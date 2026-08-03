"""
@author: HAYASHI Yasuhito (dangom_ya)

Licensed under the Apache License, Version 2.0.
"""
import argparse
import math
import os
import queue
import subprocess
import threading
import time
from datetime import datetime

import cv2
import ffmpeg
from flask import Flask, Response, request, jsonify

import global_calc as g_calc
from config_manager import load_config, save_config_snapshot

frame_streamer_instance = None


def normalize_save_settings(save_location, save_filename):
    if not isinstance(save_location, str) or not save_location.strip():
        raise ValueError("保存先フォルダを指定してください。")
    location = os.path.abspath(os.path.expanduser(save_location.strip()))
    filename = (save_filename or "").strip()
    if os.path.basename(filename) != filename or filename in {".", ".."}:
        raise ValueError("ファイル名にフォルダ区切りは使用できません。")
    os.makedirs(location, exist_ok=True)
    return location, filename

#######################################
# YAML 設定ファイルからパラメータ読み込み
#######################################
config_data = load_config()

# bothViewHW セクションからフレームカメラとイベントカメラのハードウェア情報を取得
frame_resolution = config_data["bothViewHW"]["frameCamHW"]["resolution"]
FRAME_RESOLUTION_W = int(frame_resolution[0])
FRAME_RESOLUTION_H = int(frame_resolution[1])

event_resolution = config_data["bothViewHW"]["eventCamHW"]["resolution"]
EVENT_RESOLUTION_W = int(event_resolution[0])
EVENT_RESOLUTION_H = int(event_resolution[1])

frame_pixel = config_data["bothViewHW"]["frameCamHW"]["pixel"]
FRAME_PIXCEL_W = float(frame_pixel["width"])
FRAME_PIXCEL_H = float(frame_pixel["height"])

event_pixel = config_data["bothViewHW"]["eventCamHW"]["pixel"]
EVENT_PIXCEL_W = float(event_pixel["width"])
EVENT_PIXCEL_H = float(event_pixel["height"])

FRAME_QUEUE_SIZE = 10

adjust_view = config_data["bothViewHW"]["frameCamHW"]["frame_shift"]
ADJUST_VIEW_W = int(adjust_view["width"])
ADJUST_VIEW_H = int(adjust_view["height"])
DEFAULT_FRAME_RATE = float(config_data.get("frameCam", {}).get("frameRate", 10))

#######################################
# g_value の定義（各種計算用）
#######################################
class GValue:
    def __init__(self):
        self.img_trim_width = 0
        self.img_trim_height = 0
        self.img_trim_offset_x = 0
        self.img_trim_offset_y = 0
        self.write_frame_id_x = 0
        self.write_frame_id_y = 0

g_value = GValue()

#######################################
# ffmpeg プロセス生成／破棄用関数
#######################################
def create_mp4_process(img_w: int, img_h: int, file_path: str, bitrate=2000, fps=10) -> subprocess.Popen:
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f"{img_w}x{img_h}", framerate=fps)
        .output(file_path, pix_fmt='yuv420p', vcodec='h264', vsync='passthrough',
                video_bitrate=f"{bitrate}k", loglevel='info')
        .run_async(pipe_stdin=True, overwrite_output=True)
    )
    return process

#######################################
# ffmpeg 書き込み専用スレッドクラス（sentinel 使用）
#######################################
class FfmpegWriterThread(threading.Thread):
    def __init__(self, recording_queue: 'queue.Queue', ffmpeg_process: subprocess.Popen):
        super().__init__()
        self.recording_queue = recording_queue
        self.ffmpeg_process = ffmpeg_process

    def run(self):
        while True:
            try:
                frame_np = self.recording_queue.get(timeout=0.1)
            except Exception:
                continue
            if frame_np is None:
                break
            try:
                self.ffmpeg_process.stdin.write(frame_np.tobytes())
            except BrokenPipeError:
                break
            except Exception as e:
                print("ffmpeg 書き込みエラー:", e)
        try:
            if self.ffmpeg_process.stdin:
                self.ffmpeg_process.stdin.close()
            self.ffmpeg_process.wait()
        except Exception:
            pass

#######################################
# 画像書き込み専用スレッドクラス（sentinel 使用）
#######################################
class ImageWriterThread(threading.Thread):
    def __init__(self, recording_queue: 'queue.Queue', folder_path: str):
        super().__init__()
        self.recording_queue = recording_queue
        self.folder_path = folder_path
        self.counter = 0

    def run(self):
        while True:
            try:
                frame_np = self.recording_queue.get(timeout=0.1)
            except Exception:
                continue
            if frame_np is None:
                break
            file_name = os.path.join(self.folder_path, f"{self.counter:06d}.png")
            try:
                cv2.imwrite(file_name, frame_np)
                self.counter += 1
            except Exception as e:
                print("画像保存エラー:", e)

#######################################
# カメラスレッド：vmbpy を用いてフレームを取得
#######################################
class CameraThread(threading.Thread):
    def __init__(self, frame_queue: 'queue.Queue'):
        super().__init__()
        self.frame_queue = frame_queue
        self.running = True
        self.external_callback = None
        self.cam = None  # 追加：外部アクセス用にカメラインスタンスを保持
        self.last_error = None
        self.callback_count = 0
        self.frame_count = 0
        self.last_frame_status = None
        self.allocation_mode = None

    @staticmethod
    def set_optional_feature(cam, name, value):
        """機種に存在する設定だけを適用し、映像取得自体は止めない。"""
        try:
            getattr(cam, name).set(value)
            print(f"カメラ設定: {name}={value}")
            return True
        except Exception as exc:
            print(f"カメラ設定をスキップ: {name}={value} ({exc})")
            return False

    @staticmethod
    def run_optional_command(cam, name):
        """利用可能なカメラコマンドだけを実行する。"""
        try:
            getattr(cam, name).run()
            print(f"カメラコマンド: {name}")
            return True
        except Exception as exc:
            print(f"カメラコマンドをスキップ: {name} ({exc})")
            return False

    @staticmethod
    def set_bounded_numeric_feature(cam, name, target):
        """カメラが許容する範囲内に丸めて数値Featureを設定する。"""
        try:
            feature = getattr(cam, name)
            minimum, maximum = feature.get_range()
            value = min(max(target, minimum), maximum)
            feature.set(value)
            print(f"カメラ設定: {name}={value} (範囲: {minimum}..{maximum})")
            return True
        except Exception as exc:
            print(f"カメラ設定をスキップ: {name}={target} ({exc})")
            return False

    @staticmethod
    def configure_low_bandwidth_pixel_format(cam):
        """カラー情報を保ったまま転送量の少ない8-bit Bayer形式を選ぶ。"""
        from vmbpy import PixelFormat

        preferred_formats = (
            PixelFormat.BayerRG8,
            PixelFormat.BayerGR8,
            PixelFormat.BayerGB8,
            PixelFormat.BayerBG8,
            PixelFormat.Mono8,
        )
        try:
            supported = cam.get_pixel_formats()
            for pixel_format in preferred_formats:
                if pixel_format in supported:
                    cam.set_pixel_format(pixel_format)
                    print(f"カメラ設定: PixelFormat={pixel_format}")
                    return True
            print(f"低帯域ピクセル形式を利用できません。対応形式: {supported}")
        except Exception as exc:
            print(f"ピクセル形式設定をスキップ: {exc}")
        return False

    def start_camera_streaming(self):
        """GenTLの実装差を吸収してストリーミングを開始する。"""
        from vmbpy import AllocationMode

        errors = []
        modes = (
            AllocationMode.AllocAndAnnounceFrame,
            AllocationMode.AnnounceFrame,
        )
        for allocation_mode in modes:
            try:
                self.cam.start_streaming(
                    self.frame_callback,
                    buffer_count=30,
                    allocation_mode=allocation_mode,
                )
                self.allocation_mode = allocation_mode.name
                print(f"フレームバッファ方式: {allocation_mode.name}")
                return
            except Exception as exc:
                errors.append(f"{allocation_mode.name}: {exc}")
                print(f"ストリーミング開始を再試行します ({errors[-1]})")
                self.run_optional_command(self.cam, 'AcquisitionStop')
                time.sleep(0.2)
        raise RuntimeError(" / ".join(errors))

    def run(self):
        from vmbpy import VmbSystem
        try:
            with VmbSystem.get_instance() as vmb:
                cams = vmb.get_all_cameras()
                if not cams:
                    self.last_error = "フレームカメラが見つかりません。"
                    print(self.last_error)
                    return
                cam = cams[0]
                with cam:
                    self.cam = cam
                    print(f"フレームカメラ: {cam.get_id()} / {cam.get_name()} / {cam.get_model()}")

                    # 前回の異常終了で撮像状態が残っていても、既知の状態から開始する。
                    self.run_optional_command(cam, 'AcquisitionStop')

                    # このカメラはEVSへ同期信号を出す側なので、撮像自体はフリーランにする。
                    self.set_optional_feature(cam, 'AcquisitionMode', 'Continuous')
                    self.set_optional_feature(cam, 'TriggerSelector', 'FrameStart')
                    self.set_optional_feature(cam, 'TriggerMode', 'Off')

                    # フル解像度RGBの帯域超過を避け、EVSとの同時接続に余裕を持たせる。
                    self.configure_low_bandwidth_pixel_format(cam)
                    self.set_bounded_numeric_feature(
                        cam, 'DeviceLinkThroughputLimit', 100_000_000)

                    # 同期出力は機種ごとの差が大きいため、未対応でも映像取得を継続する。
                    self.set_optional_feature(cam, 'LineSelector', 'Line0')
                    self.set_optional_feature(cam, 'LineMode', 'Output')
                    self.set_optional_feature(cam, 'LineSource', 'ExposureActive')
                    self.set_optional_feature(cam, 'LineInverter', True)
                    # 同時接続するEVSとUSB帯域を競合させない初期値にする。
                    self.set_optional_feature(cam, 'AcquisitionFrameRateEnable', True)
                    self.set_optional_feature(cam, 'AcquisitionFrameRate', DEFAULT_FRAME_RATE)

                    frame_sensor_w, frame_sensor_h = g_calc.get_cencer_size(
                        (FRAME_RESOLUTION_W, FRAME_RESOLUTION_H),
                        (FRAME_PIXCEL_W, FRAME_PIXCEL_H))
                    event_sensor_w, event_sensor_h = g_calc.get_cencer_size(
                        (EVENT_RESOLUTION_W, EVENT_RESOLUTION_H),
                        (EVENT_PIXCEL_W, EVENT_PIXCEL_H))
                    frame_trim_w, frame_trim_h = g_calc.get_trim_pixel_size(
                        (FRAME_RESOLUTION_W, FRAME_RESOLUTION_H),
                        (EVENT_RESOLUTION_W, EVENT_RESOLUTION_H),
                        (frame_sensor_w, frame_sensor_h),
                        (event_sensor_w, event_sensor_h))
                    frame_roi_w = FRAME_RESOLUTION_W - (frame_trim_w * 2)
                    frame_roi_h = FRAME_RESOLUTION_H - (frame_trim_h * 2)
                    ((frame_roi_w, frame_roi_h), (frame_trim_w, frame_trim_h)) = g_calc.get_adjusted_roi(
                        (frame_roi_w, frame_roi_h),
                        (frame_trim_w, frame_trim_h))
                    adjust_w, adjust_h = ADJUST_VIEW_W, ADJUST_VIEW_H
                    g_value.img_trim_width = int(frame_roi_w)
                    g_value.img_trim_height = int(frame_roi_h)
                    g_value.img_trim_offset_x = int(frame_trim_w - g_calc.get_adjusted_offset(adjust_w))
                    g_value.img_trim_offset_y = int(frame_trim_h + g_calc.get_adjusted_offset(adjust_h))
                    g_value.write_frame_id_x = 40 + g_value.img_trim_offset_x
                    g_value.write_frame_id_y = 60 + g_value.img_trim_offset_y
                    print("センサーサイズおよびROI設定完了")

                    self.start_camera_streaming()
                    print("フレームカメラのストリーミングを開始しました。")
                    while self.running:
                        time.sleep(0.01)
                    cam.stop_streaming()
        except Exception as exc:
            self.last_error = f"フレームカメラ初期化エラー: {exc}"
            print(self.last_error)
        finally:
            self.cam = None

    def frame_callback(self, cam, stream, frame):
        from vmbpy import FrameStatus, PixelFormat
        self.callback_count += 1
        frame_status = frame.get_status()
        self.last_frame_status = str(frame_status)
        frame_np = None
        if frame_status == FrameStatus.Complete:
            try:
                # SDKバッファは直後に返却するため、独立した領域へコピーする。
                frame_np = frame.as_opencv_image().copy()
            except ValueError:
                try:
                    converted = frame.convert_pixel_format(PixelFormat.Bgr8)
                    frame_np = converted.as_opencv_image().copy()
                except Exception as exc:
                    self.last_error = f"ピクセル形式の変換エラー: {exc}"
            if frame_np is not None and frame_np.ndim == 2:
                frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)
            if frame_np is not None:
                self.frame_count += 1
                self.last_error = None
        else:
            self.last_error = f"不完全なフレームを受信しました: {frame_status}"

        # JPEG変換などより先にSDKバッファを返し、受信バッファの枯渇を防ぐ。
        try:
            cam.queue_frame(frame)
        except Exception as e:
            print("Error re-queuing frame:", e)

        if frame_np is not None and self.external_callback:
            try:
                self.external_callback(frame_np)
            except Exception as exc:
                print("外部コールバックエラー:", exc)

    def stop(self):
        self.running = False

    def set_framerate(self, fps: float):
        """Stop streaming, set manual framerate, then restart."""
        if self.cam is None:
            return False, "Camera not initialized"
        try:
            # 一旦ストリーミングを停止
            self.cam.stop_streaming()
            # マニュアルモードにしてFPSを設定
            self.cam.AcquisitionFrameRateEnable.set(True)
            self.cam.AcquisitionFrameRate.set(float(fps))
            # ストリーミング再開
            self.start_camera_streaming()
            return True, self.cam.AcquisitionFrameRate.get()
        except Exception as e:
            return False, str(e)

#######################################
# FrameStreamer クラス
#######################################
class FrameStreamer:
    def __init__(self, save_location, display_factor=0.5):
        self.save_location = save_location
        self.display_factor = display_factor
        self.save_filename = ""  # ファイル名設定
        self.latest_frame_jpeg = None  # JPEG エンコード済みフレーム
        self.recording = False
        self.recording_file = None
        self.recording_queue = None
        self.ffmpeg_process = None
        self.ffmpeg_thread = None
        self.image_thread = None
        self.recording_mode = "mp4"

        # フレームカメラ用キュー
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.cam_thread = CameraThread(self.frame_queue)
        # 外部コールバックで最新フレームを更新し、録画中なら画像も保存
        self.cam_thread.external_callback = self.update_latest_frame
        self.cam_thread.start()
        self.recording_folder = None

    def update_latest_frame(self, frame_np):
        try:
            # 切り抜きパラメータが設定されていれば画像をクロップ
            if g_value.img_trim_width > 0 and g_value.img_trim_height > 0:
                x = g_value.img_trim_offset_x
                y = g_value.img_trim_offset_y
                w = g_value.img_trim_width
                h = g_value.img_trim_height
                cropped_frame_np = frame_np[y:y+h, x:x+w]
            else:
                cropped_frame_np = frame_np

            display_np = cropped_frame_np
            if self.display_factor != 1.0:
                display_np = cv2.resize(
                    cropped_frame_np,
                    None,
                    fx=self.display_factor,
                    fy=self.display_factor,
                    interpolation=cv2.INTER_AREA,
                )

            # JPEG エンコードを一度だけ行い、結果のバイト列を保持
            ret, jpeg_buf = cv2.imencode('.jpg', display_np)
            if ret:
                self.latest_frame_jpeg = jpeg_buf.tobytes()

            if self.recording and self.recording_queue is not None:
                try:
                    self.recording_queue.put_nowait(cropped_frame_np.copy())
                except queue.Full:
                    pass
        except Exception as e:
            print("フレーム変換エラー:", e)

    def start_recording(self, mode: str = "mp4"):
        if mode not in {"mp4", "images"}:
            print("未対応の録画形式です:", mode)
            return False
        if self.cam_thread.cam is None:
            print("フレームカメラが初期化されていません。")
            return False
        if not self.recording:
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            folder_name = f"{now}_{self.save_filename}" if self.save_filename else now
            folder_path = os.path.join(self.save_location, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            self.recording_folder = folder_path
            self.recording_mode = mode

            current_config = load_config()
            snapshot_path = save_config_snapshot(current_config, folder_path, prefix="frame_config")
            print("設定スナップショットを保存しました:", snapshot_path)

            self.recording_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)

            if mode == "mp4":
                self.recording_file = os.path.join(folder_path, "recording.mp4")
                fps_val = 10
                try:
                    cam = self.cam_thread.cam
                    if cam is not None:
                        fps_val = cam.AcquisitionFrameRate.get()
                except Exception:
                    pass
                self.ffmpeg_process = create_mp4_process(
                    g_value.img_trim_width if g_value.img_trim_width > 0 else FRAME_RESOLUTION_W,
                    g_value.img_trim_height if g_value.img_trim_height > 0 else FRAME_RESOLUTION_H,
                    self.recording_file,
                    fps=int(fps_val),
                )
                self.ffmpeg_thread = FfmpegWriterThread(self.recording_queue, self.ffmpeg_process)
                self.ffmpeg_thread.start()
                print("Frame録画開始:", self.recording_file)
            else:
                self.recording_file = None
                self.image_thread = ImageWriterThread(self.recording_queue, folder_path)
                self.image_thread.start()
                print("Frame画像保存開始:", folder_path)

            self.recording = True
            return True
        print("既に録画中です。")
        return False

    def stop_recording(self):
        if self.recording:
            self.recording = False
            if self.recording_queue is not None:
                self.recording_queue.put(None)
            if self.recording_mode == "mp4":
                if self.ffmpeg_thread is not None:
                    self.ffmpeg_thread.join()
                    self.ffmpeg_thread = None
                self.ffmpeg_process = None
                print("Frame録画停止。保存先:", self.recording_file)
                self.recording_file = None
            else:
                if self.image_thread is not None:
                    self.image_thread.join()
                    self.image_thread = None
                print("Frame画像保存停止。保存先:", self.recording_folder)
            self.recording_queue = None
            self.recording_folder = None
            return True
        print("録画していません。")
        return False

    def update_save_settings(self, save_location, save_filename):
        self.save_location, self.save_filename = normalize_save_settings(save_location, save_filename)
        print(f"保存設定更新: 保存先={self.save_location}, ファイル名のプレフィックス={self.save_filename}")
        return True

    def shutdown(self):
        self.cam_thread.stop()
        self.cam_thread.join()

#######################################
# Flask アプリの定義（Web API 部分）
#######################################
app = Flask(__name__)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            if frame_streamer_instance and frame_streamer_instance.latest_frame_jpeg:
                frame_data = frame_streamer_instance.latest_frame_jpeg
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_save', methods=['POST'])
def set_save():
    data = request.get_json(silent=True) or {}
    save_location = data.get('save_location')
    save_filename = data.get('save_filename')
    try:
        frame_streamer_instance.update_save_settings(save_location, save_filename)
    except (OSError, ValueError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    return jsonify({"status": "success", "message": "保存設定を更新しました。"})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    data = request.get_json() or {}
    mode = data.get('mode', 'mp4')
    success = frame_streamer_instance.start_recording(mode)
    message = "Frame 録画を開始しました。" if success else "Frame 録画を開始できませんでした。"
    return jsonify({"status": "success" if success else "error", "message": message}), 200 if success else 409

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    success = frame_streamer_instance.stop_recording()
    message = "Frame 録画を停止しました。" if success else "Frame は録画中ではありません。"
    return jsonify({"status": "success" if success else "error", "message": message}), 200 if success else 409


@app.route('/status')
def status():
    camera_ready = bool(frame_streamer_instance and frame_streamer_instance.cam_thread.cam)
    camera_thread = frame_streamer_instance.cam_thread if frame_streamer_instance else None
    camera_settings = {}
    if camera_ready:
        for feature_name in (
                'AcquisitionMode', 'TriggerSelector', 'TriggerMode', 'TriggerSource',
                'AcquisitionFrameRateEnable', 'AcquisitionFrameRate',
                'DeviceLinkThroughputLimit', 'ExposureAuto', 'ExposureTime',
                'Width', 'Height'):
            try:
                feature_value = getattr(camera_thread.cam, feature_name).get()
                if isinstance(feature_value, (bool, float, int, str)) or feature_value is None:
                    camera_settings[feature_name] = feature_value
                else:
                    camera_settings[feature_name] = str(feature_value)
            except Exception as exc:
                camera_settings[feature_name] = f"取得不可: {exc}"
        try:
            camera_settings['PixelFormat'] = str(camera_thread.cam.get_pixel_format())
        except Exception as exc:
            camera_settings['PixelFormat'] = f"取得不可: {exc}"
    return jsonify({
        "status": "success",
        "streaming": camera_ready,
        "frame_ready": bool(frame_streamer_instance and frame_streamer_instance.latest_frame_jpeg),
        "recording": bool(frame_streamer_instance and frame_streamer_instance.recording),
        "callback_count": camera_thread.callback_count if camera_thread else 0,
        "frame_count": camera_thread.frame_count if camera_thread else 0,
        "last_frame_status": camera_thread.last_frame_status if camera_thread else None,
        "allocation_mode": camera_thread.allocation_mode if camera_thread else None,
        "message": camera_thread.last_error if camera_thread else "カメラを初期化していません。",
        "camera_settings": camera_settings,
    })

# --- Exposure 設定用 API エンドポイント ---
@app.route('/set_exposure', methods=['POST'])
def set_exposure():
    data = request.get_json(silent=True) or {}
    mode = data.get('mode')
    if mode not in ['Once', 'Continuous', 'Manual']:
        return jsonify({"status": "error", "message": "Invalid mode specified. Use 'Once', 'Continuous', or 'Manual'."}), 400
    try:
        cam = frame_streamer_instance.cam_thread.cam
        if cam is None:
            return jsonify({"status": "error", "message": "カメラが初期化されていません。"}), 500

        if mode in ['Once', 'Continuous']:
            cam.ExposureAuto.set(mode)
            time.sleep(0.1)
            current_exposure_value = cam.ExposureTime.get()
            print("Current Exposure value:", current_exposure_value)
            return jsonify({"status": "success", "message": f"Exposure set to {current_exposure_value}."})
        elif mode == "Manual":
            manual_value = data.get("value")
            if manual_value is None:
                return jsonify({"status": "error", "message": "Manual value not provided."}), 400
            try:
                manual_value = float(manual_value)
            except Exception:
                return jsonify({"status": "error", "message": "Invalid manual value."}), 400
            try:
                min_exposure, max_exposure = cam.ExposureTime.get_range()
            except Exception as e:
                return jsonify({"status": "error", "message": f"Failed to get exposure range: {e}"}), 500
            if not (min_exposure <= manual_value <= max_exposure):
                return jsonify({"status": "error", "message": f"Manual exposure value must be between {min_exposure} and {max_exposure}."}), 400
            cam.ExposureAuto.set("Off")
            cam.ExposureTime.set(manual_value)
            time.sleep(0.1)
            current_exposure_value = cam.ExposureTime.get()
            print("Manual Exposure set to:", current_exposure_value)
            return jsonify({"status": "success", "message": f"Exposure manually set to {current_exposure_value}."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- Gain 設定用 API エンドポイント ---
@app.route('/set_gain', methods=['POST'])
def set_gain():
    data = request.get_json(silent=True) or {}
    mode = data.get('mode')
    if mode not in ['Once', 'Continuous', 'Manual']:
        return jsonify({"status": "error", "message": "Invalid mode specified. Use 'Once', 'Continuous', or 'Manual'."}), 400
    try:
        cam = frame_streamer_instance.cam_thread.cam
        if cam is None:
            return jsonify({"status": "error", "message": "カメラが初期化されていません。"}), 500

        if mode in ['Once', 'Continuous']:
            cam.GainAuto.set(mode)
            time.sleep(0.1)
            current_gain_value = cam.Gain.get()
            print("Current Gain value:", current_gain_value)
            return jsonify({"status": "success", "message": f"Gain set to {current_gain_value}."})
        elif mode == "Manual":
            manual_value = data.get("value")
            if manual_value is None:
                return jsonify({"status": "error", "message": "Manual value not provided."}), 400
            try:
                manual_value = float(manual_value)
            except Exception:
                return jsonify({"status": "error", "message": "Invalid manual value."}), 400
            try:
                min_gain, max_gain = cam.Gain.get_range()
            except Exception as e:
                return jsonify({"status": "error", "message": f"Failed to get gain range: {e}"}), 500
            if not (min_gain <= manual_value <= max_gain):
                return jsonify({"status": "error", "message": f"Manual gain value must be between {min_gain} and {max_gain}."}), 400
            cam.GainAuto.set("Off")
            cam.Gain.set(manual_value)
            time.sleep(0.1)
            current_gain_value = cam.Gain.get()
            print("Manual Gain set to:", current_gain_value)
            return jsonify({"status": "success", "message": f"Gain manually set to {current_gain_value}."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- FrameRate 設定用 API エンドポイント ---
@app.route('/set_framerate', methods=['POST'])
def set_framerate():
    data = request.get_json(silent=True) or {}
    fps = data.get('fps')
    if fps is None:
        return jsonify({"status": "error", "message": "fps not provided."}), 400
    try:
        fps = float(fps)
        if not math.isfinite(fps) or fps <= 0:
            return jsonify({"status": "error", "message": "fps must be a positive finite number."}), 400
        cam_thread = frame_streamer_instance.cam_thread
        success, result = cam_thread.set_framerate(fps)
        if not success:
            return jsonify({"status": "error", "message": result}), 500
        return jsonify({"status": "success", "fps": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- FrameRate モード切替用 API エンドポイント ---
@app.route('/set_framerate_mode', methods=['POST'])
def set_framerate_mode():
    data = request.get_json(silent=True) or {}
    mode = data.get('mode')
    if mode not in ['Manual', 'Auto']:
        return jsonify({"status": "error", "message": "Invalid mode specified. Use 'Manual' or 'Auto'."}), 400
    try:
        cam = frame_streamer_instance.cam_thread.cam
        if cam is None:
            return jsonify({"status": "error", "message": "カメラが初期化されていません。"}), 500

        enable_manual = (mode == 'Manual')
        cam.AcquisitionFrameRateEnable.set(enable_manual)
        current_mode = 'Manual' if cam.AcquisitionFrameRateEnable.get() else 'Auto'
        return jsonify({"status": "success", "message": f"Frame rate mode set to {current_mode}."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- WhiteBalance 設定用 API エンドポイント ---
@app.route('/set_whitebalance', methods=['POST'])
def set_whitebalance():
    data = request.get_json(silent=True) or {}
    mode = data.get('mode')
    if mode not in ['Once', 'Continuous']:
        return jsonify({"status": "error", "message": "Invalid mode specified. Use 'Once' or 'Continuous'."}), 400
    try:
        cam = frame_streamer_instance.cam_thread.cam
        if cam is None:
            return jsonify({"status": "error", "message": "カメラが初期化されていません。"}), 500
        cam.BalanceWhiteAuto.set(mode)
        return jsonify({"status": "success", "message": f"WhiteBalance set to {mode}."})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# --- 現在の Exposure, Gain 値を返す API エンドポイント ---
@app.route('/get_settings', methods=['GET'])
def get_settings():
    try:
        cam = frame_streamer_instance.cam_thread.cam
        if cam is None:
            return jsonify({"status": "error", "message": "カメラが初期化されていません。"}), 500
        exposure_value = cam.ExposureTime.get()
        gain_value = cam.Gain.get()
        fps_value = cam.AcquisitionFrameRate.get()
        return jsonify({
            "status": "success",
            "exposure": exposure_value,
            "gain": gain_value,
            "fps": fps_value
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def run_flask_server(port):
    app.run(host='127.0.0.1', port=port, debug=False)

#######################################
# メイン処理
#######################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Frame Streamer (Web 制御版, Tkinter 不使用)")
    parser.add_argument('--save_location', dest='save_location', default=os.getcwd(),
                        help="録画ファイルの保存先ディレクトリ")
    parser.add_argument('--port', dest='port', type=int, default=5002,
                        help="Flask サーバーのポート番号")
    parser.add_argument('--input-event-file', dest='input_event_file', default="",
                        help="入力イベントファイルのパス（フレームカメラの場合は無視）")
    parser.add_argument('--display-factor', dest='display_factor', type=float, default=0.6,
                        help="表示用に縮小する倍率 (0-1). 値を小さくするとCPU負荷を減らせます")
    args = parser.parse_args()

    if not 0 < args.display_factor <= 1:
        parser.error("--display-factor は 0 より大きく 1 以下にしてください")

    # グローバル変数に FrameStreamer インスタンスをセット
    frame_streamer_instance = FrameStreamer(args.save_location, display_factor=args.display_factor)
    # Flask サーバーを別スレッドで起動
    flask_thread = threading.Thread(target=run_flask_server, args=(args.port,), daemon=True)
    flask_thread.start()

    print("Frame Streamer 起動中。CTRL+C で終了します。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("シャットダウン中...")
        frame_streamer_instance.shutdown()
