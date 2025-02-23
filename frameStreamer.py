#!/usr/bin/env python3
import argparse
import os
import threading
import time
import io
import signal
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import ffmpeg  # pip install ffmpeg-python
import subprocess
import configparser

# vmbpy 関連のインポート
from vmbpy import VmbSystem, Frame, FrameStatus
import global_calc as g_calc
from flask import Flask, Response, request, jsonify

import queue

#######################################
# 設定ファイルからパラメータ読み込み
#######################################
config = configparser.ConfigParser()
config.read('config.ini', encoding="utf-8")
conf_camera01 = 'camera01_spec'
conf_camera02 = 'camera02_spec'
common = 'common'
user_setting = 'user_setting'
FRAME_RESOLUTION_W = int(config.get(conf_camera01, 'resolution_w'))
FRAME_RESOLUTION_H = int(config.get(conf_camera01, 'resolution_h'))
EVENT_RESOLUTION_W = int(config.get(conf_camera02, 'resolution_w'))
EVENT_RESOLUTION_H = int(config.get(conf_camera02, 'resolution_h'))
FRAME_PIXCEL_W = float(config.get(conf_camera01, 'pixcel_w'))
FRAME_PIXCEL_H = float(config.get(conf_camera01, 'pixcel_h'))
EVENT_PIXCEL_W = float(config.get(conf_camera02, 'pixcel_w'))
EVENT_PIXCEL_H = float(config.get(conf_camera02, 'pixcel_h'))
FRAME_QUEUE_SIZE = int(config.get(common, 'frame_queue_size'))
ADJUST_VIEW_W = int(config.get(user_setting, 'adjust_view_w'))
ADJUST_VIEW_H = int(config.get(user_setting, 'adjust_view_h'))

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
def create_mp4_process(img_w: int, img_h: int, file_path: str, bitrate=4000) -> subprocess.Popen:
    process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24', s=f"{img_w}x{img_h}", framerate=30)
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
# カメラスレッド：vmbpy を用いてフレームを取得
#######################################
class CameraThread(threading.Thread):
    def __init__(self, frame_queue: 'queue.Queue'):
        super().__init__()
        self.frame_queue = frame_queue
        self.running = True
        self.external_callback = None
        self.cam = None  # 追加：外部アクセス用にカメラインスタンスを保持

    def run(self):
        vmb = VmbSystem.get_instance()
        with vmb:
            cams = vmb.get_all_cameras()
            if not cams:
                print("フレームカメラが見つかりませんでした。")
                return
            cam = cams[0]
            with cam:
                self.cam = cam  # カメラインスタンスを保持
                cam.LineMode.set('Output')
                cam.LineSource.set('ExposureActive')
                cam.TriggerSource.set('Line0')
                cam.LineInverter.set(True)
                print("トリガアウト設定完了")
                # センサーサイズや ROI の設定
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
                g_value.img_trim_width  = int(frame_roi_w)
                g_value.img_trim_height = int(frame_roi_h)
                g_value.img_trim_offset_x = int(frame_trim_w - g_calc.get_adjusted_offset(adjust_w))
                g_value.img_trim_offset_y = int(frame_trim_h + g_calc.get_adjusted_offset(adjust_h))
                g_value.write_frame_id_x = 40 + g_value.img_trim_offset_x
                g_value.write_frame_id_y = 60 + g_value.img_trim_offset_y
                print("センサーサイズおよびROI設定完了")
                cam.start_streaming(self.frame_callback)
                while self.running:
                    time.sleep(0.01)
                cam.stop_streaming()

    def frame_callback(self, cam, stream, frame: Frame):
        if frame.get_status() == FrameStatus.Complete:
            try:
                frame_np = frame.as_opencv_image()
            except ValueError as e:
                if "Rgb8" in str(e):
                    h = frame.get_height()
                    w = frame.get_width()
                    frame_np = np.frombuffer(frame.get_buffer(), dtype=np.uint8).reshape((h, w, 3))
                    frame_np = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                else:
                    return
            try:
                self.frame_queue.put_nowait(frame_np)
            except queue.Full:
                pass
            if self.external_callback:
                try:
                    self.external_callback(frame_np)
                except Exception as e:
                    print("外部コールバックエラー:", e)
        try:
            cam.queue_frame(frame)
        except Exception as e:
            print("Error re-queuing frame:", e)

    def stop(self):
        self.running = False

#######################################
# FrameStreamer クラス（フレームカメラ用 headless 版）
#######################################
class FrameStreamer:
    def __init__(self, save_location, display_factor=0.5):
        self.save_location = save_location
        self.save_filename = ""  # Web から更新可能なファイル名設定
        self.latest_frame = None  # 最新フレーム（PIL Image）
        self.recording = False
        self.recording_file = None

        # フレームカメラ用キュー
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.cam_thread = CameraThread(self.frame_queue)
        # 外部コールバックで最新フレームを更新し、録画中なら録画用キューにも投入
        self.cam_thread.external_callback = self.update_latest_frame
        self.cam_thread.start()

    def update_latest_frame(self, frame_np):
        try:
            # 切り抜きパラメータが設定されていれば、画像をクロップする
            if g_value.img_trim_width > 0 and g_value.img_trim_height > 0:
                x = g_value.img_trim_offset_x
                y = g_value.img_trim_offset_y
                w = g_value.img_trim_width
                h = g_value.img_trim_height
                cropped_frame_np = frame_np[y:y+h, x:x+w]
            else:
                cropped_frame_np = frame_np

            # PIL 形式に変換して最新フレームとして保持
            pil_image = Image.fromarray(cv2.cvtColor(cropped_frame_np, cv2.COLOR_BGR2RGB))
            self.latest_frame = pil_image.copy()

            # 録画中であれば、切り抜かれたフレームを録画用キューに投入
            if self.recording:
                try:
                    self.recording_queue.put_nowait(cropped_frame_np)
                except queue.Full:
                    pass
        except Exception as e:
            print("フレーム変換エラー:", e)

    def start_recording(self):
        if not self.recording:
            frame_np = None
            start_time = time.time()
            while frame_np is None and time.time() - start_time < 3:
                try:
                    frame_np = self.frame_queue.get_nowait()
                except queue.Empty:
                    time.sleep(0.05)
            if frame_np is not None:
                h, w, _ = frame_np.shape
            else:
                print("フレームが取得できなかったため、デフォルトサイズを使用します。")
                w, h = FRAME_RESOLUTION_W, FRAME_RESOLUTION_H
            # タイムスタンプを取得
            base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            if self.save_filename:
                filename = f"{base}_{self.save_filename}.mp4"
            else:
                filename = f"{base}.mp4"
            file_path = os.path.join(self.save_location, filename)
            self.recording_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE*2)
            self.ffmpeg_process = create_mp4_process(w, h, file_path, bitrate=4000)
            self.recording = True
            self.recording_file = file_path
            print("Frame録画開始:", file_path)
            self.writer_thread = FfmpegWriterThread(self.recording_queue, self.ffmpeg_process)
            self.writer_thread.start()
            return True
        print("既に録画中です。")
        return False

    def stop_recording(self):
        if self.recording:
            self.recording = False
            self.recording_queue.put(None)
            if self.writer_thread:
                self.writer_thread.join()
                self.writer_thread = None
            try:
                if self.ffmpeg_process.stdin:
                    self.ffmpeg_process.stdin.close()
                self.ffmpeg_process.wait()
            except Exception as e:
                print("録画停止エラー:", e)
            self.ffmpeg_process = None
            print("Frame録画停止。ファイル:", self.recording_file)
            self.recording_file = None
            return True
        print("録画していません。")
        return False

    def update_save_settings(self, save_location, save_filename):
        self.save_location = save_location
        self.save_filename = save_filename
        print(f"保存設定更新: 保存先={self.save_location}, ファイル名={self.save_filename}")
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
            if frame_streamer_instance and frame_streamer_instance.latest_frame:
                buf = io.BytesIO()
                frame_streamer_instance.latest_frame.save(buf, format='JPEG')
                frame_data = buf.getvalue()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_save', methods=['POST'])
def set_save():
    data = request.get_json()
    save_location = data.get('save_location')
    save_filename = data.get('save_filename')
    if save_location is None:
        return jsonify({"status": "error", "message": "保存先が指定されていません。"}), 400
    frame_streamer_instance.update_save_settings(save_location, save_filename)
    return jsonify({"status": "success", "message": "保存設定を更新しました。"})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    success = frame_streamer_instance.start_recording()
    return jsonify({"status": "success" if success else "error", "message": "録画開始。"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    success = frame_streamer_instance.stop_recording()
    return jsonify({"status": "success" if success else "error", "message": "録画停止。"})

# --- Exposure 設定用 API エンドポイント ---
@app.route('/set_exposure', methods=['POST'])
def set_exposure():
    data = request.get_json()
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
    data = request.get_json()
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

# --- WhiteBalance 設定用 API エンドポイント ---
@app.route('/set_whitebalance', methods=['POST'])
def set_whitebalance():
    data = request.get_json()
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
        return jsonify({
            "status": "success",
            "exposure": exposure_value,
            "gain": gain_value
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def run_flask_server(port):
    app.run(host='0.0.0.0', port=port, debug=False)

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
    args = parser.parse_args()

    # グローバル変数に FrameStreamer インスタンスをセット
    frame_streamer_instance = FrameStreamer(args.save_location)
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