"""
@author: HAYASHI Yasuhito (dangom_ya)

CopyPolicy: 
    Released under the terms of the LGPLv2.1 or later.
"""
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
from flask import Flask, Response, request, jsonify
import global_calc as g_calc
from config_manager import load_config, save_config, save_config_snapshot

import queue

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
        from vmbpy import VmbSystem, Frame, FrameStatus
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

    def frame_callback(self, cam, stream, frame):
        from vmbpy import FrameStatus
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
# FrameStreamer クラス
#######################################
class FrameStreamer:
    def __init__(self, save_location, display_factor=0.5):
        self.save_location = save_location
        self.save_filename = ""  # ファイル名設定
        self.latest_frame = None  # 最新フレーム（PIL Image）
        self.recording = False
        self.recording_file = None

        # フレームカメラ用キュー
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.cam_thread = CameraThread(self.frame_queue)
        # 外部コールバックで最新フレームを更新し、録画中なら画像も保存
        self.cam_thread.external_callback = self.update_latest_frame
        self.cam_thread.start()
        # 録画時の保存先フォルダ、連番用カウンタを初期化
        self.recording_folder = None
        self.frame_counter = 0

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

            # PIL形式に変換して最新フレームとして保持
            pil_image = Image.fromarray(cv2.cvtColor(cropped_frame_np, cv2.COLOR_BGR2RGB))
            self.latest_frame = pil_image.copy()

            # 録画中なら画像を保存
            if self.recording and self.recording_folder is not None:
                filename = f"frame_{self.frame_counter:04d}.jpg"
                full_path = os.path.join(self.recording_folder, filename)
                pil_image.save(full_path)
                self.frame_counter += 1
        except Exception as e:
            print("フレーム変換エラー:", e)

    def start_recording(self):
        if not self.recording:
            # 録画開始時の時刻を取得（save_filename があればそれを含める）
            now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            folder_name = f"{now}_{self.save_filename}" if self.save_filename else now
            folder_path = os.path.join(self.save_location, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            self.recording_folder = folder_path
            self.recording = True
            self.frame_counter = 0  # カウンタ初期化

            # 録画開始時に現在の設定をスナップショットとして保存
            current_config = load_config()
            snapshot_path = save_config_snapshot(current_config, self.save_location)
            print("設定スナップショットを保存しました:", snapshot_path)

            print("Frame録画開始:", folder_path)
            return True
        print("既に録画中です。")
        return False

    def stop_recording(self):
        if self.recording:
            self.recording = False
            print("Frame録画停止。保存先:", self.recording_folder)
            self.recording_folder = None
            return True
        print("録画していません。")
        return False

    def update_save_settings(self, save_location, save_filename):
        self.save_location = save_location
        self.save_filename = save_filename
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
