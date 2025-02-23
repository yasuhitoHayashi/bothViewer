#!/usr/bin/env python3
import argparse
import os
import threading
import time
import io
import signal
from datetime import datetime

import cv2
from PIL import Image

from flask import Flask, Response, request, jsonify
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop
from metavision_core.event_io.raw_reader import initiate_device

# --------------------------------------------------
# EVSStreamer クラス（Tkinter 不使用版）
# --------------------------------------------------
class EVSStreamer:
    def __init__(self, event_file_path, save_location, display_factor=0.5):
        self.event_file_path = event_file_path
        self.save_location = save_location
        self.save_filename = ""  # Web から更新可能な保存ファイル名

        # EVS デバイスの初期化（ライブの場合）
        if event_file_path == "" or is_live_camera(event_file_path):
            self.device = initiate_device("")
            self.bias_interface = self.device.get_i_ll_biases()
            self.events_stream = self.device.get_i_events_stream()
            mv_iterator = EventsIterator.from_device(device=self.device)
        else:
            mv_iterator = EventsIterator(input_path=event_file_path, delta_t=1000)
            self.device = None
            self.bias_interface = None
            self.events_stream = None

        if self.device is None:
            mv_iterator = LiveReplayEventsIterator(mv_iterator)
        self.mv_iterator = mv_iterator

        # センサーサイズ取得（例外時は 1280x720）
        try:
            sensor_size = self.mv_iterator.get_size()
            self.orig_height, self.orig_width = sensor_size
        except Exception:
            self.orig_width, self.orig_height = 1280, 720

        self.display_factor = display_factor
        self.width = int(self.orig_width * self.display_factor)
        self.height = int(self.orig_height * self.display_factor)

        # 最新フレーム保持（オリジナルサイズの PIL Image）
        self.latest_frame = None

        # 録画状態
        self.recording = False
        self.recording_file = None

        # 固定パラメータ（累積時間、FPS）
        self.fixed_accumulation_time_ms = 33
        self.fixed_accumulation_time_us = self.fixed_accumulation_time_ms * 1000
        self.fixed_fps = 50
        self.event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=self.orig_width,
            sensor_height=self.orig_height,
            accumulation_time_us=self.fixed_accumulation_time_us,
            fps=self.fixed_fps,
            palette=ColorPalette.Dark)
        self.event_frame_gen.set_output_callback(self.on_cd_frame_cb)

        self.running = True

        # 各種設定値（Bias, Trigger In）
        self.bias_diff_on = 0
        self.bias_diff_off = 0
        self.trigger_in = False

    def on_cd_frame_cb(self, ts, cd_frame):
        # BGR→RGB 変換して PIL 画像に変換し、最新フレームとして保持
        pil_image = Image.fromarray(cv2.cvtColor(cd_frame, cv2.COLOR_BGR2RGB))
        pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
        self.latest_frame = pil_image.copy()

    def event_loop(self):
        for evs in self.mv_iterator:
            if not self.running:
                break
            EventLoop.poll_and_dispatch()
            self.event_frame_gen.process_events(evs)
        self.running = False

    def start_event_loop(self):
        self.event_thread = threading.Thread(target=self.event_loop, daemon=True)
        self.event_thread.start()

    def update_bias(self, bias_diff_on, bias_diff_off):
        self.bias_diff_on = bias_diff_on
        self.bias_diff_off = bias_diff_off
        if self.bias_interface is not None:
            self.bias_interface.set("bias_diff_on", self.bias_diff_on)
            self.bias_interface.set("bias_diff_off", self.bias_diff_off)
            print(f"Bias 更新: ON={self.bias_diff_on}, OFF={self.bias_diff_off}")
            return True
        print("Bias インターフェースが利用できません。")
        return False

    def update_trigger(self, trigger):
        self.trigger_in = trigger
        if self.device is None:
            print("デバイスが初期化されていません。")
            return False
        try:
            trigger_obj = self.device.get_i_trigger_in()
            if self.trigger_in:
                success = trigger_obj.enable(trigger_obj.Channel.MAIN)
                print("Trigger In 有効化:", success)
                return success
            else:
                success = trigger_obj.disable(trigger_obj.Channel.MAIN)
                print("Trigger In 無効化:", success)
                return success
        except Exception as e:
            print("Trigger 更新エラー:", e)
            return False

    def update_save_settings(self, save_location, save_filename):
        self.save_location = save_location
        self.save_filename = save_filename
        print(f"保存設定更新: 保存先={self.save_location}, ファイル名={self.save_filename}")
        return True

    def start_recording(self):
        if self.device is None or self.events_stream is None:
            print("録画はライブカメラのみ利用可能です。")
            return False
        if not self.recording:
            # タイムスタンプ取得
            base = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # ユーザー入力があれば、_入力 を付与
            if self.save_filename:
                filename = f"{base}_{self.save_filename}.raw"
            else:
                filename = f"{base}.raw"
            file_path = os.path.join(self.save_location, filename)
            if self.events_stream.log_raw_data(file_path):
                print("録画開始。保存先:", file_path)
                self.recording_file = file_path
                self.recording = True
                return True
            print("録画開始に失敗しました。")
            return False
        print("既に録画中です。")
        return False

    def stop_recording(self):
        if self.recording and self.events_stream is not None:
            self.events_stream.stop_log_raw_data()
            print("録画終了。ファイル:", self.recording_file)
            self.recording = False
            self.recording_file = None
            return True
        print("録画は行われていません。")
        return False

    def shutdown(self):
        self.running = False
        if self.event_thread.is_alive():
            self.event_thread.join()

# --------------------------------------------------
# Flask アプリの定義（Web API 部分）
# --------------------------------------------------
app = Flask(__name__)

# CORS 対応（どのオリジンからでもアクセス可能にする）
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
            if evs_streamer_instance and evs_streamer_instance.latest_frame:
                buf = io.BytesIO()
                evs_streamer_instance.latest_frame.save(buf, format='JPEG')
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
    evs_streamer_instance.update_save_settings(save_location, save_filename)
    return jsonify({"status": "success", "message": "保存設定を更新しました。"})

@app.route('/set_bias', methods=['POST'])
def set_bias():
    data = request.get_json()
    bias_diff_on = data.get('bias_diff_on')
    bias_diff_off = data.get('bias_diff_off')
    if bias_diff_on is None or bias_diff_off is None:
        return jsonify({"status": "error", "message": "Bias 設定の値が不足しています。"}), 400
    success = evs_streamer_instance.update_bias(bias_diff_on, bias_diff_off)
    return jsonify({"status": "success" if success else "error", "message": "Bias 設定更新完了。"})

@app.route('/set_trigger', methods=['POST'])
def set_trigger():
    data = request.get_json()
    trigger = data.get('trigger')
    if trigger is None:
        return jsonify({"status": "error", "message": "Trigger の値が不足しています。"}), 400
    success = evs_streamer_instance.update_trigger(trigger)
    return jsonify({"status": "success" if success else "error", "message": "Trigger 設定更新完了。"})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    success = evs_streamer_instance.start_recording()
    return jsonify({"status": "success" if success else "error", "message": "録画開始。"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    success = evs_streamer_instance.stop_recording()
    return jsonify({"status": "success" if success else "error", "message": "録画停止。"})

def run_flask_server(port):
    app.run(host='0.0.0.0', port=port, debug=False)

# --------------------------------------------------
# メイン処理
# --------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EVS Streamer (Web 制御版, Tkinter 不使用)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input-event-file', dest='event_file_path', default="",
                        help="入力イベントファイルのパス。未指定の場合はライブカメラを使用します。")
    parser.add_argument('--save_location', dest='save_location', default=os.getcwd(),
                        help="録画ファイルの保存先ディレクトリ")
    parser.add_argument('--port', dest='port', type=int, default=5001,
                        help="Flask サーバーのポート番号")
    args = parser.parse_args()

    # グローバル変数に EVSStreamer インスタンスをセット
    evs_streamer_instance = EVSStreamer(args.event_file_path, args.save_location)
    evs_streamer_instance.start_event_loop()

    # Flask サーバーを別スレッドで起動
    flask_thread = threading.Thread(target=run_flask_server, args=(args.port,), daemon=True)
    flask_thread.start()

    print("EVS Streamer 起動中。CTRL+C で終了します。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("シャットダウン中...")
        evs_streamer_instance.shutdown()