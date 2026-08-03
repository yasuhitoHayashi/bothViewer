"""
@author: HAYASHI Yasuhito (dangom_ya)

Licensed under the Apache License, Version 2.0.
"""
import argparse
from collections import deque
import csv
import json
import os
import signal
import threading
import time
from datetime import datetime, timezone

import cv2

from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop
from metavision_core.event_io.raw_reader import initiate_device

# 追加：YAML 設定管理用モジュール
from config_manager import (
    create_session_directory, load_config, normalize_save_settings,
    save_config, save_config_snapshot,
)
from preview_manager import LatestFramePreview, PREVIEW_PRESETS
from synchronization import build_synchronization_report
from bothviewer.api.evs import create_evs_app

evs_streamer_instance = None
config_data = load_config()
preview_config = config_data.get("preview", {})
DEFAULT_PREVIEW_PRESET = str(preview_config.get("preset", "standard"))
if DEFAULT_PREVIEW_PRESET not in PREVIEW_PRESETS:
    DEFAULT_PREVIEW_PRESET = "standard"
DEFAULT_PREVIEW_AUTO_DEGRADE = bool(preview_config.get("autoDegrade", True))


# --------------------------------------------------
# EVSStreamer クラス（Tkinter 不使用版）
# --------------------------------------------------
class EVSStreamer:
    def __init__(self, event_file_path, save_location, display_factor=0.5):
        self.event_file_path = event_file_path
        self.save_location = save_location
        self.save_filename = ""  # Web から更新可能な保存ファイル名
        self.live_mode = event_file_path == "" or is_live_camera(event_file_path)
        self.device = None
        self.bias_interface = None
        self.events_stream = None
        self.mv_iterator = None
        self.orig_width, self.orig_height = 1280, 720
        self.connection_state = "connecting"
        self.stream_epoch = 0
        self.reconnect_attempts = 0
        self.successful_reconnections = 0
        self.last_connection_error = None
        self.ever_connected = False
        self.reconnect_wakeup = threading.Event()
        self.capture_active_until = 0.0

        # 各種設定値（再接続時にも復元する）
        self.bias_diff_on = 0
        self.bias_diff_off = 0
        self.trigger_in = True

        self.display_factor = display_factor
        self.width = int(self.orig_width * self.display_factor)
        self.height = int(self.orig_height * self.display_factor)

        # 最新フレーム保持
        self.event_processing_lock = threading.RLock()
        self.preview = LatestFramePreview(
            "EVS", "evs", self.encode_preview_frame,
            preset=DEFAULT_PREVIEW_PRESET,
            auto_degrade=DEFAULT_PREVIEW_AUTO_DEGRADE,
            on_effective_change=self.apply_preview_generator_settings,
        )
        self.preview.start()
        self.processing_health_lock = threading.Lock()
        self.reset_event_processing_health()

        # 録画状態
        self.recording = False
        self.recording_file = None
        self.recording_lock = threading.RLock()
        self.recording_session_id = None
        self.recording_folder = None
        self.recording_started_utc_ns = None
        self.trigger_file = None
        self.trigger_writer = None
        self.recording_trigger_count = 0
        self.recording_trigger_rising_count = 0
        self.recording_trigger_falling_count = 0
        self.previous_trigger_timestamp = None
        self.connection_file = None
        self.connection_writer = None
        self.recording_raw_files = []
        self.raw_segment_index = 0
        self.recording_reconnect_baseline = 0
        self.initialize_trigger_monitor()

        # 固定パラメータ（累積時間、FPS）
        self.fixed_accumulation_time_ms = 33
        self.fixed_accumulation_time_us = self.fixed_accumulation_time_ms * 1000
        self.fixed_fps = PREVIEW_PRESETS[DEFAULT_PREVIEW_PRESET]["fps"]
        self.event_frame_gen = None
        self.configure_event_frame_generator()

        self.running = True

        if not self.live_mode:
            iterator = EventsIterator(input_path=event_file_path, delta_t=1000)
            self.mv_iterator = LiveReplayEventsIterator(iterator)
            try:
                self.orig_height, self.orig_width = self.mv_iterator.get_size()
                self.configure_event_frame_generator()
            except Exception:
                pass

    def configure_event_frame_generator(self):
        with self.event_processing_lock:
            settings = self.preview.current_settings()
            self.fixed_fps = settings["fps"]
            self.width = int(self.orig_width * settings["evs_scale"])
            self.height = int(self.orig_height * settings["evs_scale"])
            self.event_frame_gen = PeriodicFrameGenerationAlgorithm(
                sensor_width=self.orig_width,
                sensor_height=self.orig_height,
                accumulation_time_us=self.fixed_accumulation_time_us,
                fps=self.fixed_fps,
                palette=ColorPalette.Gray)
            self.event_frame_gen.set_colors(
                background_color=[128], on_color=[255], off_color=[0], colored=False)
            self.event_frame_gen.set_output_callback(self.on_cd_frame_cb)

    def apply_preview_generator_settings(self, _preset, settings):
        self.fixed_fps = settings["fps"]
        self.width = int(self.orig_width * settings["evs_scale"])
        self.height = int(self.orig_height * settings["evs_scale"])
        with self.event_processing_lock:
            if self.event_frame_gen is not None:
                self.event_frame_gen.set_fps(self.fixed_fps)

    def update_preview_preferences(self, preset, auto_degrade, persist=True):
        status = self.preview.set_preferences(preset, auto_degrade)
        if persist:
            config = load_config()
            config.setdefault("preview", {})["preset"] = preset
            config["preview"]["autoDegrade"] = bool(auto_degrade)
            save_config(config)
        return status

    def initialize_trigger_monitor(self):
        self.trigger_monitor_lock = threading.Lock()
        self.live_trigger_count = 0
        self.live_trigger_rising_count = 0
        self.live_trigger_falling_count = 0
        self.live_trigger_last_polarity = None
        self.live_trigger_last_sensor_timestamp_us = None
        self.live_trigger_last_host_monotonic_ns = None
        self.live_trigger_recent_timestamps = {
            0: deque(maxlen=120),
            1: deque(maxlen=120),
        }

    def reset_event_processing_health(self):
        with self.processing_health_lock:
            self.processing_samples = deque(maxlen=500)
            self.sensor_to_host_offset_ns = None
            self.decode_lag_ms = 0.0
            self.process_duration_ema_ms = 0.0
            self.processing_overload_streak = 0
            self.last_lag_fallback_monotonic = 0.0

    def update_event_processing_health(self, events, process_seconds):
        host_ns = time.monotonic_ns()
        event_count = len(events)
        sensor_timestamp_us = None
        if event_count:
            try:
                sensor_timestamp_us = int(events["t"][-1])
            except Exception:
                pass
        request_fallback = False
        with self.processing_health_lock:
            self.processing_samples.append((host_ns, event_count, process_seconds))
            cutoff = host_ns - 5_000_000_000
            while self.processing_samples and self.processing_samples[0][0] < cutoff:
                self.processing_samples.popleft()
            elapsed_ms = process_seconds * 1000
            self.process_duration_ema_ms = (
                elapsed_ms if self.process_duration_ema_ms == 0
                else self.process_duration_ema_ms * 0.9 + elapsed_ms * 0.1)
            if sensor_timestamp_us is not None:
                sensor_ns = sensor_timestamp_us * 1000
                if self.sensor_to_host_offset_ns is None:
                    self.sensor_to_host_offset_ns = host_ns - sensor_ns
                expected_host_ns = sensor_ns + self.sensor_to_host_offset_ns
                self.decode_lag_ms = max(0.0, (host_ns - expected_host_ns) / 1e6)
            if self.decode_lag_ms >= 100:
                self.processing_overload_streak += 1
            else:
                self.processing_overload_streak = 0
            now = time.monotonic()
            if (self.processing_overload_streak >= 3
                    and now - self.last_lag_fallback_monotonic >= 5):
                self.last_lag_fallback_monotonic = now
                self.processing_overload_streak = 0
                request_fallback = True
        if request_fallback:
            self.preview.request_safer_preset()

    def event_processing_status(self):
        with self.processing_health_lock:
            samples = list(self.processing_samples)
            event_rate = 0.0
            utilization = 0.0
            if len(samples) >= 2 and samples[-1][0] > samples[0][0]:
                window_seconds = (samples[-1][0] - samples[0][0]) / 1e9
                event_rate = sum(sample[1] for sample in samples) / window_seconds
                utilization = sum(sample[2] for sample in samples) / window_seconds
            return {
                "event_rate_per_second": round(event_rate, 1),
                "decode_lag_ms": round(self.decode_lag_ms, 3),
                "process_duration_ms": round(self.process_duration_ema_ms, 3),
                "processing_utilization": round(utilization, 3),
                "overloaded": bool(self.decode_lag_ms >= 100 or utilization >= 0.8),
            }

    def reset_trigger_monitor_timing(self):
        """再接続によるセンサー時刻基準の変化を周波数計算へ混ぜない。"""
        with self.trigger_monitor_lock:
            self.live_trigger_recent_timestamps[0].clear()
            self.live_trigger_recent_timestamps[1].clear()
            self.live_trigger_last_sensor_timestamp_us = None
            self.live_trigger_last_host_monotonic_ns = None

    def trigger_monitor_status(self):
        with self.trigger_monitor_lock:
            now_ns = time.monotonic_ns()
            age_ms = (
                (now_ns - self.live_trigger_last_host_monotonic_ns) / 1e6
                if self.live_trigger_last_host_monotonic_ns is not None else None)

            def frequency(polarity):
                timestamps = self.live_trigger_recent_timestamps[polarity]
                if len(timestamps) < 2 or timestamps[-1] <= timestamps[0]:
                    return 0.0
                return (len(timestamps) - 1) * 1_000_000 / (timestamps[-1] - timestamps[0])

            rising_hz = frequency(1)
            falling_hz = frequency(0)
            observed_period_ms = max(
                1000 / rising_hz if rising_hz else 0,
                1000 / falling_hz if falling_hz else 0,
            )
            active_timeout_ms = max(2000, min(30_000, observed_period_ms * 3))
            return {
                "enabled": bool(self.trigger_in),
                "active": bool(age_ms is not None and age_ms <= active_timeout_ms),
                "active_timeout_ms": round(active_timeout_ms, 1),
                "edge_count": self.live_trigger_count,
                "rising_edges": self.live_trigger_rising_count,
                "falling_edges": self.live_trigger_falling_count,
                "rising_hz": round(rising_hz, 3),
                "falling_hz": round(falling_hz, 3),
                "rising_period_ms": round(1000 / rising_hz, 3) if rising_hz else None,
                "falling_period_ms": round(1000 / falling_hz, 3) if falling_hz else None,
                "last_polarity": self.live_trigger_last_polarity,
                "last_sensor_timestamp_us": self.live_trigger_last_sensor_timestamp_us,
                "last_edge_age_ms": round(age_ms, 1) if age_ms is not None else None,
            }

    def record_connection_event(self, event, message):
        with self.recording_lock:
            if self.connection_writer is None:
                return
            self.connection_writer.writerow({
                "session_id": self.recording_session_id,
                "host_utc_ns": time.time_ns(),
                "host_monotonic_ns": time.monotonic_ns(),
                "stream_epoch": self.stream_epoch,
                "event": event,
                "message": message,
            })
            self.connection_file.flush()

    def start_recovered_raw_segment(self):
        if not self.recording or self.events_stream is None:
            return
        self.raw_segment_index += 1
        filename = f"events_{self.raw_segment_index:03d}.raw"
        path = os.path.join(self.recording_folder, filename)
        if not self.events_stream.log_raw_data(path):
            raise RuntimeError(f"復旧後のRAW記録を開始できません: {filename}")
        self.recording_file = path
        self.recording_raw_files.append(filename)

    def connect_live_device(self):
        device = initiate_device("")
        iterator = EventsIterator.from_device(device=device)
        height, width = iterator.get_size()
        self.device = device
        self.bias_interface = device.get_i_ll_biases()
        self.events_stream = device.get_i_events_stream()
        self.mv_iterator = iterator
        self.reset_trigger_monitor_timing()
        self.reset_event_processing_health()
        if (width, height) != (self.orig_width, self.orig_height):
            self.orig_width, self.orig_height = width, height
            self.configure_event_frame_generator()
        if self.bias_diff_on or self.bias_diff_off:
            self.update_bias(self.bias_diff_on, self.bias_diff_off)
        if not self.update_trigger(True):
            raise RuntimeError("EVS Trigger Inを有効化できませんでした。")
        if self.ever_connected:
            self.stream_epoch += 1
            self.successful_reconnections += 1
            with self.recording_lock:
                self.previous_trigger_timestamp = None
                self.start_recovered_raw_segment()
            self.record_connection_event("reconnected", "EVSへ再接続しました。")
        self.ever_connected = True
        self.connection_state = "connected"
        self.last_connection_error = None

    def on_cd_frame_cb(self, ts, cd_frame):
        """イベントフレーム生成時のコールバック"""
        # SDKの一時バッファだけをコピーし、反転・縮小・JPEG化は別スレッドへ渡す。
        self.preview.submit(cd_frame.copy())

    @staticmethod
    def encode_preview_frame(cd_frame, settings):
        frame_np = cv2.flip(cd_frame, 1)
        scale = settings["evs_scale"]
        if scale != 1.0:
            frame_np = cv2.resize(
                frame_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ret, jpeg_buf = cv2.imencode(
            ".jpg", frame_np,
            [cv2.IMWRITE_JPEG_QUALITY, int(settings["jpeg_quality"])])
        return jpeg_buf.tobytes() if ret else None

    def event_loop(self):
        retry_delay = 1.0
        while self.running:
            try:
                if self.live_mode and self.device is None:
                    while self.running and not self.capture_retry_active():
                        self.connection_state = "retry_paused"
                        self.reconnect_wakeup.wait(1.0)
                        self.reconnect_wakeup.clear()
                    if not self.running:
                        break
                    self.connection_state = "reconnecting" if self.ever_connected else "connecting"
                    self.reconnect_attempts += 1
                    self.connect_live_device()
                    retry_delay = 1.0
                for evs in self.mv_iterator:
                    if not self.running:
                        break
                    EventLoop.poll_and_dispatch()
                    process_started = time.monotonic()
                    with self.event_processing_lock:
                        self.event_frame_gen.process_events(evs)
                    self.update_event_processing_health(
                        evs, time.monotonic() - process_started)
                    self.record_trigger_events()
                if not self.running or not self.live_mode:
                    break
                raise RuntimeError("EVSイベントストリームが終了しました。")
            except Exception as exc:
                if not self.running:
                    break
                if not self.live_mode:
                    self.last_connection_error = str(exc)
                    break
                self.last_connection_error = str(exc)
                self.connection_state = "reconnecting"
                print("EVS切断・再接続待機:", exc)
                self.record_connection_event("disconnected", str(exc))
                with self.recording_lock:
                    if self.recording and self.events_stream is not None:
                        try:
                            self.events_stream.stop_log_raw_data()
                        except Exception:
                            pass
                try:
                    if self.events_stream is not None:
                        self.events_stream.stop()
                except Exception:
                    pass
                self.device = None
                self.bias_interface = None
                self.events_stream = None
                self.mv_iterator = None
                self.reconnect_wakeup.wait(retry_delay)
                self.reconnect_wakeup.clear()
                retry_delay = min(5.0, retry_delay * 2)
        self.connection_state = "stopped" if not self.running else "disconnected"

    def set_capture_active(self, active, lease_seconds=12.0):
        """撮影画面が見えている間だけ接続試行を許可する。"""
        self.capture_active_until = (
            time.monotonic() + max(1.0, float(lease_seconds)) if active else 0.0)
        self.reconnect_wakeup.set()
        return self.capture_retry_active()

    def capture_retry_active(self):
        return time.monotonic() < self.capture_active_until

    def request_reconnect(self):
        """接続中のストリームを明示的に閉じ、通常の復旧経路へ移行する。"""
        if not self.live_mode:
            return False, "RAW再生モードでは手動再接続できません。"
        if not self.capture_retry_active():
            return False, "撮影タブが非表示のため、EVS再接続は休止中です。"
        self.connection_state = "reconnecting"
        self.last_connection_error = "手動再接続を要求しました。"
        self.record_connection_event("manual_reconnect_requested", self.last_connection_error)
        self.reconnect_wakeup.set()
        stream = self.events_stream
        if stream is not None:
            try:
                stream.stop()
            except Exception as exc:
                self.last_connection_error = f"手動切断処理: {exc}"
        return True, "EVSの再接続を開始しました。"

    def record_trigger_events(self):
        """デコード済み外部トリガーをCSVへ追記し、SDK内部バッファを解放する。"""
        try:
            trigger_events = self.mv_iterator.get_ext_trigger_events()
        except (AssertionError, RuntimeError):
            return
        if not len(trigger_events):
            return

        host_utc_ns = time.time_ns()
        host_monotonic_ns = time.monotonic_ns()
        decoded_events = [
            (int(event["t"]), int(event["p"]), int(event["id"]))
            for event in trigger_events
        ]
        with self.trigger_monitor_lock:
            for sensor_timestamp, polarity, _channel_id in decoded_events:
                self.live_trigger_count += 1
                if polarity:
                    self.live_trigger_rising_count += 1
                else:
                    self.live_trigger_falling_count += 1
                self.live_trigger_recent_timestamps[polarity].append(sensor_timestamp)
                self.live_trigger_last_polarity = polarity
                self.live_trigger_last_sensor_timestamp_us = sensor_timestamp
                self.live_trigger_last_host_monotonic_ns = host_monotonic_ns

        with self.recording_lock:
            if self.recording and self.trigger_writer is not None:
                for sensor_timestamp, polarity, channel_id in decoded_events:
                    sensor_delta_us = None
                    if self.previous_trigger_timestamp is not None:
                        sensor_delta_us = sensor_timestamp - self.previous_trigger_timestamp
                    self.previous_trigger_timestamp = sensor_timestamp
                    self.trigger_writer.writerow({
                        "session_id": self.recording_session_id,
                        "stream_epoch": self.stream_epoch,
                        "trigger_index": self.recording_trigger_count,
                        "evs_timestamp_us": sensor_timestamp,
                        "sensor_delta_us": sensor_delta_us,
                        "polarity": polarity,
                        "channel_id": channel_id,
                        "host_decode_utc_ns": host_utc_ns,
                        "host_decode_monotonic_ns": host_monotonic_ns,
                    })
                    self.recording_trigger_count += 1
                    if polarity:
                        self.recording_trigger_rising_count += 1
                    else:
                        self.recording_trigger_falling_count += 1
                self.trigger_file.flush()
        try:
            self.mv_iterator.reader.clear_ext_trigger_events()
        except AttributeError:
            pass

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
        self.save_location, self.save_filename = normalize_save_settings(save_location, save_filename)
        print(f"保存設定更新: 保存先={self.save_location}, ファイル名={self.save_filename}")
        return True

    def start_recording(self, session_id=None):
        if self.device is None or self.events_stream is None:
            print("録画はライブカメラのみ利用可能です。")
            return False
        with self.recording_lock:
            if self.recording:
                print("既に録画中です。")
                return False
            if not session_id:
                session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            session_root = create_session_directory(self.save_location, session_id)
            evs_path = os.path.join(session_root, "evs")
            os.makedirs(evs_path, exist_ok=True)
            raw_path = os.path.join(evs_path, "events.raw")
            triggers_path = os.path.join(evs_path, "triggers.csv")
            connection_path = os.path.join(evs_path, "connection_events.csv")
            if os.path.exists(raw_path) or os.path.exists(triggers_path):
                raise ValueError("同じsession_idのEVS記録が既に存在します。")

            if not self.trigger_in and not self.update_trigger(True):
                raise RuntimeError("EVS Trigger Inを有効化できませんでした。")
            save_config_snapshot(load_config(), session_root, prefix="evs_config")
            with open(os.path.join(evs_path, "camera_settings.json"),
                      "w", encoding="utf-8") as settings_file:
                json.dump({
                    "bias_diff_on": self.bias_diff_on,
                    "bias_diff_off": self.bias_diff_off,
                    "trigger_in": True,
                    "sensor_width": self.orig_width,
                    "sensor_height": self.orig_height,
                    "display_accumulation_time_us": self.fixed_accumulation_time_us,
                    "display_fps": self.fixed_fps,
                    "preview": self.preview.status(),
                }, settings_file, ensure_ascii=False, indent=2)
            self.trigger_file = open(triggers_path, "w", newline="", encoding="utf-8")
            self.trigger_writer = csv.DictWriter(self.trigger_file, fieldnames=(
                "session_id", "stream_epoch", "trigger_index", "evs_timestamp_us", "sensor_delta_us",
                "polarity", "channel_id", "host_decode_utc_ns", "host_decode_monotonic_ns",
            ))
            self.trigger_writer.writeheader()
            self.trigger_file.flush()
            self.connection_file = open(connection_path, "w", newline="", encoding="utf-8")
            self.connection_writer = csv.DictWriter(self.connection_file, fieldnames=(
                "session_id", "host_utc_ns", "host_monotonic_ns", "stream_epoch",
                "event", "message",
            ))
            self.connection_writer.writeheader()
            self.connection_file.flush()

            if not self.events_stream.log_raw_data(raw_path):
                self.trigger_file.close()
                self.trigger_file = None
                self.trigger_writer = None
                print("録画開始に失敗しました。")
                return False

            self.recording_file = raw_path
            self.recording_folder = evs_path
            self.recording_session_id = session_id
            self.recording_started_utc_ns = time.time_ns()
            self.recording_trigger_count = 0
            self.recording_trigger_rising_count = 0
            self.recording_trigger_falling_count = 0
            self.previous_trigger_timestamp = None
            self.raw_segment_index = 0
            self.recording_raw_files = ["events.raw"]
            self.recording_reconnect_baseline = self.successful_reconnections
            self.recording = True
            self.record_connection_event("recording_started", "録画開始時の接続状態")
            print("EVS RAW・トリガー記録開始:", evs_path)
            return True

    def stop_recording(self):
        with self.recording_lock:
            if not self.recording:
                print("録画は行われていません。")
                return False
            self.recording = False
            if self.events_stream is not None:
                try:
                    self.events_stream.stop_log_raw_data()
                except Exception as exc:
                    print("EVS RAW停止エラー:", exc)
            if self.trigger_file:
                self.trigger_file.flush()
                self.trigger_file.close()
            if self.connection_file:
                self.connection_file.flush()
                self.connection_file.close()
            stopped_ns = time.time_ns()
            session_id = self.recording_session_id
            recording_folder = self.recording_folder
            session_root = os.path.dirname(recording_folder)
            summary = {
                "session_id": self.recording_session_id,
                "started_utc_ns": self.recording_started_utc_ns,
                "stopped_utc_ns": stopped_ns,
                "duration_seconds": round((stopped_ns - self.recording_started_utc_ns) / 1e9, 6),
                "trigger_events": self.recording_trigger_count,
                "rising_edges": self.recording_trigger_rising_count,
                "falling_edges": self.recording_trigger_falling_count,
                "raw_files": list(self.recording_raw_files),
                "camera_reconnections": (
                    self.successful_reconnections - self.recording_reconnect_baseline),
            }
            with open(os.path.join(recording_folder, "evs_summary.json"),
                      "w", encoding="utf-8") as summary_file:
                json.dump(summary, summary_file, ensure_ascii=False, indent=2)
            # ランチャー終了時は両プロセスがほぼ同時に停止するため、Frame側の
            # 書き込み完了を短時間待ってからセッション全体を集計する。
            frame_events_path = os.path.join(session_root, "frame", "frame_events.csv")
            frame_summary_path = os.path.join(session_root, "frame", "frame_summary.json")
            if os.path.exists(frame_events_path):
                deadline = time.monotonic() + 30
                while not os.path.exists(frame_summary_path) and time.monotonic() < deadline:
                    time.sleep(0.1)
            synchronization = build_synchronization_report(session_root, session_id)
            frame_summary = None
            if os.path.exists(frame_summary_path):
                with open(frame_summary_path, encoding="utf-8") as frame_summary_file:
                    frame_summary = json.load(frame_summary_file)
            with open(os.path.join(session_root, "session.json"),
                      "w", encoding="utf-8") as session_file:
                json.dump({
                    "schema_version": 1,
                    "session_id": session_id,
                    "frame": frame_summary,
                    "evs": summary,
                    "synchronization": synchronization,
                }, session_file, ensure_ascii=False, indent=2)
            print("EVS記録終了:", recording_folder, summary, synchronization)
            self.recording_file = None
            self.recording_folder = None
            self.recording_session_id = None
            self.trigger_file = None
            self.trigger_writer = None
            self.connection_file = None
            self.connection_writer = None
            return True

    def shutdown(self):
        if self.recording:
            self.stop_recording()
        self.running = False
        self.reconnect_wakeup.set()
        if self.events_stream is not None:
            try:
                self.events_stream.stop()
            except Exception as exc:
                print("EVSストリーム停止エラー:", exc)
        if self.event_thread.is_alive():
            self.event_thread.join(timeout=5)
        self.preview.stop()
        self.preview.join(timeout=5)

# --------------------------------------------------
# HTTP API（カメラ制御とは別モジュール）
# --------------------------------------------------
app = create_evs_app(lambda: evs_streamer_instance)


def run_flask_server(port):
    app.run(host="127.0.0.1", port=port, debug=False)


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
    parser.add_argument('--display-factor', dest='display_factor', type=float, default=1.0,
                        help="表示用に縮小する倍率 (0-1). 値を小さくするとCPU負荷を減らせます")
    args = parser.parse_args()

    if not 0 < args.display_factor <= 1:
        parser.error("--display-factor は 0 より大きく 1 以下にしてください")

    # グローバル変数に EVSStreamer インスタンスをセット
    evs_streamer_instance = EVSStreamer(
        args.event_file_path,
        args.save_location,
        display_factor=args.display_factor,
    )
    evs_streamer_instance.start_event_loop()

    # Flask サーバーを別スレッドで起動
    flask_thread = threading.Thread(target=run_flask_server, args=(args.port,), daemon=True)
    flask_thread.start()

    stop_requested = threading.Event()
    signal.signal(signal.SIGINT, lambda _signum, _frame: stop_requested.set())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_requested.set())
    print("EVS Streamer 起動中。CTRL+C で終了します。")
    try:
        while not stop_requested.wait(1):
            pass
    finally:
        print("シャットダウン中...")
        evs_streamer_instance.shutdown()
