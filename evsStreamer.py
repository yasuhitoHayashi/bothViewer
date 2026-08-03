"""
@author: HAYASHI Yasuhito (dangom_ya)

Licensed under the Apache License, Version 2.0.
"""
import argparse
import csv
import json
import os
import signal
import threading
import time
from datetime import datetime, timezone

import cv2

from flask import Flask, Response, request, jsonify
from metavision_core.event_io import EventsIterator, LiveReplayEventsIterator, is_live_camera
from metavision_sdk_core import PeriodicFrameGenerationAlgorithm, ColorPalette
from metavision_sdk_ui import EventLoop
from metavision_core.event_io.raw_reader import initiate_device

# 追加：YAML 設定管理用モジュール
from config_manager import create_session_directory, load_config, save_config_snapshot

evs_streamer_instance = None


def normalize_save_settings(save_location, save_filename):
    if not isinstance(save_location, str) or not save_location.strip():
        raise ValueError("保存先フォルダを指定してください。")
    location = os.path.abspath(os.path.expanduser(save_location.strip()))
    filename = (save_filename or "").strip()
    if os.path.basename(filename) != filename or filename in {".", ".."}:
        raise ValueError("ファイル名にフォルダ区切りは使用できません。")
    os.makedirs(location, exist_ok=True)
    return location, filename


def build_synchronization_report(session_root, session_id):
    """フレームイベントとEVSトリガーを順序対応させ、欠落を可視化する。"""
    frame_events_path = os.path.join(session_root, "frame", "frame_events.csv")
    triggers_path = os.path.join(session_root, "evs", "triggers.csv")
    saved_frames_path = os.path.join(session_root, "frame", "saved_frames.csv")
    output_path = os.path.join(session_root, "synchronization.csv")
    if not os.path.exists(frame_events_path) or not os.path.exists(triggers_path):
        return {"generated": False, "reason": "frame_events.csv または triggers.csv がありません。"}

    with open(frame_events_path, newline="", encoding="utf-8") as frame_file:
        frames = list(csv.DictReader(frame_file))
    with open(triggers_path, newline="", encoding="utf-8") as trigger_file:
        all_triggers = list(csv.DictReader(trigger_file))
    saved_filenames = set()
    if os.path.exists(saved_frames_path):
        with open(saved_frames_path, newline="", encoding="utf-8") as saved_file:
            saved_filenames = {
                row["filename"] for row in csv.DictReader(saved_file)
                if row["write_ok"] == "1"
            }

    # ExposureActiveの露光開始エッジ。反転出力時は立下り、通常出力時は立上りになる。
    reference_polarity = 1
    frame_settings_path = os.path.join(session_root, "frame", "camera_settings.json")
    if os.path.exists(frame_settings_path):
        with open(frame_settings_path, encoding="utf-8") as settings_file:
            frame_settings = json.load(settings_file)
        output_inverter = frame_settings.get(
            "LineInverter",
            frame_settings.get("external_trigger", {}).get("output_inverter"),
        )
        if output_inverter is True:
            reference_polarity = 0
    triggers = [
        row for row in all_triggers if int(row["polarity"]) == reference_polarity
    ] or all_triggers
    start_offset = 0
    if frames and triggers:
        first_frame_host_ns = int(frames[0]["host_utc_ns"])
        start_offset = min(
            range(len(triggers)),
            key=lambda index: abs(int(triggers[index]["host_decode_utc_ns"]) - first_frame_host_ns),
        )

    fields = (
        "session_id", "sync_index", "match_status", "frame_sequence",
        "frame_stream_epoch", "camera_frame_id", "frame_status", "frame_filename", "frame_host_utc_ns",
        "trigger_stream_epoch",
        "trigger_index", "evs_timestamp_us", "trigger_polarity", "trigger_channel_id",
        "trigger_host_decode_utc_ns", "host_time_difference_ms",
    )
    matched = 0
    missing_image = 0
    missing_trigger = 0
    missing_frame_events = 0
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields)
        writer.writeheader()
        output_index = 0
        for index, trigger in enumerate(triggers[:start_offset]):
            writer.writerow({
                "session_id": session_id,
                "sync_index": output_index,
                "match_status": "trigger_before_first_frame",
                "frame_sequence": "", "frame_stream_epoch": "", "camera_frame_id": "", "frame_status": "",
                "frame_filename": "", "frame_host_utc_ns": "",
                "trigger_index": trigger["trigger_index"],
                "trigger_stream_epoch": trigger.get("stream_epoch", ""),
                "evs_timestamp_us": trigger["evs_timestamp_us"],
                "trigger_polarity": trigger["polarity"],
                "trigger_channel_id": trigger["channel_id"],
                "trigger_host_decode_utc_ns": trigger["host_decode_utc_ns"],
                "host_time_difference_ms": "",
            })
            output_index += 1
        trigger_position = start_offset
        for frame in frames:
            # Frame ID欠番は、トリガーは届いたがフレームcallbackが無かった区間として残す。
            missing_before = int(frame.get("missing_before") or 0)
            for _ in range(missing_before):
                trigger = triggers[trigger_position] if trigger_position < len(triggers) else None
                writer.writerow({
                    "session_id": session_id,
                    "sync_index": output_index,
                    "match_status": "frame_event_missing",
                    "frame_sequence": "", "frame_stream_epoch": "", "camera_frame_id": "", "frame_status": "",
                    "frame_filename": "", "frame_host_utc_ns": "",
                    "trigger_index": trigger["trigger_index"] if trigger else "",
                    "trigger_stream_epoch": trigger.get("stream_epoch", "") if trigger else "",
                    "evs_timestamp_us": trigger["evs_timestamp_us"] if trigger else "",
                    "trigger_polarity": trigger["polarity"] if trigger else "",
                    "trigger_channel_id": trigger["channel_id"] if trigger else "",
                    "trigger_host_decode_utc_ns": trigger["host_decode_utc_ns"] if trigger else "",
                    "host_time_difference_ms": "",
                })
                missing_frame_events += 1
                output_index += 1
                trigger_position += 1

            trigger = triggers[trigger_position] if trigger_position < len(triggers) else None
            has_image = bool(frame["filename"]) and frame["filename"] in saved_filenames
            if not has_image:
                missing_image += 1
            if trigger:
                matched += 1
            match_status = "matched"
            if not trigger:
                match_status = "missing_trigger"
                missing_trigger += 1
            elif not has_image:
                match_status = "trigger_without_saved_image"
            host_difference_ms = ""
            if trigger:
                host_difference_ms = round(
                    (int(trigger["host_decode_utc_ns"]) - int(frame["host_utc_ns"])) / 1e6, 3)
            writer.writerow({
                "session_id": session_id,
                "sync_index": output_index,
                "match_status": match_status,
                "frame_sequence": frame["sequence"],
                "frame_stream_epoch": frame.get("stream_epoch", ""),
                "camera_frame_id": frame["camera_frame_id"],
                "frame_status": frame["frame_status"],
                "frame_filename": frame["filename"],
                "frame_host_utc_ns": frame["host_utc_ns"],
                "trigger_index": trigger["trigger_index"] if trigger else "",
                "trigger_stream_epoch": trigger.get("stream_epoch", "") if trigger else "",
                "evs_timestamp_us": trigger["evs_timestamp_us"] if trigger else "",
                "trigger_polarity": trigger["polarity"] if trigger else "",
                "trigger_channel_id": trigger["channel_id"] if trigger else "",
                "trigger_host_decode_utc_ns": trigger["host_decode_utc_ns"] if trigger else "",
                "host_time_difference_ms": host_difference_ms,
            })
            output_index += 1
            trigger_position += 1
        for trigger in triggers[trigger_position:]:
            writer.writerow({
                "session_id": session_id,
                "sync_index": output_index,
                "match_status": "trigger_without_frame_event",
                "frame_sequence": "", "frame_stream_epoch": "", "camera_frame_id": "", "frame_status": "",
                "frame_filename": "", "frame_host_utc_ns": "",
                "trigger_index": trigger["trigger_index"],
                "trigger_stream_epoch": trigger.get("stream_epoch", ""),
                "evs_timestamp_us": trigger["evs_timestamp_us"],
                "trigger_polarity": trigger["polarity"],
                "trigger_channel_id": trigger["channel_id"],
                "trigger_host_decode_utc_ns": trigger["host_decode_utc_ns"],
                "host_time_difference_ms": "",
            })
            output_index += 1

    return {
        "generated": True,
        "frame_events": len(frames),
        "reference_trigger_edges": len(triggers),
        "matched": matched,
        "frames_without_saved_image": missing_image,
        "frames_without_trigger": missing_trigger,
        "frame_events_missing_from_id_gaps": missing_frame_events,
        "triggers_before_first_frame": start_offset,
        "unmatched_reference_triggers": max(0, len(triggers) - trigger_position),
        "trigger_start_offset": start_offset,
        "reference_trigger_polarity": reference_polarity,
        "reference_edge": "ExposureActive start",
        "alignment": "nearest host decode time, then sequence",
    }

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

        # 各種設定値（再接続時にも復元する）
        self.bias_diff_on = 0
        self.bias_diff_off = 0
        self.trigger_in = True

        self.display_factor = display_factor
        self.width = int(self.orig_width * self.display_factor)
        self.height = int(self.orig_height * self.display_factor)

        # 最新フレーム保持
        self.latest_frame_jpeg = None  # JPEG エンコード済みバイト列

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

        # 固定パラメータ（累積時間、FPS）
        self.fixed_accumulation_time_ms = 33
        self.fixed_accumulation_time_us = self.fixed_accumulation_time_ms * 1000
        self.fixed_fps = 50
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
        self.width = int(self.orig_width * self.display_factor)
        self.height = int(self.orig_height * self.display_factor)
        self.event_frame_gen = PeriodicFrameGenerationAlgorithm(
            sensor_width=self.orig_width,
            sensor_height=self.orig_height,
            accumulation_time_us=self.fixed_accumulation_time_us,
            fps=self.fixed_fps,
            palette=ColorPalette.Gray)
        self.event_frame_gen.set_colors(
            background_color=[128], on_color=[255], off_color=[0], colored=False)
        self.event_frame_gen.set_output_callback(self.on_cd_frame_cb)

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
        try:
            # 左右反転
            frame_np = cv2.flip(cd_frame, 1)

            # 表示縮小
            if self.display_factor != 1.0:
                frame_np = cv2.resize(
                    frame_np,
                    None,
                    fx=self.display_factor,
                    fy=self.display_factor,
                    interpolation=cv2.INTER_AREA,
                )

            # JPEG エンコードを一度だけ実施
            ret, jpeg_buf = cv2.imencode('.jpg', frame_np)
            if ret:
                self.latest_frame_jpeg = jpeg_buf.tobytes()

        except Exception as e:
            print("EVS フレーム変換エラー:", e)

    def event_loop(self):
        retry_delay = 1.0
        while self.running:
            try:
                if self.live_mode and self.device is None:
                    self.connection_state = "reconnecting" if self.ever_connected else "connecting"
                    self.reconnect_attempts += 1
                    self.connect_live_device()
                    retry_delay = 1.0
                for evs in self.mv_iterator:
                    if not self.running:
                        break
                    EventLoop.poll_and_dispatch()
                    self.event_frame_gen.process_events(evs)
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

    def request_reconnect(self):
        """接続中のストリームを明示的に閉じ、通常の復旧経路へ移行する。"""
        if not self.live_mode:
            return False, "RAW再生モードでは手動再接続できません。"
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
        with self.recording_lock:
            if self.recording and self.trigger_writer is not None:
                for event in trigger_events:
                    sensor_timestamp = int(event["t"])
                    polarity = int(event["p"])
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
                        "channel_id": int(event["id"]),
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

# --------------------------------------------------
# Flask アプリの定義（Web API 部分）
# --------------------------------------------------
app = Flask(__name__)


def reject_settings_change_while_recording():
    if evs_streamer_instance and evs_streamer_instance.recording:
        return jsonify({
            "status": "error",
            "message": "録画中はEVS設定を変更できません。先に録画を停止してください。",
        }), 409
    return None

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
            if evs_streamer_instance and evs_streamer_instance.latest_frame_jpeg:
                frame_data = evs_streamer_instance.latest_frame_jpeg
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
            time.sleep(0.05)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_save', methods=['POST'])
def set_save():
    guard = reject_settings_change_while_recording()
    if guard:
        return guard
    data = request.get_json(silent=True) or {}
    save_location = data.get('save_location')
    save_filename = data.get('save_filename')
    try:
        evs_streamer_instance.update_save_settings(save_location, save_filename)
    except (OSError, ValueError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    return jsonify({"status": "success", "message": "保存設定を更新しました。"})

@app.route('/set_bias', methods=['POST'])
def set_bias():
    guard = reject_settings_change_while_recording()
    if guard:
        return guard
    data = request.get_json(silent=True) or {}
    bias_diff_on = data.get('bias_diff_on')
    bias_diff_off = data.get('bias_diff_off')
    if bias_diff_on is None or bias_diff_off is None:
        return jsonify({"status": "error", "message": "Bias 設定の値が不足しています。"}), 400
    success = evs_streamer_instance.update_bias(bias_diff_on, bias_diff_off)
    message = "Bias 設定を更新しました。" if success else "Bias インターフェースを利用できません。"
    return jsonify({"status": "success" if success else "error", "message": message}), 200 if success else 503

@app.route('/set_trigger', methods=['POST'])
def set_trigger():
    guard = reject_settings_change_while_recording()
    if guard:
        return guard
    data = request.get_json(silent=True) or {}
    trigger = data.get('trigger')
    if trigger is None:
        return jsonify({"status": "error", "message": "Trigger の値が不足しています。"}), 400
    if trigger is not True:
        return jsonify({
            "status": "error",
            "message": "EVS Trigger Inは同期記録のため常時有効です。無効化できません。",
        }), 409
    success = evs_streamer_instance.update_trigger(True)
    message = "EVS Trigger Inは有効です。" if success else "EVS Trigger Inを有効化できませんでした。"
    return jsonify({"status": "success" if success else "error", "message": message}), 200 if success else 503

@app.route('/start_recording', methods=['POST'])
def start_recording():
    data = request.get_json(silent=True) or {}
    try:
        success = evs_streamer_instance.start_recording(data.get('session_id'))
    except (OSError, RuntimeError, ValueError) as exc:
        return jsonify({"status": "error", "message": str(exc)}), 400
    message = "EVS 録画を開始しました。" if success else "EVS 録画を開始できませんでした。"
    return jsonify({
        "status": "success" if success else "error",
        "message": message,
        "session_id": evs_streamer_instance.recording_session_id,
    }), 200 if success else 409

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    success = evs_streamer_instance.stop_recording()
    message = "EVS 録画を停止しました。" if success else "EVS は録画中ではありません。"
    return jsonify({"status": "success" if success else "error", "message": message}), 200 if success else 409


@app.route('/reconnect', methods=['POST'])
def reconnect():
    success, message = evs_streamer_instance.request_reconnect()
    return jsonify({
        "status": "success" if success else "error",
        "message": message,
        "connection": {"state": evs_streamer_instance.connection_state},
    }), 202 if success else 409


@app.route('/status')
def status():
    return jsonify({
        "status": "success",
        "streaming": bool(
            evs_streamer_instance and evs_streamer_instance.connection_state == "connected"),
        "frame_ready": bool(evs_streamer_instance and evs_streamer_instance.latest_frame_jpeg),
        "recording": bool(evs_streamer_instance and evs_streamer_instance.recording),
        "trigger_in": bool(evs_streamer_instance and evs_streamer_instance.trigger_in),
        "connection": {
            "state": evs_streamer_instance.connection_state,
            "restart_attempts": evs_streamer_instance.reconnect_attempts,
            "successful_reconnections": evs_streamer_instance.successful_reconnections,
            "stream_epoch": evs_streamer_instance.stream_epoch,
            "last_error": evs_streamer_instance.last_connection_error,
        },
        "recording_quality": {
            "session_id": evs_streamer_instance.recording_session_id,
            "trigger_events": evs_streamer_instance.recording_trigger_count,
            "rising_edges": evs_streamer_instance.recording_trigger_rising_count,
            "falling_edges": evs_streamer_instance.recording_trigger_falling_count,
            "elapsed_seconds": round(
                (time.time_ns() - evs_streamer_instance.recording_started_utc_ns) / 1e9, 1)
                if evs_streamer_instance.recording and
                evs_streamer_instance.recording_started_utc_ns else 0,
        },
    })

def run_flask_server(port):
    app.run(host='127.0.0.1', port=port, debug=False)

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
