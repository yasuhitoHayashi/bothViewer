"""
@author: HAYASHI Yasuhito (dangom_ya)

Licensed under the Apache License, Version 2.0.
"""
import argparse
from collections import deque
import csv
import json
import os
import queue
import signal
import threading
import time
from datetime import datetime, timezone

import cv2
from bothviewer.core import geometry
from bothviewer.core.config import (
    create_session_directory, load_config, normalize_save_settings,
    save_config, save_config_snapshot,
)
from bothviewer.core.preview import LatestFramePreview, PREVIEW_PRESETS
from bothviewer.api.frame import create_frame_app

frame_streamer_instance = None


#######################################
# YAML 設定ファイルからパラメータ読み込み
#######################################
config_data = load_config()
preview_config = config_data.get("preview", {})
DEFAULT_PREVIEW_PRESET = str(preview_config.get("preset", "standard"))
if DEFAULT_PREVIEW_PRESET not in PREVIEW_PRESETS:
    DEFAULT_PREVIEW_PRESET = "standard"
DEFAULT_PREVIEW_AUTO_DEGRADE = bool(preview_config.get("autoDegrade", True))

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

FRAME_QUEUE_SIZE = 64
BANDWIDTH_PRESETS = {
    "safe": 100_000_000,
    "standard": 150_000_000,
    "high": 200_000_000,
}
BANDWIDTH_ORDER = ("safe", "standard", "high")

adjust_view = config_data["bothViewHW"]["frameCamHW"]["frame_shift"]
ADJUST_VIEW_W = int(adjust_view["width"])
ADJUST_VIEW_H = int(adjust_view["height"])
DEFAULT_FRAME_RATE = float(config_data.get("frameCam", {}).get("frameRate", 10))
DEFAULT_BANDWIDTH_PRESET = str(
    config_data.get("frameCam", {}).get("bandwidthPreset", "safe"))
if DEFAULT_BANDWIDTH_PRESET not in BANDWIDTH_PRESETS:
    DEFAULT_BANDWIDTH_PRESET = "safe"
trigger_config = config_data.get("frameCam", {}).get("externalTrigger", {})
DEFAULT_TRIGGER_ENABLED = bool(trigger_config.get("enabled", False))
DEFAULT_TRIGGER_SOURCE = str(trigger_config.get("source", "Line1"))
DEFAULT_TRIGGER_ACTIVATION = str(trigger_config.get("activation", "RisingEdge"))
TRIGGER_OUTPUT_LINE = str(trigger_config.get("outputLine", "Line0"))
TRIGGER_OUTPUT_INVERTER = bool(trigger_config.get("outputInverter", True))

#######################################
# g_value の定義（各種計算用）
#######################################
class GValue:
    def __init__(self):
        self.img_trim_width = 0
        self.img_trim_height = 0
        self.img_trim_offset_x = 0
        self.img_trim_offset_y = 0
        self.sensor_offset_x = 0
        self.sensor_offset_y = 0
        self.write_frame_id_x = 0
        self.write_frame_id_y = 0

g_value = GValue()


def calculate_evs_matching_frame_roi():
    """同じレンズ倍率を前提に、EVSセンサーと同じ物理視野のFrame ROIを返す。"""
    frame_sensor = geometry.get_sensor_size(
        (FRAME_RESOLUTION_W, FRAME_RESOLUTION_H),
        (FRAME_PIXCEL_W, FRAME_PIXCEL_H))
    event_sensor = geometry.get_sensor_size(
        (EVENT_RESOLUTION_W, EVENT_RESOLUTION_H),
        (EVENT_PIXCEL_W, EVENT_PIXCEL_H))
    trim_x, trim_y = geometry.get_trim_pixel_size(
        (FRAME_RESOLUTION_W, FRAME_RESOLUTION_H),
        (EVENT_RESOLUTION_W, EVENT_RESOLUTION_H),
        frame_sensor,
        event_sensor)
    width = FRAME_RESOLUTION_W - trim_x * 2
    height = FRAME_RESOLUTION_H - trim_y * 2
    (width, height), (trim_x, trim_y) = geometry.get_adjusted_roi(
        (width, height), (trim_x, trim_y))
    offset_x = int(trim_x - geometry.get_adjusted_offset(ADJUST_VIEW_W))
    offset_y = int(trim_y + geometry.get_adjusted_offset(ADJUST_VIEW_H))
    return {
        "width": int(width),
        "height": int(height),
        "offset_x": offset_x,
        "offset_y": offset_y,
        "frame_sensor_mm": [float(frame_sensor[0]), float(frame_sensor[1])],
        "event_sensor_mm": [float(event_sensor[0]), float(event_sensor[1])],
    }

#######################################
# Bayer画像書き込み専用スレッド
#######################################
class ImageWriterThread(threading.Thread):
    RESULT_FIELDS = (
        "session_id", "sequence", "filename", "write_started_utc_ns",
        "write_finished_utc_ns", "write_duration_ms", "write_ok",
        "file_size_bytes", "error",
    )

    def __init__(self, recording_queue: 'queue.Queue', images_path: str,
                 results_path: str, session_id: str):
        super().__init__(daemon=True)
        self.recording_queue = recording_queue
        self.images_path = images_path
        self.results_path = results_path
        self.session_id = session_id
        self.saved_count = 0
        self.failed_count = 0
        self.total_bytes = 0
        self.total_write_ns = 0

    @staticmethod
    def write_bayer_pgm(file_path, image):
        """8-bit Bayer配列を無圧縮PGMとして値を変えずに保存する。"""
        if image.dtype != "uint8" or image.ndim != 2:
            raise ValueError(
                f"PGM保存には2次元uint8画像が必要です: dtype={image.dtype}, shape={image.shape}")
        contiguous = image if image.flags.c_contiguous else image.copy(order="C")
        height, width = contiguous.shape
        header = f"P5\n{width} {height}\n255\n".encode("ascii")
        with open(file_path, "wb") as image_file:
            image_file.write(header)
            image_file.write(memoryview(contiguous).cast("B"))

    def run(self):
        with open(self.results_path, "w", newline="", encoding="utf-8") as results_file:
            writer = csv.DictWriter(results_file, fieldnames=self.RESULT_FIELDS)
            writer.writeheader()
            while True:
                packet = self.recording_queue.get()
                if packet is None:
                    break
                file_path = os.path.join(self.images_path, packet["filename"])
                started_ns = time.time_ns()
                error = ""
                write_ok = False
                try:
                    self.write_bayer_pgm(file_path, packet["image"])
                    write_ok = True
                except Exception as exc:
                    error = str(exc)
                finished_ns = time.time_ns()
                file_size = os.path.getsize(file_path) if write_ok else 0
                self.total_write_ns += finished_ns - started_ns
                if write_ok:
                    self.saved_count += 1
                    self.total_bytes += file_size
                else:
                    self.failed_count += 1
                    print("画像保存エラー:", error)
                writer.writerow({
                    "session_id": self.session_id,
                    "sequence": packet["sequence"],
                    "filename": packet["filename"],
                    "write_started_utc_ns": started_ns,
                    "write_finished_utc_ns": finished_ns,
                    "write_duration_ms": round((finished_ns - started_ns) / 1_000_000, 3),
                    "write_ok": int(write_ok),
                    "file_size_bytes": file_size,
                    "error": error,
                })
                results_file.flush()

    def performance_status(self):
        write_seconds = self.total_write_ns / 1e9
        return {
            "save_fps": round(self.saved_count / write_seconds, 3) if write_seconds else 0,
            "write_megabytes_per_second": round(
                self.total_bytes / 1_000_000 / write_seconds, 3) if write_seconds else 0,
        }

#######################################
# カメラスレッド：vmbpy を用いてフレームを取得
#######################################
class CameraThread(threading.Thread):
    SETTING_FEATURES = (
        "AcquisitionMode", "TriggerSelector", "TriggerMode", "TriggerSource",
        "AcquisitionFrameRateEnable", "AcquisitionFrameRate",
        "DeviceLinkThroughputLimit", "ExposureAuto", "ExposureTime",
        "GainAuto", "Gain", "BalanceWhiteAuto", "LineSelector", "LineMode",
        "LineSource", "LineInverter", "Width", "Height", "OffsetX", "OffsetY",
    )

    def __init__(self, free_run_fps=DEFAULT_FRAME_RATE,
                 trigger_enabled=DEFAULT_TRIGGER_ENABLED,
                 trigger_source=DEFAULT_TRIGGER_SOURCE,
                 trigger_activation=DEFAULT_TRIGGER_ACTIVATION,
                 bandwidth_preset=DEFAULT_BANDWIDTH_PRESET,
                 auto_fallback_count=0,
                 stream_epoch=0):
        super().__init__()
        self.running = True
        self.external_callback = None
        self.cam = None  # 追加：外部アクセス用にカメラインスタンスを保持
        self.last_error = None
        self.callback_count = 0
        self.frame_count = 0
        self.last_frame_status = None
        self.allocation_mode = None
        self.previous_frame_id = None
        self.incomplete_count = 0
        self.frame_id_gap_count = 0
        self.camera_control_lock = threading.RLock()
        self.external_trigger_enabled = bool(trigger_enabled)
        self.external_trigger_source = trigger_source
        self.external_trigger_activation = trigger_activation
        self.streaming_active = False
        self.free_run_fps = float(free_run_fps)
        self.bandwidth_preset = bandwidth_preset
        self.bandwidth_limit = BANDWIDTH_PRESETS[bandwidth_preset]
        self.auto_fallback_count = auto_fallback_count
        self.incomplete_timestamps_ns = deque()
        self.auto_fallback_requested = False
        self.stream_epoch = stream_epoch
        self.last_callback_monotonic_ns = None
        self.callback_timestamps_ns = deque(maxlen=120)
        self.control_requests = queue.Queue()
        self.external_trigger_monitor_lock = threading.Lock()
        self.reset_external_trigger_monitor()
        self.roi_configuration = {
            "mode": "not_configured",
            **calculate_evs_matching_frame_roi(),
        }

    def reset_external_trigger_monitor(self):
        with self.external_trigger_monitor_lock:
            self.external_trigger_callback_count = 0
            self.external_trigger_complete_count = 0
            self.external_trigger_incomplete_count = 0
            self.external_trigger_last_status = None
            self.external_trigger_last_host_monotonic_ns = None
            self.external_trigger_recent_timestamps_ns = deque(maxlen=120)

    def record_external_trigger_result(self, host_monotonic_ns, complete, frame_status):
        if not self.external_trigger_enabled:
            return
        with self.external_trigger_monitor_lock:
            self.external_trigger_callback_count += 1
            if complete:
                self.external_trigger_complete_count += 1
            else:
                self.external_trigger_incomplete_count += 1
            self.external_trigger_last_status = str(frame_status)
            self.external_trigger_last_host_monotonic_ns = host_monotonic_ns
            self.external_trigger_recent_timestamps_ns.append(host_monotonic_ns)

    def external_trigger_monitor_status(self):
        with self.external_trigger_monitor_lock:
            timestamps = self.external_trigger_recent_timestamps_ns
            measured_hz = 0.0
            if len(timestamps) >= 2 and timestamps[-1] > timestamps[0]:
                measured_hz = (len(timestamps) - 1) * 1e9 / (timestamps[-1] - timestamps[0])
            period_ms = 1000 / measured_hz if measured_hz else None
            age_ms = (
                (time.monotonic_ns() - self.external_trigger_last_host_monotonic_ns) / 1e6
                if self.external_trigger_last_host_monotonic_ns is not None else None)
            active_timeout_ms = max(2000, min(30_000, (period_ms or 0) * 3))
            return {
                "enabled": self.external_trigger_enabled,
                "active": bool(
                    self.external_trigger_enabled and age_ms is not None
                    and age_ms <= active_timeout_ms),
                "source": self.external_trigger_source,
                "activation": self.external_trigger_activation,
                "callback_frames": self.external_trigger_callback_count,
                "complete_frames": self.external_trigger_complete_count,
                "incomplete_frames": self.external_trigger_incomplete_count,
                "measured_hz": round(measured_hz, 3),
                "period_ms": round(period_ms, 3) if period_ms else None,
                "last_frame_age_ms": round(age_ms, 1) if age_ms is not None else None,
                "last_frame_status": self.external_trigger_last_status,
                "evs_output_line": TRIGGER_OUTPUT_LINE,
                "evs_output_source": "ExposureActive",
                "evs_output_inverter": TRIGGER_OUTPUT_INVERTER,
            }

    def bandwidth_status(self):
        return {
            "preset": self.bandwidth_preset,
            "limit_bytes_per_second": self.bandwidth_limit,
            "auto_fallback_count": self.auto_fallback_count,
            "fallback_pending": self.auto_fallback_requested,
        }

    def set_bandwidth_preset(self, preset, automatic=False):
        if preset not in BANDWIDTH_PRESETS:
            return False, f"不明な帯域プリセットです: {preset}"
        with self.camera_control_lock:
            if self.cam is None or not self.streaming_active:
                return False, "Camera not initialized"
            previous_preset = self.bandwidth_preset
            previous_limit = self.bandwidth_limit
            stopped = False
            try:
                self.cam.stop_streaming()
                stopped = True
                self.streaming_active = False
                self.run_optional_command(self.cam, "AcquisitionStop")
                feature = self.cam.DeviceLinkThroughputLimit
                minimum, maximum = feature.get_range()
                target = min(max(BANDWIDTH_PRESETS[preset], minimum), maximum)
                feature.set(target)
                self.bandwidth_limit = int(feature.get())
                self.bandwidth_preset = preset
                if not self.external_trigger_enabled:
                    _, fps_max = self.cam.AcquisitionFrameRate.get_range()
                    if self.free_run_fps > fps_max:
                        self.free_run_fps = float(fps_max)
                        self.cam.AcquisitionFrameRate.set(self.free_run_fps)
                self.start_camera_streaming()
                self.incomplete_timestamps_ns.clear()
                self.auto_fallback_requested = False
                if automatic:
                    self.auto_fallback_count += 1
                return True, self.bandwidth_status()
            except Exception as exc:
                if stopped:
                    try:
                        self.cam.DeviceLinkThroughputLimit.set(previous_limit)
                        self.bandwidth_preset = previous_preset
                        self.bandwidth_limit = previous_limit
                        self.start_camera_streaming()
                    except Exception as restore_exc:
                        return False, f"帯域設定失敗: {exc}; 復旧失敗: {restore_exc}"
                return False, str(exc)

    def request_control_operation(self, operation, timeout=20, **parameters):
        """VmbPy操作をカメラ所有スレッドで実行し、ネイティブ層の競合を避ける。"""
        if not self.is_alive() or not self.streaming_active:
            return False, "フレームカメラは再接続中です。"
        response_queue = queue.Queue(maxsize=1)
        self.control_requests.put({"operation": operation, **parameters,
                                   "response_queue": response_queue})
        try:
            return response_queue.get(timeout=timeout)
        except queue.Empty:
            return False, "帯域変更がタイムアウトしました。カメラは自動再接続を継続します。"

    def process_control_requests(self):
        while True:
            try:
                request_item = self.control_requests.get_nowait()
            except queue.Empty:
                return
            try:
                if request_item["operation"] == "set_bandwidth_preset":
                    result = self.set_bandwidth_preset(request_item["preset"])
                elif request_item["operation"] == "set_framerate":
                    result = self.set_framerate(request_item["fps"])
                elif request_item["operation"] == "set_external_trigger":
                    result = self.set_external_trigger_mode(
                        request_item["enabled"], request_item["source"],
                        request_item["activation"])
                else:
                    result = False, "不明なカメラ操作です。"
            except Exception as exc:
                result = False, str(exc)
            request_item["response_queue"].put(result)

    def request_safer_bandwidth(self):
        index = BANDWIDTH_ORDER.index(self.bandwidth_preset)
        if index == 0:
            return
        self.auto_fallback_requested = True

    def measured_fps(self):
        if len(self.callback_timestamps_ns) < 2:
            return 0.0
        elapsed_ns = self.callback_timestamps_ns[-1] - self.callback_timestamps_ns[0]
        if elapsed_ns <= 0:
            return 0.0
        return (len(self.callback_timestamps_ns) - 1) * 1_000_000_000 / elapsed_ns

    def framerate_capabilities(self):
        with self.camera_control_lock:
            if self.cam is None:
                raise RuntimeError("Camera not initialized")
            minimum, maximum = self.cam.AcquisitionFrameRate.get_range()
            try:
                increment = self.cam.AcquisitionFrameRate.get_increment()
            except Exception:
                increment = None
            return {
                "minimum": float(minimum),
                "maximum": float(maximum),
                "increment": float(increment) if increment else None,
                "configured": float(self.cam.AcquisitionFrameRate.get()),
                "target": float(self.free_run_fps),
                "measured": round(self.measured_fps(), 3),
            }

    def read_camera_settings(self):
        with self.camera_control_lock:
            settings = {}
            if self.cam is None:
                return settings
            for feature_name in self.SETTING_FEATURES:
                try:
                    value = getattr(self.cam, feature_name).get()
                    settings[feature_name] = (
                        value if isinstance(value, (bool, float, int, str)) or value is None
                        else str(value)
                    )
                except Exception as exc:
                    settings[feature_name] = f"取得不可: {exc}"
            try:
                settings["PixelFormat"] = str(self.cam.get_pixel_format())
            except Exception as exc:
                settings["PixelFormat"] = f"取得不可: {exc}"
            return settings

    @staticmethod
    def set_required_feature(cam, name, value):
        feature = getattr(cam, name)
        try:
            feature.set(value)
            print(f"カメラ設定: {name}={value}")
        except Exception:
            # 一部機種は現在値と同じ場合でも書込不可を返す。
            if str(feature.get()) != str(value):
                raise
            print(f"カメラ設定確認: {name}={value}")

    def configure_evs_trigger_output(self, cam):
        """フレーム露光状態をEVSへ渡す出力は、駆動モードによらず常時維持する。"""
        self.set_required_feature(cam, "LineSelector", TRIGGER_OUTPUT_LINE)
        self.set_required_feature(cam, "LineMode", "Output")
        self.set_required_feature(cam, "LineSource", "ExposureActive")
        self.set_required_feature(cam, "LineInverter", TRIGGER_OUTPUT_INVERTER)

    def apply_external_trigger_mode(self, cam, enabled, source, activation):
        """停止中のカメラへ外部トリガー設定を適用する。"""
        self.set_required_feature(cam, "TriggerSelector", "FrameStart")
        self.set_required_feature(cam, "TriggerMode", "Off")
        if enabled:
            if source == TRIGGER_OUTPUT_LINE:
                raise ValueError(f"{TRIGGER_OUTPUT_LINE} はEVS同期出力専用です。")
            self.set_required_feature(cam, "LineSelector", source)
            self.set_required_feature(cam, "LineMode", "Input")
            self.set_required_feature(cam, "TriggerSource", source)
            self.set_required_feature(cam, "TriggerActivation", activation)
            # 外部パルスを撮像基準とし、内部fps制限は無効化する。
            self.set_optional_feature(cam, "AcquisitionFrameRateEnable", False)
            self.set_required_feature(cam, "TriggerMode", "On")
        else:
            self.set_optional_feature(cam, "AcquisitionFrameRateEnable", True)
            self.set_optional_feature(cam, "AcquisitionFrameRate", self.free_run_fps)

        # LineSelectorが入力側を指したままでも、出力設定を最後に必ず再確認する。
        self.configure_evs_trigger_output(cam)
        self.external_trigger_enabled = bool(enabled)
        self.external_trigger_source = source
        self.external_trigger_activation = activation
        self.reset_external_trigger_monitor()

    def get_trigger_options(self):
        with self.camera_control_lock:
            if self.cam is None:
                return {
                    "sources": [DEFAULT_TRIGGER_SOURCE],
                    "activations": [DEFAULT_TRIGGER_ACTIVATION],
                }
            try:
                sources = [
                    str(value) for value in self.cam.TriggerSource.get_available_entries()
                    if str(value).startswith("Line") and str(value) != TRIGGER_OUTPUT_LINE
                ]
            except Exception:
                sources = [DEFAULT_TRIGGER_SOURCE]
            try:
                activations = [
                    str(value) for value in self.cam.TriggerActivation.get_available_entries()
                ]
            except Exception:
                activations = [DEFAULT_TRIGGER_ACTIVATION]
            return {"sources": sources, "activations": activations}

    def trigger_configuration(self):
        return {
            "enabled": self.external_trigger_enabled,
            "source": self.external_trigger_source,
            "activation": self.external_trigger_activation,
            "output_line": TRIGGER_OUTPUT_LINE,
            "output_source": "ExposureActive",
            "output_inverter": TRIGGER_OUTPUT_INVERTER,
        }

    def set_external_trigger_mode(self, enabled, source, activation):
        with self.camera_control_lock:
            if self.cam is None or not self.streaming_active:
                return False, "Camera not initialized"
            previous = (
                self.external_trigger_enabled,
                self.external_trigger_source,
                self.external_trigger_activation,
            )
            try:
                self.cam.stop_streaming()
                self.streaming_active = False
                self.run_optional_command(self.cam, "AcquisitionStop")
                self.apply_external_trigger_mode(self.cam, enabled, source, activation)
                self.start_camera_streaming()
                return True, self.trigger_configuration()
            except Exception as exc:
                # 切替失敗時は元の駆動モードへ戻し、映像取得を復旧させる。
                try:
                    self.apply_external_trigger_mode(self.cam, *previous)
                    self.start_camera_streaming()
                except Exception as restore_exc:
                    return False, f"{exc}; 復旧にも失敗: {restore_exc}"
                return False, str(exc)

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
    def set_aligned_numeric_feature(cam, name, target):
        """Feature固有のincrementへ丸めて設定し、実際の読戻し値を返す。"""
        feature = getattr(cam, name)
        minimum, maximum = feature.get_range()
        try:
            increment = feature.get_increment()
        except Exception:
            increment = 1
        increment = increment or 1
        bounded = min(max(float(target), float(minimum)), float(maximum))
        steps = round((bounded - float(minimum)) / float(increment))
        aligned = float(minimum) + steps * float(increment)
        aligned = min(max(aligned, float(minimum)), float(maximum))
        if all(float(value).is_integer() for value in (minimum, maximum, increment)):
            aligned = int(round(aligned))
        feature.set(aligned)
        actual = feature.get()
        print(f"カメラROI設定: {name}={actual}")
        return int(actual)

    def configure_evs_matching_hardware_roi(self, cam):
        """EVS相当の物理視野をカメラROIへ適用し、失敗時はソフトウェアcropへ戻す。"""
        desired = calculate_evs_matching_frame_roi()
        try:
            # Width/Heightを縮める前にOffsetを原点へ戻す必要がある機種がある。
            self.set_aligned_numeric_feature(cam, "OffsetX", 0)
            self.set_aligned_numeric_feature(cam, "OffsetY", 0)
            width = self.set_aligned_numeric_feature(cam, "Width", desired["width"])
            height = self.set_aligned_numeric_feature(cam, "Height", desired["height"])
            offset_x = self.set_aligned_numeric_feature(cam, "OffsetX", desired["offset_x"])
            offset_y = self.set_aligned_numeric_feature(cam, "OffsetY", desired["offset_y"])
            self.roi_configuration = {
                **desired,
                "mode": "camera_hardware_roi",
                "width": width,
                "height": height,
                "offset_x": offset_x,
                "offset_y": offset_y,
            }
            g_value.img_trim_width = width
            g_value.img_trim_height = height
            g_value.img_trim_offset_x = 0
            g_value.img_trim_offset_y = 0
            g_value.sensor_offset_x = offset_x
            g_value.sensor_offset_y = offset_y
            print("EVS相当範囲をカメラROIへ設定:", self.roi_configuration)
            return True
        except Exception as exc:
            print(f"カメラROI設定に失敗したためソフトウェアcropへ戻します: {exc}")
            # 部分適用されたROIを可能な範囲で全画素へ戻す。
            try:
                self.set_aligned_numeric_feature(cam, "OffsetX", 0)
                self.set_aligned_numeric_feature(cam, "OffsetY", 0)
                self.set_aligned_numeric_feature(cam, "Width", cam.Width.get_range()[1])
                self.set_aligned_numeric_feature(cam, "Height", cam.Height.get_range()[1])
            except Exception as restore_exc:
                print(f"全画素ROIへの復元をスキップ: {restore_exc}")
            self.roi_configuration = {**desired, "mode": "software_crop", "error": str(exc)}
            g_value.img_trim_width = desired["width"]
            g_value.img_trim_height = desired["height"]
            g_value.img_trim_offset_x = desired["offset_x"]
            g_value.img_trim_offset_y = desired["offset_y"]
            g_value.sensor_offset_x = 0
            g_value.sensor_offset_y = 0
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
                self.streaming_active = True
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

                    self.set_optional_feature(cam, 'AcquisitionMode', 'Continuous')

                    # フル解像度RGBの帯域超過を避け、EVSとの同時接続に余裕を持たせる。
                    self.configure_low_bandwidth_pixel_format(cam)
                    self.set_bounded_numeric_feature(
                        cam, 'DeviceLinkThroughputLimit', self.bandwidth_limit)
                    self.bandwidth_limit = int(cam.DeviceLinkThroughputLimit.get())

                    # EVSと同じ物理視野だけをカメラから転送する。未対応機種では
                    # 従来どおり全画素受信後のsoftware cropへ自動フォールバックする。
                    self.configure_evs_matching_hardware_roi(cam)

                    # ROIと転送帯域を先に確定すると、カメラが正しいfps上限を返せる。
                    # どちらの駆動モードでもExposureActiveはEVSへ常時出力する。
                    self.apply_external_trigger_mode(
                        cam,
                        self.external_trigger_enabled,
                        self.external_trigger_source,
                        self.external_trigger_activation,
                    )

                    self.start_camera_streaming()
                    print("フレームカメラのストリーミングを開始しました。")
                    last_health_check = 0.0
                    while self.running:
                        time.sleep(0.1)
                        self.process_control_requests()
                        now = time.monotonic()
                        if now - last_health_check >= 1.0:
                            # 外部トリガー待受中でも利用できるデバイス生存確認。
                            with self.camera_control_lock:
                                cam.DeviceLinkThroughputLimit.get()
                            last_health_check = now
                        if self.auto_fallback_requested:
                            index = BANDWIDTH_ORDER.index(self.bandwidth_preset)
                            safer = BANDWIDTH_ORDER[max(0, index - 1)]
                            success, result = self.set_bandwidth_preset(safer, automatic=True)
                            if not success:
                                self.last_error = f"帯域自動復帰エラー: {result}"
                            else:
                                try:
                                    config = load_config()
                                    config.setdefault("frameCam", {})["bandwidthPreset"] = safer
                                    config["frameCam"]["frameRate"] = self.free_run_fps
                                    save_config(config)
                                    self.last_error = (
                                        f"不完全フレームを検出したため帯域を{safer}へ下げました。")
                                except Exception as exc:
                                    self.last_error = f"帯域は復帰しましたが設定保存に失敗: {exc}"
                    cam.stop_streaming()
                    self.streaming_active = False
        except Exception as exc:
            self.last_error = f"フレームカメラ初期化エラー: {exc}"
            print(self.last_error)
        finally:
            self.streaming_active = False
            self.cam = None

    def frame_callback(self, cam, stream, frame):
        from vmbpy import FrameStatus
        host_utc_ns = time.time_ns()
        host_monotonic_ns = time.monotonic_ns()
        self.callback_count += 1
        self.callback_timestamps_ns.append(host_monotonic_ns)
        self.last_callback_monotonic_ns = host_monotonic_ns
        def safe_value(getter, default=None):
            try:
                return getter()
            except Exception:
                return default

        frame_status = safe_value(frame.get_status)
        self.last_frame_status = str(frame_status)
        self.record_external_trigger_result(
            host_monotonic_ns, frame_status == FrameStatus.Complete, frame_status)
        frame_id = safe_value(frame.get_id)
        frame_id_delta = None
        missing_before = 0
        if frame_id is not None and self.previous_frame_id is not None:
            frame_id_delta = frame_id - self.previous_frame_id
            if frame_id_delta > 1:
                missing_before = frame_id_delta - 1
                self.frame_id_gap_count += missing_before
        if frame_id is not None:
            self.previous_frame_id = frame_id

        metadata = {
            "callback_index": self.callback_count,
            "stream_epoch": self.stream_epoch,
            "host_utc_ns": host_utc_ns,
            "host_monotonic_ns": host_monotonic_ns,
            "camera_frame_id": frame_id,
            "camera_timestamp_ticks": safe_value(frame.get_timestamp),
            "frame_status": str(frame_status),
            "frame_id_delta": frame_id_delta,
            "missing_before": missing_before,
            "width": safe_value(frame.get_width),
            "height": safe_value(frame.get_height),
            "pixel_format": str(safe_value(frame.get_pixel_format, "Unknown")),
        }
        bayer_np = None
        if frame_status == FrameStatus.Complete:
            try:
                # 8-bit Bayerのセンサー値を変換せずコピーする。
                bayer_np = frame.as_numpy_ndarray().copy()
                if bayer_np.ndim == 3 and bayer_np.shape[2] == 1:
                    bayer_np = bayer_np[:, :, 0]
                self.frame_count += 1
                self.last_error = None
            except Exception as exc:
                self.last_error = f"Bayerフレーム取得エラー: {exc}"
                metadata["conversion_error"] = str(exc)
        else:
            self.incomplete_count += 1
            self.last_error = f"不完全なフレームを受信しました: {frame_status}"
            self.incomplete_timestamps_ns.append(host_monotonic_ns)
            cutoff = host_monotonic_ns - 5_000_000_000
            while self.incomplete_timestamps_ns and self.incomplete_timestamps_ns[0] < cutoff:
                self.incomplete_timestamps_ns.popleft()
            if len(self.incomplete_timestamps_ns) >= 3:
                self.request_safer_bandwidth()

        # JPEG変換などより先にSDKバッファを返し、受信バッファの枯渇を防ぐ。
        try:
            cam.queue_frame(frame)
        except Exception as e:
            print("Error re-queuing frame:", e)

        if self.external_callback:
            try:
                self.external_callback(bayer_np, metadata)
            except Exception as exc:
                print("外部コールバックエラー:", exc)

    def stop(self):
        self.running = False

    def set_framerate(self, fps: float):
        """フリーランfpsを安全に変更し、失敗時は元の設定と配信を復旧する。"""
        with self.camera_control_lock:
            if self.cam is None or not self.streaming_active:
                return False, "Camera not initialized"
            if self.external_trigger_enabled:
                return False, "外部トリガーモード中はフレームレートを設定できません。"

            try:
                minimum, maximum = self.cam.AcquisitionFrameRate.get_range()
            except Exception as exc:
                return False, f"フレームレート範囲を取得できません: {exc}"
            if not minimum <= fps <= maximum:
                return False, f"フレームレートは {minimum:g}〜{maximum:g} fps で指定してください。"

            previous_fps = float(self.cam.AcquisitionFrameRate.get())
            previous_enabled = bool(self.cam.AcquisitionFrameRateEnable.get())
            stopped = False
            try:
                self.cam.stop_streaming()
                stopped = True
                self.streaming_active = False
                self.run_optional_command(self.cam, "AcquisitionStop")
                self.cam.AcquisitionFrameRateEnable.set(True)
                self.cam.AcquisitionFrameRate.set(float(fps))
                configured_fps = float(self.cam.AcquisitionFrameRate.get())
                self.start_camera_streaming()
                self.free_run_fps = configured_fps
                self.callback_timestamps_ns.clear()
                return True, configured_fps
            except Exception as exc:
                if not stopped:
                    return False, f"ストリームを停止できないためfpsを変更しませんでした: {exc}"
                restore_error = None
                try:
                    self.run_optional_command(self.cam, "AcquisitionStop")
                    self.cam.AcquisitionFrameRateEnable.set(previous_enabled)
                    if previous_enabled:
                        self.cam.AcquisitionFrameRate.set(previous_fps)
                    self.start_camera_streaming()
                except Exception as restore_exc:
                    restore_error = restore_exc
                if restore_error:
                    return False, f"fps設定失敗: {exc}; 映像復旧にも失敗: {restore_error}"
                return False, f"fps設定に失敗したため {previous_fps:g} fpsへ戻しました: {exc}"

#######################################
# FrameStreamer クラス
#######################################
class FrameStreamer:
    EVENT_FIELDS = (
        "session_id", "sequence", "stream_epoch", "callback_index", "host_utc_iso",
        "host_utc_ns", "host_monotonic_ns", "host_delta_ms",
        "camera_frame_id", "camera_timestamp_ticks", "camera_timestamp_delta_ticks",
        "frame_status", "frame_id_delta", "missing_before", "camera_width", "camera_height",
        "width", "height", "roi_offset_x", "roi_offset_y",
        "camera_pixel_format", "pixel_format", "queue_result", "filename", "queue_depth", "error",
    )

    def __init__(self, save_location, display_factor=0.5):
        self.save_location = save_location
        self.display_factor = display_factor
        self.save_filename = ""  # ファイル名設定
        self.preview = LatestFramePreview(
            "Frame", "frame", self.encode_preview_frame,
            preset=DEFAULT_PREVIEW_PRESET,
            auto_degrade=DEFAULT_PREVIEW_AUTO_DEGRADE,
        )
        self.preview.start()
        self.recording = False
        self.recording_queue = None
        self.image_thread = None
        self.recording_lock = threading.RLock()
        self.events_file = None
        self.events_writer = None
        self.recording_session_id = None
        self.recording_folder = None
        self.recording_sequence = 0
        self.recording_started_utc_ns = None
        self.recording_previous_frame_id = None
        self.recording_previous_camera_timestamp = None
        self.recording_previous_host_monotonic_ns = None
        self.recording_complete_count = 0
        self.recording_incomplete_count = 0
        self.recording_frame_gap_count = 0
        self.recording_queue_drop_count = 0
        self.last_recording_summary = {}
        self.connection_file = None
        self.connection_writer = None
        self.recording_restart_baseline = 0
        self.recording_fallback_baseline = 0
        self.recording_previous_epoch = None
        self.recording_capture_stopped_utc_ns = None
        self.recording_finalizing = False
        self.recording_finalizer_thread = None
        self.camera_restart_count = 0
        self.successful_reconnections = 0
        self.connection_state = "connecting"
        self.camera_supervisor_running = True
        self.capture_active_until = 0.0
        self.camera_retry_wakeup = threading.Event()

        self.cam_thread = self.create_camera_thread(stream_epoch=0)
        self.camera_supervisor_thread = threading.Thread(
            target=self.camera_supervisor_loop, daemon=True)
        self.camera_supervisor_thread.start()

    def create_camera_thread(self, stream_epoch):
        previous = getattr(self, "cam_thread", None)
        thread = CameraThread(
            free_run_fps=previous.free_run_fps if previous else DEFAULT_FRAME_RATE,
            trigger_enabled=previous.external_trigger_enabled if previous else DEFAULT_TRIGGER_ENABLED,
            trigger_source=previous.external_trigger_source if previous else DEFAULT_TRIGGER_SOURCE,
            trigger_activation=(
                previous.external_trigger_activation if previous else DEFAULT_TRIGGER_ACTIVATION),
            bandwidth_preset=previous.bandwidth_preset if previous else DEFAULT_BANDWIDTH_PRESET,
            auto_fallback_count=previous.auto_fallback_count if previous else 0,
            stream_epoch=stream_epoch,
        )
        thread.external_callback = self.handle_frame
        return thread

    def record_connection_event(self, event, message, stream_epoch):
        with self.recording_lock:
            if self.connection_writer is None:
                return
            self.connection_writer.writerow({
                "session_id": self.recording_session_id,
                "host_utc_ns": time.time_ns(),
                "host_monotonic_ns": time.monotonic_ns(),
                "stream_epoch": stream_epoch,
                "event": event,
                "message": message,
            })
            self.connection_file.flush()

    def set_capture_active(self, active, lease_seconds=12.0):
        """撮影画面が見えている間だけ接続試行を許可する。"""
        self.capture_active_until = (
            time.monotonic() + max(1.0, float(lease_seconds)) if active else 0.0)
        self.camera_retry_wakeup.set()
        return self.capture_retry_active()

    def capture_retry_active(self):
        return time.monotonic() < self.capture_active_until

    def camera_supervisor_loop(self):
        was_connected = False
        retry_delay = 1.0
        while self.camera_supervisor_running:
            thread = self.cam_thread
            connected = thread.is_alive() and thread.streaming_active
            if connected and not was_connected:
                self.connection_state = "connected"
                if thread.stream_epoch > 0:
                    self.successful_reconnections += 1
                    self.record_connection_event(
                        "reconnected", "フレームカメラへ再接続しました。", thread.stream_epoch)
                was_connected = True
                retry_delay = 1.0
            if not thread.is_alive():
                previously_started = thread.ident is not None
                if was_connected:
                    self.record_connection_event(
                        "disconnected", thread.last_error or "取得スレッドが終了しました。",
                        thread.stream_epoch)
                was_connected = False
                if not self.capture_retry_active():
                    self.connection_state = "retry_paused"
                    self.camera_retry_wakeup.wait(0.5)
                    self.camera_retry_wakeup.clear()
                    continue
                self.connection_state = "reconnecting" if previously_started else "connecting"
                if not self.camera_supervisor_running:
                    break
                if previously_started:
                    self.camera_retry_wakeup.wait(retry_delay)
                    self.camera_retry_wakeup.clear()
                if not self.camera_supervisor_running:
                    break
                if not self.capture_retry_active():
                    continue
                if not previously_started:
                    thread.start()
                    time.sleep(0.25)
                    continue
                retry_delay = min(5.0, retry_delay * 2)
                self.camera_restart_count += 1
                replacement = self.create_camera_thread(thread.stream_epoch + 1)
                self.cam_thread = replacement
                replacement.start()
            time.sleep(0.25)

    @staticmethod
    def utc_iso_from_ns(timestamp_ns):
        seconds, nanoseconds = divmod(timestamp_ns, 1_000_000_000)
        base = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
        return f"{base}.{nanoseconds:09d}Z"

    @staticmethod
    def frame_filename(sequence, host_utc_ns, frame_id):
        seconds, nanoseconds = divmod(host_utc_ns, 1_000_000_000)
        timestamp = datetime.fromtimestamp(seconds, timezone.utc).strftime("%Y%m%dT%H%M%S")
        frame_id_text = "none" if frame_id is None else str(frame_id)
        return f"{timestamp}.{nanoseconds:09d}Z_seq{sequence:06d}_id{frame_id_text}.pgm"

    @staticmethod
    def bayer_to_bgr(bayer_np, pixel_format):
        # OpenCVのBayer変換名は、入力の先頭画素名と直感的に逆になる。
        # BayerRG8のRサイトをBGR出力のRへ入れるにはBG2BGRを使う。
        conversion_codes = {
            "BayerRG8": cv2.COLOR_BAYER_BG2BGR,
            "BayerGR8": cv2.COLOR_BAYER_GB2BGR,
            "BayerGB8": cv2.COLOR_BAYER_GR2BGR,
            "BayerBG8": cv2.COLOR_BAYER_RG2BGR,
        }
        code = conversion_codes.get(pixel_format)
        if code is None:
            return cv2.cvtColor(bayer_np, cv2.COLOR_GRAY2BGR)
        return cv2.cvtColor(bayer_np, code)

    @staticmethod
    def shifted_bayer_format(pixel_format, offset_x, offset_y):
        normalized = str(pixel_format).replace("PixelFormat.", "")
        shifts = {
            "BayerRG8": (("BayerRG8", "BayerGR8"), ("BayerGB8", "BayerBG8")),
            "BayerGR8": (("BayerGR8", "BayerRG8"), ("BayerBG8", "BayerGB8")),
            "BayerGB8": (("BayerGB8", "BayerBG8"), ("BayerRG8", "BayerGR8")),
            "BayerBG8": (("BayerBG8", "BayerGB8"), ("BayerGR8", "BayerRG8")),
        }
        pattern = shifts.get(normalized)
        if pattern is None:
            return normalized
        return pattern[offset_y % 2][offset_x % 2]

    def crop_frame(self, frame_np):
        if g_value.img_trim_width > 0 and g_value.img_trim_height > 0:
            x = g_value.img_trim_offset_x
            y = g_value.img_trim_offset_y
            w = g_value.img_trim_width
            h = g_value.img_trim_height
            return frame_np[y:y+h, x:x+w]
        return frame_np

    def handle_frame(self, bayer_np, metadata):
        try:
            cropped_bayer = self.crop_frame(bayer_np) if bayer_np is not None else None
            metadata = dict(metadata)
            metadata["camera_pixel_format"] = metadata["pixel_format"]
            metadata["camera_width"] = metadata["width"]
            metadata["camera_height"] = metadata["height"]
            total_offset_x = g_value.sensor_offset_x + g_value.img_trim_offset_x
            total_offset_y = g_value.sensor_offset_y + g_value.img_trim_offset_y
            metadata["roi_offset_x"] = total_offset_x
            metadata["roi_offset_y"] = total_offset_y
            if cropped_bayer is not None:
                metadata["width"] = int(cropped_bayer.shape[1])
                metadata["height"] = int(cropped_bayer.shape[0])
            if cropped_bayer is not None and g_value.img_trim_width > 0:
                metadata["pixel_format"] = self.shifted_bayer_format(
                    metadata["pixel_format"],
                    total_offset_x,
                    total_offset_y,
                )

            # 表示変換に失敗しても、受信・欠損の監査行は必ず残す。
            self.record_frame_event(cropped_bayer, metadata)

            if cropped_bayer is not None:
                self.preview.submit((cropped_bayer, metadata["pixel_format"]))

        except Exception as exc:
            print("フレーム処理エラー:", exc)

    @staticmethod
    def encode_preview_frame(payload, settings):
        bayer_np, pixel_format = payload
        display_np = FrameStreamer.bayer_to_bgr(bayer_np, pixel_format)
        scale = settings["frame_scale"]
        if scale != 1.0:
            display_np = cv2.resize(
                display_np, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ret, jpeg_buf = cv2.imencode(
            ".jpg", display_np,
            [cv2.IMWRITE_JPEG_QUALITY, int(settings["jpeg_quality"])])
        return jpeg_buf.tobytes() if ret else None

    def update_preview_preferences(self, preset, auto_degrade, persist=True):
        status = self.preview.set_preferences(preset, auto_degrade)
        if persist:
            config = load_config()
            config.setdefault("preview", {})["preset"] = preset
            config["preview"]["autoDegrade"] = bool(auto_degrade)
            save_config(config)
        return status

    def record_frame_event(self, bayer_np, metadata):
        with self.recording_lock:
            if not self.recording or self.events_writer is None:
                return

            sequence = self.recording_sequence
            self.recording_sequence += 1
            frame_id = metadata["camera_frame_id"]
            stream_epoch = metadata["stream_epoch"]
            camera_timestamp = metadata["camera_timestamp_ticks"]
            host_monotonic_ns = metadata["host_monotonic_ns"]

            if self.recording_previous_epoch != stream_epoch:
                self.recording_previous_epoch = stream_epoch
                self.recording_previous_frame_id = None
                self.recording_previous_camera_timestamp = None

            frame_id_delta = None
            missing_before = 0
            if frame_id is not None and self.recording_previous_frame_id is not None:
                frame_id_delta = frame_id - self.recording_previous_frame_id
                if frame_id_delta > 1:
                    missing_before = frame_id_delta - 1
                    self.recording_frame_gap_count += missing_before
            if frame_id is not None:
                self.recording_previous_frame_id = frame_id

            camera_timestamp_delta = None
            if camera_timestamp is not None and self.recording_previous_camera_timestamp is not None:
                camera_timestamp_delta = camera_timestamp - self.recording_previous_camera_timestamp
            if camera_timestamp is not None:
                self.recording_previous_camera_timestamp = camera_timestamp

            host_delta_ms = None
            if self.recording_previous_host_monotonic_ns is not None:
                host_delta_ms = round(
                    (host_monotonic_ns - self.recording_previous_host_monotonic_ns) / 1_000_000, 3)
            self.recording_previous_host_monotonic_ns = host_monotonic_ns

            queue_result = "not_complete"
            filename = ""
            error = metadata.get("conversion_error", "")
            if bayer_np is None:
                self.recording_incomplete_count += 1
            else:
                self.recording_complete_count += 1
                filename = self.frame_filename(sequence, metadata["host_utc_ns"], frame_id)
                try:
                    # SDKバッファはframe_callback側ですでに所有コピー済み。
                    # hardware ROIなら同じ配列をwriterへ渡し、二重コピーを避ける。
                    queue_image = (
                        bayer_np if bayer_np.flags.c_contiguous
                        else bayer_np.copy(order="C"))
                    self.recording_queue.put_nowait({
                        "sequence": sequence,
                        "filename": filename,
                        "image": queue_image,
                    })
                    queue_result = "enqueued"
                except queue.Full:
                    queue_result = "queue_full"
                    filename = ""
                    self.recording_queue_drop_count += 1

            self.events_writer.writerow({
                "session_id": self.recording_session_id,
                "sequence": sequence,
                "stream_epoch": stream_epoch,
                "callback_index": metadata["callback_index"],
                "host_utc_iso": self.utc_iso_from_ns(metadata["host_utc_ns"]),
                "host_utc_ns": metadata["host_utc_ns"],
                "host_monotonic_ns": host_monotonic_ns,
                "host_delta_ms": host_delta_ms,
                "camera_frame_id": frame_id,
                "camera_timestamp_ticks": camera_timestamp,
                "camera_timestamp_delta_ticks": camera_timestamp_delta,
                "frame_status": metadata["frame_status"],
                "frame_id_delta": frame_id_delta,
                "missing_before": missing_before,
                "camera_width": metadata["camera_width"],
                "camera_height": metadata["camera_height"],
                "width": metadata["width"],
                "height": metadata["height"],
                "roi_offset_x": metadata["roi_offset_x"],
                "roi_offset_y": metadata["roi_offset_y"],
                "camera_pixel_format": metadata.get(
                    "camera_pixel_format", metadata["pixel_format"]),
                "pixel_format": metadata["pixel_format"],
                "queue_result": queue_result,
                "filename": filename,
                "queue_depth": self.recording_queue.qsize(),
                "error": error,
            })
            self.events_file.flush()

    def start_recording(self, session_id=None):
        if self.cam_thread.cam is None or not self.cam_thread.streaming_active:
            print("フレームカメラが初期化されていません。")
            return False
        with self.recording_lock:
            if self.recording or self.recording_finalizing:
                print("録画中、または直前の保存を完了処理中です。")
                return False

            if not session_id:
                session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            session_root = create_session_directory(self.save_location, session_id)
            frame_path = os.path.join(session_root, "frame")
            images_path = os.path.join(frame_path, "images")
            os.makedirs(images_path, exist_ok=True)
            events_path = os.path.join(frame_path, "frame_events.csv")
            results_path = os.path.join(frame_path, "saved_frames.csv")
            connection_path = os.path.join(frame_path, "connection_events.csv")
            if os.path.exists(events_path) or os.path.exists(results_path):
                raise ValueError("同じsession_idのフレーム記録が既に存在します。")

            save_config_snapshot(load_config(), session_root, prefix="frame_config")
            camera_settings = self.cam_thread.read_camera_settings()
            camera_settings["external_trigger"] = self.cam_thread.trigger_configuration()
            camera_settings["recording_roi"] = dict(self.cam_thread.roi_configuration)
            camera_settings["preview"] = self.preview.status()
            with open(os.path.join(frame_path, "camera_settings.json"),
                      "w", encoding="utf-8") as settings_file:
                json.dump(camera_settings, settings_file, ensure_ascii=False, indent=2)
            self.events_file = open(events_path, "w", newline="", encoding="utf-8")
            self.events_writer = csv.DictWriter(self.events_file, fieldnames=self.EVENT_FIELDS)
            self.events_writer.writeheader()
            self.events_file.flush()
            self.connection_file = open(connection_path, "w", newline="", encoding="utf-8")
            self.connection_writer = csv.DictWriter(self.connection_file, fieldnames=(
                "session_id", "host_utc_ns", "host_monotonic_ns", "stream_epoch",
                "event", "message",
            ))
            self.connection_writer.writeheader()
            self.connection_file.flush()

            self.recording_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
            self.image_thread = ImageWriterThread(
                self.recording_queue, images_path, results_path, session_id)
            self.image_thread.start()
            self.recording_session_id = session_id
            self.recording_folder = frame_path
            self.recording_sequence = 0
            self.recording_started_utc_ns = time.time_ns()
            self.recording_capture_stopped_utc_ns = None
            self.recording_previous_frame_id = None
            self.recording_previous_camera_timestamp = None
            self.recording_previous_host_monotonic_ns = None
            self.recording_complete_count = 0
            self.recording_incomplete_count = 0
            self.recording_frame_gap_count = 0
            self.recording_queue_drop_count = 0
            self.recording_previous_epoch = None
            self.recording_restart_baseline = self.successful_reconnections
            self.recording_fallback_baseline = self.cam_thread.auto_fallback_count
            self.recording = True
            self.record_connection_event(
                "recording_started", "録画開始時の接続状態", self.cam_thread.stream_epoch)
            print("Frame Bayer画像記録開始:", frame_path)
            return True

    def stop_recording(self, wait_for_writer=False):
        """撮影受付を即時停止し、キュー排出は別スレッドで完了させる。"""
        with self.recording_lock:
            if not self.recording:
                print("録画していません。")
                return False
            self.recording = False
            capture_stopped_ns = time.time_ns()
            self.recording_capture_stopped_utc_ns = capture_stopped_ns
            if self.events_file:
                self.events_file.flush()
                self.events_file.close()
            self.events_file = None
            self.events_writer = None
            if self.connection_file:
                self.connection_file.flush()
                self.connection_file.close()
            self.connection_file = None
            self.connection_writer = None
            recording_queue = self.recording_queue
            image_thread = self.image_thread
            recording_folder = self.recording_folder
            summary_values = {
                "session_id": self.recording_session_id,
                "started_utc_ns": self.recording_started_utc_ns,
                "capture_stopped_utc_ns": capture_stopped_ns,
                "callback_events": self.recording_sequence,
                "complete_frames": self.recording_complete_count,
                "incomplete_frames": self.recording_incomplete_count,
                "frame_id_missing_count": self.recording_frame_gap_count,
                "queue_drop_count": self.recording_queue_drop_count,
                "external_trigger": self.cam_thread.trigger_configuration(),
                "recording_roi": dict(self.cam_thread.roi_configuration),
                "camera_reconnections": (
                    self.successful_reconnections - self.recording_restart_baseline),
                "bandwidth_auto_fallbacks": (
                    self.cam_thread.auto_fallback_count - self.recording_fallback_baseline),
            }
            self.recording_finalizing = True

        recording_queue.put(None)
        self.recording_finalizer_thread = threading.Thread(
            target=self._finalize_recording,
            args=(recording_queue, image_thread, recording_folder, summary_values),
            daemon=True,
        )
        self.recording_finalizer_thread.start()
        if wait_for_writer:
            self.recording_finalizer_thread.join()
        return True

    def _finalize_recording(self, recording_queue, image_thread, recording_folder,
                            summary_values):
        summary = None
        try:
            image_thread.join()
            finalized_ns = time.time_ns()
            capture_stopped_ns = summary_values["capture_stopped_utc_ns"]
            started_ns = summary_values["started_utc_ns"]
            performance = image_thread.performance_status()
            summary = {
                **summary_values,
                "stopped_utc_ns": capture_stopped_ns,
                "finalized_utc_ns": finalized_ns,
                "duration_seconds": round((capture_stopped_ns - started_ns) / 1e9, 6),
                "writer_drain_seconds": round((finalized_ns - capture_stopped_ns) / 1e9, 6),
                "saved_frames": image_thread.saved_count,
                "write_failures": image_thread.failed_count,
                "pixel_storage": "lossless uncompressed 8-bit Bayer PGM (P5)",
                **performance,
            }
            with open(os.path.join(recording_folder, "frame_summary.json"),
                      "w", encoding="utf-8") as summary_file:
                json.dump(summary, summary_file, ensure_ascii=False, indent=2)
            print("Frame画像記録・保存完了:", recording_folder, summary)
        except Exception as exc:
            print("Frame保存完了処理エラー:", exc)
        finally:
            with self.recording_lock:
                if summary is not None:
                    self.last_recording_summary = summary
                if self.recording_queue is recording_queue:
                    self.recording_queue = None
                    self.image_thread = None
                    self.recording_folder = None
                    self.recording_session_id = None
                self.recording_finalizing = False

    def wait_for_recording_finalization(self, timeout=None):
        thread = self.recording_finalizer_thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        return not (thread and thread.is_alive())

    def update_save_settings(self, save_location, save_filename):
        self.save_location, self.save_filename = normalize_save_settings(save_location, save_filename)
        config = load_config()
        config.setdefault("recording", {})["save_location"] = self.save_location
        config["recording"]["file_prefix"] = self.save_filename
        save_config(config)
        print(f"保存設定更新: 保存先={self.save_location}, ファイル名のプレフィックス={self.save_filename}")
        return True

    def shutdown(self):
        if self.recording:
            self.stop_recording(wait_for_writer=True)
        elif self.recording_finalizing:
            self.wait_for_recording_finalization()
        self.camera_supervisor_running = False
        self.camera_retry_wakeup.set()
        current_thread = self.cam_thread
        if current_thread.ident is not None:
            current_thread.stop()
        self.camera_supervisor_thread.join(timeout=5)
        if current_thread.ident is not None:
            current_thread.join(timeout=10)
        # 監視停止直前に参照が更新されていた場合も、最後のスレッドを確実に止める。
        if self.cam_thread.ident is not None:
            self.cam_thread.stop()
            self.cam_thread.join(timeout=10)
        self.preview.stop()
        self.preview.join(timeout=5)

#######################################
# HTTP API（カメラ制御とは別モジュール）
#######################################
app = create_frame_app(
    lambda: frame_streamer_instance,
    bandwidth_presets=BANDWIDTH_PRESETS,
    default_trigger_source=DEFAULT_TRIGGER_SOURCE,
    default_trigger_activation=DEFAULT_TRIGGER_ACTIVATION,
)


def run_flask_server(port):
    app.run(host="127.0.0.1", port=port, debug=False)


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

    stop_requested = threading.Event()
    signal.signal(signal.SIGINT, lambda _signum, _frame: stop_requested.set())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_requested.set())
    print("Frame Streamer 起動中。CTRL+C で終了します。")
    try:
        while not stop_requested.wait(1):
            pass
    finally:
        print("シャットダウン中...")
        frame_streamer_instance.shutdown()
