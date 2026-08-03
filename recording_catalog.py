"""Read-only catalog and preview helpers for recorded sessions."""

from __future__ import annotations

import csv
from collections import OrderedDict
from functools import lru_cache
import json
import math
import os
import re
import threading
from datetime import datetime, timezone

import cv2
import numpy as np


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+\.pgm$")
_RAW_STATE_LOCK = threading.Lock()
_RAW_STATES = OrderedDict()
_RAW_STATE_LIMIT = 4


def _read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as source:
            value = json.load(source)
            return value if isinstance(value, dict) else (default or {})
    except (OSError, ValueError):
        return default or {}


def _safe_session_path(save_location, session_id):
    if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError("セッションIDが不正です。")
    root = os.path.realpath(os.path.abspath(os.path.expanduser(save_location)))
    session_path = os.path.realpath(os.path.join(root, session_id))
    if os.path.commonpath((root, session_path)) != root:
        raise ValueError("保存先の外は参照できません。")
    if not os.path.isdir(session_path):
        raise FileNotFoundError("セッションが見つかりません。")
    return session_path


def _summary_for(session_path):
    session = _read_json(os.path.join(session_path, "session.json"))
    frame = session.get("frame") if isinstance(session.get("frame"), dict) else {}
    evs = session.get("evs") if isinstance(session.get("evs"), dict) else {}
    synchronization = (
        session.get("synchronization")
        if isinstance(session.get("synchronization"), dict) else {})
    if not frame:
        frame = _read_json(os.path.join(session_path, "frame", "frame_summary.json"))
    if not evs:
        evs = _read_json(os.path.join(session_path, "evs", "evs_summary.json"))
    return session, frame, evs, synchronization


def _utc_iso(nanoseconds, fallback_mtime):
    try:
        timestamp = float(nanoseconds) / 1_000_000_000
    except (TypeError, ValueError):
        timestamp = fallback_mtime
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _loss_counts(frame, synchronization):
    values = {
        "incomplete": int(frame.get("incomplete_frames", 0) or 0),
        "frame_id_missing": int(frame.get("frame_id_missing_count", 0) or 0),
        "queue_drops": int(frame.get("queue_drop_count", 0) or 0),
        "write_failures": int(frame.get("write_failures", 0) or 0),
        "frames_without_trigger": int(synchronization.get("frames_without_trigger", 0) or 0),
        "unmatched_triggers": int(synchronization.get("unmatched_reference_triggers", 0) or 0),
    }
    values["total"] = sum(values.values())
    return values


def list_sessions(save_location):
    root = os.path.realpath(os.path.abspath(os.path.expanduser(save_location)))
    if not os.path.isdir(root):
        return []
    sessions = []
    with os.scandir(root) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False) or not SESSION_ID_PATTERN.fullmatch(entry.name):
                continue
            session_path = entry.path
            if not (os.path.isfile(os.path.join(session_path, "session.json")) or
                    os.path.isdir(os.path.join(session_path, "frame")) or
                    os.path.isdir(os.path.join(session_path, "evs"))):
                continue
            session, frame, evs, synchronization = _summary_for(session_path)
            stat = entry.stat(follow_symlinks=False)
            losses = _loss_counts(frame, synchronization)
            sessions.append({
                "session_id": entry.name,
                "started_utc": _utc_iso(
                    frame.get("started_utc_ns") or evs.get("started_utc_ns"), stat.st_mtime),
                "duration_seconds": float(
                    frame.get("duration_seconds") or evs.get("duration_seconds") or 0),
                "saved_frames": int(frame.get("saved_frames", 0) or 0),
                "reference_edges": int(synchronization.get("reference_trigger_edges", 0) or 0),
                "matched": int(synchronization.get("matched", 0) or 0),
                "losses": losses,
                "complete": bool(session) and os.path.isfile(os.path.join(session_path, "session.json")),
            })
    sessions.sort(key=lambda item: item["started_utc"], reverse=True)
    return sessions


def _count_csv_rows(path):
    try:
        with open(path, newline="", encoding="utf-8") as source:
            return max(0, sum(1 for _ in source) - 1)
    except OSError:
        return 0


def _file_entry(session_path, relative_path, count=None):
    path = os.path.join(session_path, *relative_path.split("/"))
    if not os.path.isfile(path):
        return None
    if count is None and relative_path.endswith(".csv"):
        count = _count_csv_rows(path)
    return {
        "path": relative_path,
        "count": count,
        "size_bytes": os.path.getsize(path),
    }


def session_detail(save_location, session_id, preview_limit=4):
    session_path = _safe_session_path(save_location, session_id)
    session, frame, evs, synchronization = _summary_for(session_path)
    image_dir = os.path.join(session_path, "frame", "images")
    image_names = []
    image_bytes = 0
    if os.path.isdir(image_dir):
        with os.scandir(image_dir) as images:
            for image in images:
                if image.is_file(follow_symlinks=False) and IMAGE_PATTERN.fullmatch(image.name):
                    image_names.append(image.name)
                    image_bytes += image.stat(follow_symlinks=False).st_size
    image_names.sort()
    preview_names = image_names[:max(0, min(int(preview_limit), 12))]
    files = [{
        "path": "frame/images/*.pgm",
        "count": len(image_names),
        "size_bytes": image_bytes,
    }]
    for relative_path in (
            "frame/frame_events.csv", "frame/write_results.csv",
            "evs/triggers.csv", "synchronization.csv",
            "frame/camera_settings.json", "evs/camera_settings.json", "session.json"):
        entry = _file_entry(session_path, relative_path)
        if entry:
            files.append(entry)
    raw_dir = os.path.join(session_path, "evs")
    if os.path.isdir(raw_dir):
        for name in sorted(os.listdir(raw_dir)):
            if re.fullmatch(r"events(?:_\d+)?\.raw", name):
                entry = _file_entry(session_path, f"evs/{name}", 1)
                if entry:
                    files.append(entry)
    stat = os.stat(session_path)
    roi = frame.get("recording_roi") if isinstance(frame.get("recording_roi"), dict) else {}
    return {
        "session_id": session_id,
        "path": session_path,
        "started_utc": _utc_iso(
            frame.get("started_utc_ns") or evs.get("started_utc_ns"), stat.st_mtime),
        "duration_seconds": float(frame.get("duration_seconds") or evs.get("duration_seconds") or 0),
        "frame": frame,
        "evs": evs,
        "synchronization": synchronization,
        "losses": _loss_counts(frame, synchronization),
        "recording_roi": roi,
        "preview_images": preview_names,
        "files": files,
        "total_size_bytes": sum(item["size_bytes"] for item in files),
    }


def render_preview_jpeg(save_location, session_id, filename, max_width=720):
    session_path = _safe_session_path(save_location, session_id)
    if not isinstance(filename, str) or not IMAGE_PATTERN.fullmatch(filename):
        raise ValueError("画像名が不正です。")
    image_path = os.path.join(session_path, "frame", "images", filename)
    if not os.path.isfile(image_path):
        raise FileNotFoundError("画像が見つかりません。")
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 2:
        raise ValueError("8-bit Bayer PGMを読み込めません。")
    settings = _read_json(os.path.join(session_path, "frame", "camera_settings.json"))
    conversion = {
        "BayerRG8": cv2.COLOR_BAYER_BG2BGR,
        "BayerGR8": cv2.COLOR_BAYER_GB2BGR,
        "BayerGB8": cv2.COLOR_BAYER_GR2BGR,
        "BayerBG8": cv2.COLOR_BAYER_RG2BGR,
    }.get(str(settings.get("PixelFormat")), cv2.COLOR_BAYER_BG2BGR)
    color = cv2.cvtColor(image, conversion)
    if color.shape[1] > max_width:
        scale = max_width / color.shape[1]
        color = cv2.resize(color, (max_width, round(color.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
    success, encoded = cv2.imencode(".jpg", color, [cv2.IMWRITE_JPEG_QUALITY, 86])
    if not success:
        raise ValueError("プレビューをJPEGへ変換できません。")
    return encoded.tobytes()


def _raw_name_for_epoch(epoch):
    return "events.raw" if epoch == 0 else f"events_{epoch:03d}.raw"


def _first_raw_trigger_us(raw_path):
    from metavision_core.event_io.raw_reader import RawReader

    reader = RawReader(raw_path, do_time_shifting=True, use_external_triggers=[0])
    reader.load_delta_t(500_000)
    triggers = reader.get_ext_trigger_events()
    if not len(triggers):
        return None
    return int(triggers["t"][0])


def playback_manifest(save_location, session_id):
    session_path = _safe_session_path(save_location, session_id)
    _, frame_summary, _, _ = _summary_for(session_path)
    synchronization_path = os.path.join(session_path, "synchronization.csv")
    triggers_path = os.path.join(session_path, "evs", "triggers.csv")
    if not os.path.isfile(synchronization_path):
        raise FileNotFoundError("synchronization.csvがありません。")

    first_sensor_trigger_by_epoch = {}
    if os.path.isfile(triggers_path):
        with open(triggers_path, newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                try:
                    epoch = int(row.get("stream_epoch") or 0)
                    sensor_time = int(row["evs_timestamp_us"])
                except (KeyError, TypeError, ValueError):
                    continue
                first_sensor_trigger_by_epoch.setdefault(epoch, sensor_time)

    raw_offsets = {}
    raw_files = {}
    for epoch, sensor_time in first_sensor_trigger_by_epoch.items():
        raw_name = _raw_name_for_epoch(epoch)
        raw_path = os.path.join(session_path, "evs", raw_name)
        if not os.path.isfile(raw_path):
            continue
        first_raw_trigger = _first_raw_trigger_us(raw_path)
        if first_raw_trigger is None:
            continue
        raw_offsets[epoch] = sensor_time - first_raw_trigger
        raw_files[epoch] = raw_name

    frames = []
    images_path = os.path.join(session_path, "frame", "images")
    with open(synchronization_path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            if row.get("match_status") != "matched" or not row.get("frame_filename"):
                continue
            filename = row["frame_filename"]
            if not IMAGE_PATTERN.fullmatch(filename) or not os.path.isfile(
                    os.path.join(images_path, filename)):
                continue
            try:
                epoch = int(row.get("trigger_stream_epoch") or 0)
                sensor_time = int(row["evs_timestamp_us"])
                sequence = int(row.get("frame_sequence") or len(frames))
            except (TypeError, ValueError, KeyError):
                continue
            if epoch not in raw_offsets:
                continue
            frames.append({
                "sequence": sequence,
                "filename": filename,
                "stream_epoch": epoch,
                "raw_time_us": max(0, sensor_time - raw_offsets[epoch]),
                "camera_frame_id": row.get("camera_frame_id"),
                "host_utc_ns": int(row.get("frame_host_utc_ns") or 0),
            })
    frames.sort(key=lambda item: (item["host_utc_ns"], item["sequence"]))
    if not frames:
        raise ValueError("同期再生できる保存フレームがありません。")
    first_host_ns = frames[0]["host_utc_ns"]
    previous_ms = 0.0
    for frame in frames:
        relative_ms = max(0.0, (frame["host_utc_ns"] - first_host_ns) / 1_000_000)
        # 古い記録や不正なUTC値でも再生順序を後退させない。
        frame["relative_ms"] = round(max(previous_ms, relative_ms), 3)
        previous_ms = frame["relative_ms"]
    roi = frame_summary.get("recording_roi") if isinstance(
        frame_summary.get("recording_roi"), dict) else {}
    return {
        "session_id": session_id,
        "frames": frames,
        "raw_files": raw_files,
        "duration_ms": frames[-1]["relative_ms"],
        "frame_count": len(frames),
        "recording_roi": roi,
        "event_window_us": 33_000,
    }


class _RawState:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.reader = None

    def _new_reader(self):
        from metavision_core.event_io.raw_reader import RawReader
        self.reader = RawReader(self.path, do_time_shifting=True)

    def read_window(self, start_us, duration_us):
        with self.lock:
            if self.reader is None or start_us < self.reader.current_time:
                self._new_reader()
            if start_us > self.reader.current_time:
                self.reader.seek_time(start_us)
            return self.reader.load_delta_t(duration_us).copy()


def _raw_state(path):
    with _RAW_STATE_LOCK:
        state = _RAW_STATES.pop(path, None)
        if state is None:
            state = _RawState(path)
        _RAW_STATES[path] = state
        while len(_RAW_STATES) > _RAW_STATE_LIMIT:
            _RAW_STATES.popitem(last=False)
        return state


@lru_cache(maxsize=384)
def _render_event_jpeg(raw_path, center_us, window_us, max_width):
    half_window = max(1_000, window_us // 2)
    start_us = max(0, center_us - half_window)
    events = _raw_state(raw_path).read_window(start_us, max(2_000, window_us))
    frame = np.full((720, 1280), 128, dtype=np.uint8)
    if len(events):
        valid = ((events["x"] < 1280) & (events["y"] < 720))
        events = events[valid]
        negative = events[events["p"] == 0]
        positive = events[events["p"] != 0]
        frame[negative["y"], negative["x"]] = 0
        frame[positive["y"], positive["x"]] = 255
    frame = cv2.flip(frame, 1)
    if max_width < frame.shape[1]:
        scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, (max_width, round(frame.shape[0] * scale)),
                           interpolation=cv2.INTER_AREA)
    success, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not success:
        raise ValueError("EVSプロットをJPEGへ変換できません。")
    return encoded.tobytes()


def render_event_window_jpeg(save_location, session_id, epoch, center_us,
                             window_us=33_000, max_width=720):
    session_path = _safe_session_path(save_location, session_id)
    epoch = int(epoch)
    center_us = max(0, int(center_us))
    window_us = max(2_000, min(int(window_us), 500_000))
    max_width = max(160, min(int(max_width), 1280))
    raw_path = os.path.join(session_path, "evs", _raw_name_for_epoch(epoch))
    if not os.path.isfile(raw_path):
        raise FileNotFoundError("対応するEVS RAWセグメントがありません。")
    return _render_event_jpeg(raw_path, center_us, window_us, max_width)


@lru_cache(maxsize=384)
def _render_event_overlay_png(raw_path, center_us, window_us, max_width, max_events):
    half_window = max(1_000, window_us // 2)
    start_us = max(0, center_us - half_window)
    events = _raw_state(raw_path).read_window(start_us, max(2_000, window_us))
    overlay = np.zeros((720, 1280, 4), dtype=np.uint8)
    if len(events):
        valid = ((events["x"] < 1280) & (events["y"] < 720))
        events = events[valid]
        if max_events and len(events) > max_events:
            stride = int(math.ceil(len(events) / max_events))
            events = events[::stride]
        negative = events[events["p"] == 0]
        positive = events[events["p"] != 0]
        overlay[negative["y"], negative["x"]] = (0, 0, 0, 230)
        overlay[positive["y"], positive["x"]] = (255, 255, 255, 230)
    overlay = cv2.flip(overlay, 1)
    if max_width < overlay.shape[1]:
        scale = max_width / overlay.shape[1]
        overlay = cv2.resize(
            overlay, (max_width, round(overlay.shape[0] * scale)),
            interpolation=cv2.INTER_AREA)
    success, encoded = cv2.imencode(
        ".png", overlay, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    if not success:
        raise ValueError("EVSオーバーレイをPNGへ変換できません。")
    return encoded.tobytes()


def render_event_overlay_png(save_location, session_id, epoch, center_us,
                             window_us=33_000, max_width=960, max_events=50_000):
    session_path = _safe_session_path(save_location, session_id)
    epoch = int(epoch)
    center_us = max(0, int(center_us))
    window_us = max(2_000, min(int(window_us), 500_000))
    max_width = max(160, min(int(max_width), 1280))
    max_events = max(0, min(int(max_events), 2_000_000))
    raw_path = os.path.join(session_path, "evs", _raw_name_for_epoch(epoch))
    if not os.path.isfile(raw_path):
        raise FileNotFoundError("対応するEVS RAWセグメントがありません。")
    return _render_event_overlay_png(
        raw_path, center_us, window_us, max_width, max_events)
