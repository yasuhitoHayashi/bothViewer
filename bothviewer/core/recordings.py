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

from bothviewer.core.synchronization import load_or_rebuild_synchronization


SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
IMAGE_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+\.pgm$")
EVS_ROLES = ("evs", "evs_a", "evs_b")
_RAW_STATE_LOCK = threading.Lock()
_RAW_STATES = OrderedDict()
_RAW_STATE_LIMIT = 4
# EventCD is 16 bytes in OpenEB 5.2.  This raises the SDK default from 10M
# (about 160 MB) to 20M events (about 320 MB) without making each cached reader
# excessively large on lower-memory acquisition PCs.
RAW_READER_MAX_EVENTS = 20_000_000
RAW_INSPECTION_SLICE_US = 50_000


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
    cached_synchronization = _read_json(
        os.path.join(session_path, "synchronization_summary.json"))
    if cached_synchronization:
        synchronization = cached_synchronization
    if not synchronization or not synchronization.get("generated", True):
        synchronization, _ = load_or_rebuild_synchronization(
            session_path, os.path.basename(session_path))
    if not frame:
        frame = _read_json(os.path.join(session_path, "frame", "frame_summary.json"))
    if not evs:
        evs = _read_json(os.path.join(session_path, "evs", "evs_summary.json"))
    evs_sources = session.get("evs_sources") if isinstance(
        session.get("evs_sources"), dict) else {}
    for role in EVS_ROLES:
        summary = _read_json(os.path.join(session_path, role, "evs_summary.json"))
        if summary:
            evs_sources[role] = summary
    if not evs and evs_sources:
        evs = next(iter(evs_sources.values()))
    session["evs_sources"] = evs_sources
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
                    any(os.path.isdir(os.path.join(session_path, role)) for role in EVS_ROLES)):
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
                "evs_roles": sorted(session.get("evs_sources", {})),
                "kind": "both" if frame else (
                    "evs_dual" if len(session.get("evs_sources", {})) > 1 else "evs_single"),
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
            "frame/frame_events.csv", "frame/saved_frames.csv",
            "synchronization.jsonl", "synchronization_summary.json",
            "synchronization.csv", "frame/camera_settings.json", "session.json"):
        entry = _file_entry(session_path, relative_path)
        if entry:
            files.append(entry)
    for role in EVS_ROLES:
        for name in ("triggers.csv", "camera_settings.json", "evs_summary.json"):
            entry = _file_entry(session_path, f"{role}/{name}")
            if entry:
                files.append(entry)
        raw_dir = os.path.join(session_path, role)
        if os.path.isdir(raw_dir):
            for name in sorted(os.listdir(raw_dir)):
                if re.fullmatch(r"events(?:_\d+)?\.raw", name):
                    entry = _file_entry(session_path, f"{role}/{name}", 1)
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
        "evs_sources": session.get("evs_sources", {}),
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


def _safe_raw_path(raw_path):
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("RAWファイルを選択してください。")
    path = os.path.realpath(os.path.abspath(os.path.expanduser(raw_path.strip())))
    if os.path.splitext(path)[1].lower() != ".raw":
        raise ValueError(".rawファイルのみ開けます。")
    if not os.path.isfile(path):
        raise FileNotFoundError("RAWファイルが見つかりません。")
    return path


@lru_cache(maxsize=16)
def _inspect_raw_file_cached(raw_path, size_bytes, modified_ns):
    del size_bytes, modified_ns
    from metavision_core.event_io.raw_reader import RawReader

    reader = RawReader(
        raw_path, max_events=RAW_READER_MAX_EVENTS, do_time_shifting=True)
    height, width = reader.get_size()
    duration_us = 0
    event_count = 0
    while not reader.is_done():
        # A one-second request can exceed RawReader's rolling buffer on dense
        # recordings even though only the final timestamp is needed here.
        events = reader.load_delta_t(RAW_INSPECTION_SLICE_US)
        event_count += len(events)
        if len(events):
            duration_us = max(duration_us, int(events["t"][-1]))
    return {
        "duration_us": max(1, duration_us),
        "event_count": event_count,
        "sensor_width": int(width),
        "sensor_height": int(height),
    }


def inspect_raw_file(raw_path, interval_us=33_000):
    """Inspect a standalone Metavision RAW file and build a playback manifest."""
    path = _safe_raw_path(raw_path)
    stat = os.stat(path)
    interval_us = max(5_000, min(int(interval_us), 500_000))
    info = _inspect_raw_file_cached(path, stat.st_size, stat.st_mtime_ns)
    return {
        "filename": os.path.basename(path),
        "file_size_bytes": stat.st_size,
        "duration_us": info["duration_us"],
        "event_count": info["event_count"],
        "sensor_width": info["sensor_width"],
        "sensor_height": info["sensor_height"],
        "interval_us": interval_us,
        "segments": [{
            "epoch": 0,
            "filename": os.path.basename(path),
            "start_us": 0,
            "end_us": info["duration_us"],
        }],
        "source_type": "standalone_raw",
    }


def _validate_evs_role(role):
    if role not in EVS_ROLES:
        raise ValueError("EVS roleが不正です。")
    return role


def evs_playback_manifest(save_location, session_id, role="evs", interval_us=33_000):
    session_path = _safe_session_path(save_location, session_id)
    role = _validate_evs_role(role)
    summary = _read_json(os.path.join(session_path, role, "evs_summary.json"))
    raw_path = os.path.join(session_path, role, "events.raw")
    if not os.path.isfile(raw_path):
        raise FileNotFoundError(f"{role}/events.rawがありません。")
    duration_us = max(1, round(float(summary.get("duration_seconds", 0) or 0) * 1_000_000))
    interval_us = max(5_000, min(int(interval_us), 500_000))
    segment_starts = {0: 0}
    connection_path = os.path.join(session_path, role, "connection_events.csv")
    started_ns = int(summary.get("started_utc_ns", 0) or 0)
    if started_ns and os.path.isfile(connection_path):
        with open(connection_path, newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                if row.get("event") != "reconnected":
                    continue
                try:
                    epoch = int(row.get("stream_epoch") or 0)
                    relative_us = max(0, round(
                        (int(row["host_utc_ns"]) - started_ns) / 1000))
                except (KeyError, TypeError, ValueError):
                    continue
                segment_starts[epoch] = relative_us
    raw_files = list(summary.get("raw_files") or ["events.raw"])
    segments = []
    for index, filename in enumerate(raw_files):
        start_us = segment_starts.get(index, 0 if index == 0 else duration_us)
        next_start = segment_starts.get(index + 1, duration_us)
        segments.append({
            "epoch": index, "filename": filename, "start_us": start_us,
            "end_us": max(start_us, next_start),
        })
    times = list(range(0, duration_us + 1, interval_us))
    if times[-1] != duration_us:
        times.append(duration_us)
    return {
        "session_id": session_id,
        "role": role,
        "started_utc_ns": int(summary.get("started_utc_ns", 0) or 0),
        "duration_us": duration_us,
        "interval_us": interval_us,
        "times_us": times,
        "segments": segments,
    }


def trigger_timing_analysis(save_location, session_id, role="evs"):
    """Analyze periodic reference-trigger timing and identify suspicious edges."""
    session_path = _safe_session_path(save_location, session_id)
    role = _validate_evs_role(role)
    trigger_path = os.path.join(session_path, role, "triggers.csv")
    if not os.path.isfile(trigger_path):
        raise FileNotFoundError(f"{role}/triggers.csvがありません。")

    synchronization, _ = load_or_rebuild_synchronization(
        session_path, session_id, persist=False)
    reference_polarity = int(synchronization.get("reference_trigger_polarity", 1) or 0)
    with open(trigger_path, newline="", encoding="utf-8") as source:
        all_rows = list(csv.DictReader(source))
    rows = []
    for row in all_rows:
        try:
            if int(row.get("polarity", -1)) != reference_polarity:
                continue
            rows.append({
                "trigger_index": int(row.get("trigger_index", len(rows))),
                "stream_epoch": int(row.get("stream_epoch") or 0),
                "sensor_us": int(row["evs_timestamp_us"]),
                "host_ns": int(row.get("host_decode_utc_ns") or 0),
            })
        except (KeyError, TypeError, ValueError):
            continue
    if len(rows) < 2:
        return {
            "available": False,
            "reason": "周期判定に必要な基準エッジが2件以上ありません。",
            "reference_polarity": reference_polarity,
            "edge_count": len(rows),
        }

    intervals = []
    previous_by_epoch = {}
    for edge in rows:
        previous = previous_by_epoch.get(edge["stream_epoch"])
        previous_by_epoch[edge["stream_epoch"]] = edge
        if previous is None:
            continue
        interval_us = edge["sensor_us"] - previous["sensor_us"]
        if interval_us <= 0:
            continue
        intervals.append((edge, interval_us))
    if not intervals:
        return {
            "available": False,
            "reason": "有効なトリガー間隔を計算できません。",
            "reference_polarity": reference_polarity,
            "edge_count": len(rows),
        }

    values = np.asarray([item[1] for item in intervals], dtype=np.float64)
    expected_us = float(np.median(values))
    deviations = np.abs(values - expected_us)
    mad_us = float(np.median(deviations))
    # At least 5% is allowed; stable clocks use six MADs to avoid flagging normal
    # quantization jitter. 50 us prevents over-sensitivity at high trigger rates.
    tolerance_us = max(expected_us * 0.05, mad_us * 6.0, 50.0)
    lower_us = max(0.0, expected_us - tolerance_us)
    upper_us = expected_us + tolerance_us

    first_frame_host_ns = 0
    frame_events_path = os.path.join(session_path, "frame", "frame_events.csv")
    if os.path.isfile(frame_events_path):
        try:
            with open(frame_events_path, newline="", encoding="utf-8") as source:
                first_frame = next(csv.DictReader(source), None)
            first_frame_host_ns = int((first_frame or {}).get("host_utc_ns") or 0)
        except (OSError, TypeError, ValueError):
            first_frame_host_ns = 0
    first_host_ns = first_frame_host_ns or next(
        (edge["host_ns"] for edge in rows if edge["host_ns"]), 0)

    result_intervals = []
    counts = {"normal": 0, "early": 0, "late": 0, "missing_suspected": 0}
    estimated_missing = 0
    for sequence, (edge, interval_us) in enumerate(intervals):
        status = "normal"
        missing_count = 0
        if interval_us < lower_us:
            status = "early"
        elif interval_us > upper_us:
            ratio = interval_us / expected_us if expected_us else 0
            if ratio >= 1.5:
                status = "missing_suspected"
                missing_count = max(1, round(ratio) - 1)
                estimated_missing += missing_count
            else:
                status = "late"
        counts[status] += 1
        host_relative_ms = (
            (edge["host_ns"] - first_host_ns) / 1e6
            if edge["host_ns"] and first_host_ns else None)
        result_intervals.append({
            "sequence": sequence,
            "trigger_index": edge["trigger_index"],
            "stream_epoch": edge["stream_epoch"],
            "interval_us": interval_us,
            "interval_ms": round(interval_us / 1000.0, 6),
            "deviation_us": round(interval_us - expected_us, 3),
            "status": status,
            "estimated_missing": missing_count,
            "host_relative_ms": (
                round(host_relative_ms, 3) if host_relative_ms is not None else None),
        })

    # Preserve every anomaly. Downsample only normal points for very long runs.
    anomalies = [item for item in result_intervals if item["status"] != "normal"]
    normal = [item for item in result_intervals if item["status"] == "normal"]
    if len(normal) > 4000:
        stride = math.ceil(len(normal) / 4000)
        normal = normal[::stride]
    display_intervals = sorted(normal + anomalies, key=lambda item: item["sequence"])
    p95_deviation_us = float(np.percentile(deviations, 95))
    return {
        "available": True,
        "session_id": session_id,
        "role": role,
        "reference_polarity": reference_polarity,
        "reference_edge": synchronization.get("reference_edge", "ExposureActive start"),
        "edge_count": len(rows),
        "interval_count": len(intervals),
        "expected_period_us": round(expected_us, 3),
        "expected_period_ms": round(expected_us / 1000.0, 6),
        "expected_frequency_hz": round(1_000_000.0 / expected_us, 6),
        "tolerance_us": round(tolerance_us, 3),
        "lower_period_ms": round(lower_us / 1000.0, 6),
        "upper_period_ms": round(upper_us / 1000.0, 6),
        "mad_us": round(mad_us, 3),
        "p95_deviation_us": round(p95_deviation_us, 3),
        "maximum_deviation_us": round(float(np.max(deviations)), 3),
        "anomaly_count": len(anomalies),
        "estimated_missing_edges": estimated_missing,
        "counts": counts,
        "intervals": display_intervals,
        "anomalies": anomalies,
        "normal_points_sampled": len(normal),
    }


def _first_raw_trigger_us(raw_path):
    from metavision_core.event_io.raw_reader import RawReader

    reader = RawReader(
        raw_path, max_events=RAW_READER_MAX_EVENTS, do_time_shifting=True,
        use_external_triggers=[0])
    for _ in range(10):
        reader.load_delta_t(RAW_INSPECTION_SLICE_US)
        triggers = reader.get_ext_trigger_events()
        if len(triggers):
            return int(triggers["t"][0])
        if reader.is_done():
            break
    return None


def playback_manifest(save_location, session_id):
    session_path = _safe_session_path(save_location, session_id)
    _, frame_summary, evs_summary, _ = _summary_for(session_path)
    triggers_path = os.path.join(session_path, "evs", "triggers.csv")
    synchronization, synchronization_rows = load_or_rebuild_synchronization(
        session_path, session_id)

    first_sensor_trigger_by_epoch = {}
    anchors_by_epoch = {}
    if os.path.isfile(triggers_path):
        with open(triggers_path, newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                try:
                    epoch = int(row.get("stream_epoch") or 0)
                    sensor_time = int(row["evs_timestamp_us"])
                    host_ns = int(row.get("host_decode_utc_ns") or 0)
                except (KeyError, TypeError, ValueError):
                    continue
                first_sensor_trigger_by_epoch.setdefault(epoch, sensor_time)
                if host_ns:
                    anchors_by_epoch.setdefault(epoch, []).append(
                        {"sensor_us": sensor_time, "host_ns": host_ns})

    raw_offsets = {}
    raw_files = {}
    evs_folder = os.path.join(session_path, "evs")
    candidate_epochs = set(first_sensor_trigger_by_epoch)
    candidate_epochs.update(range(len(evs_summary.get("raw_files") or [])))
    if os.path.isdir(evs_folder):
        for name in os.listdir(evs_folder):
            match = re.fullmatch(r"events(?:_(\d+))?\.raw", name)
            if match:
                candidate_epochs.add(int(match.group(1) or 0))
    for epoch in sorted(candidate_epochs):
        raw_name = _raw_name_for_epoch(epoch)
        raw_path = os.path.join(evs_folder, raw_name)
        if not os.path.isfile(raw_path):
            continue
        raw_files[epoch] = raw_name
        sensor_time = first_sensor_trigger_by_epoch.get(epoch)
        if sensor_time is not None:
            first_raw_trigger = _first_raw_trigger_us(raw_path)
            if first_raw_trigger is not None:
                raw_offsets[epoch] = sensor_time - first_raw_trigger

    epoch_start_host_ns = {0: int(evs_summary.get("started_utc_ns", 0) or 0)}
    connection_path = os.path.join(evs_folder, "connection_events.csv")
    if os.path.isfile(connection_path):
        with open(connection_path, newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                if row.get("event") != "reconnected":
                    continue
                try:
                    epoch_start_host_ns[int(row.get("stream_epoch") or 0)] = int(
                        row["host_utc_ns"])
                except (KeyError, TypeError, ValueError):
                    continue

    exact_by_filename = {
        row.get("frame_filename"): row for row in synchronization_rows
        if row.get("match_status") == "matched" and row.get("frame_filename")}
    initial_differences = []
    for row in synchronization_rows:
        if row.get("match_status") != "matched":
            continue
        try:
            initial_differences.append(float(row["host_time_difference_ms"]))
        except (KeyError, TypeError, ValueError):
            continue
        if len(initial_differences) >= 30:
            break
    if initial_differences:
        difference_baseline_ms = float(np.median(initial_differences))
        difference_mad_ms = float(np.median(np.abs(
            np.asarray(initial_differences) - difference_baseline_ms)))
        difference_tolerance_ms = max(5.0, difference_mad_ms * 6.0)
    else:
        difference_baseline_ms = 0.0
        difference_tolerance_ms = math.inf

    images_path = os.path.join(session_path, "frame", "images")
    saved_filenames = set()
    saved_path = os.path.join(session_path, "frame", "saved_frames.csv")
    if os.path.isfile(saved_path):
        with open(saved_path, newline="", encoding="utf-8") as source:
            saved_filenames = {
                row.get("filename", "") for row in csv.DictReader(source)
                if str(row.get("write_ok", "")).lower() in ("1", "true")}

    frame_rows = []
    frame_events_path = os.path.join(session_path, "frame", "frame_events.csv")
    if os.path.isfile(frame_events_path):
        with open(frame_events_path, newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                filename = row.get("filename", "")
                if (not filename or not IMAGE_PATTERN.fullmatch(filename) or
                        not os.path.isfile(os.path.join(images_path, filename)) or
                        (saved_filenames and filename not in saved_filenames)):
                    continue
                try:
                    frame_rows.append({
                        "sequence": int(row.get("sequence") or len(frame_rows)),
                        "filename": filename,
                        "camera_frame_id": row.get("camera_frame_id"),
                        "host_utc_ns": int(row.get("host_utc_ns") or 0),
                    })
                except (TypeError, ValueError):
                    continue
    # Compatibility with old sessions that only retain synchronization.csv.
    if not frame_rows:
        for row in synchronization_rows:
            filename = row.get("frame_filename", "")
            if (row.get("match_status") != "matched" or
                    not IMAGE_PATTERN.fullmatch(filename) or
                    not os.path.isfile(os.path.join(images_path, filename))):
                continue
            try:
                frame_rows.append({
                    "sequence": int(row.get("frame_sequence") or len(frame_rows)),
                    "filename": filename,
                    "camera_frame_id": row.get("camera_frame_id"),
                    "host_utc_ns": int(row.get("frame_host_utc_ns") or 0),
                })
            except (TypeError, ValueError):
                continue

    frames = []
    for frame in frame_rows:
        exact = exact_by_filename.get(frame["filename"])
        epoch = None
        raw_time_us = None
        sync_mode = None
        anchor_distance_ms = None
        if exact:
            try:
                difference_ms = float(exact.get("host_time_difference_ms", ""))
                reliable = abs(
                    difference_ms - difference_baseline_ms) <= difference_tolerance_ms
            except (TypeError, ValueError):
                reliable = True
            try:
                exact_epoch = int(exact.get("trigger_stream_epoch") or 0)
                if reliable and exact_epoch in raw_offsets:
                    epoch = exact_epoch
                    raw_time_us = max(
                        0, int(exact["evs_timestamp_us"]) - raw_offsets[exact_epoch])
                    sync_mode = "paired_trigger"
                    try:
                        anchor_distance_ms = abs(float(exact["host_time_difference_ms"]))
                    except (KeyError, TypeError, ValueError):
                        pass
            except (KeyError, TypeError, ValueError):
                pass

        if raw_time_us is None and frame["host_utc_ns"]:
            anchor_candidates = [
                (abs(anchor["host_ns"] - frame["host_utc_ns"]), candidate_epoch, anchor)
                for candidate_epoch, anchors in anchors_by_epoch.items()
                if candidate_epoch in raw_offsets
                for anchor in anchors]
            if anchor_candidates:
                distance_ns, epoch, anchor = min(
                    anchor_candidates, key=lambda item: item[0])
                sensor_time = anchor["sensor_us"] + round(
                    (frame["host_utc_ns"] - anchor["host_ns"]) / 1000)
                raw_time_us = max(0, sensor_time - raw_offsets[epoch])
                sync_mode = "host_time_interpolated"
                anchor_distance_ms = round(distance_ns / 1e6, 3)

        if raw_time_us is None and frame["host_utc_ns"] and raw_files:
            eligible = [
                (start_ns, candidate_epoch)
                for candidate_epoch, start_ns in epoch_start_host_ns.items()
                if candidate_epoch in raw_files and start_ns and
                start_ns <= frame["host_utc_ns"]]
            if eligible:
                start_ns, epoch = max(eligible)
            else:
                epoch = min(raw_files)
                start_ns = epoch_start_host_ns.get(epoch) or frame["host_utc_ns"]
            raw_time_us = max(0, round((frame["host_utc_ns"] - start_ns) / 1000))
            sync_mode = "segment_time_estimated"

        if raw_time_us is None or epoch not in raw_files:
            continue
        frames.append({
            **frame,
            "stream_epoch": epoch,
            "raw_time_us": raw_time_us,
            "sync_mode": sync_mode,
            "trigger_anchor_distance_ms": anchor_distance_ms,
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
        "sync_mode_counts": {
            mode: sum(frame.get("sync_mode") == mode for frame in frames)
            for mode in (
                "paired_trigger", "host_time_interpolated", "segment_time_estimated")},
        "partial_synchronization": any(
            frame.get("sync_mode") != "paired_trigger" for frame in frames),
        "recording_roi": roi,
        "event_window_us": 33_000,
    }


class _RawState:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.reader = None
        self.height = 720
        self.width = 1280

    def _new_reader(self):
        from metavision_core.event_io.raw_reader import RawReader
        self.reader = RawReader(
            self.path, max_events=RAW_READER_MAX_EVENTS, do_time_shifting=True)
        self.height, self.width = self.reader.get_size()

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


def _state_geometry(state):
    height = getattr(state, "height", 720)
    width = getattr(state, "width", 1280)
    if not isinstance(height, (int, np.integer)) or height <= 0:
        height = 720
    if not isinstance(width, (int, np.integer)) or width <= 0:
        width = 1280
    return int(height), int(width)


@lru_cache(maxsize=384)
def _render_event_jpeg(raw_path, center_us, window_us, max_width, palette):
    half_window = max(1_000, window_us // 2)
    start_us = max(0, center_us - half_window)
    state = _raw_state(raw_path)
    events = state.read_window(start_us, max(2_000, window_us))
    sensor_height, sensor_width = _state_geometry(state)
    frame = (
        np.full((sensor_height, sensor_width, 3), 128, dtype=np.uint8)
        if palette == "magenta_cyan" else
        np.full((sensor_height, sensor_width), 128, dtype=np.uint8))
    if len(events):
        valid = ((events["x"] < sensor_width) & (events["y"] < sensor_height))
        events = events[valid]
        negative = events[events["p"] == 0]
        positive = events[events["p"] != 0]
        if palette == "magenta_cyan":
            frame[negative["y"], negative["x"]] = (255, 255, 0)
            frame[positive["y"], positive["x"]] = (255, 0, 255)
        else:
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
                             window_us=33_000, max_width=720, role="evs",
                             palette="mono"):
    session_path = _safe_session_path(save_location, session_id)
    epoch = int(epoch)
    center_us = max(0, int(center_us))
    window_us = max(2_000, min(int(window_us), 500_000))
    max_width = max(160, min(int(max_width), 1280))
    if palette not in {"mono", "magenta_cyan"}:
        raise ValueError("EVS配色が不正です。")
    role = _validate_evs_role(role)
    raw_path = os.path.join(session_path, role, _raw_name_for_epoch(epoch))
    if not os.path.isfile(raw_path):
        raise FileNotFoundError("対応するEVS RAWセグメントがありません。")
    return _render_event_jpeg(raw_path, center_us, window_us, max_width, palette)


def render_raw_event_window_jpeg(raw_path, center_us, window_us=33_000,
                                 max_width=720, palette="mono"):
    """Render one time window from a standalone RAW file."""
    path = _safe_raw_path(raw_path)
    center_us = max(0, int(center_us))
    window_us = max(2_000, min(int(window_us), 500_000))
    max_width = max(160, min(int(max_width), 1280))
    if palette not in {"mono", "magenta_cyan"}:
        raise ValueError("EVS配色が不正です。")
    return _render_event_jpeg(path, center_us, window_us, max_width, palette)


@lru_cache(maxsize=384)
def _render_event_overlay_png(
        raw_path, center_us, window_us, max_width, max_events, palette):
    half_window = max(1_000, window_us // 2)
    start_us = max(0, center_us - half_window)
    state = _raw_state(raw_path)
    events = state.read_window(start_us, max(2_000, window_us))
    sensor_height, sensor_width = _state_geometry(state)
    overlay = np.zeros((sensor_height, sensor_width, 4), dtype=np.uint8)
    if len(events):
        valid = ((events["x"] < sensor_width) & (events["y"] < sensor_height))
        events = events[valid]
        if max_events and len(events) > max_events:
            stride = int(math.ceil(len(events) / max_events))
            events = events[::stride]
        negative = events[events["p"] == 0]
        positive = events[events["p"] != 0]
        if palette == "magenta_cyan":
            # BGRA: negative=cyan, positive=magenta
            overlay[negative["y"], negative["x"]] = (255, 255, 0, 230)
            overlay[positive["y"], positive["x"]] = (255, 0, 255, 230)
        else:
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
                             window_us=33_000, max_width=960, max_events=50_000,
                             role="evs", palette="mono"):
    session_path = _safe_session_path(save_location, session_id)
    epoch = int(epoch)
    center_us = max(0, int(center_us))
    window_us = max(2_000, min(int(window_us), 500_000))
    max_width = max(160, min(int(max_width), 1280))
    max_events = max(0, min(int(max_events), 2_000_000))
    if palette not in {"mono", "magenta_cyan"}:
        raise ValueError("EVS配色が不正です。")
    role = _validate_evs_role(role)
    raw_path = os.path.join(session_path, role, _raw_name_for_epoch(epoch))
    if not os.path.isfile(raw_path):
        raise FileNotFoundError("対応するEVS RAWセグメントがありません。")
    return _render_event_overlay_png(
        raw_path, center_us, window_us, max_width, max_events, palette)
