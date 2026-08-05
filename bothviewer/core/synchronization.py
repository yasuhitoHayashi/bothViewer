"""Build and cache the frame-to-EVS trigger synchronization audit.

The camera audit files are the source of truth.  The synchronization index is a
derived cache and can therefore be reconstructed after an interrupted stop.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile


SYNCHRONIZATION_FIELDS = (
    "session_id", "sync_index", "match_status", "frame_sequence",
    "frame_stream_epoch", "camera_frame_id", "frame_status", "frame_filename",
    "frame_host_utc_ns", "trigger_stream_epoch", "trigger_index",
    "evs_timestamp_us", "trigger_polarity", "trigger_channel_id",
    "trigger_host_decode_utc_ns", "host_time_difference_ms",
)
INDEX_FILENAME = "synchronization.jsonl"
SUMMARY_FILENAME = "synchronization_summary.json"
LEGACY_CSV_FILENAME = "synchronization.csv"


def _paths(session_root):
    return {
        "frames": os.path.join(session_root, "frame", "frame_events.csv"),
        "triggers": os.path.join(session_root, "evs", "triggers.csv"),
        "saved": os.path.join(session_root, "frame", "saved_frames.csv"),
        "settings": os.path.join(session_root, "frame", "camera_settings.json"),
    }


def _source_fingerprint(paths):
    fingerprint = {}
    for name, path in paths.items():
        try:
            stat = os.stat(path)
        except OSError:
            continue
        fingerprint[name] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    return fingerprint


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _integer(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _empty_row(session_id, output_index, status):
    row = {field: "" for field in SYNCHRONIZATION_FIELDS}
    row.update({
        "session_id": session_id,
        "sync_index": output_index,
        "match_status": status,
    })
    return row


def _reference_polarity(settings_path):
    try:
        with open(settings_path, encoding="utf-8") as source:
            settings = json.load(source)
    except (OSError, ValueError):
        return 1
    external_trigger = settings.get("external_trigger")
    if not isinstance(external_trigger, dict):
        external_trigger = {}
    output_inverter = settings.get(
        "LineInverter", external_trigger.get("output_inverter"))
    return 0 if output_inverter is True else 1


def build_synchronization_records(session_root, session_id):
    """Return ``(summary, rows)`` reconstructed from the primary audit logs."""
    paths = _paths(session_root)
    if not os.path.isfile(paths["frames"]) or not os.path.isfile(paths["triggers"]):
        return ({
            "generated": False,
            "reason": "frame_events.csv または triggers.csv がありません。",
            "source": "primary_audit_logs",
        }, [])

    try:
        frames = _read_csv(paths["frames"])
        all_triggers = _read_csv(paths["triggers"])
        saved_rows = _read_csv(paths["saved"]) if os.path.isfile(paths["saved"]) else []
    except (OSError, csv.Error) as exc:
        return ({
            "generated": False,
            "reason": f"監査ログを読み込めません: {exc}",
            "source": "primary_audit_logs",
        }, [])

    saved_filenames = {
        row.get("filename", "") for row in saved_rows
        if str(row.get("write_ok", "")).lower() in ("1", "true")
    }
    reference_polarity = _reference_polarity(paths["settings"])
    triggers = [
        row for row in all_triggers
        if _integer(row.get("polarity"), -1) == reference_polarity
    ] or all_triggers

    start_offset = 0
    if frames and triggers:
        first_frame_host_ns = _integer(frames[0].get("host_utc_ns"))
        start_offset = min(
            range(len(triggers)),
            key=lambda index: abs(
                _integer(triggers[index].get("host_decode_utc_ns")) - first_frame_host_ns),
        )

    rows = []
    matched = 0
    missing_image = 0
    missing_trigger = 0
    missing_frame_events = 0

    def append_trigger(row, trigger):
        row.update({
            "trigger_index": trigger.get("trigger_index", ""),
            "trigger_stream_epoch": trigger.get("stream_epoch", ""),
            "evs_timestamp_us": trigger.get("evs_timestamp_us", ""),
            "trigger_polarity": trigger.get("polarity", ""),
            "trigger_channel_id": trigger.get("channel_id", ""),
            "trigger_host_decode_utc_ns": trigger.get("host_decode_utc_ns", ""),
        })

    for trigger in triggers[:start_offset]:
        row = _empty_row(session_id, len(rows), "trigger_before_first_frame")
        append_trigger(row, trigger)
        rows.append(row)

    trigger_position = start_offset
    for frame in frames:
        for _ in range(max(0, _integer(frame.get("missing_before")))):
            trigger = triggers[trigger_position] if trigger_position < len(triggers) else None
            row = _empty_row(session_id, len(rows), "frame_event_missing")
            if trigger:
                append_trigger(row, trigger)
            rows.append(row)
            missing_frame_events += 1
            trigger_position += 1

        trigger = triggers[trigger_position] if trigger_position < len(triggers) else None
        filename = frame.get("filename", "")
        has_image = bool(filename) and filename in saved_filenames
        if not has_image:
            missing_image += 1
        if trigger:
            matched += 1
        status = "matched"
        if not trigger:
            status = "missing_trigger"
            missing_trigger += 1
        elif not has_image:
            status = "trigger_without_saved_image"
        row = _empty_row(session_id, len(rows), status)
        row.update({
            "frame_sequence": frame.get("sequence", ""),
            "frame_stream_epoch": frame.get("stream_epoch", ""),
            "camera_frame_id": frame.get("camera_frame_id", ""),
            "frame_status": frame.get("frame_status", ""),
            "frame_filename": filename,
            "frame_host_utc_ns": frame.get("host_utc_ns", ""),
        })
        if trigger:
            append_trigger(row, trigger)
            row["host_time_difference_ms"] = round(
                (_integer(trigger.get("host_decode_utc_ns")) -
                 _integer(frame.get("host_utc_ns"))) / 1e6, 3)
        rows.append(row)
        trigger_position += 1

    for trigger in triggers[trigger_position:]:
        row = _empty_row(session_id, len(rows), "trigger_without_frame_event")
        append_trigger(row, trigger)
        rows.append(row)

    summary = {
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
        "source": "primary_audit_logs",
        "index_format": "jsonl",
    }
    return summary, rows


def _atomic_write(path, writer):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(path)}.", dir=os.path.dirname(path), text=True)
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as output:
            writer(output)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        try:
            os.unlink(temporary_path)
        except OSError:
            pass
        raise


def _persist_index(session_root, summary, rows, fingerprint, write_legacy_csv=True):
    metadata = {
        "record_type": "metadata",
        "schema_version": 1,
        "source_fingerprint": fingerprint,
        "summary": summary,
    }

    def write_jsonl(output):
        output.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for row in rows:
            output.write(json.dumps(
                {"record_type": "match", **row}, ensure_ascii=False) + "\n")

    _atomic_write(os.path.join(session_root, INDEX_FILENAME), write_jsonl)
    _atomic_write(
        os.path.join(session_root, SUMMARY_FILENAME),
        lambda output: json.dump(summary, output, ensure_ascii=False, indent=2),
    )
    if write_legacy_csv:
        def write_csv(output):
            writer = csv.DictWriter(output, fieldnames=SYNCHRONIZATION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        _atomic_write(os.path.join(session_root, LEGACY_CSV_FILENAME), write_csv)


def build_synchronization_report(session_root, session_id, write_legacy_csv=True):
    """Rebuild and atomically persist the derived synchronization index.

    Cache write failures do not invalidate the in-memory report because the audit
    logs remain sufficient for later reconstruction.
    """
    summary, rows = build_synchronization_records(session_root, session_id)
    if not summary.get("generated"):
        return summary
    try:
        _persist_index(
            session_root, summary, rows, _source_fingerprint(_paths(session_root)),
            write_legacy_csv=write_legacy_csv)
        summary["cache_written"] = True
    except OSError as exc:
        summary["cache_written"] = False
        summary["cache_write_error"] = str(exc)
    return summary


def _load_jsonl_index(path, expected_fingerprint):
    try:
        with open(path, encoding="utf-8") as source:
            first_line = source.readline()
            metadata = json.loads(first_line)
            if (metadata.get("record_type") != "metadata" or
                    metadata.get("source_fingerprint") != expected_fingerprint):
                return None
            rows = []
            for line in source:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.pop("record_type", None) == "match":
                    rows.append(record)
            return metadata.get("summary", {}), rows
    except (OSError, ValueError):
        return None


def _load_legacy_csv(session_root):
    path = os.path.join(session_root, LEGACY_CSV_FILENAME)
    if not os.path.isfile(path):
        return None
    try:
        rows = _read_csv(path)
    except (OSError, csv.Error):
        return None
    return ({
        "generated": True,
        "matched": sum(row.get("match_status") == "matched" for row in rows),
        "source": "legacy_synchronization_csv",
        "index_format": "csv",
    }, rows)


def load_or_rebuild_synchronization(session_root, session_id, persist=True):
    """Load a current index or reconstruct it from the primary audit logs.

    The returned rows remain usable even when the recording directory is
    read-only or disk pressure prevents the rebuilt cache from being written.
    """
    paths = _paths(session_root)
    fingerprint = _source_fingerprint(paths)
    cached = _load_jsonl_index(os.path.join(session_root, INDEX_FILENAME), fingerprint)
    if cached is not None:
        summary, rows = cached
        return {**summary, "cache_written": True}, rows

    summary, rows = build_synchronization_records(session_root, session_id)
    if summary.get("generated"):
        summary = {**summary, "reconstructed": True}
        if persist:
            try:
                _persist_index(session_root, summary, rows, fingerprint)
                summary["cache_written"] = True
            except OSError as exc:
                summary["cache_written"] = False
                summary["cache_write_error"] = str(exc)
        return summary, rows

    legacy = _load_legacy_csv(session_root)
    if legacy is not None:
        return legacy
    return summary, []
