"""Build the frame-to-EVS trigger synchronization audit."""

import csv
import json
import os


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

