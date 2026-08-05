import json
import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from bothviewer.core.recordings import (
    _render_event_overlay_png, list_sessions, playback_manifest,
    render_preview_jpeg, session_detail, trigger_timing_analysis,
)


class RecordingCatalogTests(unittest.TestCase):
    def make_session(self, root, session_id="20260803T010203.000Z_sample_abc123"):
        session = os.path.join(root, session_id)
        images = os.path.join(session, "frame", "images")
        evs = os.path.join(session, "evs")
        os.makedirs(images)
        os.makedirs(evs)
        summary = {
            "schema_version": 1,
            "session_id": session_id,
            "frame": {
                "started_utc_ns": 1_785_700_000_000_000_000,
                "duration_seconds": 2.5,
                "complete_frames": 2,
                "saved_frames": 2,
                "incomplete_frames": 1,
                "frame_id_missing_count": 0,
                "queue_drop_count": 0,
                "write_failures": 0,
                "recording_roi": {"width": 8, "height": 8},
            },
            "evs": {"trigger_events": 4},
            "synchronization": {
                "reference_trigger_edges": 2,
                "matched": 2,
                "frames_without_trigger": 0,
                "unmatched_reference_triggers": 0,
            },
        }
        with open(os.path.join(session, "session.json"), "w", encoding="utf-8") as output:
            json.dump(summary, output)
        with open(os.path.join(session, "frame", "camera_settings.json"),
                  "w", encoding="utf-8") as output:
            json.dump({"PixelFormat": "BayerRG8"}, output)
        for index in range(2):
            image = np.arange(64, dtype=np.uint8).reshape(8, 8) + index
            with open(os.path.join(images, f"frame_{index:03d}.pgm"), "wb") as output:
                output.write(b"P5\n8 8\n255\n")
                output.write(image.tobytes())
        with open(os.path.join(session, "frame", "frame_events.csv"),
                  "w", encoding="utf-8") as output:
            output.write(
                "sequence,host_utc_ns,camera_frame_id,frame_status,missing_before,filename\n"
                "0,2000000000,10,FrameStatus.Complete,0,frame_000.pgm\n"
                "1,2100000000,11,FrameStatus.Complete,0,frame_001.pgm\n")
        with open(os.path.join(session, "frame", "saved_frames.csv"),
                  "w", encoding="utf-8") as output:
            output.write(
                "filename,write_ok\n"
                "frame_000.pgm,1\n"
                "frame_001.pgm,1\n")
        with open(os.path.join(evs, "triggers.csv"), "w", encoding="utf-8") as output:
            output.write(
                "trigger_index,stream_epoch,evs_timestamp_us,polarity,channel_id,host_decode_utc_ns\n"
                "0,0,1000000,1,0,1900000000\n"
                "1,0,1010000,1,0,2000000000\n"
                "2,0,1110000,1,0,2100000000\n")
        with open(os.path.join(session, "synchronization.csv"),
                  "w", encoding="utf-8") as output:
            output.write(
                "match_status,frame_sequence,frame_filename,frame_host_utc_ns,"
                "trigger_stream_epoch,evs_timestamp_us,camera_frame_id\n"
                "matched,0,frame_000.pgm,2000000000,0,1010000,10\n"
                "matched,1,frame_001.pgm,2100000000,0,1110000,11\n")
        with open(os.path.join(evs, "events.raw"), "wb") as output:
            output.write(b"raw")
        return session_id

    def test_lists_and_summarizes_sessions(self):
        with tempfile.TemporaryDirectory() as root:
            session_id = self.make_session(root)
            sessions = list_sessions(root)
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0]["session_id"], session_id)
            self.assertEqual(sessions[0]["saved_frames"], 2)
            self.assertEqual(sessions[0]["losses"]["incomplete"], 1)

            detail = session_detail(root, session_id)
            self.assertEqual(detail["preview_images"], ["frame_000.pgm", "frame_001.pgm"])
            self.assertEqual(detail["recording_roi"], {"width": 8, "height": 8})
            self.assertGreater(detail["total_size_bytes"], 0)

    def test_rejects_paths_outside_recording_root(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                session_detail(root, "../outside")
            with self.assertRaises(ValueError):
                render_preview_jpeg(root, "bad/session", "frame.pgm")

    def test_renders_bayer_pgm_as_jpeg(self):
        with tempfile.TemporaryDirectory() as root:
            session_id = self.make_session(root)
            encoded = render_preview_jpeg(root, session_id, "frame_000.pgm")
            decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
            self.assertIsNotNone(decoded)
            self.assertEqual(decoded.shape[:2], (8, 8))

    def test_builds_real_time_synchronized_playback_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            session_id = self.make_session(root)
            with mock.patch(
                    "bothviewer.core.recordings._first_raw_trigger_us", return_value=100):
                manifest = playback_manifest(root, session_id)
            self.assertEqual(manifest["frame_count"], 2)
            self.assertEqual(manifest["duration_ms"], 100.0)
            self.assertEqual(manifest["frames"][0]["raw_time_us"], 10_100)
            self.assertEqual(manifest["frames"][1]["relative_ms"], 100.0)

    def test_rebuilds_playback_index_when_synchronization_csv_is_missing(self):
        with tempfile.TemporaryDirectory() as root:
            session_id = self.make_session(root)
            session_path = os.path.join(root, session_id)
            os.unlink(os.path.join(session_path, "synchronization.csv"))

            with mock.patch(
                    "bothviewer.core.recordings._first_raw_trigger_us", return_value=100):
                manifest = playback_manifest(root, session_id)

            self.assertEqual(manifest["frame_count"], 2)
            self.assertTrue(os.path.isfile(
                os.path.join(session_path, "synchronization.jsonl")))
            self.assertTrue(os.path.isfile(
                os.path.join(session_path, "synchronization_summary.json")))

    def test_playback_continues_after_frame_trigger_correspondence_is_lost(self):
        with tempfile.TemporaryDirectory() as root:
            session_id = self.make_session(root)
            session = os.path.join(root, session_id)
            image = np.arange(64, dtype=np.uint8).reshape(8, 8) + 2
            with open(os.path.join(session, "frame", "images", "frame_002.pgm"), "wb") as output:
                output.write(b"P5\n8 8\n255\n")
                output.write(image.tobytes())
            with open(os.path.join(session, "frame", "frame_events.csv"), "a", encoding="utf-8") as output:
                output.write("2,2200000000,12,FrameStatus.Complete,0,frame_002.pgm\n")
            with open(os.path.join(session, "frame", "saved_frames.csv"), "a", encoding="utf-8") as output:
                output.write("frame_002.pgm,1\n")

            with mock.patch(
                    "bothviewer.core.recordings._first_raw_trigger_us", return_value=100):
                manifest = playback_manifest(root, session_id)

            self.assertEqual(manifest["frame_count"], 3)
            self.assertEqual(manifest["duration_ms"], 200.0)
            self.assertEqual(manifest["frames"][-1]["sync_mode"], "host_time_interpolated")
            self.assertTrue(manifest["partial_synchronization"])

    def test_renders_magenta_cyan_event_overlay(self):
        events = np.array(
            [(10, 20, 0, 1), (11, 20, 1, 2)],
            dtype=[("x", "u2"), ("y", "u2"), ("p", "i2"), ("t", "i8")])
        state = mock.Mock()
        state.read_window.return_value = events
        _render_event_overlay_png.cache_clear()
        with mock.patch("bothviewer.core.recordings._raw_state", return_value=state):
            encoded = _render_event_overlay_png(
                "dummy.raw", 20_000, 33_000, 1280, 0, "magenta_cyan")
        decoded = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        self.assertEqual(decoded[20, 1280 - 1 - 10, :3].tolist(), [255, 255, 0])
        self.assertEqual(decoded[20, 1280 - 1 - 11, :3].tolist(), [255, 0, 255])

    def test_trigger_timing_analysis_detects_early_and_missing_edges(self):
        with tempfile.TemporaryDirectory() as root:
            session_id = "trigger-analysis"
            session = os.path.join(root, session_id)
            evs = os.path.join(session, "evs")
            os.makedirs(evs)
            with open(os.path.join(evs, "triggers.csv"), "w", encoding="utf-8") as output:
                output.write(
                    "trigger_index,stream_epoch,evs_timestamp_us,polarity,host_decode_utc_ns\n"
                    "0,0,0,1,1000000000\n"
                    "1,0,10000,1,1010000000\n"
                    "2,0,20000,1,1020000000\n"
                    "3,0,30000,1,1030000000\n"
                    "4,0,40000,1,1040000000\n"
                    "5,0,45000,1,1045000000\n"
                    "6,0,50000,1,1050000000\n"
                    "7,0,60000,1,1060000000\n"
                    "8,0,70000,1,1070000000\n"
                    "9,0,90000,1,1090000000\n")

            analysis = trigger_timing_analysis(root, session_id)

            self.assertTrue(analysis["available"])
            self.assertEqual(analysis["expected_period_ms"], 10.0)
            self.assertEqual(analysis["counts"]["early"], 2)
            self.assertEqual(analysis["counts"]["missing_suspected"], 1)
            self.assertEqual(analysis["estimated_missing_edges"], 1)
            self.assertEqual(analysis["anomaly_count"], 3)


if __name__ == "__main__":
    unittest.main()
