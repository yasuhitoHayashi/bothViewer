import json
import os
import tempfile
import unittest
from unittest import mock

import cv2
import numpy as np

from bothviewer.core.recordings import (
    list_sessions, playback_manifest, render_preview_jpeg, session_detail,
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
            output.write("sequence,status\n0,complete\n1,complete\n")
        with open(os.path.join(evs, "triggers.csv"), "w", encoding="utf-8") as output:
            output.write("stream_epoch,evs_timestamp_us,polarity\n0,1000000,1\n")
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


if __name__ == "__main__":
    unittest.main()
