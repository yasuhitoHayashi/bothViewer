import json
import os
import tempfile
import unittest
from unittest import mock

from bothviewer.api.data import create_data_app
from bothviewer.core.recordings import evs_playback_manifest, list_sessions
from launcher import MODE_VIEWERS, ProcessSupervisor


class LaunchModeTests(unittest.TestCase):
    def make_supervisor(self, save_location):
        supervisor = ProcessSupervisor.__new__(ProcessSupervisor)
        supervisor.save_location = save_location
        return supervisor

    def test_all_six_modes_have_viewers_and_expected_services(self):
        self.assertEqual(len(MODE_VIEWERS), 6)
        supervisor = self.make_supervisor("/tmp/records")
        self.assertEqual(
            set(supervisor._command_map("capture_both", ["E1", ""])),
            {"EVS", "Frame"})
        self.assertEqual(
            set(supervisor._command_map("capture_evs_single", ["E1", ""])),
            {"EVS"})
        dual = supervisor._command_map("capture_evs_dual", ["E1", "E2"])
        self.assertEqual(set(dual), {"EVS-A", "EVS-B"})
        self.assertIn("evs_a", dual["EVS-A"])
        self.assertIn("evs_b", dual["EVS-B"])
        for mode in ("review_both", "review_evs_single", "review_evs_dual"):
            self.assertEqual(set(supervisor._command_map(mode, ["", ""])), {"Data"})

    def test_dual_capture_requires_two_distinct_serials(self):
        supervisor = self.make_supervisor("/tmp/records")
        for selectors in (["", ""], ["E1", ""], ["E1", "E1"]):
            with self.subTest(selectors=selectors), self.assertRaises(ValueError):
                supervisor._command_map("capture_evs_dual", selectors)

    def test_dual_evs_session_is_catalogued_and_has_role_manifests(self):
        with tempfile.TemporaryDirectory() as root:
            session = os.path.join(root, "dual-session")
            for index, role in enumerate(("evs_a", "evs_b")):
                folder = os.path.join(session, role)
                os.makedirs(folder)
                with open(os.path.join(folder, "events.raw"), "wb") as raw_file:
                    raw_file.write(b"raw")
                with open(os.path.join(folder, "evs_summary.json"), "w", encoding="utf-8") as source:
                    json.dump({
                        "session_id": "dual-session", "role": role,
                        "started_utc_ns": 1_000_000_000 + index * 2_000_000,
                        "duration_seconds": 1.0,
                    }, source)
            sessions = list_sessions(root)
            self.assertEqual(sessions[0]["kind"], "evs_dual")
            self.assertEqual(sessions[0]["evs_roles"], ["evs_a", "evs_b"])
            manifest = evs_playback_manifest(root, "dual-session", "evs_b")
            self.assertEqual(manifest["role"], "evs_b")
            self.assertEqual(manifest["duration_us"], 1_000_000)

            client = create_data_app(root).test_client()
            response = client.get("/recordings/dual-session/evs/evs_a/playback")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json()["playback"]["role"], "evs_a")

    def test_standalone_raw_can_be_opened_without_a_session(self):
        with tempfile.TemporaryDirectory() as root:
            raw_path = os.path.join(root, "standalone.raw")
            with open(raw_path, "wb") as raw_file:
                raw_file.write(b"raw")
            app = create_data_app(root)
            client = app.test_client()
            manifest = {
                "filename": "standalone.raw", "duration_us": 2_000_000,
                "event_count": 1234, "sensor_width": 1280, "sensor_height": 720,
                "interval_us": 33_000,
                "segments": [{"epoch": 0, "start_us": 0, "end_us": 2_000_000}],
                "source_type": "standalone_raw",
            }
            with mock.patch("bothviewer.api.data.inspect_raw_file", return_value=manifest):
                response = client.post("/raw-files/open", json={"path": raw_path})
            self.assertEqual(response.status_code, 200)
            playback = response.get_json()["playback"]
            self.assertEqual(playback["filename"], "standalone.raw")
            self.assertRegex(playback["source_id"], r"^[0-9a-f]{32}$")

            with mock.patch(
                    "bothviewer.api.data.render_raw_event_window_jpeg",
                    return_value=b"jpeg") as renderer:
                frame = client.get(
                    f"/raw-files/{playback['source_id']}/100000.jpg?palette=magenta_cyan")
            self.assertEqual(frame.status_code, 200)
            self.assertEqual(frame.data, b"jpeg")
            renderer.assert_called_once()


if __name__ == "__main__":
    unittest.main()
