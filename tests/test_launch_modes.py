import json
import os
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
