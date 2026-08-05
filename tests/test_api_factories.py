import unittest

from bothviewer.api.evs import create_evs_app
from bothviewer.api.frame import create_frame_app


class ApiFactoryTests(unittest.TestCase):
    def setUp(self):
        self.frame_app = create_frame_app(
            lambda: None,
            bandwidth_presets={"safe": 100_000_000},
            default_trigger_source="Line1",
            default_trigger_activation="RisingEdge",
        )
        self.evs_app = create_evs_app(lambda: None)

    def test_uninitialized_apis_return_service_unavailable(self):
        for app in (self.frame_app, self.evs_app):
            with self.subTest(app=app.name):
                response = app.test_client().get("/status")
                self.assertEqual(response.status_code, 503)
                self.assertEqual(response.get_json()["status"], "error")
                self.assertEqual(response.headers["Access-Control-Allow-Origin"], "*")

    def test_frame_routes_are_registered(self):
        routes = {rule.rule for rule in self.frame_app.url_map.iter_rules()}
        self.assertTrue({
            "/video_feed", "/recordings", "/recordings/<session_id>",
            "/recordings/<session_id>/preview/<filename>",
            "/recordings/<session_id>/playback",
            "/recordings/<session_id>/trigger-analysis",
            "/recordings/<session_id>/events/<int:epoch>/<int:center_us>.jpg",
            "/recordings/<session_id>/events/<int:epoch>/<int:center_us>.png",
            "/set_preview", "/set_save", "/start_recording", "/stop_recording",
            "/set_capture_active",
            "/trigger_options", "/set_bandwidth_preset", "/set_external_trigger",
            "/status", "/set_exposure", "/set_gain", "/set_framerate",
            "/set_whitebalance", "/get_settings",
        }.issubset(routes))

    def test_evs_routes_are_registered(self):
        routes = {rule.rule for rule in self.evs_app.url_map.iter_rules()}
        self.assertTrue({
            "/video_feed", "/set_preview", "/set_save", "/set_bias",
            "/set_trigger", "/start_recording", "/stop_recording",
            "/set_capture_active", "/reconnect", "/status",
        }.issubset(routes))

    def test_capture_activity_endpoint_validates_and_forwards_state(self):
        class FakeStreamer:
            def __init__(self):
                self.active = None

            def set_capture_active(self, active):
                self.active = active
                return active

        for factory in (
                lambda getter: create_evs_app(getter),
                lambda getter: create_frame_app(
                    getter, bandwidth_presets={"safe": 1},
                    default_trigger_source="Line1",
                    default_trigger_activation="RisingEdge")):
            streamer = FakeStreamer()
            client = factory(lambda: streamer).test_client()
            self.assertEqual(
                client.post("/set_capture_active", json={"active": "yes"}).status_code,
                400)
            response = client.post("/set_capture_active", json={"active": True})
            self.assertEqual(response.status_code, 200)
            self.assertTrue(streamer.active)
            self.assertTrue(response.get_json()["retry_active"])


if __name__ == "__main__":
    unittest.main()
