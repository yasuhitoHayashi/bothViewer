import csv
import json
import os
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest

import cv2
import numpy as np

from bothviewer.cameras.evs import EVSStreamer
from bothviewer.cameras.frame import (
    CameraThread, FrameStreamer, ImageWriterThread, calculate_evs_matching_frame_roi)
from bothviewer.core.preview import LatestFramePreview
from bothviewer.core.synchronization import build_synchronization_report


class FakeFeature:
    def __init__(self, value=None):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class FakeFrameRateFeature(FakeFeature):
    def __init__(self, value=10.0, reject=None):
        super().__init__(value)
        self.reject = reject

    def set(self, value):
        if self.reject is not None and float(value) == self.reject:
            raise RuntimeError("rejected fps")
        super().set(float(value))

    def get_range(self):
        return 1.0, 100.0

    def get_increment(self):
        return 0.1


class FakeBandwidthFeature(FakeFeature):
    def __init__(self, value=100_000_000):
        super().__init__(value)

    def get_range(self):
        return 50_000_000, 300_000_000


class FakeRoiFeature(FakeFeature):
    def __init__(self, value, minimum, maximum, increment):
        super().__init__(value)
        self.minimum = minimum
        self.maximum = maximum
        self.increment = increment

    def get_range(self):
        return self.minimum, self.maximum

    def get_increment(self):
        return self.increment


class FakeRoiCamera:
    def __init__(self):
        self.Width = FakeRoiFeature(1936, 8, 1936, 8)
        self.Height = FakeRoiFeature(1216, 2, 1216, 2)
        self.OffsetX = FakeRoiFeature(0, 0, 1936, 2)
        self.OffsetY = FakeRoiFeature(0, 0, 1216, 2)


class FakeCommand:
    def run(self):
        return None


class FakeEventsStream:
    def __init__(self):
        self.paths = []

    def log_raw_data(self, path):
        self.paths.append(path)
        return True

    def stop(self):
        self.stopped = True


class FakeTriggerReader:
    def __init__(self):
        self.cleared = False

    def clear_ext_trigger_events(self):
        self.cleared = True


class FakeTriggerIterator:
    def __init__(self, events):
        self.events = events
        self.reader = FakeTriggerReader()

    def get_ext_trigger_events(self):
        return self.events


class FakeTriggerCamera:
    def __init__(self):
        for name in (
            "TriggerSelector", "TriggerMode", "LineSelector", "LineMode",
            "TriggerSource", "TriggerActivation", "AcquisitionFrameRateEnable",
            "AcquisitionFrameRate", "LineSource", "LineInverter",
        ):
            setattr(self, name, FakeFeature())


class FakeStreamingCamera:
    def __init__(self, reject_fps=None):
        self.AcquisitionFrameRate = FakeFrameRateFeature(10.0, reject_fps)
        self.AcquisitionFrameRateEnable = FakeFeature(True)
        self.DeviceLinkThroughputLimit = FakeBandwidthFeature()
        self.AcquisitionStop = FakeCommand()
        self.start_count = 0
        self.stop_count = 0

    def stop_streaming(self):
        self.stop_count += 1

    def start_streaming(self, _callback, buffer_count, allocation_mode):
        self.start_count += 1


class RecordingAuditTests(unittest.TestCase):
    def test_latest_preview_replaces_pending_frames_without_blocking_source(self):
        preview = LatestFramePreview(
            "test", "evs", lambda payload, _settings: payload,
            preset="standard")
        preview.submit(b"first")
        preview.submit(b"latest")

        status = preview.status()

        self.assertEqual(status["submitted_frames"], 2)
        self.assertEqual(status["skipped_source_frames"], 1)

    def test_preview_auto_degrades_after_sustained_encoder_overload(self):
        changes = []
        preview = LatestFramePreview(
            "test", "evs", lambda payload, _settings: payload,
            preset="high_quality", auto_degrade=True,
            on_effective_change=lambda preset, _settings: changes.append(preset))

        for _ in range(5):
            preview._update_adaptation(0.03, 1 / 30)

        status = preview.status()
        self.assertEqual(status["requested_preset"], "high_quality")
        self.assertEqual(status["effective_preset"], "standard")
        self.assertEqual(status["auto_degrade_count"], 1)
        self.assertEqual(changes, ["standard"])

    def test_preview_worker_encodes_latest_frame_and_stops_cleanly(self):
        preview = LatestFramePreview(
            "test", "evs", lambda payload, _settings: payload,
            preset="record_priority")
        preview.start()
        try:
            preview.submit(b"jpeg-data")
            deadline = time.monotonic() + 1
            while preview.get_jpeg() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            sequence, jpeg = preview.get_jpeg_packet()
            self.assertEqual(sequence, 1)
            self.assertEqual(jpeg, b"jpeg-data")
        finally:
            preview.stop()
            preview.join(timeout=1)
        self.assertFalse(preview.is_alive())

    def test_bayer_rg_preview_places_red_sites_in_bgr_red_channel(self):
        bayer = np.zeros((8, 8), dtype=np.uint8)
        bayer[0::2, 0::2] = 255
        preview = FrameStreamer.bayer_to_bgr(bayer, "BayerRG8")
        self.assertEqual(preview[3, 3].tolist(), [0, 0, 255])

    def test_bayer_phase_follows_crop_offset(self):
        self.assertEqual(
            FrameStreamer.shifted_bayer_format("BayerRG8", 0, 0), "BayerRG8")
        self.assertEqual(
            FrameStreamer.shifted_bayer_format("BayerRG8", 1, 0), "BayerGR8")
        self.assertEqual(
            FrameStreamer.shifted_bayer_format("BayerRG8", 0, 1), "BayerGB8")
        self.assertEqual(
            FrameStreamer.shifted_bayer_format("BayerRG8", 1, 1), "BayerBG8")

    def test_evs_recovery_starts_a_new_raw_segment(self):
        with tempfile.TemporaryDirectory() as root:
            streamer = EVSStreamer.__new__(EVSStreamer)
            streamer.recording = True
            streamer.events_stream = FakeEventsStream()
            streamer.recording_folder = root
            streamer.raw_segment_index = 0
            streamer.recording_raw_files = ["events.raw"]
            streamer.recording_file = os.path.join(root, "events.raw")

            streamer.start_recovered_raw_segment()

            self.assertEqual(streamer.recording_raw_files, ["events.raw", "events_001.raw"])
            self.assertTrue(streamer.recording_file.endswith("events_001.raw"))

    def test_manual_evs_reconnect_stops_stream_and_wakes_retry(self):
        streamer = EVSStreamer.__new__(EVSStreamer)
        streamer.live_mode = True
        streamer.connection_state = "connected"
        streamer.last_connection_error = None
        streamer.reconnect_wakeup = threading.Event()
        streamer.capture_active_until = time.monotonic() + 10
        streamer.events_stream = FakeEventsStream()
        streamer.connection_writer = None
        streamer.recording_lock = threading.RLock()

        success, _message = streamer.request_reconnect()

        self.assertTrue(success)
        self.assertEqual(streamer.connection_state, "reconnecting")
        self.assertTrue(streamer.events_stream.stopped)
        self.assertTrue(streamer.reconnect_wakeup.is_set())

    def test_camera_retry_leases_follow_capture_tab_activity(self):
        evs = EVSStreamer.__new__(EVSStreamer)
        evs.reconnect_wakeup = threading.Event()
        evs.capture_active_until = 0.0
        self.assertFalse(evs.capture_retry_active())
        self.assertTrue(evs.set_capture_active(True, lease_seconds=1))
        self.assertTrue(evs.reconnect_wakeup.is_set())
        self.assertFalse(evs.set_capture_active(False))

        frame = FrameStreamer.__new__(FrameStreamer)
        frame.camera_retry_wakeup = threading.Event()
        frame.capture_active_until = 0.0
        self.assertFalse(frame.capture_retry_active())
        self.assertTrue(frame.set_capture_active(True, lease_seconds=1))
        self.assertTrue(frame.camera_retry_wakeup.is_set())
        self.assertFalse(frame.set_capture_active(False))

    def test_trigger_monitor_counts_edges_and_reports_reference_frequency(self):
        streamer = EVSStreamer.__new__(EVSStreamer)
        streamer.initialize_trigger_monitor()
        streamer.trigger_in = True
        streamer.recording = False
        streamer.recording_lock = threading.RLock()
        streamer.trigger_writer = None
        streamer.mv_iterator = FakeTriggerIterator([
            {"t": 0, "p": 1, "id": 0},
            {"t": 500, "p": 0, "id": 0},
            {"t": 10_000, "p": 1, "id": 0},
            {"t": 10_500, "p": 0, "id": 0},
        ])

        streamer.record_trigger_events()
        status = streamer.trigger_monitor_status()

        self.assertTrue(status["active"])
        self.assertEqual(status["edge_count"], 4)
        self.assertEqual(status["rising_edges"], 2)
        self.assertEqual(status["falling_edges"], 2)
        self.assertEqual(status["rising_hz"], 100.0)
        self.assertEqual(status["falling_period_ms"], 10.0)
        self.assertTrue(streamer.mv_iterator.reader.cleared)

    def test_bandwidth_preset_restarts_stream_and_reports_limit(self):
        camera_thread = CameraThread()
        camera_thread.cam = FakeStreamingCamera()
        camera_thread.streaming_active = True

        success, status = camera_thread.set_bandwidth_preset("standard")

        self.assertTrue(success)
        self.assertEqual(status["preset"], "standard")
        self.assertEqual(status["limit_bytes_per_second"], 150_000_000)
        self.assertTrue(camera_thread.streaming_active)

    def test_bandwidth_fallback_requests_next_safer_preset(self):
        camera_thread = CameraThread(bandwidth_preset="high")
        camera_thread.request_safer_bandwidth()
        self.assertTrue(camera_thread.auto_fallback_requested)

    def test_framerate_change_restarts_stream_and_updates_target(self):
        camera_thread = CameraThread()
        camera_thread.cam = FakeStreamingCamera()
        camera_thread.streaming_active = True

        success, configured = camera_thread.set_framerate(20.0)

        self.assertTrue(success)
        self.assertEqual(configured, 20.0)
        self.assertEqual(camera_thread.free_run_fps, 20.0)
        self.assertTrue(camera_thread.streaming_active)
        self.assertEqual(camera_thread.cam.stop_count, 1)
        self.assertEqual(camera_thread.cam.start_count, 1)

    def test_framerate_failure_restores_previous_value_and_stream(self):
        camera_thread = CameraThread()
        camera_thread.cam = FakeStreamingCamera(reject_fps=20.0)
        camera_thread.streaming_active = True

        success, message = camera_thread.set_framerate(20.0)

        self.assertFalse(success)
        self.assertIn("10 fpsへ戻しました", message)
        self.assertEqual(camera_thread.cam.AcquisitionFrameRate.get(), 10.0)
        self.assertTrue(camera_thread.streaming_active)
        self.assertEqual(camera_thread.cam.start_count, 1)

    def test_external_trigger_keeps_exposure_active_output_enabled(self):
        camera_thread = CameraThread()
        camera = FakeTriggerCamera()
        camera_thread.apply_external_trigger_mode(
            camera, True, "Line1", "RisingEdge")

        self.assertEqual(camera.TriggerMode.get(), "On")
        self.assertEqual(camera.TriggerSource.get(), "Line1")
        self.assertEqual(camera.TriggerActivation.get(), "RisingEdge")
        self.assertFalse(camera.AcquisitionFrameRateEnable.get())
        self.assertEqual(camera.LineSelector.get(), "Line0")
        self.assertEqual(camera.LineMode.get(), "Output")
        self.assertEqual(camera.LineSource.get(), "ExposureActive")
        self.assertTrue(camera.LineInverter.get())

    def test_external_trigger_monitor_reports_driven_frames(self):
        camera_thread = CameraThread(trigger_enabled=True)
        now_ns = time.monotonic_ns()
        camera_thread.record_external_trigger_result(
            now_ns - 20_000_000, True, "FrameStatus.Complete")
        camera_thread.record_external_trigger_result(
            now_ns - 10_000_000, False, "FrameStatus.Incomplete")

        status = camera_thread.external_trigger_monitor_status()

        self.assertTrue(status["enabled"])
        self.assertTrue(status["active"])
        self.assertEqual(status["callback_frames"], 2)
        self.assertEqual(status["complete_frames"], 1)
        self.assertEqual(status["incomplete_frames"], 1)
        self.assertEqual(status["measured_hz"], 100.0)
        self.assertEqual(status["evs_output_source"], "ExposureActive")

    def test_image_writer_preserves_8_bit_bayer_values(self):
        with tempfile.TemporaryDirectory() as root:
            images_dir = os.path.join(root, "images")
            os.makedirs(images_dir)
            results_path = os.path.join(root, "saved_frames.csv")
            recording_queue = queue.Queue()
            writer = ImageWriterThread(
                recording_queue, images_dir, results_path, "session-test")
            source = np.arange(64, dtype=np.uint8).reshape(8, 8)
            writer.start()
            recording_queue.put({"sequence": 0, "filename": "test.pgm", "image": source})
            recording_queue.put(None)
            writer.join(timeout=5)

            restored = cv2.imread(
                os.path.join(images_dir, "test.pgm"), cv2.IMREAD_UNCHANGED)
            np.testing.assert_array_equal(restored, source)
            self.assertEqual(writer.saved_count, 1)
            self.assertEqual(writer.failed_count, 0)

    def test_hardware_roi_matches_evs_physical_sensor_area(self):
        expected = calculate_evs_matching_frame_roi()
        self.assertEqual(
            (expected["width"], expected["height"],
             expected["offset_x"], expected["offset_y"]),
            (1800, 1012, 68, 102),
        )
        camera_thread = CameraThread()
        self.assertTrue(camera_thread.configure_evs_matching_hardware_roi(FakeRoiCamera()))
        self.assertEqual(camera_thread.roi_configuration["mode"], "camera_hardware_roi")

    def test_frame_filename_contains_utc_sequence_and_camera_id(self):
        filename = FrameStreamer.frame_filename(7, 1_700_000_000_123_456_789, 42)
        self.assertRegex(
            filename,
            r"^\d{8}T\d{6}\.123456789Z_seq000007_id42\.pgm$",
        )

    def test_frame_stop_returns_before_writer_drain_and_uses_capture_duration(self):
        class BlockingWriter:
            saved_count = 1
            failed_count = 0

            def __init__(self):
                self.release = threading.Event()

            def join(self):
                self.release.wait(2)

            def performance_status(self):
                return {"save_fps": 80.0, "write_megabytes_per_second": 188.0}

        with tempfile.TemporaryDirectory() as root:
            streamer = FrameStreamer.__new__(FrameStreamer)
            streamer.recording_lock = threading.RLock()
            streamer.recording = True
            streamer.recording_finalizing = False
            streamer.recording_finalizer_thread = None
            streamer.events_file = None
            streamer.events_writer = None
            streamer.connection_file = None
            streamer.connection_writer = None
            streamer.recording_queue = queue.Queue()
            streamer.image_thread = BlockingWriter()
            streamer.recording_folder = root
            streamer.recording_session_id = "session-test"
            streamer.recording_started_utc_ns = time.time_ns() - 1_000_000_000
            streamer.recording_sequence = 80
            streamer.recording_complete_count = 80
            streamer.recording_incomplete_count = 0
            streamer.recording_frame_gap_count = 0
            streamer.recording_queue_drop_count = 0
            streamer.successful_reconnections = 0
            streamer.recording_restart_baseline = 0
            streamer.recording_fallback_baseline = 0
            streamer.last_recording_summary = {}
            streamer.cam_thread = SimpleNamespace(
                trigger_configuration=lambda: {"enabled": False}, auto_fallback_count=0,
                roi_configuration={"mode": "camera_hardware_roi", "width": 1800,
                                   "height": 1012, "offset_x": 68, "offset_y": 102})

            self.assertTrue(streamer.stop_recording(wait_for_writer=False))
            self.assertFalse(streamer.recording)
            self.assertTrue(streamer.recording_finalizing)
            streamer.image_thread.release.set()
            self.assertTrue(streamer.wait_for_recording_finalization(timeout=2))

            with open(os.path.join(root, "frame_summary.json"), encoding="utf-8") as file:
                summary = json.load(file)
            self.assertLess(summary["duration_seconds"], 1.5)
            self.assertIn("writer_drain_seconds", summary)
            self.assertIn("PGM", summary["pixel_storage"])

    def test_synchronization_uses_actual_write_results_and_frame_id_gaps(self):
        with tempfile.TemporaryDirectory() as root:
            frame_dir = os.path.join(root, "frame")
            evs_dir = os.path.join(root, "evs")
            os.makedirs(frame_dir)
            os.makedirs(evs_dir)

            frame_fields = (
                "session_id", "sequence", "callback_index", "host_utc_iso",
                "host_utc_ns", "host_monotonic_ns", "host_delta_ms",
                "camera_frame_id", "camera_timestamp_ticks", "camera_timestamp_delta_ticks",
                "frame_status", "frame_id_delta", "missing_before", "width", "height",
                "pixel_format", "queue_result", "filename", "queue_depth", "error",
            )
            with open(os.path.join(frame_dir, "frame_events.csv"), "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=frame_fields)
                writer.writeheader()
                for sequence, frame_id, missing, filename in (
                    (0, 10, 0, "frame0.png"),
                    (1, 12, 1, "frame1.png"),
                ):
                    writer.writerow({
                        "sequence": sequence, "camera_frame_id": frame_id,
                        "missing_before": missing, "host_utc_ns": 1_000_000_000 + sequence,
                        "frame_status": "FrameStatus.Complete", "filename": filename,
                        "queue_result": "enqueued",
                    })

            saved_fields = (
                "session_id", "sequence", "filename", "write_started_utc_ns",
                "write_finished_utc_ns", "write_duration_ms", "write_ok",
                "file_size_bytes", "error",
            )
            with open(os.path.join(frame_dir, "saved_frames.csv"), "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=saved_fields)
                writer.writeheader()
                writer.writerow({"filename": "frame0.png", "write_ok": 1})
                writer.writerow({"filename": "frame1.png", "write_ok": 0})

            trigger_fields = (
                "session_id", "trigger_index", "evs_timestamp_us", "sensor_delta_us",
                "polarity", "channel_id", "host_decode_utc_ns", "host_decode_monotonic_ns",
            )
            with open(os.path.join(evs_dir, "triggers.csv"), "w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=trigger_fields)
                writer.writeheader()
                for index in range(3):
                    writer.writerow({
                        "trigger_index": index, "evs_timestamp_us": index * 100_000,
                        "polarity": 1, "channel_id": 0,
                        "host_decode_utc_ns": 1_000_000_000 + index,
                    })

            summary = build_synchronization_report(root, "session-test")
            self.assertEqual(summary["matched"], 2)
            self.assertEqual(summary["frames_without_saved_image"], 1)
            self.assertEqual(summary["frame_events_missing_from_id_gaps"], 1)

            with open(os.path.join(root, "synchronization.csv"), newline="") as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(
                [row["match_status"] for row in rows],
                ["matched", "frame_event_missing", "trigger_without_saved_image"],
            )
            self.assertEqual(rows[-1]["trigger_index"], "2")


if __name__ == "__main__":
    unittest.main()
