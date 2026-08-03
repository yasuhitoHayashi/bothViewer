"""記録経路から分離した、最新フレーム優先のJPEGプレビュー処理。"""
import threading
import time
from collections import deque


PREVIEW_PRESETS = {
    "record_priority": {
        "label": "記録優先", "fps": 10.0,
        "evs_scale": 0.5, "frame_scale": 0.45, "jpeg_quality": 70,
    },
    "standard": {
        "label": "標準", "fps": 20.0,
        "evs_scale": 0.75, "frame_scale": 0.6, "jpeg_quality": 75,
    },
    "high_quality": {
        "label": "高画質", "fps": 30.0,
        "evs_scale": 1.0, "frame_scale": 0.8, "jpeg_quality": 80,
    },
}
PREVIEW_PRESET_ORDER = ("record_priority", "standard", "high_quality")


class LatestFramePreview(threading.Thread):
    """入力を塞がず最新1枚だけを指定fpsでJPEGへ変換する。"""

    def __init__(self, name, camera_kind, encoder, preset="standard",
                 auto_degrade=True, on_effective_change=None):
        super().__init__(daemon=True, name=f"{name}-preview")
        if preset not in PREVIEW_PRESETS:
            preset = "standard"
        self.preview_name = name
        self.camera_kind = camera_kind
        self.encoder = encoder
        self.on_effective_change = on_effective_change
        self.lock = threading.Lock()
        self.wakeup = threading.Event()
        self.stop_event = threading.Event()
        self.latest_payload = None
        self.latest_sequence = 0
        self.processed_sequence = 0
        self.latest_jpeg = None
        self.requested_preset = preset
        self.effective_preset = preset
        self.auto_degrade = bool(auto_degrade)
        self.submitted_frames = 0
        self.encoded_frames = 0
        self.skipped_source_frames = 0
        self.encode_failures = 0
        self.auto_degrade_count = 0
        self.auto_recover_count = 0
        self.encode_duration_ema_ms = 0.0
        self.last_encoded_monotonic_ns = None
        self.encoded_timestamps_ns = deque(maxlen=120)
        self.overload_streak = 0
        self.stable_since = time.monotonic()

    def submit(self, payload):
        with self.lock:
            if self.latest_sequence > self.processed_sequence:
                self.skipped_source_frames += 1
            self.latest_payload = payload
            self.latest_sequence += 1
            self.submitted_frames += 1
        self.wakeup.set()

    def get_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def get_jpeg_packet(self):
        with self.lock:
            return self.encoded_frames, self.latest_jpeg

    def current_settings(self):
        with self.lock:
            preset = self.effective_preset
        return dict(PREVIEW_PRESETS[preset])

    def set_preferences(self, preset, auto_degrade):
        if preset not in PREVIEW_PRESETS:
            raise ValueError("record_priority / standard / high_quality を指定してください。")
        with self.lock:
            self.requested_preset = preset
            self.effective_preset = preset
            self.auto_degrade = bool(auto_degrade)
            self.overload_streak = 0
            self.stable_since = time.monotonic()
            settings = dict(PREVIEW_PRESETS[preset])
        if self.on_effective_change:
            self.on_effective_change(preset, settings)
        self.wakeup.set()
        return self.status()

    def _change_effective_preset(self, preset, recovery=False):
        with self.lock:
            if preset == self.effective_preset:
                return
            self.effective_preset = preset
            self.overload_streak = 0
            self.stable_since = time.monotonic()
            if recovery:
                self.auto_recover_count += 1
            else:
                self.auto_degrade_count += 1
            settings = dict(PREVIEW_PRESETS[preset])
        if self.on_effective_change:
            self.on_effective_change(preset, settings)

    def request_safer_preset(self):
        """外部の処理遅延監視から表示を一段だけ軽くする。"""
        with self.lock:
            if not self.auto_degrade:
                return False
            effective_index = PREVIEW_PRESET_ORDER.index(self.effective_preset)
        if effective_index <= 0:
            return False
        self._change_effective_preset(PREVIEW_PRESET_ORDER[effective_index - 1])
        return True

    def _update_adaptation(self, encode_seconds, target_period):
        with self.lock:
            auto_degrade = self.auto_degrade
            effective = self.effective_preset
            requested = self.requested_preset
        if not auto_degrade:
            return
        utilization = encode_seconds / target_period if target_period else 0
        if utilization >= 0.75:
            self.overload_streak += 1
            self.stable_since = None
        else:
            self.overload_streak = 0
            if utilization <= 0.4:
                if self.stable_since is None:
                    self.stable_since = time.monotonic()
            else:
                self.stable_since = None

        effective_index = PREVIEW_PRESET_ORDER.index(effective)
        requested_index = PREVIEW_PRESET_ORDER.index(requested)
        if self.overload_streak >= 5 and effective_index > 0:
            self._change_effective_preset(PREVIEW_PRESET_ORDER[effective_index - 1])
        elif (effective_index < requested_index and self.stable_since is not None
              and time.monotonic() - self.stable_since >= 30):
            self._change_effective_preset(
                PREVIEW_PRESET_ORDER[effective_index + 1], recovery=True)

    def run(self):
        next_encode_at = 0.0
        while not self.stop_event.is_set():
            self.wakeup.wait(0.2)
            self.wakeup.clear()
            if self.stop_event.is_set():
                break
            settings = self.current_settings()
            target_period = 1.0 / settings["fps"]
            wait_seconds = next_encode_at - time.monotonic()
            if wait_seconds > 0 and self.stop_event.wait(wait_seconds):
                break
            with self.lock:
                if self.latest_sequence == self.processed_sequence or self.latest_payload is None:
                    continue
                payload = self.latest_payload
                sequence = self.latest_sequence
                # ここまでの最新フレームは処理対象として確保済み。以後のsubmitだけが
                # 次回までの置換候補になり、途中で上書きされた分を間引きとして数える。
                self.processed_sequence = sequence
            started = time.monotonic()
            jpeg = None
            try:
                jpeg = self.encoder(payload, settings)
            except Exception as exc:
                print(f"{self.preview_name}プレビュー変換エラー: {exc}")
            finished = time.monotonic()
            elapsed = finished - started
            with self.lock:
                if jpeg:
                    self.latest_jpeg = jpeg
                    self.encoded_frames += 1
                    self.last_encoded_monotonic_ns = time.monotonic_ns()
                    self.encoded_timestamps_ns.append(self.last_encoded_monotonic_ns)
                else:
                    self.encode_failures += 1
                elapsed_ms = elapsed * 1000
                self.encode_duration_ema_ms = (
                    elapsed_ms if self.encode_duration_ema_ms == 0
                    else self.encode_duration_ema_ms * 0.9 + elapsed_ms * 0.1)
            self._update_adaptation(elapsed, target_period)
            next_encode_at = max(next_encode_at + target_period, finished)

    def status(self):
        with self.lock:
            settings = PREVIEW_PRESETS[self.effective_preset]
            age_ms = (
                (time.monotonic_ns() - self.last_encoded_monotonic_ns) / 1e6
                if self.last_encoded_monotonic_ns is not None else None)
            measured_fps = 0.0
            if (len(self.encoded_timestamps_ns) >= 2
                    and self.encoded_timestamps_ns[-1] > self.encoded_timestamps_ns[0]):
                measured_fps = (
                    (len(self.encoded_timestamps_ns) - 1) * 1e9
                    / (self.encoded_timestamps_ns[-1] - self.encoded_timestamps_ns[0]))
            return {
                "requested_preset": self.requested_preset,
                "effective_preset": self.effective_preset,
                "label": settings["label"],
                "target_fps": settings["fps"],
                "measured_fps": round(measured_fps, 3),
                "scale": settings[f"{self.camera_kind}_scale"],
                "jpeg_quality": settings["jpeg_quality"],
                "auto_degrade": self.auto_degrade,
                "submitted_frames": self.submitted_frames,
                "encoded_frames": self.encoded_frames,
                "skipped_source_frames": self.skipped_source_frames,
                "encode_failures": self.encode_failures,
                "encode_duration_ms": round(self.encode_duration_ema_ms, 3),
                "last_preview_age_ms": round(age_ms, 1) if age_ms is not None else None,
                "auto_degrade_count": self.auto_degrade_count,
                "auto_recover_count": self.auto_recover_count,
            }

    def stop(self):
        self.stop_event.set()
        self.wakeup.set()
