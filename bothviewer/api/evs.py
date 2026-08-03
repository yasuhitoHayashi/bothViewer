"""Flask API for the event-camera service.

This module deliberately has no OpenEB imports, so its routes can be tested
without camera hardware or the Metavision SDK.
"""

import time

from flask import Flask, Response, jsonify, request

from bothviewer.core.preview import PREVIEW_PRESETS
from .common import add_cors_headers


def create_evs_app(get_streamer):
    """Create an EVS API bound to a callable returning the current streamer."""
    app = Flask("bothviewer.evs_api")
    app.after_request(add_cors_headers)

    @app.before_request
    def require_initialized_streamer():
        if request.endpoint != "video_feed" and get_streamer() is None:
            return jsonify({
                "status": "error",
                "message": "EVSサーバーを初期化していません。",
            }), 503
        return None

    def reject_settings_change_while_recording():
        streamer = get_streamer()
        if streamer and streamer.recording:
            return jsonify({
                "status": "error",
                "message": "録画中はEVS設定を変更できません。先に録画を停止してください。",
            }), 409
        return None

    @app.route("/video_feed")
    def video_feed():
        def generate():
            last_sequence = -1
            while True:
                streamer = get_streamer()
                sequence, frame_data = (
                    streamer.preview.get_jpeg_packet() if streamer else (0, None))
                if frame_data and sequence != last_sequence:
                    last_sequence = sequence
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" +
                           frame_data + b"\r\n")
                time.sleep(0.01)
        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/set_preview", methods=["POST"])
    def set_preview():
        data = request.get_json(silent=True) or {}
        preset = data.get("preset")
        auto_degrade = data.get("auto_degrade", True)
        if preset not in PREVIEW_PRESETS or not isinstance(auto_degrade, bool):
            return jsonify({"status": "error", "message": "表示設定が不正です。"}), 400
        try:
            preview_status = get_streamer().update_preview_preferences(
                preset, auto_degrade, persist=bool(data.get("persist", False)))
        except (OSError, ValueError) as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        return jsonify({
            "status": "success",
            "message": f"EVS表示を{PREVIEW_PRESETS[preset]['label']}へ変更しました。",
            "preview": preview_status,
        })

    @app.route("/set_capture_active", methods=["POST"])
    def set_capture_active():
        active = (request.get_json(silent=True) or {}).get("active")
        if not isinstance(active, bool):
            return jsonify({"status": "error", "message": "activeは真偽値で指定してください。"}), 400
        retry_active = get_streamer().set_capture_active(active)
        return jsonify({
            "status": "success",
            "retry_active": retry_active,
            "message": "EVS接続試行を有効にしました。" if active else "EVS接続試行を休止しました。",
        })

    @app.route("/set_save", methods=["POST"])
    def set_save():
        guard = reject_settings_change_while_recording()
        if guard:
            return guard
        data = request.get_json(silent=True) or {}
        try:
            get_streamer().update_save_settings(
                data.get("save_location"), data.get("save_filename"))
        except (OSError, ValueError) as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        return jsonify({"status": "success", "message": "保存設定を更新しました。"})

    @app.route("/set_bias", methods=["POST"])
    def set_bias():
        guard = reject_settings_change_while_recording()
        if guard:
            return guard
        data = request.get_json(silent=True) or {}
        bias_on = data.get("bias_diff_on")
        bias_off = data.get("bias_diff_off")
        if bias_on is None or bias_off is None:
            return jsonify({"status": "error", "message": "Bias 設定の値が不足しています。"}), 400
        success = get_streamer().update_bias(bias_on, bias_off)
        message = "Bias 設定を更新しました。" if success else "Bias インターフェースを利用できません。"
        return jsonify({"status": "success" if success else "error", "message": message}), (200 if success else 503)

    @app.route("/set_trigger", methods=["POST"])
    def set_trigger():
        guard = reject_settings_change_while_recording()
        if guard:
            return guard
        trigger = (request.get_json(silent=True) or {}).get("trigger")
        if trigger is None:
            return jsonify({"status": "error", "message": "Trigger の値が不足しています。"}), 400
        if trigger is not True:
            return jsonify({
                "status": "error",
                "message": "EVS Trigger Inは同期記録のため常時有効です。無効化できません。",
            }), 409
        success = get_streamer().update_trigger(True)
        message = "EVS Trigger Inは有効です。" if success else "EVS Trigger Inを有効化できませんでした。"
        return jsonify({"status": "success" if success else "error", "message": message}), (200 if success else 503)

    @app.route("/start_recording", methods=["POST"])
    def start_recording():
        streamer = get_streamer()
        data = request.get_json(silent=True) or {}
        try:
            success = streamer.start_recording(data.get("session_id"))
        except (OSError, RuntimeError, ValueError) as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        message = "EVS 録画を開始しました。" if success else "EVS 録画を開始できませんでした。"
        return jsonify({
            "status": "success" if success else "error",
            "message": message,
            "session_id": streamer.recording_session_id,
        }), (200 if success else 409)

    @app.route("/stop_recording", methods=["POST"])
    def stop_recording():
        success = get_streamer().stop_recording()
        message = "EVS 録画を停止しました。" if success else "EVS は録画中ではありません。"
        return jsonify({"status": "success" if success else "error", "message": message}), (200 if success else 409)

    @app.route("/reconnect", methods=["POST"])
    def reconnect():
        streamer = get_streamer()
        success, message = streamer.request_reconnect()
        return jsonify({
            "status": "success" if success else "error",
            "message": message,
            "connection": {"state": streamer.connection_state},
        }), (202 if success else 409)

    @app.route("/status")
    def status():
        streamer = get_streamer()
        return jsonify({
            "status": "success",
            "streaming": streamer.connection_state == "connected",
            "frame_ready": bool(streamer.preview.get_jpeg()),
            "preview": streamer.preview.status(),
            "processing": streamer.event_processing_status(),
            "recording": bool(streamer.recording),
            "trigger_in": bool(streamer.trigger_in),
            "trigger_monitor": streamer.trigger_monitor_status(),
            "connection": {
                "state": streamer.connection_state,
                "retry_active": streamer.capture_retry_active(),
                "restart_attempts": streamer.reconnect_attempts,
                "successful_reconnections": streamer.successful_reconnections,
                "stream_epoch": streamer.stream_epoch,
                "last_error": streamer.last_connection_error,
            },
            "recording_quality": {
                "session_id": streamer.recording_session_id,
                "trigger_events": streamer.recording_trigger_count,
                "rising_edges": streamer.recording_trigger_rising_count,
                "falling_edges": streamer.recording_trigger_falling_count,
                "elapsed_seconds": round(
                    (time.time_ns() - streamer.recording_started_utc_ns) / 1e9, 1)
                    if streamer.recording and streamer.recording_started_utc_ns else 0,
            },
        })

    return app
