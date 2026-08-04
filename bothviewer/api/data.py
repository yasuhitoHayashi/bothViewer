"""Read-only API for browsing and replaying recorded sessions."""

from flask import Flask, Response, jsonify, request

from bothviewer.api.common import add_cors_headers
from bothviewer.core.recordings import (
    evs_playback_manifest, list_sessions, playback_manifest,
    render_event_overlay_png, render_event_window_jpeg, render_preview_jpeg, session_detail,
)


def create_data_app(save_location):
    app = Flask("bothviewer.data_api")
    app.after_request(add_cors_headers)

    def error_response(exc):
        if isinstance(exc, FileNotFoundError):
            return jsonify({"status": "error", "message": str(exc)}), 404
        if isinstance(exc, (TypeError, ValueError)):
            return jsonify({"status": "error", "message": str(exc)}), 400
        return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/status")
    def status():
        return jsonify({"status": "success", "streaming": False, "recording": False})

    @app.route("/set_capture_active", methods=["POST"])
    def set_capture_active():
        return jsonify({"status": "success", "retry_active": False})

    @app.route("/recordings")
    def recordings():
        return jsonify({
            "status": "success", "save_location": save_location,
            "sessions": list_sessions(save_location),
        })

    @app.route("/recordings/<session_id>")
    def detail(session_id):
        try:
            return jsonify({"status": "success", "session": session_detail(save_location, session_id)})
        except Exception as exc:
            return error_response(exc)

    @app.route("/recordings/<session_id>/preview/<filename>")
    def preview(session_id, filename):
        try:
            return Response(render_preview_jpeg(save_location, session_id, filename), mimetype="image/jpeg")
        except Exception as exc:
            return error_response(exc)

    @app.route("/recordings/<session_id>/playback")
    def both_playback(session_id):
        try:
            return jsonify({"status": "success", "playback": playback_manifest(save_location, session_id)})
        except Exception as exc:
            return error_response(exc)

    @app.route("/recordings/<session_id>/evs/<role>/playback")
    def evs_playback(session_id, role):
        try:
            manifest = evs_playback_manifest(
                save_location, session_id, role, request.args.get("interval_us", 33_000))
            return jsonify({"status": "success", "playback": manifest})
        except Exception as exc:
            return error_response(exc)

    @app.route("/recordings/<session_id>/evs/<role>/<int:epoch>/<int:center_us>.jpg")
    def evs_frame(session_id, role, epoch, center_us):
        try:
            jpeg = render_event_window_jpeg(
                save_location, session_id, epoch, center_us,
                request.args.get("window_us", 33_000), request.args.get("width", 960), role)
            return Response(jpeg, mimetype="image/jpeg")
        except Exception as exc:
            return error_response(exc)

    @app.route("/recordings/<session_id>/events/<int:epoch>/<int:center_us>.jpg")
    def legacy_event_frame(session_id, epoch, center_us):
        return evs_frame(session_id, "evs", epoch, center_us)

    @app.route("/recordings/<session_id>/events/<int:epoch>/<int:center_us>.png")
    def legacy_event_overlay(session_id, epoch, center_us):
        try:
            png = render_event_overlay_png(
                save_location, session_id, epoch, center_us,
                request.args.get("window_us", 33_000), request.args.get("width", 960),
                request.args.get("max_events", 50_000), "evs")
            return Response(png, mimetype="image/png")
        except Exception as exc:
            return error_response(exc)

    return app
