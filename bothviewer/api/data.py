"""Read-only API for browsing and replaying recorded sessions."""

import os
import platform
import subprocess
import threading
import uuid

from flask import Flask, Response, jsonify, request

from bothviewer.api.common import add_cors_headers
from bothviewer.core.recordings import (
    evs_playback_manifest, inspect_raw_file, list_sessions, playback_manifest,
    render_event_overlay_png, render_event_window_jpeg, render_preview_jpeg,
    render_raw_event_window_jpeg, session_detail, trigger_timing_analysis,
)


def _choose_raw_file():
    """Open a native local file dialog without transferring the RAW via HTTP."""
    system = platform.system()
    if system == "Darwin":
        result = subprocess.run(
            ["osascript", "-e",
             'POSIX path of (choose file with prompt "EVS RAWファイルを選択")'],
            capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None
    if system == "Windows":
        script = (
            "Add-Type -AssemblyName System.Windows.Forms;"
            "$d=New-Object System.Windows.Forms.OpenFileDialog;"
            "$d.Filter='Metavision RAW (*.raw)|*.raw';"
            "$d.Multiselect=$false;"
            "if($d.ShowDialog() -eq 'OK'){[Console]::Write($d.FileName)}")
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-STA", "-Command", script],
            capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else None

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(filetypes=[("Metavision RAW", "*.raw")])
        root.destroy()
        return path or None
    except Exception:
        return None


def create_data_app(save_location):
    app = Flask("bothviewer.data_api")
    app.after_request(add_cors_headers)
    raw_sources = {}
    raw_sources_lock = threading.Lock()

    def open_raw(path):
        manifest = inspect_raw_file(path, request.args.get("interval_us", 33_000))
        source_id = uuid.uuid4().hex
        with raw_sources_lock:
            raw_sources[source_id] = os.path.realpath(
                os.path.abspath(os.path.expanduser(path)))
            # A viewer normally uses one source. Keep a small bounded registry so
            # repeated file selections cannot grow the server forever.
            while len(raw_sources) > 16:
                raw_sources.pop(next(iter(raw_sources)))
        return {**manifest, "source_id": source_id}

    def raw_path_for(source_id):
        with raw_sources_lock:
            path = raw_sources.get(source_id)
        if path is None:
            raise FileNotFoundError("RAWファイルをもう一度選択してください。")
        return path

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

    @app.route("/recordings/<session_id>/trigger-analysis")
    def trigger_analysis(session_id):
        try:
            analysis = trigger_timing_analysis(
                save_location, session_id, request.args.get("role", "evs"))
            return jsonify({"status": "success", "analysis": analysis})
        except Exception as exc:
            return error_response(exc)

    @app.route("/recordings/<session_id>/evs/<role>/<int:epoch>/<int:center_us>.jpg")
    def evs_frame(session_id, role, epoch, center_us):
        try:
            jpeg = render_event_window_jpeg(
                save_location, session_id, epoch, center_us,
                request.args.get("window_us", 33_000), request.args.get("width", 960), role,
                request.args.get("palette", "mono"))
            return Response(jpeg, mimetype="image/jpeg")
        except Exception as exc:
            return error_response(exc)

    @app.route("/raw-files/dialog", methods=["POST"])
    def raw_file_dialog():
        try:
            path = _choose_raw_file()
            if not path:
                return jsonify({"status": "cancelled"})
            return jsonify({"status": "success", "playback": open_raw(path)})
        except Exception as exc:
            return error_response(exc)

    @app.route("/raw-files/open", methods=["POST"])
    def raw_file_open():
        """Path-entry fallback for systems where a native dialog is unavailable."""
        try:
            payload = request.get_json(silent=True) or {}
            return jsonify({"status": "success", "playback": open_raw(payload.get("path"))})
        except Exception as exc:
            return error_response(exc)

    @app.route("/raw-files/<source_id>/<int:center_us>.jpg")
    def raw_file_frame(source_id, center_us):
        try:
            jpeg = render_raw_event_window_jpeg(
                raw_path_for(source_id), center_us,
                request.args.get("window_us", 33_000), request.args.get("width", 960),
                request.args.get("palette", "mono"))
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
                request.args.get("max_events", 50_000), "evs",
                request.args.get("palette", "mono"))
            return Response(png, mimetype="image/png")
        except Exception as exc:
            return error_response(exc)

    return app
