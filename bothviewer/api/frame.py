"""Flask API for the frame-camera service.

The API depends on a streamer getter instead of camera SDK classes. This keeps
HTTP concerns separate and allows hardware-independent route tests.
"""

import math
import time

from flask import Flask, Response, jsonify, request

from bothviewer.core.config import load_config, save_config
from bothviewer.core.preview import PREVIEW_PRESETS
from bothviewer.core.recordings import (
    list_sessions,
    playback_manifest,
    render_event_overlay_png,
    render_event_window_jpeg,
    render_preview_jpeg,
    session_detail,
)
from .common import add_cors_headers


def create_frame_app(
        get_streamer, *, bandwidth_presets, default_trigger_source,
        default_trigger_activation):
    """Create a frame-camera API bound to the current streamer instance."""
    app = Flask("bothviewer.frame_api")
    app.after_request(add_cors_headers)

    @app.before_request
    def require_initialized_streamer():
        if request.endpoint != "video_feed" and get_streamer() is None:
            return jsonify({
                "status": "error",
                "message": "Frameサーバーを初期化していません。",
            }), 503
        return None

    def reject_settings_change_while_recording():
        streamer = get_streamer()
        if streamer and (streamer.recording or streamer.recording_finalizing):
            return jsonify({
                "status": "error",
                "message": "録画中または保存完了処理中は設定を変更できません。",
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

    @app.route("/recordings")
    def recordings():
        streamer = get_streamer()
        try:
            sessions = list_sessions(streamer.save_location)
        except OSError as exc:
            return jsonify({"status": "error", "message": f"保存データを読めません: {exc}"}), 500
        return jsonify({
            "status": "success",
            "save_location": streamer.save_location,
            "sessions": sessions,
        })

    @app.route("/recordings/<session_id>")
    def recording_detail(session_id):
        try:
            detail = session_detail(get_streamer().save_location, session_id)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except FileNotFoundError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 404
        except OSError as exc:
            return jsonify({"status": "error", "message": f"保存データを読めません: {exc}"}), 500
        return jsonify({"status": "success", "session": detail})

    @app.route("/recordings/<session_id>/preview/<filename>")
    def recording_preview(session_id, filename):
        try:
            jpeg = render_preview_jpeg(get_streamer().save_location, session_id, filename)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except FileNotFoundError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 404
        except OSError as exc:
            return jsonify({"status": "error", "message": f"画像を読めません: {exc}"}), 500
        response = Response(jpeg, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "private, max-age=3600"
        return response

    @app.route("/recordings/<session_id>/playback")
    def recording_playback(session_id):
        try:
            manifest = playback_manifest(get_streamer().save_location, session_id)
        except ValueError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except FileNotFoundError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 404
        except OSError as exc:
            return jsonify({"status": "error", "message": f"再生情報を読めません: {exc}"}), 500
        return jsonify({"status": "success", "playback": manifest})

    @app.route("/recordings/<session_id>/events/<int:epoch>/<int:center_us>.jpg")
    def recording_event_window(session_id, epoch, center_us):
        try:
            jpeg = render_event_window_jpeg(
                get_streamer().save_location, session_id, epoch, center_us,
                request.args.get("window_us", 33_000), request.args.get("width", 720))
        except (TypeError, ValueError) as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except FileNotFoundError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 404
        except OSError as exc:
            return jsonify({"status": "error", "message": f"EVS RAWを読めません: {exc}"}), 500
        response = Response(jpeg, mimetype="image/jpeg")
        response.headers["Cache-Control"] = "private, max-age=3600"
        return response

    @app.route("/recordings/<session_id>/events/<int:epoch>/<int:center_us>.png")
    def recording_event_overlay(session_id, epoch, center_us):
        try:
            png = render_event_overlay_png(
                get_streamer().save_location, session_id, epoch, center_us,
                request.args.get("window_us", 33_000), request.args.get("width", 960),
                request.args.get("max_events", 50_000))
        except (TypeError, ValueError) as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        except FileNotFoundError as exc:
            return jsonify({"status": "error", "message": str(exc)}), 404
        except OSError as exc:
            return jsonify({"status": "error", "message": f"EVS RAWを読めません: {exc}"}), 500
        response = Response(png, mimetype="image/png")
        response.headers["Cache-Control"] = "private, max-age=3600"
        return response

    @app.route("/set_preview", methods=["POST"])
    def set_preview():
        data = request.get_json(silent=True) or {}
        preset = data.get("preset")
        auto_degrade = data.get("auto_degrade", True)
        if preset not in PREVIEW_PRESETS or not isinstance(auto_degrade, bool):
            return jsonify({"status": "error", "message": "表示設定が不正です。"}), 400
        try:
            preview_status = get_streamer().update_preview_preferences(
                preset, auto_degrade, persist=bool(data.get("persist", True)))
        except (OSError, ValueError) as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        return jsonify({
            "status": "success",
            "message": f"Frame表示を{PREVIEW_PRESETS[preset]['label']}へ変更しました。",
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
            "message": "Frame接続試行を有効にしました。" if active else "Frame接続試行を休止しました。",
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

    @app.route("/start_recording", methods=["POST"])
    def start_recording():
        streamer = get_streamer()
        data = request.get_json(silent=True) or {}
        try:
            success = streamer.start_recording(data.get("session_id"))
        except (OSError, ValueError) as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400
        message = "Frame 録画を開始しました。" if success else "Frame 録画を開始できませんでした。"
        return jsonify({
            "status": "success" if success else "error",
            "message": message,
            "session_id": streamer.recording_session_id,
        }), (200 if success else 409)

    @app.route("/trigger_options")
    def trigger_options():
        camera = get_streamer().cam_thread
        options = camera.get_trigger_options()
        return jsonify({
            "status": "success",
            **options,
            "configuration": camera.trigger_configuration(),
        })

    @app.route("/set_bandwidth_preset", methods=["POST"])
    def set_bandwidth_preset():
        guard = reject_settings_change_while_recording()
        if guard:
            return guard
        streamer = get_streamer()
        preset = (request.get_json(silent=True) or {}).get("preset")
        if preset not in bandwidth_presets:
            return jsonify({"status": "error", "message": "safe / standard / high を指定してください。"}), 400
        success, result = streamer.cam_thread.request_control_operation(
            "set_bandwidth_preset", preset=preset)
        if not success:
            return jsonify({"status": "error", "message": result}), 500
        config = load_config()
        config.setdefault("frameCam", {})["bandwidthPreset"] = preset
        config["frameCam"]["frameRate"] = streamer.cam_thread.free_run_fps
        save_config(config)
        return jsonify({
            "status": "success",
            "message": f"帯域プリセットを {preset} に設定しました。",
            "bandwidth": result,
            "framerate": streamer.cam_thread.framerate_capabilities(),
        })

    @app.route("/set_external_trigger", methods=["POST"])
    def set_external_trigger():
        guard = reject_settings_change_while_recording()
        if guard:
            return guard
        streamer = get_streamer()
        data = request.get_json(silent=True) or {}
        enabled = data.get("enabled")
        source = data.get("source", default_trigger_source)
        activation = data.get("activation", default_trigger_activation)
        if not isinstance(enabled, bool):
            return jsonify({"status": "error", "message": "enabledは真偽値で指定してください。"}), 400
        options = streamer.cam_thread.get_trigger_options()
        if enabled and source not in options["sources"]:
            return jsonify({"status": "error", "message": f"利用できないTriggerSourceです: {source}"}), 400
        if enabled and activation not in options["activations"]:
            return jsonify({"status": "error", "message": f"利用できないTriggerActivationです: {activation}"}), 400
        success, result = streamer.cam_thread.request_control_operation(
            "set_external_trigger", enabled=enabled, source=source, activation=activation)
        if not success:
            return jsonify({"status": "error", "message": result}), 500
        return jsonify({
            "status": "success",
            "message": "外部トリガー駆動を有効にしました。" if enabled else "フリーラン駆動へ戻しました。",
            "configuration": result,
        })

    @app.route("/stop_recording", methods=["POST"])
    def stop_recording():
        streamer = get_streamer()
        success = streamer.stop_recording(wait_for_writer=False)
        message = (
            "Frame の撮影受付を停止しました。残り画像はバックグラウンドで保存します。"
            if success else "Frame は録画中ではありません。")
        return jsonify({
            "status": "success" if success else "error",
            "message": message,
            "finalizing": bool(streamer.recording_finalizing),
        }), (200 if success else 409)

    @app.route("/status")
    def status():
        streamer = get_streamer()
        camera = streamer.cam_thread
        camera_ready = bool(camera.cam and camera.streaming_active)
        camera_settings = camera.read_camera_settings() if camera_ready else {}
        writer = streamer.image_thread
        summary = streamer.last_recording_summary
        return jsonify({
            "status": "success",
            "streaming": camera_ready,
            "frame_ready": bool(streamer.preview.get_jpeg()),
            "preview": streamer.preview.status(),
            "recording": bool(streamer.recording),
            "save_location": streamer.save_location,
            "save_filename": streamer.save_filename,
            "recording_finalizing": bool(streamer.recording_finalizing),
            "callback_count": camera.callback_count,
            "frame_count": camera.frame_count,
            "last_frame_status": camera.last_frame_status,
            "allocation_mode": camera.allocation_mode,
            "message": camera.last_error,
            "camera_settings": camera_settings,
            "trigger_configuration": camera.trigger_configuration(),
            "external_trigger_monitor": camera.external_trigger_monitor_status(),
            "recording_roi": dict(camera.roi_configuration),
            "measured_fps": round(camera.measured_fps(), 3),
            "connection": {
                "state": streamer.connection_state,
                "retry_active": streamer.capture_retry_active(),
                "restart_attempts": streamer.camera_restart_count,
                "successful_reconnections": streamer.successful_reconnections,
                "stream_epoch": camera.stream_epoch,
            },
            "bandwidth": camera.bandwidth_status(),
            "recording_quality": {
                "session_id": streamer.recording_session_id,
                "events": streamer.recording_sequence,
                "complete": streamer.recording_complete_count,
                "incomplete": streamer.recording_incomplete_count,
                "frame_id_missing": streamer.recording_frame_gap_count,
                "queue_drops": streamer.recording_queue_drop_count,
                "queue_depth": streamer.recording_queue.qsize() if streamer.recording_queue else 0,
                "saved": writer.saved_count if writer else summary.get("saved_frames", 0),
                "write_failures": writer.failed_count if writer else summary.get("write_failures", 0),
                "save_fps": writer.performance_status()["save_fps"] if writer else summary.get("save_fps", 0),
                "write_megabytes_per_second": (
                    writer.performance_status()["write_megabytes_per_second"]
                    if writer else summary.get("write_megabytes_per_second", 0)),
                "elapsed_seconds": round(
                    (time.time_ns() - streamer.recording_started_utc_ns) / 1e9, 1)
                    if streamer.recording and streamer.recording_started_utc_ns else 0,
            },
        })

    def set_auto_feature(feature_name, value_name, label):
        guard = reject_settings_change_while_recording()
        if guard:
            return guard
        data = request.get_json(silent=True) or {}
        mode = data.get("mode")
        if mode not in ("Once", "Continuous", "Manual"):
            return jsonify({"status": "error", "message": "modeの指定が不正です。"}), 400
        try:
            camera = get_streamer().cam_thread.cam
            if camera is None:
                return jsonify({"status": "error", "message": "カメラが初期化されていません。"}), 500
            auto_feature = getattr(camera, feature_name)
            value_feature = getattr(camera, value_name)
            if mode in ("Once", "Continuous"):
                auto_feature.set(mode)
                time.sleep(0.1)
                value = value_feature.get()
                return jsonify({"status": "success", "message": f"{label} set to {value}."})
            manual_value = data.get("value")
            try:
                manual_value = float(manual_value)
            except (TypeError, ValueError):
                return jsonify({"status": "error", "message": "Manual valueが不正です。"}), 400
            minimum, maximum = value_feature.get_range()
            if not minimum <= manual_value <= maximum:
                return jsonify({
                    "status": "error",
                    "message": f"Manual value must be between {minimum} and {maximum}.",
                }), 400
            auto_feature.set("Off")
            value_feature.set(manual_value)
            time.sleep(0.1)
            value = value_feature.get()
            return jsonify({"status": "success", "message": f"{label} manually set to {value}."})
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/set_exposure", methods=["POST"])
    def set_exposure():
        return set_auto_feature("ExposureAuto", "ExposureTime", "Exposure")

    @app.route("/set_gain", methods=["POST"])
    def set_gain():
        return set_auto_feature("GainAuto", "Gain", "Gain")

    @app.route("/set_framerate", methods=["POST"])
    def set_framerate():
        guard = reject_settings_change_while_recording()
        if guard:
            return guard
        data = request.get_json(silent=True) or {}
        try:
            fps = float(data.get("fps"))
        except (TypeError, ValueError):
            return jsonify({"status": "error", "message": "fps not provided or invalid."}), 400
        if not math.isfinite(fps) or fps <= 0:
            return jsonify({"status": "error", "message": "fps must be a positive finite number."}), 400
        camera = get_streamer().cam_thread
        capabilities = camera.framerate_capabilities()
        if not capabilities["minimum"] <= fps <= capabilities["maximum"]:
            return jsonify({
                "status": "error",
                "message": f"フレームレートは {capabilities['minimum']:g}〜{capabilities['maximum']:g} fps で指定してください。",
            }), 400
        try:
            success, result = camera.request_control_operation("set_framerate", fps=fps)
            if not success:
                return jsonify({"status": "error", "message": result}), 500
            persisted = True
            persistence_error = None
            try:
                config = load_config()
                config.setdefault("frameCam", {})["frameRate"] = float(result)
                save_config(config)
            except Exception as exc:
                persisted = False
                persistence_error = str(exc)
            return jsonify({
                "status": "success",
                "fps": result,
                "capabilities": camera.framerate_capabilities(),
                "persisted": persisted,
                "persistence_error": persistence_error,
                "message": (
                    f"フレームレートを {result:g} fpsに設定し、再起動後の設定にも保存しました。"
                    if persisted else
                    f"フレームレートは {result:g} fpsに設定しましたが、設定ファイルへ保存できませんでした。"),
            })
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/set_whitebalance", methods=["POST"])
    def set_whitebalance():
        guard = reject_settings_change_while_recording()
        if guard:
            return guard
        mode = (request.get_json(silent=True) or {}).get("mode")
        if mode not in ("Once", "Continuous"):
            return jsonify({"status": "error", "message": "modeの指定が不正です。"}), 400
        try:
            camera = get_streamer().cam_thread.cam
            if camera is None:
                return jsonify({"status": "error", "message": "カメラが初期化されていません。"}), 500
            camera.BalanceWhiteAuto.set(mode)
            return jsonify({"status": "success", "message": f"WhiteBalance set to {mode}."})
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    @app.route("/get_settings")
    def get_settings():
        try:
            streamer = get_streamer()
            camera = streamer.cam_thread.cam
            if camera is None:
                return jsonify({"status": "error", "message": "カメラが初期化されていません。"}), 500
            fps = streamer.cam_thread.framerate_capabilities()
            return jsonify({
                "status": "success",
                "exposure": camera.ExposureTime.get(),
                "gain": camera.Gain.get(),
                "fps": fps["configured"],
                "fps_target": fps["target"],
                "fps_measured": fps["measured"],
                "fps_min": fps["minimum"],
                "fps_max": fps["maximum"],
                "fps_increment": fps["increment"],
            })
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc)}), 500

    return app
