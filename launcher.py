"""Mode-selecting process launcher for bothViewer."""

import importlib.util
import os
import subprocess
import sys
import threading
import time
import webbrowser

from flask import Flask, jsonify, request, send_file
from werkzeug.serving import make_server

from bothviewer.core.config import load_config, resolve_recording_directory


BASE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
LAUNCHER_PORT = 5050
MODE_VIEWERS = {
    "capture_both": ("bothViewer.html", "mode=capture"),
    "capture_evs_single": ("evsViewer.html", "count=1"),
    "capture_evs_dual": ("evsViewer.html", "count=2"),
    "review_both": ("bothViewer.html", "mode=review"),
    "review_evs_single": ("evsDataViewer.html", "count=1"),
    "review_evs_dual": ("evsDataViewer.html", "count=2"),
}
COMMON_MODULES = {
    "cv2": "opencv-python", "flask": "Flask", "h5py": "h5py",
    "metavision_core": "OpenEB / Metavision SDK", "numpy": "numpy", "yaml": "PyYAML",
}


def check_dependencies(mode=None):
    required = dict(COMMON_MODULES)
    if mode == "capture_both":
        required["vmbpy"] = "Vimba X SDK付属のVmbPy wheel"
    return [package for module, package in required.items()
            if importlib.util.find_spec(module) is None]


def request_process_stop(process):
    if process.poll() is None:
        process.terminate()


def wait_process_stop(process):
    if process.poll() is not None:
        return
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


class ProcessSupervisor:
    def __init__(self, save_location):
        self.save_location = save_location
        self.lock = threading.RLock()
        self.mode = None
        self.commands = {}
        self.processes = {}
        self.running = True
        self.monitor = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor.start()

    def _command_map(self, mode, selectors):
        python = sys.executable
        evs = os.path.join(BASE_DIRECTORY, "evsStreamer.py")
        frame = os.path.join(BASE_DIRECTORY, "frameStreamer.py")
        data = os.path.join(BASE_DIRECTORY, "dataServer.py")
        common = ["--save_location", self.save_location]
        if mode == "capture_both":
            return {
                "EVS": [python, evs, "--port", "5001", *common,
                        "--camera-selector", selectors[0], "--role", "evs"],
                "Frame": [python, frame, "--port", "5002", *common],
            }
        if mode == "capture_evs_single":
            return {"EVS": [python, evs, "--port", "5001", *common,
                            "--camera-selector", selectors[0], "--role", "evs"]}
        if mode == "capture_evs_dual":
            if not selectors[0] or not selectors[1] or selectors[0] == selectors[1]:
                raise ValueError("EVS 2台撮影では異なる2つのシリアル番号を指定してください。")
            return {
                "EVS-A": [python, evs, "--port", "5001", *common,
                          "--camera-selector", selectors[0], "--role", "evs_a"],
                "EVS-B": [python, evs, "--port", "5003", *common,
                          "--camera-selector", selectors[1], "--role", "evs_b"],
            }
        if mode.startswith("review_"):
            return {"Data": [python, data, "--port", "5002", *common]}
        raise ValueError("起動モードが不正です。")

    def stop_services(self):
        with self.lock:
            processes = list(self.processes.values())
            self.processes = {}
            self.commands = {}
            self.mode = None
        for process in processes:
            request_process_stop(process)
        for process in processes:
            wait_process_stop(process)

    def launch(self, mode, selectors):
        if mode not in MODE_VIEWERS:
            raise ValueError("起動モードが不正です。")
        missing = check_dependencies(mode)
        if missing:
            raise RuntimeError("不足モジュール: " + ", ".join(missing))
        commands = self._command_map(mode, selectors)
        self.stop_services()
        with self.lock:
            self.mode = mode
            self.commands = commands
            self.processes = {
                name: subprocess.Popen(command, cwd=BASE_DIRECTORY)
                for name, command in commands.items()
            }
        filename, query = MODE_VIEWERS[mode]
        return f"http://127.0.0.1:{LAUNCHER_PORT}/{filename}?{query}"

    def _monitor_loop(self):
        while self.running:
            with self.lock:
                for name, process in list(self.processes.items()):
                    if process.poll() is not None and name in self.commands:
                        print(f"{name} サーバー終了 ({process.returncode})。再起動します。", file=sys.stderr)
                        self.processes[name] = subprocess.Popen(
                            self.commands[name], cwd=BASE_DIRECTORY)
            time.sleep(1)

    def shutdown(self):
        self.running = False
        self.stop_services()
        self.monitor.join(timeout=2)


def create_launcher_app(supervisor):
    app = Flask(
        "bothviewer.launcher",
        static_folder=os.path.join(BASE_DIRECTORY, "static"),
        static_url_path="/static",
    )

    @app.route("/")
    def index():
        return send_file(os.path.join(BASE_DIRECTORY, "modeSelector.html"))

    @app.route("/<viewer_name>")
    def viewer(viewer_name):
        allowed = {filename for filename, _query in MODE_VIEWERS.values()}
        if viewer_name not in allowed:
            return jsonify({"status": "error", "message": "画面が見つかりません。"}), 404
        return send_file(os.path.join(BASE_DIRECTORY, viewer_name))

    @app.after_request
    def cors(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.route("/api/cameras")
    def cameras():
        try:
            from metavision_hal import DeviceDiscovery
            values = [str(value) for value in DeviceDiscovery.list()]
            return jsonify({"status": "success", "cameras": values})
        except Exception as exc:
            return jsonify({"status": "error", "message": str(exc), "cameras": []}), 503

    @app.route("/api/launch", methods=["POST"])
    def launch():
        data = request.get_json(silent=True) or {}
        selectors = [str(data.get("evs1") or "").strip(), str(data.get("evs2") or "").strip()]
        try:
            viewer_url = supervisor.launch(str(data.get("mode") or ""), selectors)
            return jsonify({"status": "success", "viewer_url": viewer_url})
        except (RuntimeError, ValueError) as exc:
            return jsonify({"status": "error", "message": str(exc)}), 400

    @app.route("/api/status")
    def status():
        with supervisor.lock:
            processes = {
                name: process.poll() is None for name, process in supervisor.processes.items()
            }
            return jsonify({"status": "success", "mode": supervisor.mode, "processes": processes})

    return app


def main():
    if importlib.util.find_spec("flask") is None:
        print("Flaskが必要です。python -m pip install -r requirements.txt", file=sys.stderr)
        return 1
    config = load_config()
    configured = config.get("recording", {}).get("save_location", "./records")
    save_location = resolve_recording_directory(
        configured, application_directory=BASE_DIRECTORY)
    os.makedirs(save_location, exist_ok=True)
    supervisor = ProcessSupervisor(save_location)
    app = create_launcher_app(supervisor)
    try:
        # SDKのカメラ列挙が長引いても、モード起動APIを待たせない。
        http_server = make_server("127.0.0.1", LAUNCHER_PORT, app, threaded=True)
    except OSError as exc:
        supervisor.shutdown()
        print(f"起動モードサーバーを開始できません: {exc}", file=sys.stderr)
        return 1
    server = threading.Thread(target=http_server.serve_forever, daemon=True)
    server.start()
    webbrowser.open(f"http://127.0.0.1:{LAUNCHER_PORT}/")
    print("モード選択画面を起動しました。CTRL+Cで終了します。")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("終了処理中...")
    finally:
        http_server.shutdown()
        supervisor.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
