"""
@author: HAYASHI Yasuhito (dangom_ya)

Licensed under the Apache License, Version 2.0.
"""
import subprocess
import webbrowser
import time
import importlib.util
import os
from pathlib import Path
import sys


REQUIRED_MODULES = {
    "cv2": "opencv-python",
    "flask": "Flask",
    "h5py": "h5py",
    "metavision_core": "OpenEB / Metavision SDK",
    "numpy": "numpy",
    "vmbpy": "Vimba X SDK付属のVmbPy wheel",
    "yaml": "PyYAML",
}


def check_dependencies():
    """子プロセス起動前に、不足しているPythonモジュールをまとめて返す。"""
    return [package for module, package in REQUIRED_MODULES.items()
            if importlib.util.find_spec(module) is None]


def request_process_stop(process):
    """待たずに終了要求を送り、複数バックエンドの停止時刻を揃える。"""
    if process.poll() is not None:
        return
    process.terminate()


def wait_process_stop(process):
    """終了要求済みの子プロセスが監査ファイルを閉じるまで待つ。"""
    if process.poll() is not None:
        return
    try:
        # Bayer画像キューとCSVを閉じる時間を子プロセスへ与える。
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main():
    missing = check_dependencies()
    if missing:
        print("起動に必要なPythonモジュールが不足しています:", file=sys.stderr)
        for package in missing:
            print(f"  - {package}", file=sys.stderr)
        print("\nまず次を実行してください:", file=sys.stderr)
        print(f"  {sys.executable} -m pip install -r requirements.txt", file=sys.stderr)
        if "Vimba X SDK付属のVmbPy wheel" in missing:
            print("  Vimba X SDK付属の vmbpy-*.whl も同じ環境へインストールしてください。", file=sys.stderr)
        return 1

    base_dir = os.path.dirname(os.path.abspath(__file__))
    evs_script = os.path.join(base_dir, "evsStreamer.py")
    frame_script = os.path.join(base_dir, "frameStreamer.py")
    html_file = os.path.join(base_dir, "bothViewer.html")

    def build_commands():
        # Import only after dependency validation so a missing PyYAML still
        # produces the launcher's consolidated installation guidance.
        from config_manager import load_config, resolve_recording_directory
        config = load_config()
        configured = config.get("recording", {}).get("save_location", "./records")
        save_location = resolve_recording_directory(
            configured, application_directory=base_dir)
        os.makedirs(save_location, exist_ok=True)
        return {
            "EVS": [sys.executable, evs_script, "--port", "5001",
                    "--save_location", save_location],
            "Frame": [sys.executable, frame_script, "--port", "5002",
                      "--save_location", save_location],
        }

    commands = build_commands()
    processes = {name: subprocess.Popen(command) for name, command in commands.items()}
    
    time.sleep(5)
    
    html_url = Path(html_file).as_uri()
    webbrowser.open(html_url)
    
    print("サーバーを起動しました。CTRL+C で終了します。")
    
    exit_code = 0
    try:
        while True:
            for name, process in list(processes.items()):
                if process.poll() is not None:
                    print(
                        f"{name} サーバーが終了しました (終了コード: {process.returncode})。再起動します。",
                        file=sys.stderr,
                    )
                    time.sleep(1)
                    commands = build_commands()
                    processes[name] = subprocess.Popen(commands[name])
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("終了処理中...")
    finally:
        # 両方へ先に停止要求を送り、Frameの保存キュー排出中にEVSだけが
        # ExposureActiveを記録し続ける時間を作らない。
        request_process_stop(processes["Frame"])
        request_process_stop(processes["EVS"])
        wait_process_stop(processes["Frame"])
        wait_process_stop(processes["EVS"])
        print("終了しました。")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
