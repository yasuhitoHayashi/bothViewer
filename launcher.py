"""
@author: HAYASHI Yasuhito (dangom_ya)

Licensed under the Apache License, Version 2.0.
"""
import subprocess
import webbrowser
import time
import importlib.util
import os
import sys


REQUIRED_MODULES = {
    "cv2": "opencv-python",
    "ffmpeg": "ffmpeg-python",
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


def stop_process(process):
    """子プロセスが動作中の場合だけ、安全に終了する。"""
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
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
    
    evs_proc = subprocess.Popen([sys.executable, evs_script, "--port", "5001"])
    frame_proc = subprocess.Popen([sys.executable, frame_script, "--port", "5002"])
    
    time.sleep(5)
    
    html_url = "file://" + html_file
    webbrowser.open(html_url)
    
    print("サーバーを起動しました。CTRL+C で終了します。")
    
    exit_code = 0
    try:
        while evs_proc.poll() is None and frame_proc.poll() is None:
            time.sleep(0.5)
        failed_name = "EVS" if evs_proc.poll() is not None else "Frame"
        failed_code = evs_proc.poll() if evs_proc.poll() is not None else frame_proc.poll()
        print(f"{failed_name} サーバーが終了しました (終了コード: {failed_code})", file=sys.stderr)
        exit_code = failed_code or 1
    except KeyboardInterrupt:
        print("終了処理中...")
    finally:
        stop_process(evs_proc)
        stop_process(frame_proc)
        print("終了しました。")
    return exit_code

if __name__ == "__main__":
    sys.exit(main())
