#!/usr/bin/env python3
import subprocess
import webbrowser
import time
import os
import sys

def main():
    # スクリプトのパス（必要に応じて相対パスや絶対パスを調整）
    evs_script = os.path.join(os.path.dirname(__file__), "evsStreamer.py")
    frame_script = os.path.join(os.path.dirname(__file__), "frameStreamer.py")
    html_file = os.path.join(os.path.dirname(__file__), "bothViewer.html")
    
    evs_proc = subprocess.Popen([sys.executable, evs_script, "--port", "5001"])
    frame_proc = subprocess.Popen([sys.executable, frame_script, "--port", "5002"])
    
    # 少し待って各サーバーが起動するのを待機（必要に応じて調整）
    time.sleep(5)
    
    # bothViewer.html を既定のブラウザで開く
    html_url = "file:///" + os.path.abspath(html_file)
    webbrowser.open(html_url)
    
    print("サーバーを起動しました。CTRL+C で終了します。")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("終了処理中...")
        evs_proc.terminate()
        frame_proc.terminate()
        evs_proc.wait()
        frame_proc.wait()
        print("終了しました。")

if __name__ == "__main__":
    main()