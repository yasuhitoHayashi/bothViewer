"""
@author: HAYASHI Yasuhito (dangom_ya)

CopyPolicy: 
    Released under the terms of the LGPLv2.1 or later.
"""
import subprocess
import webbrowser
import time
import os
import sys

def main():
    evs_script = os.path.join(os.path.dirname(__file__), "evsStreamer.py")
    frame_script = os.path.join(os.path.dirname(__file__), "frameStreamer.py")
    html_file = os.path.join(os.path.dirname(__file__), "bothViewer.html")
    
    evs_proc = subprocess.Popen([sys.executable, evs_script, "--port", "5001"])
    frame_proc = subprocess.Popen([sys.executable, frame_script, "--port", "5002"])
    
    time.sleep(5)
    
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