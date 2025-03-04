"""
@author: HAYASHI Yasuhito (dangom_ya)

CopyPolicy: 
    Released under the terms of the LGPLv2.1 or later.
"""
import yaml
import os
from datetime import datetime

CONFIG_FILE = "config.yaml"

def load_config():
    if not os.path.exists(CONFIG_FILE):
        # デフォルトの設定値（config.yaml の内容と一致させる）
        default_config = {
            "eventCam": {
                "bias": {"diff_on": 20, "diff_off": 20},
                "triggerIn": True
            },
            "frameCam": {
                "exposure": "Once",
                "gain": {"mode": "Manual", "value": 0}
            },
            "recording": {
                "save_location": "./recordings",
                "file_prefix": "record"
            },
            "bothViewHW": {
                "eventCamHW": {
                    "resolution": [1280, 720],
                    "pixel": {"width": 4.86, "height": 4.86}
                },
                "frameCamHW": {
                    "resolution": [1936, 1216],
                    "pixel": {"width": 3.45, "height": 3.45},
                    "frame_shift": {"width": 0, "height": 0}
                }
            }
        }
        save_config(default_config)
        return default_config
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        yaml.safe_dump(config, f)

def save_config_snapshot(config, snapshot_dir, prefix="config_snapshot"):
    """撮影開始時の設定を snapshot_dir にタイムスタンプ付きファイルとして保存する"""
    if not os.path.exists(snapshot_dir):
        os.makedirs(snapshot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{timestamp}_{prefix}.yaml"
    snapshot_path = os.path.join(snapshot_dir, filename)
    with open(snapshot_path, "w") as f:
        yaml.safe_dump(config, f)
    return snapshot_path
