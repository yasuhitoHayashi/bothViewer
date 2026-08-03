"""
@author: HAYASHI Yasuhito (dangom_ya)

Licensed under the Apache License, Version 2.0.
"""
import os
import re
import yaml
from datetime import datetime

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "config.yaml")
SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def create_session_directory(save_location, session_id):
    """検証済みsession_idで共通の記録ディレクトリを作成する。"""
    if not isinstance(session_id, str) or not SESSION_ID_PATTERN.fullmatch(session_id):
        raise ValueError("session_id は英数字・ピリオド・ハイフン・アンダースコアのみ使用できます。")
    base_directory = os.path.abspath(os.path.expanduser(save_location))
    session_directory = os.path.join(base_directory, session_id)
    os.makedirs(session_directory, exist_ok=True)
    return session_directory

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
                "frameRate": 10,
                "bandwidthPreset": "safe",
                "externalTrigger": {
                    "enabled": False,
                    "source": "Line1",
                    "activation": "RisingEdge",
                    "outputLine": "Line0",
                    "outputInverter": True,
                },
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
    # 短いインターバル録画でも同名にならないよう、ミリ秒まで含める。
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")[:-3]
    filename = f"{timestamp}_{prefix}.yaml"
    snapshot_path = os.path.join(snapshot_dir, filename)
    with open(snapshot_path, "w") as f:
        yaml.safe_dump(config, f)
    return snapshot_path
