# Copyright (c) 2025 CenturyArks Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Process Defines
"""

import subprocess
import ffmpeg
import os

def createMp4Process(img_w: int, img_h: int, recording_name: str, crop_trim: tuple[int, int, int, int], frame_using_gpu=False, bitrate=-1) -> subprocess.Popen:

    # Constants
    _CODEC_DICT = {'GPU':'h264_nvenc', 'CPU':'h264'} # No used 'AMF':'h264_amf', 'QSV':'h264_qsv'

    v_codec = _CODEC_DICT['GPU'] if frame_using_gpu else _CODEC_DICT['CPU']
    if bitrate==-1:
        bitrate = 1000

    img_trim_offset_x, img_trim_offset_y, img_trim_width, img_trim_height = crop_trim

    file_path = recording_name + '.mp4'
    file_path = os.path.join('.', "video", file_path)

    # Memo:
    # To minimize missing frames, "vsync=passthrough" is used.
    # Normally set "vsync=vfr".
    ffmpeg_process = (
        ffmpeg
        .input('pipe:', format='rawvideo', pix_fmt='bgr24',
            s='{}x{}'.format(img_w, img_h), use_wallclock_as_timestamps=1)
        .crop(img_trim_offset_x, img_trim_offset_y, img_trim_width, img_trim_height)
        .output(file_path, pix_fmt='yuv420p', vcodec=v_codec, vsync='passthrough', video_bitrate='{}k'.format(bitrate), loglevel="info")
        .run_async(pipe_stdin=True, overwrite_output=True) # Display log on prompt
    )

    return ffmpeg_process