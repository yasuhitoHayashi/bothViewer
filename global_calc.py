# Copyright (c) 2025 CenturyArks Co.,Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.

"""
Calc Defines
"""

from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
import math

"""
Determine the size of the image sensor.
The returned sensor size is returned as a value (n.nn) rounded half up to the third decimal place.

arg: Resolution:(w,h) Pixel Size:(w,h)
return: Sensor Size:(w,h)
"""
def get_cencer_size(resolution:tuple, pixel_size:tuple) -> tuple:

    r_w, r_h = resolution
    p_w, p_h = pixel_size

    c_w = p_w * r_w * (10 ** -3)
    c_h = p_h * r_h * (10 ** -3)
    c_w = Decimal(str(c_w)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    c_h = Decimal(str(c_h)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)

    return (c_w, c_h)

"""
Determine the size of the image resolution.
Return the size of excess pixels.

arg: Frame Resolution:(w,h) Event Resolution:(w,h) Frame Sensor:(w,h) Event Sensor:(w,h)
return: Frame Trim Size:(w,h)
"""
def get_trim_pixel_size(frame_resolution:tuple, event_resolution:tuple, frame_sensor:tuple, event_sensor:tuple) -> tuple:
    frame_res_w, frame_res_h = frame_resolution
    event_res_w, event_res_h = event_resolution
    frame_sensor_w, frame_sensor_h = frame_sensor
    event_sensor_w, event_sensor_h = event_sensor

    if frame_sensor_w < event_sensor_w:
        """TODO:Not used, commented out
        #print(f'width:Event is large')
        event_res_w_calc = event_res_w * frame_sensor_w / event_sensor_w
        event_res_w_calc = Decimal(str(event_res_w_calc)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
        event_res_w_calc = int(event_res_w_calc if event_res_w_calc % 2 == 0 else event_res_w_calc + 1)
        #event_trim_w = int((event_res_w - event_res_w_calc) / 2)
        """
    else:
        # TODO:For debugging
        #print(f'width:Frame is large or same')
        frame_res_w_calc = frame_res_w * event_sensor_w / frame_sensor_w
        frame_res_w_calc = Decimal(str(frame_res_w_calc)).quantize(Decimal('0'), rounding=ROUND_DOWN)
        frame_res_w_calc = int(frame_res_w_calc if frame_res_w_calc % 2 == 0 else frame_res_w_calc - 1)
        frame_trim_w = int((frame_res_w - frame_res_w_calc) / 2)

    if frame_sensor_h < event_sensor_h:
        """TODO:Not used, commented out
        #print(f'height:Event is large')
        event_res_h_calc = event_res_h * frame_sensor_h / event_sensor_h
        event_res_h_calc = Decimal(str(event_res_h_calc)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
        event_res_h_calc = int(event_res_h_calc if event_res_h_calc % 2 == 0 else event_res_h_calc + 1)
        #event_trim_h = int((event_res_h - event_res_h_calc) / 2)
        """
    else:
        # TODO:For debugging
        #print(f'height:Frame is large or same')
        frame_res_h_calc = frame_res_h * event_sensor_h / frame_sensor_h
        frame_res_h_calc = Decimal(str(frame_res_h_calc)).quantize(Decimal('0'), rounding=ROUND_DOWN)
        frame_res_h_calc = int(frame_res_h_calc if frame_res_h_calc % 2 == 0 else frame_res_h_calc - 1)
        frame_trim_h = int((frame_res_h - frame_res_h_calc) / 2)

        return (frame_trim_w, frame_trim_h)

"""
Adjust ROI to meet frame camera specifications.
Adjust to the Interval value of the specification.(w:8px h:2px)

arg: Frame ROI:(w,h) Frame Trim:(x,y)
return: Ajdust Frame ROI:(w,h) Frame Trim:(x,y)
"""
def get_adjusted_roi(frame_roi:tuple, frame_trim:tuple):
    # Constants
    W_INTERVAL = 8
    H_INTERVAL = 2

    frame_roi_w, frame_roi_h = frame_roi
    ret_roi_w = math.floor(frame_roi_w / W_INTERVAL) * W_INTERVAL
    ret_roi_h = math.floor(frame_roi_h / H_INTERVAL) * H_INTERVAL

    frame_trim_x, frame_trim_y = frame_trim
    ret_trim_x = frame_trim_x + (frame_roi_w - ret_roi_w) / 2
    ret_trim_y = frame_trim_y + (frame_roi_h - ret_roi_h) / 2

    return ((ret_roi_w, ret_roi_h), (ret_trim_x, ret_trim_y))

"""
Adjust the position of the frame camera image.
Adjust to the Interval value of the specification.(2px)

arg: Offset(x or y)
return: Ajdust Value
"""
def get_adjusted_offset(offset:int):
    # Constants
    OFFSET_INTERVAL = 2

    is_minus = False
    if offset < 0:
        is_minus = True
        offset = offset * -1

    adjust = math.floor(offset / OFFSET_INTERVAL) * OFFSET_INTERVAL

    if is_minus:
        # Return to minus value.
        adjust = adjust * -1

    return adjust


