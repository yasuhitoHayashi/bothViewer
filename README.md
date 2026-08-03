# bothViewer

bothViewer provides a web-based control console for simultaneously viewing and recording data from an event camera and a frame camera. It runs two local Flask servers and opens a responsive, dependency-free HTML interface for monitoring both streams and changing camera settings.

The live frame preview demosaics Bayer data with crop-phase correction; recorded PGM values remain untouched. The EVS preview uses a grayscale event palette: gray background, positive events white, and negative events black.

## Installation

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Camera access also requires the vendor SDKs for your hardware:

- Prophesee OpenEB / Metavision SDK (event camera)
- Allied Vision Vimba X Python API (`vmbpy`) (frame camera)

Install and verify those SDKs using the manufacturers' instructions before launching bothViewer. Frame-camera recordings are stored as uncompressed lossless 8-bit Bayer PGM (P5) files; FFmpeg is not required. PGM removes the per-frame PNG compression bottleneck and can be read with `cv2.imread(path, cv2.IMREAD_UNCHANGED)`.

On macOS, Vimba X installs its matching VmbPy wheel separately. Install that wheel into the same environment used to launch bothViewer. A typical installation is:

```bash
python -m pip install "/Users/Shared/Allied Vision/Vimba X/VmbPy/vmbpy-1.0.4-py3-none-any.whl"
```

Use the wheel version present on your machine; it must match the installed Vimba X SDK.

## Usage

Launch both servers and open the viewer with:

```bash
python launcher.py
```

The event stream server starts on port `5001` and the frame stream server starts on port `5002`. After a short delay, `bothViewer.html` opens automatically in your default browser. Use the status indicators in the upper-right corner to confirm that each camera is streaming before recording.

Press `Ctrl+C` in the terminal to stop both servers cleanly.

## Recording output

Each recording creates one timestamped session directory shared by both cameras:

- `frame/images/*.pgm`: uncompressed lossless 8-bit Bayer frames with UTC time, sequence, and camera frame ID in each filename
- `frame/frame_events.csv`: every camera callback, including incomplete frames, frame-ID gaps, and queue results
- `frame/saved_frames.csv`: actual disk-write result and latency for every image
- `frame/camera_settings.json`: camera settings read back at recording start
- `evs/events.raw`, `events_001.raw`, ...: native EVS event-stream segments split at reconnections
- `evs/triggers.csv`: external-trigger edges with EVS and host timestamps
- `evs/camera_settings.json`: EVS bias, trigger, and sensor settings at recording start
- `frame/connection_events.csv`, `evs/connection_events.csv`: disconnect and reconnect audit trail
- `synchronization.csv`: frame-to-exposure-start correspondence and unmatched records (the reference edge follows `LineInverter`)
- `session.json`: session-wide quality summary

The recording panel shows incomplete frames, ID gaps, queue drops, and write failures while an experiment is running. Any non-zero loss counter should be treated as a quality warning.

Stopping a session first closes frame acceptance, then stops EVS while the frame writer drains in the background. `duration_seconds` therefore reports the actual capture interval; `writer_drain_seconds` separately reports post-capture save time. The GUI also reports measured save fps and MB/s. A manual **EVSを再接続** control is available; during recording, reconnection continues into a new EVS RAW segment.

The EVS view includes a live Trigger In monitor even when no recording is active. It shows recent activity, the reference-edge frequency and period, cumulative rising/falling edges, and the age and polarity of the latest edge. Frequency history resets after EVS reconnection so timestamps from different sensor epochs are never mixed.

The frame view has a corresponding external-trigger monitor. When external drive is enabled it shows the selected input line and activation edge, complete/incomplete trigger-driven frame callbacks, measured frame frequency, latest-frame age, and the `Line0 ExposureActive -> EVS Trigger In` synchronization state. Thus an external input starts the frame exposure, and that exposure simultaneously produces the timing edge recorded by EVS.

The frame camera is configured with a hardware ROI matching the EVS sensor's physical field of view (1800×1012 at the default sensor specifications, with the configured frame shift applied). This avoids transferring the unused border pixels. If the camera rejects hardware ROI control, the application restores full-sensor acquisition and automatically falls back to the same software crop. The actual mode, size, and sensor offset are stored in `camera_settings.json`, `frame_summary.json`, `/status`, and the frame audit CSV.

## Trigger topology

The EVS Trigger In and the frame camera's `Line0 = ExposureActive` output remain enabled at all times. The trigger panel controls only how the frame camera is driven:

```text
External trigger -> frame camera input line -> frame exposure
                                              -> Line0 ExposureActive -> EVS Trigger In
```

In external-trigger mode, the selected input edge determines each frame timing and the internal frame-rate limiter is disabled. Recording can be armed before the first external pulse arrives. Disable external-trigger mode to return to the configured free-run frame rate.

## Free-run frame rate

In normal free-run mode, the GUI reads the camera's supported fps range and allows the target rate to be changed directly. The stream is restarted safely, the applied value is read back from the camera, and the setting is persisted to `config.yaml`. If the camera rejects the value, the previous fps and stream are restored. The GUI shows both the configured fps and the measured callback rate, and warns when exposure time exceeds the requested frame period. Frame-rate controls are disabled in external-trigger mode.

## Bandwidth and recovery

The frame-camera GUI provides Safe (100 MB/s), Standard (150 MB/s), and High (200 MB/s) transport presets. If three incomplete frames occur within five seconds, the application automatically steps down one preset and persists the safer value. Camera disconnects are retried with exponential backoff up to five seconds. Frame recording continues in the same image directory with a new `stream_epoch`; EVS recording starts a new RAW segment because appending safely across device reconnection is not guaranteed. The browser MJPEG views also reconnect automatically.

## Performance

You can further reduce CPU usage by lowering the display scale factor. Pass
`--display-factor 0.3` (for example) when starting `frameStreamer.py` or
`evsStreamer.py` to downscale frames before encoding.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
