# bothViewer

bothViewer provides a web-based control console for simultaneously viewing and recording data from an event camera and a frame camera. It runs two local Flask servers and opens a responsive, dependency-free HTML interface for monitoring both streams and changing camera settings.

## Project layout

- `launcher.py`: dependency check, shared recording-directory resolution, and process supervision
- `modeSelector.html`: six-mode startup screen and EVS serial selection
- `evsViewer.html`: single/dual EVS live capture console
- `evsDataViewer.html`: single/dual EVS synchronized RAW playback
- `dataServer.py`: read-only recording browser service
- `frameStreamer.py`, `evsStreamer.py`: backward-compatible executable entry points
- `bothviewer/cameras/frame.py`: frame-camera lifecycle, Bayer recording, and controls
- `bothviewer/cameras/evs.py`: EVS lifecycle, RAW recording, and Trigger In monitoring
- `bothviewer/api/evs.py`: hardware-independent port 5001 HTTP API factory
- `bothviewer/api/frame.py`: hardware-independent port 5002 HTTP API factory
- `bothviewer/api/data.py`: read-only session catalog and EVS/combined playback API
- `bothviewer/api/common.py`: API response helpers shared by both servers
- `bothviewer/core/geometry.py`: physical field-of-view and camera-ROI calculations
- `bothviewer/core/preview.py`: recording-priority latest-frame JPEG workers
- `bothviewer/core/synchronization.py`: frame/trigger audit and rebuildable synchronization index
- `bothviewer/core/recordings.py`: saved-session catalog, Bayer preview, and synchronized playback rendering
- `bothviewer/core/config.py`: configuration, save-path validation, and session-directory helpers
- `bothViewer.html`: UI markup only
- `static/bothViewer.css`, `static/bothViewer.js`: UI presentation and behavior
- `tests/`: hardware-independent regression tests
- `records/`: generated experiment data; created at launch and excluded from Git

The root streamer scripts remain executable entry points for compatibility, but
their Flask routes live under `bothviewer/api`. Each API receives a callable that
returns the current streamer instance, so camera recovery can replace internal
state without leaving the web layer with a stale reference. The API modules do
not import VmbPy or OpenEB and can therefore be regression-tested without camera
hardware.

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

Open the mode selector with:

```bash
python launcher.py
```

`launcher.py` first opens a mode-selection screen. It starts only the services
required by the selected mode:

- **Frame + EVS capture**: synchronized capture using EVS on port 5001 and Frame on 5002
- **Single EVS capture**: one EVS service on port 5001
- **Dual EVS capture**: EVS A on port 5001 and EVS B on port 5003
- **Frame + EVS review**: synchronized frame/event overlay through the read-only data server
- **Single EVS review**: event-only real-time RAW playback
- **Dual EVS review**: two RAW streams aligned by recording-start host UTC

The launcher screen and control API use port 5050. In dual capture mode two distinct EVS
serial numbers must be selected. Returning to the mode-selection page and
choosing another mode stops the previous services cleanly before starting the
new set.

Press `Ctrl+C` in the terminal to stop both servers cleanly.

## Recording output

By default, `launcher.py` creates a `records` directory beside `bothViewer.html` and passes its absolute path to both camera servers. This rule is the same on macOS and Windows (for example, `/path/to/bothViewer/records` or `C:/path/to/bothViewer/records`). The GUI may still be used to select a different absolute recording directory for the current run.

Each recording creates one timestamped session directory shared by both cameras:

- `frame/images/*.pgm`: uncompressed lossless 8-bit Bayer frames with UTC time, sequence, and camera frame ID in each filename
- `frame/frame_events.csv`: every camera callback, including incomplete frames, frame-ID gaps, and queue results
- `frame/saved_frames.csv`: actual disk-write result and latency for every image
- `frame/camera_settings.json`: camera settings read back at recording start
- `evs/events.raw`, `events_001.raw`, ...: native EVS event-stream segments split at reconnections
- `evs/triggers.csv`: external-trigger edges with EVS and host timestamps
- `evs/camera_settings.json`: EVS bias, trigger, and sensor settings at recording start
- `frame/connection_events.csv`, `evs/connection_events.csv`: disconnect and reconnect audit trail
- `synchronization.jsonl`: rebuildable frame-to-exposure-start index and unmatched records (the reference edge follows `LineInverter`)
- `synchronization_summary.json`: synchronization quality summary used by the recording browser
- `synchronization.csv`: optional compatibility export of the same derived index
- `session.json`: session-wide quality summary

`frame/frame_events.csv`, `frame/saved_frames.csv`, and `evs/triggers.csv` are the
primary audit records. Synchronization files are derived indexes: if recording
shutdown is interrupted or disk/bandwidth pressure prevents them from being
written, the browser and synchronized player reconstruct the index on demand.
Index and summary updates use an atomic replacement so a partially written file
is never treated as complete.

Single-EVS recordings use `evs/`. Dual-EVS recordings use separate `evs_a/`
and `evs_b/` directories so RAW, trigger, connection, and settings files never
collide. Their summaries retain camera selector, role, and host UTC start time;
dual playback uses that host time to place both streams on one experiment
timeline.

The recording panel shows incomplete frames, ID gaps, queue drops, and write failures while an experiment is running. Any non-zero loss counter should be treated as a quality warning.

Stopping a session first closes frame acceptance, then stops EVS while the frame writer drains in the background. `duration_seconds` therefore reports the actual capture interval; `writer_drain_seconds` separately reports post-capture save time. The GUI also reports measured save fps and MB/s. A manual **EVSを再接続** control is available; during recording, reconnection continues into a new EVS RAW segment.

The EVS view includes a live Trigger In monitor even when no recording is active. It shows recent activity, the reference-edge frequency and period, cumulative rising/falling edges, and the age and polarity of the latest edge. Frequency history resets after EVS reconnection so timestamps from different sensor epochs are never mixed.

The frame view has a corresponding external-trigger monitor. When external drive is enabled it shows the selected input line and activation edge, complete/incomplete trigger-driven frame callbacks, measured frame frequency, latest-frame age, and the `Line0 ExposureActive -> EVS Trigger In` synchronization state. Thus an external input starts the frame exposure, and that exposure simultaneously produces the timing edge recorded by EVS.

The frame camera is configured with a hardware ROI matching the EVS sensor's physical field of view (1800×1012 at the default sensor specifications, with the configured frame shift applied). This avoids transferring the unused border pixels. If the camera rejects hardware ROI control, the application restores full-sensor acquisition and automatically falls back to the same software crop. The actual mode, size, and sensor offset are stored in `camera_settings.json`, `frame_summary.json`, `/status`, and the frame audit CSV.

## Saved-data browser and synchronized playback

The **保存データ** tab reads completed and interrupted sessions from the configured recording directory. It shows recording duration, saved-frame and reference-edge counts, synchronization matches, ROI, loss counters, file sizes, and on-demand Bayer previews without modifying the source data.

Synchronized playback uses each `matched` record in the derived synchronization index. If that index is absent or stale, it is rebuilt from the primary audit records before playback; the legacy CSV is not required. Frame images retain their measured host-time spacing, while the paired EVS sensor timestamp selects a short window from the corresponding RAW stream epoch. EVS background pixels are transparent and the events are overlaid directly on the Bayer frame, with positive events in white and negative events in black. Controls provide overlay opacity, pause/resume, seeking, 0.25×–4× playback speed, and 10/33/100 ms EVS accumulation windows. At 1×, display updates may skip intermediate frames if decoding cannot keep up; the player preserves experimental time instead of accumulating playback delay.

Playback does not stop when frame-to-trigger correspondence is incomplete. It
uses paired triggers where their host-time offset remains consistent, falls back
to EVS sensor/host-time interpolation after a trigger-source change or missing
edge, and finally estimates position from the RAW segment start when no trigger
anchor is available. Each displayed frame reports which synchronization mode was
used, so approximate regions remain distinguishable from precisely paired ones.

In **EVS 1台データ確認**, `RAWファイルを開く` can play a standalone
Metavision `.raw` file without a bothViewer session directory or summary files.
The local native picker is used on macOS and Windows, so large RAW files are read
directly by the data service rather than uploaded through the browser. The first
open scans the file to determine its duration, event count, and sensor geometry;
subsequent time-window rendering uses the same playback controls and palettes as
session recordings.

Playback offers monochrome and negative-cyan/positive-magenta event palettes.
The main synchronized player is capped at 30 display fps even when the recorded
frame rate is higher; experimental time remains authoritative and intermediate
display frames are skipped. The full synchronized frame/event sequence remains
available below in 60-frame pages. Only the visible page and nearby thumbnails
are rendered, avoiding an unbounded RAW-decoding burst for long recordings.

EVS playback rendering has four load presets. **軽量** updates the overlay at up to 5 fps with 20,000 events per window, **標準** uses 10 fps and 50,000 events, **高密度** uses 15 fps and 100,000 events, and **全描画** updates on every saved frame without event thinning. Thinning is uniform and display-only: the RAW file and synchronization timeline are never modified, and frame playback continues at its measured real-time cadence.

EVS windows and Bayer previews are generated only when requested and cached in memory/browser cache. Source PGM and RAW files remain unchanged.

## Recording-priority live preview

JPEG preview generation is isolated from both camera callback paths. Each camera callback only replaces a single latest-preview slot; demosaicing, resize, flip, and JPEG encoding run in dedicated best-effort workers. Superseded preview frames and duplicate MJPEG sends are intentionally skipped, while Bayer/RAW recording and trigger auditing are never thinned by the preview manager.

The GUI provides three shared preview presets:

- **記録優先**: 10 fps, EVS 0.5×, Frame 0.45×, JPEG quality 70
- **標準**: 20 fps, EVS 0.75×, Frame 0.6×, JPEG quality 75
- **高画質**: 30 fps, EVS 1.0×, Frame 0.8×, JPEG quality 80

With automatic degradation enabled, sustained JPEG utilization or EVS decode lag above 100 ms lowers only the effective preview preset. After 30 seconds of stable low utilization it recovers one level toward the requested preset. `/status` and the GUI report requested/effective presets, actual/target preview fps, encode time, skipped preview sources, EVS event rate, decode lag, and automatic degradation/recovery counts. The selected upper limit is persisted under `preview` in `config.yaml`.

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

Connection initialization and retry are active only while the capture tab is visible.
The browser renews a short activity lease for both camera services; switching to
saved data, hiding or closing the page, or losing the browser connection lets the
lease expire and changes a disconnected camera to `retry_paused`. A healthy
camera connection is left running, and retry resumes immediately when the
capture tab becomes visible again.

## Performance

You can further reduce CPU usage by lowering the display scale factor. Pass
`--display-factor 0.3` (for example) when starting `frameStreamer.py` or
`evsStreamer.py` to downscale frames before encoding.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
