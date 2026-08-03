# bothViewer

bothViewer provides a web-based control console for simultaneously viewing and recording data from an event camera and a frame camera. It runs two local Flask servers and opens a responsive, dependency-free HTML interface for monitoring both streams and changing camera settings.

## Installation

Install the Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Camera access also requires the vendor SDKs for your hardware:

- Prophesee OpenEB / Metavision SDK (event camera)
- Allied Vision Vimba X Python API (`vmbpy`) (frame camera)

Install and verify those SDKs using the manufacturers' instructions before launching bothViewer. FFmpeg must also be available on your system when saving frame-camera recordings as MP4.

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

## Performance

You can further reduce CPU usage by lowering the display scale factor. Pass
`--display-factor 0.3` (for example) when starting `frameStreamer.py` or
`evsStreamer.py` to downscale frames before encoding.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
