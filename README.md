# bothViewer

bothViewer provides a web based interface for simultaneously viewing data from an event camera and a frame camera. The project runs two Flask servers and serves a simple HTML interface to display the streams.

## Installation

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Usage

Launch both servers and open the viewer with:

```bash
python launcher.py
```

The event stream server starts on port `5001` and the frame stream server starts on port `5002`. After a short delay the `bothViewer.html` page is opened automatically in your default web browser.

## Performance

Both streamers encode frames to JPEG only once and then share the cached
bytes with connected clients. This avoids repeated conversions and lowers CPU
usage, which is especially helpful on resource constrained devices such as the
Raspberry Pi.


### Pixel Format

The frame camera's pixel format can be configured in `config.yaml` under
`frameCam.pixel_format`. The default is `RGB8`, but `Mono8` or
`YCbCr411_8_CbYYCrYY` can be selected for higher frame rates. You can also
specify the format when starting `frameStreamer.py`:

```bash
python frameStreamer.py --pixel-format Mono8
```

During runtime the format may be changed via the `/set_pixel_format` API:

```bash
curl -X POST -H "Content-Type: application/json" \
     -d '{"format":"YCbCr411_8_CbYYCrYY"}' http://localhost:5002/set_pixel_format
```

You can further reduce CPU usage by lowering the display scale factor. Pass
`--display-factor 0.3` (for example) when starting `frameStreamer.py` or
`evsStreamer.py` to downscale frames before encoding.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
