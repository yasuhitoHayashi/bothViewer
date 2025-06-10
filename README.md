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

`frameStreamer.py` encodes each captured frame to JPEG only once and streams the
cached bytes to clients. This reduces repetitive conversions and lowers CPU
usage, which is especially helpful on resource constrained devices such as the
Raspberry Pi.

## License

This project is licensed under the [Apache License 2.0](LICENSE).
