"""Compatibility entry point for the frame-camera service."""

if __name__ == "__main__":
    import runpy
    runpy.run_module("bothviewer.cameras.frame", run_name="__main__")
else:
    from bothviewer.cameras.frame import *  # noqa: F401,F403
