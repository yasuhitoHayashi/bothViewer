"""Compatibility entry point for the EVS service."""

if __name__ == "__main__":
    import runpy
    runpy.run_module("bothviewer.cameras.evs", run_name="__main__")
else:
    from bothviewer.cameras.evs import *  # noqa: F401,F403
