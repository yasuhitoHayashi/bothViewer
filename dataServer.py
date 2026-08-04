"""Read-only recording browser server."""

import argparse

from bothviewer.api.data import create_data_app


def main():
    parser = argparse.ArgumentParser(description="bothViewer recording data server")
    parser.add_argument("--save_location", required=True)
    parser.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()
    create_data_app(args.save_location).run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
