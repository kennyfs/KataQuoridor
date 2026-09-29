#!/usr/bin/env python3
"""Serve the Quoridor .sgfs viewer locally and open it on a given file.

Usage:
    python3 python/sgfs_viewer/serve.py path/to/games.sgfs [--port 8765] [--no-browser]

Without a file argument, the page opens empty and accepts drag-and-drop / "Open .sgfs...".
"""
import argparse
import http.server
import os
import urllib.parse
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help=".sgfs file (one game per line)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    data_path = os.path.abspath(args.file) if args.file else None
    if data_path and not os.path.isfile(data_path):
        ap.error(f"not a file: {data_path}")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=HERE, **kw)

        def do_GET(self):
            if data_path and urllib.parse.urlparse(self.path).path == "/data.sgfs":
                with open(data_path, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}/"
    if data_path:
        url += "?" + urllib.parse.urlencode({"src": "data.sgfs", "name": os.path.basename(data_path)})
    print(f"Viewer at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
