#!/usr/bin/env python3
"""Serve the Quoridor Q4 (four players) .sgfs viewer locally and open it on a given file.

Usage:
    python3 python/q4_sgfs_viewer/serve.py path/to/games.sgfs [--port 8765] [--host 127.0.0.1] [--no-browser]

To open it on a phone on the same network, bind to all interfaces (anyone on the LAN can then read the file):
    python3 python/q4_sgfs_viewer/serve.py games.sgfs --host 0.0.0.0
and open the printed "On your LAN" URL on the phone.

Without a file argument, the page opens empty and accepts drag-and-drop / "Open .sgfs...".
"""
import argparse
import http.server
import os
import socket
import urllib.parse
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))


def lan_ip():
    """Best-effort address of this machine on the local network (no packet is sent)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("10.255.255.255", 1))
            return sock.getsockname()[0]
    except OSError:
        return socket.gethostname()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help=".sgfs file (one game per line)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; 0.0.0.0 = reachable from the LAN)")
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
    query = ""
    if data_path:
        query = "?" + urllib.parse.urlencode({"src": "data.sgfs", "name": os.path.basename(data_path)})
    wildcard = args.host in ("0.0.0.0", "::", "")
    url = f"http://{'127.0.0.1' if wildcard else args.host}:{args.port}/{query}"
    print(f"Viewer at {url}  (Ctrl+C to stop)")
    if wildcard:
        print(f"On your LAN:  http://{lan_ip()}:{args.port}/{query}")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
