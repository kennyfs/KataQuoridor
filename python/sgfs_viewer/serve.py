#!/usr/bin/env python3
"""Serve the Quoridor .sgfs viewer locally and open it on a given file.

Usage:
    python3 python/sgfs_viewer/serve.py path/to/games.sgfs [--port 8765] [--host 127.0.0.1] [--no-browser]

To open it on a phone on the same network, bind to all interfaces (anyone on the LAN can then read the file):
    python3 python/sgfs_viewer/serve.py games.sgfs --host 0.0.0.0
and open the printed "On your LAN" URL on the phone.

With --latest, it opens the newest self-play games of ~/q1_run/run4 (or --latest RUN_DIR):
    python3 python/sgfs_viewer/serve.py --latest

Without a file argument, the page opens empty and accepts drag-and-drop / "Open .sgfs...".
"""
import argparse
import glob
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


def latest_selfplay_sgfs(run_dir):
    """The most recently written .sgfs (by mtime) under run_dir/selfplay/<model>/sgfs, the random-net data excluded.
    The newest games are those of the model being played, whatever its name (the sample counts in the names of nets
    of different sizes are not comparable)."""
    files = [f for d in glob.glob(os.path.join(run_dir, "selfplay", "*")) if os.path.isdir(d) and os.path.basename(d) != "random"
             for f in glob.glob(os.path.join(d, "sgfs", "*.sgfs"))]
    return max(files, key=os.path.getmtime) if files else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", help=".sgfs file (one game per line)")
    ap.add_argument("--latest", nargs="?", const="~/q1_run/run4", metavar="RUN_DIR",
                    help="open the newest self-play .sgfs of the newest model in RUN_DIR (default ~/q1_run/run4)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1", help="bind address (default 127.0.0.1; 0.0.0.0 = reachable from the LAN)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()

    if args.latest and args.file:
        ap.error("give either a file or --latest, not both")
    run_dir = os.path.expanduser(args.latest) if args.latest else None
    if run_dir:
        found = latest_selfplay_sgfs(run_dir)
        if not found:
            ap.error(f"no self-play .sgfs under {run_dir}/selfplay/<model>/sgfs")
        print(f"Latest self-play games now: {found} (looked up again on every page load)")
    data_path = os.path.abspath(args.file) if args.file else None
    if data_path and not os.path.isfile(data_path):
        ap.error(f"not a file: {data_path}")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=HERE, **kw)

        def do_GET(self):
            if (data_path or run_dir) and urllib.parse.urlparse(self.path).path == "/data.sgfs":
                # --latest: the newest games at the time of each request, so the same URL follows new models
                path = (latest_selfplay_sgfs(run_dir) if run_dir else None) or data_path
                if not path:
                    self.send_error(404, "no self-play .sgfs found")
                    return
                with open(path, "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                # model dir / file name, shown by the page instead of the URL's name
                self.send_header("X-Sgfs-Name", urllib.parse.quote(os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.dirname(path))))))
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def end_headers(self):
            # revalidate every file, so an updated viewer (js / css) shows without a hard refresh
            self.send_header("Cache-Control", "no-cache")
            super().end_headers()

        def log_message(self, *a):
            pass

    server = http.server.ThreadingHTTPServer((args.host, args.port), Handler)
    query = ""
    if run_dir:
        query = "?src=data.sgfs"
    elif data_path:
        query = "?" + urllib.parse.urlencode({"src": "data.sgfs", "name": os.path.basename(data_path)})
    wildcard = args.host in ("0.0.0.0", "::", "")
    url = f"http://{'127.0.0.1' if wildcard else args.host}:{args.port}/{query}"
    if wildcard:
        print(f"Viewer on your LAN at http://{lan_ip()}:{args.port}/{query}  (Ctrl+C to stop)")
    else:
        print(f"Viewer at {url}  (Ctrl+C to stop)")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
