#!/usr/bin/env python3
"""Play and analyse Four-at-a-Table (Q4) positions in the browser, with a Q4 net.

Usage:
    python3 python/q4_play_gui/serve.py --katago cpp/build/katago --model model.bin.gz \\
        [--config cpp/configs/q4/q4qtp.cfg] [--threads 2] [--port 8767] [--no-browser]

Starts one KataQuoridor q4qtp process and a local web server (127.0.0.1 by default), then opens the page.
Python 3 standard library only. See docs/q4/Q4PlayGUI.md.
"""
import argparse
import datetime
import http.server
import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import urllib.parse
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(PYTHON_DIR)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from q4_play_gui.game import KEEP, ActionError, EngineDown, Q4Game  # noqa: E402

STATIC = {"/": "index.html", "/index.html": "index.html", "/style.css": "style.css"}
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}
OWNER_TIMEOUT = 15.0   # seconds a tab keeps the game after its last request


class Owner:
    """One browser tab at a time: the first tab that asks owns the game until it has been quiet for a while."""

    def __init__(self):
        self.lock = threading.Lock()
        self.client = None
        self.seen = 0.0

    def claim(self, client):
        with self.lock:
            now = time.monotonic()
            if self.client is None or client == self.client or now - self.seen > OWNER_TIMEOUT:
                self.client = client
                self.seen = now
                return True
            return False


def make_handler(game, owner):
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "KataQuoridorQ4Play/1"

        def log_message(self, *a):
            pass

        def _send(self, status, body, ctype="application/json; charset=utf-8", extra_headers=None):
            if not isinstance(body, bytes):
                body = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra_headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status, obj):
            self._send(status, json.dumps(obj))

        def _static(self, path):
            rel = STATIC.get(path)
            if rel is None and path.startswith("/js/") and path.endswith(".js") and "/" not in path[4:]:
                rel = path[1:]
            if rel is None:
                return self._send(404, "not found", "text/plain; charset=utf-8")
            try:
                with open(os.path.join(HERE, rel), "rb") as f:
                    body = f.read()
            except OSError:
                return self._send(404, "not found", "text/plain; charset=utf-8")
            self._send(200, body, CONTENT_TYPES.get(os.path.splitext(rel)[1], "application/octet-stream"))

        def _claimed(self):
            client = self.headers.get("X-Client", "")
            if not re.match(r"^[A-Za-z0-9_-]{4,64}$", client):
                self._json(400, {"error": "missing X-Client header"})
                return False
            if not owner.claim(client):
                self._json(423, {"error": "another browser tab controls this game; close it or reload this one"})
                return False
            return True

        def _run(self, fn):
            try:
                self._json(200, fn())
            except ActionError as e:
                self._json(e.status, {"error": str(e)})
            except EngineDown as e:
                self._json(503, {"error": str(e), "engine_down": True})

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            path = parsed.path
            if path.startswith("/api/"):
                if not self._claimed():
                    return
            if path == "/api/state":
                return self._json(200, game.state())
            if path == "/api/params":
                return self._run(lambda: {"text": game.search_params()})
            if path == "/api/legal":
                q = urllib.parse.parse_qs(parsed.query)
                try:
                    seat = int(q.get("seat", ["-1"])[0])
                except ValueError:
                    return self._json(400, {"error": "seat must be 0..3"})
                return self._run(lambda: {"seat": seat, "legal": game.legal_for(seat)})
            if path == "/api/sgf":
                try:
                    sgf = game.sgf().replace("\r", "").replace("\n", "")
                except (ActionError, EngineDown) as e:
                    return self._json(getattr(e, "status", 503), {"error": str(e)})
                name = "kataquoridor-q4-%s.sgfs" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                return self._send(200, sgf + "\n", "application/x-go-sgf; charset=utf-8",
                                  {"Content-Disposition": 'attachment; filename="%s"' % name})
            return self._static(path)

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            # JSON bodies only: a cross-site form cannot send application/json without a CORS preflight, which this
            # server never approves.
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._json(415, {"error": "expected application/json"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as e:
                return self._json(400, {"error": "bad JSON: %s" % e})
            if not self._claimed():
                return
            if path == "/api/new":
                return self._run(lambda: game.new_game(body.get("rules"), body.get("seats")))
            if path == "/api/seats":
                me = body["me"] if "me" in body else KEEP
                return self._run(lambda: game.set_seats(body.get("seats", []), me=me))
            if path == "/api/play":
                return self._run(lambda: game.play(str(body.get("move", ""))))
            if path == "/api/eliminate":
                return self._run(lambda: game.eliminate(_int(body.get("seat"), -1), bool(body.get("confirm"))))
            if path == "/api/undo":
                return self._run(game.undo)
            if path == "/api/redo":
                return self._run(game.redo)
            if path == "/api/goto":
                return self._run(lambda: game.goto(body.get("ply")))
            if path == "/api/ai":
                return self._run(lambda: game.ai_move(bool(body.get("all"))))
            if path == "/api/settings":
                return self._run(lambda: game.set_settings(body))
            if path == "/api/analysis":
                return self._run(lambda: game.set_analysis(body.get("on"), body.get("cs"), body.get("cap")))
            if path == "/api/load":
                return self._run(lambda: game.load(str(body.get("text", "")), body.get("ply")))
            if path == "/api/restart":
                return self._run(_restart_now(game))
            return self._json(404, {"error": "unknown endpoint"})

    return Handler



def _int(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _restart_now(game):
    def run():
        game.restart()
        return {"ok": True}
    return run


def config_with_threads(config, threads, out_dir):
    """A copy of the QTP config with numSearchThreads set (the GUI runs few threads next to a training loop)."""
    with open(config) as f:
        lines = f.read().splitlines()
    out = []
    seen = False
    for line in lines:
        if re.match(r"^\s*numSearchThreads\s*=", line):
            out.append("numSearchThreads = %d" % threads)
            seen = True
        else:
            out.append(line)
    if not seen:
        out.append("numSearchThreads = %d" % threads)
    path = os.path.join(out_dir, "q4qtp.cfg")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")
    return path


def qtp_argv(katago, model, config):
    return [katago, "q4qtp", "-config", config, "-model", model, "-bot", "search"]


def make_server(game, host="127.0.0.1", port=8767):
    return http.server.ThreadingHTTPServer((host, port), make_handler(game, Owner()))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--katago", required=True, help="path to a KataQuoridor katago binary (CUDA or Eigen)")
    ap.add_argument("--model", required=True, help="Q4 net (.bin.gz)")
    ap.add_argument("--config", default=os.path.join(REPO, "cpp", "configs", "q4", "q4qtp.cfg"),
                    help="QTP config (default: cpp/configs/q4/q4qtp.cfg)")
    ap.add_argument("--threads", type=int, default=2, help="search threads of the engine (default 2)")
    ap.add_argument("--visits", type=int, default=400, help="default visits of an AI net seat (default 400)")
    ap.add_argument("--interval-cs", type=int, default=25, help="analysis report interval in centiseconds (default 25)")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--host", default="127.0.0.1", help="listen address (default 127.0.0.1: this machine only)")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--move-timeout", type=float, default=600.0, help="seconds before a search counts as hung")
    ap.add_argument("--log-dir", help="keep engine logs here (default: a temporary directory, removed at exit)")
    args = ap.parse_args(argv)

    for what, path in (("katago binary", args.katago), ("model", args.model), ("config", args.config)):
        if not os.path.isfile(path):
            ap.error("%s not found: %s" % (what, path))
    if not os.access(args.katago, os.X_OK):
        ap.error("katago binary is not executable: %s" % args.katago)
    if args.threads < 1 or args.visits < 1 or args.interval_cs < 1:
        ap.error("--threads, --visits and --interval-cs must be at least 1")

    log_dir = os.path.abspath(args.log_dir) if args.log_dir else tempfile.mkdtemp(prefix="kq_q4_play_")
    os.makedirs(log_dir, exist_ok=True)
    config = config_with_threads(os.path.abspath(args.config), args.threads, log_dir)
    game = Q4Game(qtp_argv(os.path.abspath(args.katago), os.path.abspath(args.model), config), log_dir,
                  move_timeout=args.move_timeout, default_visits=args.visits, interval_cs=args.interval_cs)
    game.analysis["cs"] = args.interval_cs

    try:
        server = make_server(game, args.host, args.port)
    except OSError as e:
        ap.error("cannot listen on %s:%d: %s (try --port)" % (args.host, args.port, e))
    threading.Thread(target=_start_quietly, args=(game,), daemon=True).start()
    url = "http://%s:%d/" % (args.host, server.server_address[1])
    print("KataQuoridor Q4 play GUI at %s  (Ctrl+C to stop)" % url)
    print("Engine logs: %s" % log_dir)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        game.close()
        if not args.log_dir:
            shutil.rmtree(log_dir, ignore_errors=True)


def _start_quietly(game):
    try:
        game.start_engine()
    except EngineDown:
        pass   # reported through /api/state


if __name__ == "__main__":
    main()
