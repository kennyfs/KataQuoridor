#!/usr/bin/env python3
"""Play Quoridor against KataQuoridor in the browser.

Usage:
    python3 python/play_gui/serve.py --katago cpp/build-cuda/katago --model model.bin.gz \\
        [--config cpp/configs/gtp_quoridor.cfg] [--port 8766] [--no-browser]

Starts one KataQuoridor QTP process and a local web server (127.0.0.1 by default), then opens the page.
Python 3 standard library only. See docs/PlayGUI.md.
"""
import argparse
import datetime
import http.server
import json
import os
import shutil
import sys
import tempfile
import threading
import urllib.parse
import webbrowser

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(HERE)
REPO = os.path.dirname(PYTHON_DIR)
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from play_gui.game import ActionError, EngineDown, GameController  # noqa: E402

ENGINE_OVERRIDES = ("logAllGTPCommunication=false,logSearchInfo=false,ponderingEnabled=false,"
                    "allowResignation=false,reportAnalysisWinratesAs=SIDETOMOVE")
STATIC = {"/": "index.html", "/index.html": "index.html", "/style.css": "style.css"}
CONTENT_TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8", ".svg": "image/svg+xml"}


def katago_argv(katago, model, config, log_dir, extra_overrides=""):
    overrides = ENGINE_OVERRIDES + ",logDir=" + os.path.join(log_dir, "gtp_logs")
    if extra_overrides:
        overrides += "," + extra_overrides
    return [katago, "gtp", "-model", model, "-config", config, "-override-config", overrides]


def make_handler(game):
    class Handler(http.server.BaseHTTPRequestHandler):
        server_version = "KataQuoridorPlay/1"

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
            full = os.path.join(HERE, rel)
            try:
                with open(full, "rb") as f:
                    body = f.read()
            except OSError:
                return self._send(404, "not found", "text/plain; charset=utf-8")
            self._send(200, body, CONTENT_TYPES.get(os.path.splitext(rel)[1], "application/octet-stream"))

        def _run(self, fn):
            try:
                self._json(200, fn())
            except ActionError as e:
                self._json(e.status, {"error": str(e)})
            except EngineDown as e:
                self._json(503, {"error": str(e)})

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path == "/api/state":
                return self._json(200, game.state())
            if path == "/api/sgf":
                try:
                    sgf = game.sgf().replace("\r", "").replace("\n", "")
                except (ActionError, EngineDown) as e:
                    return self._json(getattr(e, "status", 503), {"error": str(e)})
                name = "kataquoridor-%s.sgfs" % datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                return self._send(200, sgf + "\n", "application/x-go-sgf; charset=utf-8",
                                  {"Content-Disposition": 'attachment; filename="%s"' % name})
            return self._static(path)

        def do_POST(self):
            path = urllib.parse.urlparse(self.path).path
            # Only JSON bodies: a cross-site form or no-cors fetch cannot send application/json without a
            # CORS preflight, which this server never approves.
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                return self._json(415, {"error": "expected application/json"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}") if n else {}
                if not isinstance(body, dict):
                    raise ValueError("expected a JSON object")
            except ValueError as e:
                return self._json(400, {"error": "bad JSON: %s" % e})
            if path == "/api/new":
                return self._run(lambda: game.new_game(body.get("human", "b"), body.get("visits", 16)))
            if path == "/api/move":
                return self._run(lambda: game.play(str(body.get("move", ""))))
            if path == "/api/undo":
                return self._run(game.undo)
            if path == "/api/hint":
                return self._run(game.request_hint)
            if path == "/api/restart":
                threading.Thread(target=_start_quietly, args=(game,), daemon=True).start()
                return self._json(200, {"ok": True})
            return self._json(404, {"error": "unknown endpoint"})

    return Handler


def _start_quietly(game):
    try:
        game.start_engine()
    except EngineDown:
        pass  # reported through /api/state


def make_server(game, host="127.0.0.1", port=8766):
    return http.server.ThreadingHTTPServer((host, port), make_handler(game))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--katago", required=True, help="path to a KataQuoridor katago binary (CUDA or Eigen)")
    ap.add_argument("--model", required=True, help="net file (.bin.gz)")
    ap.add_argument("--config", default=os.path.join(REPO, "cpp", "configs", "gtp_quoridor.cfg"),
                    help="QTP config (default: cpp/configs/gtp_quoridor.cfg)")
    ap.add_argument("--override-config", default="",
                    help="extra katago overrides, e.g. numSearchThreads=2 (appended to the GUI's own)")
    ap.add_argument("--port", type=int, default=8766)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--hint-visits", type=int, default=400, help="visits for a hint search (default 400)")
    ap.add_argument("--move-timeout", type=float, default=600.0, help="seconds before a search counts as hung")
    ap.add_argument("--demo-delay", type=float, default=1.0, help="seconds between moves in AI vs AI mode")
    ap.add_argument("--log-dir", help="keep engine logs here (default: a temporary directory, removed at exit)")
    args = ap.parse_args(argv)

    for what, path in (("katago binary", args.katago), ("model", args.model), ("config", args.config)):
        if not os.path.isfile(path):
            ap.error("%s not found: %s" % (what, path))
    if not os.access(args.katago, os.X_OK):
        ap.error("katago binary is not executable: %s" % args.katago)

    log_dir = os.path.abspath(args.log_dir) if args.log_dir else tempfile.mkdtemp(prefix="kq_play_")
    os.makedirs(log_dir, exist_ok=True)
    model_name = os.path.basename(os.path.dirname(os.path.abspath(args.model))) \
        if os.path.basename(args.model) == "model.bin.gz" else os.path.basename(args.model)
    game = GameController(
        katago_argv(os.path.abspath(args.katago), os.path.abspath(args.model), os.path.abspath(args.config),
                    log_dir, args.override_config),
        log_dir, model_name=model_name, move_timeout=args.move_timeout, hint_visits=args.hint_visits,
        demo_delay=args.demo_delay)

    try:
        server = make_server(game, args.host, args.port)
    except OSError as e:
        ap.error("cannot listen on %s:%d: %s (try --port)" % (args.host, args.port, e))
    threading.Thread(target=_start_quietly, args=(game,), daemon=True).start()
    url = "http://%s:%d/" % (args.host, server.server_address[1])
    print("KataQuoridor play GUI at %s  (Ctrl+C to stop)" % url)
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


if __name__ == "__main__":
    main()
