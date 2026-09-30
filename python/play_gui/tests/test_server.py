"""HTTP API tests: a real server on a free port with the mock engine behind it."""
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

from play_gui.game import GameController
from play_gui.serve import katago_argv, make_server

MOCK = os.path.join(os.path.dirname(__file__), "..", "..", "quoridor_arena", "tests", "mock_engine.py")


@pytest.fixture
def api(tmp_path):
    game = GameController([sys.executable, MOCK, "--think-delay", "0.3"], str(tmp_path), model_name="mocknet",
                          move_timeout=20, start_timeout=20, hint_visits=50)
    game.start_engine()
    server = make_server(game, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = "http://127.0.0.1:%d" % server.server_address[1]

    def call(path, body=None, ctype="application/json"):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(base + path, data=data, method="GET" if body is None else "POST")
        if body is not None:
            req.add_header("Content-Type", ctype)
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if "json" in r.headers["Content-Type"] else raw.decode()), r.headers
        except urllib.error.HTTPError as e:
            raw = e.read()
            return e.code, (json.loads(raw) if "json" in e.headers["Content-Type"] else raw.decode()), e.headers

    call.game = game
    yield call
    server.shutdown()
    server.server_close()
    game.close()


def wait_idle(call, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        _, s, _ = call("/api/state")
        if not s["thinking"] and not s["busy"]:
            return s
        time.sleep(0.05)
    raise AssertionError("still thinking")


def test_static_files(api):
    code, body, headers = api("/")
    assert code == 200 and "<svg" in body and headers["Content-Type"].startswith("text/html")
    code, body, _ = api("/style.css")
    assert code == 200
    code, body, headers = api("/js/board.js")
    assert code == 200 and "javascript" in headers["Content-Type"]
    code, _, _ = api("/serve.py")
    assert code == 404
    code, _, _ = api("/js/../serve.py")
    assert code == 404


def test_full_flow(api):
    code, s, _ = api("/api/new", {"human": "b", "visits": 16})
    assert code == 200 and s["to_move"] == "b" and s["started"]
    code, s, _ = api("/api/move", {"move": "e8"})
    assert code == 200 and s["thinking"]
    # Out of turn / while thinking.
    code, err, _ = api("/api/move", {"move": "e7"})
    assert code == 409 and "thinking" in err["error"]
    code, err, _ = api("/api/undo", {})
    assert code == 409
    s = wait_idle(api)
    assert [m["move"] for m in s["moves"]] == ["e8", "e2"]
    code, hint, _ = api("/api/hint", {})
    assert code == 200 and hint["moves"][0]["move"] == "e7"
    code, err, _ = api("/api/move", {"move": "e5"})
    assert code == 400 and "illegal" in err["error"]
    code, err, _ = api("/api/move", {"move": "e7; quit"})
    assert code == 400
    code, sgf, headers = api("/api/sgf")
    assert code == 200 and sgf.startswith("(;FF[4]") and sgf.endswith(")\n") and sgf.count("\n") == 1
    assert ".sgfs" in headers["Content-Disposition"]
    code, s, _ = api("/api/undo", {})
    assert code == 200 and s["moves"] == []


def test_new_game_as_white(api):
    code, s, _ = api("/api/new", {"human": "w", "visits": 1})
    assert code == 200 and s["thinking"]
    s = wait_idle(api)
    assert s["settings"]["human"] == "w" and len(s["moves"]) == 1 and s["to_move"] == "w"


def test_rejects_non_json_posts(api):
    code, err, _ = api("/api/new", {"human": "b"}, ctype="text/plain")
    assert code == 415
    code, err, _ = api("/api/nope", {})
    assert code == 404


def test_restart_endpoint(api):
    api("/api/new", {"human": "b", "visits": 16})
    api.game.engine.kill()                   # simulate a crash
    code, err, _ = api("/api/move", {"move": "e8"})
    assert code == 503
    _, s, _ = api("/api/state")
    assert s["engine"]["status"] == "error"
    code, _, _ = api("/api/restart", {})
    assert code == 200
    t0 = time.time()
    while api("/api/state")[1]["engine"]["status"] != "ready":
        assert time.time() - t0 < 10
        time.sleep(0.05)
    code, s, _ = api("/api/move", {"move": "e8"})
    assert code == 200


def test_katago_argv():
    argv = katago_argv("/k", "/m.bin.gz", "/c.cfg", "/tmp/x", "numSearchThreads=2")
    assert argv[:7] == ["/k", "gtp", "-model", "/m.bin.gz", "-config", "/c.cfg", "-override-config"]
    ov = argv[7].split(",")
    for kv in ("ponderingEnabled=false", "allowResignation=false", "logAllGTPCommunication=false",
               "reportAnalysisWinratesAs=SIDETOMOVE", "logDir=/tmp/x/gtp_logs", "numSearchThreads=2"):
        assert kv in ov
