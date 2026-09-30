import os
import sys

import pytest

from quoridor_arena.qtp import QTPCrash, QTPEngine, QTPTimeout, ResponseReader, parse_response

MOCK = os.path.join(os.path.dirname(__file__), "mock_engine.py")


def test_parse_simple():
    assert parse_response(["= e8"]) == (True, "e8")
    assert parse_response(["="]) == (True, "")
    assert parse_response(["? illegal move"]) == (False, "illegal move")


def test_parse_with_id_and_noise():
    assert parse_response(["=12 e2h"]) == (True, "e2h")
    assert parse_response(["?3 bad"]) == (False, "bad")
    assert parse_response(["some diagnostic", "= ok"]) == (True, "ok")


def test_parse_multiline():
    ok, text = parse_response(["= MoveNum: 2", "  a b c", " 9 | . |"])
    assert ok and text.splitlines() == ["MoveNum: 2", "  a b c", " 9 | . |"]
    # A response whose first line is only the marker.
    ok, text = parse_response(["=", "line1", "line2"])
    assert ok and text == "line1\nline2"


def test_parse_no_marker():
    with pytest.raises(ValueError):
        parse_response(["hello"])


def test_reader_splits_responses():
    r = ResponseReader()
    out = []
    for line in ["junk before", "= a", "b", "", "", "? err", "", "=", ""]:
        done = r.feed(line)
        if done is not None:
            out.append(done)
    assert out == [["= a", "b"], ["? err"], ["="]]


def _engine(tmp_path, *flags, timeout=10):
    return QTPEngine("mock", [sys.executable, MOCK] + list(flags), stderr_path=str(tmp_path / "e.log"),
                     default_timeout=timeout)


def test_engine_roundtrip(tmp_path):
    e = _engine(tmp_path)
    e.start()
    try:
        assert e.send("play b e8") == (True, "")
        assert e.send("play b e7") == (False, "illegal move")
        ok, text = e.send("showboard")
        assert ok and text.count("\n") == 2
        assert e.known_command("legal_moves")
        assert not e.known_command("nonsense")
    finally:
        e.close()
    assert not e.alive()


def test_engine_crash_and_restart(tmp_path):
    e = _engine(tmp_path, "--crash-after", "1")
    e.start()
    with pytest.raises(QTPCrash):
        e.send("genmove b")
    assert not e.alive()
    e.ensure_running()
    assert e.restarts == 1
    assert e.send("clear_board") == (True, "")
    e.close()


def test_engine_timeout(tmp_path):
    e = _engine(tmp_path, "--hang-after", "1", timeout=1.0)
    e.start()
    with pytest.raises(QTPTimeout):
        e.send("genmove b")
    assert not e.alive()
    e.ensure_running()
    assert e.send("winner") == (True, "none")
    e.close()
