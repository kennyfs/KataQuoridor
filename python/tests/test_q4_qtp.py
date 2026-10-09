"""QTP protocol and rule enforcement tests for `katago q4qtp` (Plan §5.7 T10).

Tests scripted sessions:
- out-of-turn play
- illegal move
- game over
- elimination
- undo across an elimination
- rules setting
- record round trip (`printrecord` then `loadrecord` reproduces position)
- genmove
"""

import glob
import json
import os
import subprocess
import tempfile
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def find_katago_bin():
    if os.environ.get("KATAGO_BIN") and os.path.isfile(os.environ["KATAGO_BIN"]):
        return os.environ["KATAGO_BIN"]
    candidates = [
        os.path.join(REPO_ROOT, "cpp", "build-eigen", "katago"),
        os.path.join(REPO_ROOT, "cpp", "build-eigen-release", "katago"),
        os.path.join(REPO_ROOT, "build", "katago"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise RuntimeError("KataGo binary not found! Please build it first.")


class Q4QTPEngine:
    def __init__(self, bot="greedy"):
        self.p = subprocess.Popen(
            [find_katago_bin(), "q4qtp", "-bot", bot],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=REPO_ROOT,
        )

    def send(self, cmd: str):
        self.p.stdin.write(cmd + "\n")
        self.p.stdin.flush()
        lines = []
        while True:
            line = self.p.stdout.readline()
            assert line != "", f"Engine exited during {cmd}"
            line = line.rstrip("\n")
            if line == "" and lines:
                break
            if line != "" or lines:
                lines.append(line)
        header = lines[0]
        ok = header.startswith("=")
        first_line = header[2:].strip()
        rest = lines[1:]
        text = "\n".join([first_line] + rest).strip() if rest else first_line
        return ok, text

    def close(self):
        try:
            self.send("quit")
        except:
            pass
        self.p.terminate()
        self.p.wait()


def test_q4qtp_basic_and_out_of_turn():
    engine = Q4QTPEngine()
    try:
        ok, res = engine.send("name")
        assert ok and "KataQuoridor Q4" in res

        ok, res = engine.send("protocol_version")
        assert ok and res == "2"

        ok, res = engine.send("to_move")
        assert ok and res == "1"

        # Out-of-turn play: Seat 2 tries to move first -> must fail!
        ok, res = engine.send("play 2 a7")
        assert not ok
        # Turn must still be seat 1
        ok, res = engine.send("to_move")
        assert ok and res == "1"

        # Illegal move: moving through edge or jumping to invalid square
        ok, res = engine.send("play 1 f3")  # Can only move to f2, e1, g1 from f1
        assert not ok

        # Legal move: Seat 1 plays f2
        ok, res = engine.send("play 1 f2")
        assert ok
        ok, res = engine.send("to_move")
        assert ok and res == "2"  # Turn advances clockwise to seat 2
    finally:
        engine.close()


def test_q4qtp_elimination_and_undo():
    engine = Q4QTPEngine()
    try:
        # Seat 1 plays f2
        assert engine.send("play 1 f2")[0]
        # Now seat 2 is to move
        assert engine.send("to_move")[1] == "2"

        # Eliminate seat 2 (the seat to move)
        ok, _ = engine.send("eliminate 2")
        assert ok
        # Turn must immediately pass clockwise to seat 3!
        assert engine.send("to_move")[1] == "3"

        # Undo across elimination
        ok, _ = engine.send("undo")
        assert ok
        # Seat 2 should be alive and to move again!
        assert engine.send("to_move")[1] == "2"
        # Alive array shows seat 2 alive
        ok, board_str = engine.send("showboard")
        assert ok and "P2" in board_str
    finally:
        engine.close()


def test_q4qtp_rules_setting_and_max_plies():
    engine = Q4QTPEngine()
    try:
        # Set maxPlies to 4 before first move
        ok, _ = engine.send("set_rule maxPlies 4")
        assert ok

        # Check rules
        ok, rules_str = engine.send("get_rules")
        assert ok and "maxPlies=4" in rules_str

        # Play 3 plies
        assert engine.send("play 1 f2")[0]  # ply 1
        assert engine.send("play 2 b6")[0]  # ply 2
        assert engine.send("play 3 f10")[0] # ply 3

        # Setting rules after first move must be rejected
        ok, _ = engine.send("set_rule maxPlies 10")
        assert not ok

        # Ply 4: non-winning move -> Draw
        assert engine.send("play 4 j6")[0]
        ok, res = engine.send("winner")
        assert ok and res == "Draw"

        # Further moves rejected
        ok, _ = engine.send("play 1 f3")
        assert not ok
    finally:
        engine.close()


def test_q4qtp_record_round_trip():
    engine = Q4QTPEngine()
    try:
        # Play a sequence of moves and an elimination
        assert engine.send("play 1 f2")[0]
        assert engine.send("play 2 b6")[0]
        assert engine.send("eliminate 3")[0]
        assert engine.send("play 4 e5h")[0]  # Seat 4 places a wall

        # Save record
        ok, rec_line = engine.send("printrecord")
        assert ok and rec_line.startswith("(;FF[4]GM[Q4]")
        assert "e5h" in rec_line

        # Read distances and walls
        _, walls_before = engine.send("walls")
        _, dist_before = engine.send("dist")
        _, to_move_before = engine.send("to_move")

        # Save to temporary file
        with tempfile.NamedTemporaryFile("w+", suffix=".sgfs", delete=False) as tmp:
            tmp.write(rec_line + "\n")
            tmp_path = tmp.name

        try:
            # Clear board
            assert engine.send("clear_board")[0]
            assert engine.send("to_move")[1] == "1"

            # Load record
            ok, _ = engine.send(f"loadrecord {tmp_path} 1")
            assert ok

            # Verify identical state
            _, walls_after = engine.send("walls")
            _, dist_after = engine.send("dist")
            _, to_move_after = engine.send("to_move")

            assert walls_before == walls_after
            assert dist_before == dist_after
            assert to_move_before == to_move_after
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    finally:
        engine.close()


def test_q4qtp_genmove():
    engine = Q4QTPEngine(bot="greedy")
    try:
        # Seat 1 genmove
        ok, act = engine.send("genmove 1")
        assert ok
        # greedy from f1 moves towards center (5, 5) -> must be f2!
        assert act == "f2"
        # Turn advances to seat 2
        assert engine.send("to_move")[1] == "2"

        # Seat 2 genmove -> towards center from a6 is b6!
        ok, act2 = engine.send("genmove 2")
        assert ok
        assert act2 == "b6"
    finally:
        engine.close()
