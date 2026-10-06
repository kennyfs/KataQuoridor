"""Tests for Q4 Search (Round 3 Plan §8.8-§8.9 T28).

Covers:
- QTP bot=search commands (q4-search-params, q4-analyze, genmove, clear on eliminate/clear_board/undo)
- q4match seat rotation with search player kind
- T28 play match: 100 games search:b2c64@200 vs 3 greedy, 100 games vs 3 basher
"""

import json
import os
import subprocess
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def find_katago_bin():
    if os.environ.get("KATAGO_BIN") and os.path.isfile(os.environ["KATAGO_BIN"]):
        return os.environ["KATAGO_BIN"]
    candidates = [
        os.path.join(REPO_ROOT, "cpp", "build-eigen-release", "katago"),
        os.path.join(REPO_ROOT, "cpp", "build-cuda", "katago"),
        os.path.join(REPO_ROOT, "cpp", "build-eigen", "katago"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise RuntimeError("KataGo binary not found! Please build it first.")


def get_model_path():
    candidates = [
        os.path.join(REPO_ROOT, "cpp", "tests", "models", "b2c64_q4", "model.bin.gz"),
        os.path.join(REPO_ROOT, "tests", "models", "b2c64_q4", "model.bin.gz"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    # Export if needed
    cmd = [
        "/home/kenny/ml_venv/bin/python",
        os.path.join(REPO_ROOT, "python", "q4", "make_random_model.py"),
        "b2c64_q4",
        os.path.join(REPO_ROOT, "cpp", "tests", "models"),
        "--model-name",
        "b2c64_q4",
        "--scale-heads",
    ]
    subprocess.check_call(cmd, cwd=REPO_ROOT)
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise RuntimeError("Could not find or create b2c64_q4 model")


class Q4SearchQTPEngine:
    def __init__(self, model_path, config_path=None):
        cmd = [
            find_katago_bin(),
            "q4qtp",
            "-bot",
            "search",
            "-model",
            model_path,
        ]
        if config_path:
            cmd.extend(["-config", config_path])
        self.p = subprocess.Popen(
            cmd,
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
        except Exception:
            pass
        self.p.terminate()
        self.p.wait()


def test_q4qtp_search_commands():
    model_path = get_model_path()
    cfg_path = os.path.join(REPO_ROOT, "cpp", "configs", "q4", "q4search_test.cfg")
    engine = Q4SearchQTPEngine(model_path, cfg_path)
    try:
        ok, text = engine.send("name")
        assert ok
        assert "KataQuoridor Q4" in text

        # 1. q4-search-params
        ok, text = engine.send("q4-search-params")
        assert ok
        assert "maxVisits" in text
        assert "winLossUtilityFactor" in text

        # 2. q4-analyze
        ok, text = engine.send("q4-analyze 20")
        assert ok
        assert "rootValues:" in text
        assert "rootUtilities:" in text
        assert "visits" in text
        assert "pv" in text

        # 3. genmove
        ok, move = engine.send("genmove 1")
        assert ok
        assert len(move) >= 2  # e.g. f2 or e6v

        # 4. play moves
        ok, _ = engine.send("play 2 a7")
        assert ok
        ok, _ = engine.send("play 3 f10")
        assert ok
        ok, _ = engine.send("play 4 k7")
        assert ok

        # 5. genmove again
        ok, move2 = engine.send("genmove 1")
        assert ok
        assert len(move2) >= 2

        # 6. undo
        ok, _ = engine.send("undo")
        assert ok

        # 7. eliminate
        ok, _ = engine.send("eliminate 4")
        assert ok

        # 8. clear_board
        ok, _ = engine.send("clear_board")
        assert ok
    finally:
        engine.close()


def test_q4match_rotation():
    model_path = get_model_path()
    cfg_path = os.path.join(REPO_ROOT, "cpp", "configs", "q4", "q4search_test.cfg")
    cmd = [
        find_katago_bin(),
        "q4match",
        "-config",
        cfg_path,
        "-players",
        f"search:{model_path}@30,greedy,greedy,greedy",
        "-games",
        "4",
        "-rotate",
        "-maxplies",
        "60",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert res.returncode == 0, f"q4match failed: {res.stderr}"
    assert "Played 4/4 games..." in res.stdout
    assert "Seat rotation: enabled" in res.stdout


def test_t28_play_matches():
    model_path = get_model_path()
    cfg_path = os.path.join(REPO_ROOT, "cpp", "configs", "q4", "q4search_test.cfg")

    # Match 1: search:b2c64@200 vs 3 greedy (100 games)
    print("\nRunning T28 Match 1: search:b2c64@200 vs 3 greedy (100 games)...")
    cmd_greedy = [
        find_katago_bin(),
        "q4match",
        "-config",
        cfg_path,
        "-players",
        f"search:{model_path}@200,greedy,greedy,greedy",
        "-games",
        "100",
        "-rotate",
        "-maxplies",
        "200",
    ]
    res1 = subprocess.run(cmd_greedy, capture_output=True, text=True, cwd=REPO_ROOT)
    assert res1.returncode == 0, f"Match vs greedy failed: {res1.stderr}"
    assert "Played 100/100 games..." in res1.stdout
    print(res1.stdout)

    # Match 2: search:b2c64@200 vs 3 basher (100 games)
    print("\nRunning T28 Match 2: search:b2c64@200 vs 3 basher (100 games)...")
    cmd_basher = [
        find_katago_bin(),
        "q4match",
        "-config",
        cfg_path,
        "-players",
        f"search:{model_path}@200,basher,basher,basher",
        "-games",
        "100",
        "-rotate",
        "-maxplies",
        "200",
    ]
    res2 = subprocess.run(cmd_basher, capture_output=True, text=True, cwd=REPO_ROOT)
    assert res2.returncode == 0, f"Match vs basher failed: {res2.stderr}"
    assert "Played 100/100 games..." in res2.stdout
    print(res2.stdout)
