"""End-to-end tests of the scheduler with the mock engine as both arbiter and players."""
import json
import os
import sys

from quoridor_arena import arena

MOCK = os.path.join(os.path.dirname(__file__), "mock_engine.py")


def write_roster(path, extra=None):
    engines = [
        {"name": "m1", "command": sys.executable, "args": [MOCK, "--seed", "{seed}"], "seed": 1},
        {"name": "m2", "command": sys.executable, "args": [MOCK], "seed": 2},
        {"name": "m3", "command": sys.executable, "args": [MOCK]},
    ] + (extra or [])
    with open(path, "w") as f:
        json.dump({"arbiter": [sys.executable, MOCK], "engines": engines}, f)


def read_results(out):
    with open(os.path.join(out, "results.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def run(roster, out, *extra):
    return arena.main(["--roster", roster, "--out", out, "--bootstrap", "20", "--anchor", "m1",
                       "--concurrency", "2", "--skip-min-games", "0"] + list(extra))


def test_round_robin_and_resume(tmp_path):
    roster, out = str(tmp_path / "r.json"), str(tmp_path / "out")
    write_roster(roster)
    assert run(roster, out, "--games-per-pair", "4", "--verify") == 0
    res = read_results(out)
    assert len(res) == 12 and len({r["id"] for r in res}) == 12
    # Paired games: each opening is played with both colour assignments.
    for r in res:
        a, b = r["pair"]
        assert {r["black"], r["white"]} == {a, b}
        assert r["reason"] == "goal" and r["winner"] == "b"  # the toy game is a first-player win
        assert r["margin"] >= 1
    sgf_files = os.listdir(os.path.join(out, "sgfs"))
    assert len(sgf_files) == 3
    line = open(os.path.join(out, "sgfs", "m1_vs_m2.sgfs")).readline()
    assert "PB[m1]PW[m2]" in line or "PB[m2]PW[m1]" in line
    assert "RE[B+" in line and "KM[0]" in line and "startTurnIdx=4" in line

    # Simulate an interrupted run: drop two finished games and leave a truncated line.
    with open(os.path.join(out, "results.jsonl")) as f:
        lines = f.read().splitlines()
    with open(os.path.join(out, "results.jsonl"), "w") as f:
        f.write("\n".join(lines[:-2]) + "\n" + lines[-1][:20])
    assert run(roster, out, "--games-per-pair", "4") == 0
    res2 = read_results(out)
    assert len(res2) == 12 and {r["id"] for r in res2} == {r["id"] for r in res}

    # A complete rerun plays nothing; raising games-per-pair only adds the new games.
    assert run(roster, out, "--games-per-pair", "4") == 0
    assert len(read_results(out)) == 12
    assert run(roster, out, "--games-per-pair", "6") == 0
    assert len(read_results(out)) == 18
    assert os.path.exists(os.path.join(out, "report.md"))


def test_forfeits(tmp_path):
    roster, out = str(tmp_path / "r.json"), str(tmp_path / "out")
    write_roster(roster, [
        {"name": "bad", "command": sys.executable, "args": [MOCK, "--illegal"]},
        {"name": "crashy", "command": sys.executable, "args": [MOCK, "--crash-after", "2"]},
    ])
    rc = run(roster, out, "--pairs", "m1:bad,m1:crashy", "--games-per-pair", "2")
    assert rc == 0
    res = {r["id"]: r for r in read_results(out)}
    bad = [r for r in res.values() if "bad" in r["pair"]]
    assert all(r["reason"] == "illegal" and r["winner_name"] == "m1" for r in bad)
    crashy = [r for r in res.values() if "crashy" in r["pair"]]
    # crashy dies on its 2nd genmove in the first game it plays; it is restarted for the next game
    # (and dies again on its 2nd genmove, since the counter restarts with the process).
    assert all(r["reason"] == "crash" and r["winner_name"] == "m1" for r in crashy)
    assert os.path.exists(os.path.join(out, "anomalies.log"))


def test_adaptive_skip(tmp_path):
    roster, out = str(tmp_path / "r.json"), str(tmp_path / "out")
    write_roster(roster, [{"name": "bad", "command": sys.executable, "args": [MOCK, "--illegal"]}])
    rc = arena.main(["--roster", roster, "--out", out, "--bootstrap", "0", "--anchor", "m1", "--concurrency", "1",
                     "--pairs", "m1:bad", "--games-per-pair", "40", "--skip-min-games", "20"])
    assert rc == 0
    assert len(read_results(out)) == 20
