"""End-to-end tests of the scheduler with the mock engine as both arbiter and players."""
import json
import os
import sys

from quoridor_arena import arena

MOCK = os.path.join(os.path.dirname(__file__), "mock_engine.py")


def write_roster(path, extra=None, arbiter_args=(), **top):
    # The mock also answers kata-genmove_analyze (for the play GUI); m1..m3 stand for plain QTP engines.
    engines = [
        {"name": "m1", "command": sys.executable, "args": [MOCK, "--seed", "{seed}"], "seed": 1, "supports_rules": True,
         "kata": False},
        {"name": "m2", "command": sys.executable, "args": [MOCK], "seed": 2, "supports_rules": True, "kata": False},
        {"name": "m3", "command": sys.executable, "args": [MOCK], "supports_rules": True, "kata": False},
    ] + (extra or [])
    with open(path, "w") as f:
        json.dump(dict({"arbiter": [sys.executable, MOCK] + list(arbiter_args), "engines": engines}, **top), f)


def read_sgfs(out, name):
    with open(os.path.join(out, "sgfs", name)) as f:
        return [l for l in f if l.strip()]


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
    # KM / WB / WW / RE come from the arbiter: the standard game, and the toy game's B+1 is a lead of 0.5.
    assert "RE[B+0.5]" in line and "KM[-0.5]WB[10]WW[10]" in line and "startTurnIdx=4" in line
    assert all(r["result"] == "B+0.5" and r["lead"] == 0.5 and r["komi"] == -0.5 for r in res)

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


def test_komi_and_walls(tmp_path):
    """Komi and initial walls from the roster ("rules", "pair_rules") and the command line reach the engines and the
    arbiter; the winner and the SGF's KM / WB / WW / RE are the arbiter's."""
    roster, out = str(tmp_path / "r.json"), str(tmp_path / "out")
    write_roster(roster, rules={"komi": 0.5, "whiteInitialWalls": 9},
                 pair_rules=[{"pair": ["m3", "m1"], "komi": -1.5, "blackInitialWalls": 8}])
    assert run(roster, out, "--games-per-pair", "2", "--pairs", "m1:m2,m1:m3") == 0
    res = {r["id"]: r for r in read_results(out)}
    assert set(res) == {"m1_vs_m2_o000_ab_k+0.5_w10-9", "m1_vs_m2_o000_ba_k+0.5_w10-9",
                        "m1_vs_m3_o000_ab_k-1.5_w8-9", "m1_vs_m3_o000_ba_k-1.5_w8-9"}
    for r in res.values():
        if "m2" in r["pair"]:
            # The toy game ends B+1 (tempo 0): with komi +0.5 White wins by 0.5.
            assert (r["komi"], r["black_walls"], r["white_walls"]) == (0.5, 10, 9)
            assert r["winner"] == "w" and r["result"] == "W+0.5" and r["lead"] == 0.5 and r["margin"] == 1
        else:
            assert (r["komi"], r["black_walls"], r["white_walls"]) == (-1.5, 8, 9)
            assert r["winner"] == "b" and r["result"] == "B+1.5"
    line = read_sgfs(out, "m1_vs_m2.sgfs")[0]
    assert "KM[0.5]WB[10]WW[9]" in line and "RE[W+0.5]" in line and "KM[0]" not in line
    line = read_sgfs(out, "m1_vs_m3.sgfs")[0]
    assert "KM[-1.5]WB[8]WW[9]" in line and "RE[B+1.5]" in line
    report = open(os.path.join(out, "report.md")).read()
    assert "komi +0.5, walls 10/9" in report and "komi -1.5, walls 8/9" in report

    # The command line overrides the roster's "rules", and "pair_rules" override both, key by key. New rules make
    # new game ids.
    assert run(roster, out, "--games-per-pair", "2", "--pairs", "m1:m2,m1:m3", "--komi", "-0.5",
               "--white-walls", "10") == 0
    new = {r["id"]: r for r in read_results(out) if r["id"] not in res}
    assert set(new) == {"m1_vs_m2_o000_ab", "m1_vs_m2_o000_ba",
                        "m1_vs_m3_o000_ab_k-1.5_w8-10", "m1_vs_m3_o000_ba_k-1.5_w8-10"}
    assert all(new[i]["result"] == "B+0.5" and new[i]["komi"] == -0.5 for i in ("m1_vs_m2_o000_ab", "m1_vs_m2_o000_ba"))


def test_rules_need_support(tmp_path):
    """Engines without komi / walls support only play standard games."""
    roster, out = str(tmp_path / "r.json"), str(tmp_path / "out")
    write_roster(roster, [{"name": "plain", "command": sys.executable, "args": [MOCK, "--no-rules"]}])
    assert run(roster, out, "--games-per-pair", "2", "--pairs", "m1:plain") == 0
    try:
        run(roster, str(tmp_path / "out2"), "--games-per-pair", "2", "--pairs", "m1:plain", "--komi", "1.5")
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "plain" in str(e)
    # An engine that claims support but rejects the commands aborts the run.
    write_roster(roster, [{"name": "liar", "command": sys.executable, "args": [MOCK, "--no-rules"],
                           "supports_rules": True}])
    assert run(roster, str(tmp_path / "out3"), "--games-per-pair", "2", "--pairs", "m1:liar", "--komi", "1.5") == 1
    for bad in ({"komi": 1.0}, {"komi": 21.5}, {"blackInitialWalls": 11}, {"maxPlies": 5}):
        write_roster(roster, rules=bad)
        try:
            run(roster, str(tmp_path / "out4"), "--games-per-pair", "2", "--pairs", "m1:m2")
            assert False, "expected SystemExit for %r" % bad
        except SystemExit:
            pass


def test_arbiter_rule_draw(tmp_path):
    """A draw by the arbiter's own ply limit (KataQuoridor's maxPlies rule) is reported as a draw."""
    roster, out = str(tmp_path / "r.json"), str(tmp_path / "out")
    write_roster(roster, arbiter_args=["--max-plies", "10"])
    assert run(roster, out, "--games-per-pair", "2", "--pairs", "m1:m2") == 0
    res = read_results(out)
    assert len(res) == 2
    for r in res:
        assert r["winner"] is None and r["winner_name"] is None and r["reason"] == "draw10"
        assert r["result"] == "0" and r["lead"] == 0.0 and r["plies"] == 10
    assert "RE[0]" in read_sgfs(out, "m1_vs_m2.sgfs")[0]
    assert "draw rate 100.0%" in open(os.path.join(out, "report.md")).read()


def test_repetition_draw(tmp_path):
    """With the repetition rule (--repetition-draw-count, sent to the engines and the arbiter), shuffling engines draw
    at the N-th occurrence of the start position; the report counts repetition draws apart from ply-limit draws.
    Engines without rules support can't play such games."""
    roster, out = str(tmp_path / "r.json"), str(tmp_path / "out")
    write_roster(roster, [{"name": "s1", "command": sys.executable, "args": [MOCK, "--shuffle"], "supports_rules": True},
                          {"name": "s2", "command": sys.executable, "args": [MOCK, "--shuffle"], "supports_rules": True},
                          {"name": "plain", "command": sys.executable, "args": [MOCK, "--no-rules", "--shuffle"]}],
                 arbiter_args=["--max-plies", "30"])
    assert run(roster, out, "--games-per-pair", "2", "--pairs", "s1:s2", "--repetition-draw-count", "3") == 0
    res = read_results(out)
    assert {r["id"] for r in res} == {"s1_vs_s2_o000_ab_k-0.5_w10-10_r3", "s1_vs_s2_o000_ba_k-0.5_w10-10_r3"}
    for r in res:
        assert r["winner"] is None and r["reason"] == "repetition" and r["result"] == "0"
        assert r["repetition_draw_count"] == 3
        # The openings are toy walls (they reset the history), then the start position recurs every 4 plies.
        assert r["plies"] >= 8
    line = read_sgfs(out, "s1_vs_s2.sgfs")[0]
    assert "RU[Quoridor:repetitionDrawCount=3]" in line and "RE[0]" in line and "DR[repetition]" in line
    report = open(os.path.join(out, "report.md")).read()
    assert "by repetition 100.0%" in report and "repetition 2" in report and "Anomalies" not in report
    assert "komi -0.5, walls 10/10, repetition 3" in report

    # Without the rule the same engines shuffle until the arbiter's ply limit.
    out2 = str(tmp_path / "out2")
    assert run(roster, out2, "--games-per-pair", "2", "--pairs", "s1:s2") == 0
    for r in read_results(out2):
        assert r["reason"] == "draw30" and r["repetition_draw_count"] == 0
    assert "DR[maxPlies]" in read_sgfs(out2, "s1_vs_s2.sgfs")[0]

    try:
        run(roster, str(tmp_path / "out3"), "--games-per-pair", "2", "--pairs", "s1:plain", "--repetition-draw-count", "3")
        assert False, "expected SystemExit"
    except SystemExit as e:
        assert "plain" in str(e)
