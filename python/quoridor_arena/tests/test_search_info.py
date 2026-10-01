"""Search info of KataQuoridor engines (kata-genmove_analyze) on the arena's SGF move nodes."""
import os
import sys

import pytest

from quoridor_arena import referee
from quoridor_arena.tests.test_arena import MOCK, read_results, read_sgfs, run, write_roster

# The engine's output (cpp/command/gtp.cpp), side to move's view, as printed for White to move.
WHITE_TO_MOVE = (
    "\ninfo move e2 visits 63 edgeVisits 63 utility -0.03 winrate 0.484092 scoreMean -0.172573 scoreStdev 2.59 "
    "scoreLead -0.172573 scoreSelfplay -0.177383 noResultValue 0.02 prior 0.90 lcb 0.47 utilityLcb -0.05 weight 73 "
    "order 0 pv e2 e7 info move a1h visits 1 edgeVisits 1 utility 0 winrate 0.4 scoreMean 0 scoreStdev 1 "
    "scoreLead 0 scoreSelfplay 0 noResultValue 0.5 prior 0.01 lcb 0 utilityLcb 0 weight 1 order 1 pv a1h "
    "rootInfo visits 64 utility -0.03 winrate 0.484088 scoreMean -0.173807 scoreStdev 2.59 scoreLead -0.173807 "
    "scoreSelfplay -0.1779 weight 74.3 rawStWrError 0.12 rawStScoreError 0.78 rawVarTimeLeft 9.8\nplay e2")


def test_parse_white_to_move():
    mv, ev = referee.parse_kata_genmove(WHITE_TO_MOVE, "w")
    assert mv == "e2"
    assert ev["visits"] == 64
    assert ev["noResult"] == pytest.approx(0.02)  # the played move's, not a1h's
    assert ev["win"] == pytest.approx(0.484088 - 0.01)
    assert ev["win"] + ev["loss"] + ev["noResult"] == pytest.approx(1.0)
    assert ev["lead"] == pytest.approx(-0.173807) and ev["score"] == pytest.approx(-0.1779)
    assert referee.eval_comment(ev) == "0.47 0.51 0.02 -0.2 v=64 lead=-0.17"


def test_parse_black_to_move_is_flipped_to_white():
    text = WHITE_TO_MOVE.replace("play e2", "play e8").replace("move e2", "move e8")
    mv, ev = referee.parse_kata_genmove(text, "b")
    assert mv == "e8"
    assert ev["win"] == pytest.approx(1 - 0.484088 - 0.01)
    assert ev["lead"] == pytest.approx(0.173807) and ev["score"] == pytest.approx(0.1779)


def test_parse_fixed_perspective():
    _, ev = referee.parse_kata_genmove(WHITE_TO_MOVE.replace("play e2", "play e8").replace("move e2", "move e8"),
                                       "b", perspective="WHITE")
    assert ev["lead"] == pytest.approx(-0.173807)


def test_parse_without_root_info():
    assert referee.parse_kata_genmove("\nplay e2", "w") == ("e2", None)
    assert referee.parse_kata_genmove("garbage", "w") == (None, None)


def test_add_move_comments():
    sgf = "(;FF[4]C[root];B[ip];W[ib];B[gpv])"
    assert referee.add_move_comments(sgf, [None, "x y", "z"]) == "(;FF[4]C[root];B[ip];W[ib]C[x y];B[gpv]C[z])"


def test_arena_writes_search_info(tmp_path):
    roster, out = str(tmp_path / "r.json"), str(tmp_path / "out")
    kata = [{"name": "k1", "command": sys.executable, "args": [MOCK, "--kata"], "supports_rules": True},
            {"name": "k2", "command": sys.executable, "args": [MOCK, "--kata"], "supports_rules": True, "kata": False}]
    write_roster(roster, extra=kata)
    assert run(roster, out, "--games-per-pair", "2", "--pairs", "k1:m1,k1:k2") == 0
    res = read_results(out)
    assert len(res) == 4
    for r in res:
        moves_by = {"b": r["black"], "w": r["white"]}
        for i, ev in enumerate(r["evals"]):
            color = "b" if i % 2 == 0 else "w"
            if i < 4 or moves_by[color] != "k1":  # opening, plain genmove, or "kata": false
                assert ev is None
            else:  # the mock's values, White's view
                assert ev == ([0.3, -3.0, -2.5, 10] if color == "b" else [0.2, -2.0, -1.5, 10])
    for name in ("k1_vs_m1.sgfs", "k1_vs_k2.sgfs"):
        for line in read_sgfs(out, name):
            nodes = line.split(";")[2:]
            k1_black = "PB[k1]" in line
            for i, node in enumerate(nodes):
                node = node.strip().rstrip(")")
                color = "b" if i % 2 == 0 else "w"
                if i >= 4 and (color == "b") == k1_black:
                    want = "0.30 0.70 0.00 -3.0 v=10 lead=-2.50" if color == "b" else "0.20 0.80 0.00 -2.0 v=10 lead=-1.50"
                    assert node.endswith("C[%s]" % want), node
                else:
                    assert "C[" not in node, node
    # Non-kata games keep no evals.
    assert run(roster, str(tmp_path / "o2"), "--games-per-pair", "2", "--pairs", "m1:m2") == 0
    assert all(r["evals"] is None for r in read_results(str(tmp_path / "o2")))
