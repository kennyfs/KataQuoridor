"""Tests of python/q4/record.py: the SGF reader and writer against the C++ ones (docs/q4/Q4Sgf.md)."""
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q4.record import (Q4Record, format_move_comment, load_records, parse_move_comment,  # noqa: E402
                       record_dict_to_sgf_line)
from tests.q4_testutil import find_katago  # noqa: E402


def popgames(n, seed, *extra):
    proc = subprocess.run([find_katago("eigen"), "q4tool", "popgames", "-n", str(n), "-seed", str(seed), *extra],
                          capture_output=True, text=True, check=True)
    return proc.stdout.strip().splitlines()


def test_move_comment_round_trip():
    text = "0.31 0.22 0.27 0.15 0.05 v=600 weight=1.00"
    c = parse_move_comment(text)
    assert c == {"p": [0.31, 0.22, 0.27, 0.15, 0.05], "visits": 600, "weight": 1.0}
    assert format_move_comment(c) == text
    assert parse_move_comment("elim") is None
    assert parse_move_comment("0.1 0.2 v=3") is None


def test_hand_written_game():
    line = ("(;FF[4]GM[Q4]SZ[11]PS[a]PW[b]PN[c]PE[d\\]x]WS[7]WW[7]WN[5]WE[7]RU[Q4:repetitionDrawCount=3,maxPlies=50]"
            "RE[0]DR[maxPlies]C[gtype=fork,startTurnIdx=1,type0=selfplay,net0=m1,v0=600,type1=bot,type2=bot,type3=bot,table=t,opening=2,rotation=3]"
            ";S[f2]C[0.31 0.22 0.27 0.15 0.05 v=600 weight=1.00];W[b6];EL[N];E[j6])")
    rec = Q4Record.from_sgf_line(line)
    assert rec.players[3]["name"] == "d]x" and rec.rules["initialWalls"] == [7, 7, 5, 7]
    assert rec.rules["repetitionDrawCount"] == 3 and rec.rules["maxPlies"] == 50
    assert rec.result == "Draw" and rec.draw_reason == "maxPlies" and rec.gtype == "fork"
    assert rec.events == [{"a": "f2"}, {"a": "b6"}, {"elim": 3}, {"a": "j6"}]
    assert rec.match == {"table": "t", "opening": 2, "rotation": 3, "openingPlies": 1, "drawReason": "maxPlies"}
    assert rec.players[0]["net"] == "m1" and rec.players[0]["visits"] == 600
    d = rec.to_dict()
    assert d["moveComments"][0]["visits"] == 600 and d["moveComments"][1] is None
    assert rec.to_sgf_line() == line
    with pytest.raises(ValueError):
        Q4Record.from_sgf_line("(;FF[4]GM[1]SZ[17])")
    with pytest.raises(ValueError):
        Q4Record.from_sgf_line("(;FF[4]GM[Q4]SZ[11];S[f2)")


def test_cpp_population_games_round_trip(tmp_path):
    """The Python writer reproduces the line of every C++ game (eliminations, draws, mover letters)."""
    lines = popgames(80, 5, "-maxplies", "120")
    n_elim = n_draw = 0
    for line in lines:
        rec = Q4Record.from_sgf_line(line)
        assert rec.to_sgf_line() == line
        assert record_dict_to_sgf_line(rec.to_dict()) == line
        n_elim += any("elim" in e for e in rec.events)
        n_draw += rec.result == "Draw"
        pos = rec.replay()
        assert (pos.winner is None) == (rec.result in ("Draw", "none"))
    assert n_elim > 0
    path = tmp_path / "g.sgfs"
    path.write_text("\n".join(lines) + "\n")
    assert len(list(load_records(str(path)))) == len(lines)
