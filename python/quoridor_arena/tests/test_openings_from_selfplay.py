import json
import sys

from quoridor_arena import openings_from_selfplay as ofs


def test_notation_and_mirror():
    assert ofs.to_9x9("io") == "e8"       # pawn: both coordinates even
    assert ofs.to_9x9("jl") == "e6h"      # horizontal wall: both odd
    assert ofs.to_9x9("hk") == "d6v"      # vertical wall: x odd, y even
    assert ofs.mirror("e8") == "e8" and ofs.mirror("a3") == "i3"
    assert ofs.mirror("a7h") == "h7h" and ofs.mirror("c7h") == "f7h" and ofs.mirror("d6v") == "e6v"


def game(moves, km="-0.5", wb=10, ww=10, gtype="normal"):
    body = "".join(";%s[%s]" % ("BW"[i % 2], m) for i, m in enumerate(moves))
    return "(;FF[4]SZ[17]KM[%s]WB[%d]WW[%d]C[startTurnIdx=0,gtype=%s]%s)\n" % (km, wb, ww, gtype, body)


def test_suite_merges_mirrors_and_filters(tmp_path, monkeypatch):
    d = tmp_path / "run4-s1-d1"
    (d / "sgfs").mkdir(parents=True)
    a7h = ["bn", "ic", "im", "ie"]    # a7h e2 e7 e3 (wall anchor a = x 1)
    h7h = ["pn", "ic", "im", "ie"]    # its mirror h7h
    main = ["io", "ic", "im", "ie"]   # e8 e2 e7 e3
    lines = [game(main)] * 3 + [game(a7h)] * 2 + [game(h7h)] + [game(main, km="0.5"), game(main, ww=9),
                                                                   game(main, gtype="fork")]
    (d / "sgfs" / "x.sgfs").write_text("".join(lines))
    out = tmp_path / "suite.json"
    monkeypatch.setattr(sys, "argv", ["ofs", str(out), str(d), "--plies", "4", "--min-games", "1"])
    ofs.main()
    data = json.loads(out.read_text())
    assert data["standard_games"] == 6
    assert data["openings"] == [["e8", "e2", "e7", "e3"], ["a7h", "e2", "e7", "e3"]]
    assert data["games"] == [3, 3]
