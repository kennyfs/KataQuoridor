import json
import os

from quoridor_arena import blindspot as bs
from quoridor_arena import hintminer as hm


def ch(spec):
    return [{"move": m, "visits": v, "wr": wr, "lead": lead, "lcb": wr - 0.03, "prior": 0.1, "order": i, "sym": None}
            for i, (m, v, wr, lead) in enumerate(spec)]


ARGS = type("A", (), dict(margin=0.05, lead_margin=2.0, decided=0.95, fresh_visits=1000, deep_visits=10000, min_child_visits=100, min_share=0.15))()
POS = {"id": "p", "moves": "e8 e2 e7 e3 e6 e4", "symmetric": True, "ply": 6, "group": "opening"}


def test_hint_line_is_a_position_sample():
    line = hm.hint_line("e8 e2 e7 e3 e6 e4 a6h".split(), "f3h", 2.0, {"k": 1})
    d = json.loads(line)
    # the keys of Sgf::PositionSample::toJsonLine
    assert set(d) == {"xSize", "ySize", "board", "nextPla", "moveLocs", "movePlas", "initialTurnNumber", "hintLoc", "weight", "metadata"}
    assert d["nextPla"] == "B" and d["movePlas"] == list("BWBWBWB") and d["xSize"] == d["ySize"] == 17 and d["initialTurnNumber"] == 0
    back = hm.parse_hint_line(line)
    assert back["moves"][-1] == "a6h" and back["hint"] == "f3h" and back["weight"] == 2.0 and back["metadata"] == {"k": 1}


def test_make_hint_when_deep_best_is_better():
    fresh = ch([("f7h", 700, 0.5, 0)])
    deep = ch([("a6h", 6000, 0.60, 1.0), ("f7h", 3000, 0.50, 0.5), ("g1h", 100, 0.1, 0)])
    h = hm.make_hint(POS, fresh, deep, ARGS, "net")
    assert h["hint"] == "a6h" and h["moves"] == POS["moves"].split() and abs(h["metadata"]["margin"] - 0.10) < 1e-9
    assert h["weight"] == 2.0 and h["metadata"]["fresh_best"] == "f7h" and h["metadata"]["kind"] == "wr"


def test_no_hint_cases():
    deep = ch([("a6h", 6000, 0.60, 1.0), ("f7h", 3000, 0.58, 0.5)])
    assert hm.make_hint(POS, ch([("a6h", 1, 0.5, 0)]), deep, ARGS, "n") is None              # same move
    assert hm.make_hint(POS, ch([("f7h", 1, 0.5, 0)]), deep, ARGS, "n") is None              # margin 0.02, label unclear anyway
    deep2 = ch([("a6h", 6000, 0.60, 1.0), ("f7h", 2900, 0.58, 0.5), ("e5", 3000, 0.56, 0.4)])
    assert hm.make_hint(POS, ch([("e5", 1, 0.5, 0)]), deep2, ARGS, "n") is None              # not clear (rival f7h at 0.58)
    deep3 = ch([("a6h", 9000, 0.60, 1.0), ("f7h", 100, 0.30, 0.5)])
    assert hm.make_hint(POS, ch([("c1h", 1, 0.5, 0)]), deep3, ARGS, "n") is None             # the fresh move is not in the deep search
    # the fresh move is the mirror image of the deep move
    assert hm.make_hint(POS, ch([("e5", 1, 0.5, 0)]), ch([("c7h", 5000, 0.6, 1), ("e5", 20, 0.6, 1)]), ARGS, "n") is None


def test_lead_margin_only_when_decided():
    deep = ch([("a6h", 6000, 0.99, 9.0), ("f7h", 3000, 0.985, 4.0)])
    h = hm.make_hint(POS, ch([("f7h", 1, 0.5, 0)]), deep, ARGS, "n")
    assert h["metadata"]["kind"] == "lead" and h["weight"] == 1.0
    deep = ch([("a6h", 6000, 0.80, 9.0), ("f7h", 3000, 0.79, 4.0)])        # lead differs, but the game is not decided
    assert hm.make_hint(POS, ch([("f7h", 1, 0.5, 0)]), deep, ARGS, "n") is None


def test_expiry_keeps_last_runs(tmp_path):
    for i in range(5):
        hm.write_hint_file(str(tmp_path / ("mine_2026101%d-000000_n%s" % (i, hm.HINT_SUFFIX))),
                           [{"moves": ["e8", "e2"], "hint": "e7", "weight": 1.0, "metadata": {}}])
    args = type("A", (), dict(keep_runs=3, no_recheck=True))()
    hm.prune_old(str(tmp_path), args, "n")
    names = [os.path.basename(f) for f in hm.hint_files(str(tmp_path))]
    assert names == ["mine_20261013-000000_n" + hm.HINT_SUFFIX, "mine_20261014-000000_n" + hm.HINT_SUFFIX]
    assert len(hm.hint_files(str(tmp_path))) == 2       # room for the new run's file


def test_deep_move_needs_a_visit_share():
    fresh = ch([("f7h", 700, 0.5, 0)])
    deep = ch([("a6h", 1000, 0.60, 1.0), ("f7h", 100, 0.50, 0.5), ("e5", 8900, 0.45, 0)])   # best by order, 10% of the visits
    deep[0]["order"], deep[2]["order"] = 0, 2
    assert hm.make_hint(POS, fresh, deep, ARGS, "net") is None
