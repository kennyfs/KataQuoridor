import json
import os

from quoridor_arena import blindspot as bs

# kata-search_analyze output of KataQuoridor (side to move's view); c7h is the mirror image of f7h
SEARCH = ("\ninfo move e6 visits 30 edgeVisits 30 utility 0.03 winrate 0.518 scoreMean 0.11 scoreStdev 1.2 scoreLead 0.11 "
          "scoreSelfplay 0.05 prior 0.32 lcb 0.49 utilityLcb -0.03 weight 40 order 0 pv e6 e4 c7h "
          "info move f7h visits 20 edgeVisits 20 utility 0.01 winrate 0.45 scoreMean 0.01 scoreStdev 1.4 scoreLead 0.01 "
          "scoreSelfplay 0 prior 0.3 lcb 0.4 utilityLcb -0.1 weight 28 order 1 pv f7h d6h "
          "info move c7h visits 20 edgeVisits 20 utility 0.01 winrate 0.45 scoreMean 0.01 scoreStdev 1.4 scoreLead 0.01 "
          "scoreSelfplay 0 prior 0.3 lcb 0.4 utilityLcb -0.1 weight 28 isSymmetryOf f7h order 2 pv c7h e6h\nplay e6")


def test_parse_search():
    ch = bs.parse_search(SEARCH)
    assert [c["move"] for c in ch] == ["e6", "f7h", "c7h"]
    assert ch[0]["visits"] == 30 and ch[0]["wr"] == 0.518 and ch[0]["lead"] == 0.11 and ch[0]["lcb"] == 0.49
    assert ch[2]["sym"] == "f7h" and ch[1]["sym"] is None


def test_position_key_dedupes_transpositions_and_mirrors():
    k1, s1 = bs.state_key("e8 e2 e7 e3".split())
    k2, _ = bs.state_key("e8 e2 e7 e3".split())
    assert k1 == k2 and s1                        # the symmetric opening
    a, sa = bs.state_key("a7h e2 e8 e3".split())
    b, _ = bs.state_key("h7h e2 e8 e3".split())   # mirror of a7h
    assert a == b and not sa
    # the same walls placed in a different order by the same players are the same position
    x, _ = bs.state_key("a7h e2 c3h e3 e8 e4".split())
    y, _ = bs.state_key("e8 e2 c3h e3 a7h e4".split())
    z, _ = bs.state_key("c3h e2 a7h e3 e8 e4".split())
    assert x == y == z
    # who placed the wall matters (wall counts), and so does the side to move
    assert bs.state_key("a7h e2".split())[0] != bs.state_key("e8 a7h".split())[0]
    assert bs.state_key("e8 e2".split())[0] != bs.state_key("e8 e2 e7".split())[0]


def game(moves, start=0, comments=None, km="-0.5"):
    sgf_moves = []
    for i, m in enumerate(moves):
        x, y = ord(m[0]) - 97, ord(m[1]) - 97
        # 9x9 -> 17x17
        if len(m) == 2:
            xs, ys = 2 * x, 2 * (int(m[1]) - 1)
        else:
            xs, ys = 2 * x + 1, 2 * (int(m[1]) - 1) + (1 if m[2] == "h" else 0)
        c = comments[i] if comments and comments[i] else None
        sgf_moves.append(";%s[%s%s]%s" % ("BW"[i % 2], chr(97 + xs), chr(97 + ys), "C[%s]" % c if c else ""))
    return "(;FF[4]SZ[17]KM[%s]WB[10]WW[10]C[startTurnIdx=%d,gtype=normal]%s)\n" % (km, start, "".join(sgf_moves))


def test_build_groups(tmp_path):
    d = tmp_path / "run4-s1-d1" / "sgfs"
    d.mkdir(parents=True)
    main = "e8 e2 e7 e3 e6 e4".split()
    mirror_line = "e8 e2 e7 e3 e6 e4".split()
    lines = [game(main)] * 12 + [game(mirror_line + ["a6h"])] * 5
    # a game with a value swing at ply 4: White 0.9 -> 0.3 (full searches, v=1000)
    cm = ["0.50 0.50 0.00 0.0 v=1000 weight=1 lead=0"] * 8
    cm[4] = "0.90 0.10 0.00 5 v=1000 weight=1 lead=5"
    cm[5] = "0.30 0.70 0.00 -3 v=1000 weight=1 lead=-3"
    lines.append(game("e8 e2 e7 e3 b3h c3h a2h h2h".split(), comments=cm))
    lines.append(game(main, km="0.5"))        # not a standard game
    (d / "x.sgfs").write_text("".join(lines))
    args = bs.build_args(n_opening=10, n_random=3, n_swing=2, opening_min_games=5, random_min_ply=2)
    pos, info = bs.build_positions(bs.iter_games([str(tmp_path / "run4-s1-d1")]), args)
    assert info["games"] == 18
    ids = [p["id"] for p in pos]
    assert len(ids) == len(set(ids))
    opening = {p["moves"]: p for p in pos if "opening" in p["groups"]}
    assert opening["e8 e2 e7 e3"]["count"] == 18 and opening["e8 e2 e7 e3 e6 e4"]["count"] == 5   # only the games that go on
    assert all(5 <= p["count"] for p in opening.values())
    assert all(p["to_move"] == ("b" if p["ply"] % 2 == 0 else "w") for p in pos)
    swing = [p for p in pos if "swing" in p["groups"]]
    assert swing and swing[0]["swing"] > 0.3 and swing[0]["ply"] == 4
    assert all(p["group"] == next(g for g in bs.GROUPS if g in p["groups"]) for p in pos)


def children(spec):
    return [{"move": m, "visits": v, "wr": wr, "lead": lead, "lcb": wr - 0.03, "prior": 0.1, "order": i, "sym": None}
            for i, (m, v, wr, lead) in enumerate(spec)]


def test_label_clear_unclear_forced():
    clear = bs.make_label(children([("e6", 40000, 0.62, 1.0), ("f7h", 8000, 0.50, 0.2), ("a1h", 300, 0.9, 0)]), 50000, "n", 100)
    assert clear["clear"] and clear["best"] == "e6" and abs(clear["gap_wr"] - 0.12) < 1e-9
    assert clear["second"] == "f7h"            # a1h has too small a visit share to be a rival
    assert "a1h" in clear["children"] and clear["share"] > 0.8
    unclear = bs.make_label(children([("e6", 30000, 0.55, 1.0), ("f7h", 15000, 0.53, 0.2)]), 50000, "n", 100)
    assert not unclear["clear"]
    decided = bs.make_label(children([("e6", 30000, 0.99, 8.0), ("f7h", 15000, 0.985, 3.0)]), 50000, "n", 100)
    assert decided["decided"] and decided["clear"] and decided["gap_lead"] == 5.0
    forced = bs.make_label(children([("e6", 49900, 0.7, 1.0), ("f7h", 100, 0.3, 0)]), 50000, "n", 100)
    assert forced["forced"] and forced["clear"]
    # a mirror image of the best move is not a rival
    ch = children([("c7h", 25000, 0.6, 1), ("f7h", 24000, 0.6, 1)])
    ch[1]["sym"] = "c7h"
    assert bs.make_label(ch, 50000, "n", 100)["forced"]


def test_judge_agreement_regret_and_mirror():
    lab = bs.make_label(children([("e6", 40000, 0.62, 1.0), ("f7h", 8000, 0.50, 0.2), ("g1h", 500, 0.40, 0)]), 50000, "n", 100)
    pos = {"label": lab, "symmetric": False, "id": "x"}
    assert bs.judge(pos, children([("e6", 10, 0.5, 0)])) == ("e6", True, 0.0)
    chosen, agree, regret = bs.judge(pos, children([("f7h", 10, 0.5, 0)]))
    assert not agree and abs(regret - 0.12) < 1e-9
    assert bs.judge(pos, children([("a8", 10, 0.5, 0)])) == ("a8", False, None)   # not visited enough by the deep search
    # in a symmetric position the mirror image of the label move is the label move
    pos2 = {"label": bs.make_label(children([("c7h", 40000, 0.62, 1.0), ("e6", 8000, 0.5, 0)]), 50000, "n", 100), "symmetric": True}
    assert bs.judge(pos2, children([("f7h", 10, 0.5, 0)]))[1] is True
    assert bs.judge(pos, children([("e6", 1, 0, 0)]))[1] and not bs.judge({"label": lab, "symmetric": False}, children([("c7h", 1, 0, 0)]))[1]


def test_summaries_by_group_and_ply():
    rows = [{"group": "opening", "ply": 6, "count": 100, "agree": True, "regret": 0.0},
            {"group": "opening", "ply": 20, "count": 1, "agree": False, "regret": 0.1},
            {"group": "random", "ply": 50, "count": 1, "agree": False, "regret": None},
            {"group": "swing", "ply": 90, "count": 1, "agree": True, "regret": 0.0}]
    r = bs.summarize_all(rows)
    assert r["overall"]["n"] == 4 and r["overall"]["agree"] == 0.5 and r["overall"]["regret_unknown"] == 1
    assert r["overall"]["regret_mean"] == round(0.1 / 3, 4)
    assert r["by_group"]["opening"]["agree_weighted"] == round(100 / 101, 4)
    assert r["by_ply"]["0-8"]["n"] == 1 and r["by_ply"]["81+"]["n"] == 1 and "25-40" not in r["by_ply"]
