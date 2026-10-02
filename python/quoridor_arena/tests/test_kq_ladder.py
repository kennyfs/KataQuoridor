import collections
import json

import numpy as np
import pytest

from quoridor_arena import elo, kq_ladder


def fake_lister(spec):
    return spec  # the test config lists "directories" directly


def make_cfg(n_main=8, n_other=3, visit_scaling=True):
    main = ["/m/run3-s%d-d1" % int(1e6 * 1.4 ** k) for k in range(n_main)]
    other = ["/o/run1-s%d-d1" % int(1.3e6 * 2.0 ** k) for k in range(n_other)]
    cfg = {
        "series": [{"name": "run3", "prefix": "r3", "models": main},
                   {"name": "q0", "prefix": "q0", "models": other}],
        "visits": 256,
        "links": {"generation_distances": [1, 2, 3, 6], "cross": True},
        "games": {"max": 80, "min": 16, "sigma_log_samples": 0.7},
        "match": {"numGameThreads": 8, "nnMaxBatchSize": 8},
        "rules": {"timeBonusPerPly": 0.05},
    }
    if visit_scaling:
        cfg["visit_scaling"] = {"series": "run3", "net": "latest", "visits": [64, 1024], "games": 40}
    return cfg


def roster(cfg):
    return kq_ladder.make_roster(cfg, model_lister=fake_lister)


def test_model_samples():
    assert kq_ladder.model_samples("/x/run3-s914176-d563962") == 914176
    assert kq_ladder.model_samples("/x/run1-s16141056-d2846694/") == 16141056
    with pytest.raises(SystemExit):
        kq_ladder.model_samples("/x/model")


def test_roster_names_and_visit_bots():
    cfg = make_cfg()
    nets, bots = roster(cfg)
    assert [n["gen"] for n in nets["run3"]] == list(range(8))
    latest = nets["run3"][-1]["samples"]
    assert "r3-s%d-v64" % latest in bots and "r3-s%d-v1024" % latest in bots
    assert sum(1 for b in bots.values() if b["visits"] == 256) == 11
    assert len(bots) == 13


def test_games_for_distance():
    g = {"max": 80, "min": 16, "sigma_log_samples": 0.7}
    assert kq_ladder.games_for_distance(1e6, 1e6, g) == 80
    near = kq_ladder.games_for_distance(1e6, 1.2e6, g)
    far = kq_ladder.games_for_distance(1e6, 3e6, g)
    assert 80 >= near > far >= 16
    assert kq_ladder.games_for_distance(1e6, 1e8, g) == 16
    for s in (1.1e6, 1.5e6, 2.2e6, 5e6):
        assert kq_ladder.games_for_distance(1e6, s, g) % 2 == 0


def test_pairs_links_and_connectivity():
    cfg = make_cfg()
    nets, bots = roster(cfg)
    pairs = kq_ladder.make_pairs(cfg, nets, bots)
    kinds = collections.Counter(k for _, _, _, k in pairs)
    # run3 (8 nets): distances 1, 2, 3, 6 -> 7 + 6 + 5 + 2; q0 (3 nets): 2 + 1 (+0 + 0).
    assert kinds["gen1"] == 7 + 2 and kinds["gen2"] == 6 + 1 and kinds["gen3"] == 5 and kinds["gen6"] == 2
    assert kinds["visits"] == 2
    assert kinds["cross"] + kinds["cross-ends"] >= 3
    # No duplicates (either order), a is the older net, every bot is in some pair.
    seen = set()
    for a, b, n, kind in pairs:
        assert frozenset((a, b)) not in seen
        seen.add(frozenset((a, b)))
        assert n % 2 == 0 and n >= 16
        if kind != "visits":
            assert bots[a]["samples"] <= bots[b]["samples"]
    used = {x for a, b, _, _ in pairs for x in (a, b)}
    assert used == set(bots)
    # The pair graph is connected (one Bradley-Terry component).
    names = list(bots)
    idx = {n: k for k, n in enumerate(names)}
    N = np.zeros((len(names), len(names)))
    for a, b, _, _ in pairs:
        N[idx[a], idx[b]] = N[idx[b], idx[a]] = 1
    assert len(set(elo.components(N))) == 1


def test_cross_links_nearest():
    cfg = make_cfg(visit_scaling=False)
    nets, bots = roster(cfg)
    pairs = kq_ladder.make_pairs(cfg, nets, bots)
    cross = [(a, b) for a, b, _, k in pairs if k == "cross"]
    # q0's first net (1.3M samples) is nearest to run3's 1.4M net.
    assert ("r3-s1400000-v256", "q0-s1300000-v256") in cross or ("q0-s1300000-v256", "r3-s1400000-v256") in cross
    ends = [(a, b) for a, b, _, k in pairs if k == "cross-ends"]
    assert ("r3-s1000000-v256", "q0-s1300000-v256") in ends


def test_schedule_colours_openings_and_extra():
    pairs = [("a", "b", 4, "gen1"), ("a", "c", 2, "gen2")]
    sched = kq_ladder.make_schedule(pairs, {"a:c": 2})
    assert [g[0] for g in sched] == ["a_vs_b_o000_ab", "a_vs_b_o000_ba", "a_vs_b_o001_ab", "a_vs_b_o001_ba",
                                     "a_vs_c_o000_ab", "a_vs_c_o000_ba", "a_vs_c_o001_ab", "a_vs_c_o001_ba"]
    for gid, a, b, k, black, white in sched:
        assert (black, white) == ((b, a) if gid.endswith("_ba") else (a, b))
    # Games of a pair are contiguous (large NN batches per model).
    order = [(g[1], g[2]) for g in sched]
    assert order == sorted(order, key=lambda p: [("a", "b"), ("a", "c")].index(p))


def test_game_list_and_match_config():
    cfg = make_cfg()
    nets, bots = roster(cfg)
    pairs = kq_ladder.make_pairs(cfg, nets, bots)
    sched = kq_ladder.make_schedule(pairs)[:2]
    index = {n: k for k, n in enumerate(bots)}
    text = kq_ladder.game_list_text(sched, index, [["e2", "d6v"]])
    l0, l1 = text.splitlines()
    gid, b, w = l0.split()[:3]
    assert gid == sched[0][0] and int(b) == index[sched[0][4]] and int(w) == index[sched[0][5]]
    assert l0.split()[3:] == ["e2", "d6v"]
    assert l1.split()[1:3] == [w, b]  # colours swapped
    mc = kq_ladder.match_config(cfg, bots, "/g.txt", "/r.jsonl")
    assert "numBots = %d" % len(bots) in mc
    assert "repetitionDrawCount = 0" in mc and "timeBonusPerPly = 0.05" in mc and "maxPlies = 300" in mc
    assert "allowResignation = false" in mc and "handicapCompensateKomiProb = 0.0" in mc
    assert "gameListFile = /g.txt" in mc
    latest = nets["run3"][-1]["samples"]
    i = index["r3-s%d-v1024" % latest]
    assert "maxVisits%d = 1024" % i in mc


def raw(gid, black, white, winner, result, plies=50, draw_reason=""):
    return {"id": gid, "black": black, "white": white, "winner": winner, "result": result,
            "draw_reason": draw_reason, "plies": plies, "opening_plies": 4,
            "rules": "Quoridor:timeBonusPerPly=0.05", "black_walls": 10, "white_walls": 10,
            "b_time": 1.0, "w_time": 2.0}


def test_parse_lead():
    assert kq_ladder.parse_lead("W+2.5") == 2.5
    assert kq_ladder.parse_lead("B+0.5") == -0.5
    assert kq_ladder.parse_lead("0") == 0.0
    assert kq_ladder.parse_lead("W+R") is None


def test_convert_results():
    sched = kq_ladder.make_schedule([("a", "b", 4, "gen1")])
    ops = [["e2", "e8"], ["d6v", "f1"]]
    rs = [raw("a_vs_b_o000_ab", "a", "b", "w", "W+2.5"),
          raw("a_vs_b_o000_ba", "b", "a", "b", "B+0.5"),
          raw("a_vs_b_o001_ab", "a", "b", None, "0", plies=300, draw_reason="maxPlies"),
          raw("a_vs_b_o000_ab", "a", "b", "b", "B+9.5"),   # duplicate id: the first record counts
          raw("zzz", "a", "b", "b", "B+1.5")]             # not scheduled: ignored
    gs = kq_ladder.convert_all(rs, sched, ops)
    assert [g["id"] for g in gs] == ["a_vs_b_o000_ab", "a_vs_b_o000_ba", "a_vs_b_o001_ab"]
    g0, g1, g2 = gs
    assert g0["pair"] == ["a", "b"] and g0["opening"] == 0 and g0["opening_moves"] == ["e2", "e8"]
    assert g0["winner_name"] == "b" and g0["reason"] == "goal" and g0["lead"] == 2.5 and g0["margin"] == 3.0
    assert g1["winner_name"] == "b" and g1["margin"] == 1.0 and g1["lead"] == 0.5
    assert g2["winner"] is None and g2["winner_name"] is None and g2["reason"] == "draw300" and g2["margin"] == 0
    assert g0["seconds"] == 3.0
    # elo.py accepts the converted records: b won 2 of 2 decided games plus a draw.
    assert elo.game_score(g0, "b") == 1.0 and elo.game_score(g2, "a") == 0.5
    r = elo.rate(["a", "b"], gs, "a", resamples=0)
    assert r["b"][0] > 0 and r["b"][3] == 3
    # Units: the two colour-swapped games of opening 0 form one unit.
    ui, uj, us, un = elo.build_units(gs, {"a": 0, "b": 1})
    assert sorted(un.tolist()) == [1.0, 2.0]


def test_convert_rejects_colour_mismatch():
    sched = kq_ladder.make_schedule([("a", "b", 2, "gen1")])
    with pytest.raises(ValueError):
        kq_ladder.convert_all([raw("a_vs_b_o000_ab", "b", "a", "w", "W+0.5")], sched, [["e2"]])


def test_widest_pairs_and_diff_ci():
    names = ["a", "b", "c"]
    point = np.array([0.0, 100.0, 300.0])
    boot = np.array([[0.0, 90.0, 200.0], [0.0, 110.0, 400.0], [0.0, 100.0, 300.0]])
    d, lo, hi = kq_ladder.diff_ci(point, boot, 2, 1)
    assert d == 200.0 and lo < 200 < hi
    pairs = [("a", "b", 10, "gen1"), ("b", "c", 10, "gen1")]
    top = kq_ladder.widest_pairs(pairs, names, point, boot, 1)
    assert top[0][1:] == ("b", "c")
