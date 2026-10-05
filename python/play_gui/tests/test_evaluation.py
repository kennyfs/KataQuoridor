import pytest

from play_gui import evaluation as ev

# KataQuoridor 0.1.0 output (release net, 64 visits) after "play b e8", White to move: the first two info
# entries of a real kata-genmove_analyze answer, with a rootInfo entry in the engine's format appended.
GENMOVE_W = (
    "info move e2 visits 59 edgeVisits 59 utility -0.210351 winrate 0.400073 scoreMean -0.829902 scoreStdev 2.40859 "
    "scoreLead -0.829902 scoreSelfplay -0.829902 prior 0.813387 lcb 0.370163 utilityLcb -0.294099 weight 59.4455 "
    "order 0 pv e2 e7 e3 e6 e4 h6h d3h f6h e5v d6h info move h2h visits 1 edgeVisits 1 utility -0.533663 "
    "winrate 0.266911 scoreMean -2.38343 scoreStdev 2.66109 scoreLead -2.38343 scoreSelfplay -2.38343 "
    "prior 0.037042 lcb -0.983089 utilityLcb -4.03366 weight 1.12998 order 1 pv h2h rootInfo visits 64 "
    "utility -0.21 winrate 0.401 scoreMean -0.8 scoreStdev 2.3 scoreLead -0.81 scoreSelfplay -0.8 weight 66.6\n"
    "play e2")
# Real kata-raw-nn 0 output for the same position (policy and ownership arrays shortened).
RAW_NN = """symmetry 0
whiteWin 0.399697
whiteLoss 0.600303
noResult 0.000000
whiteLead -0.810
whiteScoreSelfplay -0.810
whiteScoreSelfplaySq 6.018
varTimeLeft 6.436
policy
    NAN 0.000860     NAN 0.000729
policyPass     NAN
whiteOwnership
0.0000000 0.0000000
"""


def test_side_to_move_black():
    e = ev.from_side_to_move(0.7, 2.5, "b")
    assert e["black_win"] == pytest.approx(0.7)
    assert e["white_win"] == pytest.approx(0.3)
    assert e["black_lead"] == pytest.approx(2.5)


def test_side_to_move_white_is_flipped():
    # White to move with 70% and +2.5 means Black has 30% and trails by 2.5 moves.
    e = ev.from_side_to_move(0.7, 2.5, "w")
    assert e["black_win"] == pytest.approx(0.3)
    assert e["white_win"] == pytest.approx(0.7)
    assert e["black_lead"] == pytest.approx(-2.5)


def test_side_to_move_bad_player():
    with pytest.raises(ValueError):
        ev.from_side_to_move(0.5, 0.0, "x")


def test_raw_nn_is_white_perspective():
    e = ev.from_raw_nn(ev.parse_raw_nn(RAW_NN))
    assert e["black_win"] == pytest.approx(0.600303, abs=1e-6)
    assert e["white_win"] == pytest.approx(0.399697, abs=1e-6)
    assert e["black_lead"] == pytest.approx(0.81)
    assert e["source"] == "net"


def test_raw_nn_and_search_agree_on_the_same_position():
    # The same position (after b e8, White to move) from both commands must give the same Black numbers.
    a = ev.parse_analysis(GENMOVE_W)
    root = ev.from_side_to_move(a["root"]["winrate"], a["root"]["scoreLead"], "w")
    raw = ev.from_raw_nn(ev.parse_raw_nn(RAW_NN))
    assert root["black_win"] == pytest.approx(raw["black_win"], abs=0.01)
    assert root["black_lead"] == pytest.approx(raw["black_lead"], abs=0.01)
    assert root["black_win"] > 0.5 and root["black_lead"] > 0  # Black (first player) is ahead


def test_raw_nn_with_draw_mass():
    e = ev.from_raw_nn({"whiteWin": 0.5, "whiteLoss": 0.3, "noResult": 0.2, "whiteLead": 1.0})
    assert e["black_win"] == pytest.approx(0.3)
    assert e["white_win"] == pytest.approx(0.5)


def test_final_eval():
    b = ev.final_eval("b", 3)
    assert (b["black_win"], b["white_win"], b["black_lead"], b["source"]) == (1.0, 0.0, 3.0, "final")
    w = ev.final_eval("w", 2)
    assert (w["black_win"], w["white_win"], w["black_lead"]) == (0.0, 1.0, -2.0)


def test_parse_analysis():
    a = ev.parse_analysis(GENMOVE_W)
    assert a["move"] == "e2"
    assert [d["move"] for d in a["infos"]] == ["e2", "h2h"]
    top = a["infos"][0]
    assert top["visits"] == 59 and top["winrate"] == pytest.approx(0.400073)
    assert top["pv"] == ["e2", "e7", "e3", "e6", "e4", "h6h", "d3h", "f6h", "e5v", "d6h"]
    assert a["root"]["visits"] == 64
    assert ev.info_for(a, "h2h")["pv"] == ["h2h"]
    assert ev.info_for(a, "a1h") is None


def test_parse_analysis_one_visit_has_no_info():
    a = ev.parse_analysis("play e3")
    assert a == {"move": "e3", "infos": [], "root": None}


def test_parse_analysis_uses_last_info_line():
    text = ("info move e7 visits 3 winrate 0.5 scoreLead 0 order 0 pv e7\n"
            "info move e7 visits 90 winrate 0.6 scoreLead 1 order 0 pv e7 e3\nplay e7")
    a = ev.parse_analysis(text)
    assert a["infos"][0]["visits"] == 90


def test_parse_raw_nn_missing_key():
    with pytest.raises(ValueError):
        ev.parse_raw_nn("symmetry 0 whiteWin 0.5")


def test_candidates_symmetry_pruning():
    text = (
        "info move f8 visits 50 winrate 0.6 scoreLead 1.2 order 0 pv f8 e2 "
        "info move d8 visits 50 isSymmetryOf f8 winrate 0.6 scoreLead 1.2 order 1 pv d8 e2 "
        "info move e7 visits 20 winrate 0.4 scoreLead -0.5 order 2 pv e7 e2\n"
        "play f8"
    )
    a = ev.parse_analysis(text)
    # Default: deduplicate symmetric moves and re-index order
    cands = ev.candidates(a, "b")
    assert len(cands) == 2
    assert [c["move"] for c in cands] == ["f8", "e7"]
    assert [c["order"] for c in cands] == [0, 1]
    assert cands[0]["is_symmetry_of"] is None

    # include_symmetric=True retains symmetric moves
    all_cands = ev.candidates(a, "b", include_symmetric=True)
    assert len(all_cands) == 3
    assert [c["move"] for c in all_cands] == ["f8", "d8", "e7"]
    assert all_cands[1]["is_symmetry_of"] == "f8"
    assert all_cands[1]["order"] == 1
