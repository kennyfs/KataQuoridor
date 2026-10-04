"""Controller tests with the mock QTP engine (python/quoridor_arena/tests/mock_engine.py).

The mock plays a toy race: each pawn moves straight towards its goal row or sideways, walls a1h..c1h do
nothing, and its genmove always goes straight ahead. Black (e9 -> row 1) needs 8 moves.
"""
import os
import random
import re
import sys
import time

import pytest

from play_gui.game import ActionError, EngineDown, GameController, decorate_sgf

MOCK = os.path.join(os.path.dirname(__file__), "..", "..", "quoridor_arena", "tests", "mock_engine.py")


def make_game(tmp_path, *flags, seed=0):
    g = GameController([sys.executable, MOCK] + list(flags), str(tmp_path), model_name="mocknet",
                       move_timeout=20, start_timeout=20, hint_visits=50, demo_delay=0.0, rng=random.Random(seed))
    g.start_engine()
    return g


@pytest.fixture
def game(tmp_path):
    g = make_game(tmp_path)
    yield g
    g.close()


def wait_idle(g, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = g.state()
        if not s["thinking"] and not s["busy"]:
            return s
        time.sleep(0.02)
    raise AssertionError("AI still thinking after %.0fs" % timeout)


def moves_of(s):
    return [(m["color"], m["move"]) for m in s["moves"]]


def test_engine_ready(game):
    s = game.state()
    assert s["engine"]["status"] == "ready"
    assert s["engine"]["name"] == "0.0 (mock)"
    assert not s["started"]


def test_new_game_as_black(game):
    s = game.new_game("b", 16)
    assert s["settings"]["human"] == "b" and s["settings"]["visits"] == 16
    assert s["to_move"] == "b" and not s["thinking"]
    assert s["moves"] == []
    assert s["pawns"] == {"b": "e9", "w": "e1"}
    assert s["walls_left"] == {"b": 10, "w": 10}
    assert s["dist"] == {"b": 8, "w": 8}
    assert set(s["legal_moves"]) == {"e8", "d9", "f9", "a1h", "b1h", "c1h"}
    assert s["eval"]["source"] == "net"
    assert s["eval"]["black_win"] + s["eval"]["white_win"] == pytest.approx(1.0)


def test_new_game_as_white_ai_moves_first(game):
    game.new_game("w", 16)
    s = wait_idle(game)
    assert moves_of(s) == [("b", "e8")]
    assert s["to_move"] == "w"
    assert s["pawns"]["b"] == "e8"
    # The AI's search result is used for both positions: the root before its move, the move after it.
    assert s["positions"][0]["eval"]["source"] == "search"
    assert s["positions"][1]["eval"]["source"] == "search"
    assert s["positions"][1]["eval"]["visits"] == 16


def test_new_game_random_side(tmp_path):
    seen = set()
    g = make_game(tmp_path, seed=1)
    try:
        for _ in range(12):
            g.new_game("random", 1)
            s = wait_idle(g)
            seen.add(s["settings"]["human"])
            assert s["settings"]["human_choice"] == "random"
            assert len(s["moves"]) == (0 if s["settings"]["human"] == "b" else 1)
    finally:
        g.close()
    assert seen == {"b", "w"}


def test_new_game_bad_arguments(game):
    with pytest.raises(ActionError):
        game.new_game("x", 16)
    with pytest.raises(ActionError):
        game.new_game("b", 0)
    with pytest.raises(ActionError):
        game.new_game("b", "lots")


def test_human_move_then_ai_reply(game):
    game.new_game("b", 16)
    s = game.play("e8")
    assert moves_of(s)[0] == ("b", "e8")
    s = wait_idle(game)
    assert moves_of(s) == [("b", "e8"), ("w", "e2")]
    assert s["to_move"] == "b"
    assert s["pawns"] == {"b": "e8", "w": "e2"}
    assert s["dist"] == {"b": 7, "w": 7}
    ev_after_ai = s["positions"][2]["eval"]
    assert ev_after_ai["source"] == "search"
    assert ev_after_ai["pv"] == ["e7"]    # the continuation after the AI's move (its own move is dropped)
    # the root search of the AI replaced the net-only estimate of the position after the human move
    assert s["positions"][1]["eval"]["source"] == "search"


def test_one_visit_uses_net_eval(game):
    game.new_game("b", 1)
    game.play("e8")
    s = wait_idle(game)
    assert len(s["moves"]) == 2
    assert s["positions"][1]["eval"]["source"] == "net"
    assert s["positions"][2]["eval"]["source"] == "net"


def test_walls_are_tracked(game):
    game.new_game("b", 16)
    game.play("a1h")
    s = wait_idle(game)
    assert s["walls"] == [{"move": "a1h", "color": "b", "ply": 1}]
    assert s["walls_left"] == {"b": 9, "w": 10}
    assert s["pawns"]["b"] == "e9"


def test_reject_bad_notation(game):
    game.new_game("b", 16)
    for bad in ("", "z9", "e10", "i8h", "e8 \nclear_board", "e8x"):
        with pytest.raises(ActionError) as ei:
            game.play(bad)
        assert ei.value.status == 400
    assert game.state()["moves"] == []


def test_reject_illegal_move(game):
    game.new_game("b", 16)
    with pytest.raises(ActionError) as ei:
        game.play("e5")
    assert ei.value.status == 400 and "illegal" in str(ei.value)


def test_reject_out_of_turn(game):
    game._spawn_ai = lambda *a, **k: None   # keep the AI from moving
    game.new_game("w", 1)
    with game.lock:
        game.thinking = False
    with pytest.raises(ActionError, match="not your turn"):
        game.play("e8")
    assert game.state()["moves"] == []


def test_reject_while_thinking(tmp_path):
    g = make_game(tmp_path, "--think-delay", "0.6")
    try:
        g.new_game("b", 16)
        g.play("e8")
        s = g.state()
        assert s["thinking"] and s["to_move"] == "w"
        with pytest.raises(ActionError, match="thinking"):
            g.play("e7")
        with pytest.raises(ActionError, match="thinking"):
            g.undo()
        with pytest.raises(ActionError):
            g.request_hint()
        with pytest.raises(ActionError):
            g.sgf()
        s = wait_idle(g)
        assert len(s["moves"]) == 2
    finally:
        g.close()


def test_new_game_while_thinking_discards_the_search(tmp_path):
    g = make_game(tmp_path, "--think-delay", "0.4")
    try:
        g.new_game("b", 16)
        g.play("e8")
        assert g.state()["thinking"]
        s = g.new_game("b", 16)
        assert s["moves"] == [] and not s["thinking"]
        time.sleep(0.2)
        s = g.state()
        assert s["moves"] == [] and s["legal_moves"]
    finally:
        g.close()


def test_undo_takes_back_human_move_and_ai_reply(game):
    game.new_game("b", 16)
    game.play("e8")
    wait_idle(game)
    game.play("e7")
    wait_idle(game)
    s = game.undo()
    assert moves_of(s) == [("b", "e8"), ("w", "e2")]
    assert s["to_move"] == "b" and s["pawns"] == {"b": "e8", "w": "e2"}
    assert len(s["positions"]) == 3
    assert "e7" in s["legal_moves"]
    s = game.undo()
    assert s["moves"] == [] and s["pawns"] == {"b": "e9", "w": "e1"}
    with pytest.raises(ActionError):
        game.undo()


def test_undo_as_white_keeps_ai_first_move(game):
    game.new_game("w", 16)
    wait_idle(game)
    with pytest.raises(ActionError):
        game.undo()                   # only the AI's opening move: nothing of the human's to take back
    game.play("d1")
    wait_idle(game)
    s = game.undo()
    assert moves_of(s) == [("b", "e8")] and s["to_move"] == "w"


def test_human_wins_game_over_and_undo(game):
    game.new_game("b", 16)
    for row in range(8, 0, -1):
        s = game.play("e%d" % row)
        s = wait_idle(game)
    assert s["winner"] == "b"
    assert len(s["moves"]) == 15      # Black's 8th move wins; the AI does not move after it
    assert s["margin"] == 0.5         # White stands on e8, one step from row 9: B+0.5 (komi -0.5)
    assert s["legal_moves"] == [] and s["to_move"] is None
    assert s["eval"]["source"] == "final" and s["eval"]["black_win"] == 1.0 and s["eval"]["black_lead"] == 0.5
    with pytest.raises(ActionError, match="over"):
        game.play("e2")
    with pytest.raises(ActionError):
        game.request_hint()
    # Undo after a winning human move takes back just that move.
    s = game.undo()
    assert len(s["moves"]) == 14 and s["winner"] is None and s["to_move"] == "b"
    assert "e1" in s["legal_moves"]
    assert game.engine.alive()        # kata-raw-nn was never sent on the finished position


def test_ai_wins_and_undo_takes_two(game):
    game.new_game("w", 16)
    wait_idle(game)
    side = ["d1", "e1"]
    for i in range(20):
        s = game.state()
        if s["winner"]:
            break
        game.play(side[i % 2])
        s = wait_idle(game)
    assert s["winner"] == "b" and len(s["moves"]) == 15
    assert s["margin"] == 7.5         # White never left row 1: 8 steps, B+7.5
    assert s["eval"]["black_win"] == 1.0
    s = game.undo()
    assert len(s["moves"]) == 13 and s["to_move"] == "w" and s["winner"] is None
    assert game.engine.alive()


def test_hint(game):
    game.new_game("b", 1)
    h = game.request_hint()
    assert h["ply"] == 0 and h["color"] == "b"
    assert [m["move"] for m in h["moves"]] == ["e8", "d9", "f9"]
    assert h["visits"] == 50
    for m in h["moves"]:
        assert 0 <= m["black_win"] <= 1 and m["pv"][0] == m["move"]
    s = game.state()
    assert s["hint"]["ply"] == 0 and s["settings"]["visits"] == 1
    # maxVisits was restored: the AI reply is a 1-visit search again (net-only eval)
    game.play("e8")
    s = wait_idle(game)
    assert s["hint"] is None
    assert s["positions"][2]["eval"]["source"] == "net"


def test_sgf(game):
    game.new_game("b", 16)
    game.play("e8")
    wait_idle(game)
    sgf = game.sgf()
    assert "PB[Human]" in sgf and "PW[KataQuoridor mocknet (16 visits)]" in sgf
    comments = re.findall(r";([BW])\[([^\]]*)\]C\[([^\]]*)\]", sgf)
    assert [c[:2] for c in comments] == [("B", "e8"), ("W", "e2")]
    # comment i = eval of the position before move i, White's perspective: whiteWin whiteLoss noResult score
    s = game.state()
    first = comments[0][2].split()
    e0 = s["positions"][0]["eval"]
    assert float(first[0]) == pytest.approx(e0["white_win"], abs=0.006)
    assert float(first[1]) == pytest.approx(e0["black_win"], abs=0.006)
    assert float(first[3]) == pytest.approx(-e0["black_lead"], abs=0.06)
    assert "v=16" in comments[1][2]


def test_decorate_sgf_keeps_result_comment():
    positions = [{"eval": {"white_win": 0.4, "black_win": 0.6, "black_lead": 0.8, "visits": None}},
                 {"eval": None}]
    out = decorate_sgf("(;FF[4]PB[]PW[]RE[B+1];B[io];W[ic]C[result=B+1])", positions, "Human", "Bot")
    assert out == "(;FF[4]PB[Human]PW[Bot]RE[B+1];B[io]C[0.40 0.60 0.00 -0.8];W[ic]C[result=B+1])"


def test_crash_and_restart_replays_the_game(tmp_path):
    g = make_game(tmp_path, "--crash-after", "2")
    try:
        g.new_game("b", 16)
        g.play("e8")
        s = wait_idle(g)
        assert len(s["moves"]) == 2
        g.play("e7")
        s = wait_idle(g)
        assert s["engine"]["status"] == "error" and s["engine"]["error"]
        assert len(s["moves"]) == 3 and s["to_move"] == "w"
        with pytest.raises(ActionError) as ei:
            g.play("e6")
        assert ei.value.status == 503
        # The restart replays the 3 moves into a fresh process, whose first genmove succeeds.
        g.start_engine()
        s = wait_idle(g)
        assert s["engine"]["status"] == "ready"
        assert moves_of(s) == [("b", "e8"), ("w", "e2"), ("b", "e7"), ("w", "e3")]
    finally:
        g.close()


def test_engine_missing(tmp_path):
    g = GameController([str(tmp_path / "no-such-katago")], str(tmp_path), start_timeout=5)
    with pytest.raises(EngineDown):
        g.start_engine()
    s = g.state()
    assert s["engine"]["status"] == "error" and "no-such-katago" in s["engine"]["error"]
    with pytest.raises(ActionError) as ei:
        g.new_game("b", 16)
    assert ei.value.status == 503
    g.close()


def test_ai_vs_ai_demo(game):
    game.new_game("none", 16)
    s = wait_idle(game, timeout=20)
    assert s["winner"] == "b" and len(s["moves"]) == 15
    with pytest.raises(ActionError):
        game.undo()


# -- analysis mode ------------------------------------------------------------------------------------------
def wait_for(pred, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if pred():
            return
        time.sleep(0.02)
    raise AssertionError("condition not met within %.0fs" % timeout)


def analysis_visits(g):
    cur = g.state()["analysis"]["current"]
    return cur["root"]["visits"] if cur and cur["root"] else 0


def test_analysis_mode_plays_both_sides_and_undo_redo(game):
    s = game.new_game("analysis", 16)
    assert s["mode"] == "analysis" and s["settings"]["human"] is None and not s["thinking"]
    game.play("e8")
    s = game.play("e2")
    assert moves_of(s) == [("b", "e8"), ("w", "e2")] and not s["thinking"]
    s = game.undo()
    assert moves_of(s) == [("b", "e8")] and s["redo"] == [{"color": "w", "move": "e2"}]
    s = game.goto(0)
    assert s["moves"] == [] and [m["move"] for m in s["redo"]] == ["e8", "e2"]
    assert s["pawns"] == {"b": "e9", "w": "e1"}
    s = game.goto(2)
    assert moves_of(s) == [("b", "e8"), ("w", "e2")] and s["redo"] == []
    assert s["pawns"] == {"b": "e8", "w": "e2"}
    game.undo()
    s = game.play("d1")      # a different move drops the redo line
    assert moves_of(s) == [("b", "e8"), ("w", "d1")] and s["redo"] == []
    game.goto(1)
    s = game.play("d1")      # replaying the redo move keeps the rest of the line
    assert s["redo"] == []
    with pytest.raises(ActionError):
        game.goto(5)


def test_analysis_streams_and_stops(game):
    game.new_game("analysis", 16)
    s = game.set_analysis(on=True, max_visits=0)
    assert s["analysis"]["on"]
    wait_for(lambda: analysis_visits(game) >= 8)
    s = game.state()
    cur = s["analysis"]["current"]
    assert s["analysis"]["running"] and cur["ply"] == 0 and cur["color"] == "b"
    assert cur["moves"][0]["move"] == "e8" and cur["moves"][0]["prior"] == 0.5
    assert s["positions"][0]["eval"]["source"] == "search"
    assert s["positions"][0]["net"]["source"] == "net"
    # commands still work while it streams; the analysis follows the position
    s = game.play("e8")
    wait_for(lambda: (game.state()["analysis"]["current"] or {}).get("color") == "w" and analysis_visits(game) > 0)
    assert game.state()["analysis"]["current"]["ply"] == 1
    assert game.legal  # engine still answers normally
    game.set_analysis(on=False)
    s = game.state()
    assert not s["analysis"]["running"]
    assert game._must("dist")  # the stream is closed: plain commands get their own answers


def test_analysis_visit_cap(game):
    game.new_game("analysis", 16)
    game.set_analysis(on=True, max_visits=16)
    wait_for(lambda: not game.state()["analysis"]["running"] and analysis_visits(game) >= 16)
    v = analysis_visits(game)
    time.sleep(0.3)
    assert analysis_visits(game) == v
    game.set_analysis(max_visits=0)            # raising the cap resumes it
    wait_for(lambda: analysis_visits(game) > v)
    game.set_analysis(on=False)


def test_enter_analysis_from_a_game(tmp_path):
    g = make_game(tmp_path, "--think-delay", "0.3")
    try:
        g.new_game("b", 16)
        g.play("e8")                           # the AI starts thinking
        s = g.enter_analysis()                 # the running search still plays its move
        assert s["mode"] == "analysis" and not s["thinking"]
        assert moves_of(s) == [("b", "e8"), ("w", "e2")]
        s = g.play("e7")
        assert moves_of(s)[-1] == ("b", "e7") and not s["thinking"]
        s = g.play("e3")                       # the user now plays White too
        assert moves_of(s)[-1] == ("w", "e3")
    finally:
        g.close()


def test_hint_has_candidate_details(game):
    game.new_game("b", 16)
    h = game.request_hint()
    m = h["moves"][0]
    assert m["chosen"] and m["visits"] > 0 and m["prior"] == 0.5 and m["order"] == 0
    assert 0 <= m["winrate"] <= 1 and h["root"]["visits"] == 50


def test_analysis_ai_sides(game):
    game.new_game("analysis", 16)
    s = game.set_ai_sides("w")
    assert s["analysis"]["ai_sides"] == "w" and not s["thinking"]   # Black to move: still the user
    with pytest.raises(ActionError):
        game.play("e2")                                              # not White's turn
    game.play("e8")
    s = wait_idle(game)
    assert moves_of(s) == [("b", "e8"), ("w", "e2")]                 # the AI answered for White
    s = game.set_ai_sides("bw")                                      # both: it plays on by itself
    wait_for(lambda: game.state()["winner"] is not None, timeout=20)
    s = wait_idle(game)
    assert s["mode"] == "analysis" and s["winner"] == "b"
    game.undo()
    s = game.set_ai_sides("")
    assert s["analysis"]["ai_sides"] == "" and not s["thinking"]
    game.new_game("b", 16)
    with pytest.raises(ActionError):
        game.set_ai_sides("b")                                       # only in analysis mode
