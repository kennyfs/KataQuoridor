"""The controller on the mock engine: eliminate / undo / redo flows, seat settings, the move list, SGF round trip,
malformed input, engine death and restart."""
import os
import unittest

from helpers import close_game, mock_game
from q4_play_gui.game import ActionError, EngineDown


def alive(game):
    return game.state()["position"]["alive"]


def to_move(game):
    return game.state()["position"]["toMove"]


class EliminateUndo(unittest.TestCase):
    def setUp(self):
        self.g = mock_game()

    def tearDown(self):
        close_game(self.g)

    def test_eliminate_the_seat_to_move_then_undo_brings_it_back(self):
        g = self.g
        self.assertEqual(to_move(g), 0)
        g.eliminate(0)
        self.assertEqual(alive(g), [False, True, True, True])
        self.assertEqual(to_move(g), 1)
        g.undo()
        self.assertEqual(alive(g), [True, True, True, True])
        self.assertEqual(to_move(g), 0)
        self.assertEqual(g.state()["position"]["pawns"][0], "f1")

    def test_two_eliminations_undo_twice(self):
        g = self.g
        g.eliminate(1)
        g.eliminate(3)
        self.assertEqual(alive(g), [True, False, True, False])
        g.undo()
        self.assertEqual(alive(g), [True, False, True, True])
        g.undo()
        self.assertEqual(alive(g), [True, True, True, True])
        self.assertEqual(g.state()["cursor"], 0)

    def test_eliminate_play_undo_undo(self):
        g = self.g
        g.eliminate(0)                    # S out, W to move
        g.play("b6")                      # W plays
        self.assertEqual(to_move(g), 2)
        g.undo()                          # the move
        self.assertEqual(to_move(g), 1)
        self.assertEqual(alive(g), [False, True, True, True])
        g.undo()                          # the elimination
        self.assertEqual(alive(g), [True, True, True, True])
        self.assertEqual(to_move(g), 0)

    def test_redo_replays_moves_and_eliminations(self):
        g = self.g
        g.play("f2")
        g.eliminate(1)
        g.undo()
        g.undo()
        self.assertEqual(g.state()["cursor"], 0)
        g.redo()
        self.assertEqual(g.state()["position"]["pawns"][0], "f2")
        g.redo()
        self.assertEqual(alive(g), [True, False, True, True])
        with self.assertRaises(ActionError):
            g.redo()

    def test_playing_a_move_drops_the_redo_line(self):
        g = self.g
        g.play("f2")
        g.play("b6")
        g.undo()
        g.play("a7")                      # a different move at ply 2
        st = g.state()
        self.assertEqual(st["length"], 2)
        self.assertEqual([e["a"] for e in st["events"]], ["f2", "a7"])

    def test_undo_at_the_start_is_an_error(self):
        with self.assertRaises(ActionError):
            self.g.undo()

    def test_last_but_one_seat_needs_confirmation_and_ends_the_game(self):
        g = self.g
        g.eliminate(1)
        g.eliminate(3)
        with self.assertRaises(ActionError) as cm:
            g.eliminate(2)                # leaves one seat
        self.assertIn("Confirm", str(cm.exception))
        self.assertEqual(alive(g), [True, False, True, False])
        g.eliminate(2, confirmed=True)
        st = g.state()
        self.assertTrue(st["position"]["finished"])
        self.assertEqual(st["position"]["winner"], 0)
        with self.assertRaises(ActionError):
            g.play("f2")
        g.undo()                          # undo reopens the game
        self.assertFalse(g.state()["position"]["finished"])

    def test_eliminating_a_dead_seat_is_an_error(self):
        self.g.eliminate(1)
        with self.assertRaises(ActionError):
            self.g.eliminate(1)


class MovesAndReview(unittest.TestCase):
    def setUp(self):
        self.g = mock_game()

    def tearDown(self):
        close_game(self.g)

    def test_out_of_turn_and_illegal_moves_leave_the_state_alone(self):
        g = self.g
        before = g.state()["cursor"]
        with self.assertRaises(ActionError):
            g.play("zz")                  # malformed
        with self.assertRaises(ActionError):
            g.play("a1")                  # not a legal pawn move of seat S
        self.assertEqual(g.state()["cursor"], before)
        self.assertEqual(g.state()["engine"]["status"], "ready")

    def test_goto_reviews_earlier_positions_and_playing_there_branches(self):
        g = self.g
        g.play("f2")
        g.play("b6")
        g.play("f10")
        g.goto(1)
        self.assertEqual(g.state()["cursor"], 1)
        self.assertEqual(g.state()["position"]["pawns"][0], "f2")
        g.goto(3)
        self.assertEqual(g.state()["position"]["pawns"][2], "f10")
        g.goto(1)
        other = g.state()["position"]["legal"][-1]   # a different move for W at ply 1
        g.play(other)
        st = g.state()
        self.assertEqual(st["length"], 2)            # the old ply-2 and ply-3 moves are gone

    def test_goto_out_of_range(self):
        with self.assertRaises(ActionError):
            self.g.goto(5)
        with self.assertRaises(ActionError):
            self.g.goto("x")

    def test_the_seat_to_move_has_its_legal_moves(self):
        legal = self.g.state()["position"]["legal"]
        self.assertIn("f2", legal)
        self.assertNotIn("f11", legal)

    def test_legal_for_another_seat(self):
        self.assertIn("f10", self.g.legal_for(2))


class SeatsAndSettings(unittest.TestCase):
    def setUp(self):
        self.g = mock_game()

    def tearDown(self):
        close_game(self.g)

    def test_seats_change_mid_game(self):
        g = self.g
        g.play("f2")                      # W is to move
        g.set_seats([{"kind": "human"}, {"kind": "bot", "bot": "greedy"}, {"kind": "net"}, {"kind": "human"}])
        self.assertEqual(g.state()["mode"], "play")
        g.ai_move()                       # W is a bot now
        self.assertEqual(g.state()["events"][-1]["seat"], 1)
        self.assertTrue(g.state()["events"][-1]["ai"])
        g.ai_move()                       # N is a net
        self.assertEqual(g.state()["events"][-1]["seat"], 2)
        self.assertEqual(to_move(g), 3)   # E is human: a single AI call is refused
        with self.assertRaises(ActionError):
            g.ai_move()

    def test_ai_all_stops_at_a_human_seat(self):
        g = self.g
        g.set_seats([{"kind": "net"}, {"kind": "bot", "bot": "basher"}, {"kind": "net"}, {"kind": "human"}])
        r = g.ai_move(all_seats=True)
        self.assertEqual(r["moves"], 3)
        self.assertEqual(to_move(g), 3)
        self.assertEqual(g.ai_move(all_seats=True)["moves"], 0)

    def test_grudge_needs_a_target(self):
        with self.assertRaises(ActionError):
            self.g.set_seats([{"kind": "bot", "bot": "grudge"}, {}, {}, {}])
        self.g.set_seats([{"kind": "bot", "bot": "grudge", "target": 2}, {}, {}, {}])

    def test_bad_seat_kinds(self):
        with self.assertRaises(ActionError):
            self.g.set_seats([{"kind": "robot"}, {}, {}, {}])
        with self.assertRaises(ActionError):
            self.g.set_seats([{"kind": "bot", "bot": "nope"}, {}, {}, {}])

    def test_settings_go_to_the_engine(self):
        g = self.g
        g.set_settings({"maxVisits": 123, "opponentMode": "style", "q4LeadWeightEps": 0.1})
        st = g.state()["settings"]
        self.assertEqual(st["maxVisits"], 123)
        self.assertEqual(st["opponentMode"], "style")
        with self.assertRaises(ActionError):
            g.set_settings({"opponentMode": "wild"})
        with self.assertRaises(ActionError):
            g.set_settings({"unknown": 1})

    def test_own_seat_follows_the_seats(self):
        g = self.g
        g.set_seats([{"kind": "human"}, {"kind": "net"}, {"kind": "human"}, {"kind": "net"}], me=2)
        self.assertEqual(g.state()["me"], 2)
        g.set_seats([{"kind": "human"}] * 4)        # me stays
        self.assertEqual(g.state()["me"], 2)

    def test_new_game_rules_reach_the_engine(self):
        g = self.g
        g.new_game({"maxPlies": 2, "repetitionDrawCount": 0, "initialWalls": [5, 5, 5, 5]})
        self.assertEqual(g.state()["rules"]["maxPlies"], 2)
        self.assertEqual(g.state()["position"]["wallsLeft"], [5, 5, 5, 5])
        g.play("f2")
        g.play("b6")
        self.assertTrue(g.state()["position"]["finished"])   # the mock ends at maxPlies


class RecordsAndLoading(unittest.TestCase):
    def setUp(self):
        self.g = mock_game()

    def tearDown(self):
        close_game(self.g)

    def test_sgf_round_trip_keeps_the_moves_and_eliminations(self):
        from q4.record import Q4Record   # python/q4/record.py: the reader of the .sgfs format
        g = self.g
        g.play("f2")
        g.eliminate(1)
        g.play("g11")
        text = g.sgf()
        rec = Q4Record.from_sgf_line(text)
        self.assertEqual(rec.events, [{"a": "f2"}, {"elim": 2}, {"a": "g11"}])
        g.load(text)
        st = g.state()
        self.assertEqual(st["length"], 3)
        self.assertEqual([e.get("a", "EL%d" % e["seat"]) for e in st["events"]], ["f2", "EL1", "g11"])

    def test_sgf_root_names_the_seat_kinds(self):
        g = self.g
        g.set_seats([{"kind": "human"}, {"kind": "net"}, {"kind": "bot", "bot": "basher"}, {"kind": "human"}])
        text = g.sgf()
        self.assertIn("type0=human", text)
        self.assertIn("type1=search", text)
        self.assertIn("type2=basher", text)

    def test_sgf_export_has_comments_for_analysed_positions(self):
        g = self.g
        g.set_analysis(on=True, cs=2, cap=200)
        import time
        time.sleep(0.6)
        g.set_analysis(on=False)
        g.play("f2")
        self.assertIn("C[", g.sgf())

    def test_move_list_load_checks_the_seat_to_move(self):
        g = self.g
        g.load("S f2, W b6, EL N")
        self.assertEqual(g.state()["length"], 3)
        with self.assertRaises(ActionError) as cm:
            g.load("S f2, S b6")
        self.assertIn("event 2", str(cm.exception))

    def test_load_leaves_the_cursor_at_a_ply(self):
        g = self.g
        g.load("S f2, W b6, N f10", ply=1)
        st = g.state()
        self.assertEqual(st["cursor"], 1)
        self.assertEqual(st["length"], 3)

    def test_malformed_records_are_errors(self):
        with self.assertRaises(ActionError):
            self.g.load("(;FF[4]GM[1]SZ[11])")
        with self.assertRaises(ActionError):
            self.g.load("S f2, nonsense")
        with self.assertRaises(ActionError):
            self.g.load("   ")


class EngineDeath(unittest.TestCase):
    def test_a_crash_is_reported_and_restart_replays_the_game(self):
        os.environ["MOCK_CRASH_ON"] = "eliminate"
        try:
            g = mock_game()
        finally:
            del os.environ["MOCK_CRASH_ON"]
        try:
            g.play("f2")
            g.play("b6")
            with self.assertRaises(EngineDown):
                g.eliminate(2)
            st = g.state()
            self.assertEqual(st["engine"]["status"], "down")
            self.assertEqual(st["cursor"], 2)            # the controller kept its own record
            os.environ.pop("MOCK_CRASH_ON", None)
            g.restart()
            st = g.state()
            self.assertEqual(st["engine"]["status"], "ready")
            self.assertEqual(st["engine"]["restarts"], 1)
            self.assertEqual(st["position"]["pawns"][0], "f2")
            self.assertEqual(st["position"]["pawns"][1], "b6")
            g.undo()
            self.assertEqual(g.state()["cursor"], 1)
        finally:
            close_game(g)

    def test_requests_fail_fast_while_the_engine_is_down(self):
        os.environ["MOCK_CRASH_ON"] = "play"
        try:
            g = mock_game()
        finally:
            del os.environ["MOCK_CRASH_ON"]
        try:
            with self.assertRaises(EngineDown):
                g.play("f2")
            with self.assertRaises(EngineDown):
                g.play("f2")
        finally:
            close_game(g)


if __name__ == "__main__":
    unittest.main()
