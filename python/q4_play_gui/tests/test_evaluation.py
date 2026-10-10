import unittest

from helpers import PYTHON_DIR  # noqa: F401  (import path)
from q4_play_gui import evaluation as ev


class ParseInfo(unittest.TestCase):
    def test_root_line(self):
        r = ev.parse_info("info root visits 4714 value 0.2812,0.3013,0.2465,0.1697,0.0013 "
                          "utility -0.4720,-0.3797,-0.5349,-0.7482 lead -0.4430,-0.1741,-2.3581,-3.9809")
        self.assertEqual(r["kind"], "root")
        self.assertEqual(r["visits"], 4714)
        self.assertEqual(len(r["value"]), 5)
        self.assertAlmostEqual(r["value"][4], 0.0013)
        self.assertEqual(len(r["lead"]), 4)

    def test_move_line_with_pv(self):
        m = ev.parse_info("info move e5h visits 12 prior 0.0412 utility -0.6259 lcb -2.0337 "
                          "value 0.2206,0.3147,0.2680,0.1967,0.0000 pv S e5h W f6v N j1")
        self.assertEqual(m["kind"], "move")
        self.assertEqual(m["move"], "e5h")
        self.assertEqual(m["visits"], 12)
        self.assertAlmostEqual(m["prior"], 0.0412)
        self.assertEqual(m["pv"], [("S", "e5h"), ("W", "f6v"), ("N", "j1")])

    def test_other_lines_are_ignored(self):
        self.assertIsNone(ev.parse_info("= done"))
        self.assertIsNone(ev.parse_info("play cancelled"))
        self.assertIsNone(ev.parse_info("info foo"))

    def test_malformed_values_raise(self):
        with self.assertRaises(ValueError):
            ev.parse_info("info root visits 3 value 0.1,0.2")


class ParseRawnn(unittest.TestCase):
    TEXT = """Symmetry 0, seat 3 (North) to move
Win probability by absolute seat:
  seat 1 (South):  19.69%
  seat 2 (West ):  27.35%
  seat 3 (North):  28.45%
  seat 4 (East ):  24.51%
  draw          :   0.00%
Plies to the end of the game: 120.0
Top legal moves of the search policy (203 legal):
  f10    33.00%
  e2h    12.73%
"""

    def test_values_plies_policy(self):
        r = ev.parse_rawnn(self.TEXT)
        self.assertAlmostEqual(r["value"][0], 0.1969)
        self.assertAlmostEqual(r["value"][3], 0.2451)
        self.assertAlmostEqual(sum(r["value"]), 1.0, places=3)
        self.assertEqual(r["plies"], 120.0)
        self.assertEqual(r["policy"][0], ("f10", 33.0))
        self.assertEqual(r["policy"][1], ("e2h", 12.73))


class MoveList(unittest.TestCase):
    def test_moves_eliminations_and_numbers(self):
        self.assertEqual(ev.parse_move_list("S f2, W b6, EL N, E j10v"), [
            {"seat": 0, "a": "f2"}, {"seat": 1, "a": "b6"}, {"elim": 2}, {"seat": 3, "a": "j10v"}])

    def test_numbered_lines(self):
        self.assertEqual(ev.parse_move_list("1. S f2\n2. W b6\n3) EL E"), [
            {"seat": 0, "a": "f2"}, {"seat": 1, "a": "b6"}, {"elim": 3}])

    def test_bad_tokens_name_the_token(self):
        with self.assertRaises(ValueError) as cm:
            ev.parse_move_list("S f2, W z9")
        self.assertIn("z9", str(cm.exception))
        with self.assertRaises(ValueError):
            ev.parse_move_list("EL")

    def test_seats(self):
        self.assertEqual(ev.seat_from_letter("n"), 2)
        self.assertEqual(ev.seat_from_letter("4"), 3)
        with self.assertRaises(ValueError):
            ev.seat_from_letter("X")


if __name__ == "__main__":
    unittest.main()
