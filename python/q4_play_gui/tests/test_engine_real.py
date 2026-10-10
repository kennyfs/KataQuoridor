"""The new and the used q4qtp commands against the real engine binary (skipped unless the environment names one).

  Q4_KATAGO=/path/to/katago Q4_MODEL=/path/to/model.bin.gz python3 -m unittest test_engine_real

A small net is enough (the tests check protocol and state, not play strength); the searches are kept short.
"""
import json
import os
import queue
import subprocess
import tempfile
import threading
import time
import unittest

from helpers import PYTHON_DIR  # noqa: F401  (import path)
from q4_play_gui.serve import config_with_threads

KATAGO = os.environ.get("Q4_KATAGO", "")
MODEL = os.environ.get("Q4_MODEL", "")
QTP_CFG = os.path.join(PYTHON_DIR, "..", "cpp", "configs", "q4", "q4qtp.cfg")


class Qtp:
    """A raw q4qtp process: lines in, (ok, text) responses out, and the info lines of an analysis apart."""

    def __init__(self):
        out_dir = tempfile.mkdtemp(prefix="q4qtp_test_")
        cfg = config_with_threads(os.path.abspath(QTP_CFG), 2, out_dir)
        self.proc = subprocess.Popen([KATAGO, "q4qtp", "-config", cfg, "-model", MODEL, "-bot", "search"],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lines = queue.Queue()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            self.lines.put(line.rstrip("\n"))
        self.lines.put(None)

    def write(self, text):
        self.proc.stdin.write(text + "\n")
        self.proc.stdin.flush()

    def response(self, timeout=120):
        """(ok, text, info lines) of the next response: the "=" / "?" header, then lines up to an empty line.
        Info lines (an analysis stream) are returned apart; the other lines make the text."""
        header = None
        while header is None:
            line = self.lines.get(timeout=timeout)
            if line is None:
                raise RuntimeError("engine exited")
            if line.startswith("=") or line.startswith("?"):
                header = line
        infos, body = [], []
        while True:
            line = self.lines.get(timeout=timeout)
            if line is None:
                raise RuntimeError("engine exited")
            if line.strip() == "":
                break
            if line.startswith("info "):
                infos.append(line)
            else:
                body.append(line)
        rest = header[1:].strip()
        text = "\n".join(([rest] if rest else []) + body).strip()
        return header[0] == "=", text, infos

    def cmd(self, text, timeout=120):
        self.write(text)
        ok, body, _ = self.response(timeout)
        return ok, body

    def must(self, text):
        ok, body = self.cmd(text)
        if not ok:
            raise AssertionError("%s failed: %s" % (text, body))
        return body

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=20)
        except (subprocess.TimeoutExpired, OSError):
            self.proc.kill()
        finally:
            self.proc.stdout.close()


def snapshot(q):
    return {
        "pos": json.loads(q.must("q4-position")),
        "walls": q.must("walls"),
        "legal": q.must("legal_moves"),
        "to_move": q.must("to_move"),
    }


@unittest.skipUnless(KATAGO and MODEL, "set Q4_KATAGO and Q4_MODEL to run the engine tests")
class EliminateUndo(unittest.TestCase):
    def setUp(self):
        self.q = Qtp()
        self.q.must("clear_board")

    def tearDown(self):
        self.q.close()

    def test_eliminate_the_seat_to_move_then_undo(self):
        q = self.q
        before = snapshot(q)
        q.must("eliminate 1")
        self.assertEqual(q.must("to_move"), "2")
        q.must("undo")
        self.assertEqual(snapshot(q), before)

    def test_two_eliminations_undo_twice(self):
        q = self.q
        before = snapshot(q)
        q.must("eliminate 2")
        mid = snapshot(q)
        q.must("eliminate 4")
        q.must("undo")
        self.assertEqual(snapshot(q), mid)
        q.must("undo")
        self.assertEqual(snapshot(q), before)

    def test_eliminate_play_undo_undo(self):
        q = self.q
        before = snapshot(q)
        q.must("eliminate 1")
        after_elim = snapshot(q)
        move = q.must("legal_moves 2").split()[0]
        q.must("play 2 %s" % move)
        q.must("undo")
        self.assertEqual(snapshot(q), after_elim)
        q.must("undo")
        self.assertEqual(snapshot(q), before)

    def test_undo_brings_back_the_walls_of_an_eliminated_seat(self):
        q = self.q
        q.must("play 1 f2")
        q.must("play 2 e5h")               # W places a wall: wallsLeft W = 6
        self.assertEqual(q.must("walls").split()[1], "6")
        before = snapshot(q)
        q.must("eliminate 2")              # W out: its unplaced walls are discarded
        self.assertEqual(q.must("walls").split()[1], "0")
        q.must("undo")
        self.assertEqual(snapshot(q), before)
        self.assertEqual(q.must("walls").split()[1], "6")

    def test_the_last_but_one_elimination_ends_the_game_and_undo_reopens_it(self):
        q = self.q
        q.must("eliminate 2")
        q.must("eliminate 4")
        ok, body = q.cmd("eliminate 3")
        self.assertTrue(ok)
        self.assertNotEqual(q.must("winner"), "")     # one seat is left: the game is over
        q.must("undo")
        self.assertEqual(q.must("q4-position").count('"finished":false'), 1)


@unittest.skipUnless(KATAGO and MODEL, "set Q4_KATAGO and Q4_MODEL to run the engine tests")
class Analysis(unittest.TestCase):
    def setUp(self):
        self.q = Qtp()
        self.q.must("clear_board")

    def tearDown(self):
        self.q.close()

    def test_newline_cancels_and_the_tree_is_kept(self):
        q = self.q
        q.write("q4-search_analyze_cancellable 10 100000000")   # a cap no short test reaches
        time.sleep(1.0)
        q.write("")
        ok, body, infos = q.response()
        self.assertTrue(ok)
        self.assertEqual(body, "play cancelled")
        roots = [i for i in infos if i.startswith("info root")]
        self.assertTrue(roots, "no root info line")
        first = int(roots[-1].split()[3])
        q.write("q4-search_analyze_cancellable 10 100000000")
        time.sleep(0.05)
        q.write("")
        ok, body, infos = q.response()
        second_first_root = [i for i in infos if i.startswith("info root")][0]
        self.assertGreater(int(second_first_root.split()[3]), first,
                           "the second search must start from the retained tree")

    def test_another_command_cancels_and_is_run_next(self):
        q = self.q
        q.write("q4-search_analyze_cancellable 10 100000000")
        time.sleep(0.8)
        q.write("to_move")
        ok, body, _ = q.response()
        self.assertEqual(body, "play cancelled")
        self.assertEqual(q.response()[1], "1")

    def test_visit_cap_ends_with_done(self):
        q = self.q
        q.write("q4-search_analyze_cancellable 5 120")
        ok, body, infos = q.response()
        self.assertEqual(body, "done")
        root = [i for i in infos if i.startswith("info root")][-1]
        self.assertGreaterEqual(int(root.split()[3]), 120)
        self.assertLess(int(root.split()[3]), 400)

    def test_move_lines_carry_the_pv_with_seat_letters(self):
        q = self.q
        q.write("q4-search_analyze_cancellable 5 60")
        _, _, infos = q.response()
        moves = [i for i in infos if i.startswith("info move")]
        self.assertTrue(moves)
        self.assertIn(" pv S ", moves[0] + " ")


@unittest.skipUnless(KATAGO and MODEL, "set Q4_KATAGO and Q4_MODEL to run the engine tests")
class Commands(unittest.TestCase):
    def setUp(self):
        self.q = Qtp()
        self.q.must("clear_board")

    def tearDown(self):
        self.q.close()

    def test_set_param_accepts_its_keys_and_rejects_others(self):
        q = self.q
        for cmd in ["q4-set-param maxVisits 50", "q4-set-param q4OpponentMode style",
                    "q4-set-param q4OwnSeats 1", "q4-set-param dynamicScoreUtilityFactor 0.3",
                    "q4-set-param q4LeadWeightEps 0.05", "q4-set-param cpuctExploration 1.1",
                    "q4-set-param rootPolicyTemperature 1.1"]:
            self.assertTrue(q.cmd(cmd)[0], cmd)
        self.assertIn("q4OpponentMode = style", q.must("q4-search-params"))
        self.assertIn("maxVisits = 50", q.must("q4-search-params"))
        ok, body = q.cmd("q4-set-param noSuchKey 1")
        self.assertFalse(ok)
        ok, body = q.cmd("q4-set-param q4OwnSeats 99")
        self.assertFalse(ok)
        ok, body = q.cmd("q4-set-param q4OpponentMode wild")
        self.assertFalse(ok)
        q.must("q4-set-param q4OpponentMode maxn")

    def test_bots_play_the_seat_to_move_and_grudge_needs_a_target(self):
        q = self.q
        self.assertTrue(q.cmd("q4-genmove-bot greedy")[0])
        self.assertTrue(q.cmd("q4-genmove-bot basher")[0])
        ok, body = q.cmd("q4-genmove-bot grudge")
        self.assertFalse(ok)
        self.assertTrue(q.cmd("q4-genmove-bot grudge 3")[0])

    def test_position_json_matches_the_board(self):
        q = self.q
        q.must("play 1 f2")
        pos = json.loads(q.must("q4-position"))
        self.assertEqual(pos["toMove"], 1)
        self.assertEqual(pos["pawns"][0], "f2")
        self.assertEqual(pos["wallsLeft"], [7, 7, 7, 7])


if __name__ == "__main__":
    unittest.main()
