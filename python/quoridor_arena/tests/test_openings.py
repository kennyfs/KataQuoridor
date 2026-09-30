import collections
import glob
import os
import random
import sys

import pytest

from quoridor_arena import openings
from quoridor_arena.qtp import QTPEngine
from quoridor_arena.referee import Arbiter

MOCK = os.path.join(os.path.dirname(__file__), "mock_engine.py")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _find_katago():
    """$KATAGO_BIN, else the most recently built cpp/build*/katago (like python/tests/test_qtp_rules.py)."""
    if os.environ.get("KATAGO_BIN"):
        return os.environ["KATAGO_BIN"]
    candidates = [c for c in glob.glob(os.path.join(REPO, "cpp", "build*", "katago")) if os.path.isfile(c)]
    return max(candidates, key=os.path.getmtime) if candidates else ""


ARBITER_KATAGO = _find_katago()


class FakePosition:
    """Legal moves depend on the move history; listing order is shuffled to check we sort."""

    def __init__(self, shuffle_seed=0):
        self.hist = []
        self.shuffle = random.Random(shuffle_seed)

    def clear(self):
        self.hist = []

    def winner(self):
        return None

    def legal_moves(self):
        n = len(self.hist)
        moves = ["%s%d" % (c, (n % 9) + 1) for c in "abcde"] + ["%s%dh" % (c, (n % 8) + 1) for c in "abcdefgh"]
        self.shuffle.shuffle(moves)
        return moves

    def play(self, color, mv):
        self.hist.append(mv)
        return True


def test_weights():
    moves = ["e8", "d9", "a1h", "b1h", "c1v", "d1v"]
    cands, w = openings.move_weights(moves, 0.2)
    wall_mass = sum(wi for m, wi in zip(cands, w) if openings.is_wall(m))
    assert wall_mass / sum(w) == pytest.approx(0.2)
    cands, w = openings.move_weights(moves, 0.0)
    assert all(not openings.is_wall(m) for m in cands)


def test_reproducible_and_distinct():
    a = openings.sample_openings(FakePosition(1), 30, 4, 0.2, seed=7)
    b = openings.sample_openings(FakePosition(2), 30, 4, 0.2, seed=7)
    c = openings.sample_openings(FakePosition(1), 30, 4, 0.2, seed=8)
    assert a == b  # independent of the engine's listing order
    assert a != c
    assert len({tuple(o) for o in a}) == 30
    assert all(len(o) == 4 for o in a)
    # Extending the count keeps the earlier openings.
    assert openings.sample_openings(FakePosition(3), 40, 4, 0.2, seed=7)[:30] == a


def test_wall_fraction_statistics():
    ops = openings.sample_openings(FakePosition(), 500, 4, 0.2, seed=1)
    counts = collections.Counter(openings.is_wall(m) for o in ops for m in o)
    frac = counts[True] / (counts[True] + counts[False])
    assert 0.16 < frac < 0.24


def test_cache_file(tmp_path):
    path = str(tmp_path / "openings.json")
    a = openings.load_or_create(path, FakePosition, 5, 4, 0.2, 3)
    b = openings.load_or_create(path, lambda: pytest.fail("should reuse the file"), 5, 4, 0.2, 3)
    assert a == b
    with pytest.raises(SystemExit):
        openings.load_or_create(path, FakePosition, 5, 4, 0.3, 3)


def test_mock_arbiter(tmp_path):
    arb = Arbiter(QTPEngine("arb", [sys.executable, MOCK], str(tmp_path / "a.log")))
    arb.start()
    try:
        a = openings.sample_openings(arb, 10, 4, 0.2, seed=5)
        b = openings.sample_openings(arb, 10, 4, 0.2, seed=5)
    finally:
        arb.close()
    assert a == b


@pytest.mark.skipif(not os.path.isfile(ARBITER_KATAGO), reason="katago binary not found (set KATAGO_BIN or build under cpp/build*/)")
def test_katago_arbiter(tmp_path):
    argv = [ARBITER_KATAGO, "gtp", "-model", "/dev/null", "-config",
            os.path.join(REPO, "cpp", "configs", "gtp_quoridor.cfg"), "-override-config",
            "debugSkipNeuralNet=true,logAllGTPCommunication=false,logSearchInfo=false,logDir=%s" % tmp_path]
    arb = Arbiter(QTPEngine("arb", argv, str(tmp_path / "a.log"), cwd=REPO))
    arb.start()
    try:
        a = openings.sample_openings(arb, 20, 4, 0.2, seed=11)
        b = openings.sample_openings(arb, 20, 4, 0.2, seed=11)
        # Every opening replays legally.
        for o in a:
            arb.clear()
            for k, mv in enumerate(o):
                assert arb.play("b" if k % 2 == 0 else "w", mv)
    finally:
        arb.close()
    assert a == b
    assert len({tuple(o) for o in a}) == 20
