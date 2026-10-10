"""Shared set-up of the play GUI tests: the import path, a controller on the mock engine, temporary logs."""
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
PYTHON_DIR = os.path.dirname(os.path.dirname(HERE))
if PYTHON_DIR not in sys.path:
    sys.path.insert(0, PYTHON_DIR)

from q4_play_gui.game import Q4Game  # noqa: E402

MOCK = os.path.join(HERE, "mock_qtp.py")


def mock_game(**kw):
    log_dir = tempfile.mkdtemp(prefix="q4play_test_")
    game = Q4Game([sys.executable, MOCK], log_dir, move_timeout=kw.pop("move_timeout", 20.0), **kw)
    game.start_engine()
    game._log_dir = log_dir
    return game


def close_game(game):
    game.close()
    shutil.rmtree(game._log_dir, ignore_errors=True)
