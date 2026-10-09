"""The Q4 SGF viewer (python/q4_sgfs_viewer): its pure functions (node tests) and its shortest-path distances against
the independent Python reference, on the fixture games."""
import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q4.record import load_records  # noqa: E402
from q4.reference import Pos, compute_distances_to_center, eliminate, play, str_to_action  # noqa: E402

VIEWER = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "q4_sgfs_viewer", "tests")
FIXTURE = os.path.join(VIEWER, "fixture.sgfs")

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def test_core_checks():
    out = subprocess.run(["node", os.path.join(VIEWER, "check_core.js")], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr


def test_viewer_distances_equal_reference():
    out = subprocess.run(["node", os.path.join(VIEWER, "dump_dists.js"), FIXTURE], capture_output=True, text=True,
                         check=True)
    js_games = [json.loads(line) for line in out.stdout.splitlines()]
    records = list(load_records(FIXTURE))
    assert len(js_games) == len(records) >= 10
    n = 0
    for rec, js in zip(records, js_games):
        rules = rec["rules"]
        pos = Pos(max_plies=rules["maxPlies"], repetition_draw_count=rules["repetitionDrawCount"],
                  initial_walls=rules["initialWalls"])
        for k in range(len(rec["events"]) + 1):
            dist = compute_distances_to_center(pos.hwalls, pos.vwalls)
            expected = [dist[pos.pawn[s]] if pos.alive[s] else None for s in range(4)]
            assert js[k] == expected, (k, js[k], expected)
            n += 1
            if k < len(rec["events"]):
                ev = rec["events"][k]
                pos = eliminate(pos, ev["elim"] - 1) if "elim" in ev else play(pos, str_to_action(ev["a"]))
    assert n > 300
