"""Tests of python/q4/lead_effect.py (the numbers of the lead effect check, docs/q4/Q4Lead.md)."""
import io
import contextlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q4 import lead_effect  # noqa: E402

FIXTURE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "q4_sgfs_viewer", "tests", "fixture.sgfs")


def test_fixture_numbers(tmp_path):
    out_json = tmp_path / "effect.json"
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lead_effect.main([FIXTURE, "--json", str(out_json)])
    report = json.loads(out_json.read_text())
    assert "t" in report and report["t"]["games"] == 2
    for table in report.values():
        assert table["games"] >= 1
        for p in table["players"].values():
            # shares are fractions of the hopeless moves / walls, or None when there are none
            assert p["hopelessMoves"] >= 0
            assert p["wallShare"] is None or 0.0 <= p["wallShare"] <= 1.0
            assert p["selfHurtingShare"] is None or 0.0 <= p["selfHurtingShare"] <= 1.0
    assert "table t" in buf.getvalue()
