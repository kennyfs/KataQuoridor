"""Runs the sgfs_viewer's pure-function checks (python/sgfs_viewer/tests/check_core.js) with Node, if installed."""
import os
import shutil
import subprocess

import pytest

CHECK = os.path.join(os.path.dirname(__file__), "..", "sgfs_viewer", "tests", "check_core.js")


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_sgfs_viewer_core():
    out = subprocess.run(["node", CHECK], capture_output=True, text=True)
    assert out.returncode == 0, out.stdout + out.stderr
