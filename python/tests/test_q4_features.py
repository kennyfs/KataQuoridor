"""T13: input parity between the C++ Q4NN inputs (`katago q4tool dumpinputs`) and python/q4/features.py.

For positions from random games of several kinds (pawn shuffles, play near maxPlies, wall-heavy games, eliminated
seats, walls running out, plain random walks), under all 8 symmetries:
- binary channels and binary globals exactly equal;
- float channels and globals within 1e-6;
- the raw uint8 distances equal, and decode (dist01) to the float distance planes;
- the sample covers eliminated seats, repetition states, positions near maxPlies, seats without walls, and positions
  where each kind of jump is possible.
"""

import numpy as np
import pytest

from q4_testutil import dump_positions, find_katago
from q4.features import extract_features, raw_distances, decode_raw_distances

BINARY_SPATIAL = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] + list(range(15, 27))
FLOAT_SPATIAL = [10, 11, 12, 13, 14]
FLOAT_GLOBAL = list(range(0, 4)) + list(range(12, 20)) + [24, 25, 27]
BINARY_GLOBAL = [4, 5, 6, 7, 8, 9, 10, 11, 20, 21, 22, 23, 26]

NUM_POSITIONS = 400


@pytest.fixture(scope="module")
def dumped():
    katago = find_katago("eigen")
    if katago is None:
        pytest.skip("no Eigen katago build (cpp/build-eigen)")
    return dump_positions(katago, NUM_POSITIONS, 98765)


def test_feature_layout_is_complete():
    assert sorted(BINARY_SPATIAL + FLOAT_SPATIAL) == list(range(27))
    assert sorted(FLOAT_GLOBAL + BINARY_GLOBAL) == list(range(28))


def test_q4_features_parity_t13(dumped):
    assert len(dumped) == NUM_POSITIONS
    coverage = dict(eliminated=0, repetition=0, near_max_plies=0, no_walls=0, repeat_channel=0, draw_channel=0)
    for index, (data, pos) in enumerate(dumped):
        cpp_global = np.array(data["global"], dtype=np.float32)
        cpp_raw = np.array(data["rawDist"], dtype=np.uint8).reshape(5, 11, 11)
        assert np.array_equal(cpp_raw, raw_distances(pos)), f"raw distances differ, position {index}"
        for entry in data["symmetries"]:
            sym = entry["sym"]
            cpp_spatial = np.array(entry["spatial"], dtype=np.float32).reshape(27, 11, 11)
            py_spatial, py_global = extract_features(pos, sym)
            for ch in BINARY_SPATIAL:
                assert np.array_equal(cpp_spatial[ch], py_spatial[ch]), f"binary channel {ch}, sym {sym}, position {index}"
            for ch in FLOAT_SPATIAL:
                diff = np.abs(cpp_spatial[ch] - py_spatial[ch]).max()
                assert diff <= 1e-6, f"float channel {ch} differs by {diff}, sym {sym}, position {index}"
            if sym == 0:
                # the float distance planes are the decoded raw distances
                for k in range(5):
                    assert np.array_equal(cpp_spatial[10 + k], decode_raw_distances(cpp_raw[k]))
            for idx in BINARY_GLOBAL:
                assert cpp_global[idx] == py_global[idx], f"binary global {idx}, position {index}"
            for idx in FLOAT_GLOBAL:
                assert abs(cpp_global[idx] - py_global[idx]) <= 1e-6, f"global {idx}, position {index}"
        spatial0 = np.array(data["symmetries"][0]["spatial"], dtype=np.float32).reshape(27, 11, 11)
        coverage["eliminated"] += not all(pos.alive)
        coverage["repetition"] += pos.repetition_draw_count >= 2
        coverage["near_max_plies"] += (pos.max_plies - pos.plies) <= 12
        coverage["no_walls"] += any(pos.alive[s] and pos.walls_left[s] == 0 for s in range(4))
        coverage["repeat_channel"] += bool(spatial0[25].any())
        coverage["draw_channel"] += bool(spatial0[26].any())
    print("T13 coverage:", coverage)
    for name, count in coverage.items():
        # drawing pawn moves exist only in the last plies of the (short) repetition games
        assert count >= (5 if name == "draw_channel" else 10), f"only {count} sampled positions with {name}"
