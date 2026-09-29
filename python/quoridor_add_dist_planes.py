#!/usr/bin/env python3
"""Upgrade Quoridor training npz files written before the S8-S11 fix.

Old files stored spatial input channels 8..11 (continuous BFS distances d/32) through packBits,
i.e. truncated to binary. This script recomputes the raw distances exactly from planes that
every row already stores, adds them as spatialDistNCHW (uint8 [N,4,9,9], canonical orientation,
255 = unreachable), and zeros packed planes 8..11, matching what the current C++ writer produces.

Canonical space (QuoridorNN::fillRow): row 0 is the side-to-move's goal row, row 8 the opponent's.
  ch1/ch2: my/opponent pawn.
  ch3: step to canonical row r-1 is blocked (board North for Black, board South for White,
       since fillRow swaps N/S for White together with the row flip).
  ch4: step to canonical row r+1 is blocked.  ch5: step to c+1 blocked.  ch6: step to c-1 blocked.
BFS ignores pawns, exactly like QuoridorNN::fillDistances.

Usage:
  python quoridor_add_dist_planes.py BASEDIR/selfplay [-num-processes 8]
Converts every BASEDIR/selfplay/**/tdata/*.npz in place (atomic temp file + rename); files that
already have spatialDistNCHW are skipped. Do not convert shuffleddata/; delete it instead.
"""
import argparse
import glob
import multiprocessing
import os
import sys

import numpy as np

POS_LEN = 9
FIRST_DIST_CHANNEL = 8
NUM_DIST_CHANNELS = 4
UNREACHABLE = 255
KEY = "spatialDistNCHW"


def unpack_planes(packed: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(packed, axis=2)[:, :, : POS_LEN * POS_LEN]
    return bits.reshape(packed.shape[0], packed.shape[1], POS_LEN, POS_LEN).astype(bool)


def bfs(sources: np.ndarray, blockN, blockS, blockE, blockW) -> np.ndarray:
    """Multi-source BFS vectorized over rows. sources/block*: bool [N,9,9] (row, col)."""
    n = sources.shape[0]
    dist = np.full((n, POS_LEN, POS_LEN), UNREACHABLE, dtype=np.uint8)
    dist[sources] = 0
    frontier = sources.copy()
    d = 0
    while frontier.any():
        reach = np.zeros_like(frontier)
        # Move from a frontier cell to its neighbour if that edge is open.
        reach[:, :-1, :] |= (frontier & ~blockN)[:, 1:, :]   # cell (r+1) steps north to r
        reach[:, 1:, :] |= (frontier & ~blockS)[:, :-1, :]   # cell (r-1) steps south to r
        reach[:, :, 1:] |= (frontier & ~blockE)[:, :, :-1]   # cell (c-1) steps east to c
        reach[:, :, :-1] |= (frontier & ~blockW)[:, :, 1:]   # cell (c+1) steps west to c
        frontier = reach & (dist == UNREACHABLE)
        d += 1
        assert d < UNREACHABLE
        dist[frontier] = d
    return dist


def compute_dist_planes(binaryInputNCHWPacked: np.ndarray) -> np.ndarray:
    planes = unpack_planes(binaryInputNCHWPacked)
    n = planes.shape[0]
    myPawn, oppPawn = planes[:, 1], planes[:, 2]
    assert (myPawn.reshape(n, -1).sum(axis=1) == 1).all(), "every row must have exactly one own pawn (ch1)"
    assert (oppPawn.reshape(n, -1).sum(axis=1) == 1).all(), "every row must have exactly one opponent pawn (ch2)"
    blockN, blockS, blockE, blockW = planes[:, 3], planes[:, 4], planes[:, 5], planes[:, 6]
    # Board edges are always marked blocked in the stored planes (fillRow).
    assert blockN[:, 0, :].all() and blockS[:, -1, :].all() and blockE[:, :, -1].all() and blockW[:, :, 0].all()
    myGoal = np.zeros_like(myPawn)
    myGoal[:, 0, :] = True
    oppGoal = np.zeros_like(myPawn)
    oppGoal[:, POS_LEN - 1, :] = True
    out = np.stack(
        [bfs(src, blockN, blockS, blockE, blockW) for src in (myGoal, oppGoal, myPawn, oppPawn)],
        axis=1,
    )
    return out


def truncated_bits(dist: np.ndarray) -> np.ndarray:
    """What the old writer stored in planes 8..11: (uint8_t)(d < 0 ? 1 : min(1, d/32))."""
    return (dist == UNREACHABLE) | (dist >= 32)


def upgrade_arrays(arrays: dict) -> dict:
    """Returns a new dict of arrays with spatialDistNCHW added and packed planes 8..11 zeroed.
    Checks the recomputed distances against data already in the file."""
    packed = arrays["binaryInputNCHWPacked"]
    dist = compute_dist_planes(packed)
    n = packed.shape[0]
    lo, hi = FIRST_DIST_CHANNEL, FIRST_DIST_CHANNEL + NUM_DIST_CHANNELS

    old = unpack_planes(packed)[:, lo:hi]
    # Old files hold the truncated bits; files from the new writer (key stripped) hold zeros.
    assert (old == truncated_bits(dist)).all() or not old.any(), "stored planes 8..11 disagree with recomputed distances"

    # Globals 13/14 = my/opp shortest distance (same /32 encoding) must match dist at the pawns.
    planes = unpack_planes(packed)
    for k, (pawnCh, gIdx) in enumerate([(1, 13), (2, 14)]):
        d = dist[:, k][planes[:, pawnCh]].astype(np.float32)
        assert d.shape == (n,)
        enc = np.where(d == UNREACHABLE, np.float32(1.0), np.minimum(np.float32(1.0), d / np.float32(32.0)))
        assert np.array_equal(enc, arrays["globalInputNC"][:, gIdx]), f"global {gIdx} disagrees with recomputed distances"

    newPacked = packed.copy()
    newPacked[:, lo:hi, :] = 0
    out = {}
    for key, arr in arrays.items():
        out[key] = newPacked if key == "binaryInputNCHWPacked" else arr
        if key == "binaryInputNCHWPacked":
            out[KEY] = dist
    return out


def convert_file(path: str) -> tuple:
    with np.load(path) as npz:
        if KEY in npz.files:
            return (path, "skipped", 0)
        arrays = {k: npz[k] for k in npz.files}
    out = upgrade_arrays(arrays)
    tmp = path + ".distplanes.tmp"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **out)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return (path, "converted", out[KEY].shape[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("selfplay_dir", help="BASEDIR/selfplay")
    parser.add_argument("-num-processes", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    args = parser.parse_args()

    root = os.path.abspath(args.selfplay_dir)
    if os.path.basename(root.rstrip("/")) == "shuffleddata" or "/shuffleddata" in root:
        sys.exit("Refusing to convert shuffleddata/; delete it instead, it is regenerated from selfplay/.")
    files = sorted(glob.glob(os.path.join(root, "**", "tdata", "*.npz"), recursive=True))
    stale = glob.glob(os.path.join(root, "**", "tdata", "*.distplanes.tmp"), recursive=True)
    for s in stale:
        os.remove(s)
    print(f"Found {len(files)} npz files under {root} (removed {len(stale)} stale temp files)", flush=True)

    counts = {"converted": 0, "skipped": 0}
    rows = 0
    with multiprocessing.Pool(args.num_processes) as pool:
        for i, (path, status, n) in enumerate(pool.imap_unordered(convert_file, files, chunksize=4)):
            counts[status] += 1
            rows += n
            if (i + 1) % 200 == 0 or i + 1 == len(files):
                print(f"{i + 1}/{len(files)} files, converted {counts['converted']} ({rows} rows), skipped {counts['skipped']}", flush=True)
    print("Done.", counts, f"rows converted: {rows}")


if __name__ == "__main__":
    main()
