#!/usr/bin/env python3
"""Convert Quoridor I/O v3 training data (.npz) to I/O v4 (docs/QuoridorIOv4.md). Exact: every v4 row is what the C++
writer produces for the same position.

v4 rows = v3 rows + packed spatial planes 21-24 (zero in the bit planes, as for the distance planes), the raw array
wallEffectNCHW (uint8 [N,4,9,9]: per wall placement and player, the increase of that player's shortest distance, 255 =
illegal placement), global inputs 19 / 20 (largest opponent increase / 8, number of opponent walls costing >= 2 / 16),
and the draw as the noResult outcome:

- Wall effects need only what the stored planes hold: the pawns (planes 1, 2), the placed walls (14, 15), in the
  canonical orientation (side to move = Black heading for row 0). They are computed by `katago quoridorwalleffects`, the
  C++ function the writer uses (about 3 us per row), so there is no second implementation to diverge.
- Draw targets: a row whose final outcome target is a draw has C0 = C1 = 0.5, C2 = 0 and the outcome weight on
  (1 - C35 = 1, and the u weight C27 = 1: not a side position). It becomes C0 = C1 = 0, C2 = 1. The TD value targets
  C4-C19 (mixes of later search values and the final outcome) are NOT converted: a draw's share can't be separated
  from the search values in them (in old data a search value never had a noResult share), so they keep their
  win / loss form (noResult 0).

Usage:
  python quoridor_convert_tdata_v3_to_v4.py [-katago PATH] [-num-processes N] [-dry-run] DIR_OR_FILE ...   # in place
  python quoridor_convert_tdata_v3_to_v4.py [-katago PATH] -out-dir OUT DIR ...    # copies: OUT/<path relative to DIR>
      (all files, json and txt included, are copied, so a shuffleddata/<name> directory stays usable by train.py)

Every *.npz (recursively) with 21 spatial / 19 global input channels is converted atomically (tmp file + rename), keeping
the file's mtime (shuffle.py picks its window by mtime); v4 files are skipped; anything else is an error.
"""
import argparse
import multiprocessing
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

V3 = (21, 19)
V4 = (25, 21)
POS_LEN = 9
WE_FIRST_CHANNEL = 21
WE_CHANNELS = 4
WE_ILLEGAL = 255
WE_CAP = 8
WE_BIG = 2
WE_BIG_SCALE = 16.0
GLOBAL_MAX_OPP, GLOBAL_NUM_BIG = 19, 20
KEY = "wallEffectNCHW"


def default_katago() -> str:
    env = os.environ.get("KATAGO_BIN")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "..", "cpp", "katago")


def unpack_planes(packed: np.ndarray) -> np.ndarray:
    bits = np.unpackbits(packed, axis=2)[:, :, : POS_LEN * POS_LEN]
    return bits.reshape(packed.shape[0], packed.shape[1], POS_LEN, POS_LEN).astype(bool)


def rows_for_katago(packed: np.ndarray) -> np.ndarray:
    """The 130-byte records of `katago quoridorwalleffects`: cur pawn cell, opp pawn cell, H walls [64], V walls [64]."""
    planes = unpack_planes(packed)
    n = planes.shape[0]
    cur = planes[:, 1].reshape(n, -1)
    opp = planes[:, 2].reshape(n, -1)
    assert (cur.sum(axis=1) == 1).all() and (opp.sum(axis=1) == 1).all(), "every row needs one pawn per player (planes 1, 2)"
    rec = np.zeros((n, 130), dtype=np.uint8)
    rec[:, 0] = cur.argmax(axis=1)
    rec[:, 1] = opp.argmax(axis=1)
    # Wall anchors: plane 14 = vertical, 15 = horizontal, rows / columns 0..7 of the 9x9 plane.
    rec[:, 2:66] = planes[:, 15, :8, :8].reshape(n, 64)
    rec[:, 66:130] = planes[:, 14, :8, :8].reshape(n, 64)
    assert not planes[:, 14:16, 8, :].any() and not planes[:, 14:16, :, 8].any()
    return rec


def compute_wall_effects(packed: np.ndarray, katago: str) -> np.ndarray:
    """uint8 [N,4,9,9] raw wall effects of the rows of canonical planes `packed` (v3 or v4 layout)."""
    n = packed.shape[0]
    rec = rows_for_katago(packed)
    with tempfile.TemporaryDirectory() as tmp:
        fin, fout = os.path.join(tmp, "in.bin"), os.path.join(tmp, "out.bin")
        rec.tofile(fin)
        subprocess.run([katago, "quoridorwalleffects", "-input", fin, "-output", fout], check=True, stdout=subprocess.DEVNULL)
        out = np.fromfile(fout, dtype=np.uint8)
    assert out.size == n * WE_CHANNELS * POS_LEN * POS_LEN, (out.size, n)
    return out.reshape(n, WE_CHANNELS, POS_LEN, POS_LEN)


def wall_effect_globals(raw: np.ndarray) -> np.ndarray:
    """Float32 [N,2]: globals 19 / 20 from the two opponent planes (QuoridorNN::wallEffectGlobals)."""
    opp = raw[:, 2:4].reshape(raw.shape[0], -1)
    legal = opp != WE_ILLEGAL
    vals = np.where(legal, opp, 0).astype(np.int32)
    mx = np.minimum(vals.max(axis=1), WE_CAP).astype(np.float32) / np.float32(WE_CAP)
    big = np.minimum(np.float32(1.0), (legal & (vals >= WE_BIG)).sum(axis=1).astype(np.float32) / np.float32(WE_BIG_SCALE))
    return np.stack([mx, big], axis=1).astype(np.float32)


def convert_draw_targets(globalTargetsNC: np.ndarray) -> tuple:
    """Draw rows -> C0 = C1 = 0, C2 = 1. Returns (new array, number of rows converted)."""
    g = globalTargetsNC.copy()
    draw = (g[:, 0] == 0.5) & (g[:, 1] == 0.5) & (g[:, 2] == 0.0) & (g[:, 35] == 0.0) & (g[:, 27] == 1.0)
    g[draw, 0] = 0.0
    g[draw, 1] = 0.0
    g[draw, 2] = 1.0
    return g, int(draw.sum())


def convert_arrays(arrs: dict, katago: str) -> tuple:
    packed = arrs["binaryInputNCHWPacked"]
    glob_ = arrs["globalInputNC"]
    assert packed.ndim == 3 and packed.shape[1] == V3[0] and glob_.shape[1] == V3[1], (packed.shape, glob_.shape)
    n, _, nbytes = packed.shape
    raw = compute_wall_effects(packed, katago)
    out = dict(arrs)
    out["binaryInputNCHWPacked"] = np.concatenate(
        [packed, np.zeros((n, V4[0] - V3[0], nbytes), dtype=packed.dtype)], axis=1)
    out["globalInputNC"] = np.concatenate([glob_, wall_effect_globals(raw).astype(glob_.dtype)], axis=1)
    out[KEY] = raw
    out["globalTargetsNC"], num_draw = convert_draw_targets(arrs["globalTargetsNC"])
    return out, num_draw


def version_of(path: str):
    """3 or 4, or a description of what is wrong."""
    with np.load(path) as d:
        shape = (d["binaryInputNCHWPacked"].shape[1], d["globalInputNC"].shape[1])
        has_key = KEY in d.files
    if shape == V3 and not has_key:
        return 3
    if shape == V4 and has_key:
        return 4
    return f"{shape[0]} spatial / {shape[1]} global input channels, {KEY} {'present' if has_key else 'absent'}"


def convert_file(job: tuple) -> tuple:
    src, dst, katago = job
    with np.load(src) as d:
        files = list(d.files)
        arrs = {k: d[k] for k in files}
    st = os.stat(src)
    out, num_draw = convert_arrays(arrs, katago)
    tmp = dst + ".tmp.npz"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(tmp, **out)
    os.replace(tmp, dst)
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
    return (src, "converted", arrs["globalInputNC"].shape[0], num_draw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-katago", default=default_katago(), help="katago binary with the quoridorwalleffects command")
    ap.add_argument("-num-processes", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("-out-dir", help="write converted copies here instead of converting in place (inputs must be directories)")
    ap.add_argument("-dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.dry_run and not os.path.exists(args.katago):
        raise SystemExit(f"katago binary not found: {args.katago} (-katago or KATAGO_BIN)")
    if args.out_dir:
        assert all(os.path.isdir(p) for p in args.paths), "-out-dir needs directories"
    jobs, copies, counts = [], [], {3: 0, 4: 0}
    for p in args.paths:
        entries = []
        if os.path.isfile(p):
            entries.append((p, os.path.basename(p), os.path.dirname(p) or "."))
        else:
            for root, _, fs in os.walk(p):
                for f in sorted(fs):
                    entries.append((os.path.join(root, f), os.path.relpath(os.path.join(root, f), p), p))
        for path, rel, base in entries:
            if path.endswith(".npz") and not path.endswith(".tmp.npz"):
                v = version_of(path)
                if v not in counts:
                    raise SystemExit(f"{path}: unexpected input channels / keys: {v}")
                counts[v] += 1
                dst = os.path.join(args.out_dir, os.path.basename(os.path.abspath(p)), rel) if args.out_dir else path
                if v == 3:
                    jobs.append((path, dst, args.katago))
                elif args.out_dir:
                    copies.append((path, dst))
            elif args.out_dir and not path.endswith(".tmp"):
                copies.append((path, os.path.join(args.out_dir, os.path.basename(os.path.abspath(p)), rel)))
    print(f"{counts[3]} v3 files {'to convert' if args.dry_run else 'converting'}, {counts[4]} already v4", flush=True)
    if args.dry_run:
        return 0
    for src, dst in copies:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
    rows = draws = 0
    with multiprocessing.Pool(args.num_processes) as pool:
        for i, (src, status, n, nd) in enumerate(pool.imap_unordered(convert_file, jobs, chunksize=1)):
            rows += n
            draws += nd
            if (i + 1) % 20 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)} files, {rows} rows, {draws} draw rows", flush=True)
    print(f"Done: {len(jobs)} files, {rows} rows, {draws} draw rows converted to the noResult outcome")
    return 0


if __name__ == "__main__":
    sys.exit(main())
