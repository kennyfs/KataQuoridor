#!/usr/bin/env python3
"""Convert Quoridor I/O v3 training data (.npz) to I/O v4 (docs/QuoridorIOv4.md). Exact, and needs no katago binary.

v4 rows have the same arrays as v3 rows; they differ only in the draw targets and carry a version stamp:

- Draw targets: a row whose final outcome target is a draw has C0 = C1 = 0.5, C2 = 0 and the outcome weight on
  (1 - C35 = 1, and the u weight C27 = 1: not a side position). It becomes C0 = C1 = 0, C2 = 1. The TD value targets
  C4-C19 (mixes of later search values and the final outcome) are NOT converted: a draw's share can't be separated
  from the search values in them (in old data a search value never had a noResult share), so they keep their
  win / loss form (noResult 0).
- Stamp: the key "quoridorIOVersion" (int32 [1,1], value 4), as the C++ writer writes it. The shuffler and train.py refuse
  npz files without it, which is how unconverted v3 data is detected (v3 and v4 rows can't be told apart by shape).

Usage:
  python quoridor_convert_tdata_v3_to_v4.py [-num-processes N] [-dry-run] DIR_OR_FILE ...   # in place
  python quoridor_convert_tdata_v3_to_v4.py -out-dir OUT DIR ...    # copies: OUT/<path relative to DIR>
      (all files, json and txt included, are copied, so a shuffleddata/<name> directory stays usable by train.py)

Every unstamped *.npz (recursively) with 21 spatial / 19 global input channels is converted atomically (tmp file +
rename), keeping the file's mtime (shuffle.py picks its window by mtime); stamped files are skipped; anything else is an
error.
"""
import argparse
import multiprocessing
import os
import shutil
import sys

import numpy as np

V3 = (21, 19)
IO_VERSION = 4
KEY = "quoridorIOVersion"


def convert_draw_targets(globalTargetsNC: np.ndarray) -> tuple:
    """Draw rows -> C0 = C1 = 0, C2 = 1. Returns (new array, number of rows converted)."""
    g = globalTargetsNC.copy()
    draw = (g[:, 0] == 0.5) & (g[:, 1] == 0.5) & (g[:, 2] == 0.0) & (g[:, 35] == 0.0) & (g[:, 27] == 1.0)
    g[draw, 0] = 0.0
    g[draw, 1] = 0.0
    g[draw, 2] = 1.0
    return g, int(draw.sum())


def convert_arrays(arrs: dict) -> tuple:
    """v3 arrays -> v4 arrays: the draw targets converted and the version stamp added; everything else is unchanged."""
    glob_ = arrs["globalInputNC"]
    packed = arrs["binaryInputNCHWPacked"]
    assert packed.ndim == 3 and packed.shape[1] == V3[0] and glob_.shape[1] == V3[1], (packed.shape, glob_.shape)
    out = dict(arrs)
    out["globalTargetsNC"], num_draw = convert_draw_targets(arrs["globalTargetsNC"])
    out[KEY] = np.array([[IO_VERSION]], dtype=np.int32)
    return out, num_draw


def version_of(path: str):
    """3 (unstamped, 21 / 19 channels) or 4 (stamped), or a description of what is wrong."""
    with np.load(path) as d:
        shape = (d["binaryInputNCHWPacked"].shape[1], d["globalInputNC"].shape[1])
        stamp = int(np.asarray(d[KEY]).reshape(-1)[0]) if KEY in d.files else None
    if shape != V3:
        return f"{shape[0]} spatial / {shape[1]} global input channels (expected {V3[0]} / {V3[1]})"
    if stamp is None:
        return 3
    if stamp == IO_VERSION:
        return 4
    return f"{KEY} {stamp}"


def convert_file(job: tuple) -> tuple:
    src, dst = job
    with np.load(src) as d:
        files = list(d.files)
        arrs = {k: d[k] for k in files}
    st = os.stat(src)
    out, num_draw = convert_arrays(arrs)
    tmp = dst + ".tmp.npz"
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    np.savez_compressed(tmp, **out)
    os.replace(tmp, dst)
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
    return (src, "converted", arrs["globalInputNC"].shape[0], num_draw)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("-num-processes", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("-out-dir", help="write converted copies here instead of converting in place (inputs must be directories)")
    ap.add_argument("-dry-run", action="store_true")
    args = ap.parse_args(argv)
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
                    jobs.append((path, dst))
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
