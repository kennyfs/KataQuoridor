#!/usr/bin/env python3
"""Convert Quoridor I/O v2 training data (.npz) to I/O v3 by appending the v3 repetition inputs as zeros.

v3 rows = v2 rows + spatial planes 19, 20 (repeating / drawing pawn moves; bit-packed in binaryInputNCHWPacked) and
global inputs 17, 18 (repetition rule on, repetition progress). Targets are the same in v2 and v3.

**Exact only for data written with the repetition rule off** (Rules::repetitionDrawCount = 0): then all four v3
inputs are 0 by definition. That holds for every game played by a binary from before the repetition rule (e.g. all
of run3 up to the switch to v3). For data played with the rule on, zeros would be wrong (the history needed to
compute the real inputs is not in the rows), so the tool requires -rule-was-off to confirm.

Usage:
  python quoridor_convert_tdata_v2_to_v3.py -rule-was-off DIR [DIR ...]          # convert in place
  python quoridor_convert_tdata_v2_to_v3.py -rule-was-off -dry-run DIR [DIR ...]

Every *.npz under the DIRs (recursively) with 19 spatial / 17 global input channels is rewritten atomically
(tmp file + rename); v3 files are skipped; anything else is an error. See docs/QuoridorIOv3.md for the run3 switch.
"""
import argparse
import os
import sys

import numpy as np

V2 = (19, 17)
V3 = (21, 19)


def convert_arrays(arrs: dict) -> dict:
    packed = arrs["binaryInputNCHWPacked"]
    glob_ = arrs["globalInputNC"]
    assert packed.ndim == 3 and packed.shape[1] == V2[0] and glob_.shape[1] == V2[1], (packed.shape, glob_.shape)
    n, _, nbytes = packed.shape
    out = dict(arrs)
    out["binaryInputNCHWPacked"] = np.concatenate(
        [packed, np.zeros((n, V3[0] - V2[0], nbytes), dtype=packed.dtype)], axis=1)
    out["globalInputNC"] = np.concatenate([glob_, np.zeros((n, V3[1] - V2[1]), dtype=glob_.dtype)], axis=1)
    return out


def version_of(path: str):
    with np.load(path) as d:
        c = d["binaryInputNCHWPacked"].shape[1]
        g = d["globalInputNC"].shape[1]
    return {V2: 2, V3: 3}.get((c, g), (c, g))


def convert_file(path: str):
    with np.load(path) as d:
        arrs = {k: d[k] for k in d.files}
    out = convert_arrays(arrs)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("-rule-was-off", action="store_true", help="confirm the data was played without the repetition rule")
    ap.add_argument("-dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.rule_was_off:
        raise SystemExit("refusing without -rule-was-off (zeros are only right for data played without the repetition rule)")
    files = sorted(os.path.join(root, f) for d in args.dirs for root, _, fs in os.walk(d) for f in fs
                   if f.endswith(".npz") and not f.endswith(".tmp.npz"))
    counts = {2: 0, 3: 0}
    for path in files:
        v = version_of(path)
        if v not in counts:
            raise SystemExit(f"{path}: unexpected input channels {v}")
        counts[v] += 1
        if v == 2 and not args.dry_run:
            convert_file(path)
    print(f"{len(files)} files: {counts[2]} v2 {'to convert' if args.dry_run else 'converted'}, {counts[3]} already v3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
