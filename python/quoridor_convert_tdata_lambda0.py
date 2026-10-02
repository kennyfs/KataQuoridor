#!/usr/bin/env python3
"""Convert Quoridor I/O v3 training data (.npz) written with a time bonus lambda > 0 to lambda = 0.

The utility score is u = s + sign(s) * lambda * (maxPlies - T) (docs/QuoridorIOv2.md §2.6). With lambda = 0, u = s.
Columns of globalTargetsNC derived from u, and what this tool does to each (docs/QuoridorIOv3.md §9):

  C20  final utility score u, weight C27: on decisive rows (C29 > 0: main rows of a game with a winner) set to the
       final lead C21 (exact: u = s at lambda = 0). Draw rows keep u = 0. Side rows have C27 = 0 and C20 = 0.
  C3   final score of the value targets: on main rows (C27 > 0) it equals C20, and is set to the new C20 (exact).
       On side rows it is the side search's u. Not read by the Quoridor loss; left as is.
  C15  short-term TD score (horizon index 2), a mix of the following searches' u and the final u. The searches' u
       can't be converted exactly (their time bonus depends on the expected game length, not the actual one), and
       C15 has no weight of its own (C24 also weights the TD value targets C4-14, which are unaffected). The tool
       sets C70 = 1 ("1 - short-term score weight", 0 in written data), which removes C15 from the short-term
       score error loss and the short-term optimistic policy weight (metrics_pytorch.py) on these rows.
  C7, C11, C19  the other TD score targets, same as C15: not read by the Quoridor loss; left as is.
  C58  the net's raw score (metadata, not read); left as is.

Policy targets also came from searches with lambda > 0 (the bonus slightly re-weights the score utility of won
lines); they are kept, like any off-policy data. Nothing else in a row depends on lambda.

Checks before writing, per file (anything unexpected is an error, nothing is written for that file):
  - I/O v3 inputs (21 spatial / 19 global channels), 80 global target channels, data format C63 = 3;
  - C29 > 0 only on rows with C27 > 0 (no search lead estimates, estimateLeadProb = 0), and C20 = 0 where C29 = 0;
  - on decisive rows, sign(u) = sign(s) and (u - s) * sign(s) / lambda is an integer in [0, maxPlies], with
    lambda = -lambda-old (default 0.05) and maxPlies = -max-plies (default 300);
  - on main rows, C3 = C20.
Files already converted (C70 = 1 on every row) are skipped, so the tool is idempotent. Files that are already
lambda = 0 data (C70 = 0, at least one decisive row and u = s on all of them) are skipped too. A file without
decisive rows can't tell its lambda; it is converted (only C70 changes).

Usage:
  python quoridor_convert_tdata_lambda0.py -dry-run DIR [DIR ...]
  python quoridor_convert_tdata_lambda0.py DIR [DIR ...]             # convert in place

Every *.npz under the DIRs (recursively) is rewritten atomically (tmp file + rename), with the same zip compression
as the original (C++ writes deflated files, quoridor_convert_tdata_v2_to_v3.py stored ones).
"""
import argparse
import os
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor

import numpy as np

V3 = (21, 19)
NUM_GLOBAL_TARGETS = 80
C_SHORTTERM_SCORE_UNWEIGHT = 70


def classify_and_convert(arrs: dict, lambda_old: float, max_plies: int):
    """Returns (status, stats, out_arrs). status: 'convert', 'converted' (already), 'lambda0' (native lambda = 0)."""
    packed = arrs["binaryInputNCHWPacked"]
    glob_in = arrs["globalInputNC"]
    g = arrs["globalTargetsNC"]
    if (packed.shape[1], glob_in.shape[1]) != V3:
        raise ValueError(f"not I/O v3 data: input channels {(packed.shape[1], glob_in.shape[1])}")
    if g.ndim != 2 or g.shape[1] != NUM_GLOBAL_TARGETS:
        raise ValueError(f"unexpected globalTargetsNC shape {g.shape}")
    n = g.shape[0]
    if n > 0 and not np.all(g[:, 63] == 3.0):
        raise ValueError(f"unexpected data format version C63 {np.unique(g[:, 63])}")

    marker = g[:, C_SHORTTERM_SCORE_UNWEIGHT]
    if n > 0 and np.all(marker == 1.0):
        return "converted", {"rows": n}, None
    if not np.all(marker == 0.0):
        raise ValueError(f"C{C_SHORTTERM_SCORE_UNWEIGHT} is neither all 0 nor all 1: {np.unique(marker)}")

    outcome_w, lead_w = g[:, 27], g[:, 29]
    u, s = g[:, 20], g[:, 21]
    main = outcome_w > 0
    decisive = lead_w > 0
    if np.any(decisive & ~main):
        raise ValueError(f"{int(np.sum(decisive & ~main))} rows with a lead weight but no outcome weight "
                         "(search lead estimates?): C21 is not the final lead there")
    if np.any(u[~decisive] != 0.0):
        raise ValueError("C20 != 0 on rows without a lead weight (draws / side rows)")
    du, ds = u[decisive], s[decisive]
    if np.any(ds == 0.0) or np.any(np.sign(du) != np.sign(ds)):
        raise ValueError("decisive rows with s = 0 or sign(u) != sign(s)")
    plies_left = (du - ds) * np.sign(ds) / lambda_old
    # float32 targets: u ~ 15 has a resolution of ~1e-6, i.e. ~2e-5 plies at lambda 0.05
    if np.any(np.abs(plies_left - np.round(plies_left)) > 1e-3) or np.any(plies_left < -1e-3) \
            or np.any(plies_left > max_plies + 1e-3):
        raise ValueError(f"decisive rows don't match u = s + sign(s) * {lambda_old} * (maxPlies - T)")
    if np.any(g[main, 3] != u[main]):
        raise ValueError("C3 != C20 on main rows")

    stats = {"rows": n, "decisive": int(decisive.sum()), "draws": int((main & ~decisive).sum()),
             "side": int((~main).sum())}
    if stats["decisive"] > 0 and np.all(du == ds):
        return "lambda0", stats, None

    g2 = g.copy()
    g2[decisive, 20] = s[decisive]
    g2[main, 3] = g2[main, 20]
    g2[:, C_SHORTTERM_SCORE_UNWEIGHT] = 1.0
    out = dict(arrs)
    out["globalTargetsNC"] = g2
    return "convert", stats, out


def is_compressed(path: str) -> bool:
    with zipfile.ZipFile(path) as z:
        return any(i.compress_type != zipfile.ZIP_STORED for i in z.infolist())


def process_file(path: str, lambda_old: float, max_plies: int, dry_run: bool):
    try:
        with np.load(path) as d:
            arrs = {k: d[k] for k in d.files}
        status, stats, out = classify_and_convert(arrs, lambda_old, max_plies)
    except Exception as e:
        return path, "error", {"error": str(e)}
    if status == "convert" and not dry_run:
        tmp = path + ".tmp.npz"
        (np.savez_compressed if is_compressed(path) else np.savez)(tmp, **out)
        os.replace(tmp, path)
    return path, status, stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("-lambda-old", type=float, default=0.05, help="the timeBonusPerPly the data was written with")
    ap.add_argument("-max-plies", type=int, default=300, help="the maxPlies the data was written with")
    ap.add_argument("-threads", type=int, default=8)
    ap.add_argument("-dry-run", action="store_true")
    args = ap.parse_args(argv)
    if not args.lambda_old > 0:
        raise SystemExit("-lambda-old must be > 0")
    files = sorted(os.path.join(root, f) for d in args.dirs for root, _, fs in os.walk(d) for f in fs
                   if f.endswith(".npz") and not f.endswith(".tmp.npz"))
    totals = {}
    errors = []
    with ProcessPoolExecutor(max_workers=args.threads) as ex:
        for path, status, stats in ex.map(process_file, files, [args.lambda_old] * len(files),
                                          [args.max_plies] * len(files), [args.dry_run] * len(files), chunksize=4):
            if status == "error":
                errors.append(f"{path}: {stats['error']}")
                continue
            t = totals.setdefault(status, {"files": 0})
            t["files"] += 1
            for k, v in stats.items():
                t[k] = t.get(k, 0) + v
    names = {"convert": "to convert" if args.dry_run else "converted", "converted": "already converted (skipped)",
             "lambda0": "already lambda = 0 data (skipped)"}
    print(f"{len(files)} files")
    for status, t in totals.items():
        print(f"  {names[status]}: " + ", ".join(f"{k} {v}" for k, v in t.items()))
    for e in errors:
        print(f"ERROR {e}", file=sys.stderr)
    if errors:
        print(f"{len(errors)} files with errors, not converted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
