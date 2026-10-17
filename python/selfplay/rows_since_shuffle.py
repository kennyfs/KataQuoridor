#!/usr/bin/env python3
"""Prints the number of self-play rows on disk that the last shuffle did not see (synchronous_loop.sh, ROWS_PER_CYCLE).

    rows_since_shuffle.py BASEDIR

Rows on disk: every .npz under BASEDIR/selfplay/, found and counted the way shuffle.py does (temp-like names with "_"
are skipped; rows = the length of binaryInputNCHWPacked), local and remote (python/colab) files alike, whenever they
were written: during self-play, while the loop shuffled or trained, or while it was stopped.
Rows at the last shuffle: range[1] of the newest BASEDIR/shuffleddata/*/train.json (shuffle.py's "Total rows found",
the d in the model names); without shuffled data, the largest d of the nets in BASEDIR/models/.
Row counts are cached by path, mtime and size in BASEDIR/rows_cache.json, so only new files are opened.
"""
import json, os, re, sys, zipfile

import numpy as np


def npz_rows(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.startswith("binaryInputNCHWPacked"))
        with z.open(name) as a:
            version = np.lib.format.read_magic(a)
            header = np.lib.format.read_array_header_1_0(a) if version == (1, 0) else np.lib.format.read_array_header_2_0(a)
            return header[0][0]


def rows_on_disk(basedir):
    cache_path = os.path.join(basedir, "rows_cache.json")
    try:
        with open(cache_path) as f:
            cache = json.load(f)
    except (OSError, ValueError):
        cache = {}
    new_cache, total, bad = {}, 0, 0
    for path, _dirs, files in os.walk(os.path.join(basedir, "selfplay"), followlinks=True):
        for name in files:
            if not name.endswith(".npz") or "_" in name:
                continue
            full = os.path.join(path, name)
            try:
                st = os.stat(full)
            except OSError:
                continue
            key = [st.st_mtime, st.st_size]
            hit = cache.get(full)
            if hit is not None and hit[:2] == key:
                rows = hit[2]
            else:
                try:
                    rows = int(npz_rows(full))
                except Exception as e:  # a file being written or a bad file: shuffle.py skips it too
                    print("cannot count %s: %s" % (full, e), file=sys.stderr)
                    bad += 1
                    continue
            new_cache[full] = key + [rows]
            total += rows
    tmp = cache_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(new_cache, f)
    os.replace(tmp, cache_path)
    return total, bad


def rows_at_last_shuffle(basedir):
    shuffled = os.path.join(basedir, "shuffleddata")
    dirs = sorted(d for d in os.listdir(shuffled) if not d.endswith(".tmp")) if os.path.isdir(shuffled) else []
    for d in reversed(dirs):
        try:
            with open(os.path.join(shuffled, d, "train.json")) as f:
                return int(json.load(f)["range"][1]), "shuffleddata/" + d
        except (OSError, ValueError, KeyError):
            continue
    ds = [int(m.group(1)) for n in os.listdir(os.path.join(basedir, "models"))
          if (m := re.search(r"-d(\d+)$", n))]
    if not ds:
        raise SystemExit("no shuffled data and no models with -d<rows> names in " + basedir)
    return max(ds), "models (largest d)"


def main():
    basedir = sys.argv[1]
    last, source = rows_at_last_shuffle(basedir)
    now, bad = rows_on_disk(basedir)
    print("rows on disk %d, at the last shuffle %d (%s), new %d%s" % (
        now, last, source, now - last, ", %d files not countable" % bad if bad else ""), file=sys.stderr)
    print(max(0, now - last))


if __name__ == "__main__":
    main()
