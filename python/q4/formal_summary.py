#!/usr/bin/env python3
"""One line per cycle of the Q4 formal run (Plan §15), appended to BASEDIR/summary.tsv by q4_formal_eval.sh:

    python q4/formal_summary.py BASEDIR CYCLE >> BASEDIR/summary.tsv

Columns: time, cycle, the newest model, the training position (samples), the last train and validation losses
(policy p0, value, style policy; train.py's metrics_*.json), the self-play numbers of the net that played this cycle
(the last "Final Model ... stats" block of selfplay/stdout.txt: games, draw rate with its maxPlies / repetition parts,
rows written, rows per hour of wall time since the previous summary line) and the learner win rates by opponent kind
(learner % / opponent % points per seat). A header is printed when the summary file is still empty.
"""
import json
import os
import re
import sys
import time

COLUMNS = ["time", "cycle", "model", "nsamp", "train_p0loss", "train_vloss", "train_pstyleloss", "val_p0loss",
           "val_vloss", "val_pstyleloss", "selfplay_net", "games", "draw_pct", "maxplies_pct", "rep_pct", "rows",
           "rows_per_hour", "learner_by_kind"]


def last_json_line(path):
    if not os.path.isfile(path):
        return {}
    last = None
    with open(path) as f:
        for line in f:
            if line.strip():
                last = line
    return json.loads(last) if last else {}


def last_final_block(path):
    """The last 'Final Model X stats ...' line of the self-play log and its continuation lines."""
    if not os.path.isfile(path):
        return None
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    for i in range(len(lines) - 1, -1, -1):
        if "Final Model " in lines[i]:
            block = [lines[i]]
            for nxt in lines[i + 1:i + 3]:
                if nxt.startswith("  "):
                    block.append(nxt.strip())
            return block
    return None


def parse_selfplay(block):
    out = {}
    if not block:
        return out
    head = block[0]
    m = re.search(r"Final Model (\S+) stats \((\d+) games\)", head)
    if m:
        out["selfplay_net"], out["games"] = m.group(1), int(m.group(2))
    m = re.search(r"draw rate = ([\d.]+)% \(maxPlies: ([\d.]+)%, rep: ([\d.]+)%\)", head)
    if m:
        out["draw_pct"], out["maxplies_pct"], out["rep_pct"] = m.groups()
    m = re.search(r"rows written = (\d+)", head)
    if m:
        out["rows"] = int(m.group(1))
    for line in block[1:]:
        if line.startswith("by opponent kind"):
            kinds = re.findall(r"(\w+): ([\d.]+)% / ([\d.]+)% \((\d+) games\)", line)
            out["learner_by_kind"] = " ".join(f"{k}:{a}/{b}({n})" for k, a, b, n in kinds)
    return out


def main():
    basedir, cycle = sys.argv[1], sys.argv[2]
    summary_path = os.path.join(basedir, "summary.tsv")
    print_header = not os.path.isfile(summary_path) or os.path.getsize(summary_path) == 0
    row = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "cycle": cycle}

    models_dir = os.path.join(basedir, "models")
    models = sorted(os.listdir(models_dir), key=lambda n: os.path.getmtime(os.path.join(models_dir, n)))
    row["model"] = models[-1] if models else ""

    train_dirs = [d for d in os.listdir(os.path.join(basedir, "train"))] if os.path.isdir(os.path.join(basedir, "train")) else []
    for d in train_dirs:
        tm = last_json_line(os.path.join(basedir, "train", d, "metrics_train.json"))
        vm = last_json_line(os.path.join(basedir, "train", d, "metrics_val.json"))
        if "nsamp" in tm:
            row["nsamp"] = int(tm["nsamp"])
        for key in ("p0loss", "vloss", "pstyleloss"):
            if key in tm:
                row["train_" + key] = f"{tm[key]:.4f}"
            if key in vm:
                row["val_" + key] = f"{vm[key]:.4f}"

    row.update(parse_selfplay(last_final_block(os.path.join(basedir, "selfplay", "stdout.txt"))))

    # Rows per hour of wall time since the previous summary line (the whole cycle: self-play, shuffle, train, export)
    if not print_header and "rows" in row:
        with open(summary_path) as f:
            lines = [l for l in f.read().splitlines() if l and not l.startswith("time")]
        if lines:
            prev = time.mktime(time.strptime(lines[-1].split("\t")[0], "%Y-%m-%d %H:%M:%S"))
            hours = max(1e-6, (time.time() - prev) / 3600.0)
            row["rows_per_hour"] = f"{row['rows'] / hours:.0f}"

    if print_header:
        print("\t".join(COLUMNS))
    print("\t".join(str(row.get(c, "")) for c in COLUMNS))


if __name__ == "__main__":
    main()
