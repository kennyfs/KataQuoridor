#!/usr/bin/env python3
"""Benchmark self-play throughput for Q4 (or Duel) self-play processes.

Usage (from the repo root, or anywhere):
    python python/q4/bench_selfplay.py \
        --katago cpp/build-cuda/katago \
        --models-dir /tmp/q4models/b2c64_q4 \
        --config cpp/configs/q4/training/q4_selfplay.cfg \
        --seconds 150 --warmup 30 \
        --override "nnCacheSizePowerOfTwo=20" \
        --out /tmp/bench_b1.json

Metrics collected every second during [warmup, warmup+seconds]:
  - /proc/<pid>/stat  -> utime+stime (busy CPU ticks -> average busy cores)
  - /proc/<pid>/status -> VmRSS (MB)
  - nvidia-smi         -> GPU utilization %, power W, GPU memory MB
After SIGINT, parses the katago log for
  "GPU ... finishing, processed X rows Y batches" (sum over server threads)
and the last stats line for games finished / training rows written.

NN rows/s = total_rows / (warmup + seconds) because the window-start row count
is not separately logged; this is noted in the JSON.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time


def _proc_stat_cpu_ticks(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split()
        return int(fields[13]) + int(fields[14])
    except Exception:
        return 0


def _proc_rss_mb(pid):
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _nvidia_smi():
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=utilization.gpu,power.draw,memory.used",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode().strip().split("\n")[0]
        parts = [p.strip() for p in out.split(",")]
        return {"util_pct": float(parts[0]),
                "power_w":  float(parts[1]),
                "mem_mb":   float(parts[2])}
    except Exception:
        return {}


def _parse_log_rows_batches(log_text):
    """Sum rows and batches from all 'GPU ... finishing, processed X rows Y batches' lines."""
    total_rows = 0
    total_batches = 0
    for m in re.finditer(
        r"GPU\s+[-]?\d+\s+finishing,\s+processed\s+(\d+)\s+rows\s+(\d+)\s+batches",
        log_text,
    ):
        total_rows += int(m.group(1))
        total_batches += int(m.group(2))
    return total_rows, total_batches


def _parse_log_stats_line(log_text):
    """Return (games, rows_written) from the last stats line."""
    games = 0
    rows_written = 0
    for m in re.finditer(
        r"stats\s+\((\d+)\s+games\).*?rows written\s*=\s*(\d+)",
        log_text,
    ):
        games = int(m.group(1))
        rows_written = int(m.group(2))
    return games, rows_written


def main():
    p = argparse.ArgumentParser(description="Benchmark self-play throughput")
    p.add_argument("--katago", required=True)
    p.add_argument("--models-dir", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--seconds", type=int, default=150)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--override", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--duel", action="store_true")
    args = p.parse_args()

    subcommand = "selfplay" if args.duel else "q4selfplay"
    clk_tck = os.sysconf("SC_CLK_TCK")

    output_dir = tempfile.mkdtemp(prefix="katago_bench_")
    print(f"Output dir: {output_dir}", flush=True)

    cmd = [args.katago, subcommand,
           "-models-dir", args.models_dir,
           "-output-dir", output_dir,
           "-config", args.config]
    if args.override:
        cmd += ["-override-config", args.override]

    print("Starting:", " ".join(cmd), flush=True)
    log_path = os.path.join(output_dir, "bench.log")
    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)

    pid = proc.pid
    print(f"PID: {pid}", flush=True)

    wall_start = time.monotonic()
    samples = []
    total_poll = args.warmup + args.seconds

    for i in range(total_poll):
        time.sleep(1.0)
        elapsed = time.monotonic() - wall_start
        cpu_ticks = _proc_stat_cpu_ticks(pid)
        rss_mb = _proc_rss_mb(pid)
        gpu = _nvidia_smi()
        in_window = i >= args.warmup
        samples.append({"t": elapsed, "cpu_ticks": cpu_ticks,
                         "rss_mb": rss_mb, "gpu": gpu, "in_window": in_window})
        if i == args.warmup:
            print(f"[warmup done] t={elapsed:.1f}s rss={rss_mb:.0f} MB", flush=True)
        if i % 30 == 0 and i > 0:
            print(f"  t={elapsed:.0f}s rss={rss_mb:.0f} MB "
                  f"gpu_util={gpu.get('util_pct','?')}%", flush=True)

    print("Sending SIGINT...", flush=True)
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()

    with open(log_path) as f:
        log_text = f.read()

    # Measurement window
    window = [s for s in samples if s["in_window"]]
    wall_seconds_measured = (window[-1]["t"] - window[0]["t"]) if len(window) >= 2 else 0.0

    # Average busy cores
    if len(window) >= 2:
        delta_ticks = window[-1]["cpu_ticks"] - window[0]["cpu_ticks"]
        avg_busy_cores = (delta_ticks / (wall_seconds_measured * clk_tck)
                          if wall_seconds_measured > 0 else 0.0)
    else:
        avg_busy_cores = 0.0

    rss_start = window[0]["rss_mb"] if window else 0.0
    rss_end = window[-1]["rss_mb"] if window else 0.0

    gpu_utils = [s["gpu"]["util_pct"] for s in window if "util_pct" in s["gpu"]]
    gpu_powers = [s["gpu"]["power_w"] for s in window if "power_w" in s["gpu"]]
    mean_gpu_util = (sum(gpu_utils) / len(gpu_utils)) if gpu_utils else None
    max_gpu_util = max(gpu_utils) if gpu_utils else None
    mean_power_w = (sum(gpu_powers) / len(gpu_powers)) if gpu_powers else None

    total_rows, total_batches = _parse_log_rows_batches(log_text)
    games_finished, training_rows_written = _parse_log_stats_line(log_text)

    # NN rows/s: use total_rows / (warmup+seconds) as noted in the docstring
    approx_total_wall = float(args.warmup + args.seconds)
    nn_rows_per_s = total_rows / approx_total_wall if total_rows > 0 else 0.0
    rows_note = (
        f"total_rows ({total_rows}) / (warmup+seconds) ({approx_total_wall:.0f}s); "
        "window-start rows not individually logged"
    ) if total_rows > 0 else "no 'GPU finishing' lines found in log"

    avg_batch_size = (total_rows / total_batches) if total_batches > 0 else 0.0
    cpu_us_per_row = (
        (avg_busy_cores / nn_rows_per_s) * 1e6
        if nn_rows_per_s > 0 and avg_busy_cores > 0 else 0.0
    )

    result = {
        "katago": args.katago,
        "models_dir": args.models_dir,
        "config": args.config,
        "overrides": args.override,
        "duel": args.duel,
        "warmup_s": args.warmup,
        "wall_seconds_measured": round(wall_seconds_measured, 2),
        "nn_rows_per_s": round(nn_rows_per_s, 2),
        "nn_rows_per_s_note": rows_note,
        "total_rows_from_log": total_rows,
        "total_batches_from_log": total_batches,
        "avg_batch_size": round(avg_batch_size, 2),
        "games_finished": games_finished,
        "training_rows_written": training_rows_written,
        "avg_busy_cores": round(avg_busy_cores, 3),
        "cpu_us_per_nn_row": round(cpu_us_per_row, 2),
        "mean_gpu_util_pct": round(mean_gpu_util, 1) if mean_gpu_util is not None else None,
        "max_gpu_util_pct": round(max_gpu_util, 1) if max_gpu_util is not None else None,
        "mean_power_w": round(mean_power_w, 1) if mean_power_w is not None else None,
        "rss_start_mb": round(rss_start, 1),
        "rss_end_mb": round(rss_end, 1),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResult written to {args.out}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
