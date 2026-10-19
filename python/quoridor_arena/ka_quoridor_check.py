"""Check ka_quoridor_qtp.py's move notation against KataQuoridor: random games where every position's full legal
move set must be the same in both engines (KataQuoridor's arbiter vs Ka_Quoridor's State, no nets).

Usage (from the python/ directory):
    python -m quoridor_arena.ka_quoridor_check --katago ../cpp/katago --ka-python ~/arena/ka_venv/bin/python \
        --ka-dir ~/sync/HW/Quoridor/Ka_Quoridor --data-dir ~/sync/HW/Quoridor/Ka_Quoridor/from_google_drive/application_data
"""
import argparse
import os
import random

from .qtp import QTPEngine

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago", required=True)
    ap.add_argument("--ka-python", required=True)
    ap.add_argument("--ka-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--wall-prob", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log-dir", default="/tmp/ka_quoridor_check_logs", help="the arbiter's gtp log dir")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    kq = QTPEngine("kq", [os.path.abspath(args.katago), "gtp", "-model", "/dev/null", "-config",
                          os.path.join(REPO, "cpp/configs/gtp_quoridor.cfg"), "-override-config",
                          "debugSkipNeuralNet=true,logAllGTPCommunication=false,logSearchInfo=false,logDir=" + args.log_dir],
                    cwd=REPO, default_timeout=60)
    ka = QTPEngine("ka", [args.ka_python, os.path.join(HERE, "ka_quoridor_qtp.py"), "--ka-dir", args.ka_dir,
                          "--data-dir", args.data_dir, "--no-net"], default_timeout=60)
    kq.start()
    ka.start()
    positions = mismatches = 0
    for g in range(args.games):
        kq.must("clear_board")
        ka.must("clear_board")
        for ply in range(300):
            if kq.must("winner").strip() not in ("", "none", "None", "0"):
                break
            a = set(kq.must("legal_moves").lower().split())
            b = set(ka.must("legal_moves").lower().split())
            positions += 1
            if not a:
                break
            if a != b:
                mismatches += 1
                print("game %d ply %d: only KataQuoridor %s, only Ka %s" % (g, ply, sorted(a - b), sorted(b - a)))
                if mismatches > 10:
                    raise SystemExit("too many mismatches")
            walls = [m for m in a if len(m) == 3]
            pawns = sorted(m for m in a if len(m) == 2)
            mv = rng.choice(walls) if walls and rng.random() < args.wall_prob else rng.choice(pawns or sorted(a))
            color = "b" if ply % 2 == 0 else "w"
            kq.must("play %s %s" % (color, mv))
            ka.must("play %s %s" % (color, mv))
    print("%d games, %d positions, %d mismatches" % (args.games, positions, mismatches))
    kq.close()
    ka.close()


if __name__ == "__main__":
    main()
