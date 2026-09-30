"""Quoridor arena: play a roster of QTP engines against each other and rate them.

Usage (from the python/ directory):
    python -m quoridor_arena.arena --roster quoridor_arena/roster_default.json --out ~/arena/ladder1
    python -m quoridor_arena.arena --roster R.json --out DIR --engines sq-random,sq-greedy,kq-latest-v16 \
        --games-per-pair 10 --verify
    python -m quoridor_arena.arena --roster R.json --out DIR --pairs kq-a-v256:kq-b-v256,sq-search-d4:kq-b-v1

Results go to DIR/results.jsonl (one JSON object per game; reruns skip finished games), SGFs to
DIR/sgfs/<pair>.sgfs, and the Elo report to DIR/report.md. See docs/Evaluation.md.
"""
import argparse
import collections
import json
import os
import re
import sys
import threading
import time

from . import elo, openings
from .qtp import QTPEngine, QTPError
from .referee import Arbiter, Player, RefereeError, VerifyMismatch, play_game

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


# -- roster -------------------------------------------------------------------

UNSET_ENV_MARK = "<unset environment variable "
_ENV_RE = re.compile(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}")


def _subst(s, variables):
    for _ in range(3):  # allow variables that reference other variables
        for k, v in variables.items():
            s = s.replace("{" + k + "}", str(v))
    # {env:NAME} is the environment variable NAME, for machine-specific paths. An unset one is only an
    # error for the engines actually used (see check_env_resolved).
    return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), UNSET_ENV_MARK + m.group(1) + ">"), s)


def check_env_resolved(argvs):
    for argv in argvs:
        for a in argv:
            if UNSET_ENV_MARK in a:
                name = a.split(UNSET_ENV_MARK, 1)[1].split(">", 1)[0]
                raise SystemExit("environment variable %s is not set but is needed by: %s" % (name, " ".join(argv)))


def load_roster(path, out_dir):
    """Returns a list of engine specs: {name, argv, seed, ...}.

    Entry kinds:
      {"name", "command", "args": [...], "seed"}         generic; "{seed}", "{out}", "{repo}", the roster's
                                                          "vars" and "{env:NAME}" are substituted in command/args
      {"name", "katago_model": "<dir name>", "visits": V} shorthand for a KataQuoridor engine built from
                                                          vars katago, models_dir, gtp_config, katago_overrides
    """
    with open(path) as f:
        data = json.load(f)
    variables = {"repo": REPO, "out": os.path.abspath(out_dir)}
    variables.update(data.get("vars", {}))
    specs = []
    names = set()
    for e in data["engines"]:
        if e.get("disabled"):
            continue
        name = e["name"]
        if name in names:
            raise SystemExit("duplicate engine name %r in %s" % (name, path))
        names.add(name)
        v = dict(variables, seed=e.get("seed", 0), name=name)
        if "katago_model" in e:
            v["model"] = e["katago_model"]
            v["visits"] = e["visits"]
            command = "{katago}"
            args = ["gtp", "-model", "{models_dir}/{model}/model.bin.gz", "-config", "{gtp_config}",
                    "-override-config", "maxVisits={visits},{katago_overrides}"]
        else:
            command = e["command"]
            args = e.get("args", [])
        argv = [_subst(command, v)] + [_subst(a, v) for a in args]
        spec = dict(e)
        spec["argv"] = argv
        specs.append(spec)
    arbiter_argv = [_subst(a, variables) for a in data.get("arbiter", [
        "{arbiter_katago}", "gtp", "-model", "/dev/null", "-config", "{gtp_config}", "-override-config",
        "debugSkipNeuralNet=true,logAllGTPCommunication=false,logSearchInfo=false,logDir={out}/gtp_logs"])]
    return specs, arbiter_argv


# -- schedule -----------------------------------------------------------------

def game_id(a, b, k, swap):
    return "%s_vs_%s_o%03d_%s" % (a, b, k, "ba" if swap else "ab")


def make_schedule(names, pairs, games_per_pair):
    """Ordered list of (id, (a, b), opening k, black, white). Games of one pair are contiguous, and
    each opening is played twice with colours swapped."""
    sched = []
    for a, b in pairs:
        for g in range(games_per_pair):
            k, swap = divmod(g, 2)
            black, white = (b, a) if swap else (a, b)
            sched.append((game_id(a, b, k, swap), (a, b), k, black, white))
    return sched


class Scheduler:
    """Hands out games to workers; stops pairs whose result is already lopsided."""

    def __init__(self, schedule, done_games, skip_min_games, skip_threshold):
        self.lock = threading.Lock()
        done_ids = {d["id"] for d in done_games}
        self.pending = collections.deque(g for g in schedule if g[0] not in done_ids)
        self.skip_min_games = skip_min_games
        self.skip_threshold = skip_threshold
        self.pair_n = collections.Counter()
        self.pair_score = collections.Counter()
        self.stopped = set()
        self.abort = None
        for d in done_games:
            self._account(d)

    def _account(self, g):
        pair = tuple(g["pair"])
        self.pair_n[pair] += 1
        self.pair_score[pair] += elo.game_score(g, pair[0])

    def lopsided(self, pair):
        n = self.pair_n[pair]
        if self.skip_min_games <= 0 or n < self.skip_min_games:
            return False
        s = self.pair_score[pair] / n
        return s >= self.skip_threshold or s <= 1.0 - self.skip_threshold

    def next_game(self):
        with self.lock:
            while self.pending and self.abort is None:
                g = self.pending.popleft()
                if self.lopsided(g[1]):
                    self.stopped.add(g[1])
                    continue
                return g
            return None

    def record(self, g):
        with self.lock:
            self._account(g)

    def remaining(self):
        with self.lock:
            return len(self.pending)


class Output:
    """Thread-safe writers for results.jsonl, the SGF files and the anomaly log."""

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.lock = threading.Lock()
        os.makedirs(os.path.join(out_dir, "sgfs"), exist_ok=True)
        self.results_path = os.path.join(out_dir, "results.jsonl")

    def log(self, msg):
        line = "[%s] %s" % (time.strftime("%H:%M:%S"), msg)
        with self.lock:
            print(line, flush=True)
            if msg.startswith("!!!"):
                with open(os.path.join(self.out_dir, "anomalies.log"), "a") as f:
                    f.write(line + "\n")

    def write_game(self, rec, sgf):
        a, b = rec["pair"]
        with self.lock:
            with open(os.path.join(self.out_dir, "sgfs", "%s_vs_%s.sgfs" % (a, b)), "a") as f:
                f.write(sgf + "\n")
            with open(self.results_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
                os.fsync(f.fileno())


def load_done(path):
    """Finished games from results.jsonl; tolerates a truncated last line from an interrupted run."""
    done = []
    if not os.path.exists(path):
        return done
    with open(path) as f:
        lines = f.read().split("\n")
    good = []
    for line in lines:
        if not line.strip():
            continue
        try:
            done.append(json.loads(line))
            good.append(line)
        except json.JSONDecodeError:
            print("warning: dropping unparseable line in %s: %r" % (path, line[:80]), file=sys.stderr)
    if len(good) != sum(1 for l in lines if l.strip()):
        with open(path, "w") as f:
            f.write("".join(l + "\n" for l in good))
    # Keep the first record of each id.
    seen = set()
    uniq = []
    for d in done:
        if d["id"] not in seen:
            seen.add(d["id"])
            uniq.append(d)
    return uniq


# -- workers ------------------------------------------------------------------

class Worker(threading.Thread):
    def __init__(self, wid, args, specs, arbiter_argv, sched, output, opening_list, stats):
        super().__init__(daemon=True)
        self.wid = wid
        self.args = args
        self.specs = {s["name"]: s for s in specs}
        self.arbiter_argv = arbiter_argv
        self.sched = sched
        self.output = output
        self.openings = opening_list
        self.stats = stats
        self.players = collections.OrderedDict()  # LRU cache of running engines
        self.arbiter = None
        self.error = None

    def engine_log(self, name):
        return os.path.join(self.args.out, "engine_logs", "%s.w%d.stderr.log" % (name, self.wid))

    def player(self, name, keep):
        if name in self.players:
            self.players.move_to_end(name)
            return self.players[name]
        while len(self.players) >= self.args.engine_cache:
            victim = next((n for n in self.players if n not in keep), None)
            if victim is None:
                break
            self.players.pop(victim).close()
        p = Player(self.specs[name], self.engine_log(name), cwd=REPO,
                   command_timeout=self.args.command_timeout, genmove_timeout=self.args.genmove_timeout)
        self.players[name] = p
        return p

    def run(self):
        try:
            self.arbiter = Arbiter(QTPEngine("arbiter", self.arbiter_argv, self.engine_log("arbiter"), cwd=REPO,
                                             default_timeout=self.args.command_timeout))
            self.arbiter.start()
            while True:
                g = self.sched.next_game()
                if g is None:
                    break
                gid, pair, k, black, white = g
                keep = {black, white}
                pb, pw = self.player(black, keep), self.player(white, keep)
                try:
                    res = play_game(self.arbiter, pb, pw, self.openings[k], max_plies=self.args.max_plies,
                                    verify=self.args.verify, log=self.output.log)
                except RefereeError as e:
                    self.output.log("!!! referee error in %s (game not recorded): %s" % (gid, e))
                    self.arbiter.engine.ensure_running()
                    continue
                sgf = res.pop("sgf")
                rec = {"id": gid, "pair": list(pair), "opening": k, "opening_moves": self.openings[k]}
                rec.update(res)
                rec["worker"] = self.wid
                rec["time"] = time.strftime("%FT%T")
                self.output.write_game(rec, sgf)
                self.sched.record(rec)
                self.stats.done(rec, self.sched, self.output)
        except VerifyMismatch as e:
            self.error = e
            self.output.log("!!! VERIFY MISMATCH, aborting run: %s" % e)
            with self.sched.lock:
                self.sched.abort = str(e)
        except Exception as e:  # noqa: BLE001 - report and stop this worker
            self.error = e
            self.output.log("!!! worker %d crashed: %r" % (self.wid, e))
            import traceback
            traceback.print_exc()
        finally:
            for p in self.players.values():
                p.close()
            if self.arbiter:
                self.arbiter.close()


class Stats:
    def __init__(self, total):
        self.lock = threading.Lock()
        self.t0 = time.time()
        self.n = 0
        self.total = total

    def done(self, rec, sched, output):
        with self.lock:
            self.n += 1
            n = self.n
        rem = sched.remaining()
        el = time.time() - self.t0
        eta = el / n * rem / 3600 if n else 0
        w = rec["winner_name"] or "draw"
        output.log("%4d done, %d left (<= %.1fh) | %s  %s(B) vs %s(W): %s, %s, %d plies, %.1fs" % (
            n, rem, eta, rec["id"], rec["black"], rec["white"], w, rec["reason"], rec["plies"], rec["seconds"]))


# -- main ---------------------------------------------------------------------

def parse_pairs(text, names):
    pairs = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        a, b = item.split(":")
        for n in (a, b):
            if n not in names:
                raise SystemExit("unknown engine %r in --pairs" % n)
        pairs.append((a, b))
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roster", default=os.path.join(HERE, "roster_default.json"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--engines", help="comma-separated subset of roster names (default: all)")
    ap.add_argument("--pairs", help="explicit pairs a:b,c:d instead of a full round-robin")
    ap.add_argument("--games-per-pair", type=int, default=40)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--engine-cache", type=int, default=3, help="max engine processes kept per worker")
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--wall-frac", type=float, default=0.2, help="probability mass of walls in openings")
    ap.add_argument("--seed", type=int, default=1, help="opening seed")
    ap.add_argument("--max-plies", type=int, default=300, help="draw after this many plies")
    ap.add_argument("--skip-min-games", type=int, default=20, help="0 disables the adaptive skip")
    ap.add_argument("--skip-threshold", type=float, default=0.975)
    ap.add_argument("--verify", action="store_true", help="compare legal_moves with the arbiter every ply")
    ap.add_argument("--command-timeout", type=float, default=120.0)
    ap.add_argument("--genmove-timeout", type=float, default=600.0)
    ap.add_argument("--anchor", default="sq-random")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true", help="print the schedule and engine commands only")
    args = ap.parse_args(argv)

    args.out = os.path.abspath(os.path.expanduser(args.out))
    os.makedirs(args.out, exist_ok=True)
    specs, arbiter_argv = load_roster(args.roster, args.out)
    if args.engines:
        want = [n.strip() for n in args.engines.split(",") if n.strip()]
        known = {s["name"] for s in specs}
        for n in want:
            if n not in known:
                raise SystemExit("unknown engine %r in --engines" % n)
        specs = [s for s in specs if s["name"] in want]
    names = [s["name"] for s in specs]
    if args.pairs:
        pairs = parse_pairs(args.pairs, names)
        used = {n for p in pairs for n in p}
        specs = [s for s in specs if s["name"] in used]
        names = [s["name"] for s in specs]
    else:
        pairs = [(names[i], names[j]) for i in range(len(names)) for j in range(i + 1, len(names))]

    check_env_resolved([s["argv"] for s in specs] + [arbiter_argv])

    schedule = make_schedule(names, pairs, args.games_per_pair)
    n_openings = (args.games_per_pair + 1) // 2

    # Record the roster and parameters (the report uses the roster order).
    with open(os.path.join(args.out, "roster_used.json"), "w") as f:
        json.dump({"engines": [{"name": s["name"], "argv": s["argv"]} for s in specs],
                   "arbiter": arbiter_argv, "args": vars(args)}, f, indent=1)

    if args.dry_run:
        for s in specs:
            print(s["name"], ":", " ".join(s["argv"]))
        print("arbiter :", " ".join(arbiter_argv))
        print("%d pairs, %d games scheduled" % (len(pairs), len(schedule)))
        return 0

    def arbiter_factory():
        eng = QTPEngine("arbiter", arbiter_argv, os.path.join(args.out, "engine_logs", "arbiter.openings.log"),
                        cwd=REPO, default_timeout=args.command_timeout)
        arb = Arbiter(eng)
        arb.start()
        return arb

    opening_list = openings.load_or_create(os.path.join(args.out, "openings.json"), arbiter_factory,
                                           n_openings, args.opening_plies, args.wall_frac, args.seed)

    output = Output(args.out)
    done = [d for d in load_done(output.results_path)]
    sched = Scheduler(schedule, done, args.skip_min_games, args.skip_threshold)
    todo = len(sched.pending)
    output.log("%d engines, %d pairs, %d games scheduled, %d already done, %d to play, concurrency %d" % (
        len(names), len(pairs), len(schedule), len(schedule) - todo, todo, args.concurrency))

    stats = Stats(todo)
    t0 = time.time()
    workers = [Worker(w, args, specs, arbiter_argv, sched, output, opening_list, stats)
               for w in range(max(1, args.concurrency))]
    for w in workers:
        w.start()
    try:
        for w in workers:
            while w.is_alive():
                w.join(timeout=1.0)
    except KeyboardInterrupt:
        output.log("interrupted; finished games are saved, rerun the same command to resume")
        with sched.lock:
            sched.abort = "interrupted"
        for w in workers:
            w.join(timeout=30)
        return 130
    wall = time.time() - t0
    if sched.stopped:
        output.log("adaptive skip stopped %d pairs early: %s" % (
            len(sched.stopped), ", ".join("%s:%s" % p for p in sorted(sched.stopped))))
    errors = [w.error for w in workers if w.error is not None]
    info = "Last invocation: %d games in %.1f min wall clock, concurrency %d." % (stats.n, wall / 60, args.concurrency)
    report = elo.make_report(args.out, names, args.anchor, args.bootstrap, run_info=info)
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(report)
    output.log("report written to %s (%s)" % (os.path.join(args.out, "report.md"), info))
    if sched.abort or errors:
        output.log("!!! run ended with errors: %s" % (sched.abort or errors[0]))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
