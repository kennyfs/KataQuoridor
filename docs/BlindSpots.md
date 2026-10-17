# Blind spots: a test suite and deep-search hints for self-play

run4 self-play converged on drawn lines because a 1000-visit search rarely visited low-prior winning moves that a
10000-visit search found (see [TrainingLog.md](TrainingLog.md), "Draw convergence and root exploration"). Stronger root
exploration fixed the lines it was tuned on, but deeper Black blind spots remained, and kq_ladder (random 4-ply
openings) cannot see them. Two tools:

1. **The blind-spot suite** (`python/quoridor_arena/blindspot.py`): ~370 positions from standard self-play games, each
   labelled by a 50000-visit search. Evaluating a net = a fresh v256 / v1000 search on every clear position, and
   comparing its move with the label. It takes minutes per net.
2. **Deep-search hints** (`python/quoridor_arena/hintminer.py`): positions where the self-play search and a deep search
   disagree, as KataGo `hintPosesDir` files, so that self-play starts some games where it is blind.

Everything is search-based and uses the `gtp` engine (`cpp/configs/gtp_quoridor.cfg`: komi −0.5, 10 / 10 walls,
repetition rule off), so it runs with any KataQuoridor binary and net. Moves are in 9×9 notation (`e8` pawn,
`e2h` / `e2v` walls).

**The labels are only as good as the deep search.** A 50k-visit search of one net has its own blind spots: where it
agrees with a weaker search, the position is easy for both, and where the net is wrong in both, the label is wrong.
The label net and visits are stored in the suite (`meta`), and every position keeps the deep winrate of its children,
so a suite should be **relabelled with a much newer net** from time to time (`label` on the same `positions.jsonl`;
keep the old suite file for comparisons: agreement numbers are only comparable within one suite version).

## 1. The suite

### Positions (`build`)

From the standard games (komi −0.5, 10 / 10 walls, `gtype=normal`, any repetition rule) of the newest self-play model
directories (`~/q1_run/run4/selfplay/<model>/sgfs/*.sgfs`):

| Group | What | Selection |
|---|---|---|
| `opening` | opening-tree nodes of the common lines, plies 2–24 | most common positions first (a node counts once per game; games must have searched it: ply ≥ `startTurnIdx`), at most `n_opening/8` per ply so that deep nodes are not crowded out; `count` = games reaching it |
| `swing` | the self-play search value (White's win probability in the move comments, full searches only: ≥ 500 visits) changed by more than 0.3 within 4 plies | a uniform random sample of all such positions (the position where the swing starts) |
| `random` | plies 4–80 | a uniform random sample of all searched positions |

Positions are deduped by *position* (both pawns, the walls and who placed them, side to move; so transpositions merge)
and by the **left–right mirror** (`a ↔ i`, wall anchors `a ↔ h`); a position in several groups keeps all of them in
`groups`, and `group` is the first of opening / swing / random. `symmetric` marks mirror-symmetric positions, where a
move and its mirror image are the same move (they are treated as equal when judging a net).

```bash
cd python
D=~/q1_run/run4/selfplay
python -m quoridor_arena.blindspot build --dirs $D/run4-s18713600-d8123920 $D/run4-s19619840-d8274875 \
  $D/run4-s20387072-d8403664 --out positions.jsonl     # 1-2 s
```

### Labels (`label`)

A fresh 50000-visit search (gtp params, 16 threads, `clear_board` + moves + `clear_cache` per position) per position.
Stored in `label`: the best move (the engine's own choice, `order 0`), its visits, visit `share`, winrate, lead and
LCB; every child with ≥ 100 visits (`children`: visits, winrate, lead, LCB, prior; winrates and leads are from the
side to move); the second-best move (the rival with the best winrate among children with ≥ 1% of the visits), and
`gap_wr`, `gap_lcb` (best LCB − second winrate), `gap_lead`.

* `clear`: `gap_wr ≥ 0.05`, or the position is decided (best winrate beyond 0.95 / below 0.05) and `gap_lead ≥ 0.5`,
  or the best move is the only one with a real visit share (`forced`). Only clear positions are evaluated by
  default (`--include-unclear` adds the rest).
* `forced`: no rival has 1% of the visits: an easy position (reported separately as `forced` / `contested`).

The output is JSONL: a first line `{"meta": {...}}` (label net, visits, version, rules), then one position per line.
`label` appends and skips the ids already in `--out`, so an interrupted run resumes.

```bash
python -m quoridor_arena.blindspot label --positions positions.jsonl --out suite.jsonl \
  --model ~/q1_run/run4/models/run4-s20387072-d8403664/model.bin.gz --visits 50000 --engines 1 --threads 16 --cache-bits 22
```

### Evaluation (`eval`)

For each clear position a fresh search of V visits (`--visits 256,1000`) with `--params`:

* `gtp`: `gtp_quoridor.cfg` as it is (4 threads unless `--threads`; the default);
* `selfplay`: the self-play search parameters of `selfplay_quoridor_v2.cfg` (1 thread, forced playouts 4, root policy
  temperature 1.6 / 1.5, LCB selection, root noise 10.83 / 0.25), so a result is a sample of what a self-play game
  searches;
* `selfplay-nonoise`: the same without the root noise (what the miner uses for its fresh search).

The net's move is the engine's top move. Reported (`eval_*.json`, per-position `eval_*.jsonl`, `report` makes tables):

* **agreement**: the top move equals the label's best move (± the binomial standard error); in the `opening` group
  also `agree_weighted` by the number of games reaching the node;
* **regret**: label winrate of the label move minus the label winrate of the chosen move, when the deep search gave
  the chosen move ≥ 100 visits (else *unknown*: counted separately, not as 0); the mean is over the known ones;
* the same split by group, by `forced` / `contested`, and by ply range (0–8, 9–16, 17–24, 25–40, 41–80, 81+).

`--engines N` runs N engine processes on the GPU at once (default 4).

```bash
python -m quoridor_arena.blindspot eval --suite suite.jsonl --model MODEL/model.bin.gz --visits 256,1000 \
  --params gtp --out-dir out --engines 4
python -m quoridor_arena.blindspot report out/eval_*.json
```

### As Colab experiment jobs

`python/colab/stage_exp.sh JOBDIR TEMPLATE inputs...` builds a job directory from `python/colab/exp_examples/`
(`run.sh` + the `quoridor_arena` package + `gtp_quoridor.cfg` + inputs); then, as in
[python/colab/README.md](../python/colab/README.md) ("Experiment jobs"):

```bash
cd python/colab
# labelling: one net, positions.jsonl; out/suite.jsonl
./stage_exp.sh /tmp/job_label blindspot_label ~/arena/blindspot/positions.jsonl MODEL_DIR
# evaluation: any number of nets, suite.jsonl; out/eval_*.json, out/report.md
./stage_exp.sh /tmp/job_eval blindspot_eval ~/arena/blindspot/suite.jsonl NET1_DIR NET2_DIR NET3_DIR
rsync -a /tmp/job_eval/ ws4:tmp2/kq_remote/share/exp/bs_eval/
ssh ws4 'cd ~/tmp2/kq_remote/launch && setsid -f ./colab_job.sh bs_eval L4 > /dev/null 2>&1 < /dev/null'
# results (every 2 minutes and at the end): ws4:tmp2/kq_remote/share/exp/bs_eval/out/
```

`run.sh` reads `VISITS`, `ENGINES`, `THREADS`, `PARAMS` from the environment (put them in `run.sh`).

## 2. Deep-search hints for self-play

### The mechanism for Quoridor

KataGo's `hintPosesDir` / `hintPosesProb` (`GameInitializer`, `cpp/program/play.cpp`): a game starts at a loaded
position; at its first move the search gets 4× the visits, the hint move gets +2% of the root policy and is forced to
be searched almost as often as the most-searched move (`Search::setRootHintLoc`); the value target of that move is
copied from the next turn; if the move played differs from the hint, a `hintfork` game from the position after the hint
is queued (`maybeHintForkGame`). The SGF `gtype` is `hintpos` (`hintfork` for the forks), so the analysis tools that
select `gtype=normal` ignore them. The file format is one `Sgf::PositionSample::toJsonLine` per line in
`*.hintposes.txt`.

Checked, and what Quoridor needed:

* **The move list replays.** `Board::parseBoard` returns the initial board, so a JSON sample is *the empty board plus
  its move list*: a hint position must list all its moves from the start (`initialTurnNumber` 0); `toJsonLine` of a
  mid-game `PositionSample` would lose the walls (only SGF start positions, which are in memory, keep them: below).
  Locations are 9×9 notation both ways (`Location::toString` / `ofString`). Tested: a line with walls round-trips, and
  the game replays it to the right position (`runtests quoridorselfplay`).
* **Walls.** The sample path left the board's wall counts at the default; it now sets them from the rules like a game
  from the empty board (matters for a non-10/10 config).
* **Komi.** The Go komi helper (`setKomiWithNoise`) with `komiStdev 0` gives the standard komi; `quoridorKomiRandomProb`
  and the fence handicap apply only to games from the empty board, so a hint game has komi −0.5 and 10 / 10 walls.
  The deep search that found a hint used komi −0.5, so that is what the hint is valid for. `sgfCompensateKomiProb`
  (which defaults to `forkCompensateKomiProb`, 0.8 in the v2 config, where it is set to 0 for SGF positions) would have
  moved the komi with a fair-komi search: a hint game now never asks for it, whatever the config says.
* **Repetition rule.** It was not drawn for sample games (`quoridorRepetitionDrawProb` applied to games from the empty
  board only), so hint games ran under the config's fixed `repetitionDrawCount` (3). Sample games now draw the
  per-game rule like empty-board games (75% on, N = 3 or 4; the hint is for the position, which the rule does not
  change; the mix keeps the data in line with the rest). Forks of a hint game keep its rule, as all forks do.
* **`nextPla` is the player to move at the start of the move list**, not at its end (Black, from the empty board).
  The miner first wrote the player to move at the end: every odd-length hint was then unplayable (the replay stopped
  at the first move, and the game became a plain `sgfpos` game; 43% of hint games). The unit test used only
  even-length lists; the end-to-end self-play run found it. Both are fixed and tested with an odd-length hint.
* **`hintLoc` reaches the root search.** It is set for the first search only when the root has the hint turn's
  move count and position hash (`hintPosHash`); a hint that is not legal at the position (stale file) is dropped, and
  so is a hint whose move list stops being playable. The test plays a short game from a hint position with a random net:
  the first search has 4× visits and prints `HintLoc e6` once, the next ones do not, and the SGF mode is `hintpos`
  (`sgfpos` when the hint is dropped).
* **Statistics.** `gtype=hintpos` games are not counted as standard games by `cycles.py` / `blindspot.py` (`gtype`
  filter). They also lower the cheap-search probability for the 6 turns after the hint (`play.cpp`).

### The miner (`hintminer.py mine`)

```bash
python -m quoridor_arena.hintminer mine --dirs NEWEST_SELFPLAY_DIR... --model MODEL/model.bin.gz --out-dir HINTS
```

1. **Expire old hints.** The hint files of the last `--keep-runs` (4) mining runs stay (`mine_<time>_<net>.hintposes.txt`);
   before mining, every old hint is re-searched with the current net (1000 visits, self-play params, no noise) and
   dropped if the net now plays the hint move.
2. **Positions** from the newest self-play's standard games: `--n-opening` (250) opening-tree nodes (plies 2–24, ≥ 10
   games) and `--n-swing` (150) positions before a value swing (full searches only), the same selection as the suite.
3. **Fresh search** (`--fresh-visits` 1000, self-play root params, no noise) and **deep search** (`--deep-visits` 10000,
   gtp params, 16 threads), both fresh per position.
4. **A hint** when the deep best move differs from the fresh best move (mirror images are the same move), the deep
   label is clear and the deep best has ≥ 15% of the deep visits, and it is better by a margin by the deep search's own
   numbers: `winrate(deep best) − winrate(fresh move) ≥ 0.05`, or, in a decided position (both beyond 0.95 / below
   0.05), lead margin ≥ 2.0. The hint line has all moves from the start, `hintLoc` = the deep move, `weight` =
   margin / 0.05 clipped to 1–5 (1 for a lead margin), and the numbers in `metadata`.
5. `mine_<time>_<net>.hintposes.txt` is the new file; `check` re-searches a directory of hints with a net.

As a Colab job (`stage_exp.sh JOBDIR hint_mine MODEL_DIR SELFPLAY_DIR... HINT_DIR`): `out/hints/` has the kept and
new files; copy them into the directory the config points at (below).

### Turning it on

The v2 self-play config has the lines commented out (`hintPosesDir`, `hintPosesProb = 0.03`, suggested 0.02–0.05).
Turn it on:

1. Mine (or copy a mined directory) into a directory, e.g. `~/q1_run/run4/hintposes/` (with no `*.hintposes.txt` file in it the
   self-play logs "No hint positions loaded" and runs without hints).
2. Uncomment the two lines in `cpp/configs/training/selfplay_quoridor_v2.cfg` (the loop copies the working-tree
   config) and point `hintPosesDir` at the directory; rebuild the self-play binary from a branch that has the
   `play.cpp` change above (without it, hint games run under the config's fixed repetition rule and the walls stay
   default). The files are read at start-up: **restart the self-play** after each mining run. Remote Colab workers
   have their own copy of the config (`publish_share.sh`) and need the hint files in their share.
3. Mining takes ~30 min on the home GPU / L4 for 400 positions. Every few loop cycles, run it with the newest net
   and the newest self-play directories, and restart self-play with the new files.

A hint game is 3% of the games with a 4× first search; the training data of the position itself carries the deep
search's move in the policy target (forced playouts on the hint, then a normal game), and the hint fork plays the
hint when the game did not.

### SGF start positions (`startPosesFromSgfDir`)

Report only; nothing changed in the loop. Quoridor SGF positions work in memory: `Sgf::loadAllUniquePositions` of a
Quoridor game gives samples with real boards (walls included; the sample is the board 5 turns ago plus those moves),
and a game starts from them with the right position, wall counts and side to move (tested against a straight replay
for every ply of a game). Limits:

* `startPosesFromSgfDir` collects `*.sgf` files (one game per file); self-play writes `*.sgfs` (many games, one per
  line): split them first.
* A sample from an SGF is the board a few turns earlier, so all positions of a game are separate samples
  (`startPosesTurnWeightLambda` weights by turn); `startPosesLoadProb` thins them.
* The rules of the SGF (fence handicap, repetition) are not carried: the game gets the config's rules (with the
  repetition mix now), and its board keeps the walls it had, so a handicapped game's start position and the rules'
  initial walls disagree. Use standard games only.
* Start poses also get the policy-init moves (`startPosesPolicyInitAreaProp`) and, if `sgfCompensateKomiProb > 0`, a
  fair-komi search.

As a source of opening diversity it is a heavier tool than needed: the hint games above (or `initGamesWithPolicy`
with a higher temperature) change the opening distribution too; start poses from *old* games would put the old
distribution back. See the results below for how concentrated the openings are.

## 3. Results

The suite (v1) is `python/quoridor_arena/data/blindspot_suite_v1.jsonl`: labels by run4-s20387072 at 50000 visits
(2026-10-18), positions from the standard games of run4-s18713600, s19619840 and s20387072.

### Suite composition and labelling cost

370 positions: `opening` 150 (plies 2–24, reached by ≥ 10 games), `swing` 98, `random` 122 (46 positions are in more
than one group; ply ranges 0–8: 29, 9–16: 94, 17–24: 92, 25–40: 88, 41–80: 67). After labelling, **239 are clear**
(opening 100, swing 64, random 75) and 131 unclear; of the clear ones **155 are forced** (no rival with 1% of the
50000 visits: easy), **84 contested** (opening 42, swing 30, random 12). Evaluations use the 239 clear positions;
the contested ones are where nets differ.

Labelling: **752 s for 370 positions on one L4** (50000 visits, 2 engines × 16 threads, ~2 s per position; the
home RTX 3070 is about as fast: 81 positions in ~3 minutes with 1 engine). One eval run of one net (239 positions)
takes 16 s (v256, gtp params) to 99 s (v1000, self-play params, 1 thread per search), i.e. the whole table below (3 nets
× 2 sizes × 2 param sets) is ~9 minutes of searching on an L4 plus the VM start. The miner's 370 positions
(fresh 1000 + deep 10000 visits) took 468 s.

### Three nets (Colab L4, `bs_eval`)

Top-move agreement with the label (n = 239 clear positions, ± about 2.5 points; the contested column has n = 84, ± 5.4):

| net | visits | params | agree | contested | regret (known) | opening | swing | random |
|---|---|---|---|---|---|---|---|---|
| s17109248 | 256 | gtp | 75.3% | 39.3% | 0.059 (236) | 71.0% | 68.8% | 86.7% |
| s17109248 | 1000 | gtp | 78.7% | 46.4% | 0.043 (233) | 71.0% | 73.4% | 93.3% |
| s18713600 | 256 | gtp | 75.7% | 42.9% | 0.051 (234) | 76.0% | 57.8% | 90.7% |
| s18713600 | 1000 | gtp | 75.3% | 41.7% | 0.057 (236) | 71.0% | 64.1% | 90.7% |
| s20387072 | 256 | gtp | 78.2% | 44.0% | 0.055 (237) | 77.0% | 64.1% | 92.0% |
| s20387072 | 1000 | gtp | **84.1%** | **59.5%** | 0.043 (238) | 85.0% | 75.0% | 90.7% |
| s17109248 | 256 | selfplay-nonoise | 69.5% | 35.7% | 0.068 (227) | 63.0% | 60.9% | 85.3% |
| s17109248 | 1000 | selfplay-nonoise | 73.2% | 38.1% | 0.057 (227) | 68.0% | 57.8% | 93.3% |
| s18713600 | 256 | selfplay-nonoise | 71.5% | 42.9% | 0.069 (230) | 76.0% | 43.8% | 89.3% |
| s18713600 | 1000 | selfplay-nonoise | 75.7% | 46.4% | 0.051 (230) | 70.0% | 64.1% | 93.3% |
| s20387072 | 256 | selfplay-nonoise | 80.8% | 54.8% | 0.039 (233) | 82.0% | 67.2% | 90.7% |
| s20387072 | 1000 | selfplay-nonoise | 81.6% | 52.4% | 0.039 (233) | 81.0% | 67.2% | 94.7% |

By ply range (gtp params, v1000): s17109248 68 / 71 / 65 / 86 / 100% for plies 0–8 / 9–16 / 17–24 / 25–40 / 41–80;
s18713600 68 / 70 / 61 / 78 / 98%; s20387072 84 / 82 / 72 / 86 / 98%. The label net is s20387072 itself, so its
own numbers are biased upwards (its 1000-visit search agrees with a 50000-visit search of the same net more than
another net's would): compare the older nets with each other, and treat the gap to s20387072 as an upper bound.
Findings:

* The weak spot is the opening and middlegame (plies 9–24) and the `swing` group (65–75%), and contested positions
  (40–60%); late positions (plies 41+) and `random` positions are ~90–100% (mostly forced).
* s17109248 → s18713600 is flat or slightly worse on this suite (−3 points at v1000), s18713600 → s20387072 is +9
  at v1000 (+18 points on the contested positions), in line with the exploration change that came between them. kq_ladder Elo did not move
  in that range (+7 ± 30): this suite does.
* A bigger search helps the label net more (v256 → v1000: +6 points gtp, +1 selfplay-nonoise) than the older ones;
  the self-play params without noise are 2–6 points worse than gtp params for the older nets (temperature 1.5 and
  forced playouts spread visits, and with 1 thread the search is shallower).
* Differences between single cells are within 1–2 standard errors; read the pattern, not one cell. Repeat with the
  `selfplay` (noise) param set and several seeds for a tighter comparison.

### First miner output (s20387072, `bs_mine`, Colab L4)

From 370 candidate positions (250 opening-tree nodes and 150 swing positions requested; 12039 standard games of
s19619840 and s20387072), the fresh 1000-visit and deep 10000-visit best moves differed in 98; **55 hints** (31
from opening-tree nodes, 24 from swing positions; all by the winrate margin, none by lead), median margin 0.19, 22 at
the maximum weight 5. Plies: 4 at 0–7, 17 at 8–15, 21 at 16–23, 8 at 24–31, 5 beyond. Examples (moves from the start,
9×9 notation; margin = deep winrate of the deep move minus that of the fresh move, in the deep search):

| moves | fresh best | hint | margin |
|---|---|---|---|
| e8 e2 e7 e3 e6 e4 a6h c3h c6h e6v | e3v | e4v | 0.29 |
| e8 e2 e7 e3 e6 e4 a7h f2h e4v | d5v | e5 | 0.20 |
| a7h e2 e8 e3 e7 | e6h | e4 | 0.18 |
| e8 e2 e7 e3 e6 e4 e6h h3h | e5 | e4v | 0.15 |
| e8 e2 e7 e3 f7h d6h e8v | e6v | f3h | 0.15 |
| e8 e2 e7 e3 c7h d6h f7 f6h h8h | a7v | f7v | 0.14 |

The first row is one of the blind spots the investigation found by hand (`... e6 e4 a6h c3h c6h e6v`: 1k e3v, 10k e4v),
found automatically. The biggest margins (0.5–0.95) are middlegame positions at plies 22–40 where the fresh search
plays a move that loses against the deep search's reply; a few of those may be positions the fresh search could
solve with more visits, which is what the expiry check (fresh search finds the hint) is for.

An end-to-end self-play run (CUDA build of this branch, `hintPosesProb = 0.5`, 100 visits) loaded the 55 hints
("Loaded 55 hint positions"), and 22 of 60 games were `hintpos`, 12 `hintfork`, 24 normal, 2 forks, 0 `sgfpos`
(a dropped hint), with komi −0.5 in all of them and the repetition rule on in most.

### Reproducing

`~/arena/blindspot/` has the positions, the L4 outputs and the mined hint file of this run
(`mine/hints/mine_20261017-175418_run4-s20387072-d8403664.hintposes.txt`; the time in the name is UTC).
Jobs used: `bs_label_full` (blindspot_label), `bs_eval` (blindspot_eval), `bs_mine` (hint_mine), L4.
