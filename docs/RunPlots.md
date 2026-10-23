# Run plots (Quoridor Duel)

`python/quoridor_arena/runplots.py` draws the plots used to watch a Quoridor Duel training run (q1, q2): opening frequencies, Black score
by opening line, per-net self-play stats and the training losses. It reads the run directory only (self-play SGFs,
`train/NAME/metrics_*.json`) and needs no GPU. Run it with `~/ml_venv` (matplotlib, numpy); it imports
`quoridor_arena.openings_from_selfplay` from the KataQuoridor repo.

## Commands

```bash
T=~/sync/HW/Quoridor/KataQuoridor/python/quoridor_arena/runplots.py; R=~/q2_run; M6="e8 e2 e7 e3 e6 e4"
# 1. standard games of every net (komi -0.5, 10/10 walls, no compensated / random komi), first 16 plies
python $T extract --dirs $R/selfplay/q2b*-s* --out q2.jsonl
# 2. opening frequencies (games without policy-init plies), a board diagram per line
python $T openings --data q2.jsonl --out openings.png --title q2 --event 4903432=b5 \
  --line "Race to the centre=$M6" --line "c7h f2h=$M6 c7h f2h" --line "a6h e2h=$M6 a6h e2h"
# 3. Black score per line
python $T score --data q2.jsonl --out black_score.png --title q2 --event 4903432=b5 --line "centre=$M6"
# 4. per-net self-play stats; --compare draws another run's last nets as dotted levels
python $T selfplay --dirs $R/selfplay/q2b5-s* --compare $(ls -d ~/q1_run/run4/selfplay/run4-s* | sort -t- -k2.2n | tail -5) \
  --out selfplay.png --json selfplay.json --title q2
# 5. losses: train (moving average) vs val
python $T loss --traindir $R/train/q2b5 --out loss.png --event 9459968=loop --ignore-first 4e6
```

## Conventions

- x axis: self-play data rows (the `-d<rows>` of the net's dir name) for self-play plots, so runs and net sizes
  line up; training samples only for the loss plot. Sample counts restart with a new net size (b4 -> b5).
- Lines are 9x9 notation, mirror-canonical: the first move that differs from its left-right mirror is the smaller
  string (`a6h`, not `h6h`), so a line and its mirror count together.
- Opening frequencies use only games with no policy-init plies (about 2% of the games, ~200 per net), so
  neighbouring nets are pooled to >= 600 games per point (`--pool`).
- Choosing lines: list the most common continuations first, e.g. the 7th-12th plies after the race to the centre
  per rows range (see `extract` output), then pick the lines that rise and fall. The standard game has stayed in the
  race to the centre (six pawn moves) since about 2M rows in q2; the interesting moves are plies 7-12.
- Policy KL to the target is `p0loss - ptentr`. The loop's val set is 1% of each cycle's window (~14K rows) and
  drifts with the data; the bootstrap's val set was fixed.
- Self-play "decided": the first ply after init from which every searched move has training weight <= 0.1.

## Related checks (not in runplots.py)

- Ladders: `python -m quoridor_arena.kq_ladder` (configs in `~/arena/q2ladder/`, `plot_x: "rows"`).
- A position where self-play results swing: deep search with several nets via `quoridor_arena.blindspot`'s
  `make_engine` / `run_search` (example: `~/arena/q2b5/fj/`).

## In the sgfs viewer

The cheap per-file versions are on the viewer's Statistics page (`python/sgfs_viewer`): the standard game by
policy-init length (Black score, the first search's predicted score), compensated and random komi (mean komi, Black
score, by komi value) and the standard-game line tree (no-init games, mirror-merged, two plies per step, same
canonical form as `runplots.py`). Trends across nets stay in `runplots.py`.
