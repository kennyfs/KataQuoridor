# Evaluation: the Quoridor arena and Elo ladder

`python/quoridor_arena/` is a match referee that plays any set of QTP engines against each other and
fits a Bradley-Terry Elo ladder (see §6.5 of `KataQuoridor_Review_and_Roadmap.md`). It needs Python 3
with numpy (e.g. `/home/kenny/ml_venv/bin/python`); everything else is stdlib.

```bash
cd python
# Full default ladder (roster_default.json), resumable:
python -m quoridor_arena.arena --out ~/arena/ladder1
# A subset, with legal-move cross-checking at every ply:
python -m quoridor_arena.arena --out ~/arena/smoke --engines sq-random,sq-greedy,kq-s16141056-v16 \
    --games-per-pair 10 --verify
# Explicit pairs instead of a round-robin:
python -m quoridor_arena.arena --out ~/arena/gate --pairs kq-s12213504-v256:kq-s16141056-v256
# Recompute the report only:
python -m quoridor_arena.elo ~/arena/ladder1
# Tests:
python -m pytest -o addopts="" quoridor_arena/tests
```

## Components

| File | Role |
|---|---|
| `qtp.py` | Engine process wrapper: spawn, send a command, parse `=`/`?` responses (optional ids, multi-line), per-command timeout, stderr to a per-engine log, restart after a crash. |
| `referee.py` | Plays one game. Every move is mirrored into an **arbiter** and the arbiter decides legality and the result. |
| `openings.py` | Samples the random openings. |
| `arena.py` | Roster, schedule, parallel workers, resume, adaptive skip, SGF output. |
| `elo.py` | Bradley-Terry fit, bootstrap CIs, `report.md`. |
| `roster_default.json` | The default ladder. |

### Arbiter

The arbiter is a KataQuoridor process without a net
(`katago gtp -model /dev/null ... -override-config debugSkipNeuralNet=true`). The referee never
reimplements the rules:

- `play` must succeed for every move. A move the arbiter rejects, or `resign`/`pass` from `genmove`,
  loses the game for the mover (reason `illegal`, logged with `!!!` to the console and to `anomalies.log`;
  it indicates a bug).
- `winner` decides the result, and the referee checks it before every move, so a game ends at the first
  win. (Since 0.1.0, KataQuoridor refuses every move after a win with `? game is over`, so `winner` stays
  at the result, and it refuses a `play`/`genmove` by the side not to move. Older binaries accepted both;
  the referee controls the move order and stops at the first win, so it works with either.)
- Draw rule: after 300 plies with `winner` = `none`, the game is a draw (reason `draw300`). The arbiter
  has no ply cap of its own.
- `--verify`: before every move, the arbiter's `legal_moves` is compared with the mover's (only for
  engines where `known_command legal_moves` is true). A mismatch aborts the whole run.

### Openings and pairing

Engines are (near-)deterministic, so diversity comes from the referee. Each opening is K plies
(`--opening-plies`, default 4) sampled from the arbiter's legal moves: every pawn move has weight 1, and
the walls share a total mass so that walls make up `--wall-frac` of the probability (default 0.2).
Sampling is seeded (`--seed`), independent of the engine's move-listing order, and openings are distinct.
They are cached in `<out>/openings.json`. Every opening is played twice with colours swapped (a
*paired game*), so any bias of the opening cancels. The opening moves are sent to both engines with `play`.

### Scheduling

- Full round-robin over the roster (or `--engines` subset) with `--games-per-pair` games (default 40 =
  20 openings x 2 colours), or explicit `--pairs a:b,c:d`.
- Adaptive skip: once a pair has at least `--skip-min-games` games (default 20) and a score of at least
  97.5% or at most 2.5% (`--skip-threshold`), its remaining games are dropped. Such pairs carry little
  rating information.
- `--concurrency` games run in parallel (default 4). Each worker owns its own arbiter and engine processes
  and reuses them via `clear_board` (at most `--engine-cache` engines per worker). Strength is fixed by
  visits or depth, never time, so concurrency does not change results. Timeouts (`--genmove-timeout`,
  default 600 s) only catch hung engines (reason `timeout`); a dead process loses the game (reason `crash`)
  and is restarted for the next one.

### Outputs (`--out DIR`)

| Path | Content |
|---|---|
| `results.jsonl` | One JSON object per game: `id`, `pair`, `opening`, `opening_moves`, `black`, `white`, `winner` (`b`/`w`/null), `winner_name`, `margin`, `plies`, `reason` (`goal`/`draw300`/`illegal`/`timeout`/`crash`), `detail`, `moves`, `seconds`. |
| `sgfs/<a>_vs_<b>.sgfs` | One SGF per line, same format as self-play SGFs: `PB`/`PW` = engine names, `RE` = `B+<margin>` / `W+<margin>` / `0` (draw), `KM[0]`, and a root comment with `startTurnIdx=<opening plies>` so the viewer shades the opening. Open with `python sgfs_viewer/serve.py DIR/sgfs/<file>.sgfs`. |
| `report.md` | Ratings, crosstable, per-pair statistics, anomalies. |
| `openings.json`, `roster_used.json` | Parameters of the run. |
| `anomalies.log`, `engine_logs/`, `gtp_logs/` | Forfeits and referee errors; engine stderr; KataGo logs. |

**Resume:** rerunning the same command skips every game id already in `results.jsonl` (a truncated last
line from an interrupted run is dropped). Raising `--games-per-pair` adds only the new games. Changing
the opening parameters for an existing directory is refused.

**Margin** = max(1, the loser's shortest-path distance from the arbiter's `dist`), the same definition
as the engine's terminal score. Draws have margin 0.

## Ratings

`elo.py` fits a Bradley-Terry model, P(i beats j) = 1 / (1 + 10^((R_j - R_i)/400)), by maximum likelihood
(Newton's method) on all games, with a draw counted as half a win. A weak prior adds one virtual draw
to every pair that actually played, so 100% / 0% pairs stay finite. The ratings are anchored with
`sq-random` = 0 Elo (`--anchor`). Players not connected to the anchor through played pairs get no rating.

95% confidence intervals come from 1000 bootstrap resamples over paired-game units (both games of one
opening in one pair), so the two colour-swapped games are resampled together.

`report.md` contains:

- a summary: games, Black (first-player) score, draw rate, average plies, end reasons;
- anomalies: every game that did not end by reaching the goal or by a draw;
- the rating table: Elo, 95% CI, games, score;
- the crosstable: the row player's score against each column player, with game counts;
- per pair: games, score, Black win rate, draw rate, average plies, average margin, other end reasons.

## Roster format

```json
{
  "vars": {"katago": "{main_repo}/cpp/build-cuda/katago", "models_dir": "/home/kenny/q0_run/models", ...},
  "engines": [
    {"name": "sq-greedy", "command": "{simple_quoridor}", "args": ["--qtp", "--player", "greedy", "--seed", "{seed}"], "seed": 2},
    {"name": "kq-s16141056-v256", "katago_model": "run1-s16141056-d2846694", "visits": 256}
  ],
  "arbiter": ["optional argv for the arbiter; the default uses {arbiter_katago}"]
}
```

`{seed}`, `{name}`, `{out}`, `{repo}` (this checkout), `{main_repo}` and every key of `vars` are
substituted in `command` and `args`. A `katago_model` entry expands to
`{katago} gtp -model {models_dir}/<model>/model.bin.gz -config {gtp_config} -override-config
maxVisits=<visits>,{katago_overrides}`. The default overrides are `numSearchThreads=1,
allowResignation=false, ponderingEnabled=false`, logging off, and a smaller NN cache. Set
`"disabled": true` to leave an entry out.

The default roster (`roster_default.json`) contains:

- SimpleQuoridor (`quoridor_agent --qtp`): `sq-random` (uniform over all legal moves; the anchor),
  `sq-greedy` (shortest-path pawn mover, never walls), and `sq-search-d2` / `sq-search-d4` (alpha-beta at
  fixed depth, about 10 ms and about 340 ms per move on average);
- KataQuoridor run1 snapshots s1220864, s3497984, s6290688, s9327872, s12213504 and s16141056, each at
  `maxVisits` 1 (policy only) and 256, plus the latest snapshot (s16141056) at 16, 64 and 800 visits.

The names are `kq-<samples>-v<visits>`.

## Notes

- The KataQuoridor binary must match the current `main` (the NN input contract changes). Rebuild with
  `cd cpp/build-cuda && make -j katago` (see `Compiling.md`) before a ladder run if `cpp/` changed.
- For tracking training, rerun the ladder with the new snapshot added to the roster. Earlier games are
  reused because the result file is keyed by game id.
