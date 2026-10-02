# Evaluation: the Quoridor arena and Elo ladder

`python/quoridor_arena/` is a match referee that plays any set of QTP engines against each other and
fits a Bradley-Terry Elo ladder (see §6.5 of `KataQuoridor_Review_and_Roadmap.md`). It needs Python 3
with numpy; everything else is stdlib. The default roster takes the machine-specific paths from the
environment: `KATAQUORIDOR_MODELS_DIR` (the run's `models/` directory) and `SIMPLE_QUORIDOR` (the
SimpleQuoridor `quoridor_agent` binary); the KataQuoridor binaries are `cpp/build-cuda/katago` (players)
and `cpp/build/katago` (arbiter) in this checkout.

```bash
cd python
# Full default ladder (roster_default.json), resumable:
python -m quoridor_arena.arena --out ~/arena/ladder1
# A subset, with legal-move cross-checking at every ply:
python -m quoridor_arena.arena --out ~/arena/smoke --engines sq-random,sq-greedy,kq-s16141056-v16 \
    --games-per-pair 10 --verify
# Explicit pairs instead of a round-robin:
python -m quoridor_arena.arena --out ~/arena/gate --pairs kq-s12213504-v256:kq-s16141056-v256
# Komi and fence handicap for all games (see "Komi and fence handicap" below):
python -m quoridor_arena.arena --out ~/arena/komi15 --pairs kq-s12213504-v256:kq-s16141056-v256 \
    --komi 1.5 --black-walls 9
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
| `kq_ladder.py` | KataQuoridor-only ladder in one `katago match` process, sparse pairs ([below](#kataquoridor-only-ladder-kq_ladderpy)). |

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
- Draw rule: after 300 plies (`--max-plies`) with `winner` = `none`, the game is a draw (reason `draw300`).
  Since the Quoridor I/O v2 rules ([QuoridorIOv2.md](QuoridorIOv2.md)), KataQuoridor itself ends a game at
  its `maxPlies` (default 300) and `winner` reports `Draw`; the referee records that as `draw<plies>` too, so
  keep `--max-plies` equal to the arbiter's `maxPlies`. With the repetition draw on
  ([Komi and fence handicap](#komi-and-fence-handicap)), the arbiter also says `Draw` at the N-th occurrence of a
  position, and its SGF's `DR[repetition]` makes the referee record reason `repetition`. `winner` knows the game's komi, so with a non-standard
  komi the side whose pawn arrived can lose (see [Komi and fence handicap](#komi-and-fence-handicap)).
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
| `results.jsonl` | One JSON object per game: `id`, `pair`, `opening`, `opening_moves`, `black`, `white`, `winner` (`b`/`w`/null), `winner_name`, `margin`, `lead`, `result`, `komi`, `black_walls`, `white_walls`, `plies`, `reason` (`goal`/`draw300`/`illegal`/`timeout`/`crash`), `detail`, `moves`, `evals` (see [Search info](#search-info-in-arena-sgfs); null if no player is a KataQuoridor engine), `seconds`. |
| `sgfs/<a>_vs_<b>.sgfs` | One SGF per line, same format as self-play SGFs: `PB`/`PW` = engine names; `KM`, `WB`, `WW`, `RU` from the arbiter's `printsgf`; `RE` = the arbiter's result (the lead, e.g. `W+0.5`, `B+2.5`, or `0` for a draw), `B+F` / `W+F` for a forfeit, `0` for the referee's own `--max-plies` cutoff; and a root comment with `startTurnIdx=<opening plies>` so the viewer shades the opening. Open with `python sgfs_viewer/serve.py DIR/sgfs/<file>.sgfs`. |
| `report.md` | Ratings, crosstable, per-pair statistics, anomalies. |
| `openings.json`, `roster_used.json` | Parameters of the run. |
| `anomalies.log`, `engine_logs/`, `gtp_logs/` | Forfeits and referee errors; engine stderr; KataGo logs. |

**Resume:** rerunning the same command skips every game id already in `results.jsonl` (a truncated last
line from an interrupted run is dropped). Raising `--games-per-pair` adds only the new games. Changing
the opening parameters for an existing directory is refused.

**Margin** = max(1, the shortest-path distance of the pawn that did not arrive, from the arbiter's `dist`),
the same definition as the engine's margin (`Board::whiteMarginWhenWonBy`); for a forfeit, the loser's distance.
Draws have margin 0. **Lead** = |t + komi| from the arbiter's `RE` ([QuoridorIOv2.md](QuoridorIOv2.md) §2): margin
− 0.5 in the standard game; 0 for a draw, null for a forfeit. **Result** is the SGF `RE`. Arena runs before
the Quoridor I/O v2 step 3 wrote `KM[0]` and the integer margin in `RE`; those SGFs load as standard games, and
their results have no `lead` / `result` / `komi` / walls fields (the report treats them as standard games).

### Search info in arena SGFs

An engine that knows `kata-genmove_analyze` (auto-detected with `known_command`, or set `"kata": true / false` in
its roster entry) is asked for its moves with `kata-genmove_analyze <color> rootInfo true noResultValue true` instead
of `genmove`. It is the same search as `genmove` (same `genMove` path, limits and move choice); the final analysis
is printed once before `play <move>`. The root info goes on that move's SGF node in the self-play comment format
(values of the search at the position before the move, **White's** view), plus a named `lead=` token:

```
B[ln]C[0.54 0.46 0.00 3.6 v=256 lead=0.46]
```

`win loss noResult score v=<root visits> lead=<scoreLead>`. `score` is `scoreSelfplay`, the **utility score u**
(with the time bonus, so not comparable across nets trained with different `timeBonusPerPly`); `lead` is the
predicted tempo lead `s`, comparable across nets. The root info has no noResult value; it is the played move's
`noResultValue`. The engine's analysis is from the side to move (KataQuoridor's default
`reportAnalysisWinratesAs`); set `"kata_perspective": "white"` / `"black"` in the roster entry if it is configured
otherwise. Opening moves and moves of other engines (SimpleQuoridor) have no comment. `results.jsonl` has the
same per move as `evals`: `[White win, score, lead, visits]` or null. `sgfs_viewer` shows `lead` (as the margin,
in the eval graph and the stats) when present and never shows the utility `score` then; old SGFs without `lead=` fall back to `score`, labelled
"margin (no lead in file)" (in the lead-based sort only games with `lead=` count when a file has both).
Self-play and gatekeeper SGFs write the same `lead=` token (the root search's `scoreLead`, after `v=` / `weight=`).

## Komi and fence handicap

Games can use a non-standard komi (a half-integer; White wins iff tempo + komi > 0, the standard game is −0.5),
initial walls (`blackInitialWalls` / `whiteInitialWalls`, 0–10, standard 10) and the repetition draw
(`repetitionDrawCount`: a draw at the N-th occurrence of a position, standard 0 = off), see
[QuoridorIOv2.md](QuoridorIOv2.md) §2. They are set per game, layered from least to most specific:

1. the standard game;
2. the roster's `"rules"`, for all games: `"rules": {"komi": 1.5, "whiteInitialWalls": 9}`;
3. the command line, for all games: `--komi 1.5`, `--black-walls 9`, `--white-walls 9`,
   `--repetition-draw-count 3`;
4. the roster's `"pair_rules"`, for the games of one pair (either order):
   `"pair_rules": [{"pair": ["kq-a-v256", "kq-b-v256"], "komi": -1.5}]`.

Each layer only changes the keys it gives. Both games of an opening (colours swapped) use the same rules, so each
engine plays the favoured and the disfavoured side equally often.

- The referee sends them through QTP after `clear_board`, to both engines and the arbiter: `komi <k>`,
  `kata-set-rule blackInitialWalls <n>`, `kata-set-rule whiteInitialWalls <n>`,
  `kata-set-rule repetitionDrawCount <n>`. QTP rules persist across
  `clear_board`, so the referee tracks each process's rules and sends only changes (a restarted process has the
  standard rules). The arbiter decides the winner with the komi, and the SGF records `KM`, `WB`, `WW`, `RU` (with
  the repetition rule), `RE` and, for a draw, `DR` (`repetition` or `maxPlies`).
- The repetition rule is for KataQuoridor-only matches: SimpleQuoridor doesn't implement it. Result records get
  `repetition_draw_count`, and repetition draws get reason `repetition` (listed under "End reasons" and per pair,
  and in the summary's "draw rate … (by repetition …)").
- Only engines with `"supports_rules": true` can play non-standard games. It defaults to true for
  `katago_model` entries (KataQuoridor ≥ the I/O v2 rules) and false otherwise (SimpleQuoridor). A run that
  schedules an unsupported engine in a non-standard game stops at startup; an engine that rejects a rules command
  aborts the run (`!!! RULES REJECTED`).
- Game ids of non-standard games carry the rules, e.g. `a_vs_b_o000_ab_k+1.5_w9-10` (plus `_r3` with the repetition
  rule), so rerunning a directory
  with other rules plays new games; standard ids are unchanged. The report's per-pair table gets a Rules column.
- Openings are sampled on the standard board; with very few walls (fewer than the opening's walls for a side) an
  opening can be illegal, and the game is not recorded (referee error).
- With I/O v1 nets (0.1.0), komi and the walls change the game but the nets only see the walls: they have no komi
  input, so their play ignores the komi except at terminal nodes of the search. Use this for plumbing tests or
  handicap experiments with v1, and for real komi evaluation with I/O v2 nets.

## Ratings

`elo.py` fits a Bradley-Terry model, P(i beats j) = 1 / (1 + 10^((R_j - R_i)/400)), by maximum likelihood
(Newton's method) on all games, with a draw counted as half a win. A weak prior adds one virtual draw
to every pair that actually played, so 100% / 0% pairs stay finite. The ratings are anchored with
`sq-random` = 0 Elo (`--anchor`). Players not connected to the anchor through played pairs get no rating.

95% confidence intervals come from 1000 bootstrap resamples over paired-game units (both games of one
opening in one pair), so the two colour-swapped games are resampled together.

`report.md` contains:

- a summary: games, Black (first-player) score, draw rate (and the share of repetition draws), average plies, end
  reasons;
- anomalies: every game that did not end by reaching the goal or by a draw (ply limit or repetition);
- the rating table: Elo, 95% CI, games, score;
- the crosstable: the row player's score against each column player, with game counts;
- per pair: games, score, Black win rate, draw rate, average plies, average margin, other end reasons, and the
  rules (komi, walls) when some games are not standard.

## Roster format

```json
{
  "vars": {"katago": "{repo}/cpp/build-cuda/katago", "models_dir": "{env:KATAQUORIDOR_MODELS_DIR}", ...},
  "engines": [
    {"name": "sq-greedy", "command": "{simple_quoridor}", "args": ["--qtp", "--player", "greedy", "--seed", "{seed}"], "seed": 2},
    {"name": "kq-s16141056-v256", "katago_model": "run1-s16141056-d2846694", "visits": 256}
  ],
  "arbiter": ["optional argv for the arbiter; the default uses {arbiter_katago}"],
  "rules": {"komi": -0.5, "blackInitialWalls": 10, "whiteInitialWalls": 10, "repetitionDrawCount": 0},
  "pair_rules": [{"pair": ["kq-a-v256", "kq-b-v256"], "komi": 1.5}]
}
```

`rules` and `pair_rules` are optional (see [Komi and fence handicap](#komi-and-fence-handicap)); an engine entry's
optional `"supports_rules"` says whether it understands `komi` and `kata-set-rule` for the walls and the repetition
rule.

`{seed}`, `{name}`, `{out}`, `{repo}` (this checkout), every key of `vars` and `{env:NAME}` (the environment
variable `NAME`) are substituted in `command` and `args`. An unset environment variable is an error only if an
engine that is actually played (or the arbiter) needs it. A `katago_model` entry expands to
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

## KataQuoridor-only ladder (`kq_ladder.py`)

The arena runs each player as its own QTP process with one search thread, so NN batches are tiny and a ladder of
many nets at 256 visits is slow. `kq_ladder.py` rates many KataQuoridor nets with **one `katago match` process**:
one NN evaluator per model file shared by all its bots and games, many game threads, so each model sees large
batches. It is for KataQuoridor nets only (no SimpleQuoridor).

```bash
cd python
# Schedule only:
python -m quoridor_arena.kq_ladder plan   --config quoridor_arena/kq_ladder_example.json --out ~/arena/kq_ladder1
# Play the remaining games (resumable), then write results.jsonl, report.md, elo_vs_samples.png:
python -m quoridor_arena.kq_ladder run    --config C --out DIR [--limit N]
# Optional second pass: add games to the pairs whose Elo difference has the widest CI, then `run` again:
python -m quoridor_arena.kq_ladder adapt  --config C --out DIR --pairs 10 --games 40
python -m quoridor_arena.kq_ladder report --config C --out DIR
python -m quoridor_arena.kq_ladder sgfs   --config C --out DIR   # only re-sort the SGFs per pair
```

**Engine.** `katago match` with two KataQuoridor keys (`cpp/configs/match_example.cfg`): `gameListFile`, one game
per line (`<id> <blackBot> <whiteBot> <opening moves…>`), played once each from the position after its opening
(GameRunner's start-position path), instead of random pairings; and `gameResultsFile`, one JSON line per finished
game (winner, `RE`, draw reason, plies, rules, times). Without them `match` behaves as upstream. `match` writes
the SGFs per game thread (`DIR/match/sgfs/*.sgfs`, the raw copy); after each run `kq_ladder` sorts them into one
file per pair, `DIR/sgfs/<link kind>/<a>_vs_<b>.sgfs` (kinds `gen1`, `gen2`, …, `cross`, `cross-ends`, `visits`;
games in opening order, both colours together), rebuilt from the raw files each time (`kq_ladder sgfs` does only
this). The SGFs have the self-play move comments with `lead=`, the root comment
has `startTurnIdx=<opening plies>` and `gameId=<id>`. Nets of every Quoridor I/O version (v1, v2, v3) can play in
one process. The generated `DIR/match/match.cfg` sets the rules (standard komi −0.5, walls 10/10, `maxPlies` 300,
repetition rule off, `timeBonusPerPly` from the config: a rule, so one value for all games; I/O v1 nets don't
model it), no resignation, no komi compensation, 1 search thread per bot, and copies the search parameters of
`gtp_quoridor.cfg` (the arena's), so a bot plays as the arena's `katago_model` entry with the same visits.

**Openings.** The arena's (`openings.py`, `DIR/openings.json`): every opening is played twice per pair with colours
swapped, and opening k is the same for all pairs.

**Schedule.** Nets are ordered by training samples within each series (a glob or list of model directories;
samples from the `-s<N>-` in the name). All play at one visit count (`visits`). Pairs, not a round robin:

- within a series, nets at generation distance `links.generation_distances` (e.g. 1, 2, 3, 6, 12);
- across series (`links.cross`), each net of a later series against the nearest net of the first series in log
  samples, plus oldest–oldest and latest–latest;
- visit scaling (`visit_scaling`): one net at other visit counts against itself at `visits`.

Games per pair follow a Gaussian in log-samples distance, `max(min, max_games · exp(−d²/(2σ²)))`,
`d = |ln s_a − ln s_b|`, rounded up to even. Rationale: Elo is roughly linear in log samples, and a game between
players Δ Elo apart carries information ∝ p(1−p), which falls quickly with Δ; distant pairs mainly serve to tie the
chain together (so errors don't accumulate along it), for which a few games suffice. The game list is ordered by
pair, so at any time the game threads play a few pairs and each model's batches stay large.

**Ratings.** Match results are converted to arena records (`DIR/results.jsonl`; draws count half) and fitted with
`elo.py` (Bradley–Terry, bootstrap over paired-game units). The anchor is the oldest net of the first series
(`anchor` to override); `reference` adds Elo relative to another bot, with the CI of the difference;
`compare_ladder` adds that bot-relative Elo from an arena ladder (names `kq-s<N>-v<V>` mapped to the series
prefix). `report.md`: summary (Black win rate, draws, plies, runs and games/s), rating table (series, samples,
visits, Elo, CI, games, score, Black win rate, draws, avg plies), visit scaling, Elo per doubling of samples,
schedule, and the arena's per-pair table. `elo_vs_samples.png`: Elo vs samples per series (linear and log x).

**Config** (`kq_ladder_example.json`): `katago`, `gtp_config` (for the arbiter that samples openings), `series`,
`visits`, `reference`, `compare_ladder`, `links`, `games` (`max`, `min`, `sigma_log_samples`), `visit_scaling`,
`openings`, `rules`, `search` (match search keys), `match` (`numGameThreads`, `nnMaxBatchSize`, NN cache…).
Resume: `run` skips game ids already in `DIR/match/results.jsonl`; `adapt` stores its additions in
`DIR/extra_games.json`.

## Notes

- The KataQuoridor binary must match the current `main` (the NN input contract changes). Rebuild with
  `cd cpp/build-cuda && make -j katago` (see `Compiling.md`) before a ladder run if `cpp/` changed.
- For tracking training, rerun the ladder with the new snapshot added to the roster. Earlier games are
  reused because the result file is keyed by game id.
