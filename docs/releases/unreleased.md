# Unreleased (draft for the next release)

*Draft notes for changes on the way to the Quoridor I/O v2 retrain. See
[docs/QuoridorIOv2.md](../QuoridorIOv2.md) for the design; this covers step 1 (rules and scoring).*

## Rules and scoring (Quoridor I/O v2, step 1)

KataQuoridor now has three rules of its own on top of Quoridor Duel ([GameRules.md](../../GameRules.md)):

- **Draw after 300 plies.** A game that reaches `maxPlies` plies (default 300, counted from the start of the game)
  without a pawn on its goal row is a **draw**. It is a game rule decided by the engine (`BoardHistory`), so search
  sees it at terminal nodes too, not only self-play and the arena. A pawn reaching its goal on ply 300 still wins.
  QTP `winner` reports `Draw`, and nothing more can be played.
- **Tempo and komi.** A finished game is scored in tempo: `t` = the loser's remaining distance if White's pawn
  arrived, and 1 minus it if Black's did (so an equal race, B+1, is `t = 0`). Komi is a half-integer and **White
  wins iff `t + komi > 0`**. The **standard game has komi −0.5**, under which the first pawn to arrive wins, as
  before. With another komi, the side whose pawn arrived can lose.
- **Fence handicap.** The initial walls of each player are configurable (0–10, default 10 each).

### Scores: `scoreLead` vs `scoreMean`

- `scoreLead` is the **tempo lead** `t + komi` (e.g. +0.5: White wins by one tempo). **GUIs should show it.**
- `scoreMean` (KataGo's score, what search and training use) is the **utility score**: the lead plus a time
  bonus `sign(lead) × timeBonusPerPly × (maxPlies − plies)`, which rewards the winner for finishing earlier and the
  loser for lasting longer; a win always scores above a draw. `timeBonusPerPly` defaults to 0. With a time bonus,
  `scoreMean` reads around `timeBonusPerPly × 300` at the start (e.g. +15 with 0.05) while `scoreLead` is about 0.
  That is expected. The time bonus is a property of a training run: use the value the net was trained with.
- As in upstream KataGo, `kata-analyze` and the analysis engine print the utility score as `scoreSelfplay`, and
  their `scoreMean` field is a copy of `scoreLead`.

### What changes with the default settings

With the defaults (`maxPlies = 300`, komi −0.5, no time bonus, 10/10 walls), the only changes are:

- terminal scores are the lead, **the old margin ∓ 0.5** (W+3 scores +2.5, B+1 scores −0.5). Existing I/O v1 nets
  (e.g. the 0.1.0 release net) still predict the margin; the half-tempo shift is harmless for them and they keep
  loading and playing as before;
- the **300-ply draw rule** in the engine.

Non-standard komi or a time bonus with a v1 net is possible but not meaningful: v1 nets have no komi input and
were not trained with a time bonus.

## QTP

- **`komi <k>` and `get_komi` are back**, for the Quoridor komi (half-integers only, |komi| ≤ 20.5; `komi 0` or
  `komi 7` are errors). Like upstream, komi may change mid-game, and changing it after the game re-scores it.
- **`kata-get-rules` / `kata-set-rule` / `kata-set-rules`** show and set `maxPlies`, `timeBonusPerPly`,
  `blackInitialWalls` and `whiteInitialWalls` (JSON, or `Quoridor:key=value,...`). `maxPlies` and the initial
  walls can only change before the first move.
- **`winner`** returns `Draw` after the 300-ply draw.
- **`showboard`** prints `Ply: N (draw at 300, K left)`, the komi when it isn't −0.5 and the time bonus when it
  isn't 0, and the result of a finished game (e.g. `Game finished: winner = Black (B+2.5)`).
- `set_position` and `loadsgf` count plies from 0 at the start of the position (they used to start at 2).

## SGF

- New root properties: **`WB[n]` / `WW[n]`**, the initial walls of Black and White (always written; 10 if missing).
- **`KM`** is written and read as the Quoridor komi (standard `KM[-0.5]`). SGFs without `KM`, and `KM[0]` as written
  by 0.1.0, load as the standard game.
- **`RU`** is `Quoridor` for the standard rules, or `Quoridor:maxPlies=400,timeBonusPerPly=0.05,...` listing the
  non-default rules.
- **`RE` is now the lead, a half-integer:** `W+0.5`, `B+2.5`, … (it was the integer margin, `W+1`, `B+3`), and `0`
  for a draw. `python/sgfs_viewer` handles both.

## Self-play, gatekeeper, match

- **`maxPlies`** (config key, default 300) replaces `maxMovesPerGame` / `cutoffMoves`. They may still be given but
  must equal `maxPlies`, otherwise `selfplay`, `gatekeeper` and `match` fail at startup.
- **Drawn games are training data** (value 0.5/0.5, see the I/O v2 section below) and are reported as draws in the
  self-play stats.
- **Self-play stats log:** the `Quoridor stats` line reports Black's score with a draw counted as half a win, by komi
  (with each komi's draw rate) and, for the standard komi, by repetition rule; the old "black win rate in normal
  standard games" (wins over all games), the fence-handicap slice and KataGo's `Game stats for ...` line are gone.
  See [SelfplayTraining.md](../../SelfplayTraining.md#quoridor-self-play-stats).
- **`komiMean` must be a Quoridor komi.** The Quoridor configs now say `komiMean = -0.5`; the old `komiMean = 0`
  fails at startup with an explanation. Update copies of `selfplay_quoridor.cfg` in running training directories
  before using a new binary.
- New optional config keys (also in `gtp_quoridor.cfg`, commented out): `maxPlies`, `timeBonusPerPly`,
  `blackInitialWalls`, `whiteInitialWalls`. For gtp, komi is set with the `komi` command (or forced with
  `ignoreGTPAndForceKomi`, which must be a Quoridor komi too).

## Arena

- Arena SGFs carry the search info of KataQuoridor engines on each move (`kata-genmove_analyze`): the self-play
  comment `win loss noResult score v=..` from White's view plus `lead=<scoreLead>`; `results.jsonl` has a per-move
  `evals` list. Self-play and gatekeeper SGFs also write `lead=`. `sgfs_viewer` shows and prefers `lead=` as the margin when present ([Evaluation.md](../Evaluation.md)).

- The arbiter now reports `Draw` itself at 300 plies; the referee records it as `draw300` as before. Keep
  `--max-plies` equal to the arbiter's `maxPlies`.

## Internals

- Search: the one change in `cpp/search/` reports the history's lead (not the score) as `lead` at finished-game
  terminal nodes.
- The graph-search hash and the NN cache hash now include the rules (komi, `maxPlies`, time bonus) and the ply
  count, since the same board at a different ply can have a different value.
- Upstream komi helpers were made correct for the Quoridor komi (see the table in
  [QuoridorIOv2.md](../QuoridorIOv2.md) §6.3). In particular, terminal score variance is no longer "gridded" the Go
  way.
- New `runtests quoridorv2` test group; `python/tests/test_qtp_rules.py` covers the new QTP behaviour.

## Neural net I/O v2 (Quoridor I/O v2, step 2)

Nets now declare a **Quoridor I/O version** (model option D). Version 2 is the net for the upcoming retrain; the
details are in [QuoridorIOv2.md](../QuoridorIOv2.md) §4.

- **New inputs (I/O v2):** two planes of the legal wall placements (vertical, horizontal), the plies left until the
  300-ply draw, and the komi from the side to move's view. v2 nets therefore see komi and the ply limit.
- **Separate score and lead:** a v2 net's `scoreMean` predicts the utility score (with the time bonus) and its
  `scoreLead` the tempo lead, from two heads. A new training-only head predicts how many plies the game has left.
- **I/O v1 nets (0.1.0) still work** in GTP, analysis, match and the arena, with their v1 inputs. For them
  `scoreLead` and `scoreMean` are the same number, the margin, and don't react to komi or a time bonus.
- **Training is v2 only.** Self-play writes I/O v2 training data (`inputsVersion` 2; other values are refused), the
  trainer refuses v1 models, and v1 training data can't be mixed in. Use the new presets `b2c64_quoridor_v2` or
  `tf2_b4c192_quoridor_v2` (`-model-kind`); the presets without `_v2` remain for exporting and inspecting v1 nets.
- **Draws are training data.** Games drawn at 300 plies are written like any other game (value 0.5 / 0.5, score 0,
  no lead target), with side positions and reanalysis. The self-play log reports them as draws, with a draw
  rate, instead of "hit cutoff".
- Training metrics: `smloss` (utility score), `leadloss` (lead) and the new `rtloss` (remaining plies) replace
  `gmloss`.
- Tools: `dumpnninputs -io-version`, `writesampletrainquoridor` (now with draws and komi), and the NN parity test
  cover both I/O versions. `python/model_viewer` is v1-only for now.

## Self-play setup, gatekeeper, arena (Quoridor I/O v2, step 3)

See [QuoridorIOv2.md](../QuoridorIOv2.md) §5. All new randomizations and compensations are off by default; the
existing configs create the same games as before.

- **New configs** `cpp/configs/training/selfplay_quoridor_v2.cfg` and `gatekeeper_quoridor_v2.cfg` for training
  I/O v2 nets: time bonus `timeBonusPerPly = 0.05` (a placeholder until the λ experiments),
  `staticScoreUtilityFactor = 0`, `maxPlies = 300`, and the randomizations below.
- **Komi randomization** (`quoridorKomiRandomProb`, `quoridorKomiRandomWeights`): in the v2 config, 30% of normal
  self-play games have komi −0.5 ± 1, 2 or 3 (weighted 60/30/10), the rest the standard −0.5.
- **Fence-handicap randomization** (`quoridorFenceHandicapProb`, `quoridorFenceHandicapWeights`): in the v2 config,
  10% of normal games give one side 9, 8 or 7 walls (60/30/10), independently of the komi.
- **Komi compensation** (`forkCompensateKomiProb` 0.8 for fork games, `handicapCompensateKomiProb` 0.5 for
  fence-handicap games in the v2 config) sets a fair komi with `adjustKomiToEven`. It needs nets of Quoridor I/O
  version ≥ 2; **with I/O v1 nets (e.g. 0.1.0) it is skipped** with a one-time warning. A fair komi more than 10
  tempi from the standard −0.5 is not applied (the game keeps its komi), so an untrained net can't produce games
  with an extreme komi.
- **Gatekeeper:** a draw counts **exactly half a point** for each side (it used to depend on
  `noResultUtilityForWhite`), draws are logged separately, and the gatekeeper refuses configs with komi or
  fence-handicap randomization (gating always uses the standard game).
- **Self-play log:** a new per-model `Quoridor stats` line with the draw rate, average plies, Black's win rate in
  standard games, by komi and in fence-handicap games.
- **Arena:** komi and initial walls per game (roster `"rules"` / `"pair_rules"`, `--komi`, `--black-walls`,
  `--white-walls`), sent to KataQuoridor engines and the arbiter; engines like SimpleQuoridor
  (`"supports_rules": false`) play standard games only. Arena SGFs now carry the arbiter's `KM`, `WB`, `WW` and
  `RE` (the lead, e.g. `W+0.5`; `B+F` / `W+F` for forfeits) instead of `KM[0]` and integer margins; results record
  `komi`, `black_walls`, `white_walls`, `lead` and `result`. See [Evaluation.md](../Evaluation.md).
- **sgfs_viewer:** the margin everywhere is the predicted `lead=` (the utility `score` is no longer shown when `lead=` is present; old files fall back to `score`, labelled "no lead in file"); phone-friendly responsive layout (swipe to step, collapsible game list, sticky controls); `serve.py --host 0.0.0.0` prints a LAN URL. Shows komi and initial walls when not standard, counts walls from the initial walls, and adds
  win-rate slices by komi and fence handicap to the stats page.
- New `runtests quoridorselfplay` test group; arena tests for komi, walls and arbiter draws.

## Draw by repetition (optional rule)

See [QuoridorIOv2.md](../QuoridorIOv2.md) §2.10 and §10. **Off by default**, so GTP, match, the arena against other
engines and old configs behave as before.

- **New rule `repetitionDrawCount`** (0 = off; e.g. 3): the game is a draw the moment a position (both pawns, all
  walls, walls left, side to move; not the ply) occurs for the N-th time. It ends shuffles in equilibrium positions
  (a sealed pawn that can only escape by a losing jump, and a blocker with nothing better) long before 300 plies.
  Search sees it at terminal nodes like the 300-ply draw, including from a root one move before the repetition;
  graph search keeps paths with different repetition counts apart.
- **Configs:** `repetitionDrawCount = 3` in `selfplay_quoridor_v2.cfg` and `gatekeeper_quoridor_v2.cfg`; off
  (commented) in `gtp_quoridor.cfg` — turn it on only for KataQuoridor-only matches.
- **QTP:** `kata-set-rule(s)` / `kata-get-rules` key `repetitionDrawCount` (before the first move only); `winner`
  says `Draw`; `showboard` prints the rule, the current position's count and `Game finished: Draw (repetition, …)`.
  `kata-get-rules` now always lists `repetitionDrawCount`.
- **SGF:** the rule is in `RU` (`Quoridor:repetitionDrawCount=3`); a drawn game gets a new root property `DR` with
  the reason: `repetition`, `maxPlies` or `cutoff`.
- **Training data:** a repetition draw is written exactly like a 300-ply draw (value 0.5 / 0.5, score 0, no lead,
  remaining plies to the actual end). No I/O version change.
- **Logs:** self-play `Quoridor stats` and `Game stats` and the gatekeeper's game stats count repetition draws
  separately from 300-ply draws. A draw is still half a point in gating.
- **Arena:** `--repetition-draw-count N` or roster `rules` / `pair_rules` key `repetitionDrawCount`, sent to
  KataQuoridor engines and the arbiter (SimpleQuoridor-type engines play standard games only); game ids get `_rN`;
  repetition draws get reason `repetition` and are counted in the report.
- **sgfs_viewer:** shows why a draw ended ("0 (draw by repetition)", a list tag) and the rule when on.
- New `runtests quoridorrepetition` test group.

## Neural net I/O v3 (repetition inputs)

See [QuoridorIOv3.md](../QuoridorIOv3.md).

- **I/O v3 = v2 + the repetition inputs:** two binary planes marking the pawn moves that repeat a position and those
  that draw by repetition, and two globals (rule on; how close the current position is to "one more occurrence
  draws"). All zero when the rule is off. The NN cache of v3 nets keys on this repetition state too.
- **Training is v3 only:** self-play writes v3 rows (with any net), new presets `b2c64_quoridor_v3` and
  `tf2_b4c192_quoridor_v3`. v1 and v2 nets keep working for inference everywhere.
- **Continue a v2 run as v3:** `python/quoridor_upgrade_v2_to_v3.py` adds the new input weights as zeros (model,
  SWA, optimizer state), so the upgraded net plays exactly like the v2 net (bit-identical on run3) and then learns
  the new inputs. `python/quoridor_convert_tdata_v2_to_v3.py -rule-was-off` converts the existing v2 rows of a run
  played without the repetition rule. Step-by-step switch-over in QuoridorIOv3.md §6.1.
- **The rule per self-play game:** new keys `quoridorRepetitionDrawProb`, `quoridorRepetitionDrawCounts`,
  `quoridorRepetitionDrawCountWeights` (off by default: then every game has `repetitionDrawCount`). The v2 self-play
  config plays 75% of games with the rule (N = 3: 90%, N = 4: 10%) and 25% without, so nets learn both the
  KataQuoridor game and the standard one (arena vs other engines, standard GTP). Forks keep their game's rule; the
  gatekeeper stays at N = 3. The `Quoridor stats` line reports draw rate and plies with the rule on and off.
- **From-scratch runs:** use a lower on-probability early (e.g. `quoridorRepetitionDrawProb = 0.25`) and raise it
  once the nets race purposefully. With a random net the rule turns 81–86% of games into draws (vs 15–17% without).
- Tools: `dumpnninputs -io-version 3`, `writesampletrainquoridor` (now with repetition-draw games); new
  `runtests quoridorv3`.

## Time bonus off (λ = 0), bigger net preset

See [QuoridorIOv3.md §9](../QuoridorIOv3.md#9-λ--0-2026-10-02).

- **`timeBonusPerPly = 0`** in the self-play and gatekeeper configs (was 0.05): `scoreMean` is now the lead. Nets
  trained before (run3 up to `run3-s21886976-d3911658`) used 0.05. Gatekeeper `maxVisits = 256`; self-play
  `maxRowsPerTrainFile = 10000`.
- **`python/quoridor_convert_tdata_lambda0.py`** converts λ > 0 training data: final utility score := lead on
  decisive rows, and the non-convertible short-term score target is switched off by the new weight column **C70**
  (read by the loss; written 0).
- **`tf3_b5c256_quoridor_v3`** preset (3.67M parameters, 5 nested-bottleneck blocks of 3 transformer layers, 256 trunk,
  fson normalization, SiLU), KataGo's mainline design scaled to Quoridor.
- `kq_ladder`: `extra_pairs` for pairs such as two nets at equal search time.
