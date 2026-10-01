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
