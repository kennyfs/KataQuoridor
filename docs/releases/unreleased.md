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
- **Drawn games are still discarded from training data** (as the 0.1.0 cutoff games were) and still count as
  "hit cutoff" in the self-play stats. Writing them with value 0.5/0.5 is planned for the I/O v2 net.
- **`komiMean` must be a Quoridor komi.** The Quoridor configs now say `komiMean = -0.5`; the old `komiMean = 0`
  fails at startup with an explanation. Update copies of `selfplay_quoridor.cfg` in running training directories
  before using a new binary.
- New optional config keys (also in `gtp_quoridor.cfg`, commented out): `maxPlies`, `timeBonusPerPly`,
  `blackInitialWalls`, `whiteInitialWalls`. For gtp, komi is set with the `komi` command (or forced with
  `ignoreGTPAndForceKomi`, which must be a Quoridor komi too).

## Arena

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
