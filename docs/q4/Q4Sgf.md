# Q4 game records: Duel-style SGF (`.sgfs`)

One game per line, UTF-8, `\n` terminated, `(;` … `)` without line breaks inside: the same text format as Duel's
`.sgfs` ([QuoridorIOv2.md](../QuoridorIOv2.md) §2, §5.6, §6.5), so a human can read and grep it. It replaced the JSON
record (`*.q4.jsonl`, `games.jsonl`); **old `.jsonl` files are not converted** and no tool reads them any more.

Writers and readers: `Q4Record::toSgfLine()` / `fromSgfLine()` (`cpp/q4/q4record.cpp`) and
`python/q4/record.py` (`Q4Record.to_sgf_line` / `from_sgf_line` / `load_records`), same semantics; the Python one is
checked against the C++ one on 80 C++-written games (`python/tests/test_q4_record.py`). Viewer:
`python3 python/q4_sgfs_viewer/serve.py FILE.sgfs`.

## 1. Example

A short fork game (a start position of 3 plies that no engine played), a draw by repetition (`rep 2`):

```
(;FF[4]GM[Q4]SZ[11]PS[b1c32_q4]PW[b1c32_q4]PN[b1c32_q4]PE[b1c32_q4]WS[7]WW[7]WN[7]WE[7]RU[Q4:repetitionDrawCount=2,maxPlies=100]RE[0]DR[repetition]C[gtype=fork,gameHash=c9df1ad01747946bb20cde520fb6d15f,startTurnIdx=3,type0=selfplay,net0=b1c32_q4,type1=selfplay,net1=b1c32_q4,type2=selfplay,net2=b1c32_q4,type3=selfplay,net3=b1c32_q4];S[d2h];W[a9h];N[a5v];E[k5]C[0.19 0.20 0.19 0.21 0.21 v=8 weight=0.00];S[f2]C[0.07 0.07 0.06 0.08 0.72 v=8 weight=1.10];W[a5]C[0.04 0.04 0.03 0.04 0.85 v=8 weight=0.00]; ... ;N[f11]C[0.02 0.03 0.02 0.02 0.91 v=8 weight=0.97])
```

## 2. Seats

Turn order S → W → N → E (clockwise), the compass names of [Q4Rules.md](Q4Rules.md) §2:

| letter | seat (0-based) | player | start |
|---|---|---|---|
| `S` | 0 | Player 1 | f1 |
| `W` | 1 | Player 2 | a6 |
| `N` | 2 | Player 3 | f11 |
| `E` | 3 | Player 4 | k6 |

## 3. Root node

| property | meaning |
|---|---|
| `FF[4]` `GM[Q4]` `SZ[11]` | SGF version, game (**Q4**; Duel is `GM[1]`), board size. The Q4 viewer refuses `GM[1]`, the Duel viewer refuses `GM[Q4]` with a message. |
| `PS PW PN PE` | player names of the seats S, W, N, E (`Q4PlayerInfo::name`). Self-play: the learner's model name, a population seat the kind (`weak`, ...) or its snapshot net name (`grudge` seats: `grudge>target`); q4match / gatekeeper: the player name of the config. |
| `WS WW WN WE` | initial walls of each seat (always written). |
| `RU[Q4:repetitionDrawCount=N,maxPlies=M]` | the rules; `N` = 0 is the repetition rule off. Walls are not repeated here. |
| `RE[x]` | `S`, `W`, `N` or `E` (the winner), `0` a draw, `?` unfinished. |
| `DR[r]` | `repetition` or `maxPlies` for a draw; `unfinished` when `RE[?]`. Not written for a decided game. |
| `C[...]` | comma-separated `key=value` pairs (Duel's `parseKV`), below. |

Root `C` keys: `gtype` (`normal`; `fork` for a game continued from a side position / fork; `mixed` for a population
game with a seat that is not a learner), `gameHash` (32 hex digits, hash0 then hash1: the id of the training rows of the
game, `globalTargetsNC` C44-49; only self-play), `startTurnIdx` (number of opening events that the engines did not play:
self-play `startHist` (fork / policy init), q4match / gatekeeper the random opening), and per seat `i` = 0..3 (S..E)
`type<i>` (`selfplay` = a learner seat, `weak`, `snapshot`, `greedy`, `randomPawn`, `basher`, `grudge`; q4match: `search`
or the bot name), `net<i>` (model name in self-play, the model path in q4match; when set), `v<i>` (visits, when > 0).
Match games add `table`, `opening`, `rotation` (`GameSpec`). Values must not contain `,`.

Text in `[...]` escapes `]` and `\` with a backslash; newlines in names become spaces.

## 4. Move nodes

One node per event, in order: `;S[f2]`, `;W[b6]`, `;N[e5h]`, `;E[j10v]`. The property is the seat that moves, the value
is `Q4Notation::actionToString` verbatim: a pawn destination (`f2`) or a wall by its anchor (`e5h`, `j10v`; anchor
column a..j, row 1..10). The C++ reader checks that the property is the seat to move (the engine skips a seat without a legal
action, no node is written for that).

An elimination is an external event, not a ply: `;EL[W]`. Plies (and the `maxPlies` count) count moves only.

### The comment of a move

`C[0.31 0.22 0.27 0.15 0.05 v=600 weight=1.00]` on the node of the move: the **search values at the position before that
move, from the absolute seats' point of view**: `<pS> <pW> <pN> <pE> <pDraw> v=<visits> weight=<weight>` (two decimals).

- the five numbers: win probabilities of S, W, N, E and the draw (they sum to 1);
- `v`: the unreduced visits of that search; `weight`: the training target weight of that turn (`targetWeightByTurn`,
  before the stochastic integerization, so `0.53` is possible). There is no `cheap=` field: a cheap search has a smaller
  `v` and weight 0.
- Self-play: every ply of a game after `startTurnIdx` has one: the learner's search values at that position (for a
  non-learner seat the observer search, whose weight is `q4PopulationObserverRowWeight`; its move is the seat's own).
  q4match / gatekeeper: the root values of the moving seat's search, `weight=1.00`.
- A move without a search (an opening ply, a bot's move) has no comment. An elimination node has none.

## 5. Where each command writes

| command | file |
|---|---|
| `katago q4selfplay` | `<output dir>/<model>/sgfs/<random hex>.sgfs` (flushed after every game), next to `tdata/` |
| `katago q4gatekeeper` | `<output dir>/<candidate>/games.sgfs` (+ `decision.json`) |
| `katago q4match -output FILE` | SGF lines in `FILE` (name it `*.sgfs`); `-summary` stays JSON |
| `q4_formal_eval.sh` | appends to `eval/ladder.sgfs` |
| `katago q4tool popgames` | SGF lines on stdout |
| `katago q4qtp` `printrecord` / `loadrecord FILE [n]` | one SGF line / the n-th line of a file |

`python/q4/rating.py`, `match_report.py` take `.sgfs` files (their `--json` summaries stay JSON); `style.py` takes the
dicts of `load_records`. `q4tool style` reads SGF lines (and, as a test input only, the JSON `{"rules","events"}`).

## 6. The Python dict

`load_records(path)` yields, per game, the dict the JSON record had (`rules`, `players`, `result` = `"1+".."4+"` /
`"Draw"` / `"none"`, `events` = `{"a": "f2"}` / `{"elim": seat1to4}`, `comments`, `gameHash`, `match`) plus `gtype`,
`startTurnIdx`, `drawReason` and `moveComments` (per event `None` or `{"p": [5], "visits", "weight"}`).

## 7. Viewer

`python3 python/q4_sgfs_viewer/serve.py FILE.sgfs` (`--host 0.0.0.0` for a phone on the LAN) or open
`python/q4_sgfs_viewer/index.html`'s page and drop a file: game list with filters and sorts, the board (rotate with `R`
so any seat is at the bottom, `P` toggles all shortest paths to the centre), the five win probabilities before each move,
a graph over the game, and a statistics tab. See `docs/q4/rounds/RSgf.md`.
