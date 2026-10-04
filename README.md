# KataQuoridor

KataQuoridor is a [Quoridor](GameRules.md) engine and self-play training pipeline, built as a fork of
[KataGo](https://github.com/lightvector/KataGo) v1.18.2.

- **Game:** Quoridor Duel only: 9 × 9 board, two players, 10 walls each. Race and Four-at-a-Table are not
  implemented. KataQuoridor adds a 300-ply draw, komi and a fence handicap. See [GameRules.md](GameRules.md).
- **Engine:** KataGo's Monte-Carlo graph search, neural-net evaluator and CUDA/Eigen backends, driven through a
  Quoridor dialect of GTP (called QTP here).
- **Training:** KataGo's self-play → shuffle → train → export → gatekeeper loop, with Quoridor inputs, heads and
  training targets.
- **Current release:** 0.1.0. See [docs/releases/v0.1.0.md](docs/releases/v0.1.0.md) for the recommended net,
  its Elo ladder and the known limitations.

### Credits

Nearly all of the search, neural-net, backend and training infrastructure is KataGo's, by David J Wu
("lightvector") and the [KataGo contributors](CONTRIBUTORS). KataQuoridor replaces the Go-specific parts (rules,
board, input features, heads, training data) with Quoridor ones. Please credit KataGo, and see
[LICENSE](LICENSE), if you build on this. Upstream's README is kept at
[docs/KataGo_README.md](docs/KataGo_README.md); most of it is about Go.

KataQuoridor is not affiliated with, or endorsed by, the KataGo project.

## Supported backends

| Backend | Status |
|---|---|
| CUDA | Supported (used for training and evaluation). |
| Eigen (CPU) | Supported. Build with `-DUSE_AVX2=1` if your CPU has AVX2. |
| OpenCL, TensorRT, Metal | Refuse to load a net ("... is not supported by KataQuoridor; use CUDA or Eigen"). |
| ROCm, ONNX Runtime | Not tested. |

Nets are KataGo `.bin.gz` model files (architecture version 17) that declare a Quoridor I/O version in the
header; KataQuoridor rejects a Go net with a clear error.

## Building

Follow [Compiling.md](Compiling.md) (upstream's instructions; only the CUDA and Eigen backends matter here).
In short, on Linux:

```bash
cd cpp
cmake -S . -B build-cuda -DUSE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release
make -C build-cuda -j katago
```

```bash
cd cpp
cmake -S . -B build-eigen -DUSE_BACKEND=EIGEN -DUSE_AVX2=1 -DCMAKE_BUILD_TYPE=Release
make -C build-eigen -j katago
```

Check the build with `./katago version` (it prints `KataQuoridor 0.1.0 (based on KataGo 1.18.2, git <sha>)`) and
`./katago runtests`.

## QTP quickstart

Start the engine with a net and the Quoridor config:

```bash
cpp/build-cuda/katago gtp -model model.bin.gz -config cpp/configs/gtp_quoridor.cfg
```

`gtp_quoridor.cfg` searches 800 visits per move with 4 threads and writes logs to `./gtp_logs`; override any
setting with `-override-config maxVisits=256,numSearchThreads=1`. Commands follow GTP: one command per line,
answered by `= <result>` or `? <error>` and a blank line.

### Notation

- **Colours:** Black (`b`) is the first player and starts at `e9`, the middle of the top row, heading for row 1.
  White (`w`) starts at `e1` and heads for row 9.
- **Cells:** column `a`–`i` (left to right), row `1`–`9` (bottom to top). A pawn move is written as its
  destination cell, e.g. `e8`; jumps are written the same way.
- **Walls:** the anchor cell, column `a`–`h` and row `1`–`8`, plus `h` (horizontal) or `v` (vertical). `e2h`
  lies between rows 2 and 3 under columns `e` and `f`; `e2v` lies between columns `e` and `f` beside rows 2 and 3.
  See [GameRules.md](GameRules.md) for the exact geometry and the conflict rules.

### Key commands

| Command | Result |
|---|---|
| `play <color> <move>` | Play a pawn move or a wall, e.g. `play b e8`, `play w e2h`. |
| `move [color] <cell>`, `wall [color] <wall>` | The same, restricted to pawn moves or walls; the colour defaults to the side to move. |
| `genmove <color>` | Search, play and return the engine's move. |
| `legal_moves [color]` | All legal moves, space-separated (for `color` as if it were to move). |
| `winner` | `B`, `W`, `Draw` (after the 300-ply draw) or `none`. |
| `komi <k>`, `get_komi` | Set or show the komi (a half-integer, standard −0.5; see [Scores and komi](#scores-and-komi)). |
| `kata-get-rules`, `kata-set-rule <key> <value>`, `kata-set-rules <rules>` | Show or set `maxPlies`, `timeBonusPerPly`, `blackInitialWalls`, `whiteInitialWalls` (JSON or `Quoridor:key=value,...`). `maxPlies` and the walls can only change before the first move. |
| `walls [color]` | Walls left: `B: 10 W: 9`, or one number. |
| `dist [color]` | Shortest-path distance to the goal row: `B: 7 W: 9`, or one number. |
| `showboard` | ASCII board with pawns, walls, walls left, distances, the side to move and the ply count (plies until the draw); komi when not standard. |
| `undo`, `clear_board` | Take back one move; start a new game. |
| `printsgf`, `loadsgf <file> [movenum]` | Save or load a game as SGF. |
| `kata-analyze`, `kata-genmove_analyze`, `kata-raw-nn` | KataGo's analysis commands ([docs/GTP_Extensions.md](docs/GTP_Extensions.md)); moves use the notation above. |

Rules the engine enforces:

- The players strictly alternate and there is no pass. `play`/`move`/`wall` for the side not to move fail with
  `? illegal move: not <color>'s turn`; `genmove` and the analysis commands fail with `? not <color>'s turn`.
- The game ends as soon as a pawn reaches its goal row. From then on, every move and `genmove` variant fails
  with `? game is over`, `legal_moves` is empty and `winner` keeps the result. `undo`, `clear_board` and
  `loadsgf` still work.
- A game that reaches 300 plies (`maxPlies`) without a pawn on its goal is a draw: `winner` says `Draw`, and
  nothing more can be played. Search sees the draw too. Self-play discards drawn games for now.
- The winner is decided by the tempo and komi (see below); with the standard komi −0.5 it is simply the player
  whose pawn arrives first.

Example session:

```
play b e8
=

play w e2h
=

genmove b
= e7

walls
= B: 10 W: 9

dist
= B: 7 W: 9

play w e8
? illegal move

winner
= none
```

### Scores and komi

KataQuoridor scores a finished game in tempo, from White's point of view ([GameRules.md](GameRules.md),
[docs/QuoridorIOv2.md](docs/QuoridorIOv2.md)):

- **Komi** is a half-integer; the **standard game is komi −0.5**. White wins iff `tempo + komi > 0`, where the
  tempo is the loser's remaining distance if White's pawn arrived, and 1 minus it if Black's did. An equal race
  (B+1) is tempo 0, so with komi −0.5 Black wins it.
- **`scoreLead`** (KataGo's lead) is the **tempo lead** `tempo + komi`: +0.5 means White wins by one tempo. SGF
  results are written the same way (`W+0.5`, `B+2.5`, `0` for a draw).
- **`scoreMean`** (KataGo's score; search and training use it) is the **utility score**: the lead plus a time
  bonus, `sign(lead) × timeBonusPerPly × (300 − plies)`, which rewards finishing a won game earlier. With
  `timeBonusPerPly` > 0 it reads around `timeBonusPerPly × 300` at the start (e.g. +15 with 0.05) while the lead is
  about 0; that is expected. `timeBonusPerPly` (default 0) must match what the net was trained with.
- **GUIs should show `scoreLead`.** As in upstream KataGo, `kata-analyze` and the analysis engine report the
  utility score as `scoreSelfplay`, and their `scoreMean` field is a copy of `scoreLead`; `kata-raw-nn` reports
  `whiteLead` and `whiteScoreSelfplay`.

## Play in the browser

`python3 python/play_gui/serve.py --katago <katago> --model <net.bin.gz>` starts a local web page for
playing against the engine with the mouse (hints, undo, evaluation, SGF download). See
[docs/PlayGUI.md](docs/PlayGUI.md).

## Training

The self-play training loop is KataGo's, with Quoridor configs:

- [SelfplayTraining.md](SelfplayTraining.md) describes the five parts (selfplay, shuffle, train, export,
  gatekeeper) and how they fit together.
- [python/selfplay/synchronous_loop.sh](python/selfplay/synchronous_loop.sh) runs them in one loop on one
  machine, with [cpp/configs/training/selfplay_quoridor.cfg](cpp/configs/training/selfplay_quoridor.cfg) and
  [cpp/configs/training/gatekeeper_quoridor.cfg](cpp/configs/training/gatekeeper_quoridor.cfg).
- [docs/DistPlanesUpgrade.md](docs/DistPlanesUpgrade.md) explains how to upgrade training data and checkpoints
  written before the S8–S11 distance-plane fix. Data or nets from before that fix must not be mixed with new ones.
- [docs/KataQuoridor_Review_and_Roadmap.md](docs/KataQuoridor_Review_and_Roadmap.md) documents the design
  (inputs, heads, versioning) and the roadmap.

## Evaluation

[docs/Evaluation.md](docs/Evaluation.md) describes `python/quoridor_arena`, a referee that plays QTP engines
against each other with a no-net KataQuoridor process as the arbiter, and fits a Bradley-Terry Elo ladder
anchored at a uniformly random player. Self-play and arena games can be browsed with
`python3 python/sgfs_viewer/serve.py <file>.sgfs` (the page is responsive: for a phone on the same network add
`--host 0.0.0.0` and open the printed LAN URL; swipe the board to step moves); [python/model_viewer](python/model_viewer/serve.py) shows
per-net analyses (heads, feature importance, raw net vs search) written by `analyze_model.py`.

## Releases and versions

- KataQuoridor has its own version (`0.1.0`), reported by `katago version` and QTP `version`. The upstream base
  version stays at 1.18.2 internally.
- This repository still carries upstream KataGo's tags (`v1.0` … `v1.18.x`), so KataQuoridor releases are
  tagged with a `kq-` prefix: `kq-v0.1.0`, `kq-v0.2.0`, ….
- Release notes live in [docs/releases/](docs/releases/).

## License

KataQuoridor is distributed under the same license as KataGo; see [LICENSE](LICENSE).
