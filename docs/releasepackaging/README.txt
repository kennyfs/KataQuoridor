KataQuoridor v0.1.0 (based on KataGo v1.18.2)

KataQuoridor is a Quoridor (Duel, 9x9) engine built on KataGo (https://github.com/lightvector/KataGo).
It plays through QTP, a Quoridor dialect of GTP. It has no graphical interface of its own.
(The source tree has a browser GUI for playing against it, python/play_gui; see docs/PlayGUI.md.
It is not part of these zips.)

The release notes (recommended net, strength, known limitations) are in docs/releases/v0.1.0.md of the
source tree. Release tags use a "kq-" prefix (e.g. kq-v0.1.0), because the repository also carries upstream
KataGo's v1.x tags.

-----------------------------------------------------
REQUIREMENTS
-----------------------------------------------------
Supported backends: CUDA (NVIDIA GPU) and Eigen (CPU). OpenCL, TensorRT and Metal builds refuse to load a
KataQuoridor net. Pre-built Linux executables may not work with your library versions; building from source
is usually straightforward, see Compiling.md in the source tree.

You also need a KataQuoridor net (a .bin.gz file). KataGo's Go nets do not work, and are rejected.

-----------------------------------------------------
USAGE
-----------------------------------------------------
Check that the executable works:

./katago version
./katago runtests

Run the engine (gtp_quoridor.cfg is in cpp/configs/ of the source tree):

./katago gtp -model <NEURALNET>.bin.gz -config gtp_quoridor.cfg

Settings can be overridden on the command line, e.g.:

./katago gtp -model <NEURALNET>.bin.gz -config gtp_quoridor.cfg -override-config maxVisits=256,numSearchThreads=2

Notation: Black moves first and starts at e9 (top row), heading for row 1; White starts at e1. A pawn move is
its destination cell ("e8"). A wall is its anchor cell, column a-h and row 1-8, plus "h" or "v" ("e2h").

Main commands:

play <color> <move>     play a pawn move or wall, e.g. "play b e8", "play w e2h"
genmove <color>         let the engine choose and play a move
legal_moves             list the legal moves of the side to move
winner                  B, W or none
walls / dist            walls left / shortest-path distance to the goal, for both players
showboard               print the board
undo / clear_board      take back a move / start a new game
printsgf / loadsgf      save / load a game

Players strictly alternate (there is no pass), and once a pawn reaches its goal row the game is over: moves and
genmove then fail with "? game is over". See README.md in the source tree for details.

-----------------------------------------------------
TUNING FOR PERFORMANCE
-----------------------------------------------------
The main settings are maxVisits (strength vs. time per move), numSearchThreads and nnMaxBatchSize in
gtp_quoridor.cfg. KataGo's "benchmark" and "genconfig" subcommands are Go-specific and not supported by
KataQuoridor yet.

-----------------------------------------------------
LICENSE AND CREDITS
-----------------------------------------------------
KataQuoridor is distributed under the same license as KataGo (see LICENSE). Almost all of the search,
neural-net and training code is KataGo's, by David J Wu ("lightvector") and the KataGo contributors.
