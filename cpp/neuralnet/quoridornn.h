/*
 * quoridornn.h
 *
 * The only module that knows both the 17x17 search space (Board, Loc, NNPos) and the 9x9
 * tensor space (the model's spatial grid). Everything downstream of this module (nneval.cpp,
 * the backends, desc.cpp) should only ever see one space or the other, never both.
 *
 * See docs/KataQuoridor_Review_and_Roadmap.md section 4 for the design this implements.
 */

#ifndef NEURALNET_QUORIDORNN_H_
#define NEURALNET_QUORIDORNN_H_

#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nninputs.h"

#include <utility>
#include <vector>

namespace QuoridorNN {
  // ---------------------------------------------------------------------------------------------
  // THE QUORIDOR MARGIN AND KataGo's "SCORE".
  //
  //   At game end, the winner's margin = the loser's shortest-path distance to the loser's goal row
  //   (walls only, pawns ignored; always >= 1 since the loser has not reached the goal).
  //   The margin is reported from White's perspective: positive if White's pawn arrived, negative if Black's.
  //   Implemented by Board::whiteMarginWhenWonBy.
  //
  // Since Quoridor I/O v2 (docs/QuoridorIOv2.md; definitions in rules.h), KataGo's score is not the margin but
  // derived from it: tempo t (the margin, +1 if Black arrived), lead s = t + komi (the winner is the sign of s),
  // and the utility score u = s + sign(s) * timeBonusPerPly * (maxPlies - T). BoardHistory stores
  // finalWhiteMinusBlackScore = u and finalWhiteLead = s (both 0 for a draw at maxPlies). With the default rules
  // (komi -0.5, no time bonus) u = s = margin -/+ 0.5. The margin is consumed in three places:
  //  (a) Terminal nodes: Search reads finalWhiteMinusBlackScore as scoreMean and finalWhiteLead as lead
  //      (white perspective).
  //  (b) Training targets (I/O v2, TrainingWriteBuffers::addRow), from the row's nextPlayer perspective: the
  //      game's final u (global target column 20, weight in column 27) and final s (column 21, weight in
  //      column 29; 0 for a draw, which carries no tempo information). The Python heads regress them in moves.
  //      (I/O v1 nets were trained on the final margin in column 21; v1 training data is no longer written.)
  //  (c) NN output: nneval multiplies the raw head outputs by the model's scoreMean/lead multipliers
  //      (both 1.0 for Quoridor, in moves) and flips the sign when Black is to move, yielding
  //      whiteScoreMean / whiteLead in the same white-perspective moves as (a). I/O v2 nets have separate
  //      heads for them (u and s); I/O v1 nets export their one margin head as both, so with a v1 net
  //      scoreMean and lead are the margin and ignore komi and the time bonus.
  // The Go score-belief machinery (scoreDistrN etc.) is inert for Quoridor.
  // ---------------------------------------------------------------------------------------------

  // Score utility is atan(margin / (scale * Board::sqrtBoardArea())), where sqrtBoardArea is the sqrt of the
  // real (pawn-cell) board area: 9 on the 9x9 board, the same as upstream KataGo uses for 9x9 Go, so score
  // utility behaves as in 9x9 Go. With the configs' static scale 2 and dynamicScoreCenterScale 0.5 (before the
  // *ScoreUtilityFactor), one move of margin near the center is worth ~0.035 (static, 2*9 = 18) and ~0.14
  // (dynamic, 0.5*9 = 4.5). Margins are roughly -20..+20 between weak players but likely closer to -5..+5
  // between strong ones; if score utility turns out too weak then, strengthen it via the config factors /
  // dynamicScoreCenterScale rather than changing the margin definition.

  // The model's spatial grid size in pawn cells (9x9), as opposed to Board's 17x17 search grid.
  // It must fit the largest board the engine is compiled for.
  constexpr int MODEL_LEN = Board::MAX_PAWN_LEN;
  // policy / optimistic-policy variants, each with {pawn, vertical wall, horizontal wall} planes.
  constexpr int NUM_POLICY_PLANES = 3;
  // I/O versions (model option D): 1 = KataQuoridor 0.1.0 nets, 2 = docs/QuoridorIOv2.md, 3 = v2 + the
  // repetition inputs (docs/QuoridorIOv3.md). All are supported for inference; training data is only written for
  // TRAINING_IO_VERSION.
  constexpr int MAX_SUPPORTED_IO_VERSION = 3;
  constexpr int TRAINING_IO_VERSION = 3;

  // Quoridor I/O v1 feature counts.
  constexpr int NUM_FEATURES_SPATIAL_V1 = 17;
  constexpr int NUM_FEATURES_GLOBAL_V1 = 15;
  // Quoridor I/O v2 = v1 + these channels, appended (see fillRow).
  constexpr int NUM_FEATURES_SPATIAL_V2 = 19;
  constexpr int NUM_FEATURES_GLOBAL_V2 = 17;
  constexpr int SPATIAL_LEGAL_VWALL_V2 = 17;
  constexpr int SPATIAL_LEGAL_HWALL_V2 = 18;
  constexpr int GLOBAL_PLIES_UNTIL_DRAW_V2 = 15;
  constexpr int GLOBAL_SELF_KOMI_V2 = 16;
  // Absolute scales (not relative to Rules::maxPlies / MAX_KOMI), so an input keeps its meaning across rules.
  constexpr double PLIES_UNTIL_DRAW_SCALE = 300.0;
  constexpr double SELF_KOMI_SCALE = 5.0;
  // Quoridor I/O v3 = v2 + these channels, appended (see fillRow).
  constexpr int NUM_FEATURES_SPATIAL_V3 = 21;
  constexpr int NUM_FEATURES_GLOBAL_V3 = 19;
  constexpr int SPATIAL_REPEATING_MOVE_V3 = 19;
  constexpr int SPATIAL_DRAWING_MOVE_V3 = 20;
  constexpr int GLOBAL_REPETITION_ON_V3 = 17;
  constexpr int GLOBAL_REPETITION_COUNT_V3 = 18;
  constexpr int MAX_NUM_FEATURES_SPATIAL = NUM_FEATURES_SPATIAL_V3;
  constexpr int MAX_NUM_FEATURES_GLOBAL = NUM_FEATURES_GLOBAL_V3;

  // Global 18 of v3: how close the current position (occurred `count` times) is to "one more occurrence draws":
  // (count - 1) / (N - 2), with N = rules.repetitionDrawCount, so 0 the first time and 1.0 at count N - 1 (for
  // N = 2 always 1.0). 0 when the rule is off.
  float repetitionProgress(int count, int repetitionDrawCount);
  // The legal pawn destinations of `pla` (the side to move in `hist`) whose resulting position already occurred since
  // the last wall placement, with how often (so moving there makes occurrence number count + 1). Empty when the rule
  // is off. Wall placements never repeat a position.
  void repeatingPawnMoves(const Board& board, const BoardHistory& hist, Player pla, std::vector<std::pair<Loc,int>>& out);
  // Everything the v3 repetition inputs read beyond Board / rules / ply (the current count and repeatingPawnMoves),
  // hashed; Hash128() when the rule is off. NNEvaluator folds it into the NN cache hash for v3 nets.
  Hash128 repetitionInputsHash(const Board& board, const BoardHistory& hist, Player pla);

  int numSpatialFeatures(int ioVersion);
  int numGlobalFeatures(int ioVersion);

  // BFS distance fields (in pawn steps, walls only, pawns ignored; -1 = unreachable), indexed
  // [k][c][r] in true board (not canonical) space. k: 0 = to nextPlayer's goal row, 1 = to the
  // opponent's goal row, 2 = from nextPlayer's pawn, 3 = from the opponent's pawn. These feed
  // spatial channels 8..11 of fillRow as (d < 0 ? 1 : min(1, d/32)).
  void fillDistances(const Board& board, Player nextPlayer, int dists[4][9][9]);

  // Raw distances for channels 8..11 as written to training data (npz key spatialDistNCHW):
  // out[k*81 + rCanon*9 + c], canonical (nextPlayer-relative) orientation exactly like fillRow,
  // DIST_UNREACHABLE_U8 for unreachable cells.
  constexpr int FIRST_DIST_CHANNEL = 8;
  constexpr int NUM_DIST_CHANNELS = 4;
  constexpr uint8_t DIST_UNREACHABLE_U8 = 255;
  void fillCanonicalDistancesU8(const Board& board, Player nextPlayer, uint8_t* out);

  // Fills a single row of spatial and global model inputs for the given ioVersion (1 or 2), in the
  // model's native 9x9 space, from the canonical (nextPlayer-relative, unmirrored) perspective.
  // Does not apply any input symmetry; call applyInputSymmetry afterwards for that.
  // v2 adds, after the 17 v1 spatial / 15 v1 global features:
  //   spatial 17 / 18: geometrically legal vertical / horizontal wall placements (Board::
  //     isGeometricallyLegalWallPlacement: legal for a player with a fence left, whatever the fence counts), on
  //     the 8x8 anchor grid in the 9x9 plane like the placed-wall channels 14 / 15 (row and column 8 zero);
  //   global 15: BoardHistory::pliesUntilDraw() / PLIES_UNTIL_DRAW_SCALE;
  //   global 16: BoardHistory::currentSelfKomi(nextPlayer, ...) / SELF_KOMI_SCALE (the standard komi -0.5
  //     reads +0.1 for Black and -0.1 for White).
  // v3 adds, after the 19 v2 spatial / 17 v2 global features:
  //   spatial 19: 1 on each legal pawn destination whose move repeats a position (occurrence >= 2);
  //   spatial 20: 1 on each legal pawn destination whose move draws by repetition (occurrence >= N);
  //     both pawn-cell planes like channel 1, 0 elsewhere and when the rule is off. Binary, because training data
  //     stores spatial planes as bits (a scaled plane would be truncated);
  //   global 17: 1 if the repetition rule is on (rules.repetitionDrawCount > 0);
  //   global 18: repetitionProgress(current position's occurrence count): 1.0 = one more occurrence of this
  //     position draws.
  void fillRow(
    const Board& board,
    const BoardHistory& boardHistory,
    Player nextPlayer,
    const MiscNNInputParams& nnInputParams,
    int ioVersion,
    bool useNHWC,
    float* rowSpatial,
    float* rowGlobal
  );

  // Physically mirrors a single already-filled spatial row along x (symmetry == 1), or leaves it
  // unchanged (symmetry == 0), in place. Quoridor only has two symmetries (no transpose, no
  // vertical flip: the row-flip for "which player is to move" is already baked into fillRow).
  // Pawn-cell channels map column c -> 8-c; wall-anchor channels (placed walls, the anchor domain mask
  // and, in v2, the legal-wall planes, whose domain is only the 8x8 sub-grid of anchors) map c -> 7-c; and the two board-edge-blocked channels for East and West
  // swap identities as well as mirroring, since mirroring the board turns "east" into "west".
  // The backend should always be handed symmetry 0 after this is applied.
  void applyInputSymmetry(float* rowSpatial, int ioVersion, bool useNHWC, int symmetry);

  // Maps the model's raw NUM_POLICY_PLANES*MODEL_LEN*MODEL_LEN policy output (channel-major,
  // i.e. NCHW with C = NUM_POLICY_PLANES) to the search space's 290-slot (17x17+1) policy array.
  // This undoes both the mirror applied by applyInputSymmetry and the canonical-perspective row
  // flip baked into fillRow, so the result is indexed in true board (not canonical) space.
  void mapPolicyToSearch(
    const float* rawPolicy,
    Player nextPlayer,
    float* policyProbs290,
    int symmetry
  );
}

#endif  // NEURALNET_QUORIDORNN_H_
