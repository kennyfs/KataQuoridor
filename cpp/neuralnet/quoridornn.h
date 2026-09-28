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

namespace QuoridorNN {
  // The model's spatial grid size (9x9), as opposed to Board's 17x17 search grid.
  constexpr int MODEL_LEN = 9;
  // policy / optimistic-policy variants, each with {pawn, vertical wall, horizontal wall} planes.
  constexpr int NUM_POLICY_PLANES = 3;
  constexpr int MAX_SUPPORTED_IO_VERSION = 1;

  // Quoridor I/O v1 feature counts.
  constexpr int NUM_FEATURES_SPATIAL_V1 = 17;
  constexpr int NUM_FEATURES_GLOBAL_V1 = 15;

  int numSpatialFeatures(int ioVersion);
  int numGlobalFeatures(int ioVersion);

  // Fills a single row of spatial and global model inputs for the given ioVersion, in the
  // model's native 9x9 space, from the canonical (nextPlayer-relative, unmirrored) perspective.
  // Does not apply any input symmetry; call applyInputSymmetry afterwards for that.
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
  // Pawn-cell channels map column c -> 8-c; wall-anchor channels (whose domain is only the 8x8
  // sub-grid of anchors) map c -> 7-c; and the two board-edge-blocked channels for East and West
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
