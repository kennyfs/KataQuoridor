#ifndef Q4_NN_H_
#define Q4_NN_H_

#include "../q4board.h"
#include "../q4history.h"
#include "../q4rules.h"
#include "../q4symmetry.h"
#include "q4nnconstants.h"
#include "../../neuralnet/nneval.h"
#include "../../core/global.h"
#include "../../core/hash.h"

#include <cstdint>
#include <string>
#include <vector>

// The only C++ module that knows both game space and tensor space (docs/q4/Q4IO.md).
// Inputs are always seen from the seat to move (board.toMove); outputs are decoded back to absolute seats and
// game actions.
namespace Q4NN {
  using namespace Q4NNConst;

  static constexpr int NUM_RAW_DIST_CHANNELS = 5;  // spatial channels 10..14

  // Distances (walls only, 255 = unreachable) as stored in training rows: [0] = to the center, [1 + k] = from the
  // pawn of relative seat k (all 255 if that seat is eliminated).
  struct RawDistances {
    uint8_t d[NUM_RAW_DIST_CHANNELS][POS_AREA];
  };

  // The float feature of a raw distance: 1.0 if unreachable, else min(d, 64) / 64.
  inline float dist01(uint8_t d) { return d == 255 ? 1.0f : (float)(d < 64 ? d : 64) / 64.0f; }

  void computeRawDistances(const Q4Board& board, RawDistances& out);

  // Fill the spatial (27 x 11 x 11) and global (28) input rows of the position (symmetry 0, seat = board.toMove).
  // The spatial row is NCHW, or NHWC if inputsUseNHWC. If rawDist is not null it also receives the raw distances.
  void fillRow(
    const Q4Board& board,
    const Q4History& history,
    bool inputsUseNHWC,
    float* rowSpatial,
    float* rowGlobal,
    RawDistances* rawDist = nullptr
  );

  // Apply symmetry sym (0..7) to a spatial row (same layout in and out; src and dst must not overlap unless sym == 0).
  void applyInputSymmetry(const float* srcSpatial, float* dstSpatial, int sym, bool inputsUseNHWC);

  // Raw policy logits [NUM_POLICY_VARIANTS][NUM_POLICY_PLANES][11][11] of an evaluation under symmetry sym -> logits
  // of the game actions: out[variant * NUM_ACTIONS + action].
  void mapPolicyToGame(const float* rawPolicyLogits, int sym, float* outActionLogits);

  // Softmax over the legal actions only; the other entries of outProbs (NUM_ACTIONS floats) are 0.
  void softmaxLegal(
    const float* actionLogits,
    const std::vector<int>& legalActions,
    float* outProbs,
    float temperature = 1.0f
  );

  // Softmax of the 5 relative value logits [me, next, across, previous, draw].
  void softmaxValue(const float* valueLogits, float* outRelProbs);

  // Rotate relative probabilities [me, next, across, previous, draw] to absolute [seat0..seat3, draw].
  void rotateValueToAbsolute(const float* relativeProbs, int toMove, float* outAbsProbs);

  // Trajectory logits (11 x 11, evaluated under sym) -> probabilities per game cell.
  void decodeTrajectory(const float* rawTrajectory, int sym, float* outTrajectory);

  // 128-bit key of the (position, symmetry) NN evaluation (docs/q4/Q4IO.md §7).
  Hash128 getCacheHash(const Q4Board& board, const Q4History& history, int sym);

  // Everything one evaluation returns, in game space.
  struct Eval {
    int toMove;
    int sym;
    float policyLogits[NUM_POLICY_VARIANTS][NUM_ACTIONS];  // per game action (not masked)
    float valueLogits[NUM_VALUE_LOGITS];                   // relative seats, as the net returns them
    float valueRel[NUM_VALUE_LOGITS];                      // softmax, relative seats
    float valueAbs[NUM_VALUE_LOGITS];                      // softmax, absolute seats; [4] = draw
    float misc[NUM_MISC];
    float trajectory[POS_AREA];                            // probability that my pawn visits the game cell
  };

  // Fill the rows (with symmetry sym), run them through the evaluator's Q4 raw path and decode.
  void evaluate(
    NNEvaluator& nnEval,
    NNResultBuf& buf,
    const Q4History& history,
    int sym,
    bool skipCache,
    Eval& out
  );

  // Absolute seat names (seat 0 starts at f1 = South, then clockwise).
  const char* seatName(int seat);

  // Human-readable report of an evaluation: values per absolute seat, misc outputs, the top legal moves of the
  // search policy (q4qtp q4-rawnn, q4tool evalnn).
  std::string formatEval(const Eval& eval, const std::vector<int>& legalActions, int numTopMoves);
}

#endif  // Q4_NN_H_
