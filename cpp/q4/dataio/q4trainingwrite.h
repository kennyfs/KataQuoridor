#ifndef Q4_TRAINING_WRITE_H_
#define Q4_TRAINING_WRITE_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "../../core/global.h"
#include "../../core/hash.h"
#include "../../core/rand.h"
#include "../../dataio/numpywrite.h"
#include "../q4board.h"
#include "../q4history.h"
#include "../q4playstate.h"
#include "../q4rules.h"
#include "../nn/q4nn.h"

namespace Q4Play {

struct Q4ValueTargets {
  // Absolute seats [0..3], [4] = draw
  float value[5];

  Q4ValueTargets();
  ~Q4ValueTargets() = default;
};

struct Q4PolicyTargetMove {
  int action;
  int16_t policyTarget;

  Q4PolicyTargetMove() : action(Q4Board::NULL_ACTION), policyTarget(0) {}
  Q4PolicyTargetMove(int act, int16_t target) : action(act), policyTarget(target) {}
};

struct Q4PolicyTarget {
  std::vector<Q4PolicyTargetMove>* policyTargets;
  int64_t unreducedNumVisits;

  Q4PolicyTarget() : policyTargets(nullptr), unreducedNumVisits(0) {}
  Q4PolicyTarget(std::vector<Q4PolicyTargetMove>* targets, int64_t visits)
    : policyTargets(targets), unreducedNumVisits(visits) {}
};

struct Q4NNRawStats {
  double rawNNUtility;
  double policyEntropy;

  Q4NNRawStats() : rawNNUtility(0.0), policyEntropy(0.0) {}
};

struct Q4SidePosition {
  Q4PlayState state;
  int toMove;
  int64_t unreducedNumVisits;
  std::vector<Q4PolicyTargetMove> policyTarget;
  double policySurprise;
  double policyEntropy;
  double searchEntropy;
  Q4ValueTargets valueTargets;
  Q4NNRawStats nnRawStats;
  float targetWeight;
  float targetWeightUnrounded;
  int numNeuralNetChangesSoFar;

  Q4SidePosition();
  Q4SidePosition(const Q4PlayState& s, int numNNChangesSoFar);
  ~Q4SidePosition() = default;
};

struct ChangedNeuralNet {
  std::string name;
  int turnIdx;

  ChangedNeuralNet(const std::string& n, int t) : name(n), turnIdx(t) {}
};

// Who sat in a seat of a game (population self-play, docs/q4/Q4IO.md §8); kind is a Q4SeatKind.
struct Q4SeatInfo {
  int kind = 0;
  std::string net;
  int visits = 0;
  double temperature = 0.0;
  int grudgeTarget = -1;
};

struct Q4FinishedGameData {
  std::string modelName;
  Q4PlayState startState;
  Q4History startHist;
  Q4History endHist;
  Hash128 gameHash;

  bool hitTurnLimit;
  int mode;
  bool hasFullData;

  std::vector<float> targetWeightByTurn;
  std::vector<float> targetWeightByTurnUnrounded;
  std::vector<Q4PolicyTarget> policyTargetsByTurn;
  std::vector<double> policySurpriseByTurn;
  std::vector<double> policyEntropyByTurn;
  std::vector<double> searchEntropyByTurn;
  std::vector<Q4ValueTargets> valueTargetsByTurn; // size numMoves + 1 (final entry is result)
  std::vector<Q4NNRawStats> nnRawStatsByTurn;
  std::vector<double> valueSurpriseByTurn;
  std::vector<bool> wasCheapSearchByTurn;
  std::vector<int> seatKindByTurn; // Q4SeatKind of the seat that moved; != 0 rows are observer rows (no search policy)
  Q4SeatInfo seatInfo[4];

  std::vector<Q4SidePosition*> sidePositions;
  std::vector<ChangedNeuralNet*> changedNeuralNets;
  std::vector<std::string> comments;

  double trainingWeight;
  int startPly;
  Q4Rules rules;

  static constexpr int MODE_NORMAL = 0;
  static constexpr int MODE_FORK = 2;

  Q4FinishedGameData();
  ~Q4FinishedGameData();

  Q4FinishedGameData(const Q4FinishedGameData&) = delete;
  Q4FinishedGameData& operator=(const Q4FinishedGameData&) = delete;
};

class Q4TrainingWriteBuffers {
 public:
  static constexpr int NUM_BINARY_CHANNELS = 27;
  static constexpr int PACKED_BOARD_AREA = 16;
  static constexpr int NUM_DIST_CHANNELS = 5;
  static constexpr int NUM_GLOBAL_CHANNELS = 28;
  static constexpr int POLICY_NUM_CHANNELS = 3;
  static constexpr int POLICY_SIZE = 363;
  static constexpr int GLOBAL_TARGET_NUM_CHANNELS = 64;
  static constexpr int SCORE_DISTR_LEN = 1;
  static constexpr int VALUE_TARGET_CHANNELS = 12;
  static constexpr int POS_LEN = 11;
  static constexpr int POS_AREA = 121;

  int maxRows;
  int curRows;

  NumpyBuffer<uint8_t> binaryInputNCHWPacked;
  NumpyBuffer<uint8_t> spatialDistNCHW;
  NumpyBuffer<float> globalInputNC;
  NumpyBuffer<int16_t> policyTargetsNCMove;
  NumpyBuffer<float> globalTargetsNC;
  NumpyBuffer<int8_t> scoreDistrN;
  NumpyBuffer<int8_t> valueTargetsNCHW;

  Q4TrainingWriteBuffers(int maxRws);
  ~Q4TrainingWriteBuffers() = default;

  Q4TrainingWriteBuffers(const Q4TrainingWriteBuffers&) = delete;
  Q4TrainingWriteBuffers& operator=(const Q4TrainingWriteBuffers&) = delete;

  void clear();
  int numRows() const { return curRows; }

  void writeToZipFile(const std::string& fileName);

  void addRow(
    const Q4PlayState& state,
    int turnIdx,
    float targetWeight,
    int64_t unreducedNumVisits,
    const std::vector<Q4PolicyTargetMove>* policyTarget0,
    const std::vector<Q4PolicyTargetMove>* policyTargetNext,
    int actionPlayedThisTurn,
    double policySurprise,
    double policyEntropy,
    double searchEntropy,
    const std::vector<Q4ValueTargets>& valueTargetsByTurn,
    int valueTargetsIdx,
    float valueTargetWeight,
    float tdValueTargetWeight,
    const Q4NNRawStats& nnRawStats,
    bool isSidePosition,
    Hash128 gameHash,
    int gameMode,
    int startPly,
    bool hitTurnLimit,
    int numAliveAtRow,
    int maxPlies,
    int repetitionDrawCount,
    int finalGamePlies,
    const uint8_t finalDistToCenter[4],
    const bool seatEliminatedBeforeEnd[4],
    const std::vector<Q4Board>& boardHistoryFromTurnToEnd,
    const std::vector<int>& actionsPlayedFromTurnToEnd,
    const std::vector<int>& actionSeatsFromTurnToEnd,
    int seatKind
  );

  static int actionToPolicySlot(int action);
  static void fillValueTDTargets(
    const std::vector<Q4ValueTargets>& valueTargetsByTurn,
    int idx,
    int toMove,
    double nowFactor,
    float* buf
  );
};

class Q4TrainingDataWriter {
 public:
  Q4TrainingDataWriter(
    const std::string& outputDir,
    int maxRowsPerTrainFile,
    double firstFileRandMinProp,
    uint64_t randSeed
  );
  ~Q4TrainingDataWriter();

  Q4TrainingDataWriter(const Q4TrainingDataWriter&) = delete;
  Q4TrainingDataWriter& operator=(const Q4TrainingDataWriter&) = delete;

  void writeGame(const Q4FinishedGameData& data);
  void flushIfNonempty();

  int64_t numRowsWritten() const { return totalRowsWritten.load(std::memory_order_relaxed); }
  int64_t numGamesWritten() const { return totalGamesWritten.load(std::memory_order_relaxed); }

 private:
  std::string outputDir;
  int maxRowsPerTrainFile;
  double firstFileRandMinProp;
  Rand rand;
  std::mutex writeMutex;

  std::unique_ptr<Q4TrainingWriteBuffers> writeBuffers;
  int currentFileLimit;
  std::atomic<int64_t> totalRowsWritten;
  std::atomic<int64_t> totalGamesWritten;

  void flushLocked();
};

}  // namespace Q4Play

#endif  // Q4_TRAINING_WRITE_H_
