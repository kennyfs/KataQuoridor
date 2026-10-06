#ifndef Q4SEARCH_SEARCH_H_
#define Q4SEARCH_SEARCH_H_

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "../../core/global.h"
#include "../../core/hash.h"
#include "../../core/logger.h"
#include "../../core/multithread.h"
#include "../../core/rand.h"
#include "../../core/threadsafequeue.h"
#include "../../core/threadsafecounter.h"
#include "../../neuralnet/nneval.h"
#include "../../search/distributiontable.h"
#include "../../search/mutexpool.h"
#include "../../search/searchparams.h"
#include "../../search/timecontrols.h"
#include "../q4board.h"
#include "../q4history.h"
#include "../q4playstate.h"
#include "../q4rules.h"
#include "q4analysisdata.h"
#include "q4reportedsearchvalues.h"
#include "q4searchnode.h"
#include "q4searchnodetable.h"

namespace Q4S {

struct Search;

// Per-thread state
struct SearchThread {
  int threadIdx;

  int seat; // seat to move in thread's local playout
  Q4PlayState state; // thread's local state during playout

  bool shouldCountPlayout;
  Rand rand;

  NNResultBuf nnResultBuf;
  std::vector<MoreNodeStats> statsBuf;
  double upperBoundVisitsLeft;

  std::vector<std::shared_ptr<NNOutput>*> oldNNOutputsToCleanUp;

  SearchThread(int threadIdx, const Search& search);
  ~SearchThread();

  SearchThread(const SearchThread&) = delete;
  SearchThread& operator=(const SearchThread&) = delete;
};

struct Search {
  // Constant/immutable during search
  int rootSeat;
  uint8_t rootAliveMask;
  int rootNumAlive;
  Q4PlayState rootState;
  int rootHintAction;

  SearchParams searchParams;
  int64_t numSearchesBegun;
  uint32_t searchNodeAge;
  int64_t lastSearchNumPlayouts;
  double effectiveSearchTimeCarriedOver;

  std::string randSeed;

  // Precomputed distribution for downweighting child values
  DistributionTable* valueWeightDistribution;
  double normToTApproxZ;
  std::vector<double> normToTApproxTable;

  Rand nonSearchRand;

  // Externally owned
  Logger* logger;
  NNEvaluator* nnEvaluator;

  // Mutated during search
  SearchNode* rootNode;
  SearchNodeTable* nodeTable;
  MutexPool* mutexPool;

  // Thread pool
  int numThreadsSpawned;
  std::thread* threads;
  ThreadSafeQueue<std::function<void(int)>*>* threadTasks;
  ThreadSafeCounter* threadTasksRemaining;

  std::mutex oldNNOutputsToCleanUpMutex;
  std::vector<std::shared_ptr<NNOutput>*> oldNNOutputsToCleanUp;

  Search(
    const SearchParams& params,
    NNEvaluator* nnEval,
    Logger* logger,
    const std::string& randSeed
  );
  ~Search();

  Search(const Search&) = delete;
  Search& operator=(const Search&) = delete;

  // Parameter guard (Plan §8.5, prompt B7)
  static void checkParams(const SearchParams& params);

  // Top-level outside-of-search control methods
  const Q4PlayState& getRootState() const { return rootState; }
  int getRootSeat() const { return rootState.board.toMove; }

  void setPosition(const Q4PlayState& state);
  void setPosition(const Q4History& history);
  void setRootHintAction(int action);
  void setParams(const SearchParams& params);
  void setParamsNoClearing(const SearchParams& params);
  void setNNEval(NNEvaluator* nnEval);

  void clearSearch();
  void respawnThreads();

  bool makeMove(int action);

  int runWholeSearchAndGetMove();
  int runWholeSearchAndGetMove(std::atomic<bool>* shouldStop);
  void runWholeSearch();
  void runWholeSearch(std::atomic<bool>* shouldStop);
  void runWholeSearch(std::function<bool()>* shouldStopEarly);
  void runWholeSearch(
    const std::function<void()>* searchBegun,
    std::function<bool()>* shouldStopEarly,
    const TimeControls& tc,
    double searchFactor
  );

  void beginSearch();
  bool runSinglePlayout(SearchThread& thread, double upperBoundVisitsLeft);

  // Search results and tree inspection
  int64_t getRootVisits() const;
  int getChosenMoveAction();
  bool getRootValues(ReportedSearchValues& values) const;
  void getAnalysisData(std::vector<AnalysisData>& buf) const;
  bool getPlaySelectionValues(
    std::vector<int>& actions,
    std::vector<double>& playSelectionValues,
    std::vector<double>* retVisitCounts,
    double scaleMaxToAtLeast = 0.0
  ) const;
  void getPolicySurpriseAndEntropy(double& policySurprise, double& policyEntropy) const;

  // Utility and weighting helpers (Plan §8.3)
  static double computeSeatUtility(
    int seat,
    const double value[5],
    int nAlive,
    bool isAlive,
    double winLossUtilityFactor
  );

  // Selection constants
  static constexpr double POLICY_ILLEGAL_SELECTION_VALUE = -1e50;
  static constexpr double FUTILE_VISITS_PRUNE_VALUE = -1e40;

  // Exploration helpers (PUCT, FPU, selection)
  double getExploreScaling(double totalChildWeight, double parentUtilityStdevFactor) const;
  double getExploreSelectionValue(
    double exploreScaling,
    double nnPolicyProb,
    double childWeight,
    double childUtility
  ) const;
  double getExploreSelectionValueInverse(
    double exploreScaling,
    double exploreSelectionValue,
    double nnPolicyProb,
    double childUtility
  ) const;
  double getExploreSelectionValueOfChild(
    const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
    int action,
    double exploreScaling,
    double totalChildWeight, int64_t childEdgeVisits, double fpuValue,
    double parentUtility, double parentWeightPerVisit,
    bool isDuringSearch, double maxChildWeight,
    bool countEdgeVisit,
    SearchThread* thread
  ) const;
  double getNewExploreSelectionValue(
    const SearchNode& parent,
    double exploreScaling,
    float nnPolicyProb,
    double fpuValue,
    double parentWeightPerVisit,
    double maxChildWeight,
    bool countEdgeVisit,
    SearchThread* thread
  ) const;
  double getReducedPlaySelectionWeight(
    const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
    int action,
    double exploreScaling,
    int64_t childEdgeVisits,
    double bestChildExploreSelectionValue
  ) const;
  double getFpuValueForChildrenAssumeVisited(
    const SearchNode& node, int mover, bool isRoot, double policyProbMassVisited,
    double& parentUtility, double& parentWeightPerVisit, double& parentUtilityStdevFactor
  ) const;
  void selectBestChildToDescend(
    SearchThread& thread, const SearchNode& node, SearchNodeState nodeState,
    int& numChildrenFound, int& bestChildIdx, int& bestChildMoveAction, bool& countEdgeVisit,
    bool isRoot
  ) const;

  // Update helpers
  void addLeafValue(
    SearchNode& node,
    const double value[5],
    double weight,
    bool isTerminal,
    bool assumeNoExistingWeight
  );
  void addCurrentNNOutputAsLeafValue(SearchNode& node, bool assumeNoExistingWeight);
  double computeWeightFromNNOutput(const NNOutput* nnOutput) const;
  void updateStatsAfterPlayout(SearchNode& node, SearchThread& thread, bool isRoot);
  void recomputeNodeStats(SearchNode& node, SearchThread& thread, int32_t numVisitsToAdd, bool isRoot);
  void downweightBadChildrenAndNormalizeWeight(
    int numChildren,
    double currentTotalWeight,
    double desiredTotalWeight,
    double amountToSubtract,
    double amountToPrune,
    std::vector<MoreNodeStats>& statsBuf
  ) const;
  double pruneNoiseWeight(std::vector<MoreNodeStats>& statsBuf, int numChildren, double totalChildWeight, const double* policyProbsBuf) const;

  // NN evaluation helpers
  void computeRootNNEvaluation(NNResultBuf& nnResultBuf);
  bool initNodeNNOutput(SearchThread& thread, SearchNode& node, bool isRoot, bool skipCache, bool isReInit);
  bool maybeRecomputeExistingNNOutput(SearchThread& thread, SearchNode& node, bool isRoot);

  // Multithread helpers
  int numAdditionalThreadsToUseForTasks() const;
  void spawnThreadsIfNeeded();
  void killThreads();
  void performTaskWithThreads(std::function<void(int)>* task, int capThreads);
  void applyRecursivelyPostOrderMulithreaded(const std::vector<SearchNode*>& nodes, std::function<void(SearchNode*,int)>* f);
  void applyRecursivelyPostOrderMulithreadedHelper(SearchNode* node, int threadIdx, PCG32* rand, std::unordered_set<SearchNode*>& nodeBuf, std::vector<int>& randBuf, std::function<void(SearchNode*,int)>* f);
  void applyRecursivelyAnyOrderMulithreaded(const std::vector<SearchNode*>& nodes, std::function<void(SearchNode*,int)>* f);
  void applyRecursivelyAnyOrderMulithreadedHelper(SearchNode* node, int threadIdx, PCG32* rand, std::unordered_set<SearchNode*>& nodeBuf, std::vector<int>& randBuf, std::function<void(SearchNode*,int)>* f);
  std::vector<SearchNode*> enumerateTreePostOrder();

  // Time helpers
  double numVisitsNeededToBeNonFutile(double maxVisitsMoveVisits);
  double computeUpperBoundVisitsLeftDueToTime(int64_t rootVisits, double timeUsed, double plannedTimeLimit);
  double recomputeSearchTimeLimit(const TimeControls& tc, double timeUsed, double searchFactor, int64_t rootVisits);

  // Search & allocation helpers
  uint32_t createMutexIdxForNode(SearchThread& thread) const;
  SearchNode* allocateOrFindNode(SearchThread& thread, int nextSeat, int action, bool forceNonTerminal);
  void clearOldNNOutputs();
  void transferOldNNOutputs(SearchThread& thread);
  void deleteAllOldOrAllNewTableNodesMulithreaded(bool old);
  void deleteAllTableNodesMulithreaded();
  void computeRootValues();
  bool playoutDescend(SearchThread& thread, SearchNode& node, bool isRoot);
  double interpolateEarly(double halflife, double earlyValue, double value) const;
  void getSelfUtilityLCBAndRadiusZeroVisits(double& lcbBuf, double& radiusBuf) const;
  void getSelfUtilityLCBAndRadius(const SearchNode& parent, const SearchNode* child, int64_t edgeVisits, int action, double& lcbBuf, double& radiusBuf) const;
  static uint32_t chooseIndexWithTemperature(
    Rand& rand,
    const double* relativeProbs,
    int numRelativeProbs,
    double temperature,
    double onlyBelowProb,
    double* processedRelProbsBuf
  );
  static void computeDirichletAlphaDistribution(int policySize, const float* policyProbs, double* alphaDistr);
  static void addDirichletNoise(const SearchParams& searchParams, Rand& rand, int policySize, float* policyProbs);
  std::shared_ptr<NNOutput>* maybeAddPolicyNoiseAndTemp(SearchThread& thread, bool isRoot, const NNOutput* oldNNOutput) const;

  // Analysis helpers
  AnalysisData getAnalysisDataOfSingleChild(
    const SearchNode* child,
    int64_t edgeVisits,
    std::vector<int>& scratchActions,
    std::vector<double>& scratchValues,
    int action,
    double policyProb,
    double fpuValue,
    double parentUtility,
    int maxPVDepth
  ) const;
  bool getPlaySelectionValues(
    const SearchNode& node,
    std::vector<int>& actions,
    std::vector<double>& playSelectionValues,
    std::vector<double>* retVisitCounts,
    double scaleMaxToAtLeast,
    bool allowDirectPolicyMoves,
    bool alwaysComputeLcb,
    bool neverUseLcb,
    double lcbBuf[Q4Board::NUM_ACTIONS],
    double radiusBuf[Q4Board::NUM_ACTIONS]
  ) const;
  void printPV(std::ostream& out, const std::vector<int>& buf) const;
  void debugPrintChildrenSummary(std::ostream& out, const SearchNode& node, const NNOutput* nnOutput) const;
};

}  // namespace Q4S

#endif  // Q4SEARCH_SEARCH_H_
