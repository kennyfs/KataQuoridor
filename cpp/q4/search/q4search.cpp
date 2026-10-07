#include "q4search.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>
#include <vector>

#include "../../core/fancymath.h"
#include "../../core/test.h"
#include "../../core/timer.h"
#include "q4searchnode.h"
#include "q4searchnodetable.h"

namespace Q4S {

static std::string makeSeed(const Search& search, int threadIdx) {
  std::stringstream ss;
  ss << search.randSeed;
  ss << "$searchThread$";
  ss << threadIdx;
  ss << "$";
  ss << search.rootState.board.hash;
  ss << "$";
  ss << search.rootState.plies;
  ss << "$";
  ss << search.numSearchesBegun;
  return ss.str();
}

SearchThread::SearchThread(int tIdx, const Search& search)
  : threadIdx(tIdx),
    seat(search.rootSeat),
    state(search.rootState),
    shouldCountPlayout(false),
    rand(makeSeed(search, tIdx)),
    nnResultBuf(),
    statsBuf(),
    upperBoundVisitsLeft(1e30),
    oldNNOutputsToCleanUp()
{
  statsBuf.resize(Q4Board::NUM_ACTIONS);
  oldNNOutputsToCleanUp.reserve(8);
}

SearchThread::~SearchThread() {
  for(size_t i = 0; i < oldNNOutputsToCleanUp.size(); i++)
    delete oldNNOutputsToCleanUp[i];
  oldNNOutputsToCleanUp.clear();
}

static const double VALUE_WEIGHT_DEGREES_OF_FREEDOM = 3.0;

void Search::checkParams(const SearchParams& params) {
  std::vector<std::string> offending;
  if(params.staticScoreUtilityFactor != 0.0) offending.push_back("staticScoreUtilityFactor");
  if(params.dynamicScoreUtilityFactor != 0.0) offending.push_back("dynamicScoreUtilityFactor");
  if(params.policyOptimism != 0.0) offending.push_back("policyOptimism");
  if(params.rootPolicyOptimism != 0.0) offending.push_back("rootPolicyOptimism");
  if(params.useGraphSearch) offending.push_back("useGraphSearch");
  if(params.useEvalCache) offending.push_back("useEvalCache");
  if(params.subtreeValueBiasFactor != 0.0) offending.push_back("subtreeValueBiasFactor");
  if(params.avoidRepeatedPatternUtility != 0.0) offending.push_back("avoidRepeatedPatternUtility");
  if(params.antiMirror) offending.push_back("antiMirror");
  if(params.playoutDoublingAdvantage != 0.0) offending.push_back("playoutDoublingAdvantage");
  if(params.visitCapContempt != 0.0) offending.push_back("visitCapContempt");
  if(params.rootSymmetryPruning) offending.push_back("rootSymmetryPruning");
  if(params.conservativePass) offending.push_back("conservativePass");
  if(params.enablePassingHacks) offending.push_back("enablePassingHacks");
  if(params.enableMorePassingHacks) offending.push_back("enableMorePassingHacks");
  if(params.fillDameBeforePass) offending.push_back("fillDameBeforePass");
  if(params.rootEndingBonusPoints != 0.0) offending.push_back("rootEndingBonusPoints");
  if(params.rootPruneUselessMoves) offending.push_back("rootPruneUselessMoves");
  if(params.ignorePreRootHistory) offending.push_back("ignorePreRootHistory");
  if(params.ignoreAllHistory) offending.push_back("ignoreAllHistory");
  if(params.humanSLRootExploreProbWeightless != 0.0 ||
     params.humanSLRootExploreProbWeightful != 0.0 ||
     params.humanSLPlaExploreProbWeightless != 0.0 ||
     params.humanSLPlaExploreProbWeightful != 0.0 ||
     params.humanSLOppExploreProbWeightless != 0.0 ||
     params.humanSLOppExploreProbWeightful != 0.0 ||
     params.humanSLProfile.initialized)
    offending.push_back("humanSL");

  if(!offending.empty()) {
    std::string msg = "SearchParams contains forbidden parameters for Q4: ";
    for(size_t i = 0; i < offending.size(); i++) {
      if(i > 0) msg += ", ";
      msg += offending[i];
    }
    throw StringError(msg);
  }
}

Search::Search(
  const SearchParams& params,
  NNEvaluator* nnEval,
  Logger* lg,
  const std::string& rSeed
) : rootSeat(0),
    rootAliveMask(0x0f),
    rootNumAlive(4),
    rootState(),
    rootHintAction(Q4Board::NULL_ACTION),
    searchParams(params),
    numSearchesBegun(0),
    searchNodeAge(0),
    lastSearchNumPlayouts(0),
    effectiveSearchTimeCarriedOver(0.0),
    randSeed(rSeed),
    valueWeightDistribution(NULL),
    normToTApproxZ(0.0),
    normToTApproxTable(),
    nonSearchRand(rSeed + "$nonSearchRand"),
    logger(lg),
    nnEvaluator(nnEval),
    rootNode(NULL),
    nodeTable(NULL),
    mutexPool(NULL),
    numThreadsSpawned(0),
    threads(NULL),
    threadTasks(NULL),
    threadTasksRemaining(NULL),
    oldNNOutputsToCleanUpMutex(),
    oldNNOutputsToCleanUp()
{
  checkParams(params);

  valueWeightDistribution = new DistributionTable(
    [](double z) { return FancyMath::tdistpdf(z, VALUE_WEIGHT_DEGREES_OF_FREEDOM); },
    [](double z) { return FancyMath::tdistcdf(z, VALUE_WEIGHT_DEGREES_OF_FREEDOM); },
    -50.0,
    50.0,
    2000
  );

  nodeTable = new SearchNodeTable(params.nodeTableShardsPowerOfTwo);
  mutexPool = new MutexPool(nodeTable->mutexPool->getNumMutexes());
}

Search::~Search() {
  clearSearch();
  delete valueWeightDistribution;
  delete nodeTable;
  delete mutexPool;
  clearOldNNOutputs();
  killThreads();
}

double Search::computeSeatUtility(
  int seat,
  const double value[5],
  int nAlive,
  bool isAlive,
  double winLossUtilityFactor
) {
  if(!isAlive)
    return -winLossUtilityFactor;
  if(nAlive <= 0)
    return -winLossUtilityFactor;
  return winLossUtilityFactor * (2.0 * value[seat] + (2.0 / (double)nAlive) * value[4] - 1.0);
}

void Search::setPosition(const Q4PlayState& state) {
  clearSearch();
  rootState = state;
  rootSeat = rootState.board.toMove;
  rootAliveMask = rootState.board.alive;
  rootNumAlive = rootState.board.getNumAlive();
}

void Search::setPosition(const Q4History& history) {
  clearSearch();
  rootState = history.getState();
  rootSeat = rootState.board.toMove;
  rootAliveMask = rootState.board.alive;
  rootNumAlive = rootState.board.getNumAlive();
}

void Search::setRootHintAction(int action) {
  rootHintAction = action;
}

void Search::setParams(const SearchParams& params) {
  checkParams(params);
  clearSearch();
  searchParams = params;
}

void Search::setParamsNoClearing(const SearchParams& params) {
  checkParams(params);
  searchParams = params;
}

void Search::setNNEval(NNEvaluator* nnEval) {
  clearSearch();
  nnEvaluator = nnEval;
}

void Search::clearSearch() {
  effectiveSearchTimeCarriedOver = 0.0;
  if(rootNode != NULL) {
    deleteAllTableNodesMulithreaded();
    if(rootNode != NULL) {
      delete rootNode;
      rootNode = NULL;
    }
  }
  clearOldNNOutputs();
  searchNodeAge = 0;
  lastSearchNumPlayouts = 0;
  rootHintAction = Q4Board::NULL_ACTION;
}

bool Search::makeMove(int action) {
  if(!rootState.isLegalAction(action))
    return false;

  rootState.playAssumeLegal(action);
  rootSeat = rootState.board.toMove;
  rootAliveMask = rootState.board.alive;
  rootNumAlive = rootState.board.getNumAlive();

  if(rootNode != NULL) {
    SearchNode* child = NULL;
    {
      SearchNodeChildrenReference children = rootNode->getChildren();
      int childrenCapacity = children.getCapacity();
      for(int i = 0; i < childrenCapacity; i++) {
        SearchNode* childCandidate = children[i].getIfAllocated();
        if(childCandidate == NULL)
          break;
        if(children[i].getActionRelaxed() == action) {
          child = childCandidate;
          break;
        }
      }
    }

    if(child != NULL) {
      NNOutput* nnOutput = child->getNNOutput();
      if(nnOutput == NULL)
        child = NULL;
    }

    if(child != NULL) {
      int64_t rootVisits = rootNode->stats.visits.load(std::memory_order_acquire);
      int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
      double visitProportion = (double)childVisits / (double)std::max((int64_t)1, rootVisits);
      if(visitProportion > 1.0)
        visitProportion = 1.0;
      effectiveSearchTimeCarriedOver = effectiveSearchTimeCarriedOver * visitProportion * searchParams.treeReuseCarryOverTimeFactor;

      SearchNode* oldRootNode = rootNode;
      const bool forceNonTerminal = rootState.isFinished;
      rootNode = new SearchNode(*child, forceNonTerminal);

      applyRecursivelyAnyOrderMulithreaded({rootNode}, NULL);
      bool old = true;
      deleteAllOldOrAllNewTableNodesMulithreaded(old);
      delete oldRootNode;
    }
    else {
      clearSearch();
    }
  }

  return true;
}

int Search::runWholeSearchAndGetMove() {
  runWholeSearch();
  return getChosenMoveAction();
}

int Search::runWholeSearchAndGetMove(std::atomic<bool>* shouldStop) {
  runWholeSearch(shouldStop);
  return getChosenMoveAction();
}

void Search::runWholeSearch() {
  std::function<bool()>* shouldStopEarly = NULL;
  runWholeSearch(NULL, shouldStopEarly, TimeControls(), 1.0);
}

void Search::runWholeSearch(std::atomic<bool>* shouldStop) {
  std::function<bool()> stopFunc = [shouldStop]() noexcept {
    return shouldStop != nullptr && shouldStop->load(std::memory_order_relaxed);
  };
  runWholeSearch(NULL, &stopFunc, TimeControls(), 1.0);
}

void Search::runWholeSearch(std::function<bool()>* shouldStopEarly) {
  runWholeSearch(NULL, shouldStopEarly, TimeControls(), 1.0);
}

void Search::runWholeSearch(
  const std::function<void()>* searchBegun,
  std::function<bool()>* shouldStopEarly,
  const TimeControls& tc,
  double searchFactor
) {
  ClockTimer timer;
  std::atomic<int64_t> numPlayoutsShared(0);
  std::atomic<bool> shouldStopNow(false);

  beginSearch();
  if(searchBegun != NULL)
    (*searchBegun)();
  const int64_t numNonPlayoutVisits = getRootVisits();

  int64_t maxVisits = searchParams.maxVisits;
  int64_t maxPlayouts = searchParams.maxPlayouts;
  double maxTime = searchParams.maxTime;

  if(searchFactor != 1.0) {
    double cap = (double)((int64_t)1L << 62);
    maxVisits = (int64_t)ceil(std::min(cap, maxVisits * searchFactor));
    maxPlayouts = (int64_t)ceil(std::min(cap, maxPlayouts * searchFactor));
    maxTime = maxTime * searchFactor;
  }

  int capThreads = 0x3fffFFFF;
  if(searchParams.minPlayoutsPerThread > 0.0) {
    int64_t numNewPlayouts = std::min(maxVisits - numNonPlayoutVisits, maxPlayouts);
    double cap = numNewPlayouts / searchParams.minPlayoutsPerThread;
    if(!std::isnan(cap) && cap < (double)0x3fffFFFF) {
      capThreads = std::max(1, (int)floor(cap));
    }
  }

  std::atomic<double> tcMaxTime(1e30);
  std::atomic<double> upperBoundVisitsLeftDueToTime(1e30);
  const bool hasMaxTime = maxTime < 1.0e12;
  const bool hasTc = !tc.isEffectivelyUnlimitedTime();
  if(hasTc || hasMaxTime) {
    int64_t rootVisits = numPlayoutsShared.load(std::memory_order_relaxed) + numNonPlayoutVisits;
    double timeUsed = timer.getSeconds();
    double tcLimit = 1e30;
    if(hasTc) {
      tcLimit = recomputeSearchTimeLimit(tc, timeUsed, searchFactor, rootVisits);
      tcMaxTime.store(tcLimit, std::memory_order_release);
    }
    double upperBoundVisits = computeUpperBoundVisitsLeftDueToTime(rootVisits, timeUsed, std::min(tcLimit, maxTime));
    upperBoundVisitsLeftDueToTime.store(upperBoundVisits, std::memory_order_release);
  }

  std::function<void(int)> searchLoop = [
    this, &timer, &numPlayoutsShared, numNonPlayoutVisits, &tcMaxTime, &upperBoundVisitsLeftDueToTime, &tc,
    hasMaxTime, hasTc, &shouldStopNow, shouldStopEarly, maxVisits, maxPlayouts, maxTime, searchFactor
  ](int threadIdx) {
    struct timespec tsStart, tsEnd;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &tsStart);
    SearchThread* stbuf = new SearchThread(threadIdx, *this);
    int64_t numPlayouts = numPlayoutsShared.load(std::memory_order_relaxed);
    try {
      double lastTimeUsedRecomputingTcLimit = 0.0;
      while(true) {
        double timeUsed = 0.0;
        if(hasTc || hasMaxTime)
          timeUsed = timer.getSeconds();

        double tcMaxTimeLimit = 0.0;
        if(hasTc)
          tcMaxTimeLimit = tcMaxTime.load(std::memory_order_acquire);

        bool shouldStop =
          (numPlayouts >= maxPlayouts) ||
          (numPlayouts + numNonPlayoutVisits >= maxVisits);

        if(hasMaxTime && numPlayouts >= 2 && timeUsed >= maxTime)
          shouldStop = true;
        if(hasTc && numPlayouts >= 2 && timeUsed >= tcMaxTimeLimit)
          shouldStop = true;
        if(shouldStopEarly != NULL && (*shouldStopEarly)())
          shouldStop = true;

        if(shouldStop || shouldStopNow.load(std::memory_order_relaxed)) {
          shouldStopNow.store(true, std::memory_order_relaxed);
          break;
        }

        if(hasTc && threadIdx == 0 && timeUsed >= lastTimeUsedRecomputingTcLimit + 0.1) {
          int64_t rootVisits = numPlayouts + numNonPlayoutVisits;
          double tcLimit = recomputeSearchTimeLimit(tc, timeUsed, searchFactor, rootVisits);
          tcMaxTime.store(tcLimit, std::memory_order_release);
          double upperBoundVisits = computeUpperBoundVisitsLeftDueToTime(rootVisits, timeUsed, std::min(tcLimit, maxTime));
          upperBoundVisitsLeftDueToTime.store(upperBoundVisits, std::memory_order_release);
          lastTimeUsedRecomputingTcLimit = timeUsed;
        }

        double upperBoundVisitsLeft = 1e30;
        if(hasTc)
          upperBoundVisitsLeft = upperBoundVisitsLeftDueToTime.load(std::memory_order_acquire);
        upperBoundVisitsLeft = std::min(upperBoundVisitsLeft, (double)maxPlayouts - numPlayouts);
        upperBoundVisitsLeft = std::min(upperBoundVisitsLeft, (double)maxVisits - numPlayouts - numNonPlayoutVisits);

        bool finishedPlayout = runSinglePlayout(*stbuf, upperBoundVisitsLeft);
        if(finishedPlayout) {
          numPlayouts = numPlayoutsShared.fetch_add((int64_t)1, std::memory_order_relaxed);
          numPlayouts += 1;
        }
        else {
          std::this_thread::yield();
        }
      }
    }
    catch(...) {
      clock_gettime(CLOCK_THREAD_CPUTIME_ID, &tsEnd);
      int64_t ns = (int64_t)(tsEnd.tv_sec - tsStart.tv_sec) * 1000000000LL + (int64_t)(tsEnd.tv_nsec - tsStart.tv_nsec);
      totalSearchThreadCpuTimeNs.fetch_add(ns, std::memory_order_relaxed);
      transferOldNNOutputs(*stbuf);
      delete stbuf;
      throw;
    }

    transferOldNNOutputs(*stbuf);
    delete stbuf;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &tsEnd);
    int64_t ns = (int64_t)(tsEnd.tv_sec - tsStart.tv_sec) * 1000000000LL + (int64_t)(tsEnd.tv_nsec - tsStart.tv_nsec);
    totalSearchThreadCpuTimeNs.fetch_add(ns, std::memory_order_relaxed);
  };

  performTaskWithThreads(&searchLoop, capThreads);

  if(rootNode != NULL && rootNode->nodeAge.load(std::memory_order_acquire) != searchNodeAge) {
    if(rootNode->getNNOutput() != nullptr) {
      const int threadIdx = 0;
      const bool isRoot = true;
      SearchThread thread(threadIdx, *this);
      maybeRecomputeExistingNNOutput(thread, *rootNode, isRoot);
    }
  }

  lastSearchNumPlayouts = numPlayoutsShared.load(std::memory_order_relaxed);
}

void Search::beginSearch() {
  numSearchesBegun += 1;
  searchNodeAge += 1;

  clearOldNNOutputs();

  if(rootNode == NULL) {
    const bool forceNonTerminal = true;
    SearchThread dummyThread(0, *this);
    uint32_t mutexIdx = createMutexIdxForNode(dummyThread);
    rootNode = new SearchNode(rootSeat, forceNonTerminal, mutexIdx);
  }

  if(rootNode->state.load(std::memory_order_acquire) == SearchNode::STATE_UNEVALUATED) {
    SearchThread thread(0, *this);
    initNodeNNOutput(thread, *rootNode, true, false, false);
    rootNode->initializeChildren();
    rootNode->state.store(SearchNode::STATE_EXPANDED0, std::memory_order_seq_cst);
  }
}

void Search::computeRootValues() {}

bool Search::runSinglePlayout(SearchThread& thread, double upperBoundVisitsLeft) {
  thread.upperBoundVisitsLeft = upperBoundVisitsLeft;
  thread.shouldCountPlayout = true;

  bool finishedPlayout = playoutDescend(thread, *rootNode, true);
  (void)finishedPlayout;

  thread.seat = rootSeat;
  thread.state = rootState;

  return thread.shouldCountPlayout;
}

bool Search::playoutDescend(SearchThread& thread, SearchNode& node, bool isRoot) {
  if(thread.state.isFinished && !node.forceNonTerminal) {
    nnEvaluator->waitForNextNNEvalIfAny();
    double value[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    if(thread.state.isDraw) {
      value[4] = 1.0;
    }
    else {
      int winner = thread.state.winnerSeat;
      if(winner >= 0 && winner < 4)
        value[winner] = 1.0;
      else
        value[4] = 1.0;
    }
    double weight = (searchParams.useUncertainty && nnEvaluator->supportsShorttermError()) ? searchParams.uncertaintyMaxWeight : 1.0;
    addLeafValue(node, value, weight, true, false);
    return true;
  }

  SearchNodeState nodeState = node.state.load(std::memory_order_acquire);
  if(nodeState == SearchNode::STATE_UNEVALUATED) {
    bool suc = initNodeNNOutput(thread, node, isRoot, false, false);
    if(!suc) {
      thread.shouldCountPlayout = false;
      return false;
    }

    suc = node.state.compare_exchange_strong(nodeState, SearchNode::STATE_EVALUATING, std::memory_order_seq_cst);
    if(!suc) {
      thread.shouldCountPlayout = false;
      return false;
    }
    else {
      node.initializeChildren();
      node.state.store(SearchNode::STATE_EXPANDED0, std::memory_order_seq_cst);
      return true;
    }
  }
  else if(nodeState == SearchNode::STATE_EVALUATING) {
    thread.shouldCountPlayout = false;
    return false;
  }

  assert(nodeState >= SearchNode::STATE_EXPANDED0);
  maybeRecomputeExistingNNOutput(thread, node, isRoot);

  int numChildrenFound;
  int bestChildIdx;
  int bestChildMoveAction;
  bool countEdgeVisit;

  SearchNode* child = NULL;
  while(true) {
    selectBestChildToDescend(thread, node, nodeState, numChildrenFound, bestChildIdx, bestChildMoveAction, countEdgeVisit, isRoot);

    if(bestChildIdx <= -1) {
      addCurrentNNOutputAsLeafValue(node, false);
      return true;
    }

    if(bestChildIdx >= numChildrenFound) {
      assert(bestChildIdx == numChildrenFound);
      assert(bestChildIdx < Q4Board::NUM_ACTIONS);

      bool suc = node.maybeExpandChildrenCapacityForNewChild(nodeState, numChildrenFound + 1);
      if(!suc) {
        std::this_thread::yield();
        nodeState = node.state.load(std::memory_order_acquire);
        continue;
      }

      SearchNodeChildrenReference children = node.getChildren(nodeState);

      thread.state.playAssumeLegal(bestChildMoveAction);
      thread.seat = thread.state.board.toMove;

      const bool forceNonTerminal = false;
      child = allocateOrFindNode(thread, thread.seat, bestChildMoveAction, forceNonTerminal);
      child->virtualLosses.fetch_add(1, std::memory_order_release);

      {
        std::lock_guard<std::mutex> lock(mutexPool->getMutex(node.mutexIdx));
        SearchNode* existingChild = children[bestChildIdx].getIfAllocated();
        if(existingChild == NULL) {
          SearchChildPointer& childPointer = children[bestChildIdx];
          childPointer.setActionRelaxed(bestChildMoveAction);
          childPointer.store(child);
        }
        else {
          child->virtualLosses.fetch_add(-1, std::memory_order_release);
          thread.shouldCountPlayout = false;
          return false;
        }
      }
    }
    else {
      SearchNodeChildrenReference children = node.getChildren(nodeState);
      child = children[bestChildIdx].getIfAllocated();
      assert(child != NULL);

      child->virtualLosses.fetch_add(1, std::memory_order_release);
      thread.state.playAssumeLegal(bestChildMoveAction);
      thread.seat = thread.state.board.toMove;
    }

    break;
  }

  bool shouldUpdateChildAncestors = playoutDescend(thread, *child, false);

  shouldUpdateChildAncestors = shouldUpdateChildAncestors && countEdgeVisit;
  if(shouldUpdateChildAncestors) {
    nodeState = node.state.load(std::memory_order_acquire);
    SearchNodeChildrenReference children = node.getChildren(nodeState);
    children[bestChildIdx].addEdgeVisits(1);
    updateStatsAfterPlayout(node, thread, isRoot);
  }
  child->virtualLosses.fetch_add(-1, std::memory_order_release);

  return shouldUpdateChildAncestors;
}

uint32_t Search::createMutexIdxForNode(SearchThread& thread) const {
  return thread.rand.nextUInt() & (mutexPool->getNumMutexes() - 1);
}

SearchNode* Search::allocateOrFindNode(SearchThread& thread, int nextSeat, int action, bool forceNonTerminal) {
  (void)action;
  Hash128 childHash = thread.state.board.hash ^ Hash128(thread.rand.nextUInt64(), thread.rand.nextUInt64());

  uint32_t nodeTableIdx = nodeTable->getIndex(childHash.hash0);
  std::mutex& mutex = nodeTable->mutexPool->getMutex(nodeTableIdx);
  std::lock_guard<std::mutex> lock(mutex);

  SearchNode* child = NULL;
  std::map<Hash128, SearchNode*>& nodeMap = nodeTable->entries[nodeTableIdx];

  while(true) {
    auto insertLoc = nodeMap.lower_bound(childHash);

    if(insertLoc != nodeMap.end() && insertLoc->first == childHash) {
      if(insertLoc->second->nextSeat != nextSeat) {
        childHash = thread.state.board.hash ^ Hash128(thread.rand.nextUInt64(), thread.rand.nextUInt64());
        continue;
      }
      child = insertLoc->second;
    }
    else {
      child = new SearchNode(nextSeat, forceNonTerminal, createMutexIdxForNode(thread));
      nodeMap.insert(insertLoc, std::make_pair(childHash, child));
    }
    break;
  }
  return child;
}

void Search::clearOldNNOutputs() {
  std::lock_guard<std::mutex> lock(oldNNOutputsToCleanUpMutex);
  for(size_t i = 0; i < oldNNOutputsToCleanUp.size(); i++)
    delete oldNNOutputsToCleanUp[i];
  oldNNOutputsToCleanUp.clear();
}

void Search::transferOldNNOutputs(SearchThread& thread) {
  if(thread.oldNNOutputsToCleanUp.empty())
    return;
  std::lock_guard<std::mutex> lock(oldNNOutputsToCleanUpMutex);
  for(size_t i = 0; i < thread.oldNNOutputsToCleanUp.size(); i++)
    oldNNOutputsToCleanUp.push_back(thread.oldNNOutputsToCleanUp[i]);
  thread.oldNNOutputsToCleanUp.clear();
}

void Search::deleteAllOldOrAllNewTableNodesMulithreaded(bool old) {
  int numAdditionalThreads = numAdditionalThreadsToUseForTasks();
  testAssert(numAdditionalThreads >= 0);
  std::function<void(int)> g = [&](int threadIdx) {
    size_t idx0 = (size_t)((uint64_t)(threadIdx) * nodeTable->entries.size() / (numAdditionalThreads + 1));
    size_t idx1 = (size_t)((uint64_t)(threadIdx + 1) * nodeTable->entries.size() / (numAdditionalThreads + 1));
    for(size_t i = idx0; i < idx1; i++) {
      std::map<Hash128, SearchNode*>& nodeMap = nodeTable->entries[i];
      for(auto it = nodeMap.cbegin(); it != nodeMap.cend();) {
        SearchNode* node = it->second;
        if(old == (node->nodeAge.load(std::memory_order_acquire) < searchNodeAge)) {
          delete node;
          it = nodeMap.erase(it);
        }
        else
          ++it;
      }
    }
  };
  performTaskWithThreads(&g, 0x3FFFffff);
}

void Search::deleteAllTableNodesMulithreaded() {
  int numAdditionalThreads = numAdditionalThreadsToUseForTasks();
  testAssert(numAdditionalThreads >= 0);
  std::function<void(int)> g = [&](int threadIdx) noexcept {
    size_t idx0 = (size_t)((uint64_t)(threadIdx) * nodeTable->entries.size() / (numAdditionalThreads + 1));
    size_t idx1 = (size_t)((uint64_t)(threadIdx + 1) * nodeTable->entries.size() / (numAdditionalThreads + 1));
    for(size_t i = idx0; i < idx1; i++) {
      std::map<Hash128, SearchNode*>& nodeMap = nodeTable->entries[i];
      for(auto it = nodeMap.cbegin(); it != nodeMap.cend(); ++it) {
        delete it->second;
      }
      nodeMap.clear();
    }
  };
  performTaskWithThreads(&g, 0x3FFFffff);
}

}  // namespace Q4S
