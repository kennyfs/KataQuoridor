#include "q4searchnode.h"

#include <algorithm>
#include <cassert>

#include "../../core/test.h"
#include "q4search.h"

namespace Q4S {

NNOutput::NNOutput()
  : noisedPolicyProbs(nullptr),
    stylePolicyProbs(nullptr),
    shorttermWinlossError(0.0f),
    nnHash()
{
  std::fill(policyProbs, policyProbs + Q4Board::NUM_ACTIONS, -1.0f);
  for(int i = 0; i < 5; i++) valueAbs[i] = 0.0f;
}

NNOutput::NNOutput(const NNOutput& other)
  : noisedPolicyProbs(nullptr),
    stylePolicyProbs(nullptr),
    shorttermWinlossError(other.shorttermWinlossError),
    nnHash(other.nnHash)
{
  std::copy(other.policyProbs, other.policyProbs + Q4Board::NUM_ACTIONS, policyProbs);
  std::copy(other.valueAbs, other.valueAbs + 5, valueAbs);
  if(other.noisedPolicyProbs != nullptr) {
    noisedPolicyProbs = new float[Q4Board::NUM_ACTIONS];
    std::copy(other.noisedPolicyProbs, other.noisedPolicyProbs + Q4Board::NUM_ACTIONS, noisedPolicyProbs);
  }
  if(other.stylePolicyProbs != nullptr) {
    stylePolicyProbs = new float[Q4Board::NUM_ACTIONS];
    std::copy(other.stylePolicyProbs, other.stylePolicyProbs + Q4Board::NUM_ACTIONS, stylePolicyProbs);
  }
}

NNOutput::~NNOutput() {
  if(noisedPolicyProbs != nullptr)
    delete[] noisedPolicyProbs;
  if(stylePolicyProbs != nullptr)
    delete[] stylePolicyProbs;
}

NodeStatsAtomic::NodeStatsAtomic()
  : visits(0),
    weightSum(0.0),
    weightSqSum(0.0)
{
  for(int i = 0; i < 5; i++) valueAvg[i].store(0.0, std::memory_order_relaxed);
  for(int i = 0; i < 4; i++) utilityAvg[i].store(0.0, std::memory_order_relaxed);
  for(int i = 0; i < 4; i++) utilitySqAvg[i].store(0.0, std::memory_order_relaxed);
}

NodeStatsAtomic::NodeStatsAtomic(const NodeStatsAtomic& other)
  : visits(other.visits.load(std::memory_order_acquire)),
    weightSum(other.weightSum.load(std::memory_order_acquire)),
    weightSqSum(other.weightSqSum.load(std::memory_order_acquire))
{
  for(int i = 0; i < 5; i++) valueAvg[i].store(other.valueAvg[i].load(std::memory_order_acquire), std::memory_order_relaxed);
  for(int i = 0; i < 4; i++) utilityAvg[i].store(other.utilityAvg[i].load(std::memory_order_acquire), std::memory_order_relaxed);
  for(int i = 0; i < 4; i++) utilitySqAvg[i].store(other.utilitySqAvg[i].load(std::memory_order_acquire), std::memory_order_relaxed);
}

NodeStatsAtomic::~NodeStatsAtomic() {}

NodeStats::NodeStats()
  : visits(0),
    weightSum(0.0),
    weightSqSum(0.0)
{
  for(int i = 0; i < 5; i++) valueAvg[i] = 0.0;
  for(int i = 0; i < 4; i++) utilityAvg[i] = 0.0;
  for(int i = 0; i < 4; i++) utilitySqAvg[i] = 0.0;
}

NodeStats::NodeStats(const NodeStatsAtomic& other)
  : visits(other.visits.load(std::memory_order_acquire)),
    weightSum(other.weightSum.load(std::memory_order_acquire)),
    weightSqSum(other.weightSqSum.load(std::memory_order_acquire))
{
  for(int i = 0; i < 5; i++) valueAvg[i] = other.valueAvg[i].load(std::memory_order_acquire);
  for(int i = 0; i < 4; i++) utilityAvg[i] = other.utilityAvg[i].load(std::memory_order_acquire);
  for(int i = 0; i < 4; i++) utilitySqAvg[i] = other.utilitySqAvg[i].load(std::memory_order_acquire);
}

NodeStats::~NodeStats() {}

MoreNodeStats::MoreNodeStats()
  : stats(),
    selfUtility(0.0),
    weightAdjusted(0.0),
    prevAction(Q4Board::NULL_ACTION)
{}

MoreNodeStats::~MoreNodeStats() {}

SearchChildPointer::SearchChildPointer()
  : data(nullptr),
    edgeVisits(0),
    action(Q4Board::NULL_ACTION)
{}

void SearchChildPointer::storeAll(const SearchChildPointer& other) {
  SearchNode* d = other.data.load(std::memory_order_acquire);
  int64_t e = other.edgeVisits.load(std::memory_order_acquire);
  int a = other.action.load(std::memory_order_acquire);
  action.store(a, std::memory_order_release);
  edgeVisits.store(e, std::memory_order_release);
  data.store(d, std::memory_order_release);
}

bool SearchChildPointer::storeIfNull(SearchNode* node) {
  SearchNode* expected = nullptr;
  return data.compare_exchange_strong(expected, node, std::memory_order_acq_rel);
}

bool SearchChildPointer::compexweakEdgeVisits(int64_t& expected, int64_t desired) {
  return edgeVisits.compare_exchange_weak(expected, desired, std::memory_order_acq_rel);
}

std::atomic<int64_t> SearchNode::liveNodeCount(0);

SearchNode::SearchNode(int nextSeat_, bool fnt, uint32_t mIdx)
  : nextSeat(nextSeat_),
    forceNonTerminal(fnt),
    mutexIdx(mIdx),
    state(SearchNode::STATE_UNEVALUATED),
    nnOutput(),
    nodeAge(0),
    children0(nullptr),
    children1(nullptr),
    children2(nullptr),
    stats(),
    virtualLosses(0),
    dirtyCounter(0)
{
  liveNodeCount.fetch_add(1, std::memory_order_relaxed);
}

SearchNode::SearchNode(const SearchNode& other, bool fnt)
  : nextSeat(other.nextSeat),
    forceNonTerminal(fnt),
    mutexIdx(other.mutexIdx),
    state(other.state.load(std::memory_order_acquire)),
    nnOutput(),
    nodeAge(other.nodeAge.load(std::memory_order_acquire)),
    children0(nullptr),
    children1(nullptr),
    children2(nullptr),
    stats(other.stats),
    virtualLosses(other.virtualLosses.load(std::memory_order_acquire)),
    dirtyCounter(other.dirtyCounter.load(std::memory_order_acquire))
{
  liveNodeCount.fetch_add(1, std::memory_order_relaxed);
  std::shared_ptr<NNOutput>* otherVal = other.nnOutput.load(std::memory_order_acquire);
  if(otherVal != nullptr)
    nnOutput.store(new std::shared_ptr<NNOutput>(*otherVal), std::memory_order_release);

  if(other.children0 != nullptr) {
    children0 = new SearchChildPointer[SearchChildrenSizes::SIZE0OVERFLOW];
    for(int i = 0; i < SearchChildrenSizes::SIZE0OVERFLOW; i++)
      children0[i].storeAll(other.children0[i]);
  }
  if(other.children1 != nullptr) {
    children1 = new SearchChildPointer[SearchChildrenSizes::SIZE1OVERFLOW];
    for(int i = 0; i < SearchChildrenSizes::SIZE1OVERFLOW; i++)
      children1[i].storeAll(other.children1[i]);
  }
  if(other.children2 != nullptr) {
    children2 = new SearchChildPointer[SearchChildrenSizes::SIZE2OVERFLOW];
    for(int i = 0; i < SearchChildrenSizes::SIZE2OVERFLOW; i++)
      children2[i].storeAll(other.children2[i]);
  }
}

SearchNodeChildrenReference SearchNode::getChildren() {
  return SearchNodeChildrenReference(state.load(std::memory_order_acquire), this);
}
ConstSearchNodeChildrenReference SearchNode::getChildren() const {
  return ConstSearchNodeChildrenReference(state.load(std::memory_order_acquire), this);
}
SearchNodeChildrenReference SearchNode::getChildren(SearchNodeState stateValue) {
  return SearchNodeChildrenReference(stateValue, this);
}
ConstSearchNodeChildrenReference SearchNode::getChildren(SearchNodeState stateValue) const {
  return ConstSearchNodeChildrenReference(stateValue, this);
}

static int getChildrenCapacity(SearchNodeState stateValue) {
  if(stateValue < SearchNode::STATE_EXPANDED0)
    return 0;
  else if(stateValue < SearchNode::STATE_EXPANDED1)
    return SearchChildrenSizes::SIZE0TOTAL;
  else if(stateValue < SearchNode::STATE_EXPANDED2)
    return SearchChildrenSizes::SIZE1TOTAL;
  else
    return SearchChildrenSizes::SIZE2TOTAL;
}

int SearchNodeChildrenReference::getCapacity() const {
  return getChildrenCapacity(snapshottedState);
}
int ConstSearchNodeChildrenReference::getCapacity() const {
  return getChildrenCapacity(snapshottedState);
}

int SearchNodeChildrenReference::iterateAndCountChildren() const {
  return ConstSearchNodeChildrenReference(*this).iterateAndCountChildren();
}

int ConstSearchNodeChildrenReference::iterateAndCountChildren() const {
  SearchChildPointer* arr;
  int arrCapacity;
  int offset;

  if(snapshottedState < SearchNode::STATE_EXPANDED0) {
    return 0;
  }
  else if(snapshottedState < SearchNode::STATE_EXPANDED1) {
    arr = node->children0;
    arrCapacity = SearchChildrenSizes::SIZE0OVERFLOW;
    offset = 0;
  }
  else if(snapshottedState < SearchNode::STATE_EXPANDED2) {
    arr = node->children1;
    arrCapacity = SearchChildrenSizes::SIZE1OVERFLOW;
    offset = SearchChildrenSizes::SIZE0TOTAL;
  }
  else {
    arr = node->children2;
    arrCapacity = SearchChildrenSizes::SIZE2OVERFLOW;
    offset = SearchChildrenSizes::SIZE1TOTAL;
  }

  for(int i = 0; i < arrCapacity; i++) {
    if(arr[i].getIfAllocated() == nullptr)
      return offset + i;
  }
  return offset + arrCapacity;
}

bool SearchNode::maybeExpandChildrenCapacityForNewChild(SearchNodeState& stateValue, int numChildrenFullPlusOne) {
  int capacity = getChildrenCapacity(stateValue);
  if(capacity < numChildrenFullPlusOne) {
    assert(capacity == numChildrenFullPlusOne - 1);
    return tryExpandingChildrenCapacityAssumeFull(stateValue);
  }
  return true;
}

void SearchNode::initializeChildren() {
  assert(children0 == nullptr);
  children0 = new SearchChildPointer[SearchChildrenSizes::SIZE0OVERFLOW];
}

bool SearchNode::tryExpandingChildrenCapacityAssumeFull(SearchNodeState& stateValue) {
  if(stateValue < SearchNode::STATE_EXPANDED1) {
    if(stateValue == SearchNode::STATE_GROWING1)
      return false;
    assert(stateValue == SearchNode::STATE_EXPANDED0);
    bool suc = state.compare_exchange_strong(stateValue, SearchNode::STATE_GROWING1, std::memory_order_acq_rel);
    if(!suc) return false;
    stateValue = SearchNode::STATE_GROWING1;

    SearchChildPointer* children = new SearchChildPointer[SearchChildrenSizes::SIZE1OVERFLOW];
    assert(children1 == nullptr);
    children1 = children;
    state.store(SearchNode::STATE_EXPANDED1, std::memory_order_release);
    stateValue = SearchNode::STATE_EXPANDED1;
  }
  else if(stateValue < SearchNode::STATE_EXPANDED2) {
    if(stateValue == SearchNode::STATE_GROWING2)
      return false;
    assert(stateValue == SearchNode::STATE_EXPANDED1);
    bool suc = state.compare_exchange_strong(stateValue, SearchNode::STATE_GROWING2, std::memory_order_acq_rel);
    if(!suc) return false;
    stateValue = SearchNode::STATE_GROWING2;

    SearchChildPointer* children = new SearchChildPointer[SearchChildrenSizes::SIZE2OVERFLOW];
    assert(children2 == nullptr);
    children2 = children;
    state.store(SearchNode::STATE_EXPANDED2, std::memory_order_release);
    stateValue = SearchNode::STATE_EXPANDED2;
  }
  else {
    ASSERT_UNREACHABLE;
  }
  return true;
}

void SearchNode::collapseChildrenCapacity(int numGoodChildren) {
  int stateValue = state.load(std::memory_order_acquire);
  if(numGoodChildren <= SearchChildrenSizes::SIZE1TOTAL && stateValue > SearchNode::STATE_EXPANDED1) {
    assert(stateValue == SearchNode::STATE_EXPANDED2);
    assert(children2 != nullptr);
    for(int i = 0; i < SearchChildrenSizes::SIZE2OVERFLOW; i++) {
      testAssert(children2[i].getIfAllocatedRelaxed() == nullptr);
    }
    delete[] children2;
    children2 = nullptr;
    stateValue = SearchNode::STATE_EXPANDED1;
    state.store(stateValue, std::memory_order_release);
  }
  if(numGoodChildren <= SearchChildrenSizes::SIZE0TOTAL && stateValue > SearchNode::STATE_EXPANDED0) {
    assert(stateValue == SearchNode::STATE_EXPANDED1);
    assert(children1 != nullptr);
    for(int i = 0; i < SearchChildrenSizes::SIZE1OVERFLOW; i++) {
      testAssert(children1[i].getIfAllocatedRelaxed() == nullptr);
    }
    delete[] children1;
    children1 = nullptr;
    stateValue = SearchNode::STATE_EXPANDED0;
    state.store(stateValue, std::memory_order_release);
  }
}

NNOutput* SearchNode::getNNOutput() {
  std::shared_ptr<NNOutput>* nn = nnOutput.load(std::memory_order_acquire);
  if(nn == nullptr)
    return nullptr;
  return nn->get();
}

const NNOutput* SearchNode::getNNOutput() const {
  const std::shared_ptr<NNOutput>* nn = nnOutput.load(std::memory_order_acquire);
  if(nn == nullptr)
    return nullptr;
  return nn->get();
}

bool SearchNode::storeNNOutput(std::shared_ptr<NNOutput>* newNNOutput, SearchThread& thread) {
  std::shared_ptr<NNOutput>* toCleanUp = nnOutput.exchange(newNNOutput, std::memory_order_acq_rel);
  if(toCleanUp != nullptr) {
    thread.oldNNOutputsToCleanUp.push_back(toCleanUp);
    return false;
  }
  return true;
}

bool SearchNode::storeNNOutputIfNull(std::shared_ptr<NNOutput>* newNNOutput) {
  std::shared_ptr<NNOutput>* expected = nullptr;
  return nnOutput.compare_exchange_strong(expected, newNNOutput, std::memory_order_acq_rel);
}

SearchNode::~SearchNode() {
  liveNodeCount.fetch_sub(1, std::memory_order_relaxed);
  if(children2 != nullptr)
    delete[] children2;
  if(children1 != nullptr)
    delete[] children1;
  if(children0 != nullptr)
    delete[] children0;
  std::shared_ptr<NNOutput>* nn = nnOutput.load(std::memory_order_relaxed);
  if(nn != nullptr)
    delete nn;
}

}  // namespace Q4S
