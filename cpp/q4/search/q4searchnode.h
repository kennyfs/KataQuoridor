#ifndef Q4SEARCH_SEARCHNODE_H_
#define Q4SEARCH_SEARCHNODE_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

#include "../../core/global.h"
#include "../../core/hash.h"
#include "../../core/multithread.h"
#include "../q4board.h"
#include "../q4rules.h"
#include "../q4playstate.h"

namespace Q4S {

typedef int SearchNodeState; // See SearchNode::STATE_*

struct SearchNode;
struct SearchThread;

// Neural network output in search node
struct NNOutput {
  float policyProbs[Q4Board::NUM_ACTIONS]; // variant 0, illegal actions -1.0f
  float* noisedPolicyProbs;
  float valueAbs[5];                       // absolute seats 0..3 and draw 4 (masked)
  float shorttermWinlossError;
  Hash128 nnHash;

  NNOutput();
  NNOutput(const NNOutput& other);
  ~NNOutput();
  NNOutput& operator=(const NNOutput&) = delete;

  inline const float* getPolicyProbsMaybeNoised() const {
    return noisedPolicyProbs != nullptr ? noisedPolicyProbs : policyProbs;
  }
};

struct NodeStatsAtomic {
  std::atomic<int64_t> visits;
  std::atomic<double> valueAvg[5];      // probability seat 0..3 wins, [4] = draw
  std::atomic<double> utilityAvg[4];    // utility of seats 0..3
  std::atomic<double> utilitySqAvg[4];  // utility squared of seats 0..3
  std::atomic<double> weightSum;
  std::atomic<double> weightSqSum;

  NodeStatsAtomic();
  explicit NodeStatsAtomic(const NodeStatsAtomic& other);
  ~NodeStatsAtomic();

  NodeStatsAtomic& operator=(const NodeStatsAtomic&) = delete;
  NodeStatsAtomic(NodeStatsAtomic&& other) = delete;
  NodeStatsAtomic& operator=(NodeStatsAtomic&& other) = delete;

  double getChildWeight(int64_t edgeVisits) const;
  double getChildWeight(int64_t edgeVisits, int64_t childVisits) const;
  double getChildWeightSq(int64_t edgeVisits) const;
  double getChildWeightSq(int64_t edgeVisits, int64_t childVisits) const;
};

struct NodeStats {
  int64_t visits;
  double valueAvg[5];
  double utilityAvg[4];
  double utilitySqAvg[4];
  double weightSum;
  double weightSqSum;

  NodeStats();
  explicit NodeStats(const NodeStatsAtomic& other);
  ~NodeStats();

  NodeStats(const NodeStats&) = default;
  NodeStats& operator=(const NodeStats&) = default;
  NodeStats(NodeStats&& other) = default;
  NodeStats& operator=(NodeStats&& other) = default;

  inline static double childWeight(int64_t edgeVisits, int64_t childVisits, double rawChildWeight) {
    return rawChildWeight * ((double)edgeVisits / (double)std::max(childVisits, (int64_t)1));
  }
  inline static double childWeightSq(int64_t edgeVisits, int64_t childVisits, double rawChildWeightSq) {
    return rawChildWeightSq * ((double)edgeVisits / (double)std::max(childVisits, (int64_t)1));
  }
  double getChildWeight(int64_t edgeVisits) const {
    return childWeight(edgeVisits, visits, weightSum);
  }
};

inline double NodeStatsAtomic::getChildWeight(int64_t edgeVisits) const {
  return NodeStats::childWeight(edgeVisits, visits.load(std::memory_order_acquire), weightSum.load(std::memory_order_acquire));
}
inline double NodeStatsAtomic::getChildWeight(int64_t edgeVisits, int64_t childVisits) const {
  return NodeStats::childWeight(edgeVisits, childVisits, weightSum.load(std::memory_order_acquire));
}
inline double NodeStatsAtomic::getChildWeightSq(int64_t edgeVisits) const {
  return NodeStats::childWeightSq(edgeVisits, visits.load(std::memory_order_acquire), weightSqSum.load(std::memory_order_acquire));
}
inline double NodeStatsAtomic::getChildWeightSq(int64_t edgeVisits, int64_t childVisits) const {
  return NodeStats::childWeightSq(edgeVisits, childVisits, weightSqSum.load(std::memory_order_acquire));
}

struct MoreNodeStats {
  NodeStats stats;
  double selfUtility;
  double weightAdjusted;
  int prevAction;

  MoreNodeStats();
  ~MoreNodeStats();

  MoreNodeStats(const MoreNodeStats&) = default;
  MoreNodeStats& operator=(const MoreNodeStats&) = default;
  MoreNodeStats(MoreNodeStats&& other) = default;
  MoreNodeStats& operator=(MoreNodeStats&& other) = default;
};

struct SearchChildPointer {
private:
  std::atomic<SearchNode*> data;
  std::atomic<int64_t> edgeVisits;
  std::atomic<int> action; // Q4 action (0..320, NULL_ACTION = -1)
public:
  SearchChildPointer();

  SearchChildPointer(const SearchChildPointer&) = delete;
  SearchChildPointer& operator=(const SearchChildPointer&) = delete;
  SearchChildPointer(SearchChildPointer&& other) = delete;
  SearchChildPointer& operator=(SearchChildPointer&& other) = delete;

  void storeAll(const SearchChildPointer& other);

  inline SearchNode* getIfAllocated() { return data.load(std::memory_order_acquire); }
  inline const SearchNode* getIfAllocated() const { return data.load(std::memory_order_acquire); }
  inline SearchNode* getIfAllocatedRelaxed() { return data.load(std::memory_order_relaxed); }
  inline void store(SearchNode* node) { data.store(node, std::memory_order_release); }
  inline void storeRelaxed(SearchNode* node) { data.store(node, std::memory_order_relaxed); }
  bool storeIfNull(SearchNode* node);

  inline int64_t getEdgeVisits() const { return edgeVisits.load(std::memory_order_acquire); }
  inline int64_t getEdgeVisitsRelaxed() const { return edgeVisits.load(std::memory_order_relaxed); }
  inline void setEdgeVisits(int64_t x) { edgeVisits.store(x, std::memory_order_release); }
  inline void setEdgeVisitsRelaxed(int64_t x) { edgeVisits.store(x, std::memory_order_relaxed); }
  inline void addEdgeVisits(int64_t delta) { edgeVisits.fetch_add(delta, std::memory_order_acq_rel); }
  bool compexweakEdgeVisits(int64_t& expected, int64_t desired);

  inline int getAction() const { return action.load(std::memory_order_acquire); }
  inline int getActionRelaxed() const { return action.load(std::memory_order_relaxed); }
  inline void setAction(int act) { action.store(act, std::memory_order_release); }
  inline void setActionRelaxed(int act) { action.store(act, std::memory_order_relaxed); }
};

namespace SearchChildrenSizes {
  constexpr int SIZE0TOTAL = 8;
  constexpr int SIZE1TOTAL = 64;
  constexpr int SIZE2TOTAL = Q4Board::NUM_ACTIONS; // 321
  constexpr int SIZE0OVERFLOW = SIZE0TOTAL;
  constexpr int SIZE1OVERFLOW = SIZE1TOTAL - SIZE0TOTAL;
  constexpr int SIZE2OVERFLOW = SIZE2TOTAL - SIZE1TOTAL;
}

// Abstracts children{0,1,2} in SearchNode as a single concatenated array
struct SearchNodeChildrenReference {
  SearchNodeState snapshottedState;
  SearchNode* node;

  inline SearchNodeChildrenReference() {}
  inline SearchNodeChildrenReference(SearchNodeState snapshottedState_, SearchNode* node_)
    : snapshottedState(snapshottedState_), node(node_) {}
  SearchChildPointer& operator[](int i);
  int getCapacity() const;
  int iterateAndCountChildren() const;
};

struct ConstSearchNodeChildrenReference {
  int capacity;
  SearchNodeState snapshottedState;
  const SearchNode* node;

  inline ConstSearchNodeChildrenReference() {}
  inline ConstSearchNodeChildrenReference(const SearchNodeChildrenReference& other)
    : snapshottedState(other.snapshottedState), node(other.node) {}
  inline ConstSearchNodeChildrenReference(SearchNodeState snapshottedState_, const SearchNode* node_)
    : snapshottedState(snapshottedState_), node(node_) {}
  const SearchChildPointer& operator[](int i) const;
  int getCapacity() const;
  int iterateAndCountChildren() const;
};

struct SearchNode {
  static std::atomic<int64_t> liveNodeCount;

  // Locks
  mutable std::atomic_flag statsLock = ATOMIC_FLAG_INIT;

  // Constant during search
  const int nextSeat; // Seat to move from this position (0..3)
  const bool forceNonTerminal;
  const uint32_t mutexIdx;

  // Mutable
  std::atomic<SearchNodeState> state;
  static constexpr SearchNodeState STATE_UNEVALUATED = 0;
  static constexpr SearchNodeState STATE_EVALUATING = 1;
  static constexpr SearchNodeState STATE_EXPANDED0 = 2;
  static constexpr SearchNodeState STATE_GROWING1 = 3;
  static constexpr SearchNodeState STATE_EXPANDED1 = 4;
  static constexpr SearchNodeState STATE_GROWING2 = 5;
  static constexpr SearchNodeState STATE_EXPANDED2 = 6;

  std::atomic<std::shared_ptr<NNOutput>*> nnOutput;
  std::atomic<uint32_t> nodeAge;

  SearchChildPointer* children0; // non-NULL once state >= STATE_EXPANDED0
  SearchChildPointer* children1; // non-NULL once state >= STATE_EXPANDED1
  SearchChildPointer* children2; // non-NULL once state >= STATE_EXPANDED2

  NodeStatsAtomic stats;
  std::atomic<int32_t> virtualLosses;
  std::atomic<int32_t> dirtyCounter;

  SearchNode(int nextSeat, bool forceNonTerminal, uint32_t mutexIdx);
  SearchNode(const SearchNode& other, bool forceNonTerminal);
  ~SearchNode();

  SearchNode& operator=(const SearchNode&) = delete;
  SearchNode(SearchNode&& other) = delete;
  SearchNode& operator=(SearchNode&& other) = delete;

  SearchNodeChildrenReference getChildren();
  ConstSearchNodeChildrenReference getChildren() const;
  SearchNodeChildrenReference getChildren(SearchNodeState state);
  ConstSearchNodeChildrenReference getChildren(SearchNodeState state) const;

  NNOutput* getNNOutput();
  const NNOutput* getNNOutput() const;

  bool storeNNOutput(std::shared_ptr<NNOutput>* newNNOutput, SearchThread& thread);
  bool storeNNOutputIfNull(std::shared_ptr<NNOutput>* newNNOutput);

  void initializeChildren();
  bool maybeExpandChildrenCapacityForNewChild(SearchNodeState& stateValue, int numChildrenFullPlusOne);
  void collapseChildrenCapacity(int numGoodChildren);

private:
  bool tryExpandingChildrenCapacityAssumeFull(SearchNodeState& stateValue);
};

inline SearchChildPointer& SearchNodeChildrenReference::operator[](int i) {
  if(i < SearchChildrenSizes::SIZE0TOTAL) return node->children0[i];
  if(i < SearchChildrenSizes::SIZE1TOTAL) return node->children1[i - SearchChildrenSizes::SIZE0TOTAL];
  return node->children2[i - SearchChildrenSizes::SIZE1TOTAL];
}
inline const SearchChildPointer& ConstSearchNodeChildrenReference::operator[](int i) const {
  if(i < SearchChildrenSizes::SIZE0TOTAL) return node->children0[i];
  if(i < SearchChildrenSizes::SIZE1TOTAL) return node->children1[i - SearchChildrenSizes::SIZE0TOTAL];
  return node->children2[i - SearchChildrenSizes::SIZE1TOTAL];
}

}  // namespace Q4S

#endif  // Q4SEARCH_SEARCHNODE_H_
