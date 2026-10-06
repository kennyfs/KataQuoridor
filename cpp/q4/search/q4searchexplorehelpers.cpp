#include "q4search.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "q4searchnode.h"

namespace Q4S {

static double cpuctExploration(double totalChildWeight, const SearchParams& searchParams) {
  return searchParams.cpuctExploration +
    searchParams.cpuctExplorationLog * log((totalChildWeight + searchParams.cpuctExplorationBase) / searchParams.cpuctExplorationBase);
}

// Tiny constant to add to numerator of puct formula to make it positive
// even when visits = 0.
static constexpr double TOTALCHILDWEIGHT_PUCT_OFFSET = 0.01;

double Search::getExploreScaling(
  double totalChildWeight, double parentUtilityStdevFactor
) const {
  return
    cpuctExploration(totalChildWeight, searchParams)
    * sqrt(totalChildWeight + TOTALCHILDWEIGHT_PUCT_OFFSET)
    * parentUtilityStdevFactor;
}

double Search::getExploreSelectionValue(
  double exploreScaling,
  double nnPolicyProb,
  double childWeight,
  double childUtility
) const {
  if(nnPolicyProb < 0)
    return POLICY_ILLEGAL_SELECTION_VALUE;

  double exploreComponent = exploreScaling * nnPolicyProb / (1.0 + childWeight);
  double valueComponent = childUtility;
  return exploreComponent + valueComponent;
}

// Return the childWeight that would make Search::getExploreSelectionValue return the given explore selection value.
// Or return 0, if it would be less than 0.
double Search::getExploreSelectionValueInverse(
  double exploreScaling,
  double exploreSelectionValue,
  double nnPolicyProb,
  double childUtility
) const {
  if(nnPolicyProb < 0)
    return 0;
  double valueComponent = childUtility;

  double exploreComponent = exploreSelectionValue - valueComponent;
  double exploreComponentScaling = exploreScaling * nnPolicyProb;

  // Guard against float weirdness
  if(exploreComponent <= 0)
    return 1e100;

  double childWeight = exploreComponentScaling / exploreComponent - 1;
  if(childWeight < 0)
    childWeight = 0;
  return childWeight;
}

double Search::getExploreSelectionValueOfChild(
  const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
  int action,
  double exploreScaling,
  double totalChildWeight, int64_t childEdgeVisits, double fpuValue,
  double parentUtility, double parentWeightPerVisit,
  bool isDuringSearch, double maxChildWeight,
  bool countEdgeVisit,
  SearchThread* thread
) const {
  (void)parentUtility;
  (void)totalChildWeight;
  int movePos = action;
  float nnPolicyProb = parentPolicyProbs[movePos];

  int32_t childVirtualLosses = child->virtualLosses.load(std::memory_order_acquire);
  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  int mover = parent.nextSeat;
  double utilityAvg = child->stats.utilityAvg[mover].load(std::memory_order_acquire);
  double childWeight;
  if(countEdgeVisit)
    childWeight = child->stats.getChildWeight(childEdgeVisits, childVisits);
  else
    childWeight = child->stats.weightSum.load(std::memory_order_acquire);

  double childUtility;
  if(childVisits <= 0 || childWeight <= 0.0)
    childUtility = fpuValue;
  else
    childUtility = utilityAvg;

  // Virtual losses to direct threads down different paths
  if(childVirtualLosses > 0) {
    double virtualLossWeight = childVirtualLosses * searchParams.numVirtualLossesPerThread;
    double utilityRadius = searchParams.winLossUtilityFactor;
    double virtualLossUtility = -utilityRadius;
    double virtualLossWeightFrac = (double)virtualLossWeight / (virtualLossWeight + std::max(0.25, childWeight));
    childUtility = childUtility + (virtualLossUtility - childUtility) * virtualLossWeightFrac;
    childWeight += virtualLossWeight;
  }

  if(isDuringSearch && (&parent == rootNode) && countEdgeVisit) {
    if(searchParams.futileVisitsThreshold > 0) {
      double requiredWeight = searchParams.futileVisitsThreshold * maxChildWeight;
      double averageVisitsPerWeight = (childEdgeVisits + 1.0) / (childWeight + parentWeightPerVisit);
      double estimatedRequiredVisits = requiredWeight * averageVisitsPerWeight;
      if(childVisits + thread->upperBoundVisitsLeft < estimatedRequiredVisits)
        return FUTILE_VISITS_PRUNE_VALUE;
    }
  }

  return getExploreSelectionValue(exploreScaling, nnPolicyProb, childWeight, childUtility);
}

double Search::getNewExploreSelectionValue(
  const SearchNode& parent,
  double exploreScaling,
  float nnPolicyProb,
  double fpuValue,
  double parentWeightPerVisit,
  double maxChildWeight,
  bool countEdgeVisit,
  SearchThread* thread
) const {
  double childWeight = 0;
  double childUtility = fpuValue;
  if(&parent == rootNode && countEdgeVisit) {
    if(searchParams.futileVisitsThreshold > 0) {
      double averageVisitsPerWeight = 1.0 / parentWeightPerVisit;
      double requiredWeight = searchParams.futileVisitsThreshold * maxChildWeight;
      double estimatedRequiredVisits = requiredWeight * averageVisitsPerWeight;
      if(thread->upperBoundVisitsLeft < estimatedRequiredVisits)
        return FUTILE_VISITS_PRUNE_VALUE;
    }
  }
  return getExploreSelectionValue(exploreScaling, nnPolicyProb, childWeight, childUtility);
}

double Search::getReducedPlaySelectionWeight(
  const SearchNode& parent, const float* parentPolicyProbs, const SearchNode* child,
  int action,
  double exploreScaling,
  int64_t childEdgeVisits,
  double bestChildExploreSelectionValue
) const {
  assert(&parent == rootNode);
  int movePos = action;
  float nnPolicyProb = parentPolicyProbs[movePos];

  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  int mover = parent.nextSeat;
  double utilityAvg = child->stats.utilityAvg[mover].load(std::memory_order_acquire);
  double childWeight = child->stats.getChildWeight(childEdgeVisits, childVisits);

  if(childVisits <= 0 || childWeight <= 0.0)
    return 0;

  double childUtility = utilityAvg;
  double childWeightWeRetrospectivelyWanted = getExploreSelectionValueInverse(
    bestChildExploreSelectionValue, exploreScaling, nnPolicyProb, childUtility
  );
  if(childWeight > childWeightWeRetrospectivelyWanted)
    return childWeightWeRetrospectivelyWanted;
  return childWeight;
}

double Search::getFpuValueForChildrenAssumeVisited(
  const SearchNode& node, int mover, bool isRoot, double policyProbMassVisited,
  double& parentUtility, double& parentWeightPerVisit, double& parentUtilityStdevFactor
) const {
  int64_t visits = node.stats.visits.load(std::memory_order_acquire);
  double weightSum = node.stats.weightSum.load(std::memory_order_acquire);
  double utilityAvg = node.stats.utilityAvg[mover].load(std::memory_order_acquire);
  double utilitySqAvg = node.stats.utilitySqAvg[mover].load(std::memory_order_acquire);

  assert(visits > 0);
  assert(weightSum > 0.0);
  parentWeightPerVisit = weightSum / visits;
  parentUtility = utilityAvg;
  double variancePrior = searchParams.cpuctUtilityStdevPrior * searchParams.cpuctUtilityStdevPrior;
  double variancePriorWeight = searchParams.cpuctUtilityStdevPriorWeight;
  double parentUtilityStdev;
  if(visits <= 0 || weightSum <= 1)
    parentUtilityStdev = searchParams.cpuctUtilityStdevPrior;
  else {
    double utilitySq = parentUtility * parentUtility;
    if(utilitySqAvg < utilitySq)
      utilitySqAvg = utilitySq;
    parentUtilityStdev = sqrt(
      std::max(
        0.0,
        ((utilitySq + variancePrior) * variancePriorWeight + utilitySqAvg * weightSum)
        / (variancePriorWeight + weightSum - 1.0)
        - utilitySq
      )
    );
  }
  parentUtilityStdevFactor = 1.0 + searchParams.cpuctUtilityStdevScale * (parentUtilityStdev / searchParams.cpuctUtilityStdevPrior - 1.0);

  double parentUtilityForFPU = parentUtility;
  const NNOutput* nnOutput = node.getNNOutput();
  if(nnOutput != nullptr) {
    double val[5];
    for(int i = 0; i < 5; i++) val[i] = nnOutput->valueAbs[i];
    // Approximate alive count
    int nAlive = 0;
    for(int s = 0; s < 4; s++) {
      if(val[s] > 0.0 || node.stats.valueAvg[s].load(std::memory_order_acquire) > 0.0)
        nAlive++;
    }
    if(nAlive == 0) nAlive = 4;
    double nnUtility = computeSeatUtility(mover, val, nAlive, true, searchParams.winLossUtilityFactor);

    if(searchParams.fpuParentWeightByVisitedPolicy) {
      double avgWeight = std::min(1.0, pow(policyProbMassVisited, searchParams.fpuParentWeightByVisitedPolicyPow));
      parentUtilityForFPU = avgWeight * parentUtility + (1.0 - avgWeight) * nnUtility;
    }
    else if(searchParams.fpuParentWeight > 0.0) {
      parentUtilityForFPU = searchParams.fpuParentWeight * nnUtility + (1.0 - searchParams.fpuParentWeight) * parentUtility;
    }
  }

  double fpuReductionMax = isRoot ? searchParams.rootFpuReductionMax : searchParams.fpuReductionMax;
  double fpuLossProp = isRoot ? searchParams.rootFpuLossProp : searchParams.fpuLossProp;
  double utilityRadius = searchParams.winLossUtilityFactor;

  double reduction = fpuReductionMax * sqrt(policyProbMassVisited);
  double fpuValue = parentUtilityForFPU - reduction;
  double lossValue = -utilityRadius;
  fpuValue = fpuValue + (lossValue - fpuValue) * fpuLossProp;
  if(fpuValue < -utilityRadius)
    fpuValue = -utilityRadius;

  return fpuValue;
}

void Search::selectBestChildToDescend(
  SearchThread& thread, const SearchNode& node, SearchNodeState nodeState,
  int& numChildrenFound, int& bestChildIdx, int& bestChildMoveAction, bool& countEdgeVisit,
  bool isRoot
) const {
  assert(thread.seat == node.nextSeat);

  double maxSelectionValue = POLICY_ILLEGAL_SELECTION_VALUE;
  bestChildIdx = -1;
  bestChildMoveAction = Q4Board::NULL_ACTION;
  countEdgeVisit = true;

  ConstSearchNodeChildrenReference children = node.getChildren(nodeState);
  int childrenCapacity = children.getCapacity();

  double policyProbMassVisited = 0.0;
  double maxChildWeight = 0.0;
  double totalChildWeight = 0.0;
  const NNOutput* nnOutput = node.getNNOutput();
  assert(nnOutput != NULL);
  const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();

  for(int i = 0; i < childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    int action = childPointer.getActionRelaxed();
    float nnPolicyProb = policyProbs[action];
    if(nnPolicyProb < 0)
      continue;
    policyProbMassVisited += nnPolicyProb;

    int64_t edgeVisits = childPointer.getEdgeVisits();
    double childWeight = child->stats.getChildWeight(edgeVisits);

    totalChildWeight += childWeight;
    if(childWeight > maxChildWeight)
      maxChildWeight = childWeight;
  }

  assert(policyProbMassVisited <= 1.0001);

  // First play urgency
  double parentUtility;
  double parentWeightPerVisit;
  double parentUtilityStdevFactor;
  double fpuValue = getFpuValueForChildrenAssumeVisited(
    node, thread.seat, isRoot, policyProbMassVisited,
    parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
  );

  bool posesWithChildBuf[Q4Board::NUM_ACTIONS] = { };

  double exploreScaling = getExploreScaling(totalChildWeight, parentUtilityStdevFactor);

  // Try all existing children
  numChildrenFound = 0;
  for(int i = 0; i < childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    numChildrenFound++;
    int64_t childEdgeVisits = childPointer.getEdgeVisits();

    int action = childPointer.getActionRelaxed();
    bool isDuringSearch = true;
    double selectionValue = getExploreSelectionValueOfChild(
      node, policyProbs, child,
      action,
      exploreScaling,
      totalChildWeight, childEdgeVisits, fpuValue,
      parentUtility, parentWeightPerVisit,
      isDuringSearch, maxChildWeight,
      countEdgeVisit,
      &thread
    );
    if(selectionValue > maxSelectionValue) {
      maxSelectionValue = selectionValue;
      bestChildIdx = i;
      bestChildMoveAction = action;
    }

    posesWithChildBuf[action] = true;
  }

  // Try the new child with the best policy value
  int bestNewAction = Q4Board::NULL_ACTION;
  float bestNewNNPolicyProb = -1.0f;
  for(int action = 0; action < Q4Board::NUM_ACTIONS; action++) {
    if(posesWithChildBuf[action])
      continue;

    float nnPolicyProb = policyProbs[action];
    if(nnPolicyProb < 0)
      continue;

    if(nnPolicyProb > bestNewNNPolicyProb) {
      bestNewNNPolicyProb = nnPolicyProb;
      bestNewAction = action;
    }
  }

  if(bestNewAction != Q4Board::NULL_ACTION) {
    double selectionValue = getNewExploreSelectionValue(
      node,
      exploreScaling,
      bestNewNNPolicyProb, fpuValue,
      parentWeightPerVisit,
      maxChildWeight,
      countEdgeVisit,
      &thread
    );
    if(selectionValue > maxSelectionValue) {
      maxSelectionValue = selectionValue;
      bestChildIdx = numChildrenFound;
      bestChildMoveAction = bestNewAction;
    }
  }
}

}  // namespace Q4S
