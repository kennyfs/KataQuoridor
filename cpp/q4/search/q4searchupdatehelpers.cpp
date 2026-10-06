#include "q4search.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "q4searchnode.h"
#include "../../search/distributiontable.h"

namespace Q4S {

void Search::addLeafValue(
  SearchNode& node,
  const double value[5],
  double weight,
  bool isTerminal,
  bool assumeNoExistingWeight
) {
  (void)isTerminal;
  int nAlive = 0;
  for(int s = 0; s < 4; s++) {
    if(value[s] > 0.0)
      nAlive++;
  }
  if(nAlive == 0)
    nAlive = 4;

  double utility[4];
  double utilitySq[4];
  for(int s = 0; s < 4; s++) {
    bool isAlive = (value[s] > 0.0);
    utility[s] = computeSeatUtility(s, value, nAlive, isAlive, searchParams.winLossUtilityFactor);
    utilitySq[s] = utility[s] * utility[s];
  }

  double weightSq = weight * weight;

  if(assumeNoExistingWeight) {
    for(int k = 0; k < 5; k++)
      node.stats.valueAvg[k].store(value[k], std::memory_order_release);
    for(int s = 0; s < 4; s++) {
      node.stats.utilityAvg[s].store(utility[s], std::memory_order_release);
      node.stats.utilitySqAvg[s].store(utilitySq[s], std::memory_order_release);
    }
    node.stats.weightSqSum.store(weightSq, std::memory_order_release);
    node.stats.weightSum.store(weight, std::memory_order_release);
    node.stats.visits.fetch_add(1, std::memory_order_release);
  }
  else {
    double oldWeightSum = node.stats.weightSum.load(std::memory_order_relaxed);
    double newWeightSum = oldWeightSum + weight;

    for(int k = 0; k < 5; k++) {
      double oldVal = node.stats.valueAvg[k].load(std::memory_order_relaxed);
      node.stats.valueAvg[k].store((oldVal * oldWeightSum + value[k] * weight) / newWeightSum, std::memory_order_release);
    }
    for(int s = 0; s < 4; s++) {
      double oldU = node.stats.utilityAvg[s].load(std::memory_order_relaxed);
      double oldUSq = node.stats.utilitySqAvg[s].load(std::memory_order_relaxed);
      node.stats.utilityAvg[s].store((oldU * oldWeightSum + utility[s] * weight) / newWeightSum, std::memory_order_release);
      node.stats.utilitySqAvg[s].store((oldUSq * oldWeightSum + utilitySq[s] * weight) / newWeightSum, std::memory_order_release);
    }
    node.stats.weightSqSum.store(node.stats.weightSqSum.load(std::memory_order_relaxed) + weightSq, std::memory_order_release);
    node.stats.weightSum.store(newWeightSum, std::memory_order_release);
    node.stats.visits.fetch_add(1, std::memory_order_release);
  }
}

void Search::addCurrentNNOutputAsLeafValue(SearchNode& node, bool assumeNoExistingWeight) {
  const NNOutput* nnOutput = node.getNNOutput();
  assert(nnOutput != NULL);

  double value[5];
  for(int k = 0; k < 5; k++)
    value[k] = (double)nnOutput->valueAbs[k];

  double weight = computeWeightFromNNOutput(nnOutput);
  addLeafValue(node, value, weight, false, assumeNoExistingWeight);
}

double Search::computeWeightFromNNOutput(const NNOutput* nnOutput) const {
  if(!searchParams.useUncertainty)
    return 1.0;
  if(!nnEvaluator->supportsShorttermError())
    return 1.0;

  double utilityUncertainty = searchParams.winLossUtilityFactor * nnOutput->shorttermWinlossError;

  double poweredUncertainty;
  if(searchParams.uncertaintyExponent == 1.0)
    poweredUncertainty = utilityUncertainty;
  else if(searchParams.uncertaintyExponent == 0.5)
    poweredUncertainty = sqrt(utilityUncertainty);
  else
    poweredUncertainty = pow(utilityUncertainty, searchParams.uncertaintyExponent);

  double baselineUncertainty = searchParams.uncertaintyCoeff / searchParams.uncertaintyMaxWeight;
  double weight = searchParams.uncertaintyCoeff / (poweredUncertainty + baselineUncertainty);
  return weight;
}

void Search::updateStatsAfterPlayout(SearchNode& node, SearchThread& thread, bool isRoot) {
  int32_t oldDirtyCounter = node.dirtyCounter.fetch_add(1, std::memory_order_acq_rel);
  assert(oldDirtyCounter >= 0);
  if(oldDirtyCounter > 0)
    return;
  int32_t numVisitsCompleted = 1;
  while(true) {
    recomputeNodeStats(node, thread, numVisitsCompleted, isRoot);
    oldDirtyCounter = node.dirtyCounter.fetch_add(-numVisitsCompleted, std::memory_order_acq_rel);
    int32_t newDirtyCounter = oldDirtyCounter - numVisitsCompleted;
    if(newDirtyCounter <= 0) {
      assert(newDirtyCounter == 0);
      break;
    }
    numVisitsCompleted = newDirtyCounter;
    continue;
  }
}

void Search::recomputeNodeStats(SearchNode& node, SearchThread& thread, int32_t numVisitsToAdd, bool isRoot) {
  std::vector<MoreNodeStats>& statsBuf = thread.statsBuf;
  int numGoodChildren = 0;

  ConstSearchNodeChildrenReference children = node.getChildren();
  int childrenCapacity = children.getCapacity();
  double origTotalChildWeight = 0.0;
  for(int i = 0; i < childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    MoreNodeStats& stats = statsBuf[numGoodChildren];

    int action = childPointer.getActionRelaxed();
    int64_t edgeVisits = childPointer.getEdgeVisits();
    stats.stats = NodeStats(child->stats);

    if(stats.stats.visits <= 0 || stats.stats.weightSum <= 0.0 || edgeVisits <= 0)
      continue;

    int mover = node.nextSeat;
    stats.selfUtility = stats.stats.utilityAvg[mover];
    stats.weightAdjusted = stats.stats.getChildWeight(edgeVisits);
    stats.prevAction = action;

    origTotalChildWeight += stats.weightAdjusted;
    numGoodChildren++;
  }

  double currentTotalChildWeight = origTotalChildWeight;

  if(searchParams.useNoisePruning && numGoodChildren > 0) {
    double policyProbsBuf[Q4Board::NUM_ACTIONS];
    {
      const NNOutput* nnOutput = node.getNNOutput();
      assert(nnOutput != NULL);
      const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();
      for(int i = 0; i < numGoodChildren; i++)
        policyProbsBuf[i] = std::max(1e-30, (double)policyProbs[statsBuf[i].prevAction]);
    }
    currentTotalChildWeight = pruneNoiseWeight(statsBuf, numGoodChildren, currentTotalChildWeight, policyProbsBuf);
  }

  {
    double amountToSubtract = 0.0;
    double amountToPrune = 0.0;
    if(isRoot && searchParams.rootNoiseEnabled && !searchParams.useNoisePruning) {
      double maxChildWeight = 0.0;
      for(int i = 0; i < numGoodChildren; i++) {
        if(statsBuf[i].weightAdjusted > maxChildWeight)
          maxChildWeight = statsBuf[i].weightAdjusted;
      }
      amountToSubtract = std::min(searchParams.chosenMoveSubtract, maxChildWeight / 64.0);
      amountToPrune = std::min(searchParams.chosenMovePrune, maxChildWeight / 64.0);
    }

    downweightBadChildrenAndNormalizeWeight(
      numGoodChildren, currentTotalChildWeight, currentTotalChildWeight,
      amountToSubtract, amountToPrune, statsBuf
    );
  }

  double valueSum[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
  double utilitySum[4] = {0.0, 0.0, 0.0, 0.0};
  double utilitySqSum[4] = {0.0, 0.0, 0.0, 0.0};
  double weightSqSum = 0.0;
  double weightSum = currentTotalChildWeight;

  for(int i = 0; i < numGoodChildren; i++) {
    const NodeStats& stats = statsBuf[i].stats;

    double desiredWeight = statsBuf[i].weightAdjusted;
    double weightScaling = desiredWeight / stats.weightSum;

    for(int k = 0; k < 5; k++)
      valueSum[k] += desiredWeight * stats.valueAvg[k];
    for(int s = 0; s < 4; s++) {
      utilitySum[s] += desiredWeight * stats.utilityAvg[s];
      utilitySqSum[s] += desiredWeight * stats.utilitySqAvg[s];
    }
    weightSqSum += weightScaling * weightScaling * stats.weightSqSum;
  }

  // Also add in direct evaluation of this node
  {
    const NNOutput* nnOutput = node.getNNOutput();
    assert(nnOutput != NULL);

    double nnValue[5];
    for(int k = 0; k < 5; k++)
      nnValue[k] = (double)nnOutput->valueAbs[k];

    int nAlive = 0;
    for(int s = 0; s < 4; s++) {
      if(nnValue[s] > 0.0)
        nAlive++;
    }
    if(nAlive == 0)
      nAlive = 4;

    double weight = computeWeightFromNNOutput(nnOutput);
    for(int k = 0; k < 5; k++)
      valueSum[k] += weight * nnValue[k];
    for(int s = 0; s < 4; s++) {
      bool isAlive = (nnValue[s] > 0.0);
      double u = computeSeatUtility(s, nnValue, nAlive, isAlive, searchParams.winLossUtilityFactor);
      utilitySum[s] += weight * u;
      utilitySqSum[s] += weight * u * u;
    }
    weightSqSum += weight * weight;
    weightSum += weight;
  }

  double valueAvg[5];
  for(int k = 0; k < 5; k++)
    valueAvg[k] = valueSum[k] / weightSum;

  double utilityAvg[4];
  double utilitySqAvg[4];
  for(int s = 0; s < 4; s++) {
    utilityAvg[s] = utilitySum[s] / weightSum;
    utilitySqAvg[s] = utilitySqSum[s] / weightSum;
  }

  for(int k = 0; k < 5; k++)
    node.stats.valueAvg[k].store(valueAvg[k], std::memory_order_release);
  for(int s = 0; s < 4; s++) {
    node.stats.utilityAvg[s].store(utilityAvg[s], std::memory_order_release);
    node.stats.utilitySqAvg[s].store(utilitySqAvg[s], std::memory_order_release);
  }
  node.stats.weightSqSum.store(weightSqSum, std::memory_order_release);
  node.stats.weightSum.store(weightSum, std::memory_order_release);
  node.stats.visits.fetch_add(numVisitsToAdd, std::memory_order_release);
}

void Search::downweightBadChildrenAndNormalizeWeight(
  int numChildren,
  double currentTotalWeight,
  double desiredTotalWeight,
  double amountToSubtract,
  double amountToPrune,
  std::vector<MoreNodeStats>& statsBuf
) const {
  if(numChildren <= 0 || currentTotalWeight <= 0.0)
    return;

  if(searchParams.valueWeightExponent == 0) {
    for(int i = 0; i < numChildren; i++) {
      if(statsBuf[i].weightAdjusted < amountToPrune) {
        currentTotalWeight -= statsBuf[i].weightAdjusted;
        statsBuf[i].weightAdjusted = 0.0;
        continue;
      }
      double newWeight = statsBuf[i].weightAdjusted - amountToSubtract;
      if(newWeight <= 0) {
        currentTotalWeight -= statsBuf[i].weightAdjusted;
        statsBuf[i].weightAdjusted = 0.0;
      }
      else {
        currentTotalWeight -= amountToSubtract;
        statsBuf[i].weightAdjusted = newWeight;
      }
    }

    if(currentTotalWeight != desiredTotalWeight && currentTotalWeight > 0.0) {
      double factor = desiredTotalWeight / currentTotalWeight;
      for(int i = 0; i < numChildren; i++)
        statsBuf[i].weightAdjusted *= factor;
    }
    return;
  }

  assert(numChildren <= Q4Board::NUM_ACTIONS);
  double stdevs[Q4Board::NUM_ACTIONS];
  double simpleValueSum = 0.0;
  for(int i = 0; i < numChildren; i++) {
    int64_t numVisits = statsBuf[i].stats.visits;
    assert(numVisits >= 0);
    if(numVisits == 0)
      continue;

    double weight = statsBuf[i].weightAdjusted;
    double precision = 1.5 * sqrt(weight);

    static const double minVariance = 0.00000001;
    stdevs[i] = sqrt(minVariance + 1.0 / precision);
    simpleValueSum += statsBuf[i].selfUtility * weight;
  }

  double simpleValue = simpleValueSum / currentTotalWeight;

  double totalNewUnnormWeight = 0.0;
  for(int i = 0; i < numChildren; i++) {
    if(statsBuf[i].stats.visits == 0)
      continue;

    if(statsBuf[i].weightAdjusted < amountToPrune) {
      currentTotalWeight -= statsBuf[i].weightAdjusted;
      statsBuf[i].weightAdjusted = 0.0;
      continue;
    }
    double newWeight = statsBuf[i].weightAdjusted - amountToSubtract;
    if(newWeight <= 0) {
      currentTotalWeight -= statsBuf[i].weightAdjusted;
      statsBuf[i].weightAdjusted = 0.0;
    }
    else {
      currentTotalWeight -= amountToSubtract;
      statsBuf[i].weightAdjusted = newWeight;
    }

    double z = (statsBuf[i].selfUtility - simpleValue) / stdevs[i];
    double p = valueWeightDistribution->getCdf(z) + 0.0001;
    statsBuf[i].weightAdjusted *= pow(p, searchParams.valueWeightExponent);
    totalNewUnnormWeight += statsBuf[i].weightAdjusted;
  }

  assert(totalNewUnnormWeight > 0.0);
  double factor = desiredTotalWeight / totalNewUnnormWeight;
  for(int i = 0; i < numChildren; i++)
    statsBuf[i].weightAdjusted *= factor;
}

double Search::pruneNoiseWeight(
  std::vector<MoreNodeStats>& statsBuf,
  int numChildren,
  double totalChildWeight,
  const double* policyProbsBuf
) const {
  if(numChildren <= 1 || totalChildWeight <= 0.00001)
    return totalChildWeight;

  double utilitySumSoFar = 0;
  double weightSumSoFar = 0;
  double rawPolicySumSoFar = 0;
  for(int i = 0; i < numChildren; i++) {
    double utility = statsBuf[i].selfUtility;
    double oldWeight = statsBuf[i].weightAdjusted;
    double rawPolicy = policyProbsBuf[i];

    double newWeight = oldWeight;
    if(weightSumSoFar > 0 && rawPolicySumSoFar > 0) {
      double avgUtilitySoFar = utilitySumSoFar / weightSumSoFar;
      double utilityGap = avgUtilitySoFar - utility;
      if(utilityGap > 0) {
        double weightShareFromRawPolicy = weightSumSoFar * rawPolicy / rawPolicySumSoFar;
        double lenientWeightShareFromRawPolicy = 2.0 * weightShareFromRawPolicy;
        if(oldWeight > lenientWeightShareFromRawPolicy) {
          double excessWeight = oldWeight - lenientWeightShareFromRawPolicy;
          double weightToSubtract = excessWeight * (1.0 - exp(-utilityGap / searchParams.noisePruneUtilityScale));
          if(weightToSubtract > searchParams.noisePruningCap)
            weightToSubtract = searchParams.noisePruningCap;

          newWeight = oldWeight - weightToSubtract;
          statsBuf[i].weightAdjusted = newWeight;
        }
      }
    }
    utilitySumSoFar += utility * newWeight;
    weightSumSoFar += newWeight;
    rawPolicySumSoFar += rawPolicy;
  }
  return weightSumSoFar;
}

}  // namespace Q4S
