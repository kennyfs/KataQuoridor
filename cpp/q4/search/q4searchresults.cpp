#include "q4search.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <vector>

#include "../../core/test.h"
#include "../q4notation.h"
#include "q4searchnode.h"

namespace Q4S {

int64_t Search::getRootVisits() const {
  if(rootNode == NULL)
    return 0;
  return rootNode->stats.visits.load(std::memory_order_acquire);
}

bool Search::getRootValues(ReportedSearchValues& values) const {
  if(rootNode == NULL)
    return false;
  int64_t visits = rootNode->stats.visits.load(std::memory_order_acquire);
  double weightSum = rootNode->stats.weightSum.load(std::memory_order_acquire);
  if(visits <= 0 || weightSum <= 0.0)
    return false;
  for(int k = 0; k < 5; k++)
    values.value[k] = rootNode->stats.valueAvg[k].load(std::memory_order_acquire);
  for(int s = 0; s < 4; s++)
    values.utility[s] = rootNode->stats.utilityAvg[s].load(std::memory_order_acquire);
  values.weight = weightSum;
  values.visits = visits;
  return true;
}

int Search::getChosenMoveAction() {
  if(rootNode == NULL)
    return Q4Board::NULL_ACTION;

  std::vector<int> actions;
  std::vector<double> playSelectionValues;
  bool suc = getPlaySelectionValues(actions, playSelectionValues, NULL, 0.0);
  if(!suc || actions.empty())
    return Q4Board::NULL_ACTION;

  testAssert(actions.size() == playSelectionValues.size());

  double temperature = interpolateEarly(
    searchParams.chosenMoveTemperatureHalflife, searchParams.chosenMoveTemperatureEarly, searchParams.chosenMoveTemperature
  );

  uint32_t idxChosen = chooseIndexWithTemperature(
    nonSearchRand,
    playSelectionValues.data(),
    (int)playSelectionValues.size(),
    temperature,
    searchParams.chosenMoveTemperatureOnlyBelowProb,
    NULL
  );
  return actions[idxChosen];
}

bool Search::getPlaySelectionValues(
  std::vector<int>& actions,
  std::vector<double>& playSelectionValues,
  std::vector<double>* retVisitCounts,
  double scaleMaxToAtLeast
) const {
  if(rootNode == NULL) {
    actions.clear();
    playSelectionValues.clear();
    if(retVisitCounts != NULL)
      retVisitCounts->clear();
    return false;
  }
  const bool allowDirectPolicyMoves = true;
  double lcbBuf[Q4Board::NUM_ACTIONS];
  double radiusBuf[Q4Board::NUM_ACTIONS];
  return getPlaySelectionValues(
    *rootNode, actions, playSelectionValues, retVisitCounts, scaleMaxToAtLeast,
    allowDirectPolicyMoves, false, false, lcbBuf, radiusBuf
  );
}

bool Search::getPlaySelectionValues(
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
) const {
  actions.clear();
  playSelectionValues.clear();
  if(retVisitCounts != NULL)
    retVisitCounts->clear();

  const NNOutput* nnOutput = node.getNNOutput();
  const float* policyProbs = nnOutput != NULL ? nnOutput->getPolicyProbsMaybeNoised() : NULL;

  double totalChildWeight = 0.0;

  ConstSearchNodeChildrenReference children = node.getChildren();
  const int childrenCapacity = children.getCapacity();
  for(int i = 0; i < childrenCapacity; i++) {
    const SearchChildPointer& childPointer = children[i];
    const SearchNode* child = childPointer.getIfAllocated();
    if(child == NULL)
      break;
    int action = childPointer.getActionRelaxed();
    int64_t edgeVisits = childPointer.getEdgeVisits();
    double childWeight = child->stats.getChildWeight(edgeVisits);

    actions.push_back(action);
    totalChildWeight += childWeight;

    if(policyProbs != NULL && policyProbs[action] < 0) {
      playSelectionValues.push_back(0.0);
      if(retVisitCounts != NULL)
        (*retVisitCounts).push_back(0.0);
    }
    else {
      playSelectionValues.push_back((double)childWeight);
      if(retVisitCounts != NULL)
        (*retVisitCounts).push_back((double)edgeVisits);
    }
  }

  int numChildren = (int)playSelectionValues.size();

  int nonLCBBestIdx = 0;
  double nonLCBBestChildWeight = -1e30;
  {
    double maxGoodness = -1e30;
    for(int i = 0; i < numChildren; i++) {
      double weight = playSelectionValues[i];
      const SearchChildPointer& childPointer = children[i];
      double edgeVisits = childPointer.getEdgeVisits();
      int action = childPointer.getActionRelaxed();
      double policyProb = (policyProbs != NULL) ? policyProbs[action] : 0.0;

      double g = weight * std::max(0.0, edgeVisits - 1.0) / std::max(1.0, edgeVisits) + 2.0 * policyProb;
      if(g > maxGoodness) {
        maxGoodness = g;
        nonLCBBestChildWeight = weight;
        nonLCBBestIdx = i;
      }
    }
  }

  // Possibly reduce weight on children that we spent too many visits on in retrospect
  if(&node == rootNode && numChildren > 0 && policyProbs != NULL) {
    const SearchChildPointer& bestChildPointer = children[nonLCBBestIdx];
    const SearchNode* bestChild = bestChildPointer.getIfAllocated();
    int64_t bestChildEdgeVisits = bestChildPointer.getEdgeVisits();
    int bestAction = bestChildPointer.getActionRelaxed();
    testAssert(bestChild != NULL);
    const bool isRoot = true;
    const double policyProbMassVisited = 1.0;
    double parentUtility;
    double parentWeightPerVisit;
    double parentUtilityStdevFactor;
    int mover = rootState.board.toMove;
    double fpuValue = getFpuValueForChildrenAssumeVisited(
      node, mover, isRoot, policyProbMassVisited,
      parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
    );

    bool isDuringSearch = false;
    double exploreScaling = getExploreScaling(totalChildWeight, parentUtilityStdevFactor);
    const bool countEdgeVisit = true;
    double bestChildExploreSelectionValue = getExploreSelectionValueOfChild(
      node, policyProbs, bestChild,
      bestAction,
      exploreScaling,
      totalChildWeight, bestChildEdgeVisits, fpuValue,
      parentUtility, parentWeightPerVisit,
      isDuringSearch, nonLCBBestChildWeight,
      countEdgeVisit,
      NULL
    );

    for(int i = 0; i < numChildren; i++) {
      if(i != nonLCBBestIdx) {
        const SearchChildPointer& childPointer = children[i];
        const SearchNode* child = childPointer.getIfAllocated();
        int action = childPointer.getActionRelaxed();
        int64_t edgeVisits = childPointer.getEdgeVisits();
        double reduced = getReducedPlaySelectionWeight(
          node, policyProbs, child,
          action,
          exploreScaling,
          edgeVisits,
          bestChildExploreSelectionValue
        );
        playSelectionValues[i] = ceil(reduced);
      }
    }
  }

  // Now compute play selection values taking into account LCB
  if(lcbBuf != nullptr && radiusBuf != nullptr) {
    if(!neverUseLcb && (alwaysComputeLcb || (searchParams.useLcbForSelection && numChildren > 0))) {
      double bestLcb = -1e10;
      int bestLcbIndex = -1;
      for(int i = 0; i < numChildren; i++) {
        const SearchChildPointer& childPointer = children[i];
        const SearchNode* child = childPointer.getIfAllocated();
        int64_t edgeVisits = childPointer.getEdgeVisits();
        int action = childPointer.getActionRelaxed();
        getSelfUtilityLCBAndRadius(node, child, edgeVisits, action, lcbBuf[i], radiusBuf[i]);

        double weight = playSelectionValues[i];
        if(weight > 0 && weight >= searchParams.minVisitPropForLCB * nonLCBBestChildWeight) {
          if(lcbBuf[i] > bestLcb) {
            bestLcb = lcbBuf[i];
            bestLcbIndex = i;
          }
        }
      }

      if(searchParams.useLcbForSelection && numChildren > 0 && bestLcbIndex >= 0) {
        double adjustedWeight = playSelectionValues[bestLcbIndex];
        for(int i = 0; i < numChildren; i++) {
          if(i != bestLcbIndex) {
            double excessValue = bestLcb - lcbBuf[i];
            if(excessValue < 0)
              continue;

            double radius = radiusBuf[i];
            double radiusFactor = (radius + excessValue) / (radius + 0.20 * excessValue);
            double lbound = radiusFactor * radiusFactor * playSelectionValues[i];
            if(lbound > adjustedWeight)
              adjustedWeight = lbound;
          }
        }
        playSelectionValues[bestLcbIndex] = adjustedWeight;
      }
    }
  }

  // Fallback to direct legal policy moves if no children explored
  if(numChildren == 0 && allowDirectPolicyMoves && &node == rootNode && nnOutput != NULL) {
    std::vector<int> legalActions;
    rootState.getLegalActions(legalActions);
    for(int action : legalActions) {
      double prob = (policyProbs != NULL && policyProbs[action] >= 0) ? (double)policyProbs[action] : 0.0;
      actions.push_back(action);
      playSelectionValues.push_back(prob);
      if(retVisitCounts != NULL)
        (*retVisitCounts).push_back(0.0);
      numChildren++;
    }
  }

  if(numChildren == 0)
    return false;

  double maxValue = 0.0;
  for(int i = 0; i < numChildren; i++) {
    if(playSelectionValues[i] > maxValue)
      maxValue = playSelectionValues[i];
  }

  if(maxValue <= 1e-50 && policyProbs != NULL) {
    for(int i = 0; i < numChildren; i++)
      playSelectionValues[i] = std::max(0.0, (double)policyProbs[actions[i]]);
    for(int i = 0; i < numChildren; i++) {
      if(playSelectionValues[i] > maxValue)
        maxValue = playSelectionValues[i];
    }
    if(maxValue <= 1e-50)
      return false;
  }

  testAssert(maxValue < 1e40);

  double amountToSubtract = std::min(searchParams.chosenMoveSubtract, maxValue / 64.0);
  double amountToPrune = std::min(searchParams.chosenMovePrune, maxValue / 64.0);
  for(int i = 0; i < numChildren; i++) {
    if(playSelectionValues[i] < amountToPrune)
      playSelectionValues[i] = 0.0;
    else {
      playSelectionValues[i] -= amountToSubtract;
      if(playSelectionValues[i] <= 0.0)
        playSelectionValues[i] = 0.0;
    }
  }

  if(scaleMaxToAtLeast > 0.0 && maxValue < scaleMaxToAtLeast && maxValue > 0.0) {
    double factor = scaleMaxToAtLeast / maxValue;
    for(int i = 0; i < numChildren; i++)
      playSelectionValues[i] *= factor;
  }

  return true;
}

AnalysisData Search::getAnalysisDataOfSingleChild(
  const SearchNode* child,
  int64_t edgeVisits,
  std::vector<int>& scratchActions,
  std::vector<double>& scratchValues,
  int action,
  double policyProb,
  double fpuValue,
  double parentUtility,
  int maxPVDepth
) const {
  (void)scratchActions;
  (void)scratchValues;
  (void)parentUtility;

  AnalysisData data;
  data.move = action;
  data.numVisits = (int)edgeVisits;
  data.policyPrior = policyProb;
  data.order = 0;

  int64_t childVisits = 0;
  if(child != NULL) {
    childVisits = child->stats.visits.load(std::memory_order_acquire);
    for(int k = 0; k < 5; k++)
      data.valueAvg[k] = child->stats.valueAvg[k].load(std::memory_order_acquire);
    for(int s = 0; s < 4; s++)
      data.utilityAvg[s] = child->stats.utilityAvg[s].load(std::memory_order_acquire);
  }
  else {
    for(int k = 0; k < 5; k++) data.valueAvg[k] = 0.0;
    for(int s = 0; s < 4; s++) data.utilityAvg[s] = 0.0;
  }

  int mover = rootState.board.toMove;
  if(childVisits <= 0) {
    data.utility = fpuValue;
  }
  else {
    data.utility = data.utilityAvg[mover];
  }

  data.pv.clear();
  data.pv.push_back(action);
  data.pvVisits.clear();
  data.pvVisits.push_back(childVisits);
  data.pvEdgeVisits.clear();
  data.pvEdgeVisits.push_back(edgeVisits);

  if(child != NULL && maxPVDepth > 1) {
    const SearchNode* curr = child;
    for(int depth = 1; depth < maxPVDepth; depth++) {
      ConstSearchNodeChildrenReference currChildren = curr->getChildren();
      int currCap = currChildren.getCapacity();
      const SearchNode* bestNextChild = NULL;
      int bestNextAction = Q4Board::NULL_ACTION;
      int64_t bestNextVisits = -1;
      int64_t bestNextEdgeVisits = -1;
      for(int j = 0; j < currCap; j++) {
        const SearchChildPointer& cp = currChildren[j];
        const SearchNode* c = cp.getIfAllocated();
        if(c == NULL) break;
        int64_t v = c->stats.visits.load(std::memory_order_acquire);
        if(v > bestNextVisits) {
          bestNextVisits = v;
          bestNextEdgeVisits = cp.getEdgeVisits();
          bestNextAction = cp.getActionRelaxed();
          bestNextChild = c;
        }
      }
      if(bestNextChild == NULL || bestNextVisits <= 0 || bestNextAction == Q4Board::NULL_ACTION)
        break;
      data.pv.push_back(bestNextAction);
      data.pvVisits.push_back(bestNextVisits);
      data.pvEdgeVisits.push_back(bestNextEdgeVisits);
      curr = bestNextChild;
    }
  }

  return data;
}

void Search::getAnalysisData(std::vector<AnalysisData>& buf) const {
  buf.clear();
  if(rootNode == NULL)
    return;

  ConstSearchNodeChildrenReference childrenArr = rootNode->getChildren();

  std::vector<int> actions;
  std::vector<double> playSelectionValues;
  std::vector<double> visitCounts;
  double lcbBuf[Q4Board::NUM_ACTIONS];
  double radiusBuf[Q4Board::NUM_ACTIONS];

  const bool allowDirectPolicyMoves = false;
  const bool alwaysComputeLcb = true;
  getPlaySelectionValues(*rootNode, actions, playSelectionValues, &visitCounts, 1.0, allowDirectPolicyMoves, alwaysComputeLcb, false, lcbBuf, radiusBuf);

  const NNOutput* nnOutput = rootNode->getNNOutput();
  const float* policyProbs = nnOutput != NULL ? nnOutput->getPolicyProbsMaybeNoised() : NULL;

  int mover = rootState.board.toMove;
  double parentUtility = rootNode->stats.utilityAvg[mover].load(std::memory_order_acquire);
  double parentWeightPerVisit = 1.0;
  double parentUtilityStdevFactor = 1.0;
  double fpuValue = getFpuValueForChildrenAssumeVisited(
    *rootNode, mover, true, 1.0, parentUtility, parentWeightPerVisit, parentUtilityStdevFactor
  );

  std::vector<int> scratchActions;
  std::vector<double> scratchValues;

  for(size_t i = 0; i < actions.size(); i++) {
    int action = actions[i];
    const SearchChildPointer& cp = childrenArr[(int)i];
    const SearchNode* child = cp.getIfAllocated();
    int64_t edgeVisits = cp.getEdgeVisits();
    float policyProb = policyProbs != NULL ? policyProbs[action] : 0.0f;

    AnalysisData data = getAnalysisDataOfSingleChild(
      child, edgeVisits, scratchActions, scratchValues, action, (double)policyProb, fpuValue, parentUtility, 10
    );
    data.lcb = lcbBuf[i];
    data.radius = radiusBuf[i];
    buf.push_back(data);
  }

  std::sort(buf.begin(), buf.end(), [](const AnalysisData& a, const AnalysisData& b) {
    if(a.numVisits != b.numVisits)
      return a.numVisits > b.numVisits;
    return a.policyPrior > b.policyPrior;
  });

  for(size_t i = 0; i < buf.size(); i++)
    buf[i].order = (int)i;
}

void Search::getPolicySurpriseAndEntropy(double& policySurprise, double& policyEntropy) const {
  policySurprise = 0.0;
  policyEntropy = 0.0;
  if(rootNode == NULL)
    return;

  const NNOutput* nnOutput = rootNode->getNNOutput();
  if(nnOutput == NULL)
    return;

  std::vector<int> actions;
  std::vector<double> playSelectionValues;
  std::vector<double> visitCounts;
  double lcbBuf[Q4Board::NUM_ACTIONS];
  double radiusBuf[Q4Board::NUM_ACTIONS];
  getPlaySelectionValues(*rootNode, actions, playSelectionValues, &visitCounts, 1.0, false, false, true, lcbBuf, radiusBuf);

  double visitSum = 0.0;
  for(double v : visitCounts)
    visitSum += v;

  if(visitSum <= 0.0)
    return;

  const float* policyProbs = nnOutput->getPolicyProbsMaybeNoised();

  double kl = 0.0;
  double ent = 0.0;
  for(size_t i = 0; i < actions.size(); i++) {
    double p = visitCounts[i] / visitSum;
    if(p > 1e-30) {
      ent -= p * std::log(p);
      int a = actions[i];
      double q = (policyProbs != NULL && policyProbs[a] > 1e-30f) ? (double)policyProbs[a] : 1e-30;
      kl += p * std::log(p / q);
    }
  }
  policyEntropy = ent;
  policySurprise = std::max(0.0, kl);
}

void Search::printPV(std::ostream& out, const std::vector<int>& buf) const {
  for(size_t i = 0; i < buf.size(); i++) {
    if(i > 0) out << " ";
    out << Q4Notation::actionToString(buf[i]);
  }
}

void Search::debugPrintChildrenSummary(std::ostream& out, const SearchNode& node, const NNOutput* nnOutput) const {
  ConstSearchNodeChildrenReference children = node.getChildren();
  int cap = children.getCapacity();
  const float* probs = nnOutput != nullptr ? nnOutput->getPolicyProbsMaybeNoised() : nullptr;

  out << "--- Children summary ---" << "\n";
  for(int i = 0; i < cap; i++) {
    const SearchChildPointer& cp = children[i];
    const SearchNode* child = cp.getIfAllocated();
    if(child == nullptr) break;
    int action = cp.getActionRelaxed();
    int64_t v = child->stats.visits.load(std::memory_order_acquire);
    float prior = (probs != nullptr && action >= 0) ? probs[action] : -1.0f;
    out << "Action " << Q4Notation::actionToString(action)
        << " visits " << v
        << " prior " << std::fixed << std::setprecision(4) << prior;
    out << " values [";
    for(int k = 0; k < 5; k++) {
      if(k > 0) out << ", ";
      out << std::setprecision(3) << child->stats.valueAvg[k].load(std::memory_order_acquire);
    }
    out << "]\n";
  }
}

}  // namespace Q4S
