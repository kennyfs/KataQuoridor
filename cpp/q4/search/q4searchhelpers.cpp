#include "q4search.h"

#include <algorithm>
#include <cmath>

#include "../../core/fancymath.h"
#include "../../core/test.h"
#include "q4searchnode.h"

namespace Q4S {

uint32_t Search::chooseIndexWithTemperature(
  Rand& rand,
  const double* relativeProbs,
  int numRelativeProbs,
  double temperature,
  double onlyBelowProb,
  double* processedRelProbsBuf
) {
  testAssert(numRelativeProbs > 0);
  testAssert(numRelativeProbs <= Q4Board::NUM_ACTIONS);
  double processedRelProbs[Q4Board::NUM_ACTIONS];
  if(processedRelProbsBuf == NULL)
    processedRelProbsBuf = &processedRelProbs[0];

  double maxRelProb = 0.0;
  double sumRelProb = 0.0;
  for(int i = 0; i < numRelativeProbs; i++) {
    sumRelProb += std::max(0.0, relativeProbs[i]);
    if(relativeProbs[i] > maxRelProb)
      maxRelProb = relativeProbs[i];
  }
  testAssert(maxRelProb > 0.0);
  testAssert(sumRelProb > 0.0);

  // Temperature so close to 0 that we just calculate the max directly
  if(temperature <= 1.0e-4 && onlyBelowProb >= 1.0) {
    double bestProb = relativeProbs[0];
    int bestIdx = 0;
    processedRelProbsBuf[0] = 0;
    for(int i = 1; i < numRelativeProbs; i++) {
      processedRelProbsBuf[i] = 0;
      if(relativeProbs[i] > bestProb) {
        bestProb = relativeProbs[i];
        bestIdx = i;
      }
    }
    processedRelProbsBuf[bestIdx] = 1.0;
    return bestIdx;
  }
  // Actual temperature
  else {
    double logMaxRelProb = log(maxRelProb);
    double logSumRelProb = log(sumRelProb);
    double logOnlyBelowProb = log(std::max(1e-50, onlyBelowProb));
    double sum = 0.0;
    for(int i = 0; i < numRelativeProbs; i++) {
      if(relativeProbs[i] <= 0.0)
        processedRelProbsBuf[i] = 0.0;
      else {
        double logRelProb = log(relativeProbs[i]) - logMaxRelProb;
        double logRelProbThreshold = std::min(0.0, logOnlyBelowProb + logSumRelProb - logMaxRelProb);
        double newLogRelProb;
        if(logRelProb > logRelProbThreshold)
          newLogRelProb = logRelProb;
        else
          newLogRelProb = (logRelProb - logRelProbThreshold) / temperature + logRelProbThreshold;
        processedRelProbsBuf[i] = exp(newLogRelProb);
      }
      sum += processedRelProbsBuf[i];
    }
    testAssert(sum > 0.0);
    uint32_t idxChosen = rand.nextUInt(processedRelProbsBuf, numRelativeProbs);
    return idxChosen;
  }
}

void Search::computeDirichletAlphaDistribution(int policySize, const float* policyProbs, double* alphaDistr) {
  int legalCount = 0;
  for(int i = 0; i < policySize; i++) {
    if(policyProbs[i] >= 0)
      legalCount += 1;
  }

  if(legalCount <= 0)
    throw StringError("computeDirichletAlphaDistribution: No move with nonnegative policy value");

  // We're going to generate a gamma draw on each move with alphas that sum up to searchParams.rootDirichletNoiseTotalConcentration.
  // Half of the alpha weight are uniform.
  // The other half are shaped based on the log of the existing policy.
  double logPolicySum = 0.0;
  for(int i = 0; i < policySize; i++) {
    if(policyProbs[i] >= 0) {
      alphaDistr[i] = log(std::min(0.01, (double)policyProbs[i]) + 1e-20);
      logPolicySum += alphaDistr[i];
    }
  }
  double logPolicyMean = logPolicySum / legalCount;
  double alphaPropSum = 0.0;
  for(int i = 0; i < policySize; i++) {
    if(policyProbs[i] >= 0) {
      alphaDistr[i] = std::max(0.0, alphaDistr[i] - logPolicyMean);
      alphaPropSum += alphaDistr[i];
    }
  }
  double uniformProb = 1.0 / legalCount;
  if(alphaPropSum <= 0.0) {
    for(int i = 0; i < policySize; i++) {
      if(policyProbs[i] >= 0)
        alphaDistr[i] = uniformProb;
    }
  }
  else {
    for(int i = 0; i < policySize; i++) {
      if(policyProbs[i] >= 0)
        alphaDistr[i] = 0.5 * (alphaDistr[i] / alphaPropSum + uniformProb);
    }
  }
}

void Search::addDirichletNoise(const SearchParams& searchParams, Rand& rand, int policySize, float* policyProbs) {
  double r[Q4Board::NUM_ACTIONS];
  Search::computeDirichletAlphaDistribution(policySize, policyProbs, r);

  // r now contains the proportions with which we would like to split the alpha
  // The total of the alphas is searchParams.rootDirichletNoiseTotalConcentration
  // Generate gamma draw on each move
  double rSum = 0.0;
  for(int i = 0; i < policySize; i++) {
    if(policyProbs[i] >= 0) {
      r[i] = rand.nextGamma(r[i] * searchParams.rootDirichletNoiseTotalConcentration);
      rSum += r[i];
    }
    else
      r[i] = 0.0;
  }

  // Normalized gamma draws -> dirichlet noise
  for(int i = 0; i < policySize; i++)
    r[i] /= rSum;

  // At this point, r[i] contains a dirichlet distribution draw, so add it into the nnOutput.
  for(int i = 0; i < policySize; i++) {
    if(policyProbs[i] >= 0) {
      double weight = searchParams.rootDirichletNoiseWeight;
      policyProbs[i] = (float)(r[i] * weight + policyProbs[i] * (1.0 - weight));
    }
  }
}

std::shared_ptr<NNOutput>* Search::maybeAddPolicyNoiseAndTemp(SearchThread& thread, bool isRoot, const NNOutput* oldNNOutput) const {
  if(!isRoot)
    return NULL;
  if(!searchParams.rootNoiseEnabled &&
     searchParams.rootPolicyTemperature == 1.0 &&
     searchParams.rootPolicyTemperatureEarly == 1.0 &&
     rootHintAction == Q4Board::NULL_ACTION
  )
    return NULL;
  if(oldNNOutput == NULL)
    return NULL;
  if(oldNNOutput->noisedPolicyProbs != NULL)
    return NULL;

  const int policySize = Q4Board::NUM_ACTIONS;

  // Copy nnOutput as we're about to modify its policy to add noise or temperature
  std::shared_ptr<NNOutput>* newNNOutputSharedPtr = new std::shared_ptr<NNOutput>(new NNOutput(*oldNNOutput));
  NNOutput* newNNOutput = newNNOutputSharedPtr->get();

  float* noisedPolicyProbs = new float[policySize];
  newNNOutput->noisedPolicyProbs = noisedPolicyProbs;
  std::copy(newNNOutput->policyProbs, newNNOutput->policyProbs + policySize, noisedPolicyProbs);

  if(searchParams.rootPolicyTemperature != 1.0 || searchParams.rootPolicyTemperatureEarly != 1.0) {
    double rootPolicyTemperature = interpolateEarly(
      searchParams.chosenMoveTemperatureHalflife, searchParams.rootPolicyTemperatureEarly, searchParams.rootPolicyTemperature
    );

    double maxValue = 0.0;
    for(int i = 0; i < policySize; i++) {
      double prob = noisedPolicyProbs[i];
      if(prob > maxValue)
        maxValue = prob;
    }
    testAssert(maxValue > 0.0);

    double logMaxValue = log(maxValue);
    double invTemp = 1.0 / rootPolicyTemperature;
    double sum = 0.0;

    for(int i = 0; i < policySize; i++) {
      if(noisedPolicyProbs[i] > 0) {
        // Numerically stable way to raise to power and normalize
        float p = (float)exp((log((double)noisedPolicyProbs[i]) - logMaxValue) * invTemp);
        noisedPolicyProbs[i] = p;
        sum += p;
      }
    }
    testAssert(sum > 0.0);
    for(int i = 0; i < policySize; i++) {
      if(noisedPolicyProbs[i] >= 0) {
        noisedPolicyProbs[i] = (float)(noisedPolicyProbs[i] / sum);
      }
    }
  }

  if(searchParams.rootNoiseEnabled) {
    addDirichletNoise(searchParams, thread.rand, policySize, noisedPolicyProbs);
  }

  // Move a small amount of policy to the hint action, around the same level that noising it would achieve
  if(rootHintAction != Q4Board::NULL_ACTION && rootHintAction >= 0 && rootHintAction < policySize) {
    const float propToMove = 0.02f;
    int pos = rootHintAction;
    if(noisedPolicyProbs[pos] >= 0) {
      double amountToMove = 0.0;
      for(int i = 0; i < policySize; i++) {
        if(noisedPolicyProbs[i] >= 0) {
          amountToMove += noisedPolicyProbs[i] * propToMove;
          noisedPolicyProbs[i] *= (1.0f - propToMove);
        }
      }
      noisedPolicyProbs[pos] += (float)amountToMove;
    }
  }

  return newNNOutputSharedPtr;
}

double Search::interpolateEarly(double halflife, double earlyValue, double value) const {
  double rawHalflives = (double)rootState.plies / halflife;
  return value + (earlyValue - value) * pow(0.5, rawHalflives);
}

void Search::getSelfUtilityLCBAndRadiusZeroVisits(double& lcbBuf, double& radiusBuf) const {
  // Max radius of the entire utility range
  double utilityRangeRadius = searchParams.winLossUtilityFactor;
  radiusBuf = 2.0 * utilityRangeRadius * searchParams.lcbStdevs;
  lcbBuf = -radiusBuf;
}

void Search::getSelfUtilityLCBAndRadius(
  const SearchNode& parent, const SearchNode* child, int64_t edgeVisits, int action,
  double& lcbBuf, double& radiusBuf
) const {
  (void)action;
  int mover = parent.nextSeat;
  int64_t childVisits = child->stats.visits.load(std::memory_order_acquire);
  double utilityAvg = child->stats.utilityAvg[mover].load(std::memory_order_acquire);
  double utilitySqAvg = child->stats.utilitySqAvg[mover].load(std::memory_order_acquire);
  double weightSum = child->stats.getChildWeight(edgeVisits, childVisits);
  double weightSqSum = child->stats.getChildWeightSq(edgeVisits, childVisits);

  // Max radius of the entire utility range
  double utilityRangeRadius = searchParams.winLossUtilityFactor;
  radiusBuf = 2.0 * utilityRangeRadius * searchParams.lcbStdevs;
  lcbBuf = -radiusBuf;
  if(childVisits <= 0 || weightSum <= 0.0 || weightSqSum <= 0.0)
    return;

  // Effective sample size for weighted data
  double ess = weightSum * weightSum / weightSqSum;

  double priorWeight = weightSum / (ess * ess * ess);
  utilitySqAvg = std::max(utilitySqAvg, utilityAvg * utilityAvg + 1e-8);
  utilitySqAvg = (utilitySqAvg * weightSum + (utilitySqAvg + utilityRangeRadius * utilityRangeRadius) * priorWeight) / (weightSum + priorWeight);
  weightSum += priorWeight;
  weightSqSum += priorWeight * priorWeight;

  // Recompute effective sample size now that we have the prior
  ess = weightSum * weightSum / weightSqSum;

  double selfUtility = utilityAvg;
  double utilityVariance = utilitySqAvg - utilityAvg * utilityAvg;
  double estimateStdev = sqrt(std::max(0.0, utilityVariance) / ess);
  double radius = estimateStdev * searchParams.lcbStdevs;

  lcbBuf = selfUtility - radius;
  radiusBuf = radius;
}

}  // namespace Q4S
