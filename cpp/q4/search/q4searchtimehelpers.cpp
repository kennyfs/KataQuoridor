#include "q4search.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "q4searchnode.h"

namespace Q4S {

static double applyLagBuffer(double time, double lagBuffer) {
  if(time < 0)
    return time;
  else if(time < 2.0 * lagBuffer)
    return time * 0.5;
  else
    return time - lagBuffer;
}

static void getQ4Time(
  const TimeControls& tc,
  int numPlies,
  double lagBuffer,
  double& minTime,
  double& recommendedTime,
  double& maxTime
) {
  int boardArea = 81;
  int numStonesOnBoard = numPlies;

  double approxTurnsLeftAbsolute;
  double approxTurnsLeftIncrement;
  double approxTurnsLeftByoYomi;
  {
    double typicalGameLengthToAllowForAbsolute = 0.95 * boardArea + 20.0;
    double typicalGameLengthToAllowForIncrement = 0.75 * boardArea + 15.0;
    double typicalGameLengthToAllowForByoYomi = 0.50 * boardArea + 10.0;

    double minApproxTurnsLeftAbsolute = 0.15 * boardArea + 30.0;
    double minApproxTurnsLeftIncrement = 0.10 * boardArea + 20.0;
    double minApproxTurnsLeftByoYomi = 0.02 * boardArea + 4.0;

    approxTurnsLeftAbsolute = std::max(typicalGameLengthToAllowForAbsolute - numStonesOnBoard, minApproxTurnsLeftAbsolute);
    approxTurnsLeftIncrement = std::max(typicalGameLengthToAllowForIncrement - numStonesOnBoard, minApproxTurnsLeftIncrement);
    approxTurnsLeftByoYomi = std::max(typicalGameLengthToAllowForByoYomi - numStonesOnBoard, minApproxTurnsLeftByoYomi);

    // Multiply by 0.25 since there are 4 players in Q4
    approxTurnsLeftAbsolute *= 0.25;
    approxTurnsLeftIncrement *= 0.25;
    approxTurnsLeftByoYomi *= 0.25;
  }

  auto divideTimeEvenlyForGame = [approxTurnsLeftAbsolute, approxTurnsLeftIncrement, approxTurnsLeftByoYomi, &tc](double time, bool isIncrementOrAbs, bool isByoYomi) {
    double mainTimeToUseIfAbsolute = time / approxTurnsLeftAbsolute;

    if(isIncrementOrAbs) {
      double mainTimeToUse;
      if(time <= 0)
        mainTimeToUse = time;
      else {
        mainTimeToUse = time / approxTurnsLeftIncrement;
        mainTimeToUse = std::min(mainTimeToUse, mainTimeToUseIfAbsolute + 2.0 * tc.increment);
      }
      return mainTimeToUse;
    }
    else if(isByoYomi) {
      double mainTimeToUse;
      if(tc.perPeriodTime <= 0 || tc.numStonesPerPeriod <= 0)
        mainTimeToUse = mainTimeToUseIfAbsolute;
      else {
        double byoYomiTimePerMove = tc.perPeriodTime / tc.numStonesPerPeriod;
        double theoreticalOptimalTurnsToSpendOurTime = (time / byoYomiTimePerMove) * exp(-1.0);
        double approxTurnsLeftToUse = theoreticalOptimalTurnsToSpendOurTime;

        if(approxTurnsLeftByoYomi > theoreticalOptimalTurnsToSpendOurTime)
          approxTurnsLeftToUse = std::min(approxTurnsLeftByoYomi, theoreticalOptimalTurnsToSpendOurTime * 1.75);

        if(approxTurnsLeftToUse > approxTurnsLeftAbsolute)
          approxTurnsLeftToUse = approxTurnsLeftAbsolute;
        if(approxTurnsLeftToUse < 1)
          approxTurnsLeftToUse = 1;

        mainTimeToUse = time / approxTurnsLeftToUse;
        mainTimeToUse = std::min(mainTimeToUse, mainTimeToUseIfAbsolute + 3.0 * byoYomiTimePerMove);
        if(mainTimeToUse < byoYomiTimePerMove)
          mainTimeToUse = byoYomiTimePerMove;

        if(mainTimeToUse < byoYomiTimePerMove * 1.5 && time < byoYomiTimePerMove * 1.5)
          mainTimeToUse = time + byoYomiTimePerMove;
      }
      return mainTimeToUse;
    }

    return mainTimeToUseIfAbsolute;
  };

  minTime = 0.0;
  recommendedTime = 0.0;
  maxTime = 0.0;

  double lagBufferToUse = lagBuffer;

  if(tc.increment > 0 || tc.numPeriodsLeftIncludingCurrent <= 0) {
    if(tc.mainTimeLeft <= tc.increment) {
      minTime = std::min(std::max(0.0, tc.mainTimeLeft * 0.5), std::max(0.0, tc.mainTimeLeft + tc.increment - tc.mainTimeLimit));
      recommendedTime = applyLagBuffer(tc.mainTimeLeft, lagBufferToUse);
      maxTime = tc.mainTimeLeft;
    }
    else {
      double excessMainTime = applyLagBuffer(tc.mainTimeLeft - tc.increment, lagBufferToUse);
      minTime = std::min(std::max(0.0, tc.mainTimeLeft * 0.5), std::max(0.0, tc.mainTimeLeft + tc.increment - tc.mainTimeLimit));
      recommendedTime = tc.increment + divideTimeEvenlyForGame(excessMainTime, true, false);
      maxTime = std::min(tc.mainTimeLeft, tc.increment + excessMainTime / 5.0);
    }
  }
  else {
    double effectiveMainTimeLeft = tc.mainTimeLeft;
    bool effectivelyInOvertime = tc.inOvertime;
    int effectiveNumPeriodsLeftIncludingCurrent = tc.numPeriodsLeftIncludingCurrent;
    double effectiveTimeLeftInPeriod = tc.timeLeftInPeriod;
    int effectiveNumStonesLeftInPeriod = tc.numStonesPerPeriod;

    if(effectiveMainTimeLeft < 0 && !effectivelyInOvertime) {
      effectivelyInOvertime = true;
      effectiveTimeLeftInPeriod = effectiveMainTimeLeft + tc.perPeriodTime;
      effectiveNumStonesLeftInPeriod = tc.numStonesPerPeriod;
    }
    if(effectivelyInOvertime) {
      while(effectiveTimeLeftInPeriod < 0 && effectiveNumPeriodsLeftIncludingCurrent > 1) {
        effectiveNumPeriodsLeftIncludingCurrent -= 1;
        effectiveTimeLeftInPeriod += tc.perPeriodTime;
      }
    }

    constexpr int NUM_RESERVED_PERIODS = 5;
    if(effectiveNumPeriodsLeftIncludingCurrent > NUM_RESERVED_PERIODS) {
      effectivelyInOvertime = false;
      if(!tc.inOvertime) {
        effectiveMainTimeLeft += tc.perPeriodTime * (effectiveNumPeriodsLeftIncludingCurrent - NUM_RESERVED_PERIODS);
      }
      else {
        effectiveMainTimeLeft += effectiveTimeLeftInPeriod + tc.perPeriodTime * (effectiveNumPeriodsLeftIncludingCurrent - NUM_RESERVED_PERIODS - 1);
      }
    }

    if(!effectivelyInOvertime) {
      double largeByoYomiTimePerMove = tc.perPeriodTime / (0.75 * tc.numStonesPerPeriod + 0.25);
      minTime = 0.0;
      recommendedTime = divideTimeEvenlyForGame(effectiveMainTimeLeft, false, true);
      maxTime = largeByoYomiTimePerMove + std::max(std::min(largeByoYomiTimePerMove * 1.75, effectiveMainTimeLeft), effectiveMainTimeLeft / 5.0);
      if(maxTime > effectiveMainTimeLeft && maxTime < effectiveMainTimeLeft + largeByoYomiTimePerMove)
        maxTime = effectiveMainTimeLeft + largeByoYomiTimePerMove;
      if(maxTime > effectiveMainTimeLeft && effectiveNumPeriodsLeftIncludingCurrent <= 1 && tc.numStonesPerPeriod <= 1)
        lagBufferToUse *= 2.0;
    }
    else {
      minTime = (effectiveNumStonesLeftInPeriod <= 1) ? effectiveTimeLeftInPeriod : 0.0;
      recommendedTime = effectiveTimeLeftInPeriod / effectiveNumStonesLeftInPeriod;
      maxTime = effectiveTimeLeftInPeriod / (0.75 * effectiveNumStonesLeftInPeriod + 0.25);
      if(effectiveNumPeriodsLeftIncludingCurrent <= 1 && effectiveNumStonesLeftInPeriod <= 1)
        lagBufferToUse *= 2.0;
    }
  }

  maxTime = std::min(maxTime, tc.maxTimePerMove);
  minTime = applyLagBuffer(minTime, lagBufferToUse);
  recommendedTime = applyLagBuffer(recommendedTime, lagBufferToUse);
  maxTime = applyLagBuffer(maxTime, lagBufferToUse);

  if(maxTime < 0) maxTime = 0;
  if(minTime < 0) minTime = 0;
  if(recommendedTime < 0) recommendedTime = 0;
  if(minTime > maxTime) minTime = maxTime;
  if(recommendedTime > maxTime) recommendedTime = maxTime;
}

double Search::numVisitsNeededToBeNonFutile(double maxVisitsMoveVisits) {
  double requiredVisits = searchParams.futileVisitsThreshold * maxVisitsMoveVisits;
  double chosenMoveTemperature = interpolateEarly(
    searchParams.chosenMoveTemperatureHalflife, searchParams.chosenMoveTemperatureEarly, searchParams.chosenMoveTemperature
  );
  if(chosenMoveTemperature < 1e-3)
    return requiredVisits;
  double requiredVisitsDueToTemp = maxVisitsMoveVisits * pow(0.01, chosenMoveTemperature);
  return std::min(requiredVisits, requiredVisitsDueToTemp);
}

double Search::computeUpperBoundVisitsLeftDueToTime(
  int64_t rootVisits, double timeUsed, double plannedTimeLimit
) {
  if(rootVisits <= 1)
    return 1e30;
  double timeThoughtSoFar = effectiveSearchTimeCarriedOver + timeUsed;
  double timeLeftPlanned = plannedTimeLimit - timeUsed;
  if(timeThoughtSoFar < 0.1)
    return 1e30;

  double proportionOfTimeThoughtLeft = timeLeftPlanned / timeThoughtSoFar;
  return ceil(proportionOfTimeThoughtLeft * rootVisits + searchParams.numThreads - 1);
}

double Search::recomputeSearchTimeLimit(
  const TimeControls& tc, double timeUsed, double searchFactor, int64_t rootVisits
) {
  double tcMin;
  double tcRec;
  double tcMax;
  getQ4Time(tc, rootState.plies, searchParams.lagBuffer, tcMin, tcRec, tcMax);

  tcRec *= searchParams.overallocateTimeFactor;

  if(searchParams.midgameTimeFactor != 1.0) {
    double boardAreaScale = 81.0 / 361.0;
    double presumedTurnNumber = (double)rootState.plies;
    if(presumedTurnNumber < 0) presumedTurnNumber = 0;

    double midGameWeight;
    if(presumedTurnNumber < searchParams.midgameTurnPeakTime * boardAreaScale)
      midGameWeight = presumedTurnNumber / (searchParams.midgameTurnPeakTime * boardAreaScale);
    else
      midGameWeight = exp(
        -(presumedTurnNumber - searchParams.midgameTurnPeakTime * boardAreaScale) /
        (searchParams.endgameTurnTimeDecay * boardAreaScale)
      );
    if(midGameWeight < 0) midGameWeight = 0;
    if(midGameWeight > 1) midGameWeight = 1;

    tcRec *= 1.0 + midGameWeight * (searchParams.midgameTimeFactor - 1.0);
  }

  if(searchParams.obviousMovesTimeFactor < 1.0) {
    double surprise = 0.0;
    double policyEntropy = 0.0;
    getPolicySurpriseAndEntropy(surprise, policyEntropy);
    double obviousnessByEntropy = exp(-policyEntropy / searchParams.obviousMovesPolicyEntropyTolerance);
    double obviousnessBySurprise = exp(-surprise / searchParams.obviousMovesPolicySurpriseTolerance);
    double obviousnessWeight = std::min(obviousnessByEntropy, obviousnessBySurprise);
    tcRec *= 1.0 + obviousnessWeight * (searchParams.obviousMovesTimeFactor - 1.0);
  }

  if(tcRec > 1e-20) {
    double remainingTimeNeeded = tcRec - effectiveSearchTimeCarriedOver;
    double remainingTimeNeededFactor = remainingTimeNeeded / tcRec;
    tcRec = tcRec * std::min(1.0, log(1.0 + exp(remainingTimeNeededFactor * 6.0)) / 6.0);
  }

  tcRec = tc.roundUpTimeLimitIfNeeded(searchParams.lagBuffer, timeUsed, tcRec);
  if(tcRec > tcMax) tcRec = tcMax;

  if(searchParams.futileVisitsThreshold > 0) {
    double upperBoundVisitsLeftDueToTime = computeUpperBoundVisitsLeftDueToTime(rootVisits, timeUsed, tcRec);
    if(upperBoundVisitsLeftDueToTime < searchParams.futileVisitsThreshold * rootVisits) {
      std::vector<int> actions;
      std::vector<double> playSelectionValues;
      std::vector<double> visitCounts;
      bool suc = getPlaySelectionValues(actions, playSelectionValues, &visitCounts, 1.0);
      if(suc && playSelectionValues.size() > 0) {
        if(playSelectionValues.size() == visitCounts.size()) {
          int numMoves = (int)playSelectionValues.size();
          int maxVisitsIdx = 0;
          int bestMoveIdx = 0;
          for(int i = 1; i < numMoves; i++) {
            if(playSelectionValues[i] > playSelectionValues[bestMoveIdx])
              bestMoveIdx = i;
            if(visitCounts[i] > visitCounts[maxVisitsIdx])
              maxVisitsIdx = i;
          }
          if(maxVisitsIdx == bestMoveIdx) {
            double requiredVisits = numVisitsNeededToBeNonFutile(visitCounts[maxVisitsIdx]);
            bool foundPossibleAlternativeMove = false;
            for(int i = 0; i < numMoves; i++) {
              if(i == bestMoveIdx)
                continue;
              if(visitCounts[i] + upperBoundVisitsLeftDueToTime >= requiredVisits) {
                foundPossibleAlternativeMove = true;
                break;
              }
            }
            if(!foundPossibleAlternativeMove) {
              tcRec = timeUsed * (1.0 - (1e-10));
            }
          }
        }
      }
    }
  }

  tcRec = tc.roundUpTimeLimitIfNeeded(searchParams.lagBuffer, timeUsed, tcRec);
  if(tcRec > tcMax) tcRec = tcMax;

  if(tcRec < tcMin) tcRec = tcMin;
  tcRec *= searchFactor;
  if(tcRec > tcMax) tcRec = tcMax;

  return tcRec;
}

}  // namespace Q4S
