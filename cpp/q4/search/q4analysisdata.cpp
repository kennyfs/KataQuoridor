#include "q4analysisdata.h"

#include "../q4notation.h"

namespace Q4S {

AnalysisData::AnalysisData() {
  move = Q4Board::NULL_ACTION;
  numVisits = 0;
  playSelectionValue = 0.0;
  lcb = 0.0;
  radius = 0.0;
  utility = 0.0;
  policyPrior = 0.0;
  for(int i = 0; i < 5; i++) valueAvg[i] = 0.0;
  for(int i = 0; i < 4; i++) utilityAvg[i] = 0.0;
  ess = 0.0;
  weightFactor = 0.0;
  weightSum = 0.0;
  weightSqSum = 0.0;
  utilitySqAvg = 0.0;
  childVisits = 0;
  childWeightSum = 0.0;
  order = 0;
  node = nullptr;
}

AnalysisData::AnalysisData(const AnalysisData& other)
  : move(other.move),
    numVisits(other.numVisits),
    playSelectionValue(other.playSelectionValue),
    lcb(other.lcb),
    radius(other.radius),
    utility(other.utility),
    policyPrior(other.policyPrior),
    ess(other.ess),
    weightFactor(other.weightFactor),
    weightSum(other.weightSum),
    weightSqSum(other.weightSqSum),
    utilitySqAvg(other.utilitySqAvg),
    childVisits(other.childVisits),
    childWeightSum(other.childWeightSum),
    order(other.order),
    pv(other.pv),
    pvVisits(other.pvVisits),
    pvEdgeVisits(other.pvEdgeVisits),
    node(other.node)
{
  for(int i = 0; i < 5; i++) valueAvg[i] = other.valueAvg[i];
  for(int i = 0; i < 4; i++) utilityAvg[i] = other.utilityAvg[i];
}

AnalysisData::AnalysisData(AnalysisData&& other) noexcept
  : move(other.move),
    numVisits(other.numVisits),
    playSelectionValue(other.playSelectionValue),
    lcb(other.lcb),
    radius(other.radius),
    utility(other.utility),
    policyPrior(other.policyPrior),
    ess(other.ess),
    weightFactor(other.weightFactor),
    weightSum(other.weightSum),
    weightSqSum(other.weightSqSum),
    utilitySqAvg(other.utilitySqAvg),
    childVisits(other.childVisits),
    childWeightSum(other.childWeightSum),
    order(other.order),
    pv(std::move(other.pv)),
    pvVisits(std::move(other.pvVisits)),
    pvEdgeVisits(std::move(other.pvEdgeVisits)),
    node(other.node)
{
  for(int i = 0; i < 5; i++) valueAvg[i] = other.valueAvg[i];
  for(int i = 0; i < 4; i++) utilityAvg[i] = other.utilityAvg[i];
}

AnalysisData::~AnalysisData() {}

AnalysisData& AnalysisData::operator=(const AnalysisData& other) {
  if(this == &other) return *this;
  move = other.move;
  numVisits = other.numVisits;
  playSelectionValue = other.playSelectionValue;
  lcb = other.lcb;
  radius = other.radius;
  utility = other.utility;
  policyPrior = other.policyPrior;
  for(int i = 0; i < 5; i++) valueAvg[i] = other.valueAvg[i];
  for(int i = 0; i < 4; i++) utilityAvg[i] = other.utilityAvg[i];
  ess = other.ess;
  weightFactor = other.weightFactor;
  weightSum = other.weightSum;
  weightSqSum = other.weightSqSum;
  utilitySqAvg = other.utilitySqAvg;
  childVisits = other.childVisits;
  childWeightSum = other.childWeightSum;
  order = other.order;
  pv = other.pv;
  pvVisits = other.pvVisits;
  pvEdgeVisits = other.pvEdgeVisits;
  node = other.node;
  return *this;
}

AnalysisData& AnalysisData::operator=(AnalysisData&& other) noexcept {
  if(this == &other) return *this;
  move = other.move;
  numVisits = other.numVisits;
  playSelectionValue = other.playSelectionValue;
  lcb = other.lcb;
  radius = other.radius;
  utility = other.utility;
  policyPrior = other.policyPrior;
  for(int i = 0; i < 5; i++) valueAvg[i] = other.valueAvg[i];
  for(int i = 0; i < 4; i++) utilityAvg[i] = other.utilityAvg[i];
  ess = other.ess;
  weightFactor = other.weightFactor;
  weightSum = other.weightSum;
  weightSqSum = other.weightSqSum;
  utilitySqAvg = other.utilitySqAvg;
  childVisits = other.childVisits;
  childWeightSum = other.childWeightSum;
  order = other.order;
  pv = std::move(other.pv);
  pvVisits = std::move(other.pvVisits);
  pvEdgeVisits = std::move(other.pvEdgeVisits);
  node = other.node;
  return *this;
}

void AnalysisData::writePV(std::ostream& out) const {
  for(size_t i = 0; i < pv.size(); i++) {
    if(i > 0) out << " ";
    out << Q4Notation::actionToString(pv[i]);
  }
}

void AnalysisData::writePVVisits(std::ostream& out) const {
  for(size_t i = 0; i < pvVisits.size(); i++) {
    if(i > 0) out << " ";
    out << pvVisits[i];
  }
}

void AnalysisData::writePVEdgeVisits(std::ostream& out) const {
  for(size_t i = 0; i < pvEdgeVisits.size(); i++) {
    if(i > 0) out << " ";
    out << pvEdgeVisits[i];
  }
}

bool operator<(const AnalysisData& a0, const AnalysisData& a1) {
  if(a0.playSelectionValue > a1.playSelectionValue) return true;
  if(a0.playSelectionValue < a1.playSelectionValue) return false;
  if(a0.numVisits > a1.numVisits) return true;
  if(a0.numVisits < a1.numVisits) return false;
  if(a0.policyPrior > a1.policyPrior) return true;
  if(a0.policyPrior < a1.policyPrior) return false;
  return a0.move < a1.move;
}

}  // namespace Q4S
