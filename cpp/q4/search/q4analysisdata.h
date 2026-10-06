#ifndef Q4SEARCH_ANALYSISDATA_H_
#define Q4SEARCH_ANALYSISDATA_H_

#include <cstdint>
#include <ostream>
#include <vector>

#include "../../core/global.h"
#include "../q4board.h"

namespace Q4S {

struct SearchNode;

struct AnalysisData {
  int move;                   // action (0..320, NULL_ACTION = -1)
  int64_t numVisits;
  double playSelectionValue;  // visits with LCB adjustment if applicable
  double lcb;                 // mover's LCB in utility units
  double radius;              // utility radius
  double utility;             // mover's utility
  double policyPrior;         // prior probability (0..1)
  double valueAvg[5];         // win probabilities seat 0..3, [4] = draw
  double utilityAvg[4];       // utilities of seats 0..3
  double ess;                 // effective sample size
  double weightFactor;
  double weightSum;
  double weightSqSum;
  double utilitySqAvg;        // mover's utility squared
  int64_t childVisits;
  double childWeightSum;
  int order;                  // preference order, 0 is best
  std::vector<int> pv;        // actions in principal variation
  std::vector<int64_t> pvVisits;
  std::vector<int64_t> pvEdgeVisits;

  const SearchNode* node;     // valid only as long as search is not cleared

  AnalysisData();
  AnalysisData(const AnalysisData& other);
  AnalysisData(AnalysisData&& other) noexcept;
  ~AnalysisData();

  AnalysisData& operator=(const AnalysisData& other);
  AnalysisData& operator=(AnalysisData&& other) noexcept;

  void writePV(std::ostream& out) const;
  void writePVVisits(std::ostream& out) const;
  void writePVEdgeVisits(std::ostream& out) const;
};

bool operator<(const AnalysisData& a0, const AnalysisData& a1);

}  // namespace Q4S

#endif  // Q4SEARCH_ANALYSISDATA_H_
