#ifndef Q4SEARCH_REPORTEDSEARCHVALUES_H_
#define Q4SEARCH_REPORTEDSEARCHVALUES_H_

#include "../../core/global.h"

namespace Q4S {

struct ReportedSearchValues {
  double value[5];    // win probability for seats 0..3, and [4] = draw
  double utility[4];  // utility of seats 0..3
  double weight;
  int64_t visits;

  ReportedSearchValues();
  ReportedSearchValues(
    const double valueAvg[5],
    const double utilityAvg[4],
    double totalWeight,
    int64_t totalVisits
  );
  ~ReportedSearchValues();

  friend std::ostream& operator<<(std::ostream& out, const ReportedSearchValues& values);
};

}  // namespace Q4S

#endif  // Q4SEARCH_REPORTEDSEARCHVALUES_H_
