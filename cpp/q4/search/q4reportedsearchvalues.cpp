#include "q4reportedsearchvalues.h"

namespace Q4S {

ReportedSearchValues::ReportedSearchValues() {
  for(int i = 0; i < 5; i++) value[i] = 0.0;
  for(int i = 0; i < 4; i++) utility[i] = 0.0;
  weight = 0.0;
  visits = 0;
}

ReportedSearchValues::ReportedSearchValues(
  const double valueAvg[5],
  const double utilityAvg[4],
  double totalWeight,
  int64_t totalVisits
) {
  for(int i = 0; i < 5; i++) value[i] = valueAvg[i];
  for(int i = 0; i < 4; i++) utility[i] = utilityAvg[i];
  weight = totalWeight;
  visits = totalVisits;
}

ReportedSearchValues::~ReportedSearchValues() {}

std::ostream& operator<<(std::ostream& out, const ReportedSearchValues& v) {
  out << "visits " << v.visits << " weight " << v.weight << "\n";
  for(int s = 0; s < 4; s++)
    out << "seat " << s + 1 << " win " << v.value[s] << " util " << v.utility[s] << "\n";
  out << "draw " << v.value[4] << "\n";
  return out;
}

}  // namespace Q4S
