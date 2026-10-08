#ifndef Q4_GATING_H_
#define Q4_GATING_H_

#include <cmath>
#include <string>
#include <vector>

// The decision of `katago q4gatekeeper` (docs/q4/rounds/R6.md), apart from playing the games, so it can be tested on
// canned results. A candidate net is accepted iff
//   1. its score in the ABAB / BABA tables (two copies of the candidate, two of the current net, seats rotated; a win
//      is 1 point, a draw half a point) is at least `requiredScore` (default 0.5, ties accept as Duel's gatekeeper), and
//   2. it is not worse than the current net on the mixed benchmark (candidate / strong net / two weak bots) beyond the
//      confidence interval: with d_o = (candidate's points at opening o) - (current net's points at opening o) on the
//      same openings, the net is rejected when mean(d) + z * stderr(d) < 0.
namespace Q4Gate {

struct Decision {
  bool accept = false;
  bool ababOk = false;
  bool benchChecked = false;
  bool benchOk = true;
  double ababScore = 0.0;
  double benchDiff = 0.0;      // mean of d_o
  double benchDiffHi = 0.0;    // upper end of the interval
  std::string reason;
};

inline Decision decide(
  double ababCandidatePoints,
  int ababGames,
  double requiredScore,
  const std::vector<double>& benchCandidatePerOpening,
  const std::vector<double>& benchCurrentPerOpening,
  double z = 1.96
) {
  Decision d;
  d.ababScore = ababGames > 0 ? ababCandidatePoints / ababGames : 0.0;
  d.ababOk = ababGames > 0 && ababCandidatePoints + 1e-10 >= requiredScore * ababGames;

  size_t n = benchCandidatePerOpening.size();
  if(n != benchCurrentPerOpening.size())
    n = 0;
  if(n >= 2) {
    d.benchChecked = true;
    double mean = 0.0;
    for(size_t i = 0; i < n; i++)
      mean += benchCandidatePerOpening[i] - benchCurrentPerOpening[i];
    mean /= (double)n;
    double var = 0.0;
    for(size_t i = 0; i < n; i++) {
      double x = benchCandidatePerOpening[i] - benchCurrentPerOpening[i] - mean;
      var += x * x;
    }
    var /= (double)(n - 1);
    double se = std::sqrt(var / (double)n);
    d.benchDiff = mean;
    d.benchDiffHi = mean + z * se;
    d.benchOk = !(d.benchDiffHi < 0.0);
  }

  d.accept = d.ababOk && d.benchOk;
  if(!d.ababOk)
    d.reason = "lost the ABAB tables";
  else if(!d.benchOk)
    d.reason = "worse than the current net on the mixed benchmark beyond the CI";
  else
    d.reason = "won the ABAB tables" + std::string(d.benchChecked ? " and is not worse on the benchmark" : " (no benchmark)");
  return d;
}

}  // namespace Q4Gate

#endif  // Q4_GATING_H_
