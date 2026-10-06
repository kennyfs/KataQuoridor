#ifndef Q4_NNTOOL_H_
#define Q4_NNTOOL_H_

#include "q4history.h"

#include <cstdint>
#include <string>
#include <vector>

// The neural-network subcommands of `katago q4tool`:
//   dumpinputs  inputs of sampled positions (all 8 symmetries) as JSON lines (T13)
//   evalnn      raw and decoded outputs of a model for positions (T16, T17)
//   symavg      invariance of the 8-symmetry average (T17)
//   nncache     NN cache hits and misses (T18)
//   nnbench     input-fill time and NN evaluations per second (B8)
namespace Q4NNTool {
  int run(const std::string& subcmd, const std::vector<std::string>& args);

  // n positions from random games of several kinds: pawn shuffles (repetitions), play near maxPlies, wall-heavy
  // games, games with eliminated seats and walls running out, plain random walks. Deterministic in seed.
  std::vector<Q4History> sampleHistories(uint64_t seed, int n);
}

#endif  // Q4_NNTOOL_H_
