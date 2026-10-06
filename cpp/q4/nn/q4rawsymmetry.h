#ifndef Q4_RAW_SYMMETRY_H_
#define Q4_RAW_SYMMETRY_H_

#include "q4nnconstants.h"
#include "../../neuralnet/nninputs.h"

// Dependency-free symmetry mapping tables for raw Q4 neural network tensors (Plan §7 item 1).
// This header is included by shared neural net code (neuralnet/nneval.cpp), so it must not
// depend on any game logic or board structures.
namespace Q4RawSymmetry {
  static constexpr int NUM_SYMMETRIES = 8;

  void init();

  // Cell mapping (0..120) for 11x11 board
  int applyCell(int cell, int sym);

  // Wall anchor mapping (ax, ay in 0..9)
  void applyAnchor(int ax, int ay, bool isHorizontal, int sym,
                   int& outAx, int& outAy, bool& outIsHorizontal);

  // Direction mapping (0..3)
  int applyDirection(int dir, int sym);

  // Group operations
  int compose(int sym1, int sym2); // sym1(sym2(x))
  int inverse(int sym);

  // Convert raw NN output computed under symmetry sym into symmetry 0 orientation.
  void unapply(const Q4RawNNOutput& src, Q4RawNNOutput& dst, int sym);

  // In-place conversion of raw NN output into symmetry 0 orientation.
  void unapplyInPlace(Q4RawNNOutput& raw, int sym);
}

#endif  // Q4_RAW_SYMMETRY_H_
