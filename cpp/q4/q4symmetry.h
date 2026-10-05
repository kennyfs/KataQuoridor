#ifndef Q4_SYMMETRY_H_
#define Q4_SYMMETRY_H_

#include "q4board.h"

namespace Q4Symmetry {
  static constexpr int NUM_SYMMETRIES = 8;

  void init();

  // Cell mapping (0..120)
  int applyCell(int cell, int sym);

  // Anchor mapping (ax: 0..9, ay: 0..9)
  void applyAnchor(int ax, int ay, bool isHorizontal, int sym,
                   int& outAx, int& outAy, bool& outIsHorizontal);

  // Action mapping (0..320)
  int applyAction(int action, int sym);

  // Direction mapping (0..3)
  int applyDirection(int dir, int sym);

  // Group operations
  int compose(int sym1, int sym2); // sym1(sym2(x))
  int inverse(int sym);

  // Board transformation: rotates/reflects the board geometry
  Q4Board applyBoard(const Q4Board& board, int sym);
}

#endif  // Q4_SYMMETRY_H_
