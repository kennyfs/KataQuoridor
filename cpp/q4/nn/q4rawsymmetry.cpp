#include "q4rawsymmetry.h"

#include <algorithm>
#include <mutex>

namespace {
  using namespace Q4NNConst;

  std::once_flag initFlag;
  int cellMap[8][POS_AREA];
  int anchorMap[8][100][2][2]; // [sym][a][isH][0: newA, 1: newIsH]
  int dirMap[8][4];
  int composeTable[8][8];
  int invTable[8];

  void transformCoords(int x, int y, int sym, int& ox, int& oy) {
    switch(sym) {
      case 0: ox = x;   oy = y;   break; // Identity
      case 1: ox = -y;  oy = x;   break; // Rot90 CCW
      case 2: ox = -x;  oy = -y;  break; // Rot180
      case 3: ox = y;   oy = -x;  break; // Rot270 CCW
      case 4: ox = -x;  oy = y;   break; // Flip X
      case 5: ox = x;   oy = -y;  break; // Flip Y
      case 6: ox = y;   oy = x;   break; // Transpose
      case 7: ox = -y;  oy = -x;  break; // Flip anti-diag
      default: ox = x;  oy = y;   break;
    }
  }

  bool symSwapsAxes(int sym) {
    return (sym == 1 || sym == 3 || sym == 6 || sym == 7);
  }

  void buildTables() {
    // 1. Cells: x, y in [0, 10] around center (5, 5)
    for(int sym = 0; sym < 8; sym++) {
      for(int c = 0; c < POS_AREA; c++) {
        int x = (c % POS_LEN) - 5;
        int y = (c / POS_LEN) - 5;
        int ox, oy;
        transformCoords(x, y, sym, ox, oy);
        int nx = ox + 5;
        int ny = oy + 5;
        cellMap[sym][c] = ny * POS_LEN + nx;
      }
    }

    // 2. Wall anchors: ax, ay in [0, 9] around (4.5, 4.5)
    for(int sym = 0; sym < 8; sym++) {
      bool swapH = symSwapsAxes(sym);
      for(int a = 0; a < 100; a++) {
        int ax = 2 * (a % 10) - 9;
        int ay = 2 * (a / 10) - 9;
        int oax, oay;
        transformCoords(ax, ay, sym, oax, oay);
        int nax = (oax + 9) / 2;
        int nay = (oay + 9) / 2;
        int na = nay * 10 + nax;

        for(int isH = 0; isH < 2; isH++) {
          bool newIsH = swapH ? (!isH) : (isH != 0);
          anchorMap[sym][a][isH][0] = na;
          anchorMap[sym][a][isH][1] = newIsH ? 1 : 0;
        }
      }
    }

    // 3. Directions: (0: N, 1: E, 2: S, 3: W)
    const int dirDx[4] = {0, 1, 0, -1};
    const int dirDy[4] = {1, 0, -1, 0};
    for(int sym = 0; sym < 8; sym++) {
      for(int d = 0; d < 4; d++) {
        int odx, ody;
        transformCoords(dirDx[d], dirDy[d], sym, odx, ody);
        for(int nd = 0; nd < 4; nd++) {
          if(dirDx[nd] == odx && dirDy[nd] == ody) {
            dirMap[sym][d] = nd;
            break;
          }
        }
      }
    }

    // 4. Composition and Inverse
    for(int s1 = 0; s1 < 8; s1++) {
      for(int s2 = 0; s2 < 8; s2++) {
        for(int s3 = 0; s3 < 8; s3++) {
          bool match = true;
          for(int c = 0; c < POS_AREA; c++) {
            if(cellMap[s3][c] != cellMap[s1][cellMap[s2][c]]) {
              match = false;
              break;
            }
          }
          if(match) {
            composeTable[s1][s2] = s3;
            break;
          }
        }
      }
    }

    for(int s = 0; s < 8; s++) {
      for(int inv = 0; inv < 8; inv++) {
        if(composeTable[s][inv] == 0) {
          invTable[s] = inv;
          break;
        }
      }
    }
  }
}

namespace Q4RawSymmetry {

void init() {
  std::call_once(initFlag, buildTables);
}

int applyCell(int cell, int sym) {
  init();
  return cellMap[sym][cell];
}

void applyAnchor(int ax, int ay, bool isHorizontal, int sym,
                 int& outAx, int& outAy, bool& outIsHorizontal) {
  init();
  int a = ay * 10 + ax;
  int isHIdx = isHorizontal ? 1 : 0;
  int na = anchorMap[sym][a][isHIdx][0];
  outAx = na % 10;
  outAy = na / 10;
  outIsHorizontal = (anchorMap[sym][a][isHIdx][1] == 1);
}

int applyDirection(int dir, int sym) {
  init();
  return dirMap[sym][dir];
}

int compose(int sym1, int sym2) {
  init();
  return composeTable[sym1][sym2];
}

int inverse(int sym) {
  init();
  return invTable[sym];
}

void unapply(const Q4RawNNOutput& src, Q4RawNNOutput& dst, int sym) {
  init();
  if(sym == 0) {
    dst = src;
    return;
  }

  // Values and misc are invariant under board symmetries
  std::copy(src.valueLogits, src.valueLogits + NUM_VALUE_LOGITS, dst.valueLogits);
  std::copy(src.miscValues, src.miscValues + NUM_MISC, dst.miscValues);

  // Trajectory: 11x11 cells
  for(int c = 0; c < POS_AREA; c++) {
    int tc = cellMap[sym][c];
    dst.trajectoryLogits[c] = src.trajectoryLogits[tc];
  }

  // Policy: 2 variants x 3 planes (pawn, V-wall, H-wall)
  for(int variant = 0; variant < NUM_POLICY_VARIANTS; variant++) {
    const float* srcVar = src.policyLogits + variant * POLICY_SLOTS_PER_VARIANT;
    float* dstVar = dst.policyLogits + variant * POLICY_SLOTS_PER_VARIANT;

    // Plane 0: pawn moves
    for(int c = 0; c < POS_AREA; c++) {
      int tc = cellMap[sym][c];
      dstVar[0 * POS_AREA + c] = srcVar[0 * POS_AREA + tc];
    }

    // Planes 1 & 2: walls (10x10 anchor domain in 11x11 grid)
    std::fill(dstVar + 1 * POS_AREA, dstVar + 3 * POS_AREA, 0.0f);
    for(int ay = 0; ay < 10; ay++) {
      for(int ax = 0; ax < 10; ax++) {
        int a = ay * 10 + ax;
        int c = ay * POS_LEN + ax;

        // V-wall at anchor a (plane 1, orient 0)
        int naV = anchorMap[sym][a][0][0];
        int isHV = anchorMap[sym][a][0][1];
        int naxV = naV % 10;
        int nayV = naV / 10;
        int ncV = nayV * POS_LEN + naxV;
        int planeV = isHV ? 2 : 1;
        dstVar[1 * POS_AREA + c] = srcVar[planeV * POS_AREA + ncV];

        // H-wall at anchor a (plane 2, orient 1)
        int naH = anchorMap[sym][a][1][0];
        int isHH = anchorMap[sym][a][1][1];
        int naxH = naH % 10;
        int nayH = naH / 10;
        int ncH = nayH * POS_LEN + naxH;
        int planeH = isHH ? 2 : 1;
        dstVar[2 * POS_AREA + c] = srcVar[planeH * POS_AREA + ncH];
      }
    }
  }
}

void unapplyInPlace(Q4RawNNOutput& raw, int sym) {
  if(sym == 0)
    return;
  Q4RawNNOutput tmp = raw;
  unapply(tmp, raw, sym);
}

}  // namespace Q4RawSymmetry
