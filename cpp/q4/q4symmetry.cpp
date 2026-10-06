#include "q4symmetry.h"

#include <mutex>
#include <stdexcept>

namespace {
  std::once_flag initFlag;  // init() is called from every NN evaluation, on any thread
  int cellMap[8][Q4Board::NUM_CELLS];
  int anchorMap[8][Q4Board::NUM_WALL_ANCHORS][2][2]; // [sym][a][isH][0: newA, 1: newIsH]
  int actionMap[8][Q4Board::NUM_ACTIONS];
  int dirMap[8][4];
  int composeTable[8][8];
  int invTable[8];

  // Raw coordinate transform relative to origin (0, 0)
  void transformCoords(int x, int y, int sym, int& ox, int& oy) {
    switch(sym) {
      case 0: ox = x;   oy = y;   break; // Identity
      case 1: ox = -y;  oy = x;   break; // Rot90 CCW
      case 2: ox = -x;  oy = -y;  break; // Rot180
      case 3: ox = y;   oy = -x;  break; // Rot270 CCW
      case 4: ox = -x;  oy = y;   break; // Flip X
      case 5: ox = x;   oy = -y;  break; // Flip Y
      case 6: ox = y;   oy = x;   break; // Transpose (Flip main diag)
      case 7: ox = -y;  oy = -x;  break; // Flip anti-diag
      default: ox = x;  oy = y;   break;
    }
  }

  bool symSwapsAxes(int sym) {
    return (sym == 1 || sym == 3 || sym == 6 || sym == 7);
  }
}

namespace Q4Symmetry {

static void buildTables();

void init() {
  std::call_once(initFlag, buildTables);
}

static void buildTables() {
  // 1. Cells: x, y in [0, 10] around center (5, 5)
  for(int sym = 0; sym < 8; sym++) {
    for(int c = 0; c < Q4Board::NUM_CELLS; c++) {
      int x = Q4Board::cellX(c) - 5;
      int y = Q4Board::cellY(c) - 5;
      int ox, oy;
      transformCoords(x, y, sym, ox, oy);
      int nx = ox + 5;
      int ny = oy + 5;
      cellMap[sym][c] = Q4Board::cellOf(nx, ny);
    }
  }

  // 2. Wall anchors: ax, ay in [0, 9] around (4.5, 4.5)
  // Let ax' = 2*ax - 9 in {-9, -7, ..., 9}
  for(int sym = 0; sym < 8; sym++) {
    bool swapH = symSwapsAxes(sym);
    for(int a = 0; a < Q4Board::NUM_WALL_ANCHORS; a++) {
      int ax = 2 * Q4Board::anchorX(a) - 9;
      int ay = 2 * Q4Board::anchorY(a) - 9;
      int oax, oay;
      transformCoords(ax, ay, sym, oax, oay);
      int nax = (oax + 9) / 2;
      int nay = (oay + 9) / 2;
      int na = Q4Board::anchorOf(nax, nay);

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

  // 4. Actions: 0..120 (pawns), 121..220 (vwall), 221..320 (hwall)
  for(int sym = 0; sym < 8; sym++) {
    for(int act = 0; act < Q4Board::NUM_ACTIONS; act++) {
      if(Q4Board::isPawnAction(act)) {
        actionMap[sym][act] = cellMap[sym][act];
      }
      else {
        bool isH = Q4Board::isHWallAction(act);
        int a = isH ? (act - 221) : (act - 121);
        int na = anchorMap[sym][a][isH ? 1 : 0][0];
        bool nIsH = (anchorMap[sym][a][isH ? 1 : 0][1] == 1);
        actionMap[sym][act] = nIsH ? Q4Board::actionOfHWall(na) : Q4Board::actionOfVWall(na);
      }
    }
  }

  // 5. Composition and Inverse
  for(int s1 = 0; s1 < 8; s1++) {
    for(int s2 = 0; s2 < 8; s2++) {
      for(int s3 = 0; s3 < 8; s3++) {
        bool match = true;
        for(int c = 0; c < Q4Board::NUM_CELLS; c++) {
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

int applyCell(int cell, int sym) {
  init();
  return cellMap[sym][cell];
}

void applyAnchor(int ax, int ay, bool isHorizontal, int sym,
                 int& outAx, int& outAy, bool& outIsHorizontal) {
  init();
  int a = Q4Board::anchorOf(ax, ay);
  int isHIdx = isHorizontal ? 1 : 0;
  int na = anchorMap[sym][a][isHIdx][0];
  outAx = Q4Board::anchorX(na);
  outAy = Q4Board::anchorY(na);
  outIsHorizontal = (anchorMap[sym][a][isHIdx][1] == 1);
}

int applyAction(int action, int sym) {
  init();
  return actionMap[sym][action];
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

Q4Board applyBoard(const Q4Board& board, int sym) {
  init();
  Q4Board b;
  b.alive = board.alive;
  b.toMove = board.toMove;
  std::fill(std::begin(b.occupant), std::end(b.occupant), -1);

  for(int s = 0; s < Q4Board::NUM_SEATS; s++) {
    b.wallsLeft[s] = board.wallsLeft[s];
    if(board.isAlive(s) && board.pawn[s] >= 0) {
      int nc = applyCell(board.pawn[s], sym);
      b.pawn[s] = nc;
      b.occupant[nc] = s;
    }
    else {
      b.pawn[s] = -1;
    }
  }

  for(int ay = 0; ay < Q4Board::NUM_ANCHORS; ay++) {
    for(int ax = 0; ax < Q4Board::NUM_ANCHORS; ax++) {
      int a = Q4Board::anchorOf(ax, ay);
      if(board.hWalls.test(a)) {
        int nax, nay;
        bool nIsH;
        applyAnchor(ax, ay, true, sym, nax, nay, nIsH);
        int na = Q4Board::anchorOf(nax, nay);
        if(nIsH) b.hWalls.set(na);
        else b.vWalls.set(na);
        b.addWallToBlocked(nax, nay, nIsH);
      }
      if(board.vWalls.test(a)) {
        int nax, nay;
        bool nIsH;
        applyAnchor(ax, ay, false, sym, nax, nay, nIsH);
        int na = Q4Board::anchorOf(nax, nay);
        if(nIsH) b.hWalls.set(na);
        else b.vWalls.set(na);
        b.addWallToBlocked(nax, nay, nIsH);
      }
    }
  }

  b.recomputeDistancesToCenter();
  b.hash = b.getHashFromScratch();
  return b;
}

Q4History applyHistory(const Q4History& history, int sym) {
  Q4History out(applyBoard(history.initialBoard, sym), history.rules);
  for(const Q4Event& ev : history.events) {
    if(ev.isElimination)
      out.eliminate(ev.eliminatedSeat);
    else
      out.play(applyAction(ev.action, sym));
  }
  return out;
}

}  // namespace Q4Symmetry
