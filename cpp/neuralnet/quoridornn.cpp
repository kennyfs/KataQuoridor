#include "../neuralnet/quoridornn.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "../core/test.h"

using namespace std;

int QuoridorNN::numSpatialFeatures(int ioVersion) {
  if(ioVersion == 1)
    return NUM_FEATURES_SPATIAL_V1;
  ASSERT_UNREACHABLE;
}

int QuoridorNN::numGlobalFeatures(int ioVersion) {
  if(ioVersion == 1)
    return NUM_FEATURES_GLOBAL_V1;
  ASSERT_UNREACHABLE;
}

static void setRowBin(float* rowBin, int pos, int feature, float value, int posStride, int featureStride) {
  rowBin[pos * posStride + feature * featureStride] = value;
}

void QuoridorNN::fillRow(
  const Board& board,
  const BoardHistory& boardHistory,
  Player nextPlayer,
  const MiscNNInputParams& nnInputParams,
  int ioVersion,
  bool useNHWC,
  float* rowSpatial,
  float* rowGlobal
) {
  (void)nnInputParams;
  (void)boardHistory;
  testAssert(ioVersion == 1);
  const int nnXLen = MODEL_LEN;
  const int nnYLen = MODEL_LEN;
  assert(nextPlayer == P_BLACK || nextPlayer == P_WHITE);

  std::fill(rowSpatial, rowSpatial + NUM_FEATURES_SPATIAL_V1 * nnXLen * nnYLen, 0.0f);
  std::fill(rowGlobal, rowGlobal + NUM_FEATURES_GLOBAL_V1, 0.0f);

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NUM_FEATURES_SPATIAL_V1;
  }
  else {
    featureStride = nnXLen * nnYLen;
    posStride = 1;
  }

  // Current and opponent pawns
  Loc curPawnLoc = (nextPlayer == P_BLACK) ? board.blackPawnLoc : board.whitePawnLoc;
  Loc oppPawnLoc = (nextPlayer == P_BLACK) ? board.whitePawnLoc : board.blackPawnLoc;
  int curPawnC = Location::getX(curPawnLoc, board.x_size) / 2;
  int curPawnR = Location::getY(curPawnLoc, board.x_size) / 2;
  int oppPawnC = Location::getX(oppPawnLoc, board.x_size) / 2;
  int oppPawnR = Location::getY(oppPawnLoc, board.x_size) / 2;

  // Board edge blocks
  bool blockedN[9][9];
  bool blockedS[9][9];
  bool blockedE[9][9];
  bool blockedW[9][9];

  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      Loc loc = Location::pawnLoc(c, r, board.x_size);
      blockedN[c][r] = (r == 0) || !board.canPawnStep(loc, Location::pawnLoc(c, r - 1, board.x_size));
      blockedS[c][r] = (r == 8) || !board.canPawnStep(loc, Location::pawnLoc(c, r + 1, board.x_size));
      blockedE[c][r] = (c == 8) || !board.canPawnStep(loc, Location::pawnLoc(c + 1, r, board.x_size));
      blockedW[c][r] = (c == 0) || !board.canPawnStep(loc, Location::pawnLoc(c - 1, r, board.x_size));
    }
  }

  // Placed wall anchors (c, r in [0..7]). Read directly from Board's explicit wall arrays
  // rather than reconstructing from `colors`, which is ambiguous: e.g. a horizontal wall at
  // (c, r) and a vertical wall at (c, r-1) share an arm cell, so "center occupied + arm
  // occupied" alone cannot tell which (or both) walls are actually placed.
  const bool (&hasVWall)[8][8] = board.vWalls;
  const bool (&hasHWall)[8][8] = board.hWalls;

  // BFS Distances
  auto runBFS = [&](const std::vector<std::pair<int,int>>& sources, int distMap[9][9]) {
    for(int r = 0; r < 9; r++) {
      for(int c = 0; c < 9; c++) {
        distMap[c][r] = -1;
      }
    }
    int qC[81];
    int qR[81];
    int qHead = 0;
    int qTail = 0;

    for(const auto& s : sources) {
      qC[qTail] = s.first;
      qR[qTail] = s.second;
      qTail++;
      distMap[s.first][s.second] = 0;
    }

    const int dc[4] = {0, 0, 1, -1};
    const int dr[4] = {-1, 1, 0, 0};

    while(qHead < qTail) {
      int c = qC[qHead];
      int r = qR[qHead];
      qHead++;
      int d = distMap[c][r];
      Loc currLoc = Location::pawnLoc(c, r, board.x_size);

      for(int i = 0; i < 4; i++) {
        int nc = c + dc[i];
        int nr = r + dr[i];
        if(nc >= 0 && nc < 9 && nr >= 0 && nr < 9 && distMap[nc][nr] == -1) {
          Loc nextLoc = Location::pawnLoc(nc, nr, board.x_size);
          if(board.canPawnStep(currLoc, nextLoc)) {
            distMap[nc][nr] = d + 1;
            qC[qTail] = nc;
            qR[qTail] = nr;
            qTail++;
          }
        }
      }
    }
  };

  // Black goal row on board is r=0; White goal row on board is r=8
  int curGoalR = (nextPlayer == P_BLACK) ? 0 : 8;
  int oppGoalR = (nextPlayer == P_BLACK) ? 8 : 0;

  int distToGoalCur[9][9];
  int distToGoalOpp[9][9];
  int distFromPawnCur[9][9];
  int distFromPawnOpp[9][9];

  std::vector<std::pair<int,int>> curGoalSources;
  std::vector<std::pair<int,int>> oppGoalSources;
  curGoalSources.reserve(9);
  oppGoalSources.reserve(9);
  for(int c = 0; c < 9; c++) {
    curGoalSources.push_back({c, curGoalR});
    oppGoalSources.push_back({c, oppGoalR});
  }

  runBFS(curGoalSources, distToGoalCur);
  runBFS(oppGoalSources, distToGoalOpp);
  runBFS({{curPawnC, curPawnR}}, distFromPawnCur);
  runBFS({{oppPawnC, oppPawnR}}, distFromPawnOpp);

  int shortestDistCur = distToGoalCur[curPawnC][curPawnR];
  int shortestDistOpp = distToGoalOpp[oppPawnC][oppPawnR];

  bool onPathCur[9][9];
  bool onPathOpp[9][9];
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      onPathCur[c][r] = (shortestDistCur >= 0 && distFromPawnCur[c][r] >= 0 && distToGoalCur[c][r] >= 0 &&
                         (distFromPawnCur[c][r] + distToGoalCur[c][r] == shortestDistCur));
      onPathOpp[c][r] = (shortestDistOpp >= 0 && distFromPawnOpp[c][r] >= 0 && distToGoalOpp[c][r] >= 0 &&
                         (distFromPawnOpp[c][r] + distToGoalOpp[c][r] == shortestDistOpp));
    }
  }

  // Populate 17 spatial channels in canonical perspective
  for(int rCanon = 0; rCanon < 9; rCanon++) {
    for(int c = 0; c < 9; c++) {
      int rBoard = (nextPlayer == P_WHITE) ? (8 - rCanon) : rCanon;
      int cBoard = c;
      int pos = NNPos::xyToPos(c, rCanon, nnXLen);

      // Ch 0: On-board mask
      setRowBin(rowSpatial, pos, 0, 1.0f, posStride, featureStride);

      // Ch 1: Current player pawn
      if(cBoard == curPawnC && rBoard == curPawnR)
        setRowBin(rowSpatial, pos, 1, 1.0f, posStride, featureStride);

      // Ch 2: Opponent player pawn
      if(cBoard == oppPawnC && rBoard == oppPawnR)
        setRowBin(rowSpatial, pos, 2, 1.0f, posStride, featureStride);

      // Ch 3: North-blocked edge (swapped with South if White)
      bool bN = (nextPlayer == P_WHITE) ? blockedS[cBoard][rBoard] : blockedN[cBoard][rBoard];
      if(bN)
        setRowBin(rowSpatial, pos, 3, 1.0f, posStride, featureStride);

      // Ch 4: South-blocked edge (swapped with North if White)
      bool bS = (nextPlayer == P_WHITE) ? blockedN[cBoard][rBoard] : blockedS[cBoard][rBoard];
      if(bS)
        setRowBin(rowSpatial, pos, 4, 1.0f, posStride, featureStride);

      // Ch 5: East-blocked edge
      if(blockedE[cBoard][rBoard])
        setRowBin(rowSpatial, pos, 5, 1.0f, posStride, featureStride);

      // Ch 6: West-blocked edge
      if(blockedW[cBoard][rBoard])
        setRowBin(rowSpatial, pos, 6, 1.0f, posStride, featureStride);

      // Ch 7: Goal row mask (always rCanon == 0)
      if(rCanon == 0)
        setRowBin(rowSpatial, pos, 7, 1.0f, posStride, featureStride);

      // Ch 8: Current player goal BFS distance / 32.0f
      int dCur = distToGoalCur[cBoard][rBoard];
      float fCur = (dCur < 0) ? 1.0f : std::min(1.0f, (float)dCur / 32.0f);
      setRowBin(rowSpatial, pos, 8, fCur, posStride, featureStride);

      // Ch 9: Opponent goal BFS distance / 32.0f
      int dOpp = distToGoalOpp[cBoard][rBoard];
      float fOpp = (dOpp < 0) ? 1.0f : std::min(1.0f, (float)dOpp / 32.0f);
      setRowBin(rowSpatial, pos, 9, fOpp, posStride, featureStride);

      // Ch 10: Current player pawn BFS distance / 32.0f
      int dpCur = distFromPawnCur[cBoard][rBoard];
      float fpCur = (dpCur < 0) ? 1.0f : std::min(1.0f, (float)dpCur / 32.0f);
      setRowBin(rowSpatial, pos, 10, fpCur, posStride, featureStride);

      // Ch 11: Opponent pawn BFS distance / 32.0f
      int dpOpp = distFromPawnOpp[cBoard][rBoard];
      float fpOpp = (dpOpp < 0) ? 1.0f : std::min(1.0f, (float)dpOpp / 32.0f);
      setRowBin(rowSpatial, pos, 11, fpOpp, posStride, featureStride);

      // Ch 12: Current player on-path mask
      if(onPathCur[cBoard][rBoard])
        setRowBin(rowSpatial, pos, 12, 1.0f, posStride, featureStride);

      // Ch 13: Opponent on-path mask
      if(onPathOpp[cBoard][rBoard])
        setRowBin(rowSpatial, pos, 13, 1.0f, posStride, featureStride);

      // Ch 14 & 15: Placed wall anchors
      if(c < 8 && rCanon < 8) {
        int rWallBoard = (nextPlayer == P_WHITE) ? (7 - rCanon) : rCanon;
        if(hasVWall[c][rWallBoard])
          setRowBin(rowSpatial, pos, 14, 1.0f, posStride, featureStride);
        if(hasHWall[c][rWallBoard])
          setRowBin(rowSpatial, pos, 15, 1.0f, posStride, featureStride);
      }

      // Ch 16: Valid wall anchor domain mask (1.0f on [0..7]x[0..7], 0.0f elsewhere)
      if(c < 8 && rCanon < 8) {
        setRowBin(rowSpatial, pos, 16, 1.0f, posStride, featureStride);
      }
    }
  }

  // Populate 15 global features
  // Index 0: Next player is White
  rowGlobal[0] = (nextPlayer == P_WHITE) ? 1.0f : 0.0f;

  int myFences = (nextPlayer == P_WHITE) ? board.whiteFences : board.blackFences;
  int oppFences = (nextPlayer == P_WHITE) ? board.blackFences : board.whiteFences;

  // Index 1: My fence count remaining
  rowGlobal[1] = (float)myFences / 10.0f;

  // Index 2: Opponent fence count remaining
  rowGlobal[2] = (float)oppFences / 10.0f;

  // Index 3..6: My fence exponential encoding
  if(myFences > 0) {
    float d = (float)(myFences - 1);
    rowGlobal[3] = std::exp(-d / 1.0f);
    rowGlobal[4] = std::exp(-d / 2.0f);
    rowGlobal[5] = std::exp(-d / 4.0f);
    rowGlobal[6] = std::exp(-d / 8.0f);
  }

  // Index 7: Opponent fence count present
  rowGlobal[7] = (oppFences >= 1) ? 1.0f : 0.0f;

  // Index 8..11: Opponent fence exponential encoding
  if(oppFences > 0) {
    float d = (float)(oppFences - 1);
    rowGlobal[8] = std::exp(-d / 1.0f);
    rowGlobal[9] = std::exp(-d / 2.0f);
    rowGlobal[10] = std::exp(-d / 4.0f);
    rowGlobal[11] = std::exp(-d / 8.0f);
  }

  // Index 12: Action Parity (Jump Tempo)
  // Evaluates whether nextPlayer has the jump tempo if both pawns advance directly:
  // If Manhattan distance between pawns is odd, nextPlayer reaches adjacency on opponent's turn and jumps (+1.0f).
  // If Manhattan distance is even, opponent reaches adjacency on nextPlayer's turn and opponent jumps (-1.0f).
  // / 2 because Location::getX/Y gives coordination on 17x17 board
  int c1 = Location::getX(board.blackPawnLoc, board.x_size) / 2;
  int r1 = Location::getY(board.blackPawnLoc, board.x_size) / 2;
  int c2 = Location::getX(board.whitePawnLoc, board.x_size) / 2;
  int r2 = Location::getY(board.whitePawnLoc, board.x_size) / 2;
  int manhattanDist = std::abs(c1 - c2) + std::abs(r1 - r2);
  rowGlobal[12] = (manhattanDist % 2 != 0) ? 1.0f : -1.0f;

  // Index 13: My shortest distance
  rowGlobal[13] = (shortestDistCur < 0) ? 1.0f : std::min(1.0f, (float)shortestDistCur / 32.0f);

  // Index 14: Opponent shortest distance
  rowGlobal[14] = (shortestDistOpp < 0) ? 1.0f : std::min(1.0f, (float)shortestDistOpp / 32.0f);
}

void QuoridorNN::applyInputSymmetry(float* rowSpatial, int ioVersion, bool useNHWC, int symmetry) {
  testAssert(ioVersion == 1);
  bool flipX = (symmetry % 2 != 0);
  if(!flipX)
    return;

  constexpr int H = MODEL_LEN;
  constexpr int W = MODEL_LEN;
  constexpr int C = NUM_FEATURES_SPATIAL_V1;

  auto getIdx = [&](int ch, int r, int c) {
    return useNHWC ? (r * W * C + c * C + ch) : (ch * H * W + r * W + c);
  };

  vector<float> src(rowSpatial, rowSpatial + H * W * C);

  // Standard channels: pawn, blocked N/S, distances, goal mask, etc. Mirrored across the full
  // 9-wide pawn-cell domain.
  static const int stdChannels[] = {0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13};
  for(int ch : stdChannels) {
    for(int r = 0; r < H; r++) {
      for(int c = 0; c < W; c++) {
        rowSpatial[getIdx(ch, r, 8 - c)] = src[getIdx(ch, r, c)];
      }
    }
  }

  // East-blocked (Ch 5) and West-blocked (Ch 6): mirroring the board turns east into west, so
  // these swap identities as well as flipping.
  for(int r = 0; r < H; r++) {
    for(int c = 0; c < W; c++) {
      rowSpatial[getIdx(6, r, 8 - c)] = src[getIdx(5, r, c)];
      rowSpatial[getIdx(5, r, 8 - c)] = src[getIdx(6, r, c)];
    }
  }

  // Wall channels (Ch 14: V-wall, Ch 15: H-wall). Active anchor range is [0..7]x[0..7], mirrored
  // with (7 - c); row 8 and column 8 stay 0.
  for(int ch : {14, 15}) {
    for(int r = 0; r < 8; r++) {
      for(int c = 0; c < 8; c++) {
        rowSpatial[getIdx(ch, r, 7 - c)] = src[getIdx(ch, r, c)];
      }
      rowSpatial[getIdx(ch, r, 8)] = 0.0f;
    }
    for(int c = 0; c < W; c++) {
      rowSpatial[getIdx(ch, 8, c)] = 0.0f;
    }
  }

  // Ch 16: Wall domain mask - invariant under horizontal reflection, no change needed, but keep
  // the copy for clarity/robustness in case the mask is ever made asymmetric.
  for(int r = 0; r < H; r++) {
    for(int c = 0; c < W; c++) {
      rowSpatial[getIdx(16, r, c)] = src[getIdx(16, r, c)];
    }
  }
}

void QuoridorNN::mapPolicyToSearch(
  const float* rawPolicy,
  Player nextPlayer,
  float* policyProbs290,
  int symmetry
) {
  bool flipX = (symmetry % 2 != 0);

  for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++) {
    policyProbs290[i] = -1e30f;
  }

  // Plane 0: Pawn moves to (c, r) - horizontal reflection in [0..8]
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int rCanon = (nextPlayer == P_WHITE) ? (8 - r) : r;
      int cCanon = flipX ? (8 - c) : c;
      int pos = NNPos::locToPos(Location::pawnLoc(c, r, 17), 17, 17, 17);
      policyProbs290[pos] = rawPolicy[0 * 81 + rCanon * 9 + cCanon];
    }
  }

  // Plane 1: Vertical walls at (c, r) [Upper arm] - horizontal reflection in [0..7]
  for(int r = 0; r < 8; r++) {
    for(int c = 0; c < 8; c++) {
      int rCanon = (nextPlayer == P_WHITE) ? (7 - r) : r;
      int cCanon = flipX ? (7 - c) : c;
      int pos = NNPos::locToPos(Location::vWallLoc(c, r, 17), 17, 17, 17);
      policyProbs290[pos] = rawPolicy[1 * 81 + rCanon * 9 + cCanon];
    }
  }

  // Plane 2: Horizontal walls at (c, r) [Center] - horizontal reflection in [0..7]
  for(int r = 0; r < 8; r++) {
    for(int c = 0; c < 8; c++) {
      int rCanon = (nextPlayer == P_WHITE) ? (7 - r) : r;
      int cCanon = flipX ? (7 - c) : c;
      int pos = NNPos::locToPos(Location::hWallLoc(c, r, 17), 17, 17, 17);
      policyProbs290[pos] = rawPolicy[2 * 81 + rCanon * 9 + cCanon];
    }
  }
}
