#include "../tests/tests.h"

#include <cmath>
#include <iostream>
#include <vector>

#include "../core/test.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/rules.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/quoridornn.h"

using namespace std;

static void testInitialBoardSpatialFeatures() {
  cout << "Running testInitialBoardSpatialFeatures..." << endl;
  Board board;
  Rules rules;
  BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
  MiscNNInputParams params;

  float rowSpatial[QuoridorNN::MAX_NUM_FEATURES_SPATIAL * 81];
  float rowGlobal[QuoridorNN::MAX_NUM_FEATURES_GLOBAL];

  QuoridorNN::fillRow(board, hist, P_BLACK, params, QuoridorNN::MAX_SUPPORTED_IO_VERSION, false, rowSpatial, rowGlobal);

  auto getSpatial = [&](int c, int pos) -> float {
    return rowSpatial[c * 81 + pos];
  };

  // Ch 0: On-board mask == 1.0f everywhere
  for(int pos = 0; pos < 81; pos++) {
    testAssert(getSpatial(0, pos) == 1.0f);
  }

  // Black starts at (4, 8), White at (4, 0)
  // Ch 1: Current player pawn (Black) at (4, 8)
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = r * 9 + c;
      if(c == 4 && r == 8) testAssert(getSpatial(1, pos) == 1.0f);
      else testAssert(getSpatial(1, pos) == 0.0f);
    }
  }

  // Ch 2: Opponent pawn (White) at (4, 0)
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = r * 9 + c;
      if(c == 4 && r == 0) testAssert(getSpatial(2, pos) == 1.0f);
      else testAssert(getSpatial(2, pos) == 0.0f);
    }
  }

  // Ch 3: North-blocked (only row 0 blocked on empty board)
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = r * 9 + c;
      if(r == 0) testAssert(getSpatial(3, pos) == 1.0f);
      else testAssert(getSpatial(3, pos) == 0.0f);
    }
  }

  // Ch 4: South-blocked (only row 8 blocked on empty board)
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = r * 9 + c;
      if(r == 8) testAssert(getSpatial(4, pos) == 1.0f);
      else testAssert(getSpatial(4, pos) == 0.0f);
    }
  }

  // Ch 5: East-blocked (only col 8 blocked on empty board)
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = r * 9 + c;
      if(c == 8) testAssert(getSpatial(5, pos) == 1.0f);
      else testAssert(getSpatial(5, pos) == 0.0f);
    }
  }

  // Ch 6: West-blocked (only col 0 blocked on empty board)
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = r * 9 + c;
      if(c == 0) testAssert(getSpatial(6, pos) == 1.0f);
      else testAssert(getSpatial(6, pos) == 0.0f);
    }
  }

  // Ch 7: Goal row mask (r == 0)
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = r * 9 + c;
      if(r == 0) testAssert(getSpatial(7, pos) == 1.0f);
      else testAssert(getSpatial(7, pos) == 0.0f);
    }
  }

  // Ch 8: Cur goal distance / 32.0f (dist to row 0 is r)
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = r * 9 + c;
      testAssert(abs(getSpatial(8, pos) - (float)r / 32.0f) < 1e-6);
    }
  }

  // Ch 9: Opp goal distance / 32.0f (dist to row 8 is 8 - r)
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = r * 9 + c;
      testAssert(abs(getSpatial(9, pos) - (float)(8 - r) / 32.0f) < 1e-6);
    }
  }

  // Ch 10: Cur pawn distance (from (4, 8))
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = r * 9 + c;
      float expected = (float)(abs(c - 4) + abs(r - 8)) / 32.0f;
      testAssert(abs(getSpatial(10, pos) - expected) < 1e-6);
    }
  }

  // Ch 11: Opp pawn distance (from (4, 0))
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = r * 9 + c;
      float expected = (float)(abs(c - 4) + abs(r - 0)) / 32.0f;
      testAssert(abs(getSpatial(11, pos) - expected) < 1e-6);
    }
  }

  // Ch 12: Cur on-path mask (on empty board, only straight line c=4 is shortest path from (4, 8) to row 0)
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = r * 9 + c;
      if(c == 4) testAssert(getSpatial(12, pos) == 1.0f);
      else testAssert(getSpatial(12, pos) == 0.0f);
    }
  }

  // Ch 13: Opp on-path mask (only c=4 is shortest path from (4, 0) to row 8)
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = r * 9 + c;
      if(c == 4) testAssert(getSpatial(13, pos) == 1.0f);
      else testAssert(getSpatial(13, pos) == 0.0f);
    }
  }

  // Ch 14 & 15: Wall anchors == 0.0f
  for(int pos = 0; pos < 81; pos++) {
    testAssert(getSpatial(14, pos) == 0.0f);
    testAssert(getSpatial(15, pos) == 0.0f);
  }

  // Ch 16: Wall anchor domain mask (1.0f on [0..7]x[0..7], 0.0f elsewhere)
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = r * 9 + c;
      if(c < 8 && r < 8) testAssert(getSpatial(16, pos) == 1.0f);
      else testAssert(getSpatial(16, pos) == 0.0f);
    }
  }
}

static void testCanonicalYFlipInvariance() {
  cout << "Running testCanonicalYFlipInvariance..." << endl;
  Board board;
  Rules rules;
  BoardHistory histBlack(board, P_BLACK, rules, 0, BoardHistoryModes());
  BoardHistory histWhite(board, P_WHITE, rules, 0, BoardHistoryModes());
  MiscNNInputParams params;

  float spatialBlack[QuoridorNN::MAX_NUM_FEATURES_SPATIAL * 81];
  float globalBlack[QuoridorNN::MAX_NUM_FEATURES_GLOBAL];
  float spatialWhite[QuoridorNN::MAX_NUM_FEATURES_SPATIAL * 81];
  float globalWhite[QuoridorNN::MAX_NUM_FEATURES_GLOBAL];

  QuoridorNN::fillRow(board, histBlack, P_BLACK, params, QuoridorNN::MAX_SUPPORTED_IO_VERSION, false, spatialBlack, globalBlack);
  QuoridorNN::fillRow(board, histWhite, P_WHITE, params, QuoridorNN::MAX_SUPPORTED_IO_VERSION, false, spatialWhite, globalWhite);

  // On an empty board, the canonical perspective makes Black to move and White to move completely symmetric!
  for(int ch = 0; ch < QuoridorNN::numSpatialFeatures(QuoridorNN::MAX_SUPPORTED_IO_VERSION); ch++) {
    for(int pos = 0; pos < 81; pos++) {
      float bVal = spatialBlack[ch * 81 + pos];
      float wVal = spatialWhite[ch * 81 + pos];
      testAssert(abs(bVal - wVal) < 1e-6);
    }
  }
}

static void testBFSDistanceFieldsAndOnPathMasks() {
  cout << "Running testBFSDistanceFieldsAndOnPathMasks..." << endl;
  Board board;
  Rules rules;
  // Place a horizontal wall at (4, 4)
  Loc hWall = Location::hWallLoc(4, 4, board.x_size);
  testAssert(board.isLegal(hWall, P_BLACK));
  board.playMoveAssumeLegal(hWall, P_BLACK);

  BoardHistory hist(board, P_WHITE, rules, 0, BoardHistoryModes());
  MiscNNInputParams params;

  float spatial[QuoridorNN::MAX_NUM_FEATURES_SPATIAL * 81];
  float global[QuoridorNN::MAX_NUM_FEATURES_GLOBAL];
  QuoridorNN::fillRow(board, hist, P_BLACK, params, QuoridorNN::MAX_SUPPORTED_IO_VERSION, false, spatial, global);

  auto getSpatial = [&](int c, int pos) -> float {
    return spatial[c * 81 + pos];
  };

  // Horizontal wall at (4, 4) blocks South between row 4 and row 5 for columns 4 and 5
  int pos44 = 4 * 9 + 4;
  int pos54 = 4 * 9 + 5;
  testAssert(getSpatial(4, pos44) == 1.0f); // South blocked at (4, 4)
  testAssert(getSpatial(4, pos54) == 1.0f); // South blocked at (5, 4)

  int pos45 = 5 * 9 + 4;
  int pos55 = 5 * 9 + 5;
  testAssert(getSpatial(3, pos45) == 1.0f); // North blocked at (4, 5)
  testAssert(getSpatial(3, pos55) == 1.0f); // North blocked at (5, 5)

  // Wall anchors: horizontal wall at (4, 4) in canonical view
  int posHWall = 4 * 9 + 4;
  testAssert(getSpatial(15, posHWall) == 1.0f);
  // No vertical wall placed
  for(int pos = 0; pos < 81; pos++) {
    testAssert(getSpatial(14, pos) == 0.0f);
  }

  // Shortest path for Black from (4, 8) must now go around the wall at (4, 4) and (5, 4)
  // Distance from (4, 8) to row 0 was 8; with wall blocking (4, 4)-(4, 5) and (5, 4)-(5, 5),
  // path detours via c=3 (1 horizontal step) to reach row 0 at (3, 0) -> 9 steps.
  float curGoalDistAtPawn = getSpatial(8, 8 * 9 + 4);
  testAssert(abs(curGoalDistAtPawn - (9.0f / 32.0f)) < 1e-6);

  // Detour path via column 3 is on shortest path
  testAssert(getSpatial(12, 5 * 9 + 3) == 1.0f); // (3, 5) on path
  testAssert(getSpatial(12, 4 * 9 + 3) == 1.0f); // (3, 4) on path
  // Detour via column 6 takes 2 horizontal steps (10 steps total), so it is not on shortest path
  testAssert(getSpatial(12, 5 * 9 + 6) == 0.0f);
  // (4, 4) should NOT be on-path from (4, 8) since stepping into the dead end adds extra steps
  testAssert(getSpatial(12, 4 * 9 + 4) == 0.0f);
}

static void testActionParityInvariance() {
  cout << "Running testActionParityInvariance..." << endl;
  Board board;
  Rules rules;
  BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
  MiscNNInputParams params;

  float spatial[QuoridorNN::MAX_NUM_FEATURES_SPATIAL * 81];
  float global[QuoridorNN::MAX_NUM_FEATURES_GLOBAL];

  // 1. Initial position: Black at (4, 8), White at (4, 0).
  // dist = |4 - 4| + |8 - 0| = 8 (even) -> Black (nextPlayer) does NOT have jump tempo -> -1.0f
  QuoridorNN::fillRow(board, hist, P_BLACK, params, QuoridorNN::MAX_SUPPORTED_IO_VERSION, false, spatial, global);
  testAssert(global[12] == -1.0f);

  // 2. Black moves pawn North to (4, 7): dist = 7 (odd)
  // White is nextPlayer -> White HAS jump tempo -> +1.0f
  Loc m1 = Location::pawnLoc(4, 7, board.x_size);
  hist.makeBoardMoveAssumeLegal(board, m1, P_BLACK, NULL);
  QuoridorNN::fillRow(board, hist, P_WHITE, params, QuoridorNN::MAX_SUPPORTED_IO_VERSION, false, spatial, global);
  testAssert(global[12] == 1.0f);

  // 3. White moves pawn South to (4, 1): dist = 6 (even)
  // Black is nextPlayer -> Black does NOT have jump tempo -> -1.0f
  Loc m2 = Location::pawnLoc(4, 1, board.x_size);
  hist.makeBoardMoveAssumeLegal(board, m2, P_WHITE, NULL);
  QuoridorNN::fillRow(board, hist, P_BLACK, params, QuoridorNN::MAX_SUPPORTED_IO_VERSION, false, spatial, global);
  testAssert(global[12] == -1.0f);

  // 4. Black places a horizontal wall: pawns unchanged, dist = 6 (even)
  // White is nextPlayer -> dist is even, so White does NOT have jump tempo -> -1.0f
  Loc m3 = Location::hWallLoc(0, 0, board.x_size);
  hist.makeBoardMoveAssumeLegal(board, m3, P_BLACK, NULL);
  QuoridorNN::fillRow(board, hist, P_WHITE, params, QuoridorNN::MAX_SUPPORTED_IO_VERSION, false, spatial, global);
  testAssert(global[12] == -1.0f);

  // 5. White places another wall: dist = 6 (even)
  // Black is nextPlayer -> dist is even -> Black does NOT have jump tempo -> -1.0f
  Loc m4 = Location::vWallLoc(7, 7, board.x_size);
  hist.makeBoardMoveAssumeLegal(board, m4, P_WHITE, NULL);
  QuoridorNN::fillRow(board, hist, P_BLACK, params, QuoridorNN::MAX_SUPPORTED_IO_VERSION, false, spatial, global);
  testAssert(global[12] == -1.0f);

  // 6. Black moves pawn to (4, 6): dist = 5 (odd)
  // White is nextPlayer -> dist is odd -> White HAS jump tempo -> +1.0f
  Loc m5 = Location::pawnLoc(4, 6, board.x_size);
  hist.makeBoardMoveAssumeLegal(board, m5, P_BLACK, NULL);
  QuoridorNN::fillRow(board, hist, P_WHITE, params, QuoridorNN::MAX_SUPPORTED_IO_VERSION, false, spatial, global);
  testAssert(global[12] == 1.0f);
}

static void testGlobalFeaturesValues() {
  cout << "Running testGlobalFeaturesValues..." << endl;
  Board board;
  Rules rules;
  BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
  MiscNNInputParams params;

  float spatial[QuoridorNN::MAX_NUM_FEATURES_SPATIAL * 81];
  float global[QuoridorNN::MAX_NUM_FEATURES_GLOBAL];

  QuoridorNN::fillRow(board, hist, P_BLACK, params, QuoridorNN::MAX_SUPPORTED_IO_VERSION, false, spatial, global);

  testAssert(global[0] == 0.0f); // Next player is Black
  testAssert(global[1] == 1.0f); // My fences = 10 / 10 = 1.0
  testAssert(global[2] == 1.0f); // Opp fences = 10 / 10 = 1.0
  testAssert(abs(global[3] - exp(-9.0f / 1.0f)) < 1e-6);
  testAssert(abs(global[4] - exp(-9.0f / 2.0f)) < 1e-6);
  testAssert(abs(global[5] - exp(-9.0f / 4.0f)) < 1e-6);
  testAssert(abs(global[6] - exp(-9.0f / 8.0f)) < 1e-6);
  testAssert(global[7] == 1.0f); // Opponent fence present
  testAssert(abs(global[8] - exp(-9.0f / 1.0f)) < 1e-6);
  testAssert(abs(global[9] - exp(-9.0f / 2.0f)) < 1e-6);
  testAssert(abs(global[10] - exp(-9.0f / 4.0f)) < 1e-6);
  testAssert(abs(global[11] - exp(-9.0f / 8.0f)) < 1e-6);
  testAssert(global[12] == -1.0f); // Parity: at start Black has even distance to White, White has tempo, Black is -1.0f
  testAssert(abs(global[13] - (8.0f / 32.0f)) < 1e-6); // My shortest dist
  testAssert(abs(global[14] - (8.0f / 32.0f)) < 1e-6); // Opp shortest dist

  // Exhaust all fences
  board.blackFences = 0;
  board.whiteFences = 0;
  QuoridorNN::fillRow(board, hist, P_BLACK, params, QuoridorNN::MAX_SUPPORTED_IO_VERSION, false, spatial, global);
  testAssert(global[1] == 0.0f);
  testAssert(global[2] == 0.0f);
  testAssert(global[3] == 0.0f);
  testAssert(global[4] == 0.0f);
  testAssert(global[5] == 0.0f);
  testAssert(global[6] == 0.0f);
  testAssert(global[7] == 0.0f); // Opponent fence present is 0
  testAssert(global[8] == 0.0f);
  testAssert(global[9] == 0.0f);
  testAssert(global[10] == 0.0f);
  testAssert(global[11] == 0.0f);
}

static void testPolicyScatterAndInverseMapping() {
  cout << "Running testPolicyScatterAndInverseMapping..." << endl;
  float rawPolicy[(QuoridorNN::NUM_POLICY_PLANES * QuoridorNN::MODEL_LEN * QuoridorNN::MODEL_LEN)];
  for(int i = 0; i < (QuoridorNN::NUM_POLICY_PLANES * QuoridorNN::MODEL_LEN * QuoridorNN::MODEL_LEN); i++) {
    rawPolicy[i] = 100.0f + (float)i;
  }

  float policyProbs[NNPos::MAX_NN_POLICY_SIZE];

  // 1. Black to move
  QuoridorNN::mapPolicyToSearch(rawPolicy, P_BLACK, policyProbs, 0);

  // Check Plane 0 (Pawn moves)
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = NNPos::locToPos(Location::pawnLoc(c, r, 17), 17, 17, 17);
      float expected = rawPolicy[0 * 81 + r * 9 + c];
      testAssert(policyProbs[pos] == expected);
    }
  }

  // Check Plane 1 (Vertical walls)
  for(int r = 0; r < 8; r++) {
    for(int c = 0; c < 8; c++) {
      int pos = NNPos::locToPos(Location::vWallLoc(c, r, 17), 17, 17, 17);
      float expected = rawPolicy[1 * 81 + r * 9 + c];
      testAssert(policyProbs[pos] == expected);
    }
  }

  // Check Plane 2 (Horizontal walls)
  for(int r = 0; r < 8; r++) {
    for(int c = 0; c < 8; c++) {
      int pos = NNPos::locToPos(Location::hWallLoc(c, r, 17), 17, 17, 17);
      float expected = rawPolicy[2 * 81 + r * 9 + c];
      testAssert(policyProbs[pos] == expected);
    }
  }

  // Pass move (289) must be -1e30f
  testAssert(policyProbs[289] == -1e30f);

  // 2. White to move (inversion)
  QuoridorNN::mapPolicyToSearch(rawPolicy, P_WHITE, policyProbs, 0);

  // For White: pawn (c, r) comes from rawPolicy at (c, 8 - r)
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int pos = NNPos::locToPos(Location::pawnLoc(c, r, 17), 17, 17, 17);
      float expected = rawPolicy[0 * 81 + (8 - r) * 9 + c];
      testAssert(policyProbs[pos] == expected);
    }
  }

  // Vertical wall (c, r) comes from rawPolicy at (c, 7 - r)
  for(int r = 0; r < 8; r++) {
    for(int c = 0; c < 8; c++) {
      int pos = NNPos::locToPos(Location::vWallLoc(c, r, 17), 17, 17, 17);
      float expected = rawPolicy[1 * 81 + (7 - r) * 9 + c];
      testAssert(policyProbs[pos] == expected);
    }
  }

  // Horizontal wall (c, r) comes from rawPolicy at (c, 7 - r)
  for(int r = 0; r < 8; r++) {
    for(int c = 0; c < 8; c++) {
      int pos = NNPos::locToPos(Location::hWallLoc(c, r, 17), 17, 17, 17);
      float expected = rawPolicy[2 * 81 + (7 - r) * 9 + c];
      testAssert(policyProbs[pos] == expected);
    }
  }
}

static void testMemoryLayoutNHWCvsNCHW() {
  cout << "Running testMemoryLayoutNHWCvsNCHW..." << endl;
  Board board;
  Rules rules;
  BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
  MiscNNInputParams params;

  float rowNCHW[QuoridorNN::MAX_NUM_FEATURES_SPATIAL * 81];
  float rowNHWC[QuoridorNN::MAX_NUM_FEATURES_SPATIAL * 81];
  float rowGlobalNCHW[QuoridorNN::MAX_NUM_FEATURES_GLOBAL];
  float rowGlobalNHWC[QuoridorNN::MAX_NUM_FEATURES_GLOBAL];

  QuoridorNN::fillRow(board, hist, P_BLACK, params, QuoridorNN::MAX_SUPPORTED_IO_VERSION, false, rowNCHW, rowGlobalNCHW);
  QuoridorNN::fillRow(board, hist, P_BLACK, params, QuoridorNN::MAX_SUPPORTED_IO_VERSION, true, rowNHWC, rowGlobalNHWC);

  const int numSpatial = QuoridorNN::numSpatialFeatures(QuoridorNN::MAX_SUPPORTED_IO_VERSION);
  for(int ch = 0; ch < numSpatial; ch++) {
    for(int pos = 0; pos < 81; pos++) {
      float nchwVal = rowNCHW[ch * 81 + pos];
      float nhwcVal = rowNHWC[pos * numSpatial + ch];
      testAssert(nchwVal == nhwcVal);
    }
  }

  for(int g = 0; g < QuoridorNN::numGlobalFeatures(QuoridorNN::MAX_SUPPORTED_IO_VERSION); g++) {
    testAssert(rowGlobalNCHW[g] == rowGlobalNHWC[g]);
  }
}

//------------------------------------------------------------------------------------------------
// Quoridor I/O v2 inputs

// Random games (uniform over legal moves, or a pawn move half the time), with random rules.
struct V2Position {
  Board board;
  BoardHistory hist;
  Player pla;
  vector<Loc> moves;  // from the empty board
};

static vector<V2Position> randomV2Positions(Rand& rand, int numPositions) {
  vector<V2Position> out;
  for(int game = 0; (int)out.size() < numPositions; game++) {
    Rules rules = Rules::getQuoridorRules();
    if(game % 2 == 1) {
      rules.komi = (float)(rand.nextInt(-20, 19) + 0.5);
      rules.maxPlies = rand.nextInt(1, 400);
    }
    Board board;
    Player pla = P_BLACK;
    BoardHistory hist(board, pla, rules, 0, BoardHistoryModes());
    vector<Loc> moves;
    while(!hist.isGameFinished && (int)out.size() < numPositions) {
      if(rand.nextBool(0.3))
        out.push_back(V2Position{board, hist, pla, moves});
      vector<Loc> legal;
      vector<Loc> pawn;
      for(int y = 0; y < board.y_size; y++) {
        for(int x = 0; x < board.x_size; x++) {
          Loc loc = Location::getLoc(x, y, board.x_size);
          if(hist.isLegal(board, loc, pla)) {
            legal.push_back(loc);
            if(Location::isPawnLoc(loc, board.x_size))
              pawn.push_back(loc);
          }
        }
      }
      testAssert(!legal.empty());
      Loc loc = rand.nextBool(0.5) ? pawn[rand.nextUInt((uint32_t)pawn.size())] : legal[rand.nextUInt((uint32_t)legal.size())];
      hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
      moves.push_back(loc);
      pla = getOpp(pla);
    }
  }
  return out;
}

static Loc mirrorLoc(Loc loc) {
  int x = Location::getX(loc, 17);
  int y = Location::getY(loc, 17);
  if(Location::isPawnLoc(loc, 17))
    return Location::pawnLoc(8 - x / 2, y / 2, 17);
  if(Location::isVWallLoc(loc, 17))
    return Location::vWallLoc(7 - (x - 1) / 2, y / 2, 17);
  testAssert(Location::isHWallLoc(loc, 17));
  return Location::hWallLoc(7 - (x - 1) / 2, (y - 1) / 2, 17);
}

static void testLegalWallPlanesV2() {
  cout << "Running testLegalWallPlanesV2..." << endl;
  Rand rand("testLegalWallPlanesV2");
  vector<V2Position> positions = randomV2Positions(rand, 400);
  MiscNNInputParams params;
  float spatial[QuoridorNN::MAX_NUM_FEATURES_SPATIAL * 81];
  float global[QuoridorNN::MAX_NUM_FEATURES_GLOBAL];
  int numLegal = 0;
  int numIllegal = 0;
  int numNoFences = 0;
  for(const V2Position& p : positions) {
    const Board& board = p.board;
    int myFences = p.pla == P_BLACK ? board.blackFences : board.whiteFences;
    numNoFences += myFences == 0 ? 1 : 0;
    // The side to move with at least one fence: the planes are exactly its legal wall moves.
    Board withFences = board;
    withFences.blackFences = std::max(withFences.blackFences, 1);
    withFences.whiteFences = std::max(withFences.whiteFences, 1);
    QuoridorNN::fillRow(board, p.hist, p.pla, params, 2, false, spatial, global);
    for(int rCanon = 0; rCanon < 9; rCanon++) {
      for(int c = 0; c < 9; c++) {
        float v = spatial[QuoridorNN::SPATIAL_LEGAL_VWALL_V2 * 81 + rCanon * 9 + c];
        float h = spatial[QuoridorNN::SPATIAL_LEGAL_HWALL_V2 * 81 + rCanon * 9 + c];
        if(c == 8 || rCanon == 8) {
          testAssert(v == 0.0f && h == 0.0f);
          continue;
        }
        int r = p.pla == P_WHITE ? 7 - rCanon : rCanon;
        bool vLegal = withFences.isLegalWallPlacement(c, r, true, p.pla);
        bool hLegal = withFences.isLegalWallPlacement(c, r, false, p.pla);
        testAssert(v == (vLegal ? 1.0f : 0.0f));
        testAssert(h == (hLegal ? 1.0f : 0.0f));
        // Same as the search's own legality check when the side to move has fences.
        if(myFences > 0) {
          testAssert(vLegal == p.hist.isLegal(board, Location::vWallLoc(c, r, 17), p.pla));
          testAssert(hLegal == p.hist.isLegal(board, Location::hWallLoc(c, r, 17), p.pla));
        }
        numLegal += (vLegal ? 1 : 0) + (hLegal ? 1 : 0);
        numIllegal += (vLegal ? 0 : 1) + (hLegal ? 0 : 1);
      }
    }
    // v1 rows are the first 17 / 15 features of v2 rows.
    float spatialV1[QuoridorNN::NUM_FEATURES_SPATIAL_V1 * 81];
    float globalV1[QuoridorNN::NUM_FEATURES_GLOBAL_V1];
    QuoridorNN::fillRow(board, p.hist, p.pla, params, 1, false, spatialV1, globalV1);
    for(int i = 0; i < QuoridorNN::NUM_FEATURES_SPATIAL_V1 * 81; i++)
      testAssert(spatialV1[i] == spatial[i]);
    for(int i = 0; i < QuoridorNN::NUM_FEATURES_GLOBAL_V1; i++)
      testAssert(globalV1[i] == global[i]);
  }
  testAssert(numLegal > 0 && numIllegal > 0 && numNoFences > 0);
  cout << "  " << positions.size() << " positions, " << numLegal << " legal and " << numIllegal
       << " illegal anchors, " << numNoFences << " positions with no fences for the side to move" << endl;
}

static void testMirrorSymmetryV2() {
  cout << "Running testMirrorSymmetryV2..." << endl;
  Rand rand("testMirrorSymmetryV2");
  vector<V2Position> positions = randomV2Positions(rand, 150);
  MiscNNInputParams params;
  for(int ioVersion = 1; ioVersion <= 2; ioVersion++) {
    const int C = QuoridorNN::numSpatialFeatures(ioVersion);
    const int G = QuoridorNN::numGlobalFeatures(ioVersion);
    for(bool useNHWC : {false, true}) {
      for(const V2Position& p : positions) {
        // The same game with every move mirrored left-right.
        Board mBoard;
        BoardHistory mHist(mBoard, P_BLACK, p.hist.rules, 0, BoardHistoryModes());
        Player pla = P_BLACK;
        for(Loc loc : p.moves) {
          Loc m = mirrorLoc(loc);
          testAssert(mHist.isLegal(mBoard, m, pla));
          mHist.makeBoardMoveAssumeLegal(mBoard, m, pla, NULL);
          pla = getOpp(pla);
        }
        testAssert(pla == p.pla);

        vector<float> spatial(C * 81), global(G), mSpatial(C * 81), mGlobal(G);
        QuoridorNN::fillRow(p.board, p.hist, p.pla, params, ioVersion, useNHWC, spatial.data(), global.data());
        QuoridorNN::fillRow(mBoard, mHist, pla, params, ioVersion, useNHWC, mSpatial.data(), mGlobal.data());
        QuoridorNN::applyInputSymmetry(spatial.data(), ioVersion, useNHWC, 1);
        for(int i = 0; i < C * 81; i++)
          testAssert(spatial[i] == mSpatial[i]);
        for(int i = 0; i < G; i++)
          testAssert(global[i] == mGlobal[i]);
      }
    }
  }
}

static void testGlobalFeaturesV2() {
  cout << "Running testGlobalFeaturesV2..." << endl;
  MiscNNInputParams params;
  float spatial[QuoridorNN::MAX_NUM_FEATURES_SPATIAL * 81];
  float global[QuoridorNN::MAX_NUM_FEATURES_GLOBAL];
  {
    // Standard game: 300 plies to go, komi -0.5 is +0.5 for Black and -0.5 for White.
    Board board;
    BoardHistory hist(board, P_BLACK, Rules::getQuoridorRules(), 0, BoardHistoryModes());
    QuoridorNN::fillRow(board, hist, P_BLACK, params, 2, false, spatial, global);
    testAssert(global[QuoridorNN::GLOBAL_PLIES_UNTIL_DRAW_V2] == 1.0f);
    testAssert(global[QuoridorNN::GLOBAL_SELF_KOMI_V2] == 0.1f);
    hist.makeBoardMoveAssumeLegal(board, Location::pawnLoc(4, 7, 17), P_BLACK, NULL);
    QuoridorNN::fillRow(board, hist, P_WHITE, params, 2, false, spatial, global);
    testAssert(global[QuoridorNN::GLOBAL_PLIES_UNTIL_DRAW_V2] == (float)(299.0 / 300.0));
    testAssert(global[QuoridorNN::GLOBAL_SELF_KOMI_V2] == -0.1f);
  }
  {
    // Other rules: the scales are absolute, not relative to maxPlies.
    Rules rules = Rules::getQuoridorRules();
    rules.maxPlies = 40;
    rules.komi = 2.5f;
    Board board;
    BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
    const Loc moves[] = {Location::pawnLoc(4, 7, 17), Location::pawnLoc(4, 1, 17), Location::hWallLoc(0, 0, 17)};
    Player pla = P_BLACK;
    for(Loc loc : moves) {
      hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
      pla = getOpp(pla);
    }
    testAssert(pla == P_WHITE);
    QuoridorNN::fillRow(board, hist, P_WHITE, params, 2, false, spatial, global);
    testAssert(global[QuoridorNN::GLOBAL_PLIES_UNTIL_DRAW_V2] == (float)(37.0 / 300.0));
    testAssert(global[QuoridorNN::GLOBAL_SELF_KOMI_V2] == 0.5f);
    QuoridorNN::fillRow(board, hist, P_BLACK, params, 2, false, spatial, global);
    testAssert(global[QuoridorNN::GLOBAL_SELF_KOMI_V2] == -0.5f);
    // Past the ply limit (a start position beyond it): 0, not negative.
    BoardHistory late(board, P_WHITE, rules, 0, BoardHistoryModes());
    late.initialTurnNumber = 50;
    QuoridorNN::fillRow(board, late, P_WHITE, params, 2, false, spatial, global);
    testAssert(global[QuoridorNN::GLOBAL_PLIES_UNTIL_DRAW_V2] == 0.0f);
  }
}

void Tests::runNNInputsTests() {
  cout << "Running NNInputs Quoridor Tests..." << endl;
  testInitialBoardSpatialFeatures();
  testCanonicalYFlipInvariance();
  testBFSDistanceFieldsAndOnPathMasks();
  testActionParityInvariance();
  testGlobalFeaturesValues();
  testPolicyScatterAndInverseMapping();
  testMemoryLayoutNHWCvsNCHW();
  testLegalWallPlanesV2();
  testMirrorSymmetryV2();
  testGlobalFeaturesV2();
  cout << "All NNInputs tests passed successfully!" << endl;
}
