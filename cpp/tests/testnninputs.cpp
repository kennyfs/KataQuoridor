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

using namespace std;

static void testInitialBoardSpatialFeatures() {
  cout << "Running testInitialBoardSpatialFeatures..." << endl;
  Board board;
  Rules rules;
  BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
  MiscNNInputParams params;

  float rowSpatial[NNInputs::NUM_FEATURES_SPATIAL_V1 * 81];
  float rowGlobal[NNInputs::NUM_FEATURES_GLOBAL_V1];

  NNInputs::fillRowV1(board, hist, P_BLACK, params, 9, 9, false, rowSpatial, rowGlobal);

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
}

static void testCanonicalYFlipInvariance() {
  cout << "Running testCanonicalYFlipInvariance..." << endl;
  Board board;
  Rules rules;
  BoardHistory histBlack(board, P_BLACK, rules, 0, BoardHistoryModes());
  BoardHistory histWhite(board, P_WHITE, rules, 0, BoardHistoryModes());
  MiscNNInputParams params;

  float spatialBlack[NNInputs::NUM_FEATURES_SPATIAL_V1 * 81];
  float globalBlack[NNInputs::NUM_FEATURES_GLOBAL_V1];
  float spatialWhite[NNInputs::NUM_FEATURES_SPATIAL_V1 * 81];
  float globalWhite[NNInputs::NUM_FEATURES_GLOBAL_V1];

  NNInputs::fillRowV1(board, histBlack, P_BLACK, params, 9, 9, false, spatialBlack, globalBlack);
  NNInputs::fillRowV1(board, histWhite, P_WHITE, params, 9, 9, false, spatialWhite, globalWhite);

  // On an empty board, the canonical perspective makes Black to move and White to move completely symmetric!
  for(int ch = 0; ch < NNInputs::NUM_FEATURES_SPATIAL_V1; ch++) {
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

  float spatial[NNInputs::NUM_FEATURES_SPATIAL_V1 * 81];
  float global[NNInputs::NUM_FEATURES_GLOBAL_V1];
  NNInputs::fillRowV1(board, hist, P_BLACK, params, 9, 9, false, spatial, global);

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

  float spatial[NNInputs::NUM_FEATURES_SPATIAL_V1 * 81];
  float global[NNInputs::NUM_FEATURES_GLOBAL_V1];

  // 1. Initial position: Black at (4, 8), White at (4, 0).
  // dist = |4 - 4| + |8 - 0| = 8 (even) -> Black (nextPlayer) does NOT have jump tempo -> -1.0f
  NNInputs::fillRowV1(board, hist, P_BLACK, params, 9, 9, false, spatial, global);
  testAssert(global[12] == -1.0f);

  // 2. Black moves pawn North to (4, 7): dist = 7 (odd)
  // White is nextPlayer -> White HAS jump tempo -> +1.0f
  Loc m1 = Location::pawnLoc(4, 7, board.x_size);
  hist.makeBoardMoveAssumeLegal(board, m1, P_BLACK, NULL);
  NNInputs::fillRowV1(board, hist, P_WHITE, params, 9, 9, false, spatial, global);
  testAssert(global[12] == 1.0f);

  // 3. White moves pawn South to (4, 1): dist = 6 (even)
  // Black is nextPlayer -> Black does NOT have jump tempo -> -1.0f
  Loc m2 = Location::pawnLoc(4, 1, board.x_size);
  hist.makeBoardMoveAssumeLegal(board, m2, P_WHITE, NULL);
  NNInputs::fillRowV1(board, hist, P_BLACK, params, 9, 9, false, spatial, global);
  testAssert(global[12] == -1.0f);

  // 4. Black places a horizontal wall: pawns unchanged, dist = 6 (even)
  // White is nextPlayer -> dist is even, so White does NOT have jump tempo -> -1.0f
  Loc m3 = Location::hWallLoc(0, 0, board.x_size);
  hist.makeBoardMoveAssumeLegal(board, m3, P_BLACK, NULL);
  NNInputs::fillRowV1(board, hist, P_WHITE, params, 9, 9, false, spatial, global);
  testAssert(global[12] == -1.0f);

  // 5. White places another wall: dist = 6 (even)
  // Black is nextPlayer -> dist is even -> Black does NOT have jump tempo -> -1.0f
  Loc m4 = Location::vWallLoc(7, 7, board.x_size);
  hist.makeBoardMoveAssumeLegal(board, m4, P_WHITE, NULL);
  NNInputs::fillRowV1(board, hist, P_BLACK, params, 9, 9, false, spatial, global);
  testAssert(global[12] == -1.0f);

  // 6. Black moves pawn to (4, 6): dist = 5 (odd)
  // White is nextPlayer -> dist is odd -> White HAS jump tempo -> +1.0f
  Loc m5 = Location::pawnLoc(4, 6, board.x_size);
  hist.makeBoardMoveAssumeLegal(board, m5, P_BLACK, NULL);
  NNInputs::fillRowV1(board, hist, P_WHITE, params, 9, 9, false, spatial, global);
  testAssert(global[12] == 1.0f);
}

static void testGlobalFeaturesValues() {
  cout << "Running testGlobalFeaturesValues..." << endl;
  Board board;
  Rules rules;
  BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
  MiscNNInputParams params;

  float spatial[NNInputs::NUM_FEATURES_SPATIAL_V1 * 81];
  float global[NNInputs::NUM_FEATURES_GLOBAL_V1];

  NNInputs::fillRowV1(board, hist, P_BLACK, params, 9, 9, false, spatial, global);

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
  testAssert(global[13] == 0.0f); // Move count 0
  testAssert(abs(global[14] - (8.0f / 32.0f)) < 1e-6); // My shortest dist
  testAssert(abs(global[15] - (8.0f / 32.0f)) < 1e-6); // Opp shortest dist

  // Exhaust all fences
  board.blackFences = 0;
  board.whiteFences = 0;
  NNInputs::fillRowV1(board, hist, P_BLACK, params, 9, 9, false, spatial, global);
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
  float rawPolicy[NNInputs::NN_POLICY_SIZE];
  for(int i = 0; i < NNInputs::NN_POLICY_SIZE; i++) {
    rawPolicy[i] = 100.0f + (float)i;
  }

  float policyProbs[NNPos::MAX_NN_POLICY_SIZE];

  // 1. Black to move
  NNInputs::applyPolicyMap(rawPolicy, P_BLACK, policyProbs);

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
  NNInputs::applyPolicyMap(rawPolicy, P_WHITE, policyProbs);

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

  float rowNCHW[NNInputs::NUM_FEATURES_SPATIAL_V1 * 81];
  float rowNHWC[NNInputs::NUM_FEATURES_SPATIAL_V1 * 81];
  float rowGlobalNCHW[NNInputs::NUM_FEATURES_GLOBAL_V1];
  float rowGlobalNHWC[NNInputs::NUM_FEATURES_GLOBAL_V1];

  NNInputs::fillRowV1(board, hist, P_BLACK, params, 9, 9, false, rowNCHW, rowGlobalNCHW);
  NNInputs::fillRowV1(board, hist, P_BLACK, params, 9, 9, true, rowNHWC, rowGlobalNHWC);

  for(int ch = 0; ch < NNInputs::NUM_FEATURES_SPATIAL_V1; ch++) {
    for(int pos = 0; pos < 81; pos++) {
      float nchwVal = rowNCHW[ch * 81 + pos];
      float nhwcVal = rowNHWC[pos * NNInputs::NUM_FEATURES_SPATIAL_V1 + ch];
      testAssert(nchwVal == nhwcVal);
    }
  }

  for(int g = 0; g < NNInputs::NUM_FEATURES_GLOBAL_V1; g++) {
    testAssert(rowGlobalNCHW[g] == rowGlobalNHWC[g]);
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
  cout << "All NNInputs tests passed successfully!" << endl;
}
