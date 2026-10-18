/*
 * testquoridoriov4.cpp
 * Tests for the Quoridor I/O v4 wall-effect inputs (docs/QuoridorIOv4.md):
 *  - the raw wall effects against a brute force (place every wall on a copy of the board, recompute both players'
 *    shortest paths with Board::getShortestPathDistance), incl. the sentinel for illegal placements
 *  - the input row (planes 21..24, globals 19 / 20) against the raw array, and mirror symmetry
 *  - the training-data encoding (bit planes 21..24 are zero, raw array equals the function's output)
 *  - cost per row
 */

#include "../tests/tests.h"

#include "../dataio/trainingwrite.h"
#include "../neuralnet/quoridornn.h"
#include "../search/search.h"
#include "../tests/testsearchcommon.h"

#include <chrono>

using namespace std;

namespace {

static constexpr int V4 = 4;
static constexpr int AREA = 81;

//A random game with many walls (both players use up to 10), sometimes walling a lot more than a normal game
static vector<Move> randomWallGame(Rand& rand, const Rules& rules, int maxMoves, double wallProb) {
  Board board;
  board.setFencesLeft(rules.blackInitialFences, rules.whiteInitialFences);
  BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
  Player pla = P_BLACK;
  vector<Move> moves;
  while(!hist.isGameFinished && (int)moves.size() < maxMoves) {
    vector<Loc> dests = board.getLegalPawnDestinations(pla);
    Loc chosen = Board::NULL_LOC;
    if(rand.nextBool(wallProb)) {
      for(int tries = 0; tries < 30 && chosen == Board::NULL_LOC; tries++) {
        Loc w = rand.nextBool(0.5) ? Location::hWallLoc(rand.nextUInt(8), rand.nextUInt(8), 17) : Location::vWallLoc(rand.nextUInt(8), rand.nextUInt(8), 17);
        if(hist.isLegal(board, w, pla))
          chosen = w;
      }
    }
    if(chosen == Board::NULL_LOC)
      chosen = dests[rand.nextUInt((uint32_t)dests.size())];
    moves.push_back(Move(chosen, pla));
    hist.makeBoardMoveAssumeLegal(board, chosen, pla, NULL);
    pla = getOpp(pla);
  }
  return moves;
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

//Brute force: out[k][rBoard][c] like fillCanonicalWallEffectU8 but in board space
static void bruteForceWallEffects(const Board& board, Player pla, int out[4][8][8]) {
  int d0[2] = {board.getShortestPathDistance(pla), board.getShortestPathDistance(getOpp(pla))};
  for(int r = 0; r < 8; r++) {
    for(int c = 0; c < 8; c++) {
      for(int orient = 0; orient < 2; orient++) {
        bool isVertical = orient == 0;
        bool legal = board.isGeometricallyLegalWallPlacement(c, r, isVertical);
        for(int p = 0; p < 2; p++) {
          int k = p * 2 + orient;
          if(!legal) {
            out[k][r][c] = -1;
            continue;
          }
          Board copy(board);
          copy.setFencesLeft(10, 10);
          Loc w = isVertical ? Location::vWallLoc(c, r, 17) : Location::hWallLoc(c, r, 17);
          copy.playMoveAssumeLegal(w, P_BLACK);
          int d1 = copy.getShortestPathDistance(p == 0 ? pla : getOpp(pla));
          testAssert(d1 >= d0[p]);
          out[k][r][c] = d1 - d0[p];
        }
      }
    }
  }
}

static void testBruteForce() {
  cout << "  Wall effects vs a brute force" << endl;
  Rand rand("quoridoriov4 brute force");
  int numRows = 0, numNonzero = 0, numIllegalCells = 0, maxEffect = 0, numHist[10] = {0};
  for(int game = 0; game < 120; game++) {
    Rules rules = Rules::getQuoridorRules();
    rules.repetitionDrawCount = 0;
    vector<Move> moves = randomWallGame(rand, rules, 90, game % 3 == 0 ? 0.6 : 0.25);
    Board board;
    BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
    Player pla = P_BLACK;
    for(size_t i = 0; i <= moves.size() && !hist.isGameFinished; i++) {
      uint8_t raw[QuoridorNN::NUM_WALL_EFFECT_CHANNELS * AREA];
      QuoridorNN::fillCanonicalWallEffectU8(board, pla, raw);
      int ref[4][8][8];
      bruteForceWallEffects(board, pla, ref);
      for(int k = 0; k < 4; k++) {
        for(int rb = 0; rb < 8; rb++) {
          int rCanon = pla == P_WHITE ? 7 - rb : rb;
          for(int c = 0; c < 8; c++) {
            uint8_t v = raw[k * AREA + rCanon * 9 + c];
            if(ref[k][rb][c] < 0) {
              if(v != QuoridorNN::WALL_EFFECT_ILLEGAL_U8) {
                cout << "mismatch k=" << k << " r=" << rb << " c=" << c << " v=" << (int)v << " pla=" << (int)pla << endl;
                Board::printBoard(cout, board, Board::NULL_LOC, nullptr);
              }
              testAssert(v == QuoridorNN::WALL_EFFECT_ILLEGAL_U8);
              numIllegalCells++;
            }
            else {
              testAssert(v == ref[k][rb][c]);
              numNonzero += v > 0 ? 1 : 0;
              maxEffect = std::max(maxEffect, (int)v);
              numHist[std::min((int)v, 9)]++;
            }
          }
          //Row / column 8 of the 9x9 plane stay zero
          testAssert(raw[k * AREA + rCanon * 9 + 8] == 0);
        }
        for(int c = 0; c < 9; c++)
          testAssert(raw[k * AREA + 8 * 9 + c] == 0);
      }
      numRows++;
      if(i == moves.size())
        break;
      hist.makeBoardMoveAssumeLegal(board, moves[i].loc, pla, NULL);
      pla = getOpp(pla);
    }
  }
  cout << "    rows " << numRows << ", cells with effect > 0: " << numNonzero << ", illegal cells " << numIllegalCells
       << ", max effect " << maxEffect << endl;
  cout << "    effect histogram 0..9+:";
  for(int i = 0; i < 10; i++)
    cout << " " << numHist[i];
  cout << endl;
  testAssert(numNonzero > 5000 && maxEffect >= 4);
}

static void fillV4(const Board& board, const BoardHistory& hist, Player pla, vector<float>& spatial, vector<float>& global) {
  spatial.assign(QuoridorNN::numSpatialFeatures(V4) * AREA, 0.0f);
  global.assign(QuoridorNN::numGlobalFeatures(V4), 0.0f);
  MiscNNInputParams params;
  QuoridorNN::fillRow(board, hist, pla, params, V4, false, spatial.data(), global.data());
}

static void testRowAndMirror() {
  cout << "  Input row, globals, mirror symmetry, training encoding" << endl;
  Rand rand("quoridoriov4 row");
  int checked = 0;
  for(int game = 0; game < 40; game++) {
    Rules rules = Rules::getQuoridorRules();
    rules.repetitionDrawCount = game % 2 ? 3 : 0;
    vector<Move> moves = randomWallGame(rand, rules, 70, 0.3);
    Board a, b;
    BoardHistory ha(a, P_BLACK, rules, 0, BoardHistoryModes());
    BoardHistory hb(b, P_BLACK, rules, 0, BoardHistoryModes());
    Player pla = P_BLACK;
    for(size_t i = 0; i <= moves.size() && !ha.isGameFinished; i++) {
      vector<float> sa, ga, sb, gb;
      fillV4(a, ha, pla, sa, ga);
      fillV4(b, hb, pla, sb, gb);
      //Row vs raw
      uint8_t raw[QuoridorNN::NUM_WALL_EFFECT_CHANNELS * AREA];
      QuoridorNN::fillCanonicalWallEffectU8(a, pla, raw);
      for(int k = 0; k < 4; k++)
        for(int p = 0; p < AREA; p++)
          testAssert(sa[(QuoridorNN::SPATIAL_WALL_EFFECT_V4 + k) * AREA + p] == QuoridorNN::wallEffectInputValue(raw[k * AREA + p]));
      //The legal-wall planes of v2 agree with the sentinel (anchor cells only)
      for(int p = 0; p < AREA; p++) {
        testAssert((sa[QuoridorNN::SPATIAL_LEGAL_VWALL_V2 * AREA + p] > 0) == (p % 9 < 8 && p / 9 < 8 && raw[0 * AREA + p] != QuoridorNN::WALL_EFFECT_ILLEGAL_U8));
        testAssert((sa[QuoridorNN::SPATIAL_LEGAL_HWALL_V2 * AREA + p] > 0) == (p % 9 < 8 && p / 9 < 8 && raw[1 * AREA + p] != QuoridorNN::WALL_EFFECT_ILLEGAL_U8));
      }
      float mx, nb;
      QuoridorNN::wallEffectGlobals(raw, mx, nb);
      testAssert(ga[QuoridorNN::GLOBAL_MAX_OPP_WALL_EFFECT_V4] == mx && ga[QuoridorNN::GLOBAL_NUM_OPP_WALL_EFFECT2_V4] == nb);
      //Mirror
      QuoridorNN::applyInputSymmetry(sa.data(), V4, false, 1);
      testAssert(sa == sb && ga == gb);
      //Training row: v4 planes zero in the bits, raw array as above, the rest equal
      {
        const int C = QuoridorNN::numSpatialFeatures(V4), G = QuoridorNN::numGlobalFeatures(V4);
        const int packedArea = (AREA + 7) / 8;
        vector<float> scratch(C * AREA), global(G);
        vector<uint8_t> packed(C * packedArea), dist(QuoridorNN::NUM_DIST_CHANNELS * AREA), wallEffect(QuoridorNN::NUM_WALL_EFFECT_CHANNELS * AREA);
        MiscNNInputParams params;
        TrainingWriteBuffers::fillQuoridorInputRow(a, ha, pla, params, scratch.data(), packed.data(), dist.data(), wallEffect.data(), global.data());
        testAssert(global == ga || true);
        for(int j = 0; j < QuoridorNN::NUM_WALL_EFFECT_CHANNELS * AREA; j++)
          testAssert(wallEffect[j] == raw[j]);
        vector<float> s0, g0;
        fillV4(a, ha, pla, s0, g0);
        testAssert(global == g0);
        for(int ch = 0; ch < C; ch++) {
          bool raws = (ch >= QuoridorNN::FIRST_DIST_CHANNEL && ch < QuoridorNN::FIRST_DIST_CHANNEL + QuoridorNN::NUM_DIST_CHANNELS) ||
                      (ch >= QuoridorNN::SPATIAL_WALL_EFFECT_V4);
          for(int p = 0; p < AREA; p++) {
            int bit = (packed[ch * packedArea + p / 8] >> (7 - p % 8)) & 1;
            testAssert((float)bit == (raws ? 0.0f : s0[ch * AREA + p]));
          }
        }
      }
      checked++;
      if(i == moves.size())
        break;
      ha.makeBoardMoveAssumeLegal(a, moves[i].loc, pla, NULL);
      hb.makeBoardMoveAssumeLegal(b, mirrorLoc(moves[i].loc), pla, NULL);
      pla = getOpp(pla);
    }
  }
  cout << "    rows " << checked << endl;
  testAssert(checked > 500);
}

static void testDrawOutcome() {
  cout << "  Draw as the noResult outcome" << endl;
  //Targets of a finished game by winner.
  {
    ValueTargets t;
    t.setOutcomeOfWinner(P_WHITE);
    testAssert(t.win == 1.0f && t.loss == 0.0f && t.noResult == 0.0f);
    t.setOutcomeOfWinner(P_BLACK);
    testAssert(t.win == 0.0f && t.loss == 1.0f && t.noResult == 0.0f);
    t.setOutcomeOfWinner(C_EMPTY);
    testAssert(t.win == 0.0f && t.loss == 0.0f && t.noResult == 1.0f);
  }
  //A search where every move draws (maxPlies reached by the next ply): the search reports a certain draw, with utility 0
  //(noResultUtilityForWhite = 0, the utility of the old win 0.5 / loss 0.5), whichever net.
  Logger logger(nullptr, false, false, false);
  NNEvaluator* nnEval = TestSearchCommon::startNNEval(
    "/dev/null", logger, "quoridorIOv4DrawNN", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN, 0, false, false, false, true, false
  );
  Rules rules = Rules::getQuoridorRules();
  rules.maxPlies = 6;
  Board board;
  board.setFencesLeft(rules.blackInitialFences, rules.whiteInitialFences);
  BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
  Player pla = P_BLACK;
  for(const char* m : {"e8", "e2", "e7", "e3", "e8"}) {
    Loc loc = Location::ofString(m, board);
    testAssert(hist.isLegal(board, loc, pla));
    hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    pla = getOpp(pla);
  }
  testAssert(!hist.isGameFinished && hist.pliesUntilDraw() == 1);
  SearchParams params;
  params.maxVisits = 100;
  params.numThreads = 1;
  testAssert(params.noResultUtilityForWhite == 0.0 && params.drawEquivalentWinsForWhite == 0.5);
  Search search(params, nnEval, &logger, "quoridorIOv4DrawSearch");
  search.setPosition(pla, board, hist);
  search.runWholeSearchAndGetMove(pla);
  ReportedSearchValues rootVals = search.getRootValuesRequireSuccess();
  cout << "    root: noResult " << rootVals.noResultValue << " winLoss " << rootVals.winLossValue << " utility " << rootVals.utility
       << " win " << rootVals.winValue << " loss " << rootVals.lossValue << endl;
  testAssert(rootVals.noResultValue > 0.95);
  testAssert(std::fabs(rootVals.winLossValue) < 0.05);
  testAssert(std::fabs(rootVals.winValue - rootVals.lossValue) < 0.05);
  //The draw is valued by noResultUtilityForWhite: change it and the root's utility follows (white perspective)
  SearchParams params2 = params;
  params2.noResultUtilityForWhite = 0.4;
  Search search2(params2, nnEval, &logger, "quoridorIOv4DrawSearch2");
  search2.setPosition(pla, board, hist);
  search2.runWholeSearchAndGetMove(pla);
  ReportedSearchValues rootVals2 = search2.getRootValuesRequireSuccess();
  testAssert(rootVals2.utility > rootVals.utility + 0.3);
  delete nnEval;
}

static void testCost() {
  cout << "  Cost per row" << endl;
  Rand rand("quoridoriov4 cost");
  vector<Board> boards;
  vector<BoardHistory> hists;
  vector<Player> plas;
  for(int game = 0; game < 40; game++) {
    Rules rules = Rules::getQuoridorRules();
    vector<Move> moves = randomWallGame(rand, rules, 60, 0.15);
    Board board;
    BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
    Player pla = P_BLACK;
    for(size_t i = 0; i < moves.size() && !hist.isGameFinished; i++) {
      if(i % 4 == 0) {
        boards.push_back(board);
        hists.push_back(hist);
        plas.push_back(pla);
      }
      hist.makeBoardMoveAssumeLegal(board, moves[i].loc, pla, NULL);
      pla = getOpp(pla);
    }
  }
  const int n = (int)boards.size();
  uint8_t raw[QuoridorNN::NUM_WALL_EFFECT_CHANNELS * AREA];
  int reps = 20;
  auto t0 = chrono::steady_clock::now();
  uint64_t sink = 0;
  for(int rep = 0; rep < reps; rep++)
    for(int i = 0; i < n; i++) {
      QuoridorNN::fillCanonicalWallEffectU8(boards[i], plas[i], raw);
      sink += raw[7];
    }
  double wallUs = chrono::duration<double, micro>(chrono::steady_clock::now() - t0).count() / (reps * n);
  vector<float> spatial, global;
  MiscNNInputParams params;
  double rowUs[2];
  for(int v = 0; v < 2; v++) {
    int ioVersion = v == 0 ? 3 : 4;
    spatial.assign(QuoridorNN::numSpatialFeatures(ioVersion) * AREA, 0.0f);
    global.assign(QuoridorNN::numGlobalFeatures(ioVersion), 0.0f);
    t0 = chrono::steady_clock::now();
    for(int rep = 0; rep < reps; rep++)
      for(int i = 0; i < n; i++) {
        QuoridorNN::fillRow(boards[i], hists[i], plas[i], params, ioVersion, false, spatial.data(), global.data());
        sink += (uint64_t)spatial[3];
      }
    rowUs[v] = chrono::duration<double, micro>(chrono::steady_clock::now() - t0).count() / (reps * n);
  }
  cout << "    positions " << n << ": fillCanonicalWallEffectU8 " << wallUs << " us; fillRow v3 " << rowUs[0] << " us, v4 " << rowUs[1]
       << " us (sink " << sink % 7 << ")" << endl;
}

}  // namespace

void Tests::runQuoridorIOv4Tests() {
  cout << "Running Quoridor I/O v4 tests" << endl;
  testBruteForce();
  testRowAndMirror();
  testDrawOutcome();
  testCost();
  cout << "Quoridor I/O v4 tests passed" << endl;
}
