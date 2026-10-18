/*
 * testquoridoriov4.cpp
 * Tests for Quoridor I/O v4 (docs/QuoridorIOv4.md): the v3 inputs with draws as the noResult outcome.
 *  - the v4 input row equals the v3 row (same feature counts, same values)
 *  - draw targets (ValueTargets::setOutcomeOfWinner) and a search of a certain draw (noResult, utility)
 */

#include "../tests/tests.h"

#include "../dataio/trainingwrite.h"
#include "../neuralnet/quoridornn.h"
#include "../search/search.h"
#include "../tests/testsearchcommon.h"

using namespace std;

namespace {

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

static void testInputsEqualV3() {
  cout << "  v4 rows equal v3 rows" << endl;
  testAssert(QuoridorNN::numSpatialFeatures(4) == QuoridorNN::numSpatialFeatures(3));
  testAssert(QuoridorNN::numGlobalFeatures(4) == QuoridorNN::numGlobalFeatures(3));
  testAssert(QuoridorNN::TRAINING_IO_VERSION == 4);
  Rand rand("quoridoriov4 rows");
  Rules rules = Rules::getQuoridorRules();
  rules.repetitionDrawCount = 3;
  int checked = 0;
  for(int g = 0; g < 20; g++) {
    vector<Move> moves = randomWallGame(rand, rules, 120, 0.4);
    Board board;
    board.setFencesLeft(rules.blackInitialFences, rules.whiteInitialFences);
    BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
    Player pla = P_BLACK;
    for(size_t i = 0; i <= moves.size(); i++) {
      MiscNNInputParams params;
      for(int sym = 0; sym < 2; sym++) {
        vector<float> s3(QuoridorNN::numSpatialFeatures(3) * AREA, 0.0f), s4 = s3;
        vector<float> g3(QuoridorNN::numGlobalFeatures(3), 0.0f), g4 = g3;
        QuoridorNN::fillRow(board, hist, pla, params, 3, false, s3.data(), g3.data());
        QuoridorNN::fillRow(board, hist, pla, params, 4, false, s4.data(), g4.data());
        if(sym) {
          QuoridorNN::applyInputSymmetry(s3.data(), 3, false, 1);
          QuoridorNN::applyInputSymmetry(s4.data(), 4, false, 1);
        }
        testAssert(s3 == s4 && g3 == g4);
      }
      checked++;
      if(i == moves.size())
        break;
      hist.makeBoardMoveAssumeLegal(board, moves[i].loc, pla, NULL);
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

}  // namespace

void Tests::runQuoridorIOv4Tests() {
  cout << "Running Quoridor I/O v4 tests" << endl;
  testInputsEqualV3();
  testDrawOutcome();
  cout << "Quoridor I/O v4 tests passed" << endl;
}
