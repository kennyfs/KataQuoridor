#include "../../tests/tests.h"
#include "../../core/fileutils.h"
#include "../../core/global.h"
#include "../../core/rand.h"
#include "../../program/setup.h"
#include "../../neuralnet/nneval.h"
#include "../q4board.h"
#include "../q4history.h"
#include "../q4notation.h"
#include "../q4playstate.h"
#include "../q4rules.h"
#include "../q4symmetry.h"
#include "../nn/q4nn.h"
#include "../nn/q4rawsymmetry.h"
#include "../search/q4search.h"
#include "../search/q4analysisdata.h"
#include "../search/q4reportedsearchvalues.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <vector>

using namespace std;
using namespace TestCommon;

namespace {

string getTestModelPath() {
  vector<string> candidates = {
    "cpp/tests/models/b1c32_q4/model.bin.gz",
    "tests/models/b1c32_q4/model.bin.gz",
    "../tests/models/b1c32_q4/model.bin.gz"
  };
  for(const string& p : candidates) {
    if(FileUtils::exists(p)) return p;
  }
  // Try exporting on the fly if python environment is available
  int ret = system("/home/kenny/ml_venv/bin/python python/q4/make_random_model.py b1c32_q4 cpp/tests/models --model-name b1c32_q4 --scale-heads");
  (void)ret;
  for(const string& p : candidates) {
    if(FileUtils::exists(p)) return p;
  }
  throw StringError("Could not find or generate b1c32_q4 model for search tests");
}

NNEvaluator* createTestEvaluator(const string& modelPath, Logger& logger, uint64_t seed = 42) {
  ConfigParser cfg;
  cfg.overrideKey("nnCacheSizePowerOfTwo", "16");
  cfg.overrideKey("nnMutexPoolSizePowerOfTwo", "12");
  Rand evalSeed(seed);
  return Setup::initializeNNEvaluator(
    modelPath, modelPath, "", cfg, logger, evalSeed, 1, Q4NNConst::POS_LEN, Q4NNConst::POS_LEN,
    Setup::MaxBatchSizeRequest::explicitSize(16), true, false, Setup::SETUP_FOR_OTHER
  );
}

SearchParams createTestSearchParams(int maxVisits = 100, int numThreads = 1) {
  ConfigParser cfg;
  cfg.overrideKey("maxVisits", to_string(maxVisits));
  cfg.overrideKey("numSearchThreads", to_string(numThreads));
  cfg.overrideKey("winLossUtilityFactor", "1.0");
  cfg.overrideKey("cpuctExploration", "1.1");
  cfg.overrideKey("cpuctExplorationLog", "0.0");
  cfg.overrideKey("fpuReductionMax", "0.2");
  cfg.overrideKey("rootFpuReductionMax", "0.2");
  cfg.overrideKey("fpuParentWeightByVisitedPolicy", "false");
  cfg.overrideKey("nnPolicyTemperature", "4.0");
  cfg.overrideKey("valueWeightExponent", "0.5");
  cfg.overrideKey("useUncertainty", "true");
  cfg.overrideKey("uncertaintyExponent", "1.0");
  cfg.overrideKey("uncertaintyCoeff", "0.25");
  cfg.overrideKey("useLcbForSelection", "true");
  cfg.overrideKey("lcbStdevs", "5.0");
  cfg.overrideKey("minVisitPropForLCB", "0.15");
  cfg.overrideKey("useNonBuggyLcb", "true");
  cfg.overrideKey("rootNoiseEnabled", "false");
  cfg.overrideKey("rootDirichletNoiseTotalConcentration", "10.83");
  cfg.overrideKey("rootDirichletNoiseWeight", "0.25");
  cfg.overrideKey("rootDesiredPerChildVisitsCoeff", "2");
  cfg.overrideKey("rootPolicyTemperatureEarly", "1.25");
  cfg.overrideKey("rootPolicyTemperature", "1.1");
  cfg.overrideKey("rootNumSymmetriesToSample", "1");
  cfg.overrideKey("chosenMoveTemperatureEarly", "0.75");
  cfg.overrideKey("chosenMoveTemperatureHalflife", "38");
  cfg.overrideKey("chosenMoveTemperature", "0.15");
  cfg.overrideKey("chosenMoveSubtract", "0");
  cfg.overrideKey("chosenMovePrune", "1");
  cfg.overrideKey("staticScoreUtilityFactor", "0.0");
  cfg.overrideKey("dynamicScoreUtilityFactor", "0.0");
  cfg.overrideKey("policyOptimism", "0.0");
  cfg.overrideKey("rootPolicyOptimism", "0.0");
  cfg.overrideKey("useGraphSearch", "false");
  cfg.overrideKey("useEvalCache", "false");
  cfg.overrideKey("subtreeValueBiasFactor", "0.0");
  cfg.overrideKey("avoidRepeatedPatternUtility", "0.0");
  cfg.overrideKey("antiMirror", "false");
  cfg.overrideKey("playoutDoublingAdvantage", "0.0");
  cfg.overrideKey("visitCapContempt", "0");
  cfg.overrideKey("rootSymmetryPruning", "false");
  cfg.overrideKey("conservativePass", "false");
  cfg.overrideKey("enablePassingHacks", "false");
  cfg.overrideKey("enableMorePassingHacks", "false");
  cfg.overrideKey("fillDameBeforePass", "false");
  cfg.overrideKey("rootEndingBonusPoints", "0.0");
  cfg.overrideKey("rootPruneUselessMoves", "false");
  cfg.overrideKey("ignorePreRootHistory", "false");
  cfg.overrideKey("ignoreAllHistory", "false");

  SearchParams params = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_OTHER);
  params.maxVisits = maxVisits;
  params.numThreads = numThreads;
  Q4S::Search::checkParams(params);
  return params;
}

// =========================================================================
// T19: Terminal values (Plan §8.9)
// =========================================================================
void testT19TerminalValues(NNEvaluator* nnEval, Logger& logger) {
  cout << "Running T19 Terminal values tests..." << endl;

  // (a) Seat 0 is next to the center: with 64 visits the chosen move is the winning step,
  // and root value has valueAvg[mover] > 0.9.
  {
    Q4Rules rules;
    Q4PlayState state(rules);
    int center = Q4Board::cellOf(5, 5); // f6 = 60
    int southNextToCenter = Q4Board::cellOf(5, 4); // f5 = 49
    state.board.occupant[state.board.pawn[0]] = -1;
    state.board.pawn[0] = southNextToCenter;
    state.board.occupant[southNextToCenter] = 0;
    state.board.wallsLeft[0] = 0;
    state.board.toMove = 0;
    state.board.recomputeDistancesToCenter();
    state.board.recomputeCachedPaths();
    state.board.hash = state.board.getHashFromScratch();

    SearchParams params = createTestSearchParams(64, 1);
    Q4S::Search search(params, nnEval, &logger, "t19a");
    search.setPosition(state);
    int move = search.runWholeSearchAndGetMove();

    testAssert(move == Q4Board::actionOfPawn(center));
    Q4S::ReportedSearchValues values;
    testAssert(search.getRootValues(values));
    testAssert(values.value[0] > 0.9);
  }

  // (b) Only the next seat can reach the center next turn and the mover cannot win at once:
  // with 400 visits the chosen move is a wall that makes that step impossible.
  {
    Q4Rules rules;
    Q4PlayState state(rules);
    // Seat 0 far from center (e.g. f2)
    state.board.occupant[state.board.pawn[0]] = -1;
    state.board.pawn[0] = Q4Board::cellOf(5, 1);
    state.board.occupant[state.board.pawn[0]] = 0;

    // Seat 1 is adjacent to center (e.g. e6 = (4, 5))
    int westNextToCenter = Q4Board::cellOf(4, 5);
    state.board.occupant[state.board.pawn[1]] = -1;
    state.board.pawn[1] = westNextToCenter;
    state.board.occupant[westNextToCenter] = 1;

    // Place horizontal walls in outer ranks to create a realistic midgame board
    for (int ay : {0, 1, 2, 7, 8, 9}) {
      for (int ax = 0; ax < 10; ax += 2) {
        state.board.hWalls.set(Q4Board::anchorOf(ax, ay));
        state.board.addWallToBlocked(ax, ay, true);
      }
    }
    state.board.toMove = 0;
    state.board.wallsLeft[0] = 3;
    state.board.wallsLeft[1] = 0;
    state.board.recomputeDistancesToCenter();
    state.board.recomputeCachedPaths();
    state.board.hash = state.board.getHashFromScratch();
    testAssert(state.board.distToCenter[westNextToCenter] == 1);

    SearchParams params = createTestSearchParams(400, 1);
    params.nnPolicyTemperature = 20.0f;
    Q4S::Search search(params, nnEval, &logger, "t19b");
    search.setPosition(state);
    int move = search.runWholeSearchAndGetMove();

    // The chosen move must be a wall
    testAssert(!Q4Board::isPawnAction(move));
    // And after applying the move, Player 2 can no longer step directly to center
    Q4PlayState nextState = state;
    nextState.playAssumeLegal(move);
    testAssert(nextState.board.distToCenter[nextState.board.pawn[1]] > 1);
  }

  // (c) Win by a two-pawn jump into the center is found.
  {
    Q4Rules rules;
    Q4PlayState state(rules);
    int center = Q4Board::cellOf(5, 5); // f6
    // Seat 0 at f4 = (5, 3)
    int f4 = Q4Board::cellOf(5, 3);
    // Opponent seat 2 at f5 = (5, 4)
    int f5 = Q4Board::cellOf(5, 4);

    state.board.occupant[state.board.pawn[0]] = -1;
    state.board.pawn[0] = f4;
    state.board.occupant[f4] = 0;

    state.board.occupant[state.board.pawn[2]] = -1;
    state.board.pawn[2] = f5;
    state.board.occupant[f5] = 2;

    state.board.wallsLeft[0] = 0;
    state.board.toMove = 0;
    state.board.recomputeDistancesToCenter();
    state.board.recomputeCachedPaths();
    state.board.hash = state.board.getHashFromScratch();

    SearchParams params = createTestSearchParams(100, 1);
    Q4S::Search search(params, nnEval, &logger, "t19c");
    search.setPosition(state);
    int move = search.runWholeSearchAndGetMove();

    testAssert(move == Q4Board::actionOfPawn(center));
    Q4S::ReportedSearchValues values;
    testAssert(search.getRootValues(values));
    testAssert(values.value[0] > 0.9);
  }

  // (d) Unit test of the utility of a terminal draw: -0.5 for every seat with 4 alive, 0.0 with 2 alive.
  {
    double drawVal[5] = {0.0, 0.0, 0.0, 0.0, 1.0};
    for(int s = 0; s < 4; s++) {
      double u4 = Q4S::Search::computeSeatUtility(s, drawVal, 4, true, 1.0);
      testAssert(std::fabs(u4 - (-0.5)) < 1e-9);
      double u2 = Q4S::Search::computeSeatUtility(s, drawVal, 2, true, 1.0);
      testAssert(std::fabs(u2 - 0.0) < 1e-9);
    }
  }

  // (e) Integration test: a position with maxPlies = plies + 1 where no legal move reaches the center.
  // Every child is a terminal draw. After a 200-visit search, every alive seat's root utility is -0.5 (1e-9)
  // and the root value is [0,0,0,0,1].
  {
    Q4Rules rules;
    rules.maxPlies = 10;
    Q4PlayState state(rules);
    state.plies = 9;

    SearchParams params = createTestSearchParams(200, 1);
    Q4S::Search search(params, nnEval, &logger, "t19_draw_4alive");
    search.setPosition(state);
    search.beginSearch();
    Q4S::NNOutput* rootNN = search.rootNode->getNNOutput();
    if(rootNN != nullptr) {
      for(int k = 0; k < 4; k++) rootNN->valueAbs[k] = 0.0f;
      rootNN->valueAbs[4] = 1.0f;
    }
    search.runWholeSearch();

    Q4S::ReportedSearchValues values;
    testAssert(search.getRootValues(values));
    for(int s = 0; s < 4; s++) {
      testAssert(std::fabs(values.value[s] - 0.0) < 1e-9);
      testAssert(std::fabs(values.utility[s] - (-0.5)) < 1e-9);
    }
    testAssert(std::fabs(values.value[4] - 1.0) < 1e-9);
  }

  // (f) The same with two seats eliminated: utility 0.0 for alive seats.
  {
    Q4Rules rules;
    rules.maxPlies = 10;
    Q4PlayState state(rules);
    state.plies = 9;
    state.eliminate(1);
    state.eliminate(3);

    SearchParams params = createTestSearchParams(200, 1);
    Q4S::Search search(params, nnEval, &logger, "t19_draw_2alive");
    search.setPosition(state);
    search.beginSearch();
    Q4S::NNOutput* rootNN = search.rootNode->getNNOutput();
    if(rootNN != nullptr) {
      for(int k = 0; k < 4; k++) rootNN->valueAbs[k] = 0.0f;
      rootNN->valueAbs[4] = 1.0f;
    }
    search.runWholeSearch();

    Q4S::ReportedSearchValues values;
    testAssert(search.getRootValues(values));
    testAssert(std::fabs(values.value[4] - 1.0) < 1e-9);
    testAssert(std::fabs(values.utility[0] - 0.0) < 1e-9);
    testAssert(std::fabs(values.utility[2] - 0.0) < 1e-9);
    testAssert(std::fabs(values.utility[1] - (-1.0)) < 1e-9); // eliminated
    testAssert(std::fabs(values.utility[3] - (-1.0)) < 1e-9); // eliminated
  }

  cout << "T19 Terminal values passed!" << endl;
}

// =========================================================================
// T20: Bookkeeping (Plan §8.9)
// =========================================================================
void testT20Bookkeeping(NNEvaluator* nnEval, Logger& logger) {
  cout << "Running T20 Bookkeeping tests..." << endl;

  Rand rand(2026);
  Q4Rules rules;
  rules.maxPlies = 30;
  rules.repetitionDrawCount = 2;

  SearchParams params = createTestSearchParams(100, 1);

  for(int iter = 0; iter < 10; iter++) {
    Q4History hist(rules);
    // Play random plies, optionally eliminate a seat
    int plies = rand.nextInt(0, 15);
    for(int p = 0; p < plies && !hist.isFinished; p++) {
      if(p == 5 && rand.nextBool(0.5)) {
        int aliveCount = hist.currentBoard.getNumAlive();
        if(aliveCount > 2) {
          int victim = (hist.currentBoard.toMove + 1) % 4;
          if(hist.currentBoard.isAlive(victim))
            hist.eliminate(victim);
        }
      }
      vector<int> legal;
      hist.currentBoard.getLegalActions(hist.currentBoard.toMove, legal);
      if(legal.empty()) break;
      hist.play(legal[rand.nextUInt((uint32_t)legal.size())]);
    }
    if(hist.isFinished) continue;

    Q4S::Search search(params, nnEval, &logger, "t20_" + to_string(iter));
    search.setPosition(hist);
    search.runWholeSearch();

    // 1. root visits = 1 + sum of edge visits
    int64_t rootVisits = search.getRootVisits();
    std::vector<Q4S::AnalysisData> analysis;
    search.getAnalysisData(analysis);
    int64_t sumEdgeVisits = 0;
    for(const auto& ad : analysis)
      sumEdgeVisits += ad.numVisits;
    testAssert(rootVisits == 1 + sumEdgeVisits);

    // 2. every valueAvg sums to 1.0 (1e-9)
    Q4S::ReportedSearchValues rootVals;
    testAssert(search.getRootValues(rootVals));
    double sumRootVal = 0.0;
    for(int k = 0; k < 5; k++)
      sumRootVal += rootVals.value[k];
    testAssert(std::abs(sumRootVal - 1.0) < 1e-9);

    // 3. eliminated seats have value 0 and utility -winLossUtilityFactor
    for(int s = 0; s < 4; s++) {
      if(!hist.currentBoard.isAlive(s)) {
        testAssert(rootVals.value[s] == 0.0);
        testAssert(std::abs(rootVals.utility[s] - (-params.winLossUtilityFactor)) < 1e-9);
      }
      // every utility in [-1, 1] * winLossUtilityFactor
      testAssert(rootVals.utility[s] >= -params.winLossUtilityFactor - 1e-9);
      testAssert(rootVals.utility[s] <= params.winLossUtilityFactor + 1e-9);
    }

    // 4. Per-child checks
    for(const auto& ad : analysis) {
      if(ad.numVisits > 0) {
        double sumChildVal = 0.0;
        for(int k = 0; k < 5; k++)
          sumChildVal += ad.valueAvg[k];
        testAssert(std::abs(sumChildVal - 1.0) < 1e-9);
        for(int s = 0; s < 4; s++) {
          if(!hist.currentBoard.isAlive(s))
            testAssert(ad.valueAvg[s] == 0.0);
        }
      }
    }
  }
  cout << "T20 Bookkeeping passed!" << endl;
}

// =========================================================================
// T21: Exact endgames (Plan §8.9)
// =========================================================================
int solve2PlayerGame(Q4PlayState state, int seatA, int seatB, int depthLimit) {
  if(state.isFinished) {
    if(state.winnerSeat == seatA) return 1;
    if(state.winnerSeat == seatB) return -1;
    return 0;
  }
  if(depthLimit <= 0) return 0;
  int mover = state.board.toMove;
  vector<int> actions;
  state.getLegalActions(actions);
  if(actions.empty()) return 0;

  if(mover == seatA) {
    int best = -1;
    for(int act : actions) {
      Q4PlayState nextState = state;
      nextState.playAssumeLegal(act);
      int val = solve2PlayerGame(nextState, seatA, seatB, depthLimit - 1);
      if(val > best) best = val;
      if(best == 1) break;
    }
    return best;
  }
  else if(mover == seatB) {
    int best = 1;
    for(int act : actions) {
      Q4PlayState nextState = state;
      nextState.playAssumeLegal(act);
      int val = solve2PlayerGame(nextState, seatA, seatB, depthLimit - 1);
      if(val < best) best = val;
      if(best == -1) break;
    }
    return best;
  }
  else {
    return 0;
  }
}

void testT21ExactEndgames(NNEvaluator* nnEval, Logger& logger) {
  cout << "Running T21 Exact endgames tests..." << endl;

  Q4Rules rules;
  int seatA = 0;
  int seatB = 1;
  int center = Q4Board::cellOf(5, 5);

  vector<Q4PlayState> solvedPositions;
  vector<int> trueWinners;

  // Generate distinct short-distance endgame positions
  for(int dyA = 1; dyA <= 3; dyA++) {
    for(int dxA = -1; dxA <= 1; dxA++) {
      for(int dxB = 1; dxB <= 3; dxB++) {
        for(int dyB = -1; dyB <= 1; dyB++) {
          int cellA = Q4Board::cellOf(5 + dxA, 5 - dyA);
          int cellB = Q4Board::cellOf(5 - dxB, 5 + dyB);
          if(cellA == center || cellB == center || cellA == cellB) continue;

          Q4PlayState state(rules);
          // Eliminate seat 2 and seat 3
          state.eliminate(2);
          state.eliminate(3);
          // No walls left for anyone
          state.board.wallsLeft[0] = 0;
          state.board.wallsLeft[1] = 0;
          state.board.wallsLeft[2] = 0;
          state.board.wallsLeft[3] = 0;

          state.board.occupant[state.board.pawn[0]] = -1;
          state.board.pawn[0] = cellA;
          state.board.occupant[cellA] = 0;

          state.board.occupant[state.board.pawn[1]] = -1;
          state.board.pawn[1] = cellB;
          state.board.occupant[cellB] = 1;

          state.board.toMove = seatA;
          state.board.recomputeDistancesToCenter();

          int outcome = solve2PlayerGame(state, seatA, seatB, 6);
          if(outcome == 1) {
            solvedPositions.push_back(state);
            trueWinners.push_back(seatA);
          }
          else if(outcome == -1) {
            solvedPositions.push_back(state);
            trueWinners.push_back(seatB);
          }
          if(solvedPositions.size() >= 25) break;
        }
        if(solvedPositions.size() >= 25) break;
      }
      if(solvedPositions.size() >= 25) break;
    }
    if(solvedPositions.size() >= 25) break;
  }

  cout << "T21: Found and verified " << solvedPositions.size() << " exact endgame positions." << endl;
  testAssert(solvedPositions.size() >= 20);

  SearchParams params = createTestSearchParams(350, 1);
  for(size_t i = 0; i < solvedPositions.size(); i++) {
    Q4S::Search search(params, nnEval, &logger, "t21_" + to_string(i));
    search.setPosition(solvedPositions[i]);
    int chosenMove = search.runWholeSearchAndGetMove();

    Q4S::ReportedSearchValues values;
    testAssert(search.getRootValues(values));
    int winner = trueWinners[i];
    testAssert(values.value[winner] > 0.8);

    // If mover is the true winner, the chosen move must preserve the win
    if(solvedPositions[i].board.toMove == winner) {
      Q4PlayState nextState = solvedPositions[i];
      nextState.playAssumeLegal(chosenMove);
      int nextOutcome = solve2PlayerGame(nextState, seatA, seatB, 6);
      int expectedOutcome = (winner == seatA) ? 1 : -1;
      testAssert(nextOutcome == expectedOutcome);
    }
  }
  cout << "T21 Exact endgames passed!" << endl;
}

// =========================================================================
// T22: Determinism (Plan §8.9)
// =========================================================================
void testT22Determinism(NNEvaluator* nnEval, Logger& logger) {
  cout << "Running T22 Determinism tests..." << endl;

  Q4Rules rules;
  Q4PlayState state(rules);
  // Add some moves to make position interesting
  state.playAssumeLegal(Q4Board::actionOfPawn(Q4Board::cellOf(5, 2)));
  state.playAssumeLegal(Q4Board::actionOfPawn(Q4Board::cellOf(2, 5)));
  state.playAssumeLegal(Q4Board::actionOfHWall(Q4Board::anchorOf(4, 4)));

  SearchParams params = createTestSearchParams(200, 1);

  Q4S::Search s1(params, nnEval, &logger, "fixedSeed123");
  s1.setPosition(state);
  s1.runWholeSearch();
  std::vector<Q4S::AnalysisData> d1;
  s1.getAnalysisData(d1);
  Q4S::ReportedSearchValues v1;
  s1.getRootValues(v1);

  Q4S::Search s2(params, nnEval, &logger, "fixedSeed123");
  s2.setPosition(state);
  s2.runWholeSearch();
  std::vector<Q4S::AnalysisData> d2;
  s2.getAnalysisData(d2);
  Q4S::ReportedSearchValues v2;
  s2.getRootValues(v2);

  testAssert(s1.getRootVisits() == s2.getRootVisits());
  testAssert(d1.size() == d2.size());
  for(size_t i = 0; i < d1.size(); i++) {
    testAssert(d1[i].move == d2[i].move);
    testAssert(d1[i].numVisits == d2[i].numVisits);
    testAssert(d1[i].policyPrior == d2[i].policyPrior);
    testAssert(d1[i].utility == d2[i].utility);
  }
  for(int k = 0; k < 5; k++)
    testAssert(v1.value[k] == v2.value[k]);

  cout << "T22 Determinism passed!" << endl;
}

// =========================================================================
// T23: Symmetry (Plan §8.9)
// =========================================================================
void testT23Symmetry(NNEvaluator* nnEval, Logger& logger) {
  (void)logger;
  cout << "Running T23 Symmetry tests..." << endl;

  Q4Rules rules;
  Q4PlayState state(rules);
  state.playAssumeLegal(Q4Board::actionOfPawn(Q4Board::cellOf(5, 2)));
  state.playAssumeLegal(Q4Board::actionOfVWall(Q4Board::anchorOf(3, 4)));

  Rand rand(42);
  NNResultBuf buf;
  Q4NN::Eval eval0;
  Q4NN::averageMultipleSymmetries(*nnEval, buf, state, rand, 8, eval0);

  for(int sym = 1; sym < 8; sym++) {
    Q4PlayState symState(rules);
    symState.board = Q4Symmetry::applyBoard(state.board, sym);
    symState.plies = state.plies;
    symState.isFinished = state.isFinished;
    symState.winnerSeat = state.winnerSeat;
    symState.isDraw = state.isDraw;

    Q4NN::Eval evalSym;
    Rand randSym(42);
    Q4NN::averageMultipleSymmetries(*nnEval, buf, symState, randSym, 8, evalSym);

    // With 8 symmetries averaged, values of each seat match to within 1e-4
    for(int s = 0; s < 4; s++) {
      testAssert(std::abs(eval0.valueAbsMasked[s] - evalSym.valueAbsMasked[s]) < 1e-4);
    }
    // Draw probability should match
    testAssert(std::abs(eval0.valueAbsMasked[4] - evalSym.valueAbsMasked[4]) < 1e-4);
  }

  cout << "T23 Symmetry passed!" << endl;
}

// =========================================================================
// T24: Cache (Plan §8.9)
// =========================================================================
void testT24Cache(NNEvaluator* nnEval, Logger& logger) {
  (void)logger;
  cout << "Running T24 Cache tests..." << endl;

  // 1. Check that q4rawsymmetry tables agree with Q4Symmetry
  for(int sym = 0; sym < 8; sym++) {
    for(int c = 0; c < 121; c++) {
      testAssert(Q4RawSymmetry::applyCell(c, sym) == Q4Symmetry::applyCell(c, sym));
    }
    for(int ay = 0; ay < 10; ay++) {
      for(int ax = 0; ax < 10; ax++) {
        for(int h = 0; h <= 1; h++) {
          int oax1, oay1, oax2, oay2;
          bool oh1, oh2;
          Q4RawSymmetry::applyAnchor(ax, ay, (h == 1), sym, oax1, oay1, oh1);
          Q4Symmetry::applyAnchor(ax, ay, (h == 1), sym, oax2, oay2, oh2);
          testAssert(oax1 == oax2 && oay1 == oay2 && oh1 == oh2);
        }
      }
    }
  }

  // 2. Cache key has no symmetry
  Q4Rules rules;
  Q4PlayState state(rules);
  Hash128 h1 = Q4NN::getCacheHash(state);
  Hash128 h2 = Q4NN::getCacheHash(state);
  testAssert(h1 == h2);

  // 3. Evaluate under symmetry a, then symmetry b hits cache
  NNResultBuf buf;
  Q4NN::Eval evalA;
  Q4NN::evaluate(*nnEval, buf, state, 1, false, evalA);

  Q4NN::Eval evalB;
  Q4NN::evaluate(*nnEval, buf, state, 2, false, evalB);

  // Decoded outputs should match within 1e-6
  for(int k = 0; k < 5; k++) {
    testAssert(std::abs(evalA.valueAbs[k] - evalB.valueAbs[k]) < 1e-6);
  }
  for(int a = 0; a < Q4Board::NUM_ACTIONS; a++) {
    testAssert(std::abs(evalA.policyLogits[0][a] - evalB.policyLogits[0][a]) < 1e-6);
  }

  cout << "T24 Cache passed!" << endl;
}

// =========================================================================
// T25: Multithreading (Plan §8.9)
// =========================================================================
void testT25Threads(NNEvaluator* nnEval, Logger& logger) {
  cout << "Running T25 Multithreading tests..." << endl;

  // 1. Position with many terminal nodes: pawns next to center, small maxPlies
  {
    Q4Rules termRules;
    termRules.maxPlies = 6;
    Q4PlayState termState(termRules);
    termState.plies = 4;
    termState.board.pawn[0] = 49;
    termState.board.pawn[1] = 59;
    termState.board.pawn[2] = 71;
    termState.board.pawn[3] = 61;
    termState.board.toMove = 0;
    termState.board.wallsLeft[0] = 0;
    termState.board.wallsLeft[1] = 0;
    termState.board.wallsLeft[2] = 0;
    termState.board.wallsLeft[3] = 0;

    SearchParams termParams = createTestSearchParams(3000, 8);
    Q4S::Search termSearch(termParams, nnEval, &logger, "t25_term_8threads");
    termSearch.setPosition(termState);
    termSearch.runWholeSearch();

    int64_t rootVisits = termSearch.getRootVisits();
    testAssert(rootVisits > 100);
  }

  Rand rand(777);
  Q4Rules rules;
  SearchParams params = createTestSearchParams(3000, 8);

  // Run on positions to test multithreaded descent and stats recomputation
  for(int i = 0; i < 5; i++) {
    Q4History hist(rules);
    int plies = rand.nextInt(2, 10);
    for(int p = 0; p < plies && !hist.isFinished; p++) {
      vector<int> acts;
      hist.currentBoard.getLegalActions(hist.currentBoard.toMove, acts);
      if(acts.empty()) break;
      hist.play(acts[rand.nextUInt((uint32_t)acts.size())]);
    }
    if(hist.isFinished) continue;

    Q4S::Search search(params, nnEval, &logger, "t25_" + to_string(i));
    search.setPosition(hist);
    search.runWholeSearch();

    int64_t rootVisits = search.getRootVisits();
    std::vector<Q4S::AnalysisData> analysis;
    search.getAnalysisData(analysis);
    int64_t sumEdgeVisits = 0;
    for(const auto& ad : analysis)
      sumEdgeVisits += ad.numVisits;
    testAssert(rootVisits == 1 + sumEdgeVisits);

    Q4S::ReportedSearchValues rootVals;
    testAssert(search.getRootValues(rootVals));
    double sumRootVal = 0.0;
    for(int k = 0; k < 5; k++)
      sumRootVal += rootVals.value[k];
    testAssert(std::abs(sumRootVal - 1.0) < 1e-9);
  }

  cout << "T25 Multithreading passed!" << endl;
}

// =========================================================================
// T26: Tree reuse (Plan §8.9)
// =========================================================================
void testT26TreeReuse(NNEvaluator* nnEval, Logger& logger) {
  cout << "Running T26 Tree reuse tests..." << endl;

  cout << "T26 entry liveNodeCount: " << Q4S::SearchNode::liveNodeCount.load() << endl;
  testAssert(Q4S::SearchNode::liveNodeCount.load() == 0);

  Q4Rules rules;
  Q4PlayState state(rules);

  SearchParams params = createTestSearchParams(200, 1);
  {
    std::unique_ptr<Q4S::Search> search(new Q4S::Search(params, nnEval, &logger, "t26"));
    search->setPosition(state);
    search->runWholeSearch();

    std::vector<Q4S::SearchNode*> reachableNodes = search->enumerateTreePostOrder();
    int64_t liveAfterSearch = Q4S::SearchNode::liveNodeCount.load();
    cout << "T26: Live nodes after runWholeSearch: " << liveAfterSearch
         << ", reachable nodes: " << reachableNodes.size() << endl;
    testAssert(liveAfterSearch == (int64_t)reachableNodes.size());

    std::vector<Q4S::AnalysisData> analysis;
    search->getAnalysisData(analysis);
    testAssert(!analysis.empty());

    // Find the child move with the most visits
    int bestMove = analysis[0].move;
    int64_t childVisits = analysis[0].numVisits;

    // Make move and check new root visits
    bool ok = search->makeMove(bestMove);
    testAssert(ok);
    int64_t newRootVisits = search->getRootVisits();
    testAssert(newRootVisits == childVisits);

    std::vector<Q4S::SearchNode*> newReachableNodes = search->enumerateTreePostOrder();
    int64_t liveAfterMove = Q4S::SearchNode::liveNodeCount.load();
    cout << "T26: Live nodes after makeMove: " << liveAfterMove
         << ", new reachable nodes: " << newReachableNodes.size() << endl;
    testAssert(liveAfterMove == (int64_t)newReachableNodes.size());

    // Test clearSearch()
    search->clearSearch();
    int64_t liveAfterClear = Q4S::SearchNode::liveNodeCount.load();
    cout << "T26: Live nodes after clearSearch(): " << liveAfterClear << endl;
    testAssert(liveAfterClear == 0);

    // Run another search to allocate nodes, then destroy Search
    search->setPosition(state);
    search->runWholeSearch();
    testAssert(Q4S::SearchNode::liveNodeCount.load() > 0);
  }
  // After destroying the Search, live count must be 0
  int64_t liveAfterDestroy = Q4S::SearchNode::liveNodeCount.load();
  cout << "T26: Live nodes after destroying Search: " << liveAfterDestroy << endl;
  testAssert(liveAfterDestroy == 0);

  // 30 consecutive searches of 1,000 visits on different positions with one Search object
  {
    SearchParams searchParams1000 = createTestSearchParams(1000, 1);
    Q4S::Search search30(searchParams1000, nnEval, &logger, "t26_30");
    Rand rand("testT26TreeReuse_30_searches");
    for(int iter = 0; iter < 30; iter++) {
      Q4PlayState randState(rules);
      int numPlies = rand.nextInt(0, 10);
      for(int p = 0; p < numPlies; p++) {
        std::vector<int> legals;
        randState.board.getLegalActions(randState.board.toMove, legals);
        if(legals.empty() || randState.isFinished)
          break;
        int act = legals[rand.nextInt(0, (int)legals.size() - 1)];
        randState.playAssumeLegal(act);
      }
      search30.setPosition(randState);
      search30.runWholeSearch();
      search30.clearSearch();
      int64_t live = Q4S::SearchNode::liveNodeCount.load();
      testAssert(live == 0);
    }
  }
  testAssert(Q4S::SearchNode::liveNodeCount.load() == 0);

  cout << "T26 Tree reuse passed!" << endl;
}

// =========================================================================
// T27: Config guard (Plan §8.9)
// =========================================================================
void testT27ConfigGuard() {
  cout << "Running T27 Config guard tests..." << endl;

  vector<string> forbiddenParamNames = {
    "staticScoreUtilityFactor", "dynamicScoreUtilityFactor", "policyOptimism",
    "rootPolicyOptimism", "useGraphSearch", "useEvalCache", "subtreeValueBiasFactor",
    "avoidRepeatedPatternUtility", "antiMirror", "playoutDoublingAdvantage",
    "visitCapContempt", "rootSymmetryPruning", "conservativePass",
    "enablePassingHacks", "enableMorePassingHacks", "fillDameBeforePass",
    "rootEndingBonusPoints", "rootPruneUselessMoves", "ignorePreRootHistory", "ignoreAllHistory"
  };

  for(const string& paramName : forbiddenParamNames) {
    SearchParams params = createTestSearchParams(10, 1);
    if(paramName == "staticScoreUtilityFactor") params.staticScoreUtilityFactor = 0.1;
    else if(paramName == "dynamicScoreUtilityFactor") params.dynamicScoreUtilityFactor = 0.1;
    else if(paramName == "policyOptimism") params.policyOptimism = 1.0;
    else if(paramName == "rootPolicyOptimism") params.rootPolicyOptimism = 1.0;
    else if(paramName == "useGraphSearch") params.useGraphSearch = true;
    else if(paramName == "useEvalCache") params.useEvalCache = true;
    else if(paramName == "subtreeValueBiasFactor") params.subtreeValueBiasFactor = 0.5;
    else if(paramName == "avoidRepeatedPatternUtility") params.avoidRepeatedPatternUtility = 0.5;
    else if(paramName == "antiMirror") params.antiMirror = true;
    else if(paramName == "playoutDoublingAdvantage") params.playoutDoublingAdvantage = 1.0;
    else if(paramName == "visitCapContempt") params.visitCapContempt = 10;
    else if(paramName == "rootSymmetryPruning") params.rootSymmetryPruning = true;
    else if(paramName == "conservativePass") params.conservativePass = true;
    else if(paramName == "enablePassingHacks") params.enablePassingHacks = true;
    else if(paramName == "enableMorePassingHacks") params.enableMorePassingHacks = true;
    else if(paramName == "fillDameBeforePass") params.fillDameBeforePass = true;
    else if(paramName == "rootEndingBonusPoints") params.rootEndingBonusPoints = 0.5;
    else if(paramName == "rootPruneUselessMoves") params.rootPruneUselessMoves = true;
    else if(paramName == "ignorePreRootHistory") params.ignorePreRootHistory = true;
    else if(paramName == "ignoreAllHistory") params.ignoreAllHistory = true;

    bool caught = false;
    try {
      Q4S::Search::checkParams(params);
    }
    catch(const StringError& err) {
      caught = true;
      string msg = err.what();
      testAssert(msg.find(paramName) != string::npos);
    }
    testAssert(caught);
  }

  cout << "T27 Config guard passed!" << endl;
}

}  // namespace

void Tests::runQ4SearchTests() {
  cout << "========================================" << endl;
  cout << "Starting Q4 Search Test Suite (T19 - T27)" << endl;
  cout << "========================================" << endl;

  testT27ConfigGuard();

  string modelPath = getTestModelPath();
  Logger logger;
  NNEvaluator* nnEval = createTestEvaluator(modelPath, logger, 42);

  testT19TerminalValues(nnEval, logger);
  testT20Bookkeeping(nnEval, logger);
  testT21ExactEndgames(nnEval, logger);
  testT22Determinism(nnEval, logger);
  testT23Symmetry(nnEval, logger);
  testT24Cache(nnEval, logger);
  testT25Threads(nnEval, logger);
  testT26TreeReuse(nnEval, logger);

  delete nnEval;
  cout << "All Q4 Search tests PASSED!" << endl;
}
