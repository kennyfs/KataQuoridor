/*
 * testquoridorselfplay.cpp
 * Tests for the Quoridor I/O v2 self-play setup (docs/QuoridorIOv2.md, step 3):
 *  - komi and fence-handicap randomization in GameInitializer: off by default, and the sampled distributions
 *  - komi compensation: which games ask for it, the even-komi search on the Quoridor komi grid (with a synthetic
 *    evaluator that sees komi, as an I/O v2 net would), its random rounding, and the no-op for nets that don't see komi
 *  - gatekeeper scoring: a draw is exactly half a point
 */

#include "../tests/tests.h"

#include "../core/config_parser.h"
#include "../program/play.h"
#include "../program/playutils.h"
#include "../tests/testsearchcommon.h"

using namespace std;

namespace {

static GameInitializer* makeGameInit(const string& extraConfig, const string& seed) {
  istringstream in("bSizes = 17\nbSizeRelProbs = 1\n" + extraConfig);
  ConfigParser cfg(in);
  Logger logger(nullptr, false, false, false);
  return new GameInitializer(cfg, logger, seed);
}

struct SampledGame {
  float komi;
  int blackWalls;
  int whiteWalls;
  bool makeGameFair;
};

static SampledGame sampleGame(GameInitializer& gameInit, const InitialPosition* initialPosition = NULL) {
  Board board;
  Player pla;
  BoardHistory hist;
  ExtraBlackAndKomi extraBlackAndKomi;
  OtherGameProperties otherGameProps;
  PlaySettings playSettings;
  gameInit.createGame(board, pla, hist, extraBlackAndKomi, initialPosition, playSettings, otherGameProps, NULL, BoardHistoryModes());
  testAssert(Rules::isValidKomi(hist.rules.komi));
  testAssert(extraBlackAndKomi.komiMean == hist.rules.komi);
  //The initial walls are both in the rules and on the board.
  testAssert(board.blackFences == hist.rules.blackInitialFences && board.whiteFences == hist.rules.whiteInitialFences);
  return SampledGame{hist.rules.komi, hist.rules.blackInitialFences, hist.rules.whiteInitialFences, extraBlackAndKomi.makeGameFair};
}

static void checkFraction(const string& what, int64_t num, int64_t denom, double expected, double tolerance) {
  double frac = denom > 0 ? (double)num / (double)denom : 0.0;
  cout << "    " << what << ": " << num << " / " << denom << " = " << Global::strprintf("%.4f", frac)
       << " (expected " << expected << ")" << endl;
  testAssert(std::fabs(frac - expected) <= tolerance);
}

//------------------------------------------------------------------------------------------------

static void testDefaultsOff() {
  cout << "  Randomization is off by default" << endl;
  //An old-style config (as selfplay_quoridor.cfg / gatekeeper_quoridor.cfg in 0.1.0 and step 1).
  std::unique_ptr<GameInitializer> gameInit(makeGameInit(
    "komiMean = -0.5\nkomiStdev = 0\nhandicapProb = 0\nhandicapCompensateKomiProb = 0\nforkCompensateKomiProb = 0\n",
    "testquoridorselfplay defaults"
  ));
  testAssert(!gameInit->mayCreateNonStandardGames());
  for(int i = 0; i < 500; i++) {
    SampledGame g = sampleGame(*gameInit);
    testAssert(g.komi == Rules::DEFAULT_KOMI && g.blackWalls == 10 && g.whiteWalls == 10 && !g.makeGameFair);
  }

  //Each setting that can make a game non-standard is seen by mayCreateNonStandardGames (the gatekeeper's check).
  for(const string& cfg : vector<string>{
    "quoridorKomiRandomProb = 0.3\n",
    "quoridorFenceHandicapProb = 0.1\n",
    "komiMean = 1.5\n",
    "komiStdev = 1\n",
    "komiBigStdevProb = 0.1\nkomiBigStdev = 3\n",
    "blackInitialWalls = 9\n",
    "whiteInitialWalls = 9\n",
  }) {
    std::unique_ptr<GameInitializer> g(makeGameInit(cfg, "testquoridorselfplay nonstandard"));
    testAssert(g->mayCreateNonStandardGames());
  }

  //Bad weights are rejected.
  for(const string& cfg : vector<string>{
    "quoridorKomiRandomWeights = 0,0\n",
    "quoridorFenceHandicapWeights = 1,1,1,1,1,1,1,1,1,1,1\n",
    "quoridorFenceHandicapWeights = 1,-1\n",
  }) {
    bool threw = false;
    try { std::unique_ptr<GameInitializer> g(makeGameInit(cfg, "testquoridorselfplay bad")); } catch(const StringError&) { threw = true; }
    testAssert(threw);
  }
}

//The v2 self-play distributions (docs/QuoridorIOv2.md section 5): 30% of games get komi -0.5 +/- n and,
//independently, 10% a fence handicap of n walls for one side; n = 1, 2, 3 weighted 60/30/10 for both. Half of the
//fence-handicap games ask for a fair komi (handicapCompensateKomiProb = 0.5).
static void testDistributions() {
  cout << "  Komi and fence-handicap distributions" << endl;
  std::unique_ptr<GameInitializer> gameInit(makeGameInit(
    "quoridorKomiRandomProb = 0.3\n"
    "quoridorKomiRandomWeights = 0.6,0.3,0.1\n"
    "quoridorFenceHandicapProb = 0.1\n"
    "quoridorFenceHandicapWeights = 0.6,0.3,0.1\n"
    "handicapCompensateKomiProb = 0.5\n"
    "forkCompensateKomiProb = 0.8\n",
    "testquoridorselfplay distributions"
  ));
  testAssert(gameInit->mayCreateNonStandardGames());

  const int64_t numGames = 40000;
  int64_t komiRandom = 0, komiPlus = 0;
  int64_t komiOffset[4] = {0,0,0,0};
  int64_t fence = 0, fenceBlack = 0, both = 0;
  int64_t fenceN[4] = {0,0,0,0};
  int64_t fairAsked = 0, fairAskedWithoutFence = 0;
  for(int64_t i = 0; i < numGames; i++) {
    SampledGame g = sampleGame(*gameInit);
    bool isKomiRandom = g.komi != Rules::DEFAULT_KOMI;
    bool isFence = g.blackWalls != 10 || g.whiteWalls != 10;
    if(isKomiRandom) {
      komiRandom++;
      int n = (int)std::lround(std::fabs(g.komi - Rules::DEFAULT_KOMI));
      testAssert(n >= 1 && n <= 3);
      komiOffset[n]++;
      if(g.komi > Rules::DEFAULT_KOMI)
        komiPlus++;
    }
    if(isFence) {
      fence++;
      //Exactly one side is handicapped, the other keeps 10.
      testAssert((g.blackWalls == 10) != (g.whiteWalls == 10));
      int n = 10 - std::min(g.blackWalls, g.whiteWalls);
      testAssert(n >= 1 && n <= 3);
      fenceN[n]++;
      if(g.blackWalls < 10)
        fenceBlack++;
      if(g.makeGameFair)
        fairAsked++;
    }
    else if(g.makeGameFair)
      fairAskedWithoutFence++;
    if(isKomiRandom && isFence)
      both++;
  }
  checkFraction("games with a random komi", komiRandom, numGames, 0.30, 0.01);
  checkFraction("  of which n = 1", komiOffset[1], komiRandom, 0.60, 0.02);
  checkFraction("  of which n = 2", komiOffset[2], komiRandom, 0.30, 0.02);
  checkFraction("  of which n = 3", komiOffset[3], komiRandom, 0.10, 0.015);
  checkFraction("  of which komi > -0.5", komiPlus, komiRandom, 0.50, 0.02);
  checkFraction("games with a fence handicap", fence, numGames, 0.10, 0.006);
  checkFraction("  of which n = 1", fenceN[1], fence, 0.60, 0.03);
  checkFraction("  of which n = 2", fenceN[2], fence, 0.30, 0.03);
  checkFraction("  of which n = 3", fenceN[3], fence, 0.10, 0.02);
  checkFraction("  of which Black is handicapped", fenceBlack, fence, 0.50, 0.03);
  checkFraction("  of which ask for a fair komi", fairAsked, fence, 0.50, 0.03);
  checkFraction("games with both (independent: 0.03)", both, numGames, 0.03, 0.004);
  testAssert(fairAskedWithoutFence == 0);

  //Fork games keep the komi and walls of their position and ask for a fair komi with forkCompensateKomiProb.
  {
    Rules rules = Rules::getQuoridorRules();
    rules.komi = 1.5f;
    rules.whiteInitialFences = 8;
    Board board;
    board.setFencesLeft(rules.blackInitialFences, rules.whiteInitialFences);
    BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
    InitialPosition fork(board, hist, P_BLACK, true, false, false, 1.0);
    int64_t numForks = 5000, numFair = 0;
    for(int64_t i = 0; i < numForks; i++) {
      SampledGame g = sampleGame(*gameInit, &fork);
      testAssert(g.komi == 1.5f && g.blackWalls == 10 && g.whiteWalls == 8);
      if(g.makeGameFair)
        numFair++;
    }
    checkFraction("fork games that ask for a fair komi", numFair, numForks, 0.80, 0.02);
  }
}

//------------------------------------------------------------------------------------------------

//The even-komi search behind adjustKomiToEven, on an evaluator that sees komi (like an I/O v2 net): White's lead is
//t0 + komi (t0 = White's expected tempo) and its winLoss a smooth function of the lead. The fair komi is -t0.
static void testFindEvenKomi() {
  cout << "  Even komi on the Quoridor komi grid" << endl;
  for(double t0 : vector<double>{0.0, 0.3, -0.5, 1.0, 2.3, -4.7, 7.9, -12.2, 19.6}) {
    for(double sharpness : vector<double>{0.5, 1.5, 4.0}) {
      int numEvals = 0;
      auto eval = [&](float komi) {
        testAssert(Rules::isValidKomi(komi));
        numEvals++;
        double lead = t0 + komi;
        return std::make_pair(lead, std::tanh(lead / sharpness));
      };
      for(float startKomi : vector<float>{-0.5f, 5.5f, -8.5f}) {
        numEvals = 0;
        double fair = PlayUtils::findEvenKomi(eval, startKomi);
        //The binary search brackets the zero of winLoss between two neighbouring komis and interpolates linearly;
        //tanh is odd, so the interpolation error is small.
        testAssert(std::fabs(fair - (-t0)) < 0.2);
        testAssert(numEvals <= 12);
      }
    }
  }
  //A lead the grid can't balance is clipped to the largest komi.
  {
    auto eval = [](float komi) { double lead = -40.0 + komi; return std::make_pair(lead, std::tanh(lead)); };
    double fair = PlayUtils::findEvenKomi(eval, -0.5f);
    Rand rand("testquoridorselfplay clip");
    testAssert(PlayUtils::roundKomiRandomly(fair, rand) == Rules::MAX_KOMI);
  }

  //Compensation keeps the komi when the fair komi is implausibly far from the standard -0.5 (an untrained lead
  //head); otherwise it rounds randomly to a neighbouring valid komi.
  {
    Rand rand("testquoridorselfplay compensated window");
    testAssert(PlayUtils::compensatedKomiOrKeep(20.5, -0.5f, rand) == -0.5f);
    testAssert(PlayUtils::compensatedKomiOrKeep(-11.0, 1.5f, rand) == 1.5f);
    testAssert(PlayUtils::compensatedKomiOrKeep(10.5, 1.5f, rand) == 1.5f);
    testAssert(PlayUtils::compensatedKomiOrKeep(9.5, -0.5f, rand) == 9.5f);
    testAssert(PlayUtils::compensatedKomiOrKeep(-10.5, 2.5f, rand) == -10.5f);
    testAssert(PlayUtils::compensatedKomiOrKeep(4.5, -0.5f, rand) == 4.5f);
    float k = PlayUtils::compensatedKomiOrKeep(1.2, -0.5f, rand);
    testAssert(k == 0.5f || k == 1.5f);
  }

  //Random rounding to the neighbouring valid komis, the nearer one more likely.
  {
    Rand rand("testquoridorselfplay rounding");
    int64_t upper = 0, n = 20000;
    for(int64_t i = 0; i < n; i++) {
      float k = PlayUtils::roundKomiRandomly(1.2, rand);
      testAssert(k == 0.5f || k == 1.5f);
      if(k == 1.5f)
        upper++;
    }
    checkFraction("1.2 rounds to 1.5", upper, n, 0.70, 0.015);
    testAssert(PlayUtils::roundKomiRandomly(-0.5, rand) == -0.5f);
    testAssert(PlayUtils::roundKomiRandomly(2.5, rand) == 2.5f);
  }
}

//Play::runGame skips komi compensation unless both nets see komi (I/O v2). A fork game that asks for a fair komi
//keeps its komi with an I/O v1 net.
static void testCompensationNeedsKomiInput() {
  cout << "  Komi compensation needs a net that sees komi" << endl;
  Logger logger(nullptr, false, false, false);
  //A random net, which reports the latest supported Quoridor I/O version.
  NNEvaluator* nnEval = TestSearchCommon::startNNEval(
    "/dev/null", logger, "testquoridorselfplay compensation", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN,
    0, false, false, false, true, false
  );
  const bool seesKomi = PlayUtils::nnEvalSeesKomi(nnEval);
  testAssert(seesKomi == (nnEval->getInputsVersion() >= PlayUtils::MIN_QUORIDOR_IO_VERSION_SEEING_KOMI));
  testAssert(!PlayUtils::nnEvalSeesKomi(NULL));
  cout << "    random net: Quoridor I/O version " << nnEval->getInputsVersion() << ", sees komi: " << (seesKomi ? "yes" : "no") << endl;

  SearchParams params;
  params.maxVisits = 20;
  params.numThreads = 1;
  MatchPairer::BotSpec botSpec;
  botSpec.botIdx = 0;
  botSpec.botName = "bot";
  botSpec.nnEval = nnEval;
  botSpec.baseParams = params;

  Rules rules = Rules::getQuoridorRules();
  rules.komi = 3.5f;
  Board board;
  BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
  ExtraBlackAndKomi extraBlackAndKomi;
  extraBlackAndKomi.komiMean = rules.komi;
  extraBlackAndKomi.komiStdev = 0;
  extraBlackAndKomi.makeGameFair = true;
  OtherGameProperties otherGameProps;
  otherGameProps.isFork = true;
  otherGameProps.allowPolicyInit = false;
  PlaySettings playSettings;
  playSettings.compensateKomiVisits = 10;
  Rand rand("testquoridorselfplay compensation game");
  auto shouldStop = []() noexcept { return false; };
  FinishedGameData* gameData = Play::runGame(
    board, P_BLACK, hist, extraBlackAndKomi, botSpec, botSpec, "testquoridorselfplay search",
    false, true, logger, false, false, 2, shouldStop, nullptr, playSettings, otherGameProps, rand, nullptr, nullptr
  );
  const float startKomi = gameData->startHist.rules.komi;
  cout << "    fork game komi 3.5 -> " << startKomi << endl;
  testAssert(Rules::isValidKomi(startKomi));
  if(!seesKomi)
    testAssert(startKomi == 3.5f);
  delete gameData;
  delete nnEval;
}

//------------------------------------------------------------------------------------------------

static Board boardWithPawns(const string& white, const string& black) {
  Board start;
  nlohmann::json j = Board::toJson(start);
  vector<int> colors = j["colors"].get<vector<int>>();
  colors[start.whitePawnLoc] = C_EMPTY;
  colors[start.blackPawnLoc] = C_EMPTY;
  Loc w = Location::ofString(white, start);
  Loc b = Location::ofString(black, start);
  colors[w] = C_WHITE;
  colors[b] = C_BLACK;
  j["colors"] = colors;
  j["whitePawnLoc"] = w;
  j["blackPawnLoc"] = b;
  return Board::ofJson(j);
}

//The gatekeeper scores a game 1 / 0 for a win and exactly 0.5 each for a draw, whatever drawEquivalentWinsForWhite
//and noResultUtilityForWhite are.
static void testGatekeeperScoring() {
  cout << "  Gatekeeper scoring" << endl;
  Rules rules = Rules::getQuoridorRules();
  {
    //White on e8 moves to e9: White wins.
    Board board = boardWithPawns("e8", "c5");
    BoardHistory hist(board, P_WHITE, rules, 0, BoardHistoryModes());
    Loc goal = Location::ofString("e9", board);
    testAssert(hist.isLegal(board, goal, P_WHITE));
    hist.makeBoardMoveAssumeLegal(board, goal, P_WHITE, NULL);
    testAssert(hist.isGameFinished && hist.winner == P_WHITE);
    testAssert(PlayUtils::whitePointsOfGame(hist) == 1.0);
    //With komi -4.5 the same arrival (W+4, t = 4) is a Black win.
    hist.setKomi(-4.5f);
    testAssert(hist.winner == P_BLACK);
    testAssert(PlayUtils::whitePointsOfGame(hist) == 0.0);
  }
  {
    //Two pawn moves with maxPlies = 2: a draw by the ply limit.
    Rules r = rules;
    r.maxPlies = 2;
    Board board;
    BoardHistory hist(board, P_BLACK, r, 0, BoardHistoryModes());
    for(Player pla : {P_BLACK, P_WHITE}) {
      Loc loc = Location::ofString(pla == P_BLACK ? "d9" : "d1", board);
      testAssert(hist.isLegal(board, loc, pla));
      hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    }
    testAssert(hist.isGameFinished && hist.isDraw() && hist.winner == C_EMPTY);
    testAssert(PlayUtils::whitePointsOfGame(hist) == 0.5);
  }
  {
    //A game cut off before its end.
    Board board;
    BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
    testAssert(PlayUtils::whitePointsOfGame(hist) == 0.5);
    hist.endAndScoreGameNow(board);
    testAssert(PlayUtils::whitePointsOfGame(hist) == 0.5);
  }
}

}

void Tests::runQuoridorSelfplayTests() {
  cout << "Running Quoridor self-play setup tests" << endl;
  testDefaultsOff();
  testDistributions();
  testFindEvenKomi();
  testCompensationNeedsKomiInput();
  testGatekeeperScoring();
}
