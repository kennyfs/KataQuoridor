/*
 * testquoridorscore.cpp
 * Phase 2 tests for the Quoridor margin (see QuoridorNN in neuralnet/quoridornn.h):
 *  - terminal-score test: scripted games won by each color end with finalWhiteMinusBlackScore equal to the loser's
 *    distance, positive iff White won; a cutoff draw scores 0
 *  - margin-target test: the training writer's game-margin target (global column 21, weight column 29) has that
 *    margin, from the row's nextPlayer perspective
 *  - anti-dithering search test: with an untrained net (so all guidance comes from terminal scores), the engine
 *    converts a won race quickly when score utility is on. The same test with score utility off is reported.
 */

#include "../tests/tests.h"

#include "../dataio/trainingwrite.h"
#include "../neuralnet/quoridornn.h"
#include "../search/search.h"
#include "../tests/testsearchcommon.h"

using namespace std;
using namespace TestCommon;

namespace {

static constexpr int SIZE = Board::DEFAULT_LEN;

//A position where both players have 0 walls left: all 20 walls are on the board, out of the way at the left and
//right edges, so the pawns (which stay in columns 3..5) race with plenty of room to sidestep in columns 2..6.
//Vertical walls at anchor columns 0,1,6,7 (rows 0,2,4,6) and horizontal walls at anchor column 0 (rows 1,3,5,7).
static Board makeRaceBoard(int whiteC, int whiteR, int blackC, int blackR) {
  Board start;
  nlohmann::json j = Board::toJson(start);
  vector<int> colors = j["colors"].get<vector<int>>();
  colors[start.whitePawnLoc] = C_EMPTY;
  colors[start.blackPawnLoc] = C_EMPTY;
  Loc w = Location::pawnLoc(whiteC, whiteR, SIZE);
  Loc b = Location::pawnLoc(blackC, blackR, SIZE);
  colors[w] = C_WHITE;
  colors[b] = C_BLACK;

  vector<bool> vWalls(64, false), hWalls(64, false);
  int numWalls = 0;
  auto fence = [&](int x, int y) { colors[Location::getLoc(x, y, SIZE)] = C_FENCE; };
  for(int c : {0, 1, 6, 7}) {
    for(int r : {0, 2, 4, 6}) {
      vWalls[c * 8 + r] = true;
      fence(2 * c + 1, 2 * r + 1);
      fence(2 * c + 1, 2 * r);
      fence(2 * c + 1, 2 * r + 2);
      numWalls++;
    }
  }
  for(int r : {1, 3, 5, 7}) {
    hWalls[0 * 8 + r] = true;
    fence(1, 2 * r + 1);
    fence(0, 2 * r + 1);
    fence(2, 2 * r + 1);
    numWalls++;
  }
  testAssert(numWalls == 2 * Board::MAX_FENCE_NUM);

  j["colors"] = colors;
  j["vWalls"] = vWalls;
  j["hWalls"] = hWalls;
  j["whitePawnLoc"] = w;
  j["blackPawnLoc"] = b;
  j["whiteFences"] = 0;
  j["blackFences"] = 0;
  return Board::ofJson(j);
}

//A game where each player's moves are given as a string of F(orward), L(eft), R(ight); Black moves first.
struct ScriptedGame {
  vector<Board> boards;      //boards[i] = position before ply i, boards.back() = final position
  vector<BoardHistory> hists; //same indexing
  vector<Loc> moves;
  Board board;
  BoardHistory hist;
  Player winner() const { return hist.winner; }
};

static ScriptedGame playScripted(const string& blackScript, const string& whiteScript) {
  Board board;
  BoardHistory hist(board, P_BLACK, Rules::getTrompTaylorish(), 0, BoardHistoryModes(false, false));
  ScriptedGame g{{}, {}, {}, board, hist};
  Player pla = P_BLACK;
  size_t bi = 0, wi = 0;
  while(!g.hist.isGameFinished) {
    const string& script = pla == P_BLACK ? blackScript : whiteScript;
    size_t& idx = pla == P_BLACK ? bi : wi;
    testAssert(idx < script.size());
    char c = script[idx++];
    Loc from = pla == P_BLACK ? g.board.blackPawnLoc : g.board.whitePawnLoc;
    int x = Location::getX(from, SIZE);
    int y = Location::getY(from, SIZE);
    int forward = pla == P_WHITE ? 2 : -2;
    if(c == 'F') y += forward;
    else if(c == 'L') x -= 2;
    else if(c == 'R') x += 2;
    else testAssert(false);
    Loc to = Location::getLoc(x, y, SIZE);
    testAssert(g.hist.isLegal(g.board, to, pla));
    g.boards.push_back(g.board);
    g.hists.push_back(g.hist);
    g.moves.push_back(to);
    g.hist.makeBoardMoveAssumeLegal(g.board, to, pla, NULL);
    pla = getOpp(pla);
  }
  g.boards.push_back(g.board);
  g.hists.push_back(g.hist);
  return g;
}

//White wins on ply 16; Black made 8 moves, only 3 of them forward, so Black is still 8-3 = 5 away.
static ScriptedGame whiteWinsGame() { return playScripted("LFFFLRLR", "FFFFFFFF"); }
static const float WHITE_WINS_MARGIN = 5.0f;
//Black wins on ply 15; White made 7 moves, only 2 of them forward, so White is still 8-2 = 6 away.
static ScriptedGame blackWinsGame() { return playScripted("FFFFFFFF", "LFFLRLR"); }
static const float BLACK_WINS_MARGIN = -6.0f;

//------------------------------------------------------------------------------------------------

static void testTerminalScore() {
  cout << "Running Quoridor terminal score test" << endl;
  {
    ScriptedGame g = whiteWinsGame();
    testAssert(g.hist.isGameFinished && !g.hist.isNoResult);
    testAssert(g.hist.winner == P_WHITE);
    testAssert(g.hist.isScored);
    testAssert(g.hist.finalWhiteMinusBlackScore == WHITE_WINS_MARGIN);
    testAssert(g.board.getShortestPathDistance(P_BLACK) == 5);
    //Search turns it into a white-perspective score value that favors White.
    testAssert(ScoreValue::whiteScoreValueOfScoreSmoothNoDrawAdjust(g.hist.finalWhiteMinusBlackScore, 0.0, 2.0, g.board.sqrtBoardArea()) > 0.0);
  }
  {
    ScriptedGame g = blackWinsGame();
    testAssert(g.hist.isGameFinished && !g.hist.isNoResult);
    testAssert(g.hist.winner == P_BLACK);
    testAssert(g.hist.isScored);
    testAssert(g.hist.finalWhiteMinusBlackScore == BLACK_WINS_MARGIN);
    testAssert(g.board.getShortestPathDistance(P_WHITE) == 6);
    testAssert(ScoreValue::whiteScoreValueOfScoreSmoothNoDrawAdjust(g.hist.finalWhiteMinusBlackScore, 0.0, 2.0, g.board.sqrtBoardArea()) < 0.0);
  }
  {
    //A game cut off before anyone wins is a draw with margin 0.
    Board board;
    BoardHistory hist(board, P_BLACK, Rules::getTrompTaylorish(), 0, BoardHistoryModes(false, false));
    hist.makeBoardMoveAssumeLegal(board, Location::pawnLoc(4, 7, SIZE), P_BLACK, NULL);
    testAssert(!hist.isGameFinished);
    hist.endAndScoreGameNow(board);
    testAssert(hist.isGameFinished && hist.winner == C_EMPTY);
    testAssert(hist.finalWhiteMinusBlackScore == 0.0f);
    testAssert(board.whiteMarginWhenWonBy(C_EMPTY) == 0.0f);
  }
  cout << "Quoridor terminal score test passed" << endl;
}

//------------------------------------------------------------------------------------------------

//Adds the row for the position before ply plyIdx of the scripted game, and returns the (target, weight) of the
//game-margin column.
static pair<float, float> marginTargetOfRow(
  const ScriptedGame& g, int plyIdx, bool rowHasLead, bool useFutureBoards, bool asDraw
) {
  TrainingWriteBuffers buffers(
    1, 2, QuoridorNN::NUM_FEATURES_SPATIAL_V1, QuoridorNN::NUM_FEATURES_GLOBAL_V1, QuoridorNN::MODEL_LEN, QuoridorNN::MODEL_LEN, false
  );
  BoardHistory endHist = g.hist;
  if(asDraw) {
    endHist = g.hists[plyIdx];
    endHist.endAndScoreGameNow(g.boards[plyIdx]);
  }
  Player nextPla = (plyIdx % 2 == 0) ? P_BLACK : P_WHITE;
  vector<ValueTargets> whiteValueTargets(g.boards.size());
  for(ValueTargets& t : whiteValueTargets) {
    t.win = endHist.winner == P_WHITE ? 1.0f : 0.0f;
    t.loss = 1.0f - t.win;
    t.noResult = 0.0f;
    t.score = endHist.finalWhiteMinusBlackScore;
    //The value targets of a finished game carry the final margin as lead (Play::runGame's finalValueTargets).
    t.hasLead = rowHasLead;
    t.lead = endHist.finalWhiteMinusBlackScore;
  }
  vector<QValueTargets> whiteQValueTargets(g.boards.size());
  vector<PolicyTargetMove> policyTarget;
  policyTarget.push_back(PolicyTargetMove(g.moves[plyIdx], 100));
  NNRawStats nnRawStats;
  nnRawStats.whiteWinLoss = 0.0;
  nnRawStats.whiteScoreMean = 0.0;
  nnRawStats.policyEntropy = 0.0;
  Rand rand("quoridorMarginTargetTest");
  buffers.addRow(
    g.boards[plyIdx], g.hists[plyIdx], nextPla,
    g.hists[0], endHist,
    plyIdx, 1.0f, 100,
    &policyTarget, NULL,
    0.1, 1.0, 1.0,
    whiteValueTargets, whiteQValueTargets,
    plyIdx, 1.0f, 1.0f, 1.0f,
    nnRawStats,
    &g.boards.back(), NULL, NULL, NULL,
    useFutureBoards ? &g.boards : NULL,
    !useFutureBoards, 0, 0.5, C_EMPTY, 0.0,
    Hash128(), vector<ChangedNeuralNet*>(),
    asDraw, 0, FinishedGameData::MODE_NORMAL,
    NULL, rand, ReanalysisData()
  );
  const float* gt = buffers.globalTargetsNC.data;
  return make_pair(gt[21], gt[29]);
}

static void testMarginTarget() {
  cout << "Running Quoridor margin target test" << endl;
  ScriptedGame whiteWon = whiteWinsGame();
  ScriptedGame blackWon = blackWinsGame();
  //Row 0: Black to move. Row 1: White to move. Target is from the row's nextPlayer perspective.
  for(bool rowHasLead : {false, true}) {
    pair<float, float> r;
    r = marginTargetOfRow(whiteWon, 0, rowHasLead, true, false);
    testAssert(r.first == -WHITE_WINS_MARGIN && r.second == 1.0f);  //Black to move, White won by 5
    r = marginTargetOfRow(whiteWon, 1, rowHasLead, true, false);
    testAssert(r.first == WHITE_WINS_MARGIN && r.second == 1.0f);   //White to move, White won by 5
    r = marginTargetOfRow(blackWon, 0, rowHasLead, true, false);
    testAssert(r.first == -BLACK_WINS_MARGIN && r.second == 1.0f);  //Black to move, Black won by 6
    r = marginTargetOfRow(blackWon, 1, rowHasLead, true, false);
    testAssert(r.first == BLACK_WINS_MARGIN && r.second == 1.0f);   //White to move, Black won by 6
  }
  //Side positions have no future boards and no lead: the final margin of the actual game is unknown, weight 0.
  {
    pair<float, float> r = marginTargetOfRow(whiteWon, 0, false, false, false);
    testAssert(r.second == 0.0f);
  }
  //A game cut off as a draw has no margin to learn from.
  {
    pair<float, float> r = marginTargetOfRow(whiteWon, 1, false, true, true);
    testAssert(r.second == 0.0f && r.first == 0.0f);
  }
  cout << "Quoridor margin target test passed" << endl;
}

//------------------------------------------------------------------------------------------------

struct RaceCase {
  string name;
  int whiteC, whiteR, blackC, blackR;
  Player pla;
};

//Both players have 0 walls left, so this is a pure pawn race.
//The side to move wins every one of these, and can even waste several moves and still win.
static const vector<RaceCase> RACE_CASES = {
  {"White d=2 vs Black d=8", 3, 6, 5, 8, P_WHITE},
  {"Black d=3 vs White d=8", 3, 0, 5, 3, P_BLACK},
  //The pawns face each other, so the mover can jump: BFS distance 3, but only 2 moves are needed.
  {"White d=3 (2 with a jump) vs Black d=6", 4, 5, 4, 6, P_WHITE},
  {"Black d=3 (2 with a jump) vs White d=6", 4, 2, 4, 3, P_BLACK},
};

static int shortestDist(const Board& board, Player pla) {
  return board.getShortestPathDistance(pla);
}

//Self-play from the position with the given params; returns the number of plies until someone reaches their goal
//(maxPlies + 1 if nobody did) and sets winner.
static int selfplayRace(NNEvaluator* nnEval, Logger& logger, const RaceCase& c, const SearchParams& params, int maxPlies, Player& winner) {
  Board board = makeRaceBoard(c.whiteC, c.whiteR, c.blackC, c.blackR);
  BoardHistory hist(board, c.pla, Rules::getTrompTaylorish(), 0, BoardHistoryModes(false, false));
  Search* bot = new Search(params, nnEval, &logger, "quoridorAntiDitheringTest");
  Player pla = c.pla;
  int plies = 0;
  while(!hist.isGameFinished && plies < maxPlies) {
    bot->setPosition(pla, board, hist);
    Loc move = bot->runWholeSearchAndGetMove(pla);
    testAssert(hist.isLegal(board, move, pla));
    hist.makeBoardMoveAssumeLegal(board, move, pla, NULL);
    pla = getOpp(pla);
    plies++;
  }
  delete bot;
  winner = hist.isGameFinished ? hist.winner : C_EMPTY;
  return hist.isGameFinished ? plies : maxPlies + 1;
}

static void testAntiDithering() {
  cout << "Running Quoridor anti-dithering search test" << endl;
  bool logToStdout = false, logToStderr = false, logTime = false;
  Logger logger(nullptr, logToStdout, logToStderr, logTime);
  //A random net: gaussian noise for the policy and for the value, so any progress comes from terminal scores.
  NNEvaluator* nnEval = TestSearchCommon::startNNEval(
    "/dev/null", logger, "quoridorAntiDitheringNN", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN,
    0, false, false, false, true, false
  );

  //Same values as configs/training/selfplay_quoridor.cfg.
  SearchParams on;
  on.maxVisits = 300;
  on.numThreads = 1;
  //Random gaussian logits give a spiky policy, unlike an untrained net's near-uniform one. Flatten it, so the search
  //explores every move and has to be guided by what it finds at terminal nodes rather than by the random priors.
  on.nnPolicyTemperature = 8.0f;
  on.staticScoreUtilityFactor = 0.10;
  on.dynamicScoreUtilityFactor = 0.30;
  on.dynamicScoreCenterZeroWeight = 0.25;
  on.dynamicScoreCenterScale = 0.50;
  SearchParams off = on;
  off.staticScoreUtilityFactor = 0.0;
  off.dynamicScoreUtilityFactor = 0.0;

  const int slackMoves = 1;  //Allowed to spend at most this many moves more than the shortest path.
  int totalOn = 0, totalOff = 0, totalOptimal = 0;
  for(const RaceCase& c : RACE_CASES) {
    Board board = makeRaceBoard(c.whiteC, c.whiteR, c.blackC, c.blackR);
    testAssert(board.whiteFences == 0 && board.blackFences == 0);
    int d = shortestDist(board, c.pla);
    testAssert(d < shortestDist(board, getOpp(c.pla)));  //the side to move wins the race
    int maxPlies = 2 * d - 1 + 2 * 12;
    int allowedPlies = 2 * d - 1 + 2 * slackMoves;

    Player winnerOn, winnerOff;
    int pliesOn = selfplayRace(nnEval, logger, c, on, maxPlies, winnerOn);
    int pliesOff = selfplayRace(nnEval, logger, c, off, maxPlies, winnerOff);
    cout << "  " << c.name << ": shortest path " << d << " moves (" << (2*d-1) << " plies); "
         << "score utility ON: " << pliesOn << " plies (winner " << PlayerIO::playerToString(winnerOn) << "), "
         << "OFF: " << pliesOff << " plies" << (pliesOff > maxPlies ? "+ (never finished)" : "")
         << " (winner " << PlayerIO::playerToString(winnerOff) << ")" << endl;
    testAssert(winnerOn == c.pla);
    testAssert(pliesOn <= allowedPlies);
    totalOn += pliesOn;
    totalOff += pliesOff;
    totalOptimal += 2 * d - 1;
  }
  cout << "  total plies: shortest-path bound " << totalOptimal << ", score utility ON " << totalOn << ", OFF " << totalOff << endl;
  delete nnEval;
  cout << "Quoridor anti-dithering search test passed" << endl;
}

}  // namespace

void Tests::runQuoridorScoreTests() {
  cout << "=== Running Quoridor margin / score utility tests ===" << endl;
  testTerminalScore();
  testMarginTarget();
  testAntiDithering();
  cout << "=== Quoridor margin / score utility tests passed ===" << endl;
}
