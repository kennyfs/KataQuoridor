/*
 * testquoridorrepetition.cpp
 * Tests for the repetition draw rule, Rules::repetitionDrawCount (docs/QuoridorIOv2.md section 10):
 *  - a 4-ply shuffle is drawn exactly at the N-th occurrence of a position, and not with N = 0
 *  - a position is pawns + walls + walls left + side to move (Board::getSitHash), not the ply
 *  - a wall placement resets the candidates; undo (replay), copies and copyToInitial keep the counts consistent
 *  - repetition vs maxPlies draws are told apart
 *  - Rules string / JSON / hash and SGF (RU, RE, DR) round trip
 *  - search sees the draw of the repeating move from a root already at count N - 1
 *  - graph search: a transposition (same position, same ply) with different repetition counts gets different
 *    graph hashes, although its state hash is the same
 */

#include "../tests/tests.h"

#include "../dataio/sgf.h"
#include "../game/graphhash.h"
#include "../search/search.h"
#include "../tests/testsearchcommon.h"

using namespace std;

namespace {

static bool approxEqual(double x, double y) {
  return std::fabs(x - y) < 1e-4;
}

static Rules rulesWithRepetition(int n, int fences = Rules::DEFAULT_INITIAL_FENCES) {
  Rules rules = Rules::getQuoridorRules();
  rules.repetitionDrawCount = n;
  rules.blackInitialFences = fences;
  rules.whiteInitialFences = fences;
  return rules;
}

struct Game {
  Board board;
  BoardHistory hist;
  Player pla;
  Game(const Rules& rules) {
    board = Board();
    board.setFencesLeft(rules.blackInitialFences, rules.whiteInitialFences);
    hist = BoardHistory(board, P_BLACK, rules, 0, BoardHistoryModes());
    pla = P_BLACK;
  }
  void play(const string& move) {
    Loc loc = Location::ofString(move, board);
    testAssert(hist.isLegal(board, loc, pla));
    hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    pla = getOpp(pla);
  }
  void playAll(const vector<string>& moves) {
    for(const string& m : moves)
      play(m);
  }
};

//Black e9 <-> e8 and White e1 <-> e2: after every 4 plies the start position recurs.
static const vector<string> SHUFFLE = {"e8", "e2", "e9", "e1"};

static vector<string> shuffleMoves(int n) {
  vector<string> moves;
  for(int i = 0; i < n; i++)
    moves.push_back(SHUFFLE[i % 4]);
  return moves;
}

//------------------------------------------------------------------------------------------------

static void testShuffleDraw() {
  cout << "  Shuffle reaches the draw at the N-th occurrence" << endl;
  for(int n : {2, 3, 5}) {
    Game g(rulesWithRepetition(n));
    testAssert(g.hist.currentPositionRepetitionCount() == 1);
    //The start position occurs again after 4k plies, for the (k+1)-th time.
    int drawPly = 4 * (n - 1);
    for(int ply = 1; ply <= drawPly; ply++) {
      testAssert(!g.hist.isGameFinished);
      g.play(SHUFFLE[(ply - 1) % 4]);
      //Positions after plies p and p - 4 are equal (and the start position is ply 0's).
      testAssert(g.hist.currentPositionRepetitionCount() == ply / 4 + 1);
    }
    testAssert(g.hist.isGameFinished && g.hist.isDraw() && g.hist.isRepetitionDraw() && !g.hist.isMaxPliesDraw());
    testAssert(g.hist.winner == C_EMPTY && g.hist.isScored && !g.hist.isNoResult);
    testAssert(g.hist.finalWhiteLead == 0.0f && g.hist.finalWhiteMinusBlackScore == 0.0f);
    testAssert(g.hist.getCurrentTurnNumber() == drawPly);
    testAssert(g.hist.currentPositionRepetitionCount() == n);
    //No more moves.
    testAssert(!g.hist.isLegal(g.board, Location::ofString("e8", g.board), P_BLACK));
  }

  //N = 0 (the default): never a repetition draw, but the count is still kept.
  {
    Rules rules = Rules::getQuoridorRules();
    testAssert(rules.repetitionDrawCount == 0);
    Game g(rules);
    g.playAll(shuffleMoves(40));
    testAssert(!g.hist.isGameFinished);
    testAssert(g.hist.currentPositionRepetitionCount() == 11);
    testAssert(g.hist.positionsSinceLastWall.size() == 41);
  }

  //Ply limit and repetition on the same ply: a repetition draw.
  {
    Rules rules = rulesWithRepetition(3);
    rules.maxPlies = 8;
    Game g(rules);
    g.playAll(shuffleMoves(8));
    testAssert(g.hist.isRepetitionDraw() && !g.hist.isMaxPliesDraw());
    //One ply less to the limit: a maxPlies draw.
    rules.maxPlies = 7;
    Game g2(rules);
    g2.playAll(shuffleMoves(7));
    testAssert(g2.hist.isMaxPliesDraw() && !g2.hist.isRepetitionDraw());
  }

  //A pawn reaching its goal wins even if that position repeats (it can't, but the goal check comes first).
  //A position is the side to move too: the same pawns with the other side to move are a different position.
  {
    Board b;
    testAssert(b.getSitHash(P_BLACK) != b.getSitHash(P_WHITE));
    //Walls left are part of the position.
    Board c;
    c.setFencesLeft(9, 10);
    testAssert(b.getSitHash(P_BLACK) != c.getSitHash(P_BLACK));
  }
}

//------------------------------------------------------------------------------------------------

static void testWallResets() {
  cout << "  A wall placement resets the candidates" << endl;
  Game g(rulesWithRepetition(3));
  g.playAll(shuffleMoves(4));
  testAssert(g.hist.currentPositionRepetitionCount() == 2);
  testAssert(g.hist.positionsSinceLastWall.size() == 5);
  g.play("a2h");
  testAssert(g.hist.positionsSinceLastWall.size() == 1);
  testAssert(g.hist.currentPositionRepetitionCount() == 1);
  //White and Black shuffle again: the start position (with a wall) can't recur, the new ones count from 1.
  //After the wall it is White to move; White e1->e2, Black e9->e8, White e2->e1, Black e8->e9.
  vector<string> after = {"e2", "e8", "e1", "e9"};
  for(int round = 0; round < 2; round++) {
    for(size_t i = 0; i < after.size(); i++) {
      testAssert(!g.hist.isGameFinished);
      g.play(after[i]);
    }
  }
  //The position right after the wall occurred after plies 5, 9 and 13: the third time is a draw.
  testAssert(g.hist.isRepetitionDraw());
  testAssert(g.hist.getCurrentTurnNumber() == 13);
  testAssert(g.hist.positionsSinceLastWall.size() == 9);
}

//------------------------------------------------------------------------------------------------

static void testUndoAndCopies() {
  cout << "  Undo, copies and copyToInitial keep the counts consistent" << endl;
  Rules rules = rulesWithRepetition(3);
  //A shuffle, Black's wall on ply 5, and a shuffle until the position after the wall occurs a third time.
  vector<string> moves = shuffleMoves(4);
  for(const string& m : {"a2h", "e2", "e8", "e1", "e9", "e2", "e8", "e1", "e9"})
    moves.push_back(m);
  Game full(rules);
  vector<int> counts;
  vector<size_t> sizes;
  for(const string& m : moves) {
    if(full.hist.isGameFinished)
      break;
    full.play(m);
    counts.push_back(full.hist.currentPositionRepetitionCount());
    sizes.push_back(full.hist.positionsSinceLastWall.size());
  }
  //Undo as QTP does it: rebuild from the initial position and replay the first k moves.
  for(size_t k = 0; k <= counts.size(); k++) {
    BoardHistory hist = full.hist.copyToInitial();
    testAssert(hist.currentPositionRepetitionCount() == 1 && hist.positionsSinceLastWall.size() == 1);
    Board board = hist.initialBoard;
    Player pla = hist.initialPla;
    for(size_t i = 0; i < k; i++) {
      testAssert(hist.makeBoardMoveTolerant(board, full.hist.moveHistory[i].loc, pla));
      pla = getOpp(pla);
    }
    if(k > 0) {
      testAssert(hist.currentPositionRepetitionCount() == counts[k-1]);
      testAssert(hist.positionsSinceLastWall.size() == sizes[k-1]);
    }
    //A copy continues identically.
    BoardHistory copy(hist);
    testAssert(copy.positionsSinceLastWall == hist.positionsSinceLastWall);
    testAssert(copy.currentPositionRepetitionCount() == hist.currentPositionRepetitionCount());
  }
  testAssert(full.hist.isRepetitionDraw() && counts.size() == moves.size());
  //clear() resets to the new position.
  BoardHistory h(full.hist);
  Board b;
  h.clear(b, P_BLACK, rules, 0);
  testAssert(h.currentPositionRepetitionCount() == 1 && h.positionsSinceLastWall.size() == 1 && !h.isRepetitionDrawFlag);
}

//------------------------------------------------------------------------------------------------

static bool contains(const string& s, const string& sub) {
  return s.find(sub) != string::npos;
}

static void testRulesAndSgf() {
  cout << "  Rules string, JSON, hash and SGF" << endl;
  Rules r = rulesWithRepetition(3);
  testAssert(r.toString() == "Quoridor:repetitionDrawCount=3");
  testAssert(r.toStringNoKomi() == "Quoridor:repetitionDrawCount=3");
  testAssert(Rules::parseRules(r.toString()) == r);
  testAssert(Rules::parseRules(r.toJsonString()) == r);
  testAssert(Rules::parseRules("{\"repetitionDrawCount\":3}") == r);
  testAssert(Rules::updateRules("repetitionDrawCount", "3", Rules::getQuoridorRules()) == r);
  testAssert(Rules::updateRules("repetitionDrawCount", "0", r) == Rules::getQuoridorRules());
  testAssert(r != Rules::getQuoridorRules() && !r.equalsIgnoringKomi(Rules::getQuoridorRules()));
  testAssert(Rules::getQuoridorRules().toJsonStringNoKomi() ==
    "{\"blackInitialWalls\":10,\"maxPlies\":300,\"repetitionDrawCount\":0,\"timeBonusPerPly\":0.0,\"whiteInitialWalls\":10}");
  Rules buf;
  testAssert(!Rules::tryParseRules("Quoridor:repetitionDrawCount=1", buf));
  testAssert(!Rules::tryParseRules("Quoridor:repetitionDrawCount=-1", buf));
  testAssert(!Rules::tryParseRules("Quoridor:repetitionDrawCount=1001", buf));
  testAssert(Rules::tryParseRules("Quoridor:repetitionDrawCount=2", buf) && buf.repetitionDrawCount == 2);

  //The rules hash (graph search and NN cache) depends on the rule.
  {
    Board board;
    BoardHistory off(board, P_BLACK, Rules::getQuoridorRules(), 0, BoardHistoryModes());
    BoardHistory on(board, P_BLACK, r, 0, BoardHistoryModes());
    testAssert(
      BoardHistory::getSituationRulesAndKoHash(board, off, P_BLACK, 0.5) !=
      BoardHistory::getSituationRulesAndKoHash(board, on, P_BLACK, 0.5)
    );
  }

  //SGF: RU carries the rule, RE[0] and DR[repetition] the result.
  {
    Game g(r);
    g.playAll(shuffleMoves(8));
    testAssert(g.hist.isRepetitionDraw());
    ostringstream out;
    WriteSgf::writeSgf(out, "black", "white", g.hist, NULL, true, false);
    string sgf = out.str();
    testAssert(contains(sgf, "RU[Quoridor:repetitionDrawCount=3]RE[0]DR[repetition]"));
    std::unique_ptr<CompactSgf> c = CompactSgf::parse(sgf);
    testAssert(c->getRulesOrFail() == r);
    Board board;
    Player pla;
    BoardHistory hist;
    c->setupBoardAndHistAssumeLegal(c->getRulesOrFail(), board, pla, hist, (int64_t)c->moves.size(), BoardHistoryModes());
    testAssert(hist.isRepetitionDraw() && hist.getCurrentTurnNumber() == 8);
    testAssert(WriteSgf::drawReason(hist) == "repetition");
  }
  //A maxPlies draw: DR[maxPlies]; a game cut off by a controller: DR[cutoff]; a win: no DR.
  {
    Rules m = Rules::getQuoridorRules();
    m.maxPlies = 6;
    Game g(m);
    g.playAll(shuffleMoves(6));
    ostringstream out;
    WriteSgf::writeSgf(out, "black", "white", g.hist, NULL, true, false);
    testAssert(contains(out.str(), "RE[0]DR[maxPlies]"));

    Game cut(Rules::getQuoridorRules());
    cut.playAll(shuffleMoves(3));
    cut.hist.endAndScoreGameNow(cut.board);
    testAssert(WriteSgf::drawReason(cut.hist) == "cutoff");
    testAssert(WriteSgf::drawReason(Game(m).hist) == "");
  }
  //Old SGFs (no rule in RU): off.
  {
    std::unique_ptr<CompactSgf> c = CompactSgf::parse("(;FF[4]GM[1]SZ[17]KM[-0.5]RU[Quoridor:maxPlies=400];B[io])");
    testAssert(c->getRulesOrFail().repetitionDrawCount == 0);
  }
}

//------------------------------------------------------------------------------------------------

static void testGraphHashTransposition() {
  cout << "  Graph hash: a transposition with different repetition counts" << endl;
  Rules rules = rulesWithRepetition(3, 0);
  //Both paths reach Black e9, White e2, White to move, on ply 7.
  //Path A goes through it twice (plies 3 and 7); path B once.
  Game a(rules);
  a.playAll({"e8", "e2", "e9", "e1", "e8", "e2", "e9"});
  Game b(rules);
  b.playAll({"e8", "f1", "d8", "f2", "d9", "e2", "e9"});
  testAssert(a.board.pos_hash == b.board.pos_hash && a.pla == b.pla);
  testAssert(a.hist.getCurrentTurnNumber() == b.hist.getCurrentTurnNumber());
  testAssert(a.hist.currentPositionRepetitionCount() == 2 && b.hist.currentPositionRepetitionCount() == 1);

  //The state hash alone (board, ply, rules) can't tell them apart...
  testAssert(GraphHash::getStateHash(a.hist, a.pla, 0.5) == GraphHash::getStateHash(b.hist, b.pla, 0.5));
  //...but White e1 is the third occurrence of the start position on path A only.
  Game a2 = a;
  a2.play("e1");
  Game b2 = b;
  b2.play("e1");
  testAssert(a2.hist.isRepetitionDraw());
  testAssert(!b2.hist.isGameFinished && b2.hist.currentPositionRepetitionCount() == 2);
  //Graph search must not merge them: the graph hash chains the path after pawn moves.
  for(int repBound : {3, 11, 50}) {
    testAssert(GraphHash::getGraphHashFromScratch(a.hist, a.pla, repBound, 0.5) != GraphHash::getGraphHashFromScratch(b.hist, b.pla, repBound, 0.5));
    testAssert(GraphHash::getGraphHashFromScratch(a2.hist, a2.pla, repBound, 0.5) != GraphHash::getGraphHashFromScratch(b2.hist, b2.pla, repBound, 0.5));
  }

  //After a wall placement the state hash is the graph hash: the same wall from both positions merges, which is
  //safe because no earlier position can recur.
  {
    Rules wr = rulesWithRepetition(3, 10);
    Game c(wr);
    c.playAll({"e8", "e2", "e9", "e1", "e8", "e2", "e9"});
    Game d(wr);
    d.playAll({"e8", "f1", "d8", "f2", "d9", "e2", "e9"});
    Hash128 hc = GraphHash::getGraphHashFromScratch(c.hist, c.pla, 11, 0.5);
    Hash128 hd = GraphHash::getGraphHashFromScratch(d.hist, d.pla, 11, 0.5);
    testAssert(hc != hd);
    c.play("a2h");
    d.play("a2h");
    testAssert(c.hist.currentPositionRepetitionCount() == 1 && d.hist.currentPositionRepetitionCount() == 1);
    testAssert(GraphHash::getGraphHash(hc, c.hist, c.pla, 11, 0.5) == GraphHash::getGraphHash(hd, d.hist, d.pla, 11, 0.5));
  }

  //And in a search: from both roots, White's e1 is searched as a child; on path A it is a terminal draw, on path B
  //it is not.
  Logger logger(nullptr, false, false, false);
  NNEvaluator* nnEval = TestSearchCommon::startNNEval(
    "/dev/null", logger, "quoridorRepetitionNN", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN,
    0, false, false, false, true, false
  );
  SearchParams params;
  params.maxVisits = 400;
  params.numThreads = 1;
  params.useGraphSearch = true;
  params.nnPolicyTemperature = 8.0f;
  auto searchE1 = [&](const Game& g) {
    Search* search = new Search(params, nnEval, &logger, "quoridorRepetitionSearch");
    search->setPosition(g.pla, g.board, g.hist);
    search->runWholeSearch(g.pla);
    vector<AnalysisData> data;
    search->getAnalysisData(data, 1, false, 2, false);
    Loc e1 = Location::ofString("e1", g.board);
    AnalysisData found;
    bool wasFound = false;
    for(const AnalysisData& d : data) {
      if(d.move == e1) {
        found = d;
        wasFound = true;
      }
    }
    testAssert(wasFound && found.numVisits > 0);
    delete search;
    return found;
  };
  {
    AnalysisData d = searchE1(a);
    testAssert(approxEqual(d.winLossValue, 0.0));
    testAssert(approxEqual(d.lead, 0.0));
    testAssert(approxEqual(d.scoreMean, 0.0));
    testAssert(approxEqual(d.scoreStdev, 0.0));
  }
  {
    AnalysisData d = searchE1(b);
    //A random net's value is (almost surely) not an exact draw.
    testAssert(!(approxEqual(d.winLossValue, 0.0) && approxEqual(d.lead, 0.0) && approxEqual(d.scoreMean, 0.0)));
  }
  delete nnEval;
}

//------------------------------------------------------------------------------------------------

static void testSearchSeesDraw() {
  cout << "  Search sees the repetition draw" << endl;
  Logger logger(nullptr, false, false, false);
  NNEvaluator* nnEval = TestSearchCommon::startNNEval(
    "/dev/null", logger, "quoridorRepetitionNN2", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN,
    0, false, false, false, true, false
  );
  for(bool useGraphSearch : {false, true}) {
    SearchParams params;
    params.maxVisits = 300;
    params.numThreads = 1;
    params.useGraphSearch = useGraphSearch;
    params.nnPolicyTemperature = 8.0f;
    //The root is one move before the third occurrence (its own count is 2 = N - 1).
    Game g(rulesWithRepetition(3, 0));
    g.playAll(shuffleMoves(7));
    testAssert(g.hist.currentPositionRepetitionCount() == 2 && g.pla == P_WHITE);
    Search* search = new Search(params, nnEval, &logger, "quoridorRepetitionSearch2");
    search->setPosition(g.pla, g.board, g.hist);
    search->runWholeSearch(g.pla);
    vector<AnalysisData> data;
    search->getAnalysisData(data, 1, false, 2, false);
    Loc e1 = Location::ofString("e1", g.board);
    bool found = false;
    for(const AnalysisData& d : data) {
      if(d.move != e1)
        continue;
      found = true;
      testAssert(d.numVisits > 0);
      testAssert(approxEqual(d.winLossValue, 0.0));
      testAssert(approxEqual(d.lead, 0.0));
      testAssert(approxEqual(d.scoreMean, 0.0));
    }
    testAssert(found);
    delete search;

    //Without the rule, the same move is not terminal.
    Rules off = rulesWithRepetition(0, 0);
    Game h(off);
    h.playAll(shuffleMoves(7));
    BoardHistory next(h.hist);
    Board nb(h.board);
    next.makeBoardMoveAssumeLegal(nb, e1, P_WHITE, NULL);
    testAssert(!next.isGameFinished);
  }
  delete nnEval;
}

}  // namespace

void Tests::runQuoridorRepetitionTests() {
  cout << "Running Quoridor repetition draw tests" << endl;
  testShuffleDraw();
  testWallResets();
  testUndoAndCopies();
  testRulesAndSgf();
  testGraphHashTransposition();
  testSearchSeesDraw();
  cout << "Quoridor repetition draw tests passed" << endl;
}
