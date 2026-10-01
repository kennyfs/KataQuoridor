/*
 * testquoridoriov3.cpp
 * Tests for the Quoridor I/O v3 repetition inputs (docs/QuoridorIOv3.md):
 *  - the features against a brute-force recount over the whole game (not only since the last wall), on random games
 *    with walls, step-backs and several rule values, including undo (rebuild by replay) and the rule off
 *  - Board::getSitHashAfterPawnMove against playing the move
 *  - mirror symmetry: the inputs of a mirrored game are applyInputSymmetry of the original's
 *  - the NN cache hash: equal positions on equal plies with different repetition state don't share an entry
 *  - the training-data encoding (packed bits + float globals) of a repetition-draw game
 */

#include "../tests/tests.h"

#include "../dataio/trainingwrite.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/quoridornn.h"
#include "../tests/testsearchcommon.h"

using namespace std;

namespace {

static constexpr int V3 = 3;
static constexpr int AREA = 81;

struct Row {
  vector<float> spatial;
  vector<float> global;
};

static Row fill(const Board& board, const BoardHistory& hist, Player pla) {
  Row r;
  r.spatial.resize(QuoridorNN::numSpatialFeatures(V3) * AREA);
  r.global.resize(QuoridorNN::numGlobalFeatures(V3));
  MiscNNInputParams params;
  QuoridorNN::fillRow(board, hist, pla, params, V3, false, r.spatial.data(), r.global.data());
  return r;
}

static int canonPos(Loc loc, Player pla) {
  int c = Location::getX(loc, 17) / 2;
  int r = Location::getY(loc, 17) / 2;
  int rCanon = pla == P_WHITE ? 8 - r : r;
  return rCanon * 9 + c;
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

//A random game where players often step back to where their pawn was two plies ago, sometimes place a wall, and
//otherwise play a random pawn move. Returns the moves; stops when the game ends or after maxMoves.
static vector<Move> randomShuffleGame(Rand& rand, const Rules& rules, int maxMoves, double wallProb) {
  Board board;
  board.setFencesLeft(rules.blackInitialFences, rules.whiteInitialFences);
  BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
  Player pla = P_BLACK;
  vector<Move> moves;
  while(!hist.isGameFinished && (int)moves.size() < maxMoves) {
    vector<Loc> dests = board.getLegalPawnDestinations(pla);
    Loc chosen = Board::NULL_LOC;
    if(rand.nextBool(wallProb)) {
      for(int tries = 0; tries < 20 && chosen == Board::NULL_LOC; tries++) {
        Loc w = rand.nextBool(0.5) ? Location::hWallLoc(rand.nextUInt(8), rand.nextUInt(8), 17) : Location::vWallLoc(rand.nextUInt(8), rand.nextUInt(8), 17);
        if(hist.isLegal(board, w, pla))
          chosen = w;
      }
    }
    if(chosen == Board::NULL_LOC && hist.moveHistory.size() >= 2 && rand.nextBool(0.6)) {
      const Board& twoAgo = hist.getRecentBoard(2);
      Loc prev = pla == P_BLACK ? twoAgo.blackPawnLoc : twoAgo.whitePawnLoc;
      for(Loc d : dests)
        if(d == prev)
          chosen = d;
    }
    if(chosen == Board::NULL_LOC)
      chosen = dests[rand.nextUInt((uint32_t)dests.size())];
    moves.push_back(Move(chosen, pla));
    hist.makeBoardMoveAssumeLegal(board, chosen, pla, NULL);
    pla = getOpp(pla);
  }
  return moves;
}

//------------------------------------------------------------------------------------------------

static void testBruteForce() {
  cout << "  Features vs a brute-force recount, with undo" << endl;
  Rand rand("quoridoriov3 brute force");
  int numRows = 0, numRepeating = 0, numDrawing = 0, numUndo = 0;
  for(int game = 0; game < 80; game++) {
    Rules rules = Rules::getQuoridorRules();
    rules.repetitionDrawCount = game % 5 == 4 ? 0 : 2 + game % 4;  //0 (off), 2, 3, 4, 5
    vector<Move> moves = randomShuffleGame(rand, rules, 120, game % 2 == 0 ? 0.05 : 0.0);

    Board board;
    BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
    Player pla = P_BLACK;
    vector<Hash128> allPositions = {board.getSitHash(pla)};
    vector<Row> rows;
    for(size_t i = 0; i <= moves.size(); i++) {
      if(hist.isGameFinished)
        break;
      Row row = fill(board, hist, pla);
      rows.push_back(row);
      numRows++;
      const int n = rules.repetitionDrawCount;
      //Brute force: over every position of the game so far.
      int count = 0;
      for(const Hash128& h : allPositions)
        count += h == board.getSitHash(pla) ? 1 : 0;
      testAssert(count == hist.currentPositionRepetitionCount());
      testAssert(row.global[QuoridorNN::GLOBAL_REPETITION_ON_V3] == (n > 0 ? 1.0f : 0.0f));
      float expectedProgress = n <= 0 ? 0.0f : n == 2 ? 1.0f : (float)(count - 1) / (float)(n - 2);
      testAssert(std::fabs(row.global[QuoridorNN::GLOBAL_REPETITION_COUNT_V3] - expectedProgress) < 1e-6);
      testAssert(n <= 0 || count < n);

      vector<float> expRepeat(AREA, 0.0f), expDraw(AREA, 0.0f);
      for(Loc d : board.getLegalPawnDestinations(pla)) {
        Board copy(board);
        copy.playMoveAssumeLegal(d, pla);
        Hash128 h = copy.getSitHash(getOpp(pla));
        testAssert(board.getSitHashAfterPawnMove(d, pla) == h);
        int occ = 0;
        for(const Hash128& g : allPositions)
          occ += g == h ? 1 : 0;
        if(n > 0 && occ >= 1)
          expRepeat[canonPos(d, pla)] = 1.0f;
        if(n > 0 && occ + 1 >= n)
          expDraw[canonPos(d, pla)] = 1.0f;
      }
      for(int p = 0; p < AREA; p++) {
        testAssert(row.spatial[QuoridorNN::SPATIAL_REPEATING_MOVE_V3 * AREA + p] == expRepeat[p]);
        testAssert(row.spatial[QuoridorNN::SPATIAL_DRAWING_MOVE_V3 * AREA + p] == expDraw[p]);
        numRepeating += expRepeat[p] > 0 ? 1 : 0;
        numDrawing += expDraw[p] > 0 ? 1 : 0;
      }
      //Wall placements never repeat, so the wall planes carry nothing new: the v2 channels are unchanged.
      {
        vector<float> s2(QuoridorNN::numSpatialFeatures(2) * AREA), g2(QuoridorNN::numGlobalFeatures(2));
        MiscNNInputParams params;
        QuoridorNN::fillRow(board, hist, pla, params, 2, false, s2.data(), g2.data());
        for(size_t k = 0; k < s2.size(); k++)
          testAssert(s2[k] == row.spatial[k]);
        for(size_t k = 0; k < g2.size(); k++)
          testAssert(g2[k] == row.global[k]);
      }

      if(i == moves.size())
        break;
      hist.makeBoardMoveAssumeLegal(board, moves[i].loc, moves[i].pla, NULL);
      pla = getOpp(pla);
      allPositions.push_back(board.getSitHash(pla));

      //Undo, as QTP does it: rebuild from the initial position and replay a prefix; the rows must match.
      if(rand.nextBool(0.1)) {
        size_t k = rand.nextUInt((uint32_t)rows.size());
        BoardHistory h2 = hist.copyToInitial();
        Board b2 = h2.initialBoard;
        Player p2 = h2.initialPla;
        for(size_t j = 0; j < k; j++) {
          testAssert(h2.makeBoardMoveTolerant(b2, moves[j].loc, p2));
          p2 = getOpp(p2);
        }
        Row r2 = fill(b2, h2, p2);
        testAssert(r2.spatial == rows[k].spatial && r2.global == rows[k].global);
        numUndo++;
      }
    }
  }
  cout << "    rows " << numRows << ", repeating-move cells " << numRepeating << ", drawing-move cells " << numDrawing
       << ", undo checks " << numUndo << endl;
  testAssert(numRepeating > 100 && numDrawing > 50 && numUndo > 50);
}

//------------------------------------------------------------------------------------------------

static void testMirror() {
  cout << "  Mirror symmetry" << endl;
  Rand rand("quoridoriov3 mirror");
  int checked = 0, nonzero = 0;
  for(int game = 0; game < 30; game++) {
    Rules rules = Rules::getQuoridorRules();
    rules.repetitionDrawCount = 2 + game % 3;
    vector<Move> moves = randomShuffleGame(rand, rules, 60, 0.05);
    Board a, b;
    BoardHistory ha(a, P_BLACK, rules, 0, BoardHistoryModes());
    BoardHistory hb(b, P_BLACK, rules, 0, BoardHistoryModes());
    Player pla = P_BLACK;
    for(size_t i = 0; i <= moves.size() && !ha.isGameFinished; i++) {
      Row ra = fill(a, ha, pla);
      Row rb = fill(b, hb, pla);
      QuoridorNN::applyInputSymmetry(ra.spatial.data(), V3, false, 1);
      testAssert(ra.spatial == rb.spatial && ra.global == rb.global);
      for(int p = 0; p < 2 * AREA; p++)
        nonzero += rb.spatial[QuoridorNN::SPATIAL_REPEATING_MOVE_V3 * AREA + p] > 0 ? 1 : 0;
      checked++;
      if(i == moves.size())
        break;
      ha.makeBoardMoveAssumeLegal(a, moves[i].loc, pla, NULL);
      hb.makeBoardMoveAssumeLegal(b, mirrorLoc(moves[i].loc), pla, NULL);
      pla = getOpp(pla);
    }
  }
  cout << "    rows " << checked << ", nonzero repetition cells " << nonzero << endl;
  testAssert(nonzero > 50);
}

//------------------------------------------------------------------------------------------------

struct Game {
  Board board;
  BoardHistory hist;
  Player pla;
  Game(const Rules& rules) {
    board.setFencesLeft(rules.blackInitialFences, rules.whiteInitialFences);
    hist = BoardHistory(board, P_BLACK, rules, 0, BoardHistoryModes());
    pla = P_BLACK;
  }
  void play(const vector<string>& ms) {
    for(const string& m : ms) {
      Loc loc = Location::ofString(m, board);
      testAssert(hist.isLegal(board, loc, pla));
      hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
      pla = getOpp(pla);
    }
  }
};

static void testCacheHash() {
  cout << "  NN cache hash" << endl;
  Rules rules = Rules::getQuoridorRules();
  rules.repetitionDrawCount = 3;
  rules.blackInitialFences = 0;
  rules.whiteInitialFences = 0;
  //Black e9 / White e2, White to move, ply 7: twice on the direct shuffle, once on the detour. White's e1 draws on
  //the first path only.
  Game a(rules), b(rules);
  a.play({"e8", "e2", "e9", "e1", "e8", "e2", "e9"});
  b.play({"e8", "f1", "d8", "f2", "d9", "e2", "e9"});
  MiscNNInputParams params;
  testAssert(NNInputs::getHash(a.board, a.hist, a.pla, params) == NNInputs::getHash(b.board, b.hist, b.pla, params));
  testAssert(QuoridorNN::repetitionInputsHash(a.board, a.hist, a.pla) != QuoridorNN::repetitionInputsHash(b.board, b.hist, b.pla));
  Row ra = fill(a.board, a.hist, a.pla), rb = fill(b.board, b.hist, b.pla);
  testAssert(ra.spatial != rb.spatial);
  testAssert(ra.spatial[QuoridorNN::SPATIAL_DRAWING_MOVE_V3 * AREA + canonPos(Location::ofString("e1", a.board), P_WHITE)] == 1.0f);
  testAssert(rb.spatial[QuoridorNN::SPATIAL_DRAWING_MOVE_V3 * AREA + canonPos(Location::ofString("e1", b.board), P_WHITE)] == 0.0f);
  //Rule off: no repetition state at all.
  {
    Rules off = rules;
    off.repetitionDrawCount = 0;
    Game c(off);
    c.play({"e8", "e2", "e9", "e1", "e8", "e2", "e9"});
    testAssert(QuoridorNN::repetitionInputsHash(c.board, c.hist, c.pla) == Hash128());
  }

  //Through NNEvaluator, with the random test net (the latest I/O version).
  Logger logger(nullptr, false, false, false);
  NNEvaluator* nnEval = TestSearchCommon::startNNEval(
    "/dev/null", logger, "quoridorIOv3CacheNN", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN, 0, false, false, false, true, false
  );
  testAssert(nnEval->getInputsVersion() == QuoridorNN::MAX_SUPPORTED_IO_VERSION && nnEval->getInputsVersion() == 3);
  NNResultBuf buf;
  nnEval->evaluate(a.board, a.hist, a.pla, params, buf, false, false);
  Hash128 ha = buf.result->nnHash;
  uint64_t hits0 = nnEval->numCacheHits();
  nnEval->evaluate(b.board, b.hist, b.pla, params, buf, false, false);
  testAssert(buf.result->nnHash != ha);
  testAssert(nnEval->numCacheHits() == hits0);
  nnEval->evaluate(a.board, a.hist, a.pla, params, buf, false, false);
  testAssert(nnEval->numCacheHits() == hits0 + 1);
  //A third path with the same repetition-relevant state as b (once each) shares b's entry.
  Game c(rules);
  c.play({"d9", "f1", "d8", "f2", "e8", "e2", "e9"});
  testAssert(c.board.pos_hash == b.board.pos_hash);
  testAssert(QuoridorNN::repetitionInputsHash(c.board, c.hist, c.pla) == QuoridorNN::repetitionInputsHash(b.board, b.hist, b.pla));
  nnEval->evaluate(c.board, c.hist, c.pla, params, buf, false, false);
  testAssert(nnEval->numCacheHits() == hits0 + 2);
  delete nnEval;
}

//------------------------------------------------------------------------------------------------

static void testTrainingEncoding() {
  cout << "  Training-data encoding of a repetition-draw game" << endl;
  Rules rules = Rules::getQuoridorRules();
  rules.repetitionDrawCount = 3;
  Game g(rules);
  const vector<string> shuffle = {"e8", "e2", "e9", "e1", "e8", "e2", "e9", "e1"};
  const int C = QuoridorNN::numSpatialFeatures(QuoridorNN::TRAINING_IO_VERSION);
  const int G = QuoridorNN::numGlobalFeatures(QuoridorNN::TRAINING_IO_VERSION);
  testAssert(QuoridorNN::TRAINING_IO_VERSION == 3 && C == 21 && G == 19);
  const int packedArea = (AREA + 7) / 8;
  int drawingRows = 0;
  for(size_t i = 0; i < shuffle.size(); i++) {
    vector<float> scratch(C * AREA), global(G);
    vector<uint8_t> packed(C * packedArea), dist(QuoridorNN::NUM_DIST_CHANNELS * AREA);
    MiscNNInputParams params;
    TrainingWriteBuffers::fillQuoridorInputRow(g.board, g.hist, g.pla, params, scratch.data(), packed.data(), dist.data(), global.data());
    Row row = fill(g.board, g.hist, g.pla);
    testAssert(global == row.global);
    for(int ch = 0; ch < C; ch++) {
      if(ch >= QuoridorNN::FIRST_DIST_CHANNEL && ch < QuoridorNN::FIRST_DIST_CHANNEL + QuoridorNN::NUM_DIST_CHANNELS)
        continue;
      for(int p = 0; p < AREA; p++) {
        int bit = (packed[ch * packedArea + p / 8] >> (7 - p % 8)) & 1;
        testAssert((float)bit == row.spatial[ch * AREA + p]);
      }
    }
    int drawing = 0;
    for(int p = 0; p < AREA; p++)
      drawing += row.spatial[QuoridorNN::SPATIAL_DRAWING_MOVE_V3 * AREA + p] > 0 ? 1 : 0;
    //Only the last row (before White's e1, the third occurrence of the start) has a drawing move.
    testAssert(drawing == (i == shuffle.size() - 1 ? 1 : 0));
    drawingRows += drawing;
    g.play({shuffle[i]});
  }
  testAssert(drawingRows == 1 && g.hist.isRepetitionDraw());
}

}  // namespace

void Tests::runQuoridorIOv3Tests() {
  cout << "Running Quoridor I/O v3 tests" << endl;
  testBruteForce();
  testMirror();
  testCacheHash();
  testTrainingEncoding();
  cout << "Quoridor I/O v3 tests passed" << endl;
}
