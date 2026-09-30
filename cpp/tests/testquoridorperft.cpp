/*
 * testquoridorperft.cpp
 * Phase 0 game-rule safety net (see docs/KataQuoridor_Review_and_Roadmap.md §5):
 *  - perft move counts from fixed positions, pinned against the independent Python reference
 *    python/tests/test_quoridor_perft_reference.py
 *  - lazy-BFS fuzz: in random games, Board's lazy wall legality == a naive full-BFS check, for all 128 walls
 *  - no-legal-move invariant: every non-terminal position of the fuzz games has at least one legal move
 *  - undo round-trip: playMoveRecorded + undo restores the board exactly, including the hash
 *
 * The naive reference below works on 9x9 cells and an explicit list of wall anchors that the test tracks
 * itself, so it shares no code with Board's 17x17 representation or its cached paths.
 */

#include "../tests/tests.h"

#include "../game/graphhash.h"

#include <algorithm>
#include <set>

using namespace std;
using namespace TestCommon;

namespace {

static constexpr int N = 9;
static constexpr int NUM_ANCHORS = N - 1;
static constexpr int SIZE = Board::DEFAULT_LEN;

//------------------------------------------------------------------------------------------------
// Independent reference model

struct Cell {
  int c;
  int r;
  bool operator==(const Cell& o) const { return c == o.c && r == o.r; }
  bool operator<(const Cell& o) const { return c < o.c || (c == o.c && r < o.r); }
};

static Cell cellOfLoc(Loc loc) {
  int x = Location::getX(loc, SIZE);
  int y = Location::getY(loc, SIZE);
  testAssert(x >= 0 && x < SIZE && y >= 0 && y < SIZE && x % 2 == 0 && y % 2 == 0);
  return Cell{x / 2, y / 2};
}

struct RefWalls {
  // Indexed [c][r] over anchors 0..7.
  bool h[NUM_ANCHORS][NUM_ANCHORS] = {};
  bool v[NUM_ANCHORS][NUM_ANCHORS] = {};
  int left[3] = {0, Board::MAX_FENCE_NUM, Board::MAX_FENCE_NUM}; // indexed by Player

  bool hasH(int c, int r) const { return c >= 0 && c < NUM_ANCHORS && r >= 0 && r < NUM_ANCHORS && h[c][r]; }
  bool hasV(int c, int r) const { return c >= 0 && c < NUM_ANCHORS && r >= 0 && r < NUM_ANCHORS && v[c][r]; }

  bool blocked(Cell a, Cell b) const {
    if(a.c == b.c) {
      int r = min(a.r, b.r);
      return hasH(a.c, r) || hasH(a.c - 1, r);
    }
    int c = min(a.c, b.c);
    return hasV(c, a.r) || hasV(c, a.r - 1);
  }
  bool canStep(Cell a, Cell b) const {
    return b.c >= 0 && b.c < N && b.r >= 0 && b.r < N && !blocked(a, b);
  }
};

static const int DIRS[4][2] = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};

static bool refReachesGoal(const RefWalls& w, Cell start, int goalRow) {
  bool seen[N][N] = {};
  Cell q[N * N];
  int head = 0;
  int tail = 0;
  q[tail++] = start;
  seen[start.c][start.r] = true;
  while(head < tail) {
    Cell cur = q[head++];
    if(cur.r == goalRow)
      return true;
    for(int d = 0; d < 4; d++) {
      Cell nxt{cur.c + DIRS[d][0], cur.r + DIRS[d][1]};
      if(w.canStep(cur, nxt) && !seen[nxt.c][nxt.r]) {
        seen[nxt.c][nxt.r] = true;
        q[tail++] = nxt;
      }
    }
  }
  return false;
}

static int goalRowOf(Player pla) {
  return pla == P_BLACK ? 0 : N - 1;
}

// Returns 0 if legal, 1 if illegal for a local reason (no walls left, overlap, crossing), 2 if illegal
// only because it would cut some pawn off from its goal.
static int refWallStatus(const Board& board, const RefWalls& w, int c, int r, bool isVertical, Player pla) {
  if(w.left[pla] <= 0)
    return 1;
  if(w.h[c][r] || w.v[c][r])
    return 1;
  RefWalls after = w;
  if(isVertical) {
    if(w.hasV(c, r - 1) || w.hasV(c, r + 1))
      return 1;
    after.v[c][r] = true;
  }
  else {
    if(w.hasH(c - 1, r) || w.hasH(c + 1, r))
      return 1;
    after.h[c][r] = true;
  }
  if(!refReachesGoal(after, cellOfLoc(board.blackPawnLoc), goalRowOf(P_BLACK)))
    return 2;
  if(!refReachesGoal(after, cellOfLoc(board.whitePawnLoc), goalRowOf(P_WHITE)))
    return 2;
  return 0;
}

static set<Cell> refPawnMoves(const Board& board, const RefWalls& w, Player pla, bool& sawDiagonalJump) {
  Cell me = cellOfLoc(pla == P_BLACK ? board.blackPawnLoc : board.whitePawnLoc);
  Cell opp = cellOfLoc(pla == P_BLACK ? board.whitePawnLoc : board.blackPawnLoc);
  set<Cell> out;
  for(int d = 0; d < 4; d++) {
    int dc = DIRS[d][0];
    int dr = DIRS[d][1];
    Cell adj{me.c + dc, me.r + dr};
    if(!w.canStep(me, adj))
      continue;
    if(!(adj == opp)) {
      out.insert(adj);
      continue;
    }
    Cell behind{adj.c + dc, adj.r + dr};
    if(w.canStep(adj, behind)) {
      out.insert(behind);
      continue;
    }
    Cell sides[2] = {{adj.c + dr, adj.r + dc}, {adj.c - dr, adj.r - dc}};
    for(const Cell& side : sides) {
      if(w.canStep(adj, side)) {
        out.insert(side);
        sawDiagonalJump = true;
      }
    }
  }
  return out;
}

static void refApplyMove(RefWalls& w, Loc loc, Player pla) {
  int x = Location::getX(loc, SIZE);
  int y = Location::getY(loc, SIZE);
  if(x % 2 == 0 && y % 2 == 0)
    return;
  testAssert(x % 2 == 1);
  int c = x / 2;
  int r = y / 2;
  if(y % 2 == 1)
    w.h[c][r] = true;
  else
    w.v[c][r] = true;
  w.left[pla]--;
}

//------------------------------------------------------------------------------------------------
// Helpers on the real Board

static bool isTerminal(const Board& board) {
  return Location::getY(board.blackPawnLoc, SIZE) == 0 || Location::getY(board.whitePawnLoc, SIZE) == SIZE - 1;
}

static vector<Loc> legalMoves(const Board& board, Player pla) {
  vector<Loc> moves;
  for(int y = 0; y < SIZE; y++) {
    for(int x = 0; x < SIZE; x++) {
      Loc loc = Location::getLoc(x, y, SIZE);
      if(board.isLegal(loc, pla))
        moves.push_back(loc);
    }
  }
  return moves;
}

static Hash128 hashFromScratch(const Board& b) {
  Hash128 h = Board::ZOBRIST_SIZE_X_HASH[b.x_size] ^ Board::ZOBRIST_SIZE_Y_HASH[b.y_size];
  h ^= Board::ZOBRIST_BOARD_HASH[b.whitePawnLoc][C_WHITE];
  h ^= Board::ZOBRIST_BOARD_HASH[b.blackPawnLoc][C_BLACK];
  h ^= Board::ZOBRIST_FENCENUM_HASH[b.blackFences][0];
  h ^= Board::ZOBRIST_FENCENUM_HASH[b.whiteFences][1];
  for(int y = 0; y < b.y_size; y++) {
    for(int x = 0; x < b.x_size; x++) {
      Loc loc = Location::getLoc(x, y, b.x_size);
      if(b.colors[loc] == C_FENCE)
        h ^= Board::ZOBRIST_BOARD_HASH[loc][C_FENCE];
    }
  }
  return h;
}

// The lazy wall check is sound only if each cached path is a real, currently unblocked path
// from the pawn to its goal row.
static void checkCachedPath(const Board& b, const CompactPath& path, Loc pawn, int goalY) {
  testAssert(path.size() >= 1);
  testAssert(path[0] == pawn);
  testAssert(Location::getY(path[path.size() - 1], SIZE) == goalY);
  for(size_t i = 0; i + 1 < path.size(); i++)
    testAssert(b.canPawnStep(path[i], path[i + 1]));
}

static void checkBoardInvariants(const Board& b) {
  b.checkConsistency();
  testAssert(b.colors[b.blackPawnLoc] == C_BLACK);
  testAssert(b.colors[b.whitePawnLoc] == C_WHITE);
  testAssert(b.pos_hash == hashFromScratch(b));
  checkCachedPath(b, b.cachedPathP1, b.blackPawnLoc, 0);
  checkCachedPath(b, b.cachedPathP2, b.whitePawnLoc, SIZE - 1);
}

static void checkSameBoard(const Board& a, const Board& b) {
  testAssert(a.isEqualForTesting(b));
  testAssert(a.pos_hash == b.pos_hash);
  testAssert(a.movenum == b.movenum);
  for(int i = 0; i < Board::MAX_ARR_SIZE; i++)
    testAssert(a.colors[i] == b.colors[i]);
  for(int c = 0; c < NUM_ANCHORS; c++) {
    for(int r = 0; r < NUM_ANCHORS; r++) {
      testAssert(a.hWalls[c][r] == b.hWalls[c][r]);
      testAssert(a.vWalls[c][r] == b.vWalls[c][r]);
    }
  }
}

static Board boardFromMoves(const string& moves, Player& pla) {
  Board board;
  pla = P_BLACK;
  for(const string& s : Global::split(Global::trim(moves), ' ')) {
    if(s.empty())
      continue;
    testAssert(!isTerminal(board));
    Loc loc = Location::ofString(s, board);
    testAssert(board.playMove(loc, pla));
    pla = getOpp(pla);
  }
  return board;
}

//------------------------------------------------------------------------------------------------
// Perft

static uint64_t perftCopy(const Board& board, Player pla, int depth) {
  if(isTerminal(board) || depth == 0)
    return 1;
  vector<Loc> moves = legalMoves(board, pla);
  testAssert(!moves.empty());
  if(depth == 1)
    return moves.size();
  uint64_t total = 0;
  for(Loc loc : moves) {
    Board copy = board;
    copy.playMoveAssumeLegal(loc, pla);
    total += perftCopy(copy, getOpp(pla), depth - 1);
  }
  return total;
}

static uint64_t perftUndo(Board& board, Player pla, int depth) {
  if(isTerminal(board) || depth == 0)
    return 1;
  vector<Loc> moves = legalMoves(board, pla);
  testAssert(!moves.empty());
  if(depth == 1)
    return moves.size();
  uint64_t total = 0;
  for(Loc loc : moves) {
    Board::MoveRecord rec = board.playMoveRecorded(loc, pla);
    total += perftUndo(board, getOpp(pla), depth - 1);
    board.undo(rec);
  }
  return total;
}

static uint64_t perftHist(const Board& board, const BoardHistory& hist, Player pla, int depth) {
  if(hist.isGameFinished || depth == 0)
    return 1;
  uint64_t total = 0;
  for(int y = 0; y < SIZE; y++) {
    for(int x = 0; x < SIZE; x++) {
      Loc loc = Location::getLoc(x, y, SIZE);
      if(!hist.isLegal(board, loc, pla))
        continue;
      if(depth == 1) {
        total++;
        continue;
      }
      Board b2 = board;
      BoardHistory h2 = hist;
      h2.makeBoardMoveAssumeLegal(b2, loc, pla, NULL);
      total += perftHist(b2, h2, getOpp(pla), depth - 1);
    }
  }
  return total;
}

struct PerftCase {
  const char* name;
  const char* moves;
  vector<uint64_t> counts; // perft(1), perft(2), ...
};

// Keep in sync with PERFT_POSITIONS in python/tests/test_quoridor_perft_reference.py.
static const vector<PerftCase> PERFT_CASES = {
  {"start", "", {131, 16677, 2062264}},
  {"face-to-face, white can jump straight", "e8 e2 e7 e3 e6 e4 e5", {132, 16938, 2111842}},
  {"wall behind white, black gets diagonal jumps", "e8 e2 e7 e3 e6 e4 e5 e3h", {129, 15922, 1936376}},
  {"white at e8 under black at e9 on the edge, diagonal jumps win",
   "a2h e2 c2h e3 a4h e4 c4h e5 a6h e6 c6h e7 a8h e8 c8h", {109, 11083, 1129546}},
  {"black out of walls, white has all ten",
   "a2h e2 c2h e1 a4h e2 c4h e1 a6h e2 c6h e1 a8h e2 c8h e1 g2h e2 g4h e1", {3, 296, 878}},
};

static void testPerft() {
  cout << "  Quoridor perft..." << endl;
  for(const PerftCase& pc : PERFT_CASES) {
    Player pla;
    Board board = boardFromMoves(pc.moves, pla);
    const Board initial = board;
    for(size_t i = 0; i < pc.counts.size(); i++) {
      int depth = (int)i + 1;
      uint64_t viaCopy = perftCopy(board, pla, depth);
      uint64_t viaUndo = perftUndo(board, pla, depth);
      checkSameBoard(board, initial);
      checkBoardInvariants(board);
      if(viaCopy != pc.counts[i] || viaUndo != pc.counts[i]) {
        cout << "    " << pc.name << " depth " << depth << ": expected " << pc.counts[i]
             << " copy " << viaCopy << " undo " << viaUndo << endl;
      }
      testAssert(viaCopy == pc.counts[i]);
      testAssert(viaUndo == pc.counts[i]);
      if(depth <= 2) {
        BoardHistory hist(board, pla, Rules::getQuoridorRules(), 0, BoardHistoryModes());
        testAssert(perftHist(board, hist, pla, depth) == pc.counts[i]);
      }
    }
    cout << "    " << pc.name << ": ok" << endl;
  }
}

//------------------------------------------------------------------------------------------------
// Fuzz: lazy BFS vs naive BFS, no-legal-move invariant, undo round-trip

struct FuzzStats {
  uint64_t positions = 0;
  uint64_t terminalGames = 0;
  uint64_t wallChecks = 0;
  uint64_t wallsRejectedForBlocking = 0;
  uint64_t positionsWithDiagonalJump = 0;
  uint64_t undoChecks = 0;
};

static void checkPosition(const Board& board, const RefWalls& w, Player pla, Rand& rand, FuzzStats& stats) {
  stats.positions++;
  checkBoardInvariants(board);
  testAssert(board.blackFences == w.left[P_BLACK]);
  testAssert(board.whiteFences == w.left[P_WHITE]);

  // Board's explicit wall arrays (the single source of truth for wall placement, used by
  // QuoridorNN::fillRow) must exactly equal the walls replayed independently from move history
  // (RefWalls), and `colors` must equal what those walls imply: this is the regression test for
  // the wall-anchor ambiguity bug where a horizontal wall at (c, r) and a vertical wall at
  // (c, r-1) share an arm cell, so "center + one arm occupied" alone cannot tell them apart.
  for(int c = 0; c < NUM_ANCHORS; c++) {
    for(int r = 0; r < NUM_ANCHORS; r++) {
      testAssert(board.hWalls[c][r] == w.h[c][r]);
      testAssert(board.vWalls[c][r] == w.v[c][r]);

      Loc center = Location::hWallLoc(c, r);
      bool colorsImplyCenter = board.colors[center] == C_FENCE;
      testAssert(colorsImplyCenter == (board.hWalls[c][r] || board.vWalls[c][r]));
      if(board.hWalls[c][r]) {
        testAssert(board.colors[center + board.adj_offsets[1]] == C_FENCE);
        testAssert(board.colors[center + board.adj_offsets[2]] == C_FENCE);
      }
      if(board.vWalls[c][r]) {
        testAssert(board.colors[center + board.adj_offsets[0]] == C_FENCE);
        testAssert(board.colors[center + board.adj_offsets[3]] == C_FENCE);
      }
    }
  }

  // Walls: lazy (isLegalWallPlacement, isLegal) vs naive full BFS, for both players' fence counts.
  for(int c = 0; c < NUM_ANCHORS; c++) {
    for(int r = 0; r < NUM_ANCHORS; r++) {
      for(int vert = 0; vert < 2; vert++) {
        bool isVertical = vert == 1;
        Loc loc = isVertical ? Location::vWallLoc(c, r) : Location::hWallLoc(c, r);
        for(Player p : {P_BLACK, P_WHITE}) {
          int status = refWallStatus(board, w, c, r, isVertical, p);
          bool lazy = board.isLegalWallPlacement(c, r, isVertical, p);
          if(lazy != (status == 0)) {
            cout << board << endl;
            cout << "Wall " << Location::toString(loc, board) << " for " << PlayerIO::playerToString(p)
                 << ": lazy " << lazy << " naive status " << status << endl;
          }
          testAssert(lazy == (status == 0));
          testAssert(board.isLegal(loc, p) == (status == 0));
          stats.wallChecks++;
          if(status == 2)
            stats.wallsRejectedForBlocking++;
        }
      }
    }
  }

  // Pawn moves: isLegal over all cells vs getLegalPawnDestinations vs naive.
  bool sawDiagonalJump = false;
  set<Cell> naivePawn = refPawnMoves(board, w, pla, sawDiagonalJump);
  if(sawDiagonalJump)
    stats.positionsWithDiagonalJump++;
  set<Cell> viaIsLegal;
  for(int c = 0; c < N; c++)
    for(int r = 0; r < N; r++)
      if(board.isLegal(Location::pawnLoc(c, r), pla))
        viaIsLegal.insert(Cell{c, r});
  set<Cell> viaDest;
  for(Loc loc : board.getLegalPawnDestinations(pla))
    viaDest.insert(cellOfLoc(loc));
  testAssert(viaIsLegal == naivePawn);
  testAssert(viaDest == naivePawn);

  // No-legal-move invariant, and nothing outside pawn cells / wall anchors is ever legal.
  vector<Loc> moves = legalMoves(board, pla);
  testAssert(!board.isLegal(Board::PASS_LOC, pla));
  testAssert(!board.isLegal(Board::NULL_LOC, pla));
  size_t numWallMoves = 0;
  for(Loc loc : moves) {
    if(Location::isHWallLoc(loc) || Location::isVWallLoc(loc))
      numWallMoves++;
    else
      testAssert(Location::isPawnLoc(loc));
  }
  testAssert(moves.size() == naivePawn.size() + numWallMoves);
  if(!isTerminal(board))
    testAssert(!moves.empty());
  testAssert(!naivePawn.empty()); // Stronger: under no-full-block, a pawn is never boxed in.

  // Undo round-trip for every legal move.
  const Board before = board;
  size_t sampled = moves.empty() ? 0 : (size_t)rand.nextUInt((uint32_t)moves.size());
  for(size_t i = 0; i < moves.size(); i++) {
    Board b = board;
    Board::MoveRecord rec = b.playMoveRecorded(moves[i], pla);
    testAssert(b.pos_hash == hashFromScratch(b));
    Board fresh = board;
    fresh.playMoveAssumeLegal(moves[i], pla);
    checkSameBoard(b, fresh);
    b.undo(rec);
    checkSameBoard(b, before);
    checkBoardInvariants(b);
    if(i == sampled)
      testAssert(legalMoves(b, pla) == moves);
    stats.undoChecks++;
  }
}

static void testFuzz() {
  cout << "  Quoridor lazy-BFS / no-legal-move / undo fuzz..." << endl;
  Rand rand("testquoridorperft fuzz");
  FuzzStats stats;
  const int numGames = 150;
  const int maxPlies = 400;

  for(int game = 0; game < numGames; game++) {
    // Mix of move pickers, so we see both wall-heavy middlegames and games that reach the goal:
    // 0 = uniform over all legal moves; 1 = pawn move half the time; 2 = mostly shortest-path pawn moves.
    int style = game % 3;
    Board board;
    RefWalls walls;
    Player pla = P_BLACK;
    // The fuzz plays past the standard 300-ply draw, to exercise long wall-heavy games.
    Rules rules = Rules::getQuoridorRules();
    rules.maxPlies = maxPlies + 1;
    BoardHistory hist(board, pla, rules, 0, BoardHistoryModes());
    vector<Board::MoveRecord> records;
    vector<Board> snapshots;

    for(int ply = 0; ply < maxPlies && !isTerminal(board); ply++) {
      checkPosition(board, walls, pla, rand, stats);
      testAssert(!hist.isGameFinished);

      vector<Loc> moves = legalMoves(board, pla);
      vector<Loc> pawnMoves;
      for(Loc loc : moves)
        if(Location::isPawnLoc(loc))
          pawnMoves.push_back(loc);

      Loc chosen;
      double u = rand.nextDouble();
      if(style == 2 && u < 0.8) {
        vector<Loc> path = board.findShortestPath(pla);
        testAssert(path.size() >= 2);
        chosen = board.isLegal(path[1], pla) ? path[1] : pawnMoves[rand.nextUInt((uint32_t)pawnMoves.size())];
      }
      else if(style >= 1 && u < 0.5)
        chosen = pawnMoves[rand.nextUInt((uint32_t)pawnMoves.size())];
      else
        chosen = moves[rand.nextUInt((uint32_t)moves.size())];

      testAssert(hist.isLegal(board, chosen, pla));
      snapshots.push_back(board);
      Board histBoard = board;
      hist.makeBoardMoveAssumeLegal(histBoard, chosen, pla, NULL);
      records.push_back(board.playMoveRecorded(chosen, pla));
      checkSameBoard(board, histBoard);
      refApplyMove(walls, chosen, pla);
      pla = getOpp(pla);
    }

    if(isTerminal(board)) {
      stats.terminalGames++;
      testAssert(hist.isGameFinished);
      bool blackWon = Location::getY(board.blackPawnLoc, SIZE) == 0;
      testAssert(hist.winner == (blackWon ? P_BLACK : P_WHITE));
      checkBoardInvariants(board);
    }

    // Unwind the whole game with undo; every intermediate board must match its snapshot.
    while(!records.empty()) {
      board.undo(records.back());
      records.pop_back();
      checkSameBoard(board, snapshots.back());
      checkBoardInvariants(board);
      snapshots.pop_back();
    }
    checkSameBoard(board, Board());
  }

  cout << "    positions " << stats.positions << ", finished games " << stats.terminalGames << "/" << numGames
       << ", wall checks " << stats.wallChecks << " (rejected for blocking " << stats.wallsRejectedForBlocking
       << "), positions with diagonal jumps " << stats.positionsWithDiagonalJump
       << ", undo checks " << stats.undoChecks << endl;
  // Make sure the fuzz actually exercises the interesting cases.
  testAssert(stats.terminalGames > 0);
  testAssert(stats.wallsRejectedForBlocking > 0);
  testAssert(stats.positionsWithDiagonalJump > 0);
}

//------------------------------------------------------------------------------------------------
// Board printing (showboard)

static void testBoardSizes() {
  cout << "  Board sizes" << endl;
  // Board size in pawn cells, which board-area based scales use, vs the search grid.
  Board board;
  testAssert(board.x_size == SIZE && board.y_size == SIZE);
  testAssert(board.pawnXSize() == N && board.pawnYSize() == N);
  testAssert(board.pawnArea() == N * N);
  testAssert(board.sqrtBoardArea() == (double)N);
  testAssert(Board::DEFAULT_PAWN_LEN == Board::pawnLenOfGridLen(Board::DEFAULT_LEN));
}

static Hash128 graphHashAfterMoves(const string& moves) {
  Board board;
  Player pla = P_BLACK;
  BoardHistory hist(board, pla, Rules::getQuoridorRules(), 0, BoardHistoryModes(false, false));
  for(const string& s : Global::split(Global::trim(moves), ' ')) {
    Loc loc = Location::ofString(s, board);
    testAssert(hist.isLegal(board, loc, pla));
    hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    pla = getOpp(pla);
  }
  return GraphHash::getGraphHashFromScratch(hist, pla, 11, 0.5);
}

static void testGraphHashTranspositions() {
  cout << "  Graph hash transpositions" << endl;
  // Same position reached in different orders, last move a wall: merged (the position cannot repeat).
  testAssert(graphHashAfterMoves("d3h e7h f3h") == graphHashAfterMoves("f3h e7h d3h"));
  // Same position, last move a pawn move: kept apart, since pawn moves can cycle.
  testAssert(graphHashAfterMoves("e8 e2 d8") != graphHashAfterMoves("d9 e2 d8"));
  // A wall after those makes the positions identical and irreversible again, so they merge.
  testAssert(graphHashAfterMoves("e8 e2 d8 a2h") == graphHashAfterMoves("d9 e2 d8 a2h"));
  // Different positions never merge.
  testAssert(graphHashAfterMoves("d3h e7h f3h") != graphHashAfterMoves("d3h e7h g3h"));
}

static void testPrintBoard() {
  cout << "  Board printing" << endl;
  Player pla;
  Board board = boardFromMoves("d3h e5v a8h e2", pla);
  BoardHistory hist(board, pla, Rules::getQuoridorRules(), 0, BoardHistoryModes(false, false));
  vector<Move> moves = {Move(Location::ofString("d3h", board), P_BLACK), Move(Location::ofString("e5v", board), P_WHITE)};

  {
    // Marking a wall that is not placed yet (as a search hint would).
    ostringstream out;
    Board::printBoard(out, board, Location::ofString("g6v", board), &moves);
    string expected = R"%%(
MoveNum: 2 HASH: 1A1440A04BD749E4AC8283D4AB8EF701
     a   b   c   d   e   f   g   h   i
   +---+---+---+---+---+---+---+---+---+
 9 | .   .   .   .   B   .   .   .   . |
   +---+---+   +   +   +   +   +   +   +
 8 | .   .   .   .   .   .   .   .   . |
   +   +   +   +   +   +   +   +   +   +
 7 | .   .   .   .   .   .   . # .   . |
   +   +   +   +   +   +   +   +   +   +
 6 | .   .   .   .   . | .   . # .   . |
   +   +   +   +   +   +   +   +   +   +
 5 | .   .   .   .   . | .   .   .   . |
   +   +   +   +   +   +   +   +   +   +
 4 | .   .   .   .   .   .   .   .   . |
   +   +   +   +---+---+   +   +   +   +
 3 | .   .   .   .   .   .   .   .   . |
   +   +   +   +   +   +   +   +   +   +
 2 | .   .   .   .   W   .   .   .   . |
   +   +   +   +   +   +   +   +   +   +
 1 | .   .   .   .   .   .   .   .   . |
   +---+---+---+---+---+---+---+---+---+
Black (B): e9 walls: 8 dist: 9 (goal row 1)
White (W): e2 walls: 9 dist: 8 (goal row 9)
)%%";
    expect("printBoard with marked vertical wall", out, expected);
  }

  {
    // Marking a pawn cell and a horizontal wall.
    ostringstream out;
    Board::printBoard(out, board, Location::ofString("e8", board), nullptr);
    Board::printBoard(out, board, Location::ofString("h1h", board), nullptr);
    string expected = R"%%(
HASH: 1A1440A04BD749E4AC8283D4AB8EF701
     a   b   c   d   e   f   g   h   i
   +---+---+---+---+---+---+---+---+---+
 9 | .   .   .   .   B   .   .   .   . |
   +---+---+   +   +   +   +   +   +   +
 8 | .   .   .   .  [.]  .   .   .   . |
   +   +   +   +   +   +   +   +   +   +
 7 | .   .   .   .   .   .   .   .   . |
   +   +   +   +   +   +   +   +   +   +
 6 | .   .   .   .   . | .   .   .   . |
   +   +   +   +   +   +   +   +   +   +
 5 | .   .   .   .   . | .   .   .   . |
   +   +   +   +   +   +   +   +   +   +
 4 | .   .   .   .   .   .   .   .   . |
   +   +   +   +---+---+   +   +   +   +
 3 | .   .   .   .   .   .   .   .   . |
   +   +   +   +   +   +   +   +   +   +
 2 | .   .   .   .   W   .   .   .   . |
   +   +   +   +   +   +   +   +   +   +
 1 | .   .   .   .   .   .   .   .   . |
   +---+---+---+---+---+---+---+---+---+
Black (B): e9 walls: 8 dist: 9 (goal row 1)
White (W): e2 walls: 9 dist: 8 (goal row 9)
HASH: 1A1440A04BD749E4AC8283D4AB8EF701
     a   b   c   d   e   f   g   h   i
   +---+---+---+---+---+---+---+---+---+
 9 | .   .   .   .   B   .   .   .   . |
   +---+---+   +   +   +   +   +   +   +
 8 | .   .   .   .   .   .   .   .   . |
   +   +   +   +   +   +   +   +   +   +
 7 | .   .   .   .   .   .   .   .   . |
   +   +   +   +   +   +   +   +   +   +
 6 | .   .   .   .   . | .   .   .   . |
   +   +   +   +   +   +   +   +   +   +
 5 | .   .   .   .   . | .   .   .   . |
   +   +   +   +   +   +   +   +   +   +
 4 | .   .   .   .   .   .   .   .   . |
   +   +   +   +---+---+   +   +   +   +
 3 | .   .   .   .   .   .   .   .   . |
   +   +   +   +   +   +   +   +   +   +
 2 | .   .   .   .   W   .   .   .   . |
   +   +   +   +   +   +   +   +===+===+
 1 | .   .   .   .   .   .   .   .   . |
   +---+---+---+---+---+---+---+---+---+
Black (B): e9 walls: 8 dist: 9 (goal row 1)
White (W): e2 walls: 9 dist: 8 (goal row 9)
)%%";
    expect("printBoard with marked pawn cell and horizontal wall", out, expected);
  }

  {
    // Hitting the move cap ends the game via endAndScoreGameNow, which is the only way to draw.
    hist.endAndScoreGameNow(board);
    ostringstream out;
    hist.printBasicInfo(out, board);
    vector<string> lines = Global::split(Global::trim(out.str()), '\n');
    testAssert(lines.back() == "Game finished: Draw (move cutoff)");
    testAssert(out.str().find("Rules") == string::npos);
    testAssert(out.str().find("komi") == string::npos);
  }
}

} // namespace

void Tests::runQuoridorRuleTests() {
  cout << "=== Running Quoridor perft / fuzz / undo tests ===" << endl;
  testPerft();
  testFuzz();
  testBoardSizes();
  testGraphHashTranspositions();
  testPrintBoard();
  cout << "=== Quoridor perft / fuzz / undo tests passed ===" << endl;
}
