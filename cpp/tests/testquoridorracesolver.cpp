/*
 * testquoridorracesolver.cpp
 * Tests for the pawn race solver (docs/QuoridorRaceSolver.md):
 *  - the solver's pawn move generation equals Board::getLegalPawnDestinations on random walls and pawns
 *  - hand-built races: straight jump, diagonal jump at a wall, a corridor block, a race won because of a jump
 *  - the solved value equals a brute-force minimax over full BoardHistory play (small horizon)
 *  - optimal play on the real BoardHistory ends with the solved result in the solved number of plies
 *  - evaluate(): which positions count as solved (exact, one-sided, plies left, repetition history)
 *  - the cache
 */

#include "../tests/tests.h"

#include "../game/racesolver.h"

#include <chrono>

using namespace std;

namespace {

static int sq(int c, int r) { return RaceSolution::sqOf(c, r); }

static void setPawns(Board& board, int blackSq, int whiteSq) {
  board.setStone(board.blackPawnLoc, C_EMPTY);
  board.setStone(board.whitePawnLoc, C_EMPTY);
  board.blackPawnLoc = RaceSolution::locOfSq(blackSq);
  board.whitePawnLoc = RaceSolution::locOfSq(whiteSq);
  board.setStone(board.blackPawnLoc, C_BLACK);
  board.setStone(board.whitePawnLoc, C_WHITE);
}

static void placeWall(Board& board, int c, int r, bool vertical, Player pla) {
  Loc loc = vertical ? Location::vWallLoc(c, r) : Location::hWallLoc(c, r);
  testAssert(board.isLegalWallPlacement(c, r, vertical, pla));
  testAssert(board.playMove(loc, pla));
}

static Board randomWallBoard(Rand& rand, int numWallAttempts) {
  Board board;
  board.setFencesLeft(10, 10);
  for(int i = 0; i < numWallAttempts; i++) {
    Player pla = rand.nextBool(0.5) ? P_BLACK : P_WHITE;
    int c = rand.nextInt(0, 7), r = rand.nextInt(0, 7);
    bool v = rand.nextBool(0.5);
    if(board.isLegalWallPlacement(c, r, v, pla))
      board.playMove(v ? Location::vWallLoc(c, r) : Location::hWallLoc(c, r), pla);
  }
  board.setFencesLeft(0, 0);
  return board;
}

static set<Loc> solverMoves(const RaceSolution& sol, const Board& board, Player pla) {
  int b = RaceSolution::sqOfLoc(board.blackPawnLoc, board.x_size);
  int w = RaceSolution::sqOfLoc(board.whitePawnLoc, board.x_size);
  int8_t moves[RaceSolution::MAX_MOVES];
  int n = sol.genMoves(pla == P_BLACK ? b : w, pla == P_BLACK ? w : b, moves);
  set<Loc> ret;
  for(int i = 0; i < n; i++)
    ret.insert(RaceSolution::locOfSq(moves[i]));
  return ret;
}

static set<Loc> boardMoves(const Board& board, Player pla) {
  vector<Loc> v = board.getLegalPawnDestinations(pla);
  return set<Loc>(v.begin(), v.end());
}

static BoardHistory makeHist(const Board& board, Player pla, int maxPlies = 300, int repetition = 0, float komi = -0.5f) {
  Rules rules = Rules::getQuoridorRules();
  rules.maxPlies = maxPlies;
  rules.repetitionDrawCount = repetition;
  rules.komi = komi;
  return BoardHistory(board, pla, rules, 0, BoardHistoryModes());
}

static int valueAt(const RaceSolution& sol, const Board& board, Player pla) {
  int b = RaceSolution::sqOfLoc(board.blackPawnLoc, board.x_size);
  int w = RaceSolution::sqOfLoc(board.whitePawnLoc, board.x_size);
  return sol.val2[RaceSolution::stateIndex(pla, b, w)];
}
static int rankAt(const RaceSolution& sol, const Board& board, Player pla) {
  int b = RaceSolution::sqOfLoc(board.blackPawnLoc, board.x_size);
  int w = RaceSolution::sqOfLoc(board.whitePawnLoc, board.x_size);
  return sol.plies[RaceSolution::stateIndex(pla, b, w)];
}

static int forcedAt(const RaceSolution& sol, const Board& board, Player pla) {
  int b = RaceSolution::sqOfLoc(board.blackPawnLoc, board.x_size);
  int w = RaceSolution::sqOfLoc(board.whitePawnLoc, board.x_size);
  return sol.forced[RaceSolution::stateIndex(pla, b, w)];
}

//------------------------------------------------------------------------------------------------

static void testMoveGeneration() {
  cout << "  Move generation equals Board::getLegalPawnDestinations" << endl;
  Rand rand("testquoridorracesolver movegen");
  int numChecked = 0, numJumps = 0;
  for(int iter = 0; iter < 60; iter++) {
    Board board = randomWallBoard(rand, rand.nextInt(0, 40));
    shared_ptr<RaceSolution> sol = RaceSolver::solve(board, -0.5f);
    for(int k = 0; k < 150; k++) {
      int b = rand.nextInt(0, 80), w = rand.nextInt(0, 80);
      // Half of the positions put the pawns next to each other, where the jumps are.
      if(k % 2 == 0) {
        int d = rand.nextInt(0, 3);
        int c = b % 9 + (d == 1 ? -1 : d == 2 ? 1 : 0), r = b / 9 + (d == 0 ? -1 : d == 3 ? 1 : 0);
        if(c < 0 || c > 8 || r < 0 || r > 8)
          continue;
        w = sq(c, r);
      }
      if(b == w)
        continue;
      setPawns(board, b, w);
      for(Player pla : {P_BLACK, P_WHITE}) {
        set<Loc> a = solverMoves(*sol, board, pla);
        set<Loc> e = boardMoves(board, pla);
        if(a != e) {
          cout << Board::toStringSimple(board) << " pla " << (int)pla << endl;
          testAssert(false);
        }
        for(Loc loc : e)
          testAssert(board.isLegalPawnMove(loc, pla));
        int me = pla == P_BLACK ? b : w, opp = pla == P_BLACK ? w : b;
        for(Loc loc : e) {
          int t = RaceSolution::sqOfLoc(loc, board.x_size);
          if(std::abs(t % 9 - me % 9) + std::abs(t / 9 - me / 9) != 1)
            numJumps++;
        }
        (void)opp;
        numChecked++;
      }
    }
  }
  cout << "    " << numChecked << " position/player pairs, " << numJumps << " jump moves" << endl;
  testAssert(numJumps > 100);
}

static void testHandBuiltRaces() {
  cout << "  Hand-built races" << endl;
  // A straight jump: Black e5 (4,4), White e4 (4,3) in front of it, empty board. Black jumps to (4,2).
  {
    Board board;
    board.setFencesLeft(0, 0);
    setPawns(board, sq(4, 4), sq(4, 3));
    shared_ptr<RaceSolution> sol = RaceSolver::solve(board, -0.5f);
    set<Loc> moves = solverMoves(*sol, board, P_BLACK);
    testAssert(moves.count(RaceSolution::locOfSq(sq(4, 2))) == 1);
    testAssert(moves.count(RaceSolution::locOfSq(sq(4, 3))) == 0);
    testAssert(moves == boardMoves(board, P_BLACK));
    // Black: jump (ply 1), two more steps (plies 3, 5). White ran twice (plies 2, 4) to row 5, 3 from its goal:
    // B+3, t = -2, s = -2.5.
    testAssert(valueAt(*sol, board, P_BLACK) == -5);
    testAssert(rankAt(*sol, board, P_BLACK) == 5);
  }
  // A diagonal jump: a wall behind White blocks the straight jump, so Black has 5 moves.
  {
    Board board;
    board.setFencesLeft(10, 10);
    placeWall(board, 3, 2, false, P_WHITE); // horizontal, blocks columns 3 and 4 between rows 2 and 3
    board.setFencesLeft(0, 0);
    setPawns(board, sq(4, 4), sq(4, 3));
    shared_ptr<RaceSolution> sol = RaceSolver::solve(board, -0.5f);
    set<Loc> moves = solverMoves(*sol, board, P_BLACK);
    testAssert(moves.size() == 5);
    testAssert(moves.count(RaceSolution::locOfSq(sq(3, 3))) == 1 && moves.count(RaceSolution::locOfSq(sq(5, 3))) == 1);
    testAssert(moves.count(RaceSolution::locOfSq(sq(4, 2))) == 0);
    testAssert(moves == boardMoves(board, P_BLACK));
  }
  // A race won by the player who is behind on distance, because of a jump: Black e4 (4,5), White (4,4), Black to
  // move, empty board. Black needs 5, White 4, so a pure distance rule says White. But Black jumps to row 3.
  {
    Board board;
    board.setFencesLeft(0, 0);
    setPawns(board, sq(4, 5), sq(4, 4));
    shared_ptr<RaceSolution> sol = RaceSolver::solve(board, -0.5f);
    testAssert(sol->distBlack[sq(4, 5)] == 5 && sol->distWhite[sq(4, 4)] == 4);
    // Black arrives on ply 7 with White 1 away: B+1, s = -0.5.
    testAssert(valueAt(*sol, board, P_BLACK) == -1);
    testAssert(rankAt(*sol, board, P_BLACK) == 7);
    // With White to move in the same position White is ahead (4 vs 5, no jump for White: Black is behind it? White
    // at row 4 moving south meets Black at row 5 and jumps): the solver decides, the brute force below checks it.
    testAssert(valueAt(*sol, board, P_WHITE) > 0);
  }
  // A corridor block: column 0 is a corridor (vertical walls east of it along rows 1..6), White sits in it above
  // Black with a wall above White, so Black cannot jump and has to back off.
  {
    Board board;
    board.setFencesLeft(10, 10);
    for(int r = 1; r <= 5; r += 2)
      placeWall(board, 0, r, true, P_WHITE); // vertical walls east of column 0, rows 1..6
    placeWall(board, 0, 2, false, P_BLACK); // horizontal, rows 2 | 3, columns 0 and 1
    board.setFencesLeft(0, 0);
    setPawns(board, sq(0, 4), sq(0, 3));
    shared_ptr<RaceSolution> sol = RaceSolver::solve(board, -0.5f);
    set<Loc> moves = solverMoves(*sol, board, P_BLACK);
    testAssert(moves == boardMoves(board, P_BLACK));
    testAssert(moves.size() == 1 && moves.count(RaceSolution::locOfSq(sq(0, 5))) == 1);
    testAssert(valueAt(*sol, board, P_BLACK) != RaceSolution::INVALID);
  }
}

//------------------------------------------------------------------------------------------------

static float bruteForce(const Board& board, const BoardHistory& hist, Player pla, float alpha, float beta) {
  if(hist.isGameFinished)
    return hist.finalWhiteLead;
  vector<Loc> moves = board.getLegalPawnDestinations(pla);
  // A pawn can be boxed in by walls and the other pawn (rare); the solver values such a state as a draw.
  if(moves.empty())
    return 0.0f;
  if(pla == P_WHITE) {
    float best = -1e9f;
    for(Loc m : moves) {
      Board b2(board);
      BoardHistory h2(hist);
      h2.makeBoardMoveAssumeLegal(b2, m, pla, NULL);
      best = std::max(best, bruteForce(b2, h2, P_BLACK, alpha, beta));
      alpha = std::max(alpha, best);
      if(alpha >= beta)
        break;
    }
    return best;
  }
  else {
    float best = 1e9f;
    for(Loc m : moves) {
      Board b2(board);
      BoardHistory h2(hist);
      h2.makeBoardMoveAssumeLegal(b2, m, pla, NULL);
      best = std::min(best, bruteForce(b2, h2, P_WHITE, alpha, beta));
      beta = std::min(beta, best);
      if(alpha >= beta)
        break;
    }
    return best;
  }
}

static void testBruteForce() {
  cout << "  Solved value equals brute-force minimax over BoardHistory play" << endl;
  Rand rand("testquoridorracesolver brute");
  const int HORIZON = 9;
  int numCompared = 0, numWins = 0, numDraws = 0;
  for(int iter = 0; iter < 400 && numCompared < 150; iter++) {
    Board board = randomWallBoard(rand, rand.nextInt(0, 14));
    int b = sq(rand.nextInt(0, 8), rand.nextInt(1, 3));
    int w = sq(rand.nextInt(0, 8), rand.nextInt(5, 7));
    if(rand.nextBool(0.3)) // near each other
      w = sq(b % 9, std::min(7, b / 9 + 1));
    if(b == w)
      continue;
    setPawns(board, b, w);
    float komi = rand.nextBool(0.7) ? -0.5f : (float)rand.nextInt(-3, 3) + 0.5f;
    Player pla = rand.nextBool(0.5) ? P_BLACK : P_WHITE;
    shared_ptr<RaceSolution> sol = RaceSolver::solve(board, komi);
    int v = valueAt(*sol, board, pla), r = forcedAt(*sol, board, pla);
    if(v == RaceSolution::INVALID)
      continue;
    if(v != 0 && r > HORIZON)
      continue;
    BoardHistory hist = makeHist(board, pla, HORIZON, 0, komi);
    float bf = bruteForce(board, hist, pla, -1e9f, 1e9f);
    if(std::fabs(bf - (float)v / 2.0f) > 1e-4f) {
      cout << Board::toStringSimple(board) << " pla " << (int)pla << " komi " << komi << " solver " << v / 2.0 << " rank " << r
           << " brute " << bf << endl;
      testAssert(false);
    }
    numCompared++;
    if(v == 0)
      numDraws++;
    else
      numWins++;
  }
  // Draw states: neither side can force a result, so the brute force with any horizon must also find 0.
  for(int iter = 0; iter < 300 && numDraws < 25; iter++) {
    Board board = randomWallBoard(rand, rand.nextInt(10, 60));
    shared_ptr<RaceSolution> sol = RaceSolver::solve(board, -0.5f);
    vector<int> draws;
    for(int i = 0; i < RaceSolution::NUM_STATES; i++)
      if(sol->val2[i] == 0)
        draws.push_back(i);
    if(draws.empty())
      continue;
    int i = draws[rand.nextUInt((uint32_t)draws.size())];
    Player pla = i < RaceSolution::NUM_SQ * RaceSolution::NUM_SQ ? P_BLACK : P_WHITE;
    int rest = i % (RaceSolution::NUM_SQ * RaceSolution::NUM_SQ);
    setPawns(board, rest / RaceSolution::NUM_SQ, rest % RaceSolution::NUM_SQ);
    BoardHistory hist = makeHist(board, pla, HORIZON, 0, -0.5f);
    testAssert(bruteForce(board, hist, pla, -1e9f, 1e9f) == 0.0f);
    numCompared++;
    numDraws++;
  }
  cout << "    " << numCompared << " positions compared (" << numWins << " decided, " << numDraws << " draws)" << endl;
  testAssert(numCompared >= 100 && numWins > 30 && numDraws >= 10);
}

static void testOptimalPlay() {
  cout << "  Optimal play on BoardHistory ends with the solved result in the solved number of plies" << endl;
  Rand rand("testquoridorracesolver play");
  int numGames = 0, numLongest = 0;
  for(int iter = 0; iter < 80; iter++) {
    Board board = randomWallBoard(rand, rand.nextInt(0, 30));
    int b = sq(rand.nextInt(0, 8), rand.nextInt(1, 7));
    int w = sq(rand.nextInt(0, 8), rand.nextInt(1, 7));
    if(b == w || b / 9 > w / 9 + 3)
      continue;
    setPawns(board, b, w);
    float komi = rand.nextBool(0.7) ? -0.5f : (float)rand.nextInt(-4, 4) + 0.5f;
    Player pla = rand.nextBool(0.5) ? P_BLACK : P_WHITE;
    BoardHistory hist = makeHist(board, pla, 300, 0, komi);
    RaceResult res = RaceSolver::evaluate(board, hist, pla);
    if(!res.solved || res.winner == C_EMPTY)
      continue;
    testAssert(res.exact);
    Board cur(board);
    Player p = pla;
    int plies = 0;
    while(!hist.isGameFinished) {
      vector<Loc> moves = RaceSolver::optimalMoves(cur, hist, p);
      testAssert(!moves.empty());
      Loc m = moves[rand.nextUInt((uint32_t)moves.size())];
      testAssert(hist.isLegal(cur, m, p));
      hist.makeBoardMoveAssumeLegal(cur, m, p, NULL);
      p = getOpp(p);
      plies++;
      testAssert(plies <= res.plies);
    }
    testAssert(plies == res.plies);
    testAssert(hist.winner == res.winner);
    testAssert(std::fabs(hist.finalWhiteLead - (float)res.lead2 / 2.0f) < 1e-4f);
    numGames++;
    numLongest = std::max(numLongest, plies);
  }
  cout << "    " << numGames << " games, longest " << numLongest << " plies" << endl;
  testAssert(numGames >= 30);
}

static void testEvaluate() {
  cout << "  evaluate(): exact, one-sided, plies left, repetition history" << endl;
  // The jump race of testHandBuiltRaces: Black to move wins in 7 plies.
  auto jumpRace = [](int blackWalls, int whiteWalls) {
    Board board;
    board.setFencesLeft(0, 0);
    setPawns(board, sq(4, 5), sq(4, 4));
    board.setFencesLeft(blackWalls, whiteWalls);
    return board;
  };
  {
    Board board = jumpRace(0, 0);
    BoardHistory hist = makeHist(board, P_BLACK);
    RaceResult r = RaceSolver::evaluate(board, hist, P_BLACK);
    testAssert(r.solved && r.exact && r.winner == P_BLACK && r.lead2 == -1 && r.plies == 7);
    // Plies left: the win needs 7 plies; with maxPlies 7 it still fits (a goal on the last ply is a win), with 6 not.
    testAssert(RaceSolver::evaluate(board, makeHist(board, P_BLACK, 7), P_BLACK).solved);
    testAssert(!RaceSolver::evaluate(board, makeHist(board, P_BLACK, 6), P_BLACK).solved);
  }
  {
    // Black has walls, White none: Black's pawn-only win is proven, not exact.
    Board board = jumpRace(3, 0);
    BoardHistory hist = makeHist(board, P_BLACK);
    RaceResult r = RaceSolver::evaluate(board, hist, P_BLACK);
    testAssert(r.solved && !r.exact && r.winner == P_BLACK && r.lead2 == -1);
    // White has walls, Black none: the race is won by Black, who has no walls: nothing is proven.
    Board board2 = jumpRace(0, 3);
    testAssert(!RaceSolver::evaluate(board2, makeHist(board2, P_BLACK), P_BLACK).solved);
    // Both have walls: unsolved.
    Board board3 = jumpRace(3, 3);
    testAssert(!RaceSolver::evaluate(board3, makeHist(board3, P_BLACK), P_BLACK).solved);
  }
  {
    // A drawn race is a draw only when both have no walls.
    Board board;
    board.setFencesLeft(0, 0);
    // Pawns far from the goals but shuffling forever: a standard start is not a draw (someone wins), so look for any
    // draw state in the empty-board solution.
    shared_ptr<RaceSolution> sol = RaceSolver::solve(board, -0.5f);
    int drawIdx = -1;
    for(int i = 0; i < RaceSolution::NUM_STATES; i++)
      if(sol->val2[i] == 0) {
        drawIdx = i;
        break;
      }
    if(drawIdx >= 0) {
      Player pla = drawIdx < RaceSolution::NUM_SQ * RaceSolution::NUM_SQ ? P_BLACK : P_WHITE;
      int rest = drawIdx % (RaceSolution::NUM_SQ * RaceSolution::NUM_SQ);
      setPawns(board, rest / RaceSolution::NUM_SQ, rest % RaceSolution::NUM_SQ);
      RaceResult r = RaceSolver::evaluate(board, makeHist(board, pla), pla);
      testAssert(r.solved && r.winner == C_EMPTY && r.lead2 == 0);
      board.setFencesLeft(2, 0);
      testAssert(!RaceSolver::evaluate(board, makeHist(board, pla), pla).solved);
    }
  }
  {
    // Repetition history: with the rule on (N = 3), an earlier position that has already occurred twice makes a solved
    // win unreliable; a position that occurred once does not.
    Board board = jumpRace(0, 0);
    BoardHistory hist = makeHist(board, P_BLACK, 300, 3);
    Board cur(board);
    Player p = P_BLACK;
    // Black steps back to (4,6) and forward to (4,5) while White waits at (4,4)? White cannot wait; both shuffle.
    auto play = [&](int c, int r) {
      Loc m = RaceSolution::locOfSq(sq(c, r));
      testAssert(hist.isLegal(cur, m, p));
      hist.makeBoardMoveAssumeLegal(cur, m, p, NULL);
      p = getOpp(p);
    };
    // Black (4,5)->(4,6), White (4,4)->(4,3), Black (4,6)->(4,5), White (4,3)->(4,4): back at the start (count 2).
    play(4, 6); play(4, 3); play(4, 5); play(4, 4);
    testAssert(hist.currentPositionRepetitionCount() == 2);
    // The current position has occurred twice, but a solved line never returns to it: still solved.
    RaceResult r = RaceSolver::evaluate(cur, hist, p);
    testAssert(r.solved && r.winner == P_BLACK);
    // Once more around: the intermediate positions have each occurred twice, so a line through them would be a draw.
    play(4, 6); play(4, 3);
    testAssert(hist.currentPositionRepetitionCount() == 2 || !hist.isGameFinished);
    play(4, 5);
    // Now the position after White's next shuffle would be its third occurrence; at this position the earlier
    // positions with count 2 exist (the start, and the others once around), so nothing is solved here.
    RaceResult r2 = RaceSolver::evaluate(cur, hist, p);
    testAssert(!r2.solved);
    // The same history with the rule off is solved.
    BoardHistory histOff = makeHist(board, P_BLACK, 300, 0);
    Board cur2(board);
    Player p2 = P_BLACK;
    for(auto cr : vector<pair<int, int>>{{4, 6}, {4, 3}, {4, 5}, {4, 4}, {4, 6}, {4, 3}, {4, 5}}) {
      Loc m = RaceSolution::locOfSq(sq(cr.first, cr.second));
      histOff.makeBoardMoveAssumeLegal(cur2, m, p2, NULL);
      p2 = getOpp(p2);
    }
    testAssert(RaceSolver::evaluate(cur2, histOff, p2).solved);
  }
}

static void testSolveCost() {
  cout << "  Solve cost" << endl;
  Rand rand("testquoridorracesolver cost");
  int numBoards = 100;
  int numDrawStates = 0, numBoardsWithDraws = 0;
  double totalMs = 0, maxMs = 0;
  for(int iter = 0; iter < numBoards; iter++) {
    Board board = randomWallBoard(rand, iter < 20 ? 0 : rand.nextInt(1, 60));
    auto t0 = std::chrono::steady_clock::now();
    shared_ptr<RaceSolution> sol = RaceSolver::solve(board, -0.5f);
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    totalMs += ms;
    maxMs = std::max(maxMs, ms);
    int draws = 0;
    for(int i = 0; i < RaceSolution::NUM_STATES; i++)
      if(sol->val2[i] == 0)
        draws++;
    numDrawStates += draws;
    if(draws > 0)
      numBoardsWithDraws++;
  }
  cout << "    " << numBoards << " wall configurations: mean " << totalMs / numBoards << " ms, max " << maxMs << " ms; "
       << numBoardsWithDraws << " have draw states (" << numDrawStates / numBoards << " per configuration on average)" << endl;
}

static void testCache() {
  cout << "  Cache" << endl;
  RaceSolver::clearCache();
  RaceSolver::resetCacheStats();
  Board board;
  board.setFencesLeft(0, 0);
  auto s1 = RaceSolver::getCached(board, -0.5f);
  auto s2 = RaceSolver::getCached(board, -0.5f);
  testAssert(s1 == s2);
  auto s3 = RaceSolver::getCached(board, 0.5f);
  testAssert(s3 != s1 && s3->komi2 == 1 && s1->komi2 == -1);
  Board board2(board);
  board2.setFencesLeft(10, 10);
  testAssert(board2.isLegalWallPlacement(3, 3, true, P_BLACK));
  board2.playMove(Location::vWallLoc(3, 3), P_BLACK);
  auto s4 = RaceSolver::getCached(board2, -0.5f);
  testAssert(s4 != s1);
  // Pawn placements and wall counts don't matter for the key.
  Board board3(board);
  setPawns(board3, sq(1, 1), sq(7, 7));
  testAssert(RaceSolver::getCached(board3, -0.5f) == s1);
  RaceSolver::CacheStats st = RaceSolver::getCacheStats();
  testAssert(st.misses == 3 && st.hits == 2 && st.entries == 3);
  // Bounded size.
  size_t oldMax = RaceSolver::getMaxCacheEntries();
  RaceSolver::setMaxCacheEntries(2);
  testAssert(RaceSolver::getCacheStats().entries == 2);
  RaceSolver::setMaxCacheEntries(oldMax);
  RaceSolver::clearCache();
}

}  // namespace

void Tests::runQuoridorRaceSolverTests() {
  cout << "Running Quoridor race solver tests" << endl;
  testMoveGeneration();
  testHandBuiltRaces();
  testBruteForce();
  testOptimalPlay();
  testEvaluate();
  testSolveCost();
  testCache();
  cout << "Quoridor race solver tests passed" << endl;
}
