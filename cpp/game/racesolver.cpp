#include "../game/racesolver.h"

#include <algorithm>
#include <chrono>
#include <deque>

using namespace std;

// Directions: 0 north (row - 1), 1 west, 2 east, 3 south (row + 1), the order of Board's pawn moves.
static const int DIR_DC[4] = {0, -1, 1, 0};
static const int DIR_DR[4] = {-1, 0, 0, 1};

//----------------------------------------------------------------------------------------------------------------------
// RaceSolution
//----------------------------------------------------------------------------------------------------------------------

int RaceSolution::genMoves(int me, int opp, int8_t out[MAX_MOVES]) const {
  int n = 0;
  for(int d = 0; d < 4; d++) {
    int step1 = nbr[me][d];
    if(step1 < 0)
      continue;
    if(step1 != opp) {
      out[n++] = (int8_t)step1;
      continue;
    }
    // The opponent is adjacent: jump straight over it, or diagonally if the straight jump is blocked by a wall or the edge.
    int straight = nbr[opp][d];
    if(straight >= 0) {
      out[n++] = (int8_t)straight;
    }
    else {
      const int pd0 = (d == 0 || d == 3) ? 1 : 0;
      const int pd1 = (d == 0 || d == 3) ? 2 : 3;
      int diag0 = nbr[opp][pd0];
      int diag1 = nbr[opp][pd1];
      if(diag0 >= 0)
        out[n++] = (int8_t)diag0;
      if(diag1 >= 0)
        out[n++] = (int8_t)diag1;
    }
  }
  return n;
}

int RaceSolution::terminalVal2(Player mover, int loserSq) const {
  if(mover == P_WHITE) {
    int d = std::max(1, (int)distBlack[loserSq]);
    return 2 * d + komi2;
  }
  else {
    int d = std::max(1, (int)distWhite[loserSq]);
    return 2 * (1 - d) + komi2;
  }
}

int RaceSolution::numSolvedStates() const {
  int n = 0;
  for(int i = 0; i < NUM_STATES; i++)
    if(val2[i] != INVALID)
      n++;
  return n;
}

//----------------------------------------------------------------------------------------------------------------------
// Solver
//----------------------------------------------------------------------------------------------------------------------

namespace {
  struct Graph {
    // edges of state i: edgeStart[i] .. edgeStart[i+1]; edgeTo = successor state or -1 for an arrival at the goal
    vector<int32_t> edgeStart;
    vector<int32_t> edgeTo;
    vector<int16_t> edgeVal; // for terminal edges
    // predecessors of state j: revFrom[revStart[j] .. revStart[j+1]]
    vector<int32_t> revStart;
    vector<int32_t> revFrom;
    // terminal edges (source state, value), sorted by value ascending
    vector<pair<int16_t, int32_t>> terminals;
    vector<uint8_t> outDeg;
    vector<uint8_t> valid;
  };

  struct Attractor {
    vector<uint8_t> won;
    vector<int16_t> rank;
    vector<uint8_t> cnt;
    vector<int32_t> queue;
  };
}

static void buildGraph(RaceSolution& sol, Graph& g) {
  const int S = RaceSolution::NUM_STATES;
  const int NS = RaceSolution::NUM_SQ;
  g.edgeStart.assign(S + 1, 0);
  g.edgeTo.clear();
  g.edgeVal.clear();
  g.edgeTo.reserve(S * 4);
  g.edgeVal.reserve(S * 4);
  g.outDeg.assign(S, 0);
  g.valid.assign(S, 0);
  g.terminals.clear();
  vector<int32_t> predCount(S + 1, 0);

  for(int pi = 0; pi < 2; pi++) {
    Player pla = pi == 0 ? P_BLACK : P_WHITE;
    for(int b = 0; b < NS; b++) {
      for(int w = 0; w < NS; w++) {
        int i = (pi * NS + b) * NS + w;
        g.edgeStart[i] = (int32_t)g.edgeTo.size();
        sol.val2[i] = RaceSolution::INVALID;
        sol.plies[i] = 0;
        sol.forced[i] = 0;
        if(b == w || b < 9 || w >= 72)
          continue; // same square, or a pawn already on its goal row: not part of the race
        g.valid[i] = 1;
        sol.val2[i] = 0;
        int me = pla == P_BLACK ? b : w;
        int opp = pla == P_BLACK ? w : b;
        int8_t moves[RaceSolution::MAX_MOVES];
        int n = sol.genMoves(me, opp, moves);
        g.outDeg[i] = (uint8_t)n;
        for(int k = 0; k < n; k++) {
          int t = moves[k];
          bool arrived = pla == P_BLACK ? (t < 9) : (t >= 72);
          if(arrived) {
            int16_t v = (int16_t)sol.terminalVal2(pla, opp);
            g.edgeTo.push_back(-1);
            g.edgeVal.push_back(v);
            g.terminals.emplace_back(v, (int32_t)i);
          }
          else {
            int nb = pla == P_BLACK ? t : b;
            int nw = pla == P_BLACK ? w : t;
            int j = RaceSolution::stateIndex(getOpp(pla), nb, nw);
            g.edgeTo.push_back(j);
            g.edgeVal.push_back(0);
            predCount[j]++;
          }
        }
      }
    }
  }
  g.edgeStart[S] = (int32_t)g.edgeTo.size();

  g.revStart.assign(S + 1, 0);
  for(int j = 0; j < S; j++)
    g.revStart[j + 1] = g.revStart[j] + predCount[j];
  g.revFrom.assign(g.revStart[S], 0);
  vector<int32_t> fill(g.revStart.begin(), g.revStart.begin() + S);
  for(int i = 0; i < S; i++) {
    for(int e = g.edgeStart[i]; e < g.edgeStart[i + 1]; e++) {
      int j = g.edgeTo[e];
      if(j >= 0)
        g.revFrom[fill[j]++] = i;
    }
  }
  std::sort(g.terminals.begin(), g.terminals.end());
}

// Computes the states from which `attackerIsWhite`'s player can force the game to end with 2s >= thr (White) or 2s <= thr
// (Black) within finitely many plies, with the number of plies of that forced end (the attacker minimizing, the
// defender maximizing).
static void computeAttractor(const Graph& g, bool attackerIsWhite, int thr, Attractor& a) {
  const int S = RaceSolution::NUM_STATES;
  const int NS = RaceSolution::NUM_SQ;
  const int whiteBase = NS * NS; // states >= whiteBase have White to move
  a.won.assign(S, 0);
  a.cnt = g.outDeg;
  a.queue.clear();
  if((int)a.rank.size() != S)
    a.rank.assign(S, 0);

  auto isAttackerMover = [&](int i) { return (i >= whiteBase) == attackerIsWhite; };

  // Seed with the terminal edges that reach the target.
  if(attackerIsWhite) {
    for(auto it = g.terminals.rbegin(); it != g.terminals.rend() && it->first >= thr; ++it) {
      int i = it->second;
      if(a.won[i])
        continue;
      if(isAttackerMover(i)) {
        a.won[i] = 1;
        a.rank[i] = 1;
        a.queue.push_back(i);
      }
      else if(--a.cnt[i] == 0) {
        a.won[i] = 1;
        a.rank[i] = 1;
        a.queue.push_back(i);
      }
    }
  }
  else {
    for(auto it = g.terminals.begin(); it != g.terminals.end() && it->first <= thr; ++it) {
      int i = it->second;
      if(a.won[i])
        continue;
      if(isAttackerMover(i)) {
        a.won[i] = 1;
        a.rank[i] = 1;
        a.queue.push_back(i);
      }
      else if(--a.cnt[i] == 0) {
        a.won[i] = 1;
        a.rank[i] = 1;
        a.queue.push_back(i);
      }
    }
  }
  // The seeds above are pushed in an order that is not by rank (all rank 1), which is fine. Defender states that need
  // several terminal edges to resolve, plus later propagation, keep the FIFO order by rank.
  for(size_t qi = 0; qi < a.queue.size(); qi++) {
    int j = a.queue[qi];
    int16_t rj = a.rank[j];
    for(int k = g.revStart[j]; k < g.revStart[j + 1]; k++) {
      int p = g.revFrom[k];
      if(a.won[p])
        continue;
      if(isAttackerMover(p)) {
        a.won[p] = 1;
        a.rank[p] = (int16_t)(rj + 1);
        a.queue.push_back(p);
      }
      else if(--a.cnt[p] == 0) {
        a.won[p] = 1;
        a.rank[p] = (int16_t)(rj + 1);
        a.queue.push_back(p);
      }
    }
  }
}

// Gives the states that the attractor of `level` wins, and that are not decided at a better level, their value, the
// forced number of plies (the attractor rank) and the number of plies of optimal play that keeps the value. The queue is
// in order of rank, and the successors that keep the value of a state have a smaller rank, so they are done before it.
static void assignLevel(const Graph& g, RaceSolution& sol, const Attractor& a, bool attackerIsWhite, int level, vector<uint8_t>& assigned) {
  const int whiteBase = RaceSolution::NUM_SQ * RaceSolution::NUM_SQ;
  for(int32_t i : a.queue) {
    if(assigned[i])
      continue;
    assigned[i] = 1;
    sol.val2[i] = (int16_t)level;
    sol.forced[i] = a.rank[i];
    const bool moverIsAttacker = (i >= whiteBase) == attackerIsWhite;
    int best = moverIsAttacker ? 32000 : -1;
    for(int e = g.edgeStart[i]; e < g.edgeStart[i + 1]; e++) {
      int to = g.edgeTo[e];
      int sv, sr;
      if(to < 0) {
        sv = g.edgeVal[e];
        sr = 0;
      }
      else {
        sv = sol.val2[to];
        sr = sol.plies[to];
      }
      if(sv != level)
        continue;
      best = moverIsAttacker ? std::min(best, sr) : std::max(best, sr);
    }
    sol.plies[i] = (int16_t)(best + 1);
  }
}

shared_ptr<RaceSolution> RaceSolver::solve(const Board& board, float komi) {
  auto solp = std::make_shared<RaceSolution>();
  RaceSolution& sol = *solp;
  sol.komi2 = (int)lround(2.0 * (double)komi);

  const int xSize = board.x_size;
  int32_t dist[81];
  board.calDistMap(P_BLACK, dist);
  for(int sq = 0; sq < 81; sq++)
    sol.distBlack[sq] = (int8_t)dist[sq];
  board.calDistMap(P_WHITE, dist);
  for(int sq = 0; sq < 81; sq++)
    sol.distWhite[sq] = (int8_t)dist[sq];
  for(int sq = 0; sq < 81; sq++) {
    int c = sq % 9, r = sq / 9;
    Loc from = Location::pawnLoc(c, r, xSize);
    for(int d = 0; d < 4; d++) {
      int nc = c + DIR_DC[d], nr = r + DIR_DR[d];
      if(nc < 0 || nc >= 9 || nr < 0 || nr >= 9) {
        sol.nbr[sq][d] = -1;
        continue;
      }
      Loc to = Location::pawnLoc(nc, nr, xSize);
      sol.nbr[sq][d] = board.canPawnStep(from, to) ? (int8_t)(nr * 9 + nc) : (int8_t)-1;
    }
  }

  Graph g;
  buildGraph(sol, g);

  // Distinct terminal values.
  vector<int16_t> positive, negative;
  for(const auto& t : g.terminals) {
    if(t.first > 0) {
      if(positive.empty() || positive.back() != t.first)
        positive.push_back(t.first);
    }
    else {
      if(negative.empty() || negative.back() != t.first)
        negative.push_back(t.first);
    }
  }

  const int S = RaceSolution::NUM_STATES;
  vector<uint8_t> assigned(S, 0);
  Attractor a;
  // White wins: the largest threshold first, so that a state gets the largest lead it can force.
  for(auto it = positive.rbegin(); it != positive.rend(); ++it) {
    computeAttractor(g, true, *it, a);
    assignLevel(g, sol, a, true, *it, assigned);
  }
  // Black wins: the most negative threshold first.
  for(auto it = negative.begin(); it != negative.end(); ++it) {
    computeAttractor(g, false, *it, a);
    assignLevel(g, sol, a, false, *it, assigned);
  }
  return solp;
}

//----------------------------------------------------------------------------------------------------------------------
// Cache
//----------------------------------------------------------------------------------------------------------------------

namespace {
  struct CacheKey {
    uint64_t v;
    uint64_t h;
    int komi2;
    bool operator==(const CacheKey& o) const { return v == o.v && h == o.h && komi2 == o.komi2; }
  };
  struct CacheKeyHash {
    size_t operator()(const CacheKey& k) const {
      uint64_t x = k.v * 0x9E3779B97F4A7C15ULL;
      x ^= (k.h + 0x7F4A7C15ULL) * 0xC2B2AE3D27D4EB4FULL;
      x ^= (uint64_t)(uint32_t)k.komi2 * 0x165667B19E3779F9ULL;
      x ^= x >> 29;
      return (size_t)x;
    }
  };

  struct Cache {
    mutex mtx;
    unordered_map<CacheKey, shared_ptr<const RaceSolution>, CacheKeyHash> map;
    deque<CacheKey> order;
    size_t maxEntries = 1024;
    atomic<uint64_t> hits{0};
    atomic<uint64_t> misses{0};
    atomic<uint64_t> solveMicros{0};
  };

  Cache& cache() {
    static Cache c;
    return c;
  }

  CacheKey makeKey(const Board& board, float komi) {
    CacheKey k;
    k.v = 0;
    k.h = 0;
    for(int c = 0; c < 8; c++) {
      for(int r = 0; r < 8; r++) {
        int bit = c * 8 + r;
        if(board.vWalls[c][r])
          k.v |= (uint64_t)1 << bit;
        if(board.hWalls[c][r])
          k.h |= (uint64_t)1 << bit;
      }
    }
    k.komi2 = (int)lround(2.0 * (double)komi);
    return k;
  }
}

shared_ptr<const RaceSolution> RaceSolver::getCached(const Board& board, float komi) {
  Cache& c = cache();
  CacheKey key = makeKey(board, komi);
  {
    lock_guard<mutex> lock(c.mtx);
    auto it = c.map.find(key);
    if(it != c.map.end()) {
      c.hits.fetch_add(1, memory_order_relaxed);
      return it->second;
    }
  }
  c.misses.fetch_add(1, memory_order_relaxed);
  auto t0 = chrono::steady_clock::now();
  shared_ptr<const RaceSolution> sol = solve(board, komi);
  auto t1 = chrono::steady_clock::now();
  c.solveMicros.fetch_add((uint64_t)chrono::duration_cast<chrono::microseconds>(t1 - t0).count(), memory_order_relaxed);
  {
    lock_guard<mutex> lock(c.mtx);
    if(c.map.find(key) == c.map.end()) {
      while(c.map.size() >= c.maxEntries && !c.order.empty()) {
        c.map.erase(c.order.front());
        c.order.pop_front();
      }
      c.map[key] = sol;
      c.order.push_back(key);
    }
  }
  return sol;
}

void RaceSolver::setMaxCacheEntries(size_t n) {
  Cache& c = cache();
  lock_guard<mutex> lock(c.mtx);
  c.maxEntries = std::max((size_t)1, n);
  while(c.map.size() > c.maxEntries && !c.order.empty()) {
    c.map.erase(c.order.front());
    c.order.pop_front();
  }
}

size_t RaceSolver::getMaxCacheEntries() {
  Cache& c = cache();
  lock_guard<mutex> lock(c.mtx);
  return c.maxEntries;
}

void RaceSolver::clearCache() {
  Cache& c = cache();
  lock_guard<mutex> lock(c.mtx);
  c.map.clear();
  c.order.clear();
}

RaceSolver::CacheStats RaceSolver::getCacheStats() {
  Cache& c = cache();
  CacheStats s;
  s.hits = c.hits.load();
  s.misses = c.misses.load();
  s.solveMicros = c.solveMicros.load();
  {
    lock_guard<mutex> lock(c.mtx);
    s.entries = c.map.size();
  }
  return s;
}

void RaceSolver::resetCacheStats() {
  Cache& c = cache();
  c.hits = 0;
  c.misses = 0;
  c.solveMicros = 0;
}

//----------------------------------------------------------------------------------------------------------------------
// Evaluation of positions
//----------------------------------------------------------------------------------------------------------------------

bool RaceSolver::isSupportedBoard(const Board& board) {
  return board.x_size == 17 && board.y_size == 17;
}

// With the repetition rule on, an earlier position of the history that has already occurred N-1 times (other than the
// current position, which a solved line never revisits) turns a line that enters it into a draw. Such a history makes
// the solved value unreliable.
static bool repetitionHistoryIsClean(const BoardHistory& hist) {
  const int n = hist.rules.repetitionDrawCount;
  if(n <= 0)
    return true;
  const vector<Hash128>& ps = hist.positionsSinceLastWall;
  if(ps.size() < 2)
    return true;
  Hash128 cur = ps.back();
  vector<Hash128> prev;
  prev.reserve(ps.size());
  for(size_t i = 0; i + 1 < ps.size(); i++)
    if(!(ps[i] == cur))
      prev.push_back(ps[i]);
  if((int)prev.size() < n - 1)
    return true;
  std::sort(prev.begin(), prev.end(), [](const Hash128& a, const Hash128& b) {
    return a.hash0 != b.hash0 ? a.hash0 < b.hash0 : a.hash1 < b.hash1;
  });
  int run = 1;
  for(size_t i = 1; i <= prev.size(); i++) {
    if(i < prev.size() && prev[i] == prev[i - 1]) {
      run++;
    }
    else {
      if(run >= n - 1)
        return false;
      run = 1;
    }
  }
  return true;
}

RaceResult RaceSolver::evaluate(const Board& board, const BoardHistory& hist, Player nextPla) {
  RaceResult res;
  if(hist.isGameFinished || !isSupportedBoard(board))
    return res;
  const int bf = board.blackFences, wf = board.whiteFences;
  if(bf > 0 && wf > 0)
    return res;
  const bool exact = bf == 0 && wf == 0;
  const Player wallPla = bf > 0 ? P_BLACK : (wf > 0 ? P_WHITE : C_EMPTY);

  shared_ptr<const RaceSolution> sol = getCached(board, hist.rules.komi);
  int b = RaceSolution::sqOfLoc(board.blackPawnLoc, board.x_size);
  int w = RaceSolution::sqOfLoc(board.whitePawnLoc, board.x_size);
  int idx = RaceSolution::stateIndex(nextPla, b, w);
  int v = sol->val2[idx];
  if(v == RaceSolution::INVALID)
    return res;

  if(v == 0) {
    if(!exact)
      return res; // a draw of the race is not a draw of the game for the player with walls
    res.solved = true;
    res.exact = true;
    res.winner = C_EMPTY;
    res.lead2 = 0;
    res.plies = 0;
  }
  else {
    Player winner = v > 0 ? P_WHITE : P_BLACK;
    if(!exact && winner != wallPla)
      return res;
    int plies = sol->plies[idx];
    if((int64_t)sol->forced[idx] > hist.pliesUntilDraw())
      return res;
    if(!repetitionHistoryIsClean(hist))
      return res;
    res.solved = true;
    res.exact = exact;
    res.winner = winner;
    res.lead2 = v;
    res.plies = plies;
  }
  return res;
}

vector<Loc> RaceSolver::optimalMoves(const Board& board, const BoardHistory& hist, Player pla) {
  vector<Loc> out;
  RaceResult r = evaluate(board, hist, pla);
  if(!r.solved || !r.exact)
    return out;
  shared_ptr<const RaceSolution> sol = getCached(board, hist.rules.komi);
  int b = RaceSolution::sqOfLoc(board.blackPawnLoc, board.x_size);
  int w = RaceSolution::sqOfLoc(board.whitePawnLoc, board.x_size);
  int me = pla == P_BLACK ? b : w;
  int opp = pla == P_BLACK ? w : b;
  int v = sol->val2[RaceSolution::stateIndex(pla, b, w)];
  int8_t moves[RaceSolution::MAX_MOVES];
  int n = sol->genMoves(me, opp, moves);
  const bool moverWins = (v > 0 && pla == P_WHITE) || (v < 0 && pla == P_BLACK);

  int bestRank = 0;
  vector<int> keep;
  for(int k = 0; k < n; k++) {
    int t = moves[k];
    bool arrived = pla == P_BLACK ? (t < 9) : (t >= 72);
    int succVal, succRank;
    if(arrived) {
      succVal = sol->terminalVal2(pla, opp);
      succRank = 0;
    }
    else {
      int nb = pla == P_BLACK ? t : b;
      int nw = pla == P_BLACK ? w : t;
      int j = RaceSolution::stateIndex(getOpp(pla), nb, nw);
      succVal = sol->val2[j];
      succRank = sol->plies[j];
    }
    if(succVal != v)
      continue;
    if(v == 0) {
      keep.push_back(t);
      continue;
    }
    if(keep.empty()) {
      keep.push_back(t);
      bestRank = succRank;
    }
    else if((moverWins && succRank < bestRank) || (!moverWins && succRank > bestRank)) {
      keep.assign(1, t);
      bestRank = succRank;
    }
    else if(succRank == bestRank) {
      keep.push_back(t);
    }
  }
  for(int t : keep)
    out.push_back(RaceSolution::locOfSq(t));
  return out;
}
