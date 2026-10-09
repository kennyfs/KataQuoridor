#include "q4board.h"

#include "../core/rand.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>

const int Q4Board::DIR_OFFSET[4] = {+11, +1, -11, -1};
const int Q4Board::DIR_OPPOSITE[4] = {2, 3, 0, 1};
const int Q4Board::START_CELLS[4] = {5, 55, 115, 65};

bool Q4Board::IS_ZOBRIST_INITIALIZED = false;

#ifndef NDEBUG
static Q4Bits121 columnCells(int x) {
  Q4Bits121 r;
  for(int y = 0; y < Q4Board::BOARD_SIZE; y++)
    r.set(Q4Board::cellOf(x, y));
  return r;
}
static const Q4Bits121 COL_X0 = columnCells(0);
static const Q4Bits121 COL_X10 = columnCells(Q4Board::BOARD_SIZE - 1);
#endif
Hash128 Q4Board::ZOBRIST_PAWN[NUM_SEATS][NUM_CELLS];
Hash128 Q4Board::ZOBRIST_HWALL[NUM_WALL_ANCHORS];
Hash128 Q4Board::ZOBRIST_VWALL[NUM_WALL_ANCHORS];
Hash128 Q4Board::ZOBRIST_WALLS_LEFT[NUM_SEATS][11];
Hash128 Q4Board::ZOBRIST_TO_MOVE[NUM_SEATS];
Hash128 Q4Board::ZOBRIST_ALIVE[NUM_SEATS];

void Q4Board::initHash() {
  if(IS_ZOBRIST_INITIALIZED)
    return;
  Rand rand("Q4Board::initHash() for KataQuoridor Four at a Table");

  auto nextHash = [&rand]() {
    uint64_t h0 = rand.nextUInt64();
    uint64_t h1 = rand.nextUInt64();
    return Hash128(h0, h1);
  };

  for(int s = 0; s < NUM_SEATS; s++) {
    for(int c = 0; c < NUM_CELLS; c++)
      ZOBRIST_PAWN[s][c] = nextHash();
    for(int w = 0; w <= 10; w++)
      ZOBRIST_WALLS_LEFT[s][w] = nextHash();
    ZOBRIST_TO_MOVE[s] = nextHash();
    ZOBRIST_ALIVE[s] = nextHash();
  }

  for(int a = 0; a < NUM_WALL_ANCHORS; a++) {
    ZOBRIST_HWALL[a] = nextHash();
    ZOBRIST_VWALL[a] = nextHash();
  }

  IS_ZOBRIST_INITIALIZED = true;
}

Q4Board::Q4Board() : Q4Board(Q4Rules()) {}

Q4Board::Q4Board(const Q4Rules& rules) {
  initHash();
  alive = 0x0F;  // Seats 0, 1, 2, 3 all alive
  toMove = 0;

  std::fill(std::begin(occupant), std::end(occupant), -1);
  for(int s = 0; s < NUM_SEATS; s++) {
    pawn[s] = START_CELLS[s];
    occupant[START_CELLS[s]] = s;
    wallsLeft[s] = rules.initialWalls[s];
  }

  hWalls.reset();
  vWalls.reset();

  initGridBoundaries();
  recomputeDistancesToCenter();

  hash = getHashFromScratch();
}

void Q4Board::initGridBoundaries() {
  for(int c = 0; c < NUM_CELLS; c++) {
    int x = cellX(c);
    int y = cellY(c);
    uint8_t mask = 0;
    if(y == 10) mask |= (1 << DIR_N);
    if(x == 10) mask |= (1 << DIR_E);
    if(y == 0)  mask |= (1 << DIR_S);
    if(x == 0)  mask |= (1 << DIR_W);
    blocked[c] = mask;
    syncCanMasks(c);
  }
}

void Q4Board::syncCanMasks(int c) {
  uint8_t b = blocked[c];
  canN.assign(c, !(b & (1 << DIR_N)));
  canE.assign(c, !(b & (1 << DIR_E)));
  canS.assign(c, !(b & (1 << DIR_S)));
  canW.assign(c, !(b & (1 << DIR_W)));
}

void Q4Board::addWallToBlocked(int ax, int ay, bool isHorizontal) {
  if(isHorizontal) {
    int c0 = ay * BOARD_SIZE + ax;
    int c1 = c0 + 1;
    blocked[c0] |= (1 << DIR_N);
    blocked[c0 + DIR_OFFSET[DIR_N]] |= (1 << DIR_S);
    blocked[c1] |= (1 << DIR_N);
    blocked[c1 + DIR_OFFSET[DIR_N]] |= (1 << DIR_S);
    syncCanMasks(c0); syncCanMasks(c1);
    syncCanMasks(c0 + BOARD_SIZE); syncCanMasks(c1 + BOARD_SIZE);
  }
  else {
    int c0 = ay * BOARD_SIZE + ax;
    int c1 = c0 + DIR_OFFSET[DIR_N];
    blocked[c0] |= (1 << DIR_E);
    blocked[c0 + DIR_OFFSET[DIR_E]] |= (1 << DIR_W);
    blocked[c1] |= (1 << DIR_E);
    blocked[c1 + DIR_OFFSET[DIR_E]] |= (1 << DIR_W);
    syncCanMasks(c0); syncCanMasks(c0 + 1);
    syncCanMasks(c1); syncCanMasks(c1 + 1);
  }
}

void Q4Board::removeWallFromBlocked(int ax, int ay, bool isHorizontal) {
  if(isHorizontal) {
    int c0 = ay * BOARD_SIZE + ax;
    int c1 = c0 + 1;
    blocked[c0] &= ~(1 << DIR_N);
    blocked[c0 + DIR_OFFSET[DIR_N]] &= ~(1 << DIR_S);
    blocked[c1] &= ~(1 << DIR_N);
    blocked[c1 + DIR_OFFSET[DIR_N]] &= ~(1 << DIR_S);
    syncCanMasks(c0); syncCanMasks(c1);
    syncCanMasks(c0 + BOARD_SIZE); syncCanMasks(c1 + BOARD_SIZE);
  }
  else {
    int c0 = ay * BOARD_SIZE + ax;
    int c1 = c0 + DIR_OFFSET[DIR_N];
    blocked[c0] &= ~(1 << DIR_E);
    blocked[c0 + DIR_OFFSET[DIR_E]] &= ~(1 << DIR_W);
    blocked[c1] &= ~(1 << DIR_E);
    blocked[c1 + DIR_OFFSET[DIR_E]] &= ~(1 << DIR_W);
    syncCanMasks(c0); syncCanMasks(c0 + 1);
    syncCanMasks(c1); syncCanMasks(c1 + 1);
  }
}

int Q4Board::getNumAlive() const {
  int count = 0;
  for(int s = 0; s < NUM_SEATS; s++) {
    if(alive & (1 << s))
      count++;
  }
  return count;
}

int Q4Board::getNextAlive(int currSeat) const {
  for(int step = 1; step <= NUM_SEATS; step++) {
    int s = (currSeat + step) % NUM_SEATS;
    if(alive & (1 << s))
      return s;
  }
  return currSeat;
}

bool Q4Board::wallConflicts(int ax, int ay, bool isHorizontal) const {
  if(ax < 0 || ax >= NUM_ANCHORS || ay < 0 || ay >= NUM_ANCHORS)
    return true;
  int a = anchorOf(ax, ay);
  if(isHorizontal) {
    if(vWalls.test(a)) return true;  // Crossing
    if(hWalls.test(a)) return true;  // Overlap
    if(ax > 0 && hWalls.test(a - 1)) return true; // End-to-end adjacent left
    if(ax < NUM_ANCHORS - 1 && hWalls.test(a + 1)) return true; // End-to-end adjacent right
  }
  else {
    if(hWalls.test(a)) return true;  // Crossing
    if(vWalls.test(a)) return true;  // Overlap
    if(ay > 0 && vWalls.test(a - NUM_ANCHORS)) return true; // End-to-end adjacent down
    if(ay < NUM_ANCHORS - 1 && vWalls.test(a + NUM_ANCHORS)) return true; // End-to-end adjacent up
  }
  return false;
}

bool Q4Board::isGeometricallyLegalWall(int ax, int ay, bool isHorizontal) const {
  if(wallConflicts(ax, ay, isHorizontal))
    return false;

  if(isHorizontal) {
    int e1 = ay * BOARD_SIZE + ax;
    int e2 = ay * BOARD_SIZE + (ax + 1);
    if(!hPathEdges.test(e1) && !hPathEdges.test(e2))
      return true;
  }
  else {
    int e1 = ay * NUM_ANCHORS + ax;
    int e2 = (ay + 1) * NUM_ANCHORS + ax;
    if(!vPathEdges.test(e1) && !vPathEdges.test(e2))
      return true;
  }

  return isLegalWallBruteForce(ax, ay, isHorizontal);
}

bool Q4Board::isLegalWallBruteForce(int ax, int ay, bool isHorizontal) const {
  if(wallConflicts(ax, ay, isHorizontal))
    return false;

  Q4Bits121 needed;
  for(int s = 0; s < NUM_SEATS; s++) {
    int p = pawn[s];
    if(isAlive(s) && p >= 0 && p != CENTER_CELL)
      needed.set(p);
  }
  if(needed == Q4Bits121())
    return true;

  // Masks with the edges of the new wall closed
  Q4Bits121 cN = canN, cE = canE, cS = canS, cW = canW;
  int c0 = cellOf(ax, ay);
  if(isHorizontal) {
    cN.clear(c0);
    cN.clear(c0 + 1);
    cS.clear(c0 + BOARD_SIZE);
    cS.clear(c0 + BOARD_SIZE + 1);
  }
  else {
    cE.clear(c0);
    cE.clear(c0 + BOARD_SIZE);
    cW.clear(c0 + 1);
    cW.clear(c0 + BOARD_SIZE + 1);
  }

  // Board-edge bits in blocked[] must keep the shifts from wrapping across rows
  assert((cE & COL_X10) == Q4Bits121());
  assert((cW & COL_X0) == Q4Bits121());

  Q4Bits121 reached;
  reached.set(CENTER_CELL);
  while(true) {
    Q4Bits121 next = reached
      | (reached & cN).shl<BOARD_SIZE>()
      | (reached & cE).shl<1>()
      | (reached & cS).shr<BOARD_SIZE>()
      | (reached & cW).shr<1>();
    if((next & needed) == needed)
      return true;
    if(next == reached)
      return false;
    reached = next;
  }
}

bool Q4Board::isLegalWallBruteForceBFS(int ax, int ay, bool isHorizontal) const {
  if(wallConflicts(ax, ay, isHorizontal))
    return false;

  // Local copy of blocked to eliminate const_cast and data races
  uint8_t localBlocked[NUM_CELLS];
  std::memcpy(localBlocked, blocked, sizeof(blocked));

  int c0 = cellOf(ax, ay);
  if(isHorizontal) {
    localBlocked[c0] |= (1 << DIR_N);
    localBlocked[c0 + BOARD_SIZE] |= (1 << DIR_S);
    localBlocked[c0 + 1] |= (1 << DIR_N);
    localBlocked[c0 + BOARD_SIZE + 1] |= (1 << DIR_S);
  }
  else {
    localBlocked[c0] |= (1 << DIR_E);
    localBlocked[c0 + 1] |= (1 << DIR_W);
    localBlocked[c0 + BOARD_SIZE] |= (1 << DIR_E);
    localBlocked[c0 + BOARD_SIZE + 1] |= (1 << DIR_W);
  }

  // BFS from CENTER_CELL
  uint8_t q[NUM_CELLS];
  bool visited[NUM_CELLS] = {false};
  int head = 0, tail = 0;
  q[tail++] = CENTER_CELL;
  visited[CENTER_CELL] = true;

  int neededMask = 0;
  for(int s = 0; s < NUM_SEATS; s++) {
    if(isAlive(s)) {
      int p = pawn[s];
      if(p >= 0 && p != CENTER_CELL)
        neededMask |= (1 << s);
    }
  }
  if(neededMask == 0)
    return true;

  int foundMask = 0;
  while(head < tail) {
    int curr = q[head++];
    for(int dir = 0; dir < 4; dir++) {
      if(!(localBlocked[curr] & (1 << dir))) {
        int nxt = curr + DIR_OFFSET[dir];
        if(!visited[nxt]) {
          visited[nxt] = true;
          int occ = occupant[nxt];
          if(occ >= 0 && (neededMask & (1 << occ))) {
            foundMask |= (1 << occ);
            if(foundMask == neededMask)
              return true;
          }
          q[tail++] = nxt;
        }
      }
    }
  }

  for(int s = 0; s < NUM_SEATS; s++) {
    if(isAlive(s)) {
      int p = pawn[s];
      if(p >= 0 && visited[p])
        foundMask |= (1 << s);
    }
  }
  return foundMask == neededMask;
}

bool Q4Board::isLegalWall(int ax, int ay, bool isHorizontal) const {
  if(wallsLeft[toMove] <= 0)
    return false;
  return isGeometricallyLegalWall(ax, ay, isHorizontal);
}

void Q4Board::recomputeDistancesToCenter() {
  std::fill(std::begin(distToCenter), std::end(distToCenter), 255);
  uint8_t q[NUM_CELLS];
  int head = 0, tail = 0;
  distToCenter[CENTER_CELL] = 0;
  q[tail++] = CENTER_CELL;

  while(head < tail) {
    int curr = q[head++];
    uint8_t d = distToCenter[curr];
    for(int dir = 0; dir < 4; dir++) {
      if(!(blocked[curr] & (1 << dir))) {
        int nxt = curr + DIR_OFFSET[dir];
        if(distToCenter[nxt] == 255) {
          distToCenter[nxt] = d + 1;
          q[tail++] = nxt;
        }
      }
    }
  }

  recomputeCachedPaths();
}

void Q4Board::recomputeCachedPaths() {
  hPathEdges.reset();
  vPathEdges.reset();
  for(int s = 0; s < NUM_SEATS; s++) {
    if(isAlive(s)) {
      int p = pawn[s];
      if(p >= 0 && distToCenter[p] < 255) {
        int curr = p;
        while(curr != CENTER_CELL) {
          bool stepped = false;
          for(int dir = 0; dir < 4; dir++) {
            if(!(blocked[curr] & (1 << dir))) {
              int nxt = curr + DIR_OFFSET[dir];
              if(distToCenter[nxt] == distToCenter[curr] - 1) {
                int cx = curr % BOARD_SIZE;
                int cy = curr / BOARD_SIZE;
                if(dir == DIR_N) {
                  hPathEdges.set(cy * BOARD_SIZE + cx);
                }
                else if(dir == DIR_S) {
                  hPathEdges.set((cy - 1) * BOARD_SIZE + cx);
                }
                else if(dir == DIR_E) {
                  vPathEdges.set(cy * NUM_ANCHORS + cx);
                }
                else if(dir == DIR_W) {
                  vPathEdges.set(cy * NUM_ANCHORS + (cx - 1));
                }
                curr = nxt;
                stepped = true;
                break;
              }
            }
          }
          if(!stepped)
            break;
        }
      }
    }
  }
}

int Q4Board::getPawnMoves(int seat, int* outMoves) const {
  if(!isAlive(seat))
    return 0;
  int me = pawn[seat];
  if(me < 0)
    return 0;

  int ordinary[16];
  int numOrdinary = 0;
  int twoPawn[4];
  int numTwoPawn = 0;

  for(int d = 0; d < 4; d++) {
    if(blocked[me] & (1 << d))
      continue;
    int c1 = me + DIR_OFFSET[d];
    if(occupant[c1] < 0) {
      ordinary[numOrdinary++] = c1;
      continue;
    }

    // c1 is occupied by an alive pawn
    if(!(blocked[c1] & (1 << d))) {
      int c2 = c1 + DIR_OFFSET[d];
      if(occupant[c2] < 0) {
        // Direct jump
        ordinary[numOrdinary++] = c2;
      }
      else {
        // c2 is occupied: diagonal jumps forbidden, check two-pawn jump
        if(!(blocked[c2] & (1 << d))) {
          int c3 = c2 + DIR_OFFSET[d];
          if(occupant[c3] < 0) {
            twoPawn[numTwoPawn++] = c3;
          }
        }
      }
    }
    else {
      // Straight step from c1 is blocked by wall or board edge: diagonal jumps permitted
      int dPerp1 = (d + 1) & 3;
      int dPerp2 = (d + 3) & 3;
      if(!(blocked[c1] & (1 << dPerp1))) {
        int cd1 = c1 + DIR_OFFSET[dPerp1];
        if(occupant[cd1] < 0)
          ordinary[numOrdinary++] = cd1;
      }
      if(!(blocked[c1] & (1 << dPerp2))) {
        int cd2 = c1 + DIR_OFFSET[dPerp2];
        if(occupant[cd2] < 0)
          ordinary[numOrdinary++] = cd2;
      }
    }
  }

  // Distance restriction: two-pawn jump permitted only if every ordinary move strictly increases distance
  bool canTwoPawn = true;
  uint8_t myDist = distToCenter[me];
  for(int i = 0; i < numOrdinary; i++) {
    if(distToCenter[ordinary[i]] <= myDist) {
      canTwoPawn = false;
      break;
    }
  }

  int count = 0;
  bool seen[NUM_CELLS] = {false};
  for(int i = 0; i < numOrdinary; i++) {
    int c = ordinary[i];
    if(!seen[c]) {
      seen[c] = true;
      outMoves[count++] = c;
    }
  }
  if(canTwoPawn) {
    for(int i = 0; i < numTwoPawn; i++) {
      int c = twoPawn[i];
      if(!seen[c]) {
        seen[c] = true;
        outMoves[count++] = c;
      }
    }
  }
  std::sort(outMoves, outMoves + count);
  return count;
}

void Q4Board::getPawnMoves(int seat, std::vector<int>& outMoves) const {
  outMoves.resize(16);
  int n = getPawnMoves(seat, outMoves.data());
  outMoves.resize(n);
}

int Q4Board::getLegalActions(int seat, int* outActions) const {
  if(isFinished() || !isAlive(seat))
    return 0;

  int count = getPawnMoves(seat, outActions);

  if(wallsLeft[seat] > 0) {
    for(int ay = 0; ay < NUM_ANCHORS; ay++) {
      for(int ax = 0; ax < NUM_ANCHORS; ax++) {
        int a = anchorOf(ax, ay);
        if(isGeometricallyLegalWall(ax, ay, false))
          outActions[count++] = actionOfVWall(a);
        if(isGeometricallyLegalWall(ax, ay, true))
          outActions[count++] = actionOfHWall(a);
      }
    }
  }
  return count;
}

void Q4Board::getLegalActions(int seat, std::vector<int>& outActions) const {
  outActions.resize(NUM_ACTIONS);
  int n = getLegalActions(seat, outActions.data());
  outActions.resize(n);
}

bool Q4Board::isLegalAction(int action, int seat) const {
  if(action < 0 || action >= NUM_ACTIONS || !isAlive(seat) || isFinished())
    return false;

  if(isPawnAction(action)) {
    std::vector<int> moves;
    getPawnMoves(seat, moves);
    return std::find(moves.begin(), moves.end(), action) != moves.end();
  }

  if(wallsLeft[seat] <= 0)
    return false;

  if(isVWallAction(action)) {
    int a = action - 121;
    return isGeometricallyLegalWall(anchorX(a), anchorY(a), false);
  }
  if(isHWallAction(action)) {
    int a = action - 221;
    return isGeometricallyLegalWall(anchorX(a), anchorY(a), true);
  }
  return false;
}

Hash128 Q4Board::getHashAfterPawnMove(int destCell) const {
  // Exactly the hash that applyPawnMove(destCell) produces: a move into the center ends the game and the turn does
  // not pass.
  int seat = toMove;
  Hash128 h = hash ^ ZOBRIST_PAWN[seat][pawn[seat]] ^ ZOBRIST_PAWN[seat][destCell] ^ ZOBRIST_TO_MOVE[seat];
  if(destCell != CENTER_CELL)
    h ^= ZOBRIST_TO_MOVE[getNextAlive(toMove)];
  return h;
}

void Q4Board::applyPawnMove(int destCell) {
  int seat = toMove;
  int oldCell = pawn[seat];

  occupant[oldCell] = -1;
  occupant[destCell] = seat;
  pawn[seat] = destCell;

  hash ^= ZOBRIST_PAWN[seat][oldCell];
  hash ^= ZOBRIST_PAWN[seat][destCell];
  hash ^= ZOBRIST_TO_MOVE[toMove];

  if(destCell != CENTER_CELL) {
    toMove = getNextAlive(toMove);
    hash ^= ZOBRIST_TO_MOVE[toMove];
  }

  recomputeCachedPaths();
}

void Q4Board::applyWall(int ax, int ay, bool isHorizontal) {
  int seat = toMove;
  int a = anchorOf(ax, ay);
  int oldWalls = wallsLeft[seat];
  wallsLeft[seat]--;
  int newWalls = wallsLeft[seat];

  if(isHorizontal)
    hWalls.set(a);
  else
    vWalls.set(a);

  addWallToBlocked(ax, ay, isHorizontal);
  recomputeDistancesToCenter();

  hash ^= (isHorizontal ? ZOBRIST_HWALL[a] : ZOBRIST_VWALL[a]);
  hash ^= ZOBRIST_WALLS_LEFT[seat][oldWalls];
  hash ^= ZOBRIST_WALLS_LEFT[seat][newWalls];
  hash ^= ZOBRIST_TO_MOVE[toMove];

  toMove = getNextAlive(toMove);
  hash ^= ZOBRIST_TO_MOVE[toMove];
}

void Q4Board::applyAction(int action) {
  if(isPawnAction(action)) {
    applyPawnMove(action);
  }
  else if(isVWallAction(action)) {
    int a = action - 121;
    applyWall(anchorX(a), anchorY(a), false);
  }
  else if(isHWallAction(action)) {
    int a = action - 221;
    applyWall(anchorX(a), anchorY(a), true);
  }
  else {
    throw StringError("Invalid action in applyAction: " + Global::intToString(action));
  }
}

void Q4Board::eliminateSeat(int seat) {
  if(!isAlive(seat))
    return;

  alive &= ~(1 << seat);
  hash ^= ZOBRIST_ALIVE[seat];

  int oldCell = pawn[seat];
  if(oldCell >= 0) {
    occupant[oldCell] = -1;
    pawn[seat] = -1;
    hash ^= ZOBRIST_PAWN[seat][oldCell];
  }

  int oldWalls = wallsLeft[seat];
  wallsLeft[seat] = 0;
  hash ^= ZOBRIST_WALLS_LEFT[seat][oldWalls];
  hash ^= ZOBRIST_WALLS_LEFT[seat][0];

  if(toMove == seat) {
    int nextSeat = getNextAlive(seat);
    hash ^= ZOBRIST_TO_MOVE[toMove];
    toMove = nextSeat;
    hash ^= ZOBRIST_TO_MOVE[toMove];
  }

  recomputeCachedPaths();
}

bool Q4Board::isFinished() const {
  if(occupant[CENTER_CELL] >= 0)
    return true;
  return getNumAlive() <= 1;
}

int Q4Board::getWinner() const {
  if(occupant[CENTER_CELL] >= 0)
    return occupant[CENTER_CELL];
  if(getNumAlive() == 1) {
    for(int s = 0; s < NUM_SEATS; s++) {
      if(isAlive(s))
        return s;
    }
  }
  return -1;
}

Hash128 Q4Board::getHashFromScratch() const {
  Hash128 h;
  for(int s = 0; s < NUM_SEATS; s++) {
    if(isAlive(s)) {
      h ^= ZOBRIST_ALIVE[s];
      if(pawn[s] >= 0)
        h ^= ZOBRIST_PAWN[s][pawn[s]];
      h ^= ZOBRIST_WALLS_LEFT[s][wallsLeft[s]];
    }
    else {
      h ^= ZOBRIST_WALLS_LEFT[s][0];
    }
  }

  for(int a = 0; a < NUM_WALL_ANCHORS; a++) {
    if(hWalls.test(a))
      h ^= ZOBRIST_HWALL[a];
    if(vWalls.test(a))
      h ^= ZOBRIST_VWALL[a];
  }

  h ^= ZOBRIST_TO_MOVE[toMove];
  return h;
}

bool Q4Board::checkInvariants() const {
  if(hash != getHashFromScratch())
    return false;
  for(int s = 0; s < NUM_SEATS; s++) {
    if(isAlive(s)) {
      int p = pawn[s];
      if(p < 0 || p >= NUM_CELLS) return false;
      if(occupant[p] != s) return false;
      if(wallsLeft[s] < 0 || wallsLeft[s] > 10) return false;
    }
    else {
      if(pawn[s] != -1) return false;
      if(wallsLeft[s] != 0) return false;
    }
  }
  return true;
}

bool Q4Board::toMoveHasLegalAction() const {
  int seat = toMove;
  if(seat < 0 || !isAlive(seat))
    return false;
  int moves[NUM_ACTIONS];
  if(getPawnMoves(seat, moves) > 0)
    return true;
  if(wallsLeft[seat] <= 0)
    return false;
  for(int a = 121; a < NUM_ACTIONS; a++)
    if(isLegalAction(a, seat))
      return true;
  return false;
}

void Q4Board::skipSeatsWithoutLegalAction() {
  if(isFinished())
    return;
  for(int i = 0; i < NUM_SEATS && !toMoveHasLegalAction(); i++) {
    hash ^= ZOBRIST_TO_MOVE[toMove];
    toMove = getNextAlive(toMove);
    hash ^= ZOBRIST_TO_MOVE[toMove];
  }
}
