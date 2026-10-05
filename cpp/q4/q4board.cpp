#include "q4board.h"

#include "../core/rand.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

const int Q4Board::DIR_OFFSET[4] = {+11, +1, -11, -1};
const int Q4Board::DIR_OPPOSITE[4] = {2, 3, 0, 1};
const int Q4Board::START_CELLS[4] = {5, 55, 115, 65};

bool Q4Board::IS_ZOBRIST_INITIALIZED = false;
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
  }
}

void Q4Board::addWallToBlocked(int ax, int ay, bool isHorizontal) {
  if(isHorizontal) {
    int c0 = ay * BOARD_SIZE + ax;
    int c1 = c0 + 1;
    blocked[c0] |= (1 << DIR_N);
    blocked[c0 + DIR_OFFSET[DIR_N]] |= (1 << DIR_S);
    blocked[c1] |= (1 << DIR_N);
    blocked[c1 + DIR_OFFSET[DIR_N]] |= (1 << DIR_S);
  }
  else {
    int c0 = ay * BOARD_SIZE + ax;
    int c1 = c0 + DIR_OFFSET[DIR_N];
    blocked[c0] |= (1 << DIR_E);
    blocked[c0 + DIR_OFFSET[DIR_E]] |= (1 << DIR_W);
    blocked[c1] |= (1 << DIR_E);
    blocked[c1 + DIR_OFFSET[DIR_E]] |= (1 << DIR_W);
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
  }
  else {
    int c0 = ay * BOARD_SIZE + ax;
    int c1 = c0 + DIR_OFFSET[DIR_N];
    blocked[c0] &= ~(1 << DIR_E);
    blocked[c0 + DIR_OFFSET[DIR_E]] &= ~(1 << DIR_W);
    blocked[c1] &= ~(1 << DIR_E);
    blocked[c1 + DIR_OFFSET[DIR_E]] &= ~(1 << DIR_W);
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
  return !wallConflicts(ax, ay, isHorizontal);
}

bool Q4Board::isLegalWallBruteForce(int ax, int ay, bool isHorizontal) const {
  if(wallConflicts(ax, ay, isHorizontal))
    return false;

  // Tentatively place wall on grid
  const_cast<Q4Board*>(this)->addWallToBlocked(ax, ay, isHorizontal);

  // BFS from CENTER_CELL
  uint8_t q[NUM_CELLS];
  bool visited[NUM_CELLS] = {false};
  int head = 0, tail = 0;
  q[tail++] = CENTER_CELL;
  visited[CENTER_CELL] = true;

  while(head < tail) {
    int curr = q[head++];
    for(int dir = 0; dir < 4; dir++) {
      if(!(blocked[curr] & (1 << dir))) {
        int nxt = curr + DIR_OFFSET[dir];
        if(!visited[nxt]) {
          visited[nxt] = true;
          q[tail++] = nxt;
        }
      }
    }
  }

  bool ok = true;
  for(int s = 0; s < NUM_SEATS; s++) {
    if(isAlive(s)) {
      int p = pawn[s];
      if(p >= 0 && !visited[p]) {
        ok = false;
        break;
      }
    }
  }

  const_cast<Q4Board*>(this)->removeWallFromBlocked(ax, ay, isHorizontal);
  return ok;
}

bool Q4Board::isLegalWall(int ax, int ay, bool isHorizontal) const {
  if(wallsLeft[toMove] <= 0)
    return false;
  return isLegalWallBruteForce(ax, ay, isHorizontal);
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
}

void Q4Board::getPawnMoves(int seat, std::vector<int>& outMoves) const {
  outMoves.clear();
  if(!isAlive(seat))
    return;
  int me = pawn[seat];
  if(me < 0)
    return;

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

  bool seen[NUM_CELLS] = {false};
  for(int i = 0; i < numOrdinary; i++) {
    int c = ordinary[i];
    if(!seen[c]) {
      seen[c] = true;
      outMoves.push_back(c);
    }
  }
  if(canTwoPawn) {
    for(int i = 0; i < numTwoPawn; i++) {
      int c = twoPawn[i];
      if(!seen[c]) {
        seen[c] = true;
        outMoves.push_back(c);
      }
    }
  }
  std::sort(outMoves.begin(), outMoves.end());
}

void Q4Board::getLegalActions(int seat, std::vector<int>& outActions) const {
  outActions.clear();
  if(isFinished() || !isAlive(seat))
    return;

  std::vector<int> pawnDests;
  getPawnMoves(seat, pawnDests);
  for(int c : pawnDests)
    outActions.push_back(actionOfPawn(c));

  if(wallsLeft[seat] > 0) {
    for(int ay = 0; ay < NUM_ANCHORS; ay++) {
      for(int ax = 0; ax < NUM_ANCHORS; ax++) {
        int a = anchorOf(ax, ay);
        if(isLegalWallBruteForce(ax, ay, false))
          outActions.push_back(actionOfVWall(a));
        if(isLegalWallBruteForce(ax, ay, true))
          outActions.push_back(actionOfHWall(a));
      }
    }
  }
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
    return isLegalWallBruteForce(anchorX(a), anchorY(a), false);
  }
  if(isHWallAction(action)) {
    int a = action - 221;
    return isLegalWallBruteForce(anchorX(a), anchorY(a), true);
  }
  return false;
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
