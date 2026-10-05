#include "q4bots.h"

#include "../core/global.h"

#include <algorithm>
#include <limits>
#include <vector>

namespace Q4Bots {

int arrivalEstimate(const Q4Board& board, int seat) {
  if(!board.isAlive(seat) || board.pawn[seat] < 0)
    return 999999;
  uint8_t d = board.distToCenter[board.pawn[seat]];
  if(d == 255)
    return 999999;
  if(d == 0)
    return 0;

  int nAlive = board.getNumAlive();
  int me = board.toMove;
  int order = 0;
  int curr = me;
  while(order < nAlive) {
    if(curr == seat)
      break;
    curr = board.getNextAlive(curr);
    order++;
  }

  return (int)(d - 1) * nAlive + order + 1;
}

int findRaceLeader(const Q4Board& board) {
  int bestSeat = -1;
  int minEst = 999999;
  for(int s = 0; s < Q4Board::NUM_SEATS; s++) {
    if(board.isAlive(s)) {
      int est = arrivalEstimate(board, s);
      if(est < minEst) {
        minEst = est;
        bestSeat = s;
      }
    }
  }
  return bestSeat;
}

class RandomBot : public Q4Bot {
  Rand rand;
public:
  explicit RandomBot(uint64_t seed) : rand(seed) {}
  int getMove(const Q4Board& board) override {
    std::vector<int> actions;
    board.getLegalActions(board.toMove, actions);
    if(actions.empty())
      return Q4Board::NULL_ACTION;
    size_t idx = rand.nextUInt((uint32_t)actions.size());
    return actions[idx];
  }
  std::string getName() const override { return "random"; }
};

class RandomPawnBot : public Q4Bot {
  Rand rand;
public:
  explicit RandomPawnBot(uint64_t seed) : rand(seed) {}
  int getMove(const Q4Board& board) override {
    std::vector<int> pawnDests;
    board.getPawnMoves(board.toMove, pawnDests);

    // 5% chance of wall if walls left and walls available
    if(board.wallsLeft[board.toMove] > 0 && rand.nextUInt(100) < 5) {
      std::vector<int> walls;
      for(int ay = 0; ay < Q4Board::NUM_ANCHORS; ay++) {
        for(int ax = 0; ax < Q4Board::NUM_ANCHORS; ax++) {
          int a = Q4Board::anchorOf(ax, ay);
          if(board.isLegalWall(ax, ay, false))
            walls.push_back(Q4Board::actionOfVWall(a));
          if(board.isLegalWall(ax, ay, true))
            walls.push_back(Q4Board::actionOfHWall(a));
        }
      }
      if(!walls.empty()) {
        size_t idx = rand.nextUInt((uint32_t)walls.size());
        return walls[idx];
      }
    }

    if(!pawnDests.empty()) {
      size_t idx = rand.nextUInt((uint32_t)pawnDests.size());
      return Q4Board::actionOfPawn(pawnDests[idx]);
    }

    // Fallback: any legal action
    std::vector<int> actions;
    board.getLegalActions(board.toMove, actions);
    if(actions.empty())
      return Q4Board::NULL_ACTION;
    size_t idx = rand.nextUInt((uint32_t)actions.size());
    return actions[idx];
  }
  std::string getName() const override { return "randomPawn"; }
};

class GreedyBot : public Q4Bot {
  Rand rand;
public:
  explicit GreedyBot(uint64_t seed) : rand(seed) {}
  int getMove(const Q4Board& board) override {
    std::vector<int> pawnDests;
    board.getPawnMoves(board.toMove, pawnDests);
    if(pawnDests.empty()) {
      std::vector<int> actions;
      board.getLegalActions(board.toMove, actions);
      if(actions.empty())
        return Q4Board::NULL_ACTION;
      return actions[0];
    }

    uint8_t minD = 255;
    for(int c : pawnDests) {
      if(board.distToCenter[c] < minD)
        minD = board.distToCenter[c];
    }

    std::vector<int> bestMoves;
    for(int c : pawnDests) {
      if(board.distToCenter[c] == minD)
        bestMoves.push_back(c);
    }

    size_t idx = rand.nextUInt((uint32_t)bestMoves.size());
    return Q4Board::actionOfPawn(bestMoves[idx]);
  }
  std::string getName() const override { return "greedy"; }
};

class BasherBot : public Q4Bot {
  Rand rand;
public:
  explicit BasherBot(uint64_t seed) : rand(seed) {}

  int getMove(const Q4Board& board) override {
    int me = board.toMove;
    int leader = findRaceLeader(board);

    if(leader != me && leader >= 0 && board.wallsLeft[me] > 0) {
      int leaderEst = arrivalEstimate(board, leader);
      int myEst = arrivalEstimate(board, me);

      if(leaderEst < myEst) {
        // Try to wall the leader
        uint8_t origLeaderDist = board.distToCenter[board.pawn[leader]];
        uint8_t origMyDist = board.distToCenter[board.pawn[me]];

        int maxLeaderDistInc = 0;
        int minMyDistInc = 999999;
        std::vector<int> bestWalls;

        for(int ay = 0; ay < Q4Board::NUM_ANCHORS; ay++) {
          for(int ax = 0; ax < Q4Board::NUM_ANCHORS; ax++) {
            for(int isH = 0; isH < 2; isH++) {
              bool isHorizontal = (isH == 1);
              if(!board.isLegalWall(ax, ay, isHorizontal))
                continue;

              // Tentative apply wall to a copy or on board
              Q4Board bCopy = board;
              bCopy.applyWall(ax, ay, isHorizontal);

              uint8_t newLeaderDist = bCopy.distToCenter[bCopy.pawn[leader]];
              uint8_t newMyDist = bCopy.distToCenter[bCopy.pawn[me]];

              int leaderInc = (int)newLeaderDist - (int)origLeaderDist;
              int myInc = (int)newMyDist - (int)origMyDist;

              if(leaderInc > maxLeaderDistInc) {
                maxLeaderDistInc = leaderInc;
                minMyDistInc = myInc;
                bestWalls.clear();
                int a = Q4Board::anchorOf(ax, ay);
                bestWalls.push_back(isHorizontal ? Q4Board::actionOfHWall(a) : Q4Board::actionOfVWall(a));
              }
              else if(leaderInc == maxLeaderDistInc && leaderInc > 0) {
                if(myInc < minMyDistInc) {
                  minMyDistInc = myInc;
                  bestWalls.clear();
                  int a = Q4Board::anchorOf(ax, ay);
                  bestWalls.push_back(isHorizontal ? Q4Board::actionOfHWall(a) : Q4Board::actionOfVWall(a));
                }
                else if(myInc == minMyDistInc) {
                  int a = Q4Board::anchorOf(ax, ay);
                  bestWalls.push_back(isHorizontal ? Q4Board::actionOfHWall(a) : Q4Board::actionOfVWall(a));
                }
              }
            }
          }
        }

        if(maxLeaderDistInc >= 1 && !bestWalls.empty()) {
          size_t idx = rand.nextUInt((uint32_t)bestWalls.size());
          return bestWalls[idx];
        }
      }
    }

    // Fall back to greedy
    GreedyBot greedy(rand.nextUInt64());
    return greedy.getMove(board);
  }

  std::string getName() const override { return "basher"; }
};

class GrudgeBot : public Q4Bot {
  Rand rand;
  int targetSeat;
public:
  GrudgeBot(uint64_t seed, int target) : rand(seed), targetSeat(target) {}

  int getMove(const Q4Board& board) override {
    int me = board.toMove;
    int target = targetSeat;
    if(target < 0 || target == me) {
      target = (me + 1) % Q4Board::NUM_SEATS;
    }

    if(board.isAlive(target) && board.pawn[target] >= 0 && board.wallsLeft[me] > 0) {
      uint8_t origTargetDist = board.distToCenter[board.pawn[target]];
      uint8_t origMyDist = board.distToCenter[board.pawn[me]];

      int maxTargetDistInc = 0;
      int minMyDistInc = 999999;
      std::vector<int> bestWalls;

      for(int ay = 0; ay < Q4Board::NUM_ANCHORS; ay++) {
        for(int ax = 0; ax < Q4Board::NUM_ANCHORS; ax++) {
          for(int isH = 0; isH < 2; isH++) {
            bool isHorizontal = (isH == 1);
            if(!board.isLegalWall(ax, ay, isHorizontal))
              continue;

            Q4Board bCopy = board;
            bCopy.applyWall(ax, ay, isHorizontal);

            uint8_t newTargetDist = bCopy.distToCenter[bCopy.pawn[target]];
            uint8_t newMyDist = bCopy.distToCenter[bCopy.pawn[me]];

            int targetInc = (int)newTargetDist - (int)origTargetDist;
            int myInc = (int)newMyDist - (int)origMyDist;

            if(targetInc > maxTargetDistInc) {
              maxTargetDistInc = targetInc;
              minMyDistInc = myInc;
              bestWalls.clear();
              int a = Q4Board::anchorOf(ax, ay);
              bestWalls.push_back(isHorizontal ? Q4Board::actionOfHWall(a) : Q4Board::actionOfVWall(a));
            }
            else if(targetInc == maxTargetDistInc && targetInc > 0) {
              if(myInc < minMyDistInc) {
                minMyDistInc = myInc;
                bestWalls.clear();
                int a = Q4Board::anchorOf(ax, ay);
                bestWalls.push_back(isHorizontal ? Q4Board::actionOfHWall(a) : Q4Board::actionOfVWall(a));
              }
              else if(myInc == minMyDistInc) {
                int a = Q4Board::anchorOf(ax, ay);
                bestWalls.push_back(isHorizontal ? Q4Board::actionOfHWall(a) : Q4Board::actionOfVWall(a));
              }
            }
          }
        }
      }

      if(maxTargetDistInc >= 1 && !bestWalls.empty()) {
        size_t idx = rand.nextUInt((uint32_t)bestWalls.size());
        return bestWalls[idx];
      }
    }

    GreedyBot greedy(rand.nextUInt64());
    return greedy.getMove(board);
  }

  std::string getName() const override { return "grudge"; }
};

std::unique_ptr<Q4Bot> makeBot(const std::string& type, uint64_t seed, int targetSeat) {
  std::string t = Global::toLower(Global::trim(type));
  if(t == "random")
    return std::make_unique<RandomBot>(seed);
  if(t == "randompawn" || t == "random_pawn")
    return std::make_unique<RandomPawnBot>(seed);
  if(t == "greedy")
    return std::make_unique<GreedyBot>(seed);
  if(t == "basher")
    return std::make_unique<BasherBot>(seed);
  if(t == "grudge")
    return std::make_unique<GrudgeBot>(seed, targetSeat);
  throw StringError("Unknown Q4 bot type: " + type);
}

}  // namespace Q4Bots
