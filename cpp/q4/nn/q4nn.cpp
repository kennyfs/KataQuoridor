#include "q4nn.h"
#include "../q4notation.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace Q4NN {

namespace {
  // Spatial channel numbers (docs/q4/Q4IO.md §3)
  constexpr int CH_ON_BOARD = 0;
  constexpr int CH_PAWN = 1;           // 1..4
  constexpr int CH_GOAL = 5;
  constexpr int CH_BLOCKED = 6;        // 6..9: N E S W
  constexpr int CH_DIST_CENTER = 10;
  constexpr int CH_DIST_PAWN = 11;     // 11..14
  constexpr int CH_ON_PATH = 15;       // 15..18
  constexpr int CH_VWALL = 19;
  constexpr int CH_HWALL = 20;
  constexpr int CH_ANCHOR_DOMAIN = 21;
  constexpr int CH_LEGAL_VWALL = 22;
  constexpr int CH_LEGAL_HWALL = 23;
  constexpr int CH_LEGAL_PAWN = 24;
  constexpr int CH_REPEAT_PAWN = 25;
  constexpr int CH_DRAW_PAWN = 26;

  // Global feature indices
  constexpr int G_WALLS = 0;           // 0..3
  constexpr int G_HAS_WALL = 4;        // 4..7
  constexpr int G_ALIVE = 8;           // 8..11
  constexpr int G_DIST = 12;           // 12..15
  constexpr int G_ARRIVAL = 16;        // 16..19
  constexpr int G_LEADER = 20;         // 20..23
  constexpr int G_NUM_ALIVE = 24;
  constexpr int G_PLIES_LEFT = 25;
  constexpr int G_REP_ON = 26;
  constexpr int G_REP_PROGRESS = 27;

  constexpr float MAX_WALLS_NORM = 7.0f;
  constexpr float PLIES_NORM = 400.0f;
  constexpr float ARRIVAL_NORM = 64.0f;

  inline int idx(int ch, int c, bool nhwc) {
    return nhwc ? c * NUM_SPATIAL_CHANNELS + ch : ch * POS_AREA + c;
  }

  // Number of earlier occurrences (since the last wall or elimination) of the position after the pawn move.
  int occurrencesAfterPawnMove(const Q4Board& board, const Q4History& history, int dest) {
    Hash128 nextHash = board.getHashAfterPawnMove(dest);
    int occ = 0;
    for(const Hash128& h : history.repetitionHashes)
      if(h == nextHash)
        occ++;
    return occ;
  }
}

void computeRawDistances(const Q4Board& board, RawDistances& out) {
  std::copy(board.distToCenter, board.distToCenter + POS_AREA, out.d[0]);
  for(int k = 0; k < 4; k++) {
    uint8_t* dist = out.d[1 + k];
    std::fill(dist, dist + POS_AREA, (uint8_t)255);
    int s = (board.toMove + k) % 4;
    if(!board.isAlive(s))
      continue;
    int queue[POS_AREA];
    int head = 0, tail = 0;
    int start = board.pawn[s];
    dist[start] = 0;
    queue[tail++] = start;
    while(head < tail) {
      int cur = queue[head++];
      for(int dir = 0; dir < 4; dir++) {
        if(board.blocked[cur] & (1 << dir))
          continue;
        int nxt = cur + Q4Board::DIR_OFFSET[dir];
        if(dist[nxt] == 255) {
          dist[nxt] = (uint8_t)(dist[cur] + 1);
          queue[tail++] = nxt;
        }
      }
    }
  }
}

void fillRow(
  const Q4Board& board,
  const Q4History& history,
  bool inputsUseNHWC,
  float* rowSpatial,
  float* rowGlobal,
  RawDistances* rawDistOut
) {
  Q4Symmetry::init();
  const int toMove = board.toMove;

  // Built in NCHW, converted at the end if needed.
  float sp[NUM_SPATIAL_CHANNELS * POS_AREA];
  std::fill(sp, sp + NUM_SPATIAL_CHANNELS * POS_AREA, 0.0f);
  std::fill(rowGlobal, rowGlobal + NUM_GLOBAL_FEATURES, 0.0f);
  auto S = [&](int ch, int c) -> float& { return sp[ch * POS_AREA + c]; };

  for(int c = 0; c < POS_AREA; c++)
    S(CH_ON_BOARD, c) = 1.0f;
  for(int k = 0; k < 4; k++) {
    int s = (toMove + k) % 4;
    if(board.isAlive(s))
      S(CH_PAWN + k, board.pawn[s]) = 1.0f;
  }
  S(CH_GOAL, Q4Board::CENTER_CELL) = 1.0f;

  for(int c = 0; c < POS_AREA; c++)
    for(int d = 0; d < 4; d++)
      if(board.blocked[c] & (1 << d))
        S(CH_BLOCKED + d, c) = 1.0f;

  RawDistances localDist;
  RawDistances& rd = rawDistOut != nullptr ? *rawDistOut : localDist;
  computeRawDistances(board, rd);
  for(int c = 0; c < POS_AREA; c++) {
    S(CH_DIST_CENTER, c) = dist01(rd.d[0][c]);
    for(int k = 0; k < 4; k++)
      S(CH_DIST_PAWN + k, c) = dist01(rd.d[1 + k][c]);
  }

  // Cells on some shortest path to the center of each alive seat.
  for(int k = 0; k < 4; k++) {
    int s = (toMove + k) % 4;
    if(!board.isAlive(s))
      continue;
    uint8_t pawnToCenter = board.distToCenter[board.pawn[s]];
    if(pawnToCenter == 255)
      continue;
    for(int c = 0; c < POS_AREA; c++) {
      uint8_t df = rd.d[1 + k][c];
      uint8_t dc = board.distToCenter[c];
      if(df != 255 && dc != 255 && (int)df + (int)dc == (int)pawnToCenter)
        S(CH_ON_PATH + k, c) = 1.0f;
    }
  }

  for(int ay = 0; ay < Q4Board::NUM_ANCHORS; ay++) {
    for(int ax = 0; ax < Q4Board::NUM_ANCHORS; ax++) {
      int a = Q4Board::anchorOf(ax, ay);
      int c = ay * POS_LEN + ax;
      if(board.vWalls.test(a)) S(CH_VWALL, c) = 1.0f;
      if(board.hWalls.test(a)) S(CH_HWALL, c) = 1.0f;
      S(CH_ANCHOR_DOMAIN, c) = 1.0f;
      if(board.isGeometricallyLegalWall(ax, ay, false)) S(CH_LEGAL_VWALL, c) = 1.0f;
      if(board.isGeometricallyLegalWall(ax, ay, true)) S(CH_LEGAL_HWALL, c) = 1.0f;
    }
  }

  const int repN = history.rules.repetitionDrawCount;
  const bool repOn = repN >= 2;
  std::vector<int> pawnDests;
  board.getPawnMoves(toMove, pawnDests);
  for(int c : pawnDests) {
    S(CH_LEGAL_PAWN, c) = 1.0f;
    if(repOn) {
      int occ = occurrencesAfterPawnMove(board, history, c);
      if(occ > 0) S(CH_REPEAT_PAWN, c) = 1.0f;
      if(occ + 1 >= repN) S(CH_DRAW_PAWN, c) = 1.0f;
    }
  }

  if(inputsUseNHWC) {
    for(int ch = 0; ch < NUM_SPATIAL_CHANNELS; ch++)
      for(int c = 0; c < POS_AREA; c++)
        rowSpatial[idx(ch, c, true)] = sp[ch * POS_AREA + c];
  }
  else {
    std::copy(sp, sp + NUM_SPATIAL_CHANNELS * POS_AREA, rowSpatial);
  }

  // Global features
  const int nAlive = board.getNumAlive();
  int order[4] = {-1, -1, -1, -1};  // position of relative slot k in the upcoming alive turn order
  int next = 0;
  for(int k = 0; k < 4; k++)
    if(board.isAlive((toMove + k) % 4))
      order[k] = next++;

  int leader = -1;
  float leaderEstimate = 0.0f;
  for(int k = 0; k < 4; k++) {
    int s = (toMove + k) % 4;
    rowGlobal[G_WALLS + k] = (float)board.wallsLeft[s] / MAX_WALLS_NORM;
    rowGlobal[G_HAS_WALL + k] = board.wallsLeft[s] > 0 ? 1.0f : 0.0f;
    rowGlobal[G_ALIVE + k] = board.isAlive(s) ? 1.0f : 0.0f;
    if(!board.isAlive(s))
      continue;  // distance, arrival: 0
    uint8_t d = board.distToCenter[board.pawn[s]];
    rowGlobal[G_DIST + k] = dist01(d);
    if(d == 255) {
      rowGlobal[G_ARRIVAL + k] = 1.0f;
    }
    else {
      float estimate = ((float)(d - 1) * nAlive + order[k] + 1) / ARRIVAL_NORM;
      rowGlobal[G_ARRIVAL + k] = estimate;
      if(leader < 0 || estimate < leaderEstimate) {
        leader = k;
        leaderEstimate = estimate;
      }
    }
  }
  if(leader >= 0)
    rowGlobal[G_LEADER + leader] = 1.0f;

  rowGlobal[G_NUM_ALIVE] = (float)nAlive / 4.0f;
  rowGlobal[G_PLIES_LEFT] = (float)std::max(0, history.rules.maxPlies - history.plies) / PLIES_NORM;
  rowGlobal[G_REP_ON] = repOn ? 1.0f : 0.0f;
  if(repOn) {
    int count = history.currentPositionRepetitionCount();
    rowGlobal[G_REP_PROGRESS] =
      repN == 2 ? 1.0f : std::min(1.0f, (float)std::max(0, count - 1) / (float)(repN - 2));
  }
}

void applyInputSymmetry(const float* src, float* dst, int sym, bool nhwc) {
  Q4Symmetry::init();
  if(sym == 0) {
    if(src != dst)
      std::copy(src, src + NUM_SPATIAL_CHANNELS * POS_AREA, dst);
    return;
  }
  std::fill(dst, dst + NUM_SPATIAL_CHANNELS * POS_AREA, 0.0f);

  // Cell channels: plain cell map.
  static const int cellChannels[] = {0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 25, 26};
  for(int c = 0; c < POS_AREA; c++) {
    int nc = Q4Symmetry::applyCell(c, sym);
    for(int ch : cellChannels)
      dst[idx(ch, nc, nhwc)] = src[idx(ch, c, nhwc)];
    // Blocked-direction channels are permuted along with the cell.
    for(int d = 0; d < 4; d++)
      dst[idx(CH_BLOCKED + Q4Symmetry::applyDirection(d, sym), nc, nhwc)] = src[idx(CH_BLOCKED + d, c, nhwc)];
  }

  // Anchor channels: the anchor map, and V <-> H when the symmetry exchanges the axes.
  for(int ay = 0; ay < Q4Board::NUM_ANCHORS; ay++) {
    for(int ax = 0; ax < Q4Board::NUM_ANCHORS; ax++) {
      int c = ay * POS_LEN + ax;
      for(int orient = 0; orient < 2; orient++) {  // 0 = vertical, 1 = horizontal
        int nax, nay;
        bool nIsH;
        Q4Symmetry::applyAnchor(ax, ay, orient == 1, sym, nax, nay, nIsH);
        int nc = nay * POS_LEN + nax;
        dst[idx(CH_VWALL + (nIsH ? 1 : 0), nc, nhwc)] = src[idx(CH_VWALL + orient, c, nhwc)];
        dst[idx(CH_LEGAL_VWALL + (nIsH ? 1 : 0), nc, nhwc)] = src[idx(CH_LEGAL_VWALL + orient, c, nhwc)];
      }
      dst[idx(CH_ANCHOR_DOMAIN, c, nhwc)] = src[idx(CH_ANCHOR_DOMAIN, c, nhwc)];
    }
  }
}

void mapPolicyToGame(const float* raw, int sym, float* out) {
  Q4Symmetry::init();
  for(int variant = 0; variant < NUM_POLICY_VARIANTS; variant++) {
    const float* rawVariant = raw + variant * POLICY_SLOTS_PER_VARIANT;
    float* outVariant = out + variant * NUM_ACTIONS;
    for(int act = 0; act < NUM_ACTIONS; act++) {
      int a = Q4Symmetry::applyAction(act, sym);
      int plane, cell;
      if(Q4Board::isPawnAction(a)) {
        plane = 0;
        cell = a;
      }
      else if(Q4Board::isVWallAction(a)) {
        plane = 1;
        int anchor = a - 121;
        cell = Q4Board::anchorY(anchor) * POS_LEN + Q4Board::anchorX(anchor);
      }
      else {
        plane = 2;
        int anchor = a - 221;
        cell = Q4Board::anchorY(anchor) * POS_LEN + Q4Board::anchorX(anchor);
      }
      outVariant[act] = rawVariant[plane * POS_AREA + cell];
    }
  }
}

void softmaxLegal(const float* logits, const std::vector<int>& legal, float* out, float temperature) {
  std::fill(out, out + NUM_ACTIONS, 0.0f);
  if(legal.empty())
    return;
  float maxLogit = logits[legal[0]];
  for(int a : legal)
    maxLogit = std::max(maxLogit, logits[a]);
  const float invTemp = 1.0f / std::max(1e-4f, temperature);
  double sum = 0.0;
  for(int a : legal) {
    float p = std::exp((logits[a] - maxLogit) * invTemp);
    out[a] = p;
    sum += p;
  }
  for(int a : legal)
    out[a] = (float)(out[a] / sum);
}

void softmaxValue(const float* logits, float* out) {
  float maxLogit = *std::max_element(logits, logits + NUM_VALUE_LOGITS);
  double sum = 0.0;
  for(int i = 0; i < NUM_VALUE_LOGITS; i++) {
    out[i] = std::exp(logits[i] - maxLogit);
    sum += out[i];
  }
  for(int i = 0; i < NUM_VALUE_LOGITS; i++)
    out[i] = (float)(out[i] / sum);
}

void rotateValueToAbsolute(const float* rel, int toMove, float* abs) {
  for(int s = 0; s < 4; s++)
    abs[s] = rel[(s - toMove + 4) % 4];
  abs[4] = rel[4];
}

void decodeTrajectory(const float* raw, int sym, float* out) {
  Q4Symmetry::init();
  for(int c = 0; c < POS_AREA; c++)
    out[c] = 1.0f / (1.0f + std::exp(-raw[Q4Symmetry::applyCell(c, sym)]));
}

Hash128 getCacheHash(const Q4Board& board, const Q4History& history, int sym) {
  auto mix = [](uint64_t tag, uint64_t v) {
    uint64_t a = Hash::splitMix64(tag ^ v);
    return Hash128(a, Hash::nasam(a ^ tag));
  };
  Hash128 h = board.hash;
  h ^= mix(0x5132526c73ULL, (uint64_t)(uint32_t)history.rules.maxPlies | ((uint64_t)(uint32_t)history.rules.repetitionDrawCount << 32));
  h ^= mix(0x506c696573ULL, (uint64_t)std::max(0, history.rules.maxPlies - history.plies));
  h ^= mix(0x53796d6dULL, (uint64_t)sym);

  if(history.rules.repetitionDrawCount >= 2) {
    uint64_t rep = Hash::splitMix64(0x5245504554ULL ^ (uint64_t)history.currentPositionRepetitionCount());
    std::vector<int> dests;
    board.getPawnMoves(board.toMove, dests);
    for(int c : dests) {
      int occ = occurrencesAfterPawnMove(board, history, c);
      if(occ > 0)
        rep = Hash::splitMix64(rep ^ ((uint64_t)c << 16) ^ (uint64_t)occ);
    }
    h ^= mix(0x5265704861ULL, rep);
  }
  return h;
}

void evaluate(NNEvaluator& nnEval, NNResultBuf& buf, const Q4History& history, int sym, bool skipCache, Eval& out) {
  const Q4Board& board = history.currentBoard;
  const bool nhwc = nnEval.getInputsUseNHWC();
  std::vector<float> spatial(NUM_SPATIAL_CHANNELS * POS_AREA), symSpatial(NUM_SPATIAL_CHANNELS * POS_AREA);
  std::vector<float> global(NUM_GLOBAL_FEATURES);
  fillRow(board, history, nhwc, spatial.data(), global.data());
  applyInputSymmetry(spatial.data(), symSpatial.data(), sym, nhwc);
  nnEval.evaluateQ4Raw(symSpatial.data(), global.data(), getCacheHash(board, history, sym), buf, skipCache);
  const Q4RawNNOutput* raw = buf.result->q4Raw.get();
  if(raw == nullptr)
    throw StringError("Q4NN::evaluate: the evaluator returned no Q4 raw output (not a Q4 model?)");

  out.toMove = board.toMove;
  out.sym = sym;
  mapPolicyToGame(raw->policyLogits, sym, &out.policyLogits[0][0]);
  std::copy(raw->valueLogits, raw->valueLogits + NUM_VALUE_LOGITS, out.valueLogits);
  softmaxValue(raw->valueLogits, out.valueRel);
  rotateValueToAbsolute(out.valueRel, out.toMove, out.valueAbs);
  std::copy(raw->miscValues, raw->miscValues + NUM_MISC, out.misc);
  decodeTrajectory(raw->trajectoryLogits, sym, out.trajectory);
}

const char* seatName(int seat) {
  static const char* names[4] = {"South", "West", "North", "East"};
  return names[seat & 3];
}

std::string formatEval(const Eval& e, const std::vector<int>& legalActions, int numTopMoves) {
  static const char* relNames[4] = {"me", "next", "across", "previous"};
  std::string out = Global::strprintf("Symmetry %d, seat %d (%s) to move\n", e.sym, e.toMove + 1, seatName(e.toMove));
  out += "Win probability by absolute seat:\n";
  for(int s = 0; s < 4; s++)
    out += Global::strprintf("  seat %d (%-5s): %6.2f%%\n", s + 1, seatName(s), e.valueAbs[s] * 100.0);
  out += Global::strprintf("  draw          : %6.2f%%\n", e.valueAbs[4] * 100.0);
  out += Global::strprintf("Plies to the end of the game: %.1f\n", e.misc[0] * 100.0);
  out += "Walls-only distance to the center at the end of the game:\n";
  for(int k = 0; k < 4; k++)
    out += Global::strprintf("  %-8s (seat %d): %.2f\n", relNames[k], (e.toMove + k) % 4 + 1, e.misc[1 + k] * 32.0);
  out += Global::strprintf("Short-term value error: %.4f\n", e.misc[5]);

  std::vector<float> probs(NUM_ACTIONS);
  softmaxLegal(&e.policyLogits[0][0], legalActions, probs.data(), 1.0f);
  std::vector<std::pair<float, int>> ranked;
  for(int a : legalActions)
    ranked.push_back({probs[a], a});
  std::sort(ranked.begin(), ranked.end(), [](const std::pair<float, int>& x, const std::pair<float, int>& y) {
    return x.first != y.first ? x.first > y.first : x.second < y.second;
  });
  out += Global::strprintf("Top legal moves of the search policy (%d legal):\n", (int)legalActions.size());
  for(int i = 0; i < (int)ranked.size() && i < numTopMoves; i++)
    out += Global::strprintf("  %-5s %6.2f%%\n", Q4Notation::actionToString(ranked[i].second).c_str(), ranked[i].first * 100.0);
  return out;
}

} // namespace Q4NN
