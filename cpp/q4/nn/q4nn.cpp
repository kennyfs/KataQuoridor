#include "q4nn.h"
#include "q4rawsymmetry.h"
#include "../q4notation.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace Q4NN {

namespace {
  // Feature plane indices in the spatial tensor [27, 11, 11]
  static constexpr int CH_ON_BOARD = 0;
  static constexpr int CH_PAWN = 1;          // 1..4: me, next, across, previous
  static constexpr int CH_GOAL = 5;
  static constexpr int CH_BLOCKED = 6;       // 6..9: N, E, S, W
  static constexpr int CH_DIST_CENTER = 10;
  static constexpr int CH_DIST_PAWN = 11;    // 11..14: me, next, across, previous
  static constexpr int CH_ON_PATH = 15;      // 15..18: me, next, across, previous
  static constexpr int CH_VWALL = 19;
  static constexpr int CH_HWALL = 20;
  static constexpr int CH_ANCHOR_DOMAIN = 21;
  static constexpr int CH_LEGAL_VWALL = 22;
  static constexpr int CH_LEGAL_HWALL = 23;
  static constexpr int CH_LEGAL_PAWN = 24;
  static constexpr int CH_REPEATING_PAWN = 25;
  static constexpr int CH_DRAWING_PAWN = 26;

  // Global feature indices in the global tensor [28]
  static constexpr int G_WALLS = 0;          // 0..3: me, next, across, previous
  static constexpr int G_HAS_WALL = 4;       // 4..7: me, next, across, previous
  static constexpr int G_ALIVE = 8;          // 8..11: me, next, across, previous
  static constexpr int G_DIST = 12;          // 12..15: me, next, across, previous
  static constexpr int G_ARRIVAL = 16;       // 16..19: me, next, across, previous
  static constexpr int G_LEADER = 20;        // 20..23: me, next, across, previous
  static constexpr int G_NUM_ALIVE = 24;
  static constexpr int G_PLIES_LEFT = 25;
  static constexpr int G_REP_ON = 26;
  static constexpr int G_REP_PROGRESS = 27;

  static constexpr float MAX_WALLS_NORM = 7.0f;
  static constexpr float ARRIVAL_NORM = 64.0f;
  static constexpr float PLIES_NORM = 400.0f;

  inline size_t idx(int channel, int cell, bool nhwc) {
    return nhwc ? (size_t)cell * NUM_SPATIAL_CHANNELS + channel
                : (size_t)channel * POS_AREA + cell;
  }

  int occurrencesAfterPawnMove(const Q4PlayState& state, int dest) {
    Hash128 nextHash = state.board.getHashAfterPawnMove(dest);
    int occ = 0;
    for(const Hash128& h : state.repetitionHashes)
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
  const Q4PlayState& state,
  bool inputsUseNHWC,
  float* rowSpatial,
  float* rowGlobal,
  RawDistances* rawDistOut,
  int* outLegalActions,
  int* outNumLegalActions
) {
  const Q4Board& board = state.board;
  const int toMove = board.toMove;
  static thread_local float nchwSpatialBuf[NUM_SPATIAL_CHANNELS * POS_AREA];
  float* sp = rowSpatial;
  if(inputsUseNHWC) {
    sp = nchwSpatialBuf;
  }
  std::fill_n(sp, NUM_SPATIAL_CHANNELS * POS_AREA, 0.0f);
  std::fill_n(rowGlobal, (size_t)NUM_GLOBAL_FEATURES, 0.0f);
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

  const int repN = state.rules.repetitionDrawCount;
  const bool repOn = repN >= 2;
  int pawnDests[16];
  int numPawnDests = board.getPawnMoves(toMove, pawnDests);
  for(int i = 0; i < numPawnDests; i++) {
    int c = pawnDests[i];
    S(CH_LEGAL_PAWN, c) = 1.0f;
    if(repOn) {
      int occ = occurrencesAfterPawnMove(state, c);
      if(occ > 0)
        S(CH_REPEATING_PAWN, c) = 1.0f;
      if(occ + 1 >= repN)
        S(CH_DRAWING_PAWN, c) = 1.0f;
    }
  }

  int legalActionCount = 0;
  if(outLegalActions != nullptr) {
    for(int i = 0; i < numPawnDests; i++)
      outLegalActions[legalActionCount++] = pawnDests[i];
  }

  bool canPlaceWalls = board.wallsLeft[toMove] > 0;
  for(int ay = 0; ay < Q4Board::NUM_ANCHORS; ay++) {
    for(int ax = 0; ax < Q4Board::NUM_ANCHORS; ax++) {
      int a = Q4Board::anchorOf(ax, ay);
      int c = ay * POS_LEN + ax;
      if(board.vWalls.test(a)) S(CH_VWALL, c) = 1.0f;
      if(board.hWalls.test(a)) S(CH_HWALL, c) = 1.0f;
      S(CH_ANCHOR_DOMAIN, c) = 1.0f;
      bool vLegal = board.isGeometricallyLegalWall(ax, ay, false);
      if(vLegal) {
        S(CH_LEGAL_VWALL, c) = 1.0f;
        if(outLegalActions != nullptr && canPlaceWalls)
          outLegalActions[legalActionCount++] = Q4Board::actionOfVWall(a);
      }
      bool hLegal = board.isGeometricallyLegalWall(ax, ay, true);
      if(hLegal) {
        S(CH_LEGAL_HWALL, c) = 1.0f;
        if(outLegalActions != nullptr && canPlaceWalls)
          outLegalActions[legalActionCount++] = Q4Board::actionOfHWall(a);
      }
    }
  }
  if(outNumLegalActions != nullptr)
    *outNumLegalActions = legalActionCount;

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
      float unclampedEstimate = ((float)(d - 1) * nAlive + order[k] + 1) / ARRIVAL_NORM;
      float estimate = std::min(1.0f, unclampedEstimate); // A2: arrival estimate clamp
      rowGlobal[G_ARRIVAL + k] = estimate;
      if(leader < 0 || unclampedEstimate < leaderEstimate) {
        leader = k;
        leaderEstimate = unclampedEstimate;
      }
    }
  }
  if(leader >= 0)
    rowGlobal[G_LEADER + leader] = 1.0f;

  rowGlobal[G_NUM_ALIVE] = (float)nAlive / 4.0f;
  rowGlobal[G_PLIES_LEFT] = (float)std::max(0, state.rules.maxPlies - state.plies) / PLIES_NORM;
  rowGlobal[G_REP_ON] = repOn ? 1.0f : 0.0f;
  if(repOn) {
    int count = state.currentPositionRepetitionCount();
    rowGlobal[G_REP_PROGRESS] = repN == 2 ? 1.0f : std::min(1.0f, (float)std::max(0, count - 1) / (float)(repN - 2));
  }

  if(inputsUseNHWC) {
    for(int c = 0; c < POS_AREA; c++)
      for(int ch = 0; ch < NUM_SPATIAL_CHANNELS; ch++)
        rowSpatial[(size_t)c * NUM_SPATIAL_CHANNELS + ch] = sp[(size_t)ch * POS_AREA + c];
  }
}

void fillRow(
  const Q4Board& board,
  const Q4History& history,
  bool inputsUseNHWC,
  float* rowSpatial,
  float* rowGlobal,
  RawDistances* rawDistOut,
  int* outLegalActions,
  int* outNumLegalActions
) {
  Q4PlayState s = history.state;
  s.board = board;
  s.rules = history.rules;
  s.plies = history.plies;
  s.isFinished = history.isFinished;
  s.winnerSeat = history.winnerSeat;
  s.isDraw = history.isDraw;
  s.repetitionHashes = history.repetitionHashes;
  fillRow(s, inputsUseNHWC, rowSpatial, rowGlobal, rawDistOut, outLegalActions, outNumLegalActions);
}

static int inputSymMap[8][2][NUM_SPATIAL_CHANNELS * POS_AREA];
static int16_t policySlotMap[8][NUM_ACTIONS];
static std::once_flag precomputedTablesFlag;

static void initPrecomputedTables() {
  Q4Symmetry::init();
  Q4RawSymmetry::init();

  for(int sym = 0; sym < 8; sym++) {
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
      policySlotMap[sym][act] = (int16_t)(plane * POS_AREA + cell);
    }
  }

  static const int cellChannels[] = {0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 25, 26};
  for(int sym = 0; sym < 8; sym++) {
    for(int nhwc = 0; nhwc < 2; nhwc++) {
      int* map = inputSymMap[sym][nhwc];
      std::fill_n(map, NUM_SPATIAL_CHANNELS * POS_AREA, -1);

      if(sym == 0) {
        for(int i = 0; i < NUM_SPATIAL_CHANNELS * POS_AREA; i++)
          map[i] = i;
        continue;
      }

      for(int c = 0; c < POS_AREA; c++) {
        int nc = Q4Symmetry::applyCell(c, sym);
        for(int ch : cellChannels) {
          int dstI = idx(ch, nc, nhwc != 0);
          int srcI = idx(ch, c, nhwc != 0);
          map[dstI] = srcI;
        }
        for(int d = 0; d < 4; d++) {
          int dstI = idx(CH_BLOCKED + Q4Symmetry::applyDirection(d, sym), nc, nhwc != 0);
          int srcI = idx(CH_BLOCKED + d, c, nhwc != 0);
          map[dstI] = srcI;
        }
      }

      for(int ay = 0; ay < Q4Board::NUM_ANCHORS; ay++) {
        for(int ax = 0; ax < Q4Board::NUM_ANCHORS; ax++) {
          int c = ay * POS_LEN + ax;
          for(int orient = 0; orient < 2; orient++) {
            int nax, nay;
            bool nIsH;
            Q4Symmetry::applyAnchor(ax, ay, orient == 1, sym, nax, nay, nIsH);
            int nc = nay * POS_LEN + nax;
            int dstV = idx(CH_VWALL + (nIsH ? 1 : 0), nc, nhwc != 0);
            int srcV = idx(CH_VWALL + orient, c, nhwc != 0);
            map[dstV] = srcV;
            int dstL = idx(CH_LEGAL_VWALL + (nIsH ? 1 : 0), nc, nhwc != 0);
            int srcL = idx(CH_LEGAL_VWALL + orient, c, nhwc != 0);
            map[dstL] = srcL;
          }
          int dstA = idx(CH_ANCHOR_DOMAIN, c, nhwc != 0);
          int srcA = idx(CH_ANCHOR_DOMAIN, c, nhwc != 0);
          map[dstA] = srcA;
        }
      }
    }
  }
}

void applyInputSymmetry(const float* src, float* dst, int sym, bool nhwc) {
  std::call_once(precomputedTablesFlag, initPrecomputedTables);
  if(sym == 0) {
    if(src != dst)
      std::copy(src, src + NUM_SPATIAL_CHANNELS * POS_AREA, dst);
    return;
  }
  std::fill_n(dst, NUM_SPATIAL_CHANNELS * POS_AREA, 0.0f);
  const int* map = inputSymMap[sym][nhwc ? 1 : 0];
  for(int i = 0; i < NUM_SPATIAL_CHANNELS * POS_AREA; i++) {
    int s = map[i];
    if(s >= 0)
      dst[i] = src[s];
  }
}

void mapPolicyToGame(const float* raw, int sym, float* out) {
  std::call_once(precomputedTablesFlag, initPrecomputedTables);
  const int16_t* map = policySlotMap[sym];
  for(int variant = 0; variant < NUM_POLICY_VARIANTS; variant++) {
    const float* rawVariant = raw + variant * POLICY_SLOTS_PER_VARIANT;
    float* outVariant = out + variant * NUM_ACTIONS;
    for(int act = 0; act < NUM_ACTIONS; act++) {
      outVariant[act] = rawVariant[map[act]];
    }
  }
}

void softmaxLegal(
  const float* logits,
  const int* legal,
  int numLegal,
  float* out,
  float temperature
) {
  std::fill_n(out, NUM_ACTIONS, 0.0f);
  if(numLegal <= 0)
    return;
  float maxLogit = logits[legal[0]];
  for(int i = 1; i < numLegal; i++)
    maxLogit = std::max(maxLogit, logits[legal[i]]);
  const float invTemp = 1.0f / std::max(1e-4f, temperature);
  double sum = 0.0;
  for(int i = 0; i < numLegal; i++) {
    int a = legal[i];
    float p = std::exp((logits[a] - maxLogit) * invTemp);
    out[a] = p;
    sum += p;
  }
  float invSum = (float)(1.0 / sum);
  for(int i = 0; i < numLegal; i++) {
    int a = legal[i];
    out[a] *= invSum;
  }
}

void softmaxLegal(const float* logits, const std::vector<int>& legal, float* out, float temperature) {
  softmaxLegal(logits, legal.data(), (int)legal.size(), out, temperature);
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

void computeMaskedValue(const Q4Board& board, const float* valueAbs, float* valueAbsMasked) {
  double maskedSum = 0.0;
  for(int s = 0; s < 4; s++) {
    if(board.isAlive(s)) {
      valueAbsMasked[s] = valueAbs[s];
      maskedSum += valueAbs[s];
    } else {
      valueAbsMasked[s] = 0.0f;
    }
  }
  valueAbsMasked[4] = valueAbs[4];
  maskedSum += valueAbs[4];
  if(maskedSum > 0.0) {
    for(int i = 0; i < NUM_VALUE_LOGITS; i++)
      valueAbsMasked[i] = (float)(valueAbsMasked[i] / maskedSum);
  } else {
    valueAbsMasked[4] = 1.0f;
  }
}

void decodeTrajectory(const float* raw, int sym, float* out) {
  if(sym == 0) {
    for(int c = 0; c < POS_AREA; c++)
      out[c] = 1.0f / (1.0f + std::exp(-raw[c]));
    return;
  }
  for(int c = 0; c < POS_AREA; c++) {
    int sc = Q4RawSymmetry::applyCell(c, sym);
    out[c] = 1.0f / (1.0f + std::exp(-raw[sc]));
  }
}

Hash128 getCacheHash(const Q4PlayState& state) {
  auto mix = [](uint64_t tag, uint64_t v) {
    uint64_t a = Hash::splitMix64(tag ^ v);
    return Hash128(a, Hash::nasam(a ^ tag));
  };
  Hash128 h = state.board.hash;
  h ^= mix(0x5132526c73ULL, (uint64_t)(uint32_t)state.rules.maxPlies | ((uint64_t)(uint32_t)state.rules.repetitionDrawCount << 32));
  h ^= mix(0x506c696573ULL, (uint64_t)std::max(0, state.rules.maxPlies - state.plies));
  // Note: symmetry is NO LONGER included in the cache key (Plan §7 item 1)

  if(state.rules.repetitionDrawCount >= 2) {
    uint64_t rep = Hash::splitMix64(0x5245504554ULL ^ (uint64_t)state.currentPositionRepetitionCount());
    int dests[16];
    int numDests = state.board.getPawnMoves(state.board.toMove, dests);
    for(int i = 0; i < numDests; i++) {
      int c = dests[i];
      int occ = occurrencesAfterPawnMove(state, c);
      if(occ > 0)
        rep = Hash::splitMix64(rep ^ ((uint64_t)c << 16) ^ (uint64_t)occ);
    }
    h ^= mix(0x5265704861ULL, rep);
  }
  return h;
}

Hash128 getCacheHash(const Q4Board& board, const Q4History& history, int /*sym*/) {
  Q4PlayState s = history.state;
  s.board = board;
  s.rules = history.rules;
  s.plies = history.plies;
  s.isFinished = history.isFinished;
  s.winnerSeat = history.winnerSeat;
  s.isDraw = history.isDraw;
  s.repetitionHashes = history.repetitionHashes;
  return getCacheHash(s);
}

void evaluate(
  NNEvaluator& nnEval,
  NNResultBuf& buf,
  const Q4PlayState& state,
  int sym,
  bool skipCache,
  Eval& out,
  Rand* rand,
  float nnPolicyTemperature
) {
  const Hash128 key = getCacheHash(state);
  const bool nhwc = nnEval.getInputsUseNHWC();

  bool hit = false;
  if(!skipCache) {
    hit = nnEval.getCacheTableEntry(key, buf.result);
  }

  int symUsed = 0;
  if(hit) {
    buf.hasResult = true;
    symUsed = 0;
    out.numLegalActions = state.board.getLegalActions(state.board.toMove, out.legalActions);
  } else {
    int symToUse = sym;
    if(symToUse < 0 || symToUse > 7) {
      if(nnEval.getDoRandomize()) {
        static thread_local Rand tlsRand;
        symToUse = (rand != nullptr) ? (int)rand->nextUInt(8) : (int)tlsRand.nextUInt(8);
      } else {
        int defSym = nnEval.getDefaultSymmetry();
        symToUse = (defSym >= 0) ? (defSym % 8) : 0;
      }
    }
    symUsed = symToUse;

    static thread_local float spatialBuf[NUM_SPATIAL_CHANNELS * POS_AREA];
    static thread_local float symSpatialBuf[NUM_SPATIAL_CHANNELS * POS_AREA];
    static thread_local float globalBuf[NUM_GLOBAL_FEATURES];
    fillRow(state, nhwc, spatialBuf, globalBuf, nullptr, out.legalActions, &out.numLegalActions);
    applyInputSymmetry(spatialBuf, symSpatialBuf, symToUse, nhwc);
    nnEval.evaluateQ4Raw(symSpatialBuf, globalBuf, key, buf, skipCache, symToUse);
  }

  const Q4RawNNOutput* raw = buf.result->q4Raw.get();
  if(raw == nullptr)
    throw StringError("Q4NN::evaluate: the evaluator returned no Q4 raw output (not a Q4 model?)");

  out.toMove = state.board.toMove;
  out.sym = symUsed;

  // Since raw output has already been un-symmetrized in server thread before caching,
  // we always decode using symmetry 0.
  mapPolicyToGame(raw->policyLogits, 0, &out.policyLogits[0][0]);
  std::copy(raw->valueLogits, raw->valueLogits + NUM_VALUE_LOGITS, out.valueLogits);
  softmaxValue(raw->valueLogits, out.valueRel);
  rotateValueToAbsolute(out.valueRel, out.toMove, out.valueAbs);

  // A4: Masked values (eliminated seats set to 0 and renormalized)
  computeMaskedValue(state.board, out.valueAbs, out.valueAbsMasked);

  std::copy(raw->miscValues, raw->miscValues + NUM_MISC, out.misc);
  // A3: Short-term value error
  out.shorttermWinlossError = decodeShorttermValueError(raw->miscValues[5], nnEval.getPostProcessParams().shorttermValueErrorMultiplier);

  decodeTrajectory(raw->trajectoryLogits, 0, out.trajectory);

  // Policy probabilities over legal actions
  softmaxLegal(out.policyLogits[0], out.legalActions, out.numLegalActions, out.policyProbs[0], nnPolicyTemperature);
  softmaxLegal(out.policyLogits[1], out.legalActions, out.numLegalActions, out.policyProbs[1], 1.0f);
}

void evaluate(
  NNEvaluator& nnEval,
  NNResultBuf& buf,
  const Q4History& history,
  int sym,
  bool skipCache,
  Eval& out,
  Rand* rand,
  float nnPolicyTemperature
) {
  Q4PlayState s = history.state;
  s.board = history.currentBoard;
  s.rules = history.rules;
  s.plies = history.plies;
  s.isFinished = history.isFinished;
  s.winnerSeat = history.winnerSeat;
  s.isDraw = history.isDraw;
  s.repetitionHashes = history.repetitionHashes;
  evaluate(nnEval, buf, s, sym, skipCache, out, rand, nnPolicyTemperature);
}

void averageMultipleSymmetries(
  NNEvaluator& nnEval,
  NNResultBuf& buf,
  const Q4PlayState& state,
  Rand& rand,
  int numSymmetries,
  Eval& out,
  float nnPolicyTemperature
) {
  int numToSample = std::max(1, std::min(numSymmetries, 8));
  std::array<int, 8> symIndices = {0, 1, 2, 3, 4, 5, 6, 7};
  for(int i = 0; i < numToSample; i++) {
    int j = (int)rand.nextInt(i, 7);
    std::swap(symIndices[i], symIndices[j]);
  }

  out.toMove = state.board.toMove;
  out.sym = -1;
  std::fill(&out.policyLogits[0][0], &out.policyLogits[0][0] + NUM_POLICY_VARIANTS * NUM_ACTIONS, 0.0f);
  std::fill(&out.policyProbs[0][0], &out.policyProbs[0][0] + NUM_POLICY_VARIANTS * NUM_ACTIONS, 0.0f);
  std::fill(out.valueLogits, out.valueLogits + NUM_VALUE_LOGITS, 0.0f);
  std::fill(out.valueRel, out.valueRel + NUM_VALUE_LOGITS, 0.0f);
  std::fill(out.valueAbs, out.valueAbs + NUM_VALUE_LOGITS, 0.0f);
  std::fill(out.valueAbsMasked, out.valueAbsMasked + NUM_VALUE_LOGITS, 0.0f);
  std::fill(out.misc, out.misc + NUM_MISC, 0.0f);
  out.shorttermWinlossError = 0.0f;
  std::fill(out.trajectory, out.trajectory + POS_AREA, 0.0f);

  for(int i = 0; i < numToSample; i++) {
    Eval single;
    evaluate(nnEval, buf, state, symIndices[i], /*skipCache=*/true, single, &rand, nnPolicyTemperature);
    if(i == 0) {
      out.numLegalActions = single.numLegalActions;
      std::copy(single.legalActions, single.legalActions + single.numLegalActions, out.legalActions);
    }
    for(int v = 0; v < NUM_POLICY_VARIANTS; v++) {
      for(int a = 0; a < NUM_ACTIONS; a++) {
        out.policyLogits[v][a] += single.policyLogits[v][a];
        out.policyProbs[v][a] += single.policyProbs[v][a];
      }
    }
    for(int k = 0; k < NUM_VALUE_LOGITS; k++) {
      out.valueLogits[k] += single.valueLogits[k];
      out.valueRel[k] += single.valueRel[k];
      out.valueAbs[k] += single.valueAbs[k];
      out.valueAbsMasked[k] += single.valueAbsMasked[k];
    }
    for(int m = 0; m < NUM_MISC; m++) {
      out.misc[m] += single.misc[m];
    }
    out.shorttermWinlossError += single.shorttermWinlossError;
    for(int c = 0; c < POS_AREA; c++) {
      out.trajectory[c] += single.trajectory[c];
    }
  }

  const float invN = 1.0f / (float)numToSample;
  for(int v = 0; v < NUM_POLICY_VARIANTS; v++) {
    for(int a = 0; a < NUM_ACTIONS; a++) {
      out.policyLogits[v][a] *= invN;
      out.policyProbs[v][a] *= invN;
    }
  }
  for(int k = 0; k < NUM_VALUE_LOGITS; k++) {
    out.valueLogits[k] *= invN;
    out.valueRel[k] *= invN;
    out.valueAbs[k] *= invN;
    out.valueAbsMasked[k] *= invN;
  }
  for(int m = 0; m < NUM_MISC; m++) {
    out.misc[m] *= invN;
  }
  out.shorttermWinlossError *= invN;
  for(int c = 0; c < POS_AREA; c++) {
    out.trajectory[c] *= invN;
  }
}

void averageMultipleSymmetries(
  NNEvaluator& nnEval,
  NNResultBuf& buf,
  const Q4History& history,
  Rand& rand,
  int numSymmetries,
  Eval& out,
  float nnPolicyTemperature
) {
  averageMultipleSymmetries(nnEval, buf, history.state, rand, numSymmetries, out, nnPolicyTemperature);
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
