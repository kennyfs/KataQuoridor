#include "../../tests/tests.h"
#include "../../core/rand.h"
#include "../../neuralnet/nninputs.h"
#include "../q4board.h"
#include "../q4history.h"
#include "../q4notation.h"
#include "../q4rules.h"
#include "../q4symmetry.h"
#include "../nn/q4nn.h"
#include "../command/q4nntool.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
using namespace TestCommon;

namespace {

void testActionSymmetries() {
  for(int sym = 0; sym < Q4Symmetry::NUM_SYMMETRIES; sym++) {
    int invSym = Q4Symmetry::inverse(sym);
    for(int act = 0; act < Q4Board::NUM_ACTIONS; act++)
      testAssert(Q4Symmetry::applyAction(Q4Symmetry::applyAction(act, sym), invSym) == act);
    for(int d = 0; d < 4; d++)
      testAssert(Q4Symmetry::applyDirection(Q4Symmetry::applyDirection(d, sym), invSym) == d);
  }
  for(int s1 = 0; s1 < Q4Symmetry::NUM_SYMMETRIES; s1++) {
    for(int s2 = 0; s2 < Q4Symmetry::NUM_SYMMETRIES; s2++) {
      int s12 = Q4Symmetry::compose(s1, s2);
      for(int act = 0; act < Q4Board::NUM_ACTIONS; act++)
        testAssert(Q4Symmetry::applyAction(act, s12) == Q4Symmetry::applyAction(Q4Symmetry::applyAction(act, s2), s1));
    }
  }
}

// T14: a one-hot policy on each of the 321 actions, in both variants, round-trips through every symmetry.
void testPolicyRoundTrip() {
  using namespace Q4NNConst;
  vector<float> raw(POLICY_SLOTS), out(NUM_POLICY_VARIANTS * NUM_ACTIONS);
  for(int sym = 0; sym < Q4Symmetry::NUM_SYMMETRIES; sym++) {
    for(int variant = 0; variant < NUM_POLICY_VARIANTS; variant++) {
      for(int act = 0; act < NUM_ACTIONS; act++) {
        // The slot the net would use for this game action when evaluating the transformed position.
        int a = Q4Symmetry::applyAction(act, sym);
        int plane = Q4Board::isPawnAction(a) ? 0 : (Q4Board::isVWallAction(a) ? 1 : 2);
        int cell = a;
        if(plane == 1) cell = Q4Board::anchorY(a - 121) * POS_LEN + Q4Board::anchorX(a - 121);
        if(plane == 2) cell = Q4Board::anchorY(a - 221) * POS_LEN + Q4Board::anchorX(a - 221);
        fill(raw.begin(), raw.end(), 0.0f);
        raw[variant * POLICY_SLOTS_PER_VARIANT + plane * POS_AREA + cell] = 1.0f;
        Q4NN::mapPolicyToGame(raw.data(), sym, out.data());
        for(int v = 0; v < NUM_POLICY_VARIANTS; v++)
          for(int b = 0; b < NUM_ACTIONS; b++)
            testAssert(out[v * NUM_ACTIONS + b] == ((v == variant && b == act) ? 1.0f : 0.0f));
      }
    }
  }
}

// T14: the inputs of the transformed position (symmetry 0) equal the transformed inputs of the position, for all
// channels and under every symmetry. Both sides are computed by the C++ code; this tests applyInputSymmetry against
// an independent computation of the same features on the geometrically transformed game.
void testInputSymmetryConsistency() {
  using namespace Q4NNConst;
  vector<Q4History> positions = Q4NNTool::sampleHistories(4242, 1200);
  testAssert(positions.size() == 1200);
  vector<float> spA(NUM_SPATIAL_CHANNELS * POS_AREA), spB(NUM_SPATIAL_CHANNELS * POS_AREA), spT(NUM_SPATIAL_CHANNELS * POS_AREA);
  vector<float> glA(NUM_GLOBAL_FEATURES), glB(NUM_GLOBAL_FEATURES);
  int numWithElim = 0, numWithRepeat = 0;
  for(size_t i = 0; i < positions.size(); i += 3) {
    const Q4History& h = positions[i];
    Q4NN::fillRow(h.currentBoard, h, false, spA.data(), glA.data());
    if(h.currentBoard.getNumAlive() < 4) numWithElim++;
    for(int p = 0; p < POS_AREA; p++) if(spA[25 * POS_AREA + p] > 0.0f) { numWithRepeat++; break; }
    for(int sym = 0; sym < Q4Symmetry::NUM_SYMMETRIES; sym++) {
      Q4History t = Q4Symmetry::applyHistory(h, sym);
      Q4NN::fillRow(t.currentBoard, t, false, spB.data(), glB.data());
      Q4NN::applyInputSymmetry(spA.data(), spT.data(), sym, false);
      for(int k = 0; k < NUM_SPATIAL_CHANNELS * POS_AREA; k++) {
        if(spB[k] != spT[k]) {
          cout << "position " << i << " sym " << sym << " channel " << k / POS_AREA << " cell " << k % POS_AREA << ": "
               << spB[k] << " vs " << spT[k] << endl;
        }
        testAssert(spB[k] == spT[k]);
      }
      for(int k = 0; k < NUM_GLOBAL_FEATURES; k++)
        testAssert(glA[k] == glB[k]);

      // NHWC <-> NCHW: same content, and the symmetry works the same in both layouts.
      vector<float> nhwc(NUM_SPATIAL_CHANNELS * POS_AREA), nhwcSym(NUM_SPATIAL_CHANNELS * POS_AREA);
      vector<float> glC(NUM_GLOBAL_FEATURES);
      Q4NN::fillRow(h.currentBoard, h, true, nhwc.data(), glC.data());
      Q4NN::applyInputSymmetry(nhwc.data(), nhwcSym.data(), sym, true);
      for(int ch = 0; ch < NUM_SPATIAL_CHANNELS; ch++)
        for(int c = 0; c < POS_AREA; c++)
          testAssert(nhwcSym[c * NUM_SPATIAL_CHANNELS + ch] == spT[ch * POS_AREA + c]);
    }
  }
  cout << "  symmetry test positions with eliminated seats: " << numWithElim << ", with repeating pawn moves: " << numWithRepeat << endl;
  testAssert(numWithElim > 5 && numWithRepeat > 5);
}

// Structural properties of the inputs that do not depend on any symmetry.
void testInputStructure() {
  using namespace Q4NNConst;
  vector<Q4History> positions = Q4NNTool::sampleHistories(777, 600);
  vector<float> sp(NUM_SPATIAL_CHANNELS * POS_AREA), gl(NUM_GLOBAL_FEATURES);
  auto at = [&](int ch, int c) { return sp[ch * POS_AREA + c]; };
  for(const Q4History& h : positions) {
    const Q4Board& b = h.currentBoard;
    Q4NN::RawDistances rd;
    Q4NN::fillRow(b, h, false, sp.data(), gl.data(), &rd);
    int toMove = b.toMove;
    for(int c = 0; c < POS_AREA; c++) {
      testAssert(at(0, c) == 1.0f);
      testAssert(at(5, c) == (c == Q4Board::CENTER_CELL ? 1.0f : 0.0f));
      // Raw distances and the float planes agree.
      testAssert(at(10, c) == Q4NN::dist01(rd.d[0][c]));
      for(int k = 0; k < 4; k++)
        testAssert(at(11 + k, c) == Q4NN::dist01(rd.d[1 + k][c]));
    }
    for(int k = 0; k < 4; k++) {
      int s = (toMove + k) % 4;
      float count = 0.0f, pathCount = 0.0f;
      for(int c = 0; c < POS_AREA; c++) {
        count += at(1 + k, c);
        pathCount += at(15 + k, c);
      }
      testAssert(count == (b.isAlive(s) ? 1.0f : 0.0f));
      if(b.isAlive(s)) {
        testAssert(at(1 + k, b.pawn[s]) == 1.0f);
        testAssert(rd.d[1 + k][b.pawn[s]] == 0);
        testAssert(at(15 + k, b.pawn[s]) == 1.0f && at(15 + k, Q4Board::CENTER_CELL) == 1.0f);
        testAssert(gl[8 + k] == 1.0f);
        testAssert(gl[0 + k] == (float)b.wallsLeft[s] / 7.0f);
        testAssert(gl[4 + k] == (b.wallsLeft[s] > 0 ? 1.0f : 0.0f));
      }
      else {
        testAssert(pathCount == 0.0f);
        testAssert(gl[4 + k] == 0.0f);
        for(int c = 0; c < POS_AREA; c++)
          testAssert(rd.d[1 + k][c] == 255);
      }
    }
    // Pawn destinations and legal walls agree with the board.
    vector<int> dests;
    b.getPawnMoves(toMove, dests);
    float destCount = 0.0f;
    for(int c = 0; c < POS_AREA; c++) destCount += at(24, c);
    testAssert((int)destCount == (int)dests.size());
    for(int c : dests) testAssert(at(24, c) == 1.0f);
    // The hash after a pawn move (used for the repetition inputs) is the hash of the board after really making the move.
    for(int c : dests) {
      Q4Board after = b;
      after.applyAction(Q4Board::actionOfPawn(c));
      testAssert(b.getHashAfterPawnMove(c) == after.hash);
    }
    for(int ay = 0; ay < 10; ay++) {
      for(int ax = 0; ax < 10; ax++) {
        testAssert(at(21, ay * POS_LEN + ax) == 1.0f);
        testAssert(at(22, ay * POS_LEN + ax) == (b.isGeometricallyLegalWall(ax, ay, false) ? 1.0f : 0.0f));
        testAssert(at(23, ay * POS_LEN + ax) == (b.isGeometricallyLegalWall(ax, ay, true) ? 1.0f : 0.0f));
        testAssert(at(19, ay * POS_LEN + ax) == (b.vWalls.test(Q4Board::anchorOf(ax, ay)) ? 1.0f : 0.0f));
        testAssert(at(20, ay * POS_LEN + ax) == (b.hWalls.test(Q4Board::anchorOf(ax, ay)) ? 1.0f : 0.0f));
      }
      testAssert(at(21, ay * POS_LEN + 10) == 0.0f && at(19, ay * POS_LEN + 10) == 0.0f);
    }
    for(int ax = 0; ax < POS_LEN; ax++)
      testAssert(at(21, 10 * POS_LEN + ax) == 0.0f && at(20, 10 * POS_LEN + ax) == 0.0f);
    // Repetition channels are off when the rule is off.
    if(h.rules.repetitionDrawCount < 2) {
      for(int c = 0; c < POS_AREA; c++) testAssert(at(25, c) == 0.0f && at(26, c) == 0.0f);
      testAssert(gl[26] == 0.0f && gl[27] == 0.0f);
    }
    else {
      testAssert(gl[26] == 1.0f);
      for(int c = 0; c < POS_AREA; c++) testAssert(at(26, c) <= at(25, c));  // a drawing move is also a repeating one
    }
    // Race leader: exactly one, or none if nobody can reach the center.
    float leaders = gl[20] + gl[21] + gl[22] + gl[23];
    testAssert(leaders == 1.0f || leaders == 0.0f);
    testAssert(gl[24] == (float)b.getNumAlive() / 4.0f);
    testAssert(gl[25] == (float)max(0, h.rules.maxPlies - h.plies) / 400.0f);
  }
}

// setQ4RawNNOutput reads the backends' two policy layouts the same way.
void testRawOutputLayouts() {
  using namespace Q4NNConst;
  const int numCh = NUM_POLICY_VARIANTS * NUM_POLICY_PLANES;
  vector<float> nchw(numCh * POS_AREA), nhwc(numCh * POS_AREA);
  for(int ch = 0; ch < numCh; ch++)
    for(int pos = 0; pos < POS_AREA; pos++) {
      float v = (float)(ch * 1000 + pos);
      nchw[ch * POS_AREA + pos] = v;
      nhwc[pos * numCh + ch] = v;
    }
  float value[NUM_VALUE_LOGITS] = {1, 2, 3, 4, 5}, misc[NUM_MISC] = {6, 7, 8, 9, 10, 11};
  vector<float> traj(TRAJECTORY_SLOTS);
  for(int i = 0; i < TRAJECTORY_SLOTS; i++) traj[i] = (float)i;
  NNOutput a, b;
  setQ4RawNNOutput(&a, nchw.data(), false, value, misc, traj.data());
  setQ4RawNNOutput(&b, nhwc.data(), true, value, misc, traj.data());
  testAssert(a.q4Raw != nullptr && b.q4Raw != nullptr);
  for(int k = 0; k < POLICY_SLOTS; k++) {
    testAssert(a.q4Raw->policyLogits[k] == (float)((k / POS_AREA) * 1000 + k % POS_AREA));
    testAssert(a.q4Raw->policyLogits[k] == b.q4Raw->policyLogits[k]);
  }
  for(int i = 0; i < NUM_VALUE_LOGITS; i++) testAssert(a.q4Raw->valueLogits[i] == value[i]);
  for(int i = 0; i < NUM_MISC; i++) testAssert(b.q4Raw->miscValues[i] == misc[i]);
  for(int i = 0; i < TRAJECTORY_SLOTS; i++) testAssert(b.q4Raw->trajectoryLogits[i] == traj[i]);
  NNOutput c(b);  // copies share the raw block
  testAssert(c.q4Raw == b.q4Raw);
}

void testValueRotationAndSoftmax() {
  float rel[5] = {0.4f, 0.3f, 0.2f, 0.1f, 0.0f}, abs[5];
  for(int toMove = 0; toMove < 4; toMove++) {
    Q4NN::rotateValueToAbsolute(rel, toMove, abs);
    for(int k = 0; k < 4; k++)
      testAssert(abs[(toMove + k) % 4] == rel[k]);
    testAssert(abs[4] == rel[4]);
  }
  float logits[5] = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f}, probs[5];
  Q4NN::softmaxValue(logits, probs);
  float sum = 0.0f;
  for(int i = 0; i < 5; i++) sum += probs[i];
  testAssert(std::abs(sum - 1.0f) < 1e-6f && probs[3] > probs[2]);
}

// T18, the key part: what the cache key does and does not depend on. (The real cache, with a network, is tested
// by q4tool nncache in python/tests/test_q4_export_load.py.)
void testCacheKey() {
  Q4Rules rules;
  rules.repetitionDrawCount = 3;
  Q4History hist(rules);
  Hash128 base = Q4NN::getCacheHash(hist.currentBoard, hist, 0);
  testAssert(base == Q4NN::getCacheHash(hist.currentBoard, hist, 0));

  // The symmetry is part of the key; every symmetry differs.
  for(int sym = 1; sym < 8; sym++)
    testAssert(Q4NN::getCacheHash(hist.currentBoard, hist, sym) != base);
  // The position (and with it the seat to move) is part of the key.
  { Q4History h2 = hist; vector<int> a; h2.currentBoard.getLegalActions(0, a); h2.play(a[0]);
    testAssert(Q4NN::getCacheHash(h2.currentBoard, h2, 0) != base); }
  // Plies until the draw.
  { Q4History h2 = hist; h2.plies = 10; testAssert(Q4NN::getCacheHash(h2.currentBoard, h2, 0) != base); }
  // Rules the inputs read.
  { Q4Rules r2 = rules; r2.maxPlies = 150; Q4History h2(r2); testAssert(Q4NN::getCacheHash(h2.currentBoard, h2, 0) != base); }
  { Q4Rules r2 = rules; r2.repetitionDrawCount = 2; Q4History h2(r2); testAssert(Q4NN::getCacheHash(h2.currentBoard, h2, 0) != base); }
  { Q4Rules r2 = rules; r2.repetitionDrawCount = 0; Q4History h2(r2); testAssert(Q4NN::getCacheHash(h2.currentBoard, h2, 0) != base); }

  // Repetition state: the same board reached with a different repetition count gets a different key, and the keys
  // follow the inputs: two histories with equal inputs have equal keys, different inputs different keys.
  vector<Q4History> positions = Q4NNTool::sampleHistories(99, 2500);
  vector<float> spA(Q4NNConst::NUM_SPATIAL_CHANNELS * Q4NNConst::POS_AREA), glA(Q4NNConst::NUM_GLOBAL_FEATURES);
  vector<float> spB(spA.size()), glB(glA.size());
  int pairsDifferentInputsSameBoard = 0;
  for(size_t i = 0; i + 1 < positions.size(); i++) {
    for(size_t j = i + 1; j < std::min(positions.size(), i + 40); j++) {
      const Q4History& x = positions[i];
      const Q4History& y = positions[j];
      if(x.currentBoard.hash != y.currentBoard.hash)
        continue;
      Q4NN::fillRow(x.currentBoard, x, false, spA.data(), glA.data());
      Q4NN::fillRow(y.currentBoard, y, false, spB.data(), glB.data());
      bool sameInputs = spA == spB && glA == glB;
      bool sameKey = Q4NN::getCacheHash(x.currentBoard, x, 0) == Q4NN::getCacheHash(y.currentBoard, y, 0);
      testAssert(sameInputs == sameKey);
      if(!sameInputs) pairsDifferentInputsSameBoard++;
    }
  }
  testAssert(pairsDifferentInputsSameBoard > 0);
}

}  // namespace

void Tests::runQ4NNTests() {
  cout << "Running Q4 NN I/O tests" << endl;
  testActionSymmetries();
  testPolicyRoundTrip();
  testInputSymmetryConsistency();
  testInputStructure();
  testRawOutputLayouts();
  testValueRotationAndSoftmax();
  testCacheKey();
  cout << "Q4 NN I/O tests passed" << endl;
}
