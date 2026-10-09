#include "../../tests/tests.h"
#include "../../core/fileutils.h"
#include "../../core/global.h"
#include "../../core/makedir.h"
#include "../../core/rand.h"
#include "../../program/setup.h"
#include "../../neuralnet/nneval.h"
#include "../q4board.h"
#include "../q4history.h"
#include "../q4notation.h"
#include "../q4playstate.h"
#include "../q4rules.h"
#include "../q4record.h"
#include "../nn/q4nn.h"
#include "../nn/q4nnconstants.h"
#include "../dataio/q4trainingwrite.h"
#include "../play/q4play.h"
#include "../play/q4playsettings.h"
#include "../play/q4population.h"
#include "../play/q4selfplaymanager.h"

#include <cmath>
#include <iostream>
#include <vector>

using namespace std;
using namespace TestCommon;
using namespace Q4Play;

namespace {

string getTestModelPath() {
  vector<string> candidates = {
    "cpp/tests/models/b1c32_q4/model.bin.gz",
    "tests/models/b1c32_q4/model.bin.gz",
    "../tests/models/b1c32_q4/model.bin.gz"
  };
  for(const string& p : candidates) {
    if(FileUtils::exists(p)) return p;
  }
  const char* pyEnv = getenv("KATAGO_TEST_PYTHON");
  string pythonCmd = (pyEnv != nullptr && strlen(pyEnv) > 0) ? string(pyEnv) : "python3";
  string cmd = pythonCmd + " python/q4/make_random_model.py b1c32_q4 cpp/tests/models --model-name b1c32_q4 --scale-heads";
  int ret = system(cmd.c_str());
  (void)ret;
  for(const string& p : candidates) {
    if(FileUtils::exists(p)) return p;
  }
  throw StringError("Could not find or generate b1c32_q4 model for q4selfplay tests");
}

void testActionToPolicySlot() {
  cout << "Testing action to policy slot..." << endl;
  // Pawn moves
  for(int a = 0; a < 121; a++) {
    int slot = Q4TrainingWriteBuffers::actionToPolicySlot(a);
    testAssert(slot == a);
  }
  // V-walls: 121 + ay * 11 + ax
  for(int anchor = 0; anchor < 100; anchor++) {
    int act = 121 + anchor;
    int ay = Q4Board::anchorY(anchor);
    int ax = Q4Board::anchorX(anchor);
    int slot = Q4TrainingWriteBuffers::actionToPolicySlot(act);
    testAssert(slot == 121 + ay * 11 + ax);
  }
  // H-walls: 242 + ay * 11 + ax
  for(int anchor = 0; anchor < 100; anchor++) {
    int act = 221 + anchor;
    int ay = Q4Board::anchorY(anchor);
    int ax = Q4Board::anchorX(anchor);
    int slot = Q4TrainingWriteBuffers::actionToPolicySlot(act);
    testAssert(slot == 242 + ay * 11 + ax);
  }
  testAssert(Q4TrainingWriteBuffers::actionToPolicySlot(Q4Board::NULL_ACTION) == -1);
}

void testTDTargetsMath() {
  cout << "Testing TD targets math and seat rotation..." << endl;
  std::vector<Q4ValueTargets> vts;
  Q4ValueTargets vt0;
  vt0.value[0] = 0.4f; vt0.value[1] = 0.3f; vt0.value[2] = 0.2f; vt0.value[3] = 0.1f; vt0.value[4] = 0.0f;
  vts.push_back(vt0);

  Q4ValueTargets vt1;
  vt1.value[0] = 0.1f; vt1.value[1] = 0.6f; vt1.value[2] = 0.1f; vt1.value[3] = 0.1f; vt1.value[4] = 0.1f;
  vts.push_back(vt1);

  // Result: seat 1 wins
  Q4ValueTargets finalResult;
  finalResult.value[0] = 0.0f; finalResult.value[1] = 1.0f; finalResult.value[2] = 0.0f; finalResult.value[3] = 0.0f; finalResult.value[4] = 0.0f;
  vts.push_back(finalResult);

  float buf[5];

  // nowFactor = 0.0 (final outcome) from perspective of seat 0
  Q4TrainingWriteBuffers::fillValueTDTargets(vts, 0, 0, 0.0, buf);
  testAssert(std::abs(buf[0] - 0.0f) < 1e-6);
  testAssert(std::abs(buf[1] - 1.0f) < 1e-6);
  testAssert(std::abs(buf[2] - 0.0f) < 1e-6);
  testAssert(std::abs(buf[3] - 0.0f) < 1e-6);
  testAssert(std::abs(buf[4] - 0.0f) < 1e-6);

  // nowFactor = 0.0 from perspective of seat 1: seat 1 is relative seat 0 (me)!
  Q4TrainingWriteBuffers::fillValueTDTargets(vts, 0, 1, 0.0, buf);
  testAssert(std::abs(buf[0] - 1.0f) < 1e-6); // me won!
  testAssert(std::abs(buf[1] - 0.0f) < 1e-6);

  // nowFactor = 1.0 from perspective of seat 0 at turn 0: turn 0's value
  Q4TrainingWriteBuffers::fillValueTDTargets(vts, 0, 0, 1.0, buf);
  testAssert(std::abs(buf[0] - 0.4f) < 1e-6);
  testAssert(std::abs(buf[1] - 0.3f) < 1e-6);
  testAssert(std::abs(buf[2] - 0.2f) < 1e-6);
  testAssert(std::abs(buf[3] - 0.1f) < 1e-6);

  // nowFactor = 1.0 from perspective of seat 2 at turn 0: relative seat 0 is absolute seat 2
  Q4TrainingWriteBuffers::fillValueTDTargets(vts, 0, 2, 1.0, buf);
  testAssert(std::abs(buf[0] - 0.2f) < 1e-6); // me = seat 2
  testAssert(std::abs(buf[1] - 0.1f) < 1e-6); // next = seat 3
  testAssert(std::abs(buf[2] - 0.4f) < 1e-6); // across = seat 0
  testAssert(std::abs(buf[3] - 0.3f) < 1e-6); // prev = seat 1
}

void addPopulationKeys(ConfigParser& cfg, const string& mixedProb) {
  cfg.overrideKey("q4PopulationMixedProb", mixedProb);
  cfg.overrideKey("q4PopulationWeights", "weak:1, snapshot:1, greedy:0.5, randomPawn:0.5, basher:0.5, grudge:0.5");
  cfg.overrideKey("q4PopulationWeakVisits", "1, 2, 4, 8, 16, 32");
  cfg.overrideKey("q4PopulationWeakTemperatureMin", "0.5");
  cfg.overrideKey("q4PopulationWeakTemperatureMax", "1.5");
  cfg.overrideKey("q4PopulationNumSnapshots", "3");
  cfg.overrideKey("q4PopulationSnapshotVisits", "100");
  cfg.overrideKey("q4PopulationSnapshotChosenMoveTemperatureEarly", "0.6");
  cfg.overrideKey("q4PopulationSnapshotChosenMoveTemperature", "0.2");
  cfg.overrideKey("q4PopulationSnapshotCacheSizePowerOfTwo", "16");
}

// P2: the composition sampler on 10,000 draws matches the configured probabilities (docs/q4/rounds/R7.md).
void testP2CompositionSampler() {
  cout << "Running P2 composition sampler..." << endl;
  ConfigParser cfg;
  addPopulationKeys(cfg, "0.5");
  Q4PopulationSettings pop = Q4PopulationSettings::load(cfg);
  testAssert(pop.mixedProb == 0.5 && pop.numSnapshots == 3 && pop.snapshotVisits == 100);
  testAssert(pop.kindWeights[SEAT_WEAK] == 1.0 && pop.kindWeights[SEAT_GRUDGE] == 0.5);

  const int N = 10000;
  Rand rand("p2_sampler");
  int numMixed = 0;
  int learnersCount[5] = {0, 0, 0, 0, 0};
  int learnerSeatCount[4] = {0, 0, 0, 0};   // among mixed games with 1 learner
  int kindCount[NUM_SEAT_KINDS] = {0, 0, 0, 0, 0, 0, 0};
  int weakVisitsCount[64] = {0};
  int snapshotIdxCount[3] = {0, 0, 0};
  double weakTempSum = 0.0, weakTempMin = 1e9, weakTempMax = -1e9;
  int numWeak = 0, numNonLearner = 0, numSnapshot = 0, numGrudgeInTwoLearner = 0;
  for(int i = 0; i < N; i++) {
    Q4Composition c = sampleComposition(rand, pop, 3);
    int nl = c.numLearners();
    testAssert(nl >= 1 && nl <= 4);
    learnersCount[nl]++;
    if(nl == 4)
      continue;
    numMixed++;
    if(nl == 1)
      for(int s = 0; s < 4; s++)
        if(c.seats[s].kind == SEAT_LEARNER)
          learnerSeatCount[s]++;
    for(int s = 0; s < 4; s++) {
      const Q4SeatSpec& spec = c.seats[s];
      if(spec.kind == SEAT_LEARNER)
        continue;
      numNonLearner++;
      kindCount[spec.kind]++;
      if(spec.kind == SEAT_WEAK) {
        numWeak++;
        testAssert(spec.weakVisits >= 1 && spec.weakVisits <= 32 && (spec.weakVisits & (spec.weakVisits - 1)) == 0);
        weakVisitsCount[spec.weakVisits]++;
        testAssert(spec.weakTemperature >= 0.5 && spec.weakTemperature <= 1.5);
        weakTempSum += spec.weakTemperature;
        weakTempMin = std::min(weakTempMin, spec.weakTemperature);
        weakTempMax = std::max(weakTempMax, spec.weakTemperature);
      }
      if(spec.kind == SEAT_SNAPSHOT) {
        numSnapshot++;
        testAssert(spec.snapshotIdx >= 0 && spec.snapshotIdx < 3);
        snapshotIdxCount[spec.snapshotIdx]++;
      }
      if(spec.kind == SEAT_GRUDGE) {
        // the target is a learner seat
        testAssert(spec.grudgeTarget >= 0 && spec.grudgeTarget < 4 && c.seats[spec.grudgeTarget].kind == SEAT_LEARNER);
        if(nl == 2) {
          numGrudgeInTwoLearner++;
        }
      }
    }
  }
  auto near = [](double observed, double expected, double sigmas, double n) {
    double sd = std::sqrt(expected * (1.0 - expected) / n);
    return std::fabs(observed - expected) <= sigmas * sd;
  };
  // all-learner games: probability 1 - mixedProb
  testAssert(near((double)learnersCount[4] / N, 0.5, 4.0, N));
  // among the mixed games the number of learners is uniform in {1,2,3}
  for(int nl = 1; nl <= 3; nl++)
    testAssert(near((double)learnersCount[nl] / numMixed, 1.0 / 3.0, 4.0, numMixed));
  // the seat of a single learner is uniform
  int numOneLearner = learnersCount[1];
  for(int s = 0; s < 4; s++)
    testAssert(near((double)learnerSeatCount[s] / numOneLearner, 0.25, 4.0, numOneLearner));
  // the kinds of the other seats follow the weights (chi-square, 5 degrees of freedom: 20.5 is the 0.001 quantile)
  double totalWeight = 0.0;
  for(int k = SEAT_WEAK; k < NUM_SEAT_KINDS; k++)
    totalWeight += pop.kindWeights[k];
  double chi2 = 0.0;
  for(int k = SEAT_WEAK; k < NUM_SEAT_KINDS; k++) {
    double expected = numNonLearner * pop.kindWeights[k] / totalWeight;
    chi2 += (kindCount[k] - expected) * (kindCount[k] - expected) / expected;
  }
  cout << "  P2: " << N << " draws, all-learner " << learnersCount[4] << ", mixed " << numMixed << " (1/2/3 learners "
       << learnersCount[1] << "/" << learnersCount[2] << "/" << learnersCount[3] << "), kind chi2 " << chi2 << endl;
  testAssert(chi2 < 20.5);
  // weak players: visits uniform over {1,2,4,8,16,32}, temperature uniform in [0.5,1.5]
  for(int v = 1; v <= 32; v *= 2)
    testAssert(near((double)weakVisitsCount[v] / numWeak, 1.0 / 6.0, 4.0, numWeak));
  testAssert(std::fabs(weakTempSum / numWeak - 1.0) < 0.02 && weakTempMin < 0.51 && weakTempMax > 1.49);
  // snapshots uniform; grudge targets uniform over the 2 learner seats of a 2-learner game
  for(int i = 0; i < 3; i++)
    testAssert(near((double)snapshotIdxCount[i] / numSnapshot, 1.0 / 3.0, 4.0, numSnapshot));
  testAssert(numGrudgeInTwoLearner > 100);

  // No snapshot available: the kind is never drawn
  {
    Rand r2("p2_nosnap");
    for(int i = 0; i < 2000; i++) {
      Q4Composition c = sampleComposition(r2, pop, 0);
      for(int s = 0; s < 4; s++)
        testAssert(c.seats[s].kind != SEAT_SNAPSHOT);
    }
  }
  // mixedProb 0: always all learners; mixedProb 1: never
  {
    ConfigParser c0, c1;
    addPopulationKeys(c0, "0");
    addPopulationKeys(c1, "1");
    Q4PopulationSettings p0 = Q4PopulationSettings::load(c0), p1 = Q4PopulationSettings::load(c1);
    Rand r3("p2_extremes");
    for(int i = 0; i < 2000; i++) {
      testAssert(sampleComposition(r3, p0, 3).numLearners() == 4);
      testAssert(sampleComposition(r3, p1, 3).numLearners() < 4);
    }
  }
  // config errors are hard errors
  {
    ConfigParser bad;
    addPopulationKeys(bad, "0.5");
    bad.overrideKey("q4PopulationWeights", "weak:1, snapshot:1");  // kinds missing
    bool threw = false;
    try { Q4PopulationSettings::load(bad); } catch(const StringError&) { threw = true; }
    testAssert(threw);
    ConfigParser bad2;
    addPopulationKeys(bad2, "0.5");
    bad2.overrideKey("q4PopulationWeights", "weak:1, snapshot:1, greedy:1, randomPawn:1, basher:1, grudge:1, learner:1");
    threw = false;
    try { Q4PopulationSettings::load(bad2); } catch(const StringError&) { threw = true; }
    testAssert(threw);
    ConfigParser bad3;
    threw = false;
    try { Q4PopulationSettings::load(bad3); } catch(const StringError&) { threw = true; }  // no keys at all
    testAssert(threw);
  }
  cout << "P2 passed!" << endl;
}

void testPlaySettingsValidation() {
  cout << "Testing PlaySettings config rejection..." << endl;
  {
    ConfigParser cfg;
    cfg.overrideKey("allowResignation", "true");
    bool threw = false;
    try { Q4PlaySettings::loadForSelfplay(cfg); }
    catch(const StringError&) { threw = true; }
    testAssert(threw);
  }
  {
    ConfigParser cfg;
    cfg.overrideKey("useReanalyze", "true");
    bool threw = false;
    try { Q4PlaySettings::loadForSelfplay(cfg); }
    catch(const StringError&) { threw = true; }
    testAssert(threw);
  }
  {
    ConfigParser cfg;
    cfg.overrideKey("estimateLeadProb", "0.5");
    bool threw = false;
    try { Q4PlaySettings::loadForSelfplay(cfg); }
    catch(const StringError&) { threw = true; }
    testAssert(threw);
  }
  {
    ConfigParser cfg;
    cfg.overrideKey("sekiForkHackProb", "0.1");
    bool threw = false;
    try { Q4PlaySettings::loadForSelfplay(cfg); }
    catch(const StringError&) { threw = true; }
    testAssert(threw);
  }
  {
    ConfigParser cfg;
    cfg.overrideKey("quoridorKomiRandomProb", "0.2");
    bool threw = false;
    try { Q4PlaySettings::loadForSelfplay(cfg); }
    catch(const StringError&) { threw = true; }
    testAssert(threw);
  }
}

void testSelfplayIntegration() {
  cout << "Testing Q4 selfplay game run and writing integration..." << endl;
  string modelPath = getTestModelPath();
  Logger logger;

  ConfigParser cfg;
  cfg.overrideKey("nnCacheSizePowerOfTwo", "16");
  cfg.overrideKey("nnMutexPoolSizePowerOfTwo", "12");
  cfg.overrideKey("maxVisits", "16");
  cfg.overrideKey("numSearchThreads", "1");
  cfg.overrideKey("winLossUtilityFactor", "1.0");
  cfg.overrideKey("cpuctExploration", "1.1");
  cfg.overrideKey("fpuReductionMax", "0.2");
  cfg.overrideKey("nnPolicyTemperature", "4.0");
  cfg.overrideKey("valueWeightExponent", "0.5");
  cfg.overrideKey("useUncertainty", "true");
  cfg.overrideKey("useLcbForSelection", "false");
  cfg.overrideKey("rootNoiseEnabled", "false");
  cfg.overrideKey("rootNumSymmetriesToSample", "1");
  cfg.overrideKey("staticScoreUtilityFactor", "0.0");
  cfg.overrideKey("dynamicScoreUtilityFactor", "0.0");
  cfg.overrideKey("useGraphSearch", "false");
  cfg.overrideKey("rootEndingBonusPoints", "0.0");
  cfg.overrideKey("rootPruneUselessMoves", "false");
  cfg.overrideKey("maxPlies", "30");
  cfg.overrideKey("initGamesWithPolicy", "false");
  cfg.overrideKey("cheapSearchProb", "0.0");
  cfg.overrideKey("cheapSearchVisits", "1");
  cfg.overrideKey("cheapSearchTargetWeight", "0.0");
  cfg.overrideKey("reduceVisits", "false");
  cfg.overrideKey("reduceVisitsThreshold", "0.5");
  cfg.overrideKey("reduceVisitsThresholdLookback", "1");
  cfg.overrideKey("reducedVisitsMin", "1");
  cfg.overrideKey("reducedVisitsWeight", "1.0");
  cfg.overrideKey("earlyForkGameProb", "0.0");
  cfg.overrideKey("earlyForkGameExpectedMoveProp", "0.0");
  cfg.overrideKey("forkGameProb", "0.0");
  cfg.overrideKey("forkGameMinChoices", "1");
  cfg.overrideKey("earlyForkGameMaxChoices", "1");
  cfg.overrideKey("forkGameMaxChoices", "1");
  cfg.overrideKey("sidePositionProb", "0.0");
  cfg.overrideKey("policySurpriseDataWeight", "0.0");
  cfg.overrideKey("valueSurpriseDataWeight", "0.0");
  cfg.overrideKey("scaleDataWeight", "1.0");
  cfg.overrideKey("logSearchInfo", "false");
  cfg.overrideKey("logMoves", "false");
  cfg.overrideKey("q4EliminationProb", "0.5"); // Force high chance of elimination
  addPopulationKeys(cfg, "0");

  Q4PlaySettings playSettings = Q4PlaySettings::loadForSelfplay(cfg);
  Q4GameRunner runner(cfg, playSettings, logger);

  Rand rand("testselfplay");
  NNEvaluator* nnEval = Setup::initializeNNEvaluator(
    "b1c32_q4", modelPath, "", cfg, logger, rand, 1, Q4NNConst::POS_LEN, Q4NNConst::POS_LEN,
    Setup::MaxBatchSizeRequest::explicitSize(16), true, false, Setup::SETUP_FOR_OTHER
  );

  Q4GameRunner::BotSpec botSpec;
  botSpec.botIdx = 0;
  botSpec.botName = "b1c32_q4";
  botSpec.nnEval = nnEval;
  botSpec.baseParams = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_OTHER);

  Q4FinishedGameData* gameData = runner.runGame(
    "test_seed", botSpec, nullptr, logger,
    []() { return false; }, nullptr, nullptr, nullptr, nullptr
  );

  testAssert(gameData != nullptr);
  testAssert(!gameData->valueTargetsByTurn.empty());
  testAssert(!gameData->comments.empty());
  testAssert(gameData->comments.size() == gameData->endHist.events.size());

  // Check move comment format
  for(size_t i = 0; i < gameData->endHist.events.size(); i++) {
    if(!gameData->endHist.events[i].isElimination) {
      testAssert(gameData->comments[i].find("v=[") != string::npos);
      testAssert(gameData->comments[i].find("visits=") != string::npos);
      testAssert(gameData->comments[i].find("cheap=") != string::npos);
    }
  }

  // Check record generation
  Q4Record rec;
  rec.rules = gameData->rules;
  rec.players.clear();
  for(int s = 0; s < 4; s++) {
    Q4PlayerInfo p;
    p.name = "Seat " + to_string(s);
    p.type = "selfplay";
    p.net = "b1c32_q4";
    rec.players.push_back(p);
  }
  rec.result = gameData->endHist.isDraw ? "Draw" : (to_string(gameData->endHist.winnerSeat + 1) + "+");
  rec.events = gameData->endHist.events;
  rec.comments = gameData->comments;
  rec.hasGameHash = true;
  rec.gameHash0 = gameData->gameHash.hash0;
  rec.gameHash1 = gameData->gameHash.hash1;
  string jsonLine = rec.toJsonLine();
  testAssert(!jsonLine.empty());
  Q4Record parsedRec = Q4Record::fromJsonLine(jsonLine);
  testAssert(parsedRec.events.size() == rec.events.size());
  testAssert(parsedRec.comments.size() == rec.comments.size());
  testAssert(parsedRec.hasGameHash);
  testAssert(parsedRec.gameHash0 == gameData->gameHash.hash0);
  testAssert(parsedRec.gameHash1 == gameData->gameHash.hash1);

  // Check data writer
  string scratchDir = "cpp/tests/scratch";
  if(!FileUtils::exists(scratchDir))
    MakeDir::make(scratchDir);
  string tmpDir = "cpp/tests/scratch/q4_selfplay_test";
  if(!FileUtils::exists(tmpDir))
    MakeDir::make(tmpDir);
  Q4TrainingDataWriter writer(tmpDir, 100, 0.0, 12345);
  writer.writeGame(*gameData);
  writer.flushIfNonempty();
  testAssert(writer.numGamesWritten() == 1);
  testAssert(writer.numRowsWritten() > 0);

  delete gameData;
  delete nnEval;
}

} // namespace

void Tests::runQ4SelfplayTests() {
  testActionToPolicySlot();
  testTDTargetsMath();
  testPlaySettingsValidation();
  testP2CompositionSampler();
  testSelfplayIntegration();
  cout << "All Q4 selfplay C++ tests passed!" << endl;
}
