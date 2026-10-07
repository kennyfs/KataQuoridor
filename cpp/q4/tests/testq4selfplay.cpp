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
  string jsonLine = rec.toJsonLine();
  testAssert(!jsonLine.empty());
  Q4Record parsedRec = Q4Record::fromJsonLine(jsonLine);
  testAssert(parsedRec.events.size() == rec.events.size());
  testAssert(parsedRec.comments.size() == rec.comments.size());

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
  testSelfplayIntegration();
  cout << "All Q4 selfplay C++ tests passed!" << endl;
}
