#include "../../game/board.h"
#include "../../core/global.h"
#include "../../core/datetime.h"
#include "../../core/fileutils.h"
#include "../../core/makedir.h"
#include "../../core/config_parser.h"
#include "../../core/timer.h"
#include "../../dataio/loadmodel.h"
#include "../../program/setup.h"
#include "../../command/commandline.h"
#include "../../core/test.h"
#include "../../main.h"

#include "../nn/q4nnconstants.h"
#include "../play/q4play.h"
#include "../play/q4playsettings.h"
#include "../play/q4selfplaymanager.h"
#include "../search/q4search.h"

#include <algorithm>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <ghc/filesystem.hpp>

using namespace std;
namespace gfs = ghc::filesystem;

static void createDirectoriesRecursive(const string& path) {
  gfs::path gfsPath(gfs::u8path(path));
  std::error_code ec;
  gfs::create_directories(gfsPath, ec);
  if(ec)
    throw StringError("Error creating directory: " + ec.message());
}

namespace {

struct ListedModel {
  string name;
  string file;
  ghc::filesystem::file_time_type time;
};

// All models under modelsDir, named as LoadModel::findLatestModel names them (the directory for a generic file name).
vector<ListedModel> listModels(const string& modelsDir) {
  static const vector<string> suffixes = {".bin.gz", ".bin", "model.txt.gz", "model.txt"};
  static const vector<string> genericNames = {"model.bin.gz", "model.bin", "model.txt.gz", "model.txt"};
  vector<ListedModel> models;
  for(const auto& entry : gfs::recursive_directory_iterator(gfs::u8path(modelsDir))) {
    const gfs::path& path = entry.path();
    if(!gfs::is_regular_file(path))
      continue;
    string filename = path.filename().u8string();
    bool acceptable = false;
    for(const string& suffix : suffixes)
      acceptable = acceptable || Global::isSuffix(filename, suffix);
    if(!acceptable)
      continue;
    ListedModel m;
    m.file = path.u8string();
    m.time = gfs::last_write_time(path);
    bool generic = std::find(genericNames.begin(), genericNames.end(), filename) != genericNames.end();
    m.name = generic ? path.parent_path().filename().u8string() : filename;
    models.push_back(m);
  }
  return models;
}

}  // namespace

static std::atomic<bool> sigReceived(false);
static std::atomic<bool> shouldStop(false);
static void signalHandler(int signal) {
  if(signal == SIGINT || signal == SIGTERM) {
    sigReceived.store(true);
    shouldStop.store(true);
  }
}

int MainCmds::q4selfplay(const vector<string>& args) {
  // NNEvaluator's transformer warmup builds a Duel Board, which needs the Duel Zobrist tables.
  Board::initHash();
  Rand seedRand;

  ConfigParser cfg;
  string modelsDir;
  string outputDir;
  string seedStr;
  int64_t maxGamesTotal = ((int64_t)1) << 62;
  int64_t maxValidGamesTotal = ((int64_t)1) << 62;
  int64_t maxRowsTotal = ((int64_t)1) << 62;

  try {
    KataGoCommandLine cmd("Generate Q4 training data via self play.");
    cmd.addConfigFileArg("", "");
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> modelsDirArg("", "models-dir", "Dir to poll and load models from", true, string(), "DIR");
    TCLAP::ValueArg<string> outputDirArg("", "output-dir", "Dir to output files", true, string(), "DIR");
    TCLAP::ValueArg<string> maxGamesTotalArg("", "max-games-total", "Terminate after this many games", false, string(), "NGAMES");
    TCLAP::ValueArg<string> maxValidGamesTotalArg("", "max-valid-games-total", "Terminate after this many valid non-discarded games", false, string(), "NGAMES");
    TCLAP::ValueArg<string> maxRowsTotalArg("", "max-rows-total", "Terminate after this many valid data rows", false, string(), "NROWS");
    TCLAP::ValueArg<string> seedArg("", "seed", "Base seed string for deterministic runs", false, string(), "SEED");
    cmd.add(modelsDirArg);
    cmd.add(outputDirArg);
    cmd.add(maxGamesTotalArg);
    cmd.add(maxValidGamesTotalArg);
    cmd.add(maxRowsTotalArg);
    cmd.add(seedArg);
    cmd.parseArgs(args);

    modelsDir = modelsDirArg.getValue();
    outputDir = outputDirArg.getValue();
    if(seedArg.isSet())
      seedStr = seedArg.getValue();
    string maxGamesTotalStr = maxGamesTotalArg.getValue();
    if(!maxGamesTotalStr.empty()) {
      bool suc = Global::tryStringToInt64(maxGamesTotalStr, maxGamesTotal);
      if(!suc || maxGamesTotal <= 0)
        throw StringError("-max-games-total must be a positive integer");
    }
    string maxValidGamesTotalStr = maxValidGamesTotalArg.getValue();
    if(!maxValidGamesTotalStr.empty()) {
      bool suc = Global::tryStringToInt64(maxValidGamesTotalStr, maxValidGamesTotal);
      if(!suc || maxValidGamesTotal <= 0)
        throw StringError("-max-valid-games-total must be a positive integer");
    }
    string maxRowsTotalStr = maxRowsTotalArg.getValue();
    if(!maxRowsTotalStr.empty()) {
      bool suc = Global::tryStringToInt64(maxRowsTotalStr, maxRowsTotal);
      if(!suc || maxRowsTotal <= 0)
        throw StringError("-max-rows-total must be a positive integer");
    }

    auto checkDirNonEmpty = [](const char* flag, const string& s) {
      if(s.length() <= 0)
        throw StringError("Empty directory specified for " + string(flag));
    };
    checkDirNonEmpty("models-dir", modelsDir);
    checkDirNonEmpty("output-dir", outputDir);

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  createDirectoriesRecursive(outputDir);
  createDirectoriesRecursive(modelsDir);

  Logger logger(&cfg);

  const int numGameThreads = cfg.getInt("numGameThreads", 1, 16384);
  const int maxDataQueueSize = cfg.getInt("maxDataQueueSize", 1, 16384);
  const int maxRowsPerTrainFile = cfg.getInt("maxRowsPerTrainFile", 1, 1000000);
  const double firstFileRandMinProp = cfg.contains("firstFileRandMinProp") ? cfg.getDouble("firstFileRandMinProp", 0.0, 1.0) : 0.0;
  const int64_t logGamesEvery = cfg.getInt64("logGamesEvery", 1, 1000000000);
  const bool switchNetsMidGame = cfg.getBool("switchNetsMidGame");

  string gameSeedBase;
  if(!seedStr.empty())
    gameSeedBase = seedStr;
  else if(cfg.contains("gameSeedBase"))
    gameSeedBase = cfg.getString("gameSeedBase");
  else if(cfg.contains("seed"))
    gameSeedBase = cfg.getString("seed");
  else
    gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  SearchParams baseParams = Setup::loadSingleParams(cfg, Setup::SETUP_FOR_OTHER);
  Q4S::Search::checkParams(baseParams);

  if(cfg.contains("maxGamesTotal")) {
    int64_t m = cfg.getInt64("maxGamesTotal", 1, ((int64_t)1) << 62);
    maxGamesTotal = std::min(maxGamesTotal, m);
  }
  if(cfg.contains("maxValidGamesTotal")) {
    int64_t v = cfg.getInt64("maxValidGamesTotal", 1, ((int64_t)1) << 62);
    maxValidGamesTotal = std::min(maxValidGamesTotal, v);
  }
  if(cfg.contains("maxRowsTotal")) {
    int64_t r = cfg.getInt64("maxRowsTotal", 1, ((int64_t)1) << 62);
    maxRowsTotal = std::min(maxRowsTotal, r);
  }

  Q4Play::Q4PlaySettings playSettings = Q4Play::Q4PlaySettings::loadForSelfplay(cfg);
  const Q4Play::Q4PopulationSettings popSettings = playSettings.population;
  Q4Play::Q4GameRunner* gameRunner = new Q4Play::Q4GameRunner(cfg, gameSeedBase + ":gameInit", playSettings, logger);
  bool autoCleanupAllButLatestIfUnused = true;
  Q4Play::Q4SelfPlayManager* manager = new Q4Play::Q4SelfPlayManager(
    maxDataQueueSize, &logger, logGamesEvery, autoCleanupAllButLatestIfUnused
  );

  // Snapshot nets of population games: the models q4PopulationSnapshotAges exports back, kept in a second manager (no
  // data-writing loops, as KataGo keeps models whose games write no data), refreshed when the current net changes.
  Q4Play::Q4SelfPlayManager* snapshotManager = new Q4Play::Q4SelfPlayManager(
    1, &logger, 1, false
  );
  std::mutex wantedSnapshotsMutex;
  vector<string> wantedSnapshots;
  auto refreshSnapshots = [&](const string& currentName) {
    if(!popSettings.usesSnapshots())
      return;
    vector<ListedModel> models = listModels(modelsDir);
    const ListedModel* current = nullptr;
    for(const ListedModel& m : models)
      if(m.name == currentName)
        current = &m;
    vector<const ListedModel*> older;
    if(current != nullptr) {
      for(const ListedModel& m : models)
        if(m.name != currentName && m.time < current->time)
          older.push_back(&m);
    }
    std::sort(older.begin(), older.end(), [](const ListedModel* a, const ListedModel* b) {
      return a->time != b->time ? a->time > b->time : a->name > b->name;
    });
    vector<const ListedModel*> chosen;
    if(!older.empty()) {
      for(int age : popSettings.snapshotAges)
        chosen.push_back(older[std::min((size_t)age, older.size()) - 1]);
    }
    vector<string> wanted;
    for(const ListedModel* m : chosen) {
      if(std::find(wanted.begin(), wanted.end(), m->name) != wanted.end())
        continue;
      if(!snapshotManager->hasModel(m->name)) {
        ConfigParser snapCfg = cfg;
        snapCfg.overrideKey("nnCacheSizePowerOfTwo", Global::intToString(popSettings.snapshotCacheSizePowerOfTwo));
        Rand snapRand(gameSeedBase + ":snapshot:" + m->name);
        NNEvaluator* snapEval = Setup::initializeNNEvaluator(
          m->name, m->file, "", snapCfg, logger, snapRand, cfg.getInt("numSearchThreads") * numGameThreads,
          Q4NNConst::POS_LEN, Q4NNConst::POS_LEN, Setup::MaxBatchSizeRequest::requireFromConfig(),
          true, false, Setup::SETUP_FOR_OTHER
        );
        snapshotManager->loadModelNoDataWritingLoop(snapEval, nullptr, nullptr);
        logger.write("Loaded snapshot neural net " + m->name + " from: " + m->file);
      }
      wanted.push_back(m->name);
    }
    {
      std::lock_guard<std::mutex> lock(wantedSnapshotsMutex);
      wantedSnapshots = wanted;
    }
    snapshotManager->unloadUnusedModelsNotIn(wanted);
    string list;
    for(const string& w : wanted)
      list += " " + w;
    logger.write("Snapshots for population games (current " + currentName + "):" + (list.empty() ? " none" : list));
  };
  Q4Play::Q4SnapshotSource snapshotSource;
  snapshotSource.acquireAll = [&]() {
    vector<string> names;
    {
      std::lock_guard<std::mutex> lock(wantedSnapshotsMutex);
      names = wantedSnapshots;
    }
    std::sort(names.begin(), names.end());
    vector<Q4Play::Q4SnapshotRef> refs;
    for(const string& n : names) {
      NNEvaluator* eval = snapshotManager->acquireModel(n);
      if(eval != nullptr)
        refs.push_back({n, eval});
    }
    return refs;
  };
  snapshotSource.releaseAll = [&](const vector<Q4Play::Q4SnapshotRef>& refs) {
    for(const Q4Play::Q4SnapshotRef& r : refs)
      snapshotManager->release(r.name);
  };

  Setup::initializeSession(cfg);

  logger.write("Loaded all config stuff, starting Q4 self play");
  if(!logger.isLoggingToStdout())
    cout << "Loaded all config stuff, starting Q4 self play" << endl;

  ClockTimer selfplayTimer;

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  auto loadLatestNeuralNetIntoManager =
    [&manager, maxRowsPerTrainFile, firstFileRandMinProp, &refreshSnapshots,
     &modelsDir, &outputDir, &logger, &cfg, numGameThreads, gameSeedBase](const string* lastNetName) -> bool {

    string modelName;
    string modelFile;
    string modelDir;
    time_t modelTime;
    bool foundModel = LoadModel::findLatestModel(modelsDir, logger, modelName, modelFile, modelDir, modelTime);

    if(!foundModel || (lastNetName != nullptr && *lastNetName == modelName))
      return false;
    if(modelName == "random" && lastNetName != nullptr && *lastNetName != "random") {
      logger.write("WARNING: " + *lastNetName + " was previous model, but now no model found. Continuing with prev model");
      return false;
    }

    logger.write("Found new neural net " + modelName);

    const int expectedConcurrentEvals = cfg.getInt("numSearchThreads") * numGameThreads;
    const bool defaultRequireExactNNLen = true;
    const bool disableFP16 = false;
    const string expectedSha256 = "";

    Rand rand(gameSeedBase + ":manager:" + modelName);
    NNEvaluator* nnEval = Setup::initializeNNEvaluator(
      modelName, modelFile, expectedSha256, cfg, logger, rand, expectedConcurrentEvals,
      Q4NNConst::POS_LEN, Q4NNConst::POS_LEN, Setup::MaxBatchSizeRequest::requireFromConfig(),
      defaultRequireExactNNLen, disableFP16, Setup::SETUP_FOR_OTHER
    );
    logger.write("Loaded latest neural net " + modelName + " from: " + modelFile);

    string modelOutputDir = outputDir + "/" + modelName;
    string recordsOutputDir = modelOutputDir + "/records";
    string tdataOutputDir = modelOutputDir + "/tdata";

    int maxTries = 5;
    for(int i = 0; i < maxTries; i++) {
      bool success = false;
      try {
        createDirectoriesRecursive(modelOutputDir);
        createDirectoriesRecursive(recordsOutputDir);
        createDirectoriesRecursive(tdataOutputDir);
        success = true;
      }
      catch(const StringError& e) {
        logger.write(string("WARNING, error making directories, trying again shortly: ") + e.what());
        success = false;
      }

      if(success)
        break;
      else {
        if(i == maxTries - 1) {
          logger.write("ERROR: Could not make selfplay model directories");
          return false;
        }
        double sleepTime = 5.0 + rand.nextDouble() * 10.0;
        std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
      }
    }

    {
      ofstream out;
      FileUtils::open(out, modelOutputDir + "/selfplay-" + Global::uint64ToHexString(rand.nextUInt64()) + ".cfg");
      out << cfg.getContents();
      out.close();
    }

    auto* tdataWriter = new Q4Play::Q4TrainingDataWriter(
      tdataOutputDir, maxRowsPerTrainFile, firstFileRandMinProp, rand.nextUInt64()
    );
    ofstream* recordsOut = new ofstream();
    FileUtils::open(*recordsOut, recordsOutputDir + "/" + Global::uint64ToHexString(rand.nextUInt64()) + ".q4.jsonl");

    logger.write("Model loading loop loaded new neural net " + nnEval->getModelName());
    manager->loadModelAndStartDataWriting(nnEval, tdataWriter, recordsOut);
    refreshSnapshots(modelName);
    return true;
  };

  {
    bool success = loadLatestNeuralNetIntoManager(nullptr);
    if(!success)
      throw StringError("Could not load initial neural net from " + modelsDir);
  }

  cfg.warnUnusedKeys(cerr, &logger);

  std::atomic<int64_t> numGamesStarted(0);
  std::atomic<int64_t> numValidGamesFinished(0);
  std::atomic<int64_t> numDataRowsEnqueued(0);
  auto* forkData = new Q4Play::Q4ForkData();

  auto gameLoop = [
    &gameRunner,
    &manager,
    &logger,
    switchNetsMidGame,
    &numGamesStarted,
    &numValidGamesFinished,
    &numDataRowsEnqueued,
    &forkData,
    maxGamesTotal,
    maxValidGamesTotal,
    maxRowsTotal,
    &baseParams,
    &gameSeedBase,
    &snapshotSource
  ](int threadIdx) {
    auto shouldStopFunc = []() noexcept {
      return shouldStop.load();
    };
    WaitableFlag* shouldPause = nullptr;

    string prevModelName;
    Rand thisLoopSeedRand(gameSeedBase + ":thread:" + Global::intToString(threadIdx));
    while(true) {
      if(shouldStop.load())
        break;
      NNEvaluator* nnEval = manager->acquireLatest();
      testAssert(nnEval != nullptr);

      if(prevModelName != nnEval->getModelName()) {
        prevModelName = nnEval->getModelName();
        logger.write("Game loop thread " + Global::intToString(threadIdx) + " starting game on new neural net: " + prevModelName);
      }

      std::function<NNEvaluator*()> checkForNewNNEval = [&manager, &nnEval, &prevModelName, &logger, threadIdx]() -> NNEvaluator* {
        NNEvaluator* newNNEval = manager->acquireLatest();
        testAssert(newNNEval != nullptr);
        if(newNNEval == nnEval) {
          manager->release(newNNEval);
          return nullptr;
        }
        manager->release(nnEval);
        nnEval = newNNEval;
        prevModelName = nnEval->getModelName();
        logger.write("Game loop thread " + Global::intToString(threadIdx) + " changing midgame to new net: " + prevModelName);
        return nnEval;
      };

      Q4Play::Q4FinishedGameData* gameData = nullptr;
      bool canStart = true;
      if(numValidGamesFinished.load(std::memory_order_relaxed) >= maxValidGamesTotal)
        canStart = false;
      if(numDataRowsEnqueued.load(std::memory_order_relaxed) >= maxRowsTotal)
        canStart = false;
      if(maxGamesTotal < ((int64_t)1 << 60)) {
        int64_t gIdx = numGamesStarted.fetch_add(1, std::memory_order_acq_rel);
        if(gIdx >= maxGamesTotal)
          canStart = false;
      }

      if(canStart) {
        if(maxGamesTotal >= ((int64_t)1 << 60))
          numGamesStarted.fetch_add(1, std::memory_order_relaxed);

        manager->countOneGameStarted(nnEval);
        Q4Play::Q4GameRunner::BotSpec botSpec;
        botSpec.botIdx = 0;
        botSpec.botName = nnEval->getModelName();
        botSpec.nnEval = nnEval;
        botSpec.baseParams = baseParams;

        string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
        gameData = gameRunner->runGame(
          seed, botSpec, forkData, logger,
          shouldStopFunc,
          shouldPause,
          (switchNetsMidGame ? checkForNewNNEval : nullptr),
          nullptr,
          nullptr,
          &snapshotSource
        );
      }

      bool shouldContinue = (gameData != nullptr);
      if(gameData != nullptr) {
        manager->countQ4GameResult(nnEval, *gameData);
        if(gameData->hitTurnLimit) {
          manager->countOneGameHitCutoff(nnEval, (int64_t)gameData->endHist.plies);
          delete gameData;
        }
        else {
          int64_t rows = 0;
          for(float w : gameData->targetWeightByTurn)
            rows += (int64_t)std::round(w);
          for(const auto* sp : gameData->sidePositions)
            rows += (int64_t)std::round(sp->targetWeight);
          numValidGamesFinished.fetch_add(1, std::memory_order_relaxed);
          numDataRowsEnqueued.fetch_add(rows, std::memory_order_relaxed);
          manager->enqueueDataToWrite(nnEval, gameData);
        }
      }

      manager->release(nnEval);

      if(!shouldContinue)
        break;
    }

    logger.write("Game loop thread " + Global::intToString(threadIdx) + " terminating");
  };

  auto gameLoopProtected = [&logger, &gameLoop](int threadIdx) {
    Logger::logThreadUncaught("game loop", &logger, [&]() { gameLoop(threadIdx); });
  };

  std::mutex modelLoadMutex;
  std::condition_variable modelLoadSleepVar;
  auto modelLoadLoop = [&modelLoadMutex, &modelLoadSleepVar, &logger, &manager, &loadLatestNeuralNetIntoManager]() {
    logger.write("Model loading loop thread starting");
    while(true) {
      if(shouldStop.load())
        break;
      string lastNetName = manager->getLatestModelName();
      loadLatestNeuralNetIntoManager(&lastNetName);

      if(shouldStop.load())
        break;

      std::unique_lock<std::mutex> lock(modelLoadMutex);
      modelLoadSleepVar.wait_for(lock, std::chrono::seconds(20), []() { return shouldStop.load(); });
    }
    logger.write("Model loading loop thread terminating");
  };
  auto modelLoadLoopProtected = [&logger, &modelLoadLoop]() {
    Logger::logThreadUncaught("model load loop", &logger, modelLoadLoop);
  };

  vector<std::thread> threads;
  threads.reserve(numGameThreads);
  for(int i = 0; i < numGameThreads; i++) {
    threads.emplace_back(gameLoopProtected, i);
  }
  std::thread modelLoadLoopThread(modelLoadLoopProtected);

  for(size_t i = 0; i < threads.size(); i++)
    threads[i].join();

  shouldStop.store(true);
  {
    std::lock_guard<std::mutex> lock(modelLoadMutex);
    modelLoadSleepVar.notify_all();
  }
  modelLoadLoopThread.join();

  delete manager;
  delete snapshotManager;
  delete gameRunner;
  delete forkData;

  logger.write("Total games started: " + Global::int64ToString(numGamesStarted.load(std::memory_order_relaxed)));
  logger.write("Total valid games: " + Global::int64ToString(numValidGamesFinished.load(std::memory_order_relaxed)));
  logger.write("Total data rows: " + Global::int64ToString(numDataRowsEnqueued.load(std::memory_order_relaxed)));
  logger.write("Total selfplay runtime (seconds): " + Global::doubleToString(selfplayTimer.getSeconds()));

  return 0;
}
