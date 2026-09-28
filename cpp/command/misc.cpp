#include "../core/global.h"
#include "../core/fileutils.h"
#include "../core/fancymath.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/parallel.h"
#include "../core/timer.h"
#include "../core/test.h"
#include "../dataio/sgf.h"
#include "../dataio/poswriter.h"
#include "../dataio/files.h"
#include "../dataio/trainingwrite.h"
#include "../dataio/numpywrite.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../search/asyncbot.h"
#include "../program/setup.h"
#include "../program/playutils.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../tests/tests.h"
#include "../main.h"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <mutex>
#include <thread>

using namespace std;

static std::atomic<bool> sigReceived(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM)
    sigReceived.store(true);
}

int MainCmds::printclockinfo(const vector<string>& args) {
  (void)args;
#ifdef OS_IS_WINDOWS
  cout << "Does nothing on windows, disabled" << endl;
#endif
#ifdef OS_IS_UNIX_OR_APPLE
  cout << "Tick unit in seconds: " << std::chrono::steady_clock::period::num << " / " <<  std::chrono::steady_clock::period::den << endl;
  cout << "Ticks since epoch: " << std::chrono::steady_clock::now().time_since_epoch().count() << endl;
#endif
  return 0;
}

int MainCmds::sampleinitializations(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  int numToGen;
  bool evaluate;
  try {
    KataGoCommandLine cmd("View startposes");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<int> numToGenArg("","num","Num to gen",false,1,"N");
    TCLAP::SwitchArg evaluateArg("","evaluate","Print out values and scores on the inited poses");
    cmd.add(numToGenArg);
    cmd.add(evaluateArg);
    cmd.parseArgs(args);
    numToGen = numToGenArg.getValue();
    evaluate = evaluateArg.getValue();

    cmd.getConfigAllowEmpty(cfg);
    if(cfg.getFileName() != "")
      modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  NNEvaluator* nnEval = NULL;
  if(cfg.getFileName() != "") {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    {
      Setup::initializeSession(cfg);
      const int expectedConcurrentEvals = params.numThreads;
      const bool defaultRequireExactNNLen = false;
      const bool disableFP16 = false;
      const string expectedSha256 = "";
      nnEval = Setup::initializeNNEvaluator(
        modelFile,modelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
        Board::MAX_LEN,Board::MAX_LEN,Setup::MaxBatchSizeRequest::fromConcurrency(),defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_GTP
      );
    }
    logger.write("Loaded neural net");
  }

  AsyncBot* evalBot;
  {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_DISTRIBUTED);
    params.maxVisits = 20;
    params.numThreads = 1;
    string seed = Global::uint64ToString(rand.nextUInt64());
    evalBot = new AsyncBot(params, nnEval, &logger, seed);
  }

  //Play no moves in game, since we're sampling initializations
  cfg.overrideKey("maxMovesPerGame","0");

  const bool isDistributed = false;
  PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, isDistributed);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);

  for(int i = 0; i<numToGen; i++) {
    string seed = Global::uint64ToString(rand.nextUInt64());
    MatchPairer::BotSpec botSpec;
    botSpec.botIdx = 0;
    botSpec.botName = "";
    botSpec.nnEval = nnEval;
    botSpec.baseParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_DISTRIBUTED);

    FinishedGameData* data = gameRunner->runGame(
      seed,
      botSpec,
      botSpec,
      NULL,
      NULL,
      logger,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr
    );

    cout << data->startHist.rules.toString() << endl;
    Board::printBoard(cout, data->startBoard, Board::NULL_LOC, &(data->startHist.moveHistory));
    cout << endl;
    if(evaluate) {
      evalBot->setPosition(data->startPla, data->startBoard, data->startHist);
      evalBot->genMoveSynchronous(data->startPla,TimeControls());
      ReportedSearchValues values = evalBot->getSearchStopAndWait()->getRootValuesRequireSuccess();
      cout << "Winloss: " << values.winLossValue << endl;
      cout << "Lead: " << values.lead << endl;
    }

    delete data;
  }

  delete gameRunner;
  delete evalBot;
  if(nnEval != NULL)
    delete nnEval;

  ScoreValue::freeTables();
  return 0;
}

int MainCmds::evalrandominits(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  try {
    KataGoCommandLine cmd("Eval random inits");
    cmd.addConfigFileArg("","");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    cmd.getConfigAllowEmpty(cfg);
    if(cfg.getFileName() != "")
      modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);

  NNEvaluator* nnEval = NULL;
  if(cfg.getFileName() != "") {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    {
      Setup::initializeSession(cfg);
      const int expectedConcurrentEvals = params.numThreads;
      const bool defaultRequireExactNNLen = false;
      const bool disableFP16 = false;
      const string expectedSha256 = "";
      nnEval = Setup::initializeNNEvaluator(
        modelFile,modelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
        Board::MAX_LEN,Board::MAX_LEN,Setup::MaxBatchSizeRequest::fromConcurrency(),defaultRequireExactNNLen,disableFP16,
        Setup::SETUP_FOR_GTP
      );
    }
    logger.write("Loaded neural net");
  }

  Search* evalBot;
  {
    SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_DISTRIBUTED);
    params.maxVisits = 40;
    params.numThreads = 1;
    string seed = Global::uint64ToString(rand.nextUInt64());
    evalBot = new Search(params, nnEval, &logger, seed);
  }

  Rand gameRand;
  while(true) {
    Board board(19,19);
    Player pla = P_BLACK;
    Rules rules = Rules::parseRules("japanese");
    //Keep pass-alive computations (e.g. endGameIfAllPassAlive below) consistent with how the bot's
    //own searches are performing them.
    BoardHistory hist(board,pla,rules,0,evalBot->getRootHist().modes);
    int numInitialMovesToPlay = (int)gameRand.nextUInt(200);
    double temperature = 1.0;
    for(int i = 0; i<numInitialMovesToPlay; i++) {
      NNResultBuf buf;
      Loc loc = PlayUtils::getGameInitializationMove(evalBot, evalBot, board, hist, pla, buf, gameRand, temperature);

      assert(hist.isLegal(board,loc,pla));
      hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
      pla = getOpp(pla);

      hist.endGameIfAllPassAlive(board);
      if(hist.isGameFinished)
        break;
    }

    evalBot->setPosition(pla,board,hist);
    evalBot->runWholeSearch(pla);
    ReportedSearchValues values = evalBot->getRootValuesRequireSuccess();
    cout << numInitialMovesToPlay << "," << values.winLossValue << "," << values.lead << endl;
  }
  delete evalBot;
  if(nnEval != NULL)
    delete nnEval;

  ScoreValue::freeTables();
  return 0;
}

int MainCmds::searchentropyanalysis(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  string boardSizeDataset;
  try {
    KataGoCommandLine cmd("Analyze search entropy across datasets");
    cmd.addConfigFileArg(KataGoCommandLine::defaultGtpConfigFileName(),"gtp_example.cfg");
    cmd.addModelFileArg();
    TCLAP::ValueArg<string> boardSizeDatasetArg("","boardsizedataset", "Dataset to analyze (9,13,19,10x14,rectangle)", true, "", "SIZE");
    cmd.add(boardSizeDatasetArg);
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    modelFile = cmd.getModelFile();
    boardSizeDataset = boardSizeDatasetArg.getValue();
    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand rand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);
  logger.write("Search Entropy Analysis");
  logger.write("Model: " + modelFile);
  logger.write("Dataset: " + boardSizeDataset);

  SearchParams params = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
  NNEvaluator* nnEval = NULL;
  {
    Setup::initializeSession(cfg);
    const int expectedConcurrentEvals = params.numThreads;
    const bool defaultRequireExactNNLen = false;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,rand,expectedConcurrentEvals,
      Board::MAX_LEN,Board::MAX_LEN,Setup::MaxBatchSizeRequest::fromConcurrency(),defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_GTP
    );
  }
  logger.write("Loaded neural net");

  Search* bot;
  {
    SearchParams searchParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_GTP);
    string seed = Global::uint64ToString(rand.nextUInt64());
    bot = new Search(searchParams, nnEval, &logger, seed);
  }

  vector<string> sgfData;
  if(boardSizeDataset == "9")
    sgfData = TestCommon::getMultiGameSize9Data();
  else if(boardSizeDataset == "13")
    sgfData = TestCommon::getMultiGameSize13Data();
  else if(boardSizeDataset == "19")
    sgfData = TestCommon::getMultiGameSize19Data();
  else if(boardSizeDataset == "10x14")
    sgfData = TestCommon::getMultiGameSize10x14Data();
  else if(boardSizeDataset == "rectangle")
    sgfData = TestCommon::getMultiGameRectangleData();
  else {
    throw StringError("Unknown dataset to test gpu error on: " + boardSizeDataset);
  }

  // Build list of all (sgfIdx, turnIdx) pairs, then deterministically shuffle
  vector<std::pair<int,int>> positions;
  {
    vector<std::unique_ptr<CompactSgf>> sgfObjs;
    for(const string& sgf : sgfData)
      sgfObjs.push_back(CompactSgf::parse(sgf));
    for(int sgfIdx = 0; sgfIdx < (int)sgfObjs.size(); sgfIdx++) {
      for(int turnIdx = 0; turnIdx < (int)sgfObjs[sgfIdx]->moves.size(); turnIdx++) {
        positions.push_back({sgfIdx, turnIdx});
      }
    }
  }
  {
    Rand shuffleRand("searchentropyanalysis_shuffle_seed");
    shuffleRand.shuffle(positions);
  }
  logger.write("Total positions to search: " + Global::intToString((int)positions.size()));

  vector<double> searchEntropies;
  vector<double> searchSurprises;
  int numPositions = 0;

  auto printRunningStats = [&]() {
    cout << "Searched numPositions: " << numPositions << endl;
    if(numPositions > 0) {
      double entropyMean = 0.0;
      for(double v : searchEntropies) entropyMean += v;
      entropyMean /= numPositions;
      double entropyVar = 0.0;
      for(double v : searchEntropies) { double d = v - entropyMean; entropyVar += d * d; }
      entropyVar /= numPositions;
      cout << "Mean search entropy: " << entropyMean << " standard deviation: " << sqrt(entropyVar) << endl;

      double surpriseMean = 0.0;
      for(double v : searchSurprises) surpriseMean += v;
      surpriseMean /= numPositions;
      double surpriseVar = 0.0;
      for(double v : searchSurprises) { double d = v - surpriseMean; surpriseVar += d * d; }
      surpriseVar /= numPositions;
      cout << "Mean search surprise: " << surpriseMean << " standard deviation: " << sqrt(surpriseVar) << endl;
    }
  };

  for(const auto& pos : positions) {
    int sgfIdx = pos.first;
    int turnIdx = pos.second;

    std::unique_ptr<CompactSgf> sgfObj = CompactSgf::parse(sgfData[sgfIdx]);

    Board board;
    Player pla;
    BoardHistory hist;
    Rules initialRules;
    sgfObj->setupInitialBoardAndHist(initialRules, board, pla, hist, bot->getRootHist().modes);

    for(int i = 0; i < turnIdx; i++) {
      Loc moveLoc = sgfObj->moves[i].loc;
      if(moveLoc != Board::NULL_LOC && hist.isLegal(board, moveLoc, pla)) {
        hist.makeBoardMoveAssumeLegal(board, moveLoc, pla, NULL);
        pla = getOpp(pla);
      }
    }

    if(!hist.isGameFinished) {
      bot->setPosition(pla, board, hist);
      bot->runWholeSearch(pla);

      double surprise, searchEntropy, policyEntropy;
      if(bot->getPolicySurpriseAndEntropy(surprise, searchEntropy, policyEntropy)) {
        searchEntropies.push_back(searchEntropy);
        searchSurprises.push_back(surprise);
        numPositions++;
        if(numPositions % 10 == 0)
          printRunningStats();
      }
    }
  }

  cout << modelFile << endl;
  cout << "Dataset: " << boardSizeDataset << endl;
  printRunningStats();

  delete bot;
  delete nnEval;
  ScoreValue::freeTables();
  return 0;
}

//One CSV row per recorded turn of the game, containing the stats that drive surprise-based selection of
//positions (policy surprise, value surprise) along with the per-game and per-turn covariates needed to
//analyze what they vary with.
static void writeSurpriseStatsRows(ostream& out, int64_t gameIdx, const FinishedGameData& data) {
  size_t numMoves = data.targetWeightByTurn.size();
  testAssert(data.hasFullData);
  testAssert(data.policySurpriseByTurn.size() == numMoves);
  testAssert(data.policyEntropyByTurn.size() == numMoves);
  testAssert(data.searchEntropyByTurn.size() == numMoves);
  testAssert(data.valueSurpriseByTurn.size() == numMoves);
  testAssert(data.wasCheapSearchByTurn.size() == numMoves);
  testAssert(data.nnRawStatsByTurn.size() == numMoves);
  testAssert(data.policyTargetsByTurn.size() == numMoves);
  testAssert(data.reanalysisByTurn.size() == numMoves);
  testAssert(data.whiteValueTargetsByTurn.size() == numMoves + 1);

  //Last entry of the value targets is the actual game outcome.
  const ValueTargets& outcome = data.whiteValueTargetsByTurn[numMoves];
  double pdaWhite =
    data.playoutDoublingAdvantagePla == P_WHITE ? data.playoutDoublingAdvantage :
    data.playoutDoublingAdvantagePla == P_BLACK ? -data.playoutDoublingAdvantage : 0.0;
  //Turn number of the first recorded turn, counting handicap/policy-init/fork setup moves and any initial
  //turn number offset of positions begun from sgfs, so that turn number is comparable across game modes.
  int64_t startTurnIdx = data.startHist.initialTurnNumber + (int64_t)data.startHist.moveHistory.size();

  for(size_t t = 0; t<numMoves; t++) {
    const ValueTargets& mcts = data.whiteValueTargetsByTurn[t];
    const NNRawStats& raw = data.nnRawStatsByTurn[t];
    const ReanalysisData& rean = data.reanalysisByTurn[t];
    out << gameIdx
        << "," << data.gameHash.toString()
        << "," << data.startBoard.x_size
        << "," << data.startBoard.y_size
        << "," << data.startHist.rules.komi
        << "," << pdaWhite
        << "," << data.numExtraBlack
        << "," << data.mode
        << "," << (data.hitTurnLimit ? 1 : 0)
        << "," << numMoves
        << "," << outcome.win
        << "," << outcome.loss
        << "," << outcome.noResult
        << "," << outcome.score
        << "," << (startTurnIdx + (int64_t)t)
        << "," << t
        << "," << (data.wasCheapSearchByTurn[t] ? 1 : 0)
        << "," << data.policyTargetsByTurn[t].unreducedNumVisits
        << "," << data.targetWeightByTurn[t]
        << "," << data.policySurpriseByTurn[t]
        << "," << data.valueSurpriseByTurn[t]
        << "," << data.policyEntropyByTurn[t]
        << "," << data.searchEntropyByTurn[t]
        << "," << raw.whiteWinLoss
        << "," << raw.whiteScoreMean
        << "," << mcts.win
        << "," << mcts.loss
        << "," << mcts.noResult
        << "," << mcts.score
        << "," << (rean.wasReanalyzed ? 1 : 0)
        << "," << rean.selectionPolicySurprise
        << "," << rean.selectionValueSurprise
        << "," << rean.originalNumVisits
        << "\n";
  }
}

int MainCmds::selfplaysurprisedump(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  string outputFile;
  int64_t numGamesTotal;
  bool reanalyzeAll;
  try {
    KataGoCommandLine cmd("Run selfplay games with a fixed model and dump per-position policy and value surprise stats to csv");
    cmd.addConfigFileArg("","selfplay config file");
    cmd.addModelFileArg();
    cmd.addOverrideConfigArg();

    TCLAP::ValueArg<string> outputArg("","output","Csv file to write one row per recorded turn to",true,string(),"FILE");
    TCLAP::ValueArg<int> numGamesArg("","num-games","Number of games to run",true,0,"N");
    TCLAP::SwitchArg reanalyzeAllArg("","reanalyze-all","Instead of disabling reanalysis, reanalyze every cheap-search position, so each such row records both the cheap search's (before) and the full search's (after) surprise stats");
    cmd.add(outputArg);
    cmd.add(numGamesArg);
    cmd.add(reanalyzeAllArg);
    cmd.parseArgs(args);
    outputFile = outputArg.getValue();
    numGamesTotal = numGamesArg.getValue();
    reanalyzeAll = reanalyzeAllArg.getValue();
    if(numGamesTotal <= 0)
      throw StringError("-num-games must be positive");

    cmd.getConfig(cfg);
    modelFile = cmd.getModelFile();
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Rand seedRand;

  const bool logToStdoutDefault = true;
  Logger logger(&cfg, logToStdoutDefault);
  logger.write("Selfplay surprise stats dump");
  logger.write("Model: " + modelFile);

  const int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  const SearchParams baseParams = Setup::loadSingleParams(cfg,Setup::SETUP_FOR_OTHER);
  const bool isDistributed = false;
  PlaySettings playSettings = PlaySettings::loadForSelfplay(cfg, isDistributed);
  if(reanalyzeAll) {
    //Reanalyze every cheap-search position, so that every cheap-search row records both the cheap search's
    //surprise stats (the ones that would drive reanalysis selection, preserved as origPolicySurprise and
    //origValueSurprise) and the full search's stats, without any selection bias in which positions get both.
    //Note that the post-reanalysis value surprise of a turn also absorbs the value changes of other reanalyzed
    //turns after it, via the smoothed forward-outcome track, so unlike policy surprise it is not a pure
    //cheap-vs-full comparison of this turn's search alone.
    logger.write("Reanalyzing all cheap-search positions to record before and after surprise stats");
    playSettings.useReanalyze = true;
    playSettings.reanalyzeProp = 1.0;
  }
  else {
    //Force no reanalysis, so that every turn's recorded stats are those of the search the game actually played
    //with - exactly the stats that would drive reanalysis selection - rather than partly overwritten by
    //reanalysis searches.
    if(playSettings.useReanalyze && playSettings.reanalyzeProp > 0.0)
      logger.write("Config has reanalyzeProp > 0, forcing it to 0 so that recorded stats are the pre-reanalysis ones");
    playSettings.reanalyzeProp = 0.0;
  }

  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);

  const int minBoardXSizeUsed = gameRunner->getGameInitializer()->getMinBoardXSize();
  const int minBoardYSizeUsed = gameRunner->getGameInitializer()->getMinBoardYSize();
  const int maxBoardXSizeUsed = gameRunner->getGameInitializer()->getMaxBoardXSize();
  const int maxBoardYSizeUsed = gameRunner->getGameInitializer()->getMaxBoardYSize();

  Setup::initializeSession(cfg);
  NNEvaluator* nnEval;
  {
    const int expectedConcurrentEvals = cfg.getInt("numSearchThreads") * numGameThreads;
    const bool defaultRequireExactNNLen = minBoardXSizeUsed == maxBoardXSizeUsed && minBoardYSizeUsed == maxBoardYSizeUsed;
    const bool disableFP16 = false;
    const string expectedSha256 = "";
    nnEval = Setup::initializeNNEvaluator(
      modelFile,modelFile,expectedSha256,cfg,logger,seedRand,expectedConcurrentEvals,
      maxBoardXSizeUsed,maxBoardYSizeUsed,Setup::MaxBatchSizeRequest::requireFromConfig(),defaultRequireExactNNLen,disableFP16,
      Setup::SETUP_FOR_OTHER
    );
  }
  logger.write("Loaded neural net");

  cfg.warnUnusedKeys(cerr,&logger);

  ofstream out;
  FileUtils::open(out,outputFile);
  out.precision(9);
  out << "gameIdx,gameHash,xSize,ySize,komi,pdaWhite,numExtraBlack,mode,hitTurnLimit,gameNumMoves"
      << ",outcomeWhiteWin,outcomeWhiteLoss,outcomeWhiteNoResult,outcomeWhiteScore"
      << ",turnIdx,turnAfterStart,wasCheapSearch,unreducedNumVisits,targetWeight"
      << ",policySurprise,valueSurprise,policyEntropy,searchEntropy"
      << ",rawNNWhiteWinLoss,rawNNWhiteScoreMean,mctsWhiteWin,mctsWhiteLoss,mctsWhiteNoResult,mctsWhiteScore"
      << ",wasReanalyzed,origPolicySurprise,origValueSurprise,origNumVisits"
      << "\n";

  if(!std::atomic_is_lock_free(&sigReceived))
    throw StringError("sigReceived is not lock free, signal-quitting mechanism will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);

  std::mutex writeMutex;
  std::atomic<int64_t> numGamesStarted(0);
  int64_t numGamesWritten = 0;
  ForkData* forkData = new ForkData();

  auto gameLoop = [
    &gameRunner,&logger,&numGamesStarted,&forkData,numGamesTotal,&baseParams,&gameSeedBase,
    &nnEval,&writeMutex,&out,&numGamesWritten
  ](int threadIdx) {
    (void)threadIdx;
    auto shouldStopFunc = []() noexcept {
      return sigReceived.load();
    };
    Rand thisLoopSeedRand;
    while(true) {
      if(sigReceived.load())
        break;
      int64_t gameIdx = numGamesStarted.fetch_add(1,std::memory_order_acq_rel);
      if(gameIdx >= numGamesTotal)
        break;

      MatchPairer::BotSpec botSpec;
      botSpec.botIdx = 0;
      botSpec.botName = nnEval->getModelName();
      botSpec.nnEval = nnEval;
      botSpec.baseParams = baseParams;

      string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
      FinishedGameData* gameData = gameRunner->runGame(
        seed, botSpec, botSpec, forkData, NULL, logger,
        shouldStopFunc, nullptr, nullptr, nullptr, nullptr
      );
      //Happens when interrupted by the signal handler mid-game
      if(gameData == NULL)
        break;

      {
        std::lock_guard<std::mutex> lock(writeMutex);
        writeSurpriseStatsRows(out, gameIdx, *gameData);
        out.flush();
        numGamesWritten += 1;
        logger.write(
          "Finished game " + Global::int64ToString(gameIdx) + " (" +
          Global::int64ToString(numGamesWritten) + "/" + Global::int64ToString(numGamesTotal) + " written)"
        );
      }
      delete gameData;
    }
  };

  vector<std::thread> threads;
  for(int i = 0; i<numGameThreads; i++)
    threads.push_back(std::thread(gameLoop,i));
  for(int i = 0; i<numGameThreads; i++)
    threads[i].join();

  out.close();
  logger.write("Wrote stats for " + Global::int64ToString(numGamesWritten) + " games to " + outputFile);

  delete forkData;
  delete gameRunner;
  delete nnEval;
  ScoreValue::freeTables();
  return 0;
}

int MainCmds::writesampletrainquoridor(const vector<string>& args) {
  if(args.size() < 2) {
    cout << "Usage: katago writesampletrainquoridor OUTDIR [NUM_FILES] [NUM_ROWS_PER_FILE]" << endl;
    return 1;
  }
  string outDir = args[1];
  int numFiles = (args.size() >= 3) ? Global::stringToInt(args[2]) : 2;
  int rowsPerFile = (args.size() >= 4) ? Global::stringToInt(args[3]) : 16;

  Board::initHash();
  ScoreValue::initTables();

  MakeDir::make(outDir);

  Rand rand("writesampletrainquoridor");
  int inputsVersion = 1;
  int nnXLen = 9;
  int nnYLen = 9;

  for(int f = 0; f < numFiles; f++) {
    TrainingWriteBuffers buffers(
      inputsVersion, rowsPerFile, NNInputs::NUM_FEATURES_SPATIAL_V1, NNInputs::NUM_FEATURES_GLOBAL_V1, nnXLen, nnYLen, false
    );

    while(buffers.curRows < rowsPerFile) {
      Board startBoard(17, 17);
      Player startPla = P_BLACK;
      Rules rules = Rules::getTrompTaylorish();
      BoardHistory startHist(startBoard, startPla, rules, 0, BoardHistoryModes(false, false));

      vector<Board> posHist;
      posHist.push_back(startBoard);

      Board currBoard = startBoard;
      BoardHistory currHist = startHist;
      Player currPla = startPla;

      int maxTurns = 30;
      int turnsPlayed = 0;

      for(int t = 0; t < maxTurns; t++) {
        std::vector<Loc> pawnDests = currBoard.getLegalPawnDestinations(currPla);
        Loc chosenMove = Board::NULL_LOC;
        if(!pawnDests.empty()) {
          int bestDist = 999;
          for(Loc dest : pawnDests) {
            int d = (currPla == P_BLACK) ? Location::getY(dest, 17) : (16 - Location::getY(dest, 17));
            if(d < bestDist) {
              bestDist = d;
              chosenMove = dest;
            }
          }
        }
        if(chosenMove == Board::NULL_LOC) break;

        currHist.makeBoardMoveAssumeLegal(currBoard, chosenMove, currPla, NULL);
        posHist.push_back(currBoard);
        turnsPlayed++;
        if(currHist.isGameFinished) break;
        currPla = getOpp(currPla);
      }

      if(!currHist.isGameFinished) {
        currHist.isGameFinished = true;
        currHist.winner = P_BLACK;
        currHist.isNoResult = false;
      }

      vector<ValueTargets> whiteValueTargets(turnsPlayed + 1);
      for(size_t i = 0; i <= (size_t)turnsPlayed; i++) {
        whiteValueTargets[i].win = (currHist.winner == P_WHITE) ? 1.0f : 0.0f;
        whiteValueTargets[i].loss = (currHist.winner == P_BLACK) ? 1.0f : 0.0f;
        whiteValueTargets[i].noResult = 0.0f;
        whiteValueTargets[i].score = (currHist.winner == P_WHITE) ? 5.0f : -5.0f;
        whiteValueTargets[i].hasLead = true;
        whiteValueTargets[i].lead = whiteValueTargets[i].score;
      }
      vector<QValueTargets> whiteQValueTargets(turnsPlayed + 1);
      NNRawStats nnRawStats;
      nnRawStats.whiteWinLoss = 0.0;
      nnRawStats.whiteScoreMean = 0.0;
      nnRawStats.policyEntropy = 0.0;

      for(int turn = 0; turn < turnsPlayed && buffers.curRows < rowsPerFile; turn++) {
        Player p = (turn % 2 == 0) ? P_BLACK : P_WHITE;
        vector<PolicyTargetMove> policyTarget0;
        Loc moveLoc = currHist.moveHistory[turn].loc;
        policyTarget0.push_back(PolicyTargetMove(moveLoc, 100));

        vector<PolicyTargetMove> policyTarget1;
        if(turn + 1 < turnsPlayed) {
          policyTarget1.push_back(PolicyTargetMove(currHist.moveHistory[turn + 1].loc, 100));
        }

        Board bAtTurn = posHist[turn];
        BoardHistory hAtTurn(bAtTurn, p, rules, turn, BoardHistoryModes(false, false));

        buffers.addRow(
          bAtTurn, hAtTurn, p,
          startHist, currHist,
          turn, 1.0f, 100,
          &policyTarget0, (turn + 1 < turnsPlayed ? &policyTarget1 : NULL),
          0.1, 1.0, 1.0,
          whiteValueTargets, whiteQValueTargets,
          turn, 1.0f, 1.0f, 1.0f,
          nnRawStats,
          &currBoard, NULL, NULL, NULL,
          &posHist,
          false, 0, 0.5, C_EMPTY, 0.0,
          Hash128(), vector<ChangedNeuralNet*>(),
          false, 0, FinishedGameData::MODE_NORMAL,
          NULL, rand, ReanalysisData()
        );
      }
    }

    string fileName = outDir + "/train" + Global::intToString(f) + ".npz";
    buffers.writeToZipFile(fileName);
    cout << "Wrote " << buffers.curRows << " rows to " << fileName << endl;
  }

  ScoreValue::freeTables();
  return 0;
}

int MainCmds::dumpnninputs(const vector<string>& args) {
  string outputFile = "dumpnninputs.npz";
  int numRows = 200;
  string seed = "dumpnninputs-seed-quoridor";

  for(size_t i = 1; i < args.size(); i++) {
    if(args[i] == "-output" && i + 1 < args.size()) {
      outputFile = args[++i];
    }
    else if(args[i] == "-n" && i + 1 < args.size()) {
      numRows = Global::stringToInt(args[++i]);
    }
    else if(args[i] == "-seed" && i + 1 < args.size()) {
      seed = args[++i];
    }
    else if(args[i] == "-h" || args[i] == "--help") {
      cout << "Usage: katago dumpnninputs [-output PATH.npz] [-n NUM_ROWS] [-seed SEED_STR]" << endl;
      return 0;
    }
    else if(i == 1 && args[i][0] != '-') {
      outputFile = args[i];
    }
    else if(i == 2 && args[i][0] != '-') {
      numRows = Global::stringToInt(args[i]);
    }
    else if(i == 3 && args[i][0] != '-') {
      seed = args[i];
    }
  }

  if(numRows <= 0) {
    cerr << "Error: numRows must be > 0" << endl;
    return 1;
  }

  Board::initHash();
  ScoreValue::initTables();

  cout << "Dumping " << numRows << " Quoridor NN input rows to " << outputFile << " (seed: " << seed << ")..." << endl;

  Rand rand(seed);

  NumpyBuffer<float> binaryInputNCHW(std::vector<int64_t>({numRows, 17, 9, 9}));
  NumpyBuffer<float> globalInputNC(std::vector<int64_t>({numRows, 15}));
  NumpyBuffer<uint8_t> legalMovesMask(std::vector<int64_t>({numRows, 290}));
  NumpyBuffer<int32_t> nextPlayer(std::vector<int64_t>({numRows, 1}));

  int curRows = 0;
  Rules rules = Rules::getTrompTaylorish();
  MiscNNInputParams params;

  while(curRows < numRows) {
    Board board(17, 17);
    Player pla = P_BLACK;
    BoardHistory hist(board, pla, rules, 0, BoardHistoryModes());

    for(int step = 0; step < 300 && curRows < numRows; step++) {
      int blackY = Location::getY(board.blackPawnLoc, board.x_size);
      int whiteY = Location::getY(board.whitePawnLoc, board.x_size);
      if(blackY == 0 || whiteY == board.y_size - 1)
        break;

      vector<Loc> legalMoves;
      uint8_t* maskPtr = legalMovesMask.data + curRows * 290;
      std::fill(maskPtr, maskPtr + 290, (uint8_t)0);

      for(int y = 0; y < board.y_size; y++) {
        for(int x = 0; x < board.x_size; x++) {
          Loc loc = Location::getLoc(x, y, board.x_size);
          if(board.isLegal(loc, pla)) {
            legalMoves.push_back(loc);
            int pos = NNPos::locToPos(loc, 17, 17, 17);
            if(pos >= 0 && pos < 290)
              maskPtr[pos] = 1;
          }
        }
      }

      if(legalMoves.empty())
        break;

      float* rowSpatial = binaryInputNCHW.data + curRows * (17 * 81);
      float* rowGlobal = globalInputNC.data + curRows * 15;
      NNInputs::fillRowV1(board, hist, pla, params, 9, 9, false, rowSpatial, rowGlobal);

      nextPlayer.data[curRows] = (int32_t)pla;
      curRows++;

      Loc chosen = legalMoves[(size_t)rand.nextUInt((uint32_t)legalMoves.size())];
      hist.makeBoardMoveAssumeLegal(board, chosen, pla, NULL);
      pla = getOpp(pla);
    }
  }

  size_t lastSlash = outputFile.find_last_of("/\\");
  if(lastSlash != string::npos) {
    MakeDir::make(outputFile.substr(0, lastSlash));
  }

  ZipFile zipFile(outputFile);
  uint64_t numBytes;

  numBytes = binaryInputNCHW.prepareHeaderWithNumRows(numRows);
  zipFile.writeBuffer("binaryInputNCHW", binaryInputNCHW.dataIncludingHeader, numBytes);

  numBytes = globalInputNC.prepareHeaderWithNumRows(numRows);
  zipFile.writeBuffer("globalInputNC", globalInputNC.dataIncludingHeader, numBytes);

  numBytes = legalMovesMask.prepareHeaderWithNumRows(numRows);
  zipFile.writeBuffer("legalMovesMask", legalMovesMask.dataIncludingHeader, numBytes);

  numBytes = nextPlayer.prepareHeaderWithNumRows(numRows);
  zipFile.writeBuffer("nextPlayer", nextPlayer.dataIncludingHeader, numBytes);

  zipFile.close();
  cout << "Successfully dumped " << numRows << " rows to " << outputFile << endl;

  string binFile = outputFile;
  if(Global::isSuffix(binFile, ".npz"))
    binFile = binFile.substr(0, binFile.size() - 4) + ".bin";
  else
    binFile += ".bin";

  ofstream outBin(binFile, ios::binary);
  if(outBin.is_open()) {
    int32_t header[4] = { numRows, 17 * 81, 15, 290 };
    outBin.write(reinterpret_cast<const char*>(header), sizeof(header));
    outBin.write(reinterpret_cast<const char*>(binaryInputNCHW.data), numRows * 17 * 81 * sizeof(float));
    outBin.write(reinterpret_cast<const char*>(globalInputNC.data), numRows * 15 * sizeof(float));
    outBin.write(reinterpret_cast<const char*>(legalMovesMask.data), numRows * 290 * sizeof(uint8_t));
    outBin.write(reinterpret_cast<const char*>(nextPlayer.data), numRows * sizeof(int32_t));
    outBin.close();
    cout << "Also wrote raw binary dump to " << binFile << endl;
  }

  ScoreValue::freeTables();
  return 0;
}

int MainCmds::evalnninputs(const vector<string>& args) {
  string modelFile = "";
  string configFile = "";
  string refBinFile = "";
  string outputBinFile = "";
  int numRows = 200;
  string seed = "dumpnninputs-seed-quoridor";

  for(size_t i = 1; i < args.size(); i++) {
    if(args[i] == "-model" && i + 1 < args.size()) {
      modelFile = args[++i];
    }
    else if(args[i] == "-config" && i + 1 < args.size()) {
      configFile = args[++i];
    }
    else if(args[i] == "-n" && i + 1 < args.size()) {
      numRows = Global::stringToInt(args[++i]);
    }
    else if(args[i] == "-seed" && i + 1 < args.size()) {
      seed = args[++i];
    }
    else if(args[i] == "-ref-bin" && i + 1 < args.size()) {
      refBinFile = args[++i];
    }
    else if(args[i] == "-output-bin" && i + 1 < args.size()) {
      outputBinFile = args[++i];
    }
    else if(args[i] == "-h" || args[i] == "--help") {
      cout << "Usage: katago evalnninputs -model MODEL.bin.gz [-config CONFIG.cfg] [-n NUM_ROWS] [-seed SEED] [-ref-bin REF.bin] [-output-bin OUT.bin]" << endl;
      return 0;
    }
  }

  if(modelFile == "") {
    cerr << "Error: -model MODEL_FILE must be specified" << endl;
    return 1;
  }

  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  if(configFile != "" && FileUtils::exists(configFile)) {
    cfg.initialize(configFile);
  }
  if(!cfg.contains("nnMaxBatchSize")) cfg.overrideKey("nnMaxBatchSize", "16");
  if(!cfg.contains("numNNServerThreadsPerModel")) cfg.overrideKey("numNNServerThreadsPerModel", "1");

  Logger logger(NULL, true, true, false);
  Rand seedRand(seed);
  int maxBatchSize = cfg.getInt("nnMaxBatchSize", 1, 64);
  int nnLen = 17;

  cout << "Loading model for Quoridor eval parity: " << modelFile << endl;

  NNEvaluator* nnEval = NULL;
  try {
    nnEval = Setup::initializeNNEvaluator(
      modelFile, modelFile, "", cfg, logger, seedRand,
      maxBatchSize, nnLen, nnLen, Setup::MaxBatchSizeRequest::explicitSize(maxBatchSize),
      false, false, Setup::SETUP_FOR_BENCHMARKNN
    );
  }
  catch(const std::exception& e) {
    cout << "DIFF: NNEvaluator initialization failed on current contract: " << e.what() << endl;
    ScoreValue::freeTables();
    return 2;
  }

  if(nnEval == NULL) {
    cout << "DIFF: NNEvaluator was NULL" << endl;
    ScoreValue::freeTables();
    return 2;
  }

  cout << "Evaluating " << numRows << " positions at symmetries 0 and 1..." << endl;

  Rand rand(seed);
  Rules rules = Rules::getTrompTaylorish();
  int curRows = 0;

  vector<float> predPolicySym0(numRows * 290, -1e30f);
  vector<float> predPolicySym1(numRows * 290, -1e30f);
  vector<float> predValue(numRows * 3, 0.0f);

  bool evalSuccess = true;
  string evalFailReason = "";

  while(curRows < numRows && evalSuccess) {
    Board board(17, 17);
    Player pla = P_BLACK;
    BoardHistory hist(board, pla, rules, 0, BoardHistoryModes());

    for(int step = 0; step < 300 && curRows < numRows && evalSuccess; step++) {
      int blackY = Location::getY(board.blackPawnLoc, board.x_size);
      int whiteY = Location::getY(board.whitePawnLoc, board.x_size);
      if(blackY == 0 || whiteY == board.y_size - 1)
        break;

      vector<Loc> legalMoves;
      vector<bool> isLegalMask(290, false);
      for(int y = 0; y < board.y_size; y++) {
        for(int x = 0; x < board.x_size; x++) {
          Loc loc = Location::getLoc(x, y, board.x_size);
          if(board.isLegal(loc, pla)) {
            legalMoves.push_back(loc);
            int pos = NNPos::locToPos(loc, 17, 17, 17);
            if(pos >= 0 && pos < 290)
              isLegalMask[pos] = true;
          }
        }
      }
      if(legalMoves.empty()) break;

      try {
        NNResultBuf buf0;
        MiscNNInputParams params0;
        params0.symmetry = 0;
        nnEval->evaluate(board, hist, pla, params0, buf0, true, false);

        for(int pos = 0; pos < 290; pos++) {
          float logit = buf0.result->policyProbs[pos];
          predPolicySym0[curRows * 290 + pos] = isLegalMask[pos] ? logit : -1e30f;
        }
        predValue[curRows * 3 + 0] = buf0.result->whiteWinProb;
        predValue[curRows * 3 + 1] = buf0.result->whiteLossProb;
        predValue[curRows * 3 + 2] = buf0.result->whiteNoResultProb;

        NNResultBuf buf1;
        MiscNNInputParams params1;
        params1.symmetry = 1;
        nnEval->evaluate(board, hist, pla, params1, buf1, true, false);

        for(int pos = 0; pos < 290; pos++) {
          float logit = buf1.result->policyProbs[pos];
          predPolicySym1[curRows * 290 + pos] = isLegalMask[pos] ? logit : -1e30f;
        }
      }
      catch(const std::exception& e) {
        evalSuccess = false;
        evalFailReason = e.what();
        break;
      }

      curRows++;
      Loc chosen = legalMoves[(size_t)rand.nextUInt((uint32_t)legalMoves.size())];
      hist.makeBoardMoveAssumeLegal(board, chosen, pla, NULL);
      pla = getOpp(pla);
    }
  }

  if(!evalSuccess) {
    cout << "DIFF: Evaluation encountered error on current contract: " << evalFailReason << endl;
  }
  else {
    cout << "Successfully evaluated " << curRows << " rows in C++." << endl;
  }

  if(outputBinFile != "") {
    size_t lastSlash = outputBinFile.find_last_of("/\\");
    if(lastSlash != string::npos) MakeDir::make(outputBinFile.substr(0, lastSlash));
    ofstream outBin(outputBinFile, ios::binary);
    if(outBin.is_open()) {
      int32_t header[3] = { curRows, 290, 3 };
      outBin.write(reinterpret_cast<const char*>(header), sizeof(header));
      outBin.write(reinterpret_cast<const char*>(predPolicySym0.data()), curRows * 290 * sizeof(float));
      outBin.write(reinterpret_cast<const char*>(predPolicySym1.data()), curRows * 290 * sizeof(float));
      outBin.write(reinterpret_cast<const char*>(predValue.data()), curRows * 3 * sizeof(float));
      outBin.close();
      cout << "Wrote evaluation predictions to " << outputBinFile << endl;
    }
  }

  int exitCode = 0;
  if(refBinFile != "" && FileUtils::exists(refBinFile) && evalSuccess) {
    ifstream refIn(refBinFile, ios::binary);
    if(refIn.is_open()) {
      int32_t refHeader[3];
      refIn.read(reinterpret_cast<char*>(refHeader), sizeof(refHeader));
      int refRows = refHeader[0];
      int refValChannels = refHeader[2];

      vector<float> refPolicySym0(refRows * 290);
      vector<float> refPolicySym1(refRows * 290);
      vector<float> refVal(refRows * refValChannels);

      refIn.read(reinterpret_cast<char*>(refPolicySym0.data()), refRows * 290 * sizeof(float));
      refIn.read(reinterpret_cast<char*>(refPolicySym1.data()), refRows * 290 * sizeof(float));
      refIn.read(reinterpret_cast<char*>(refVal.data()), refRows * refValChannels * sizeof(float));
      refIn.close();

      int compareRows = std::min(curRows, refRows);
      float maxDiffPol0 = 0.0f;
      float maxDiffPol1 = 0.0f;
      float maxDiffVal = 0.0f;

      for(int r = 0; r < compareRows; r++) {
        for(int p = 0; p < 290; p++) {
          float cp0 = predPolicySym0[r * 290 + p];
          float rp0 = refPolicySym0[r * 290 + p];
          if(rp0 > -1e20f && cp0 > -1e20f) {
            maxDiffPol0 = std::max(maxDiffPol0, std::abs(cp0 - rp0));
          }
          float cp1 = predPolicySym1[r * 290 + p];
          float rp1 = refPolicySym1[r * 290 + p];
          if(rp1 > -1e20f && cp1 > -1e20f) {
            maxDiffPol1 = std::max(maxDiffPol1, std::abs(cp1 - rp1));
          }
        }
        for(int v = 0; v < std::min(2, refValChannels); v++) {
          maxDiffVal = std::max(maxDiffVal, std::abs(predValue[r * 3 + v] - refVal[r * refValChannels + v]));
        }
      }

      cout << "=== NN Parity Comparison Results (" << compareRows << " rows) ===" << endl;
      cout << "  Max Policy Diff (sym 0): " << maxDiffPol0 << endl;
      cout << "  Max Policy Diff (sym 1): " << maxDiffPol1 << endl;
      cout << "  Max Value Diff:          " << maxDiffVal << endl;

      float tol = 1e-4f;
      if(maxDiffPol0 <= tol && maxDiffPol1 <= tol && maxDiffVal <= tol) {
        cout << "PASS: Parity verified within tolerance " << tol << "!" << endl;
      }
      else {
        cout << "DIFF: Outputs exceed tolerance " << tol << "!" << endl;
        exitCode = 1;
      }
    }
  }

  delete nnEval;
  ScoreValue::freeTables();
  return evalSuccess ? exitCode : 2;
}
