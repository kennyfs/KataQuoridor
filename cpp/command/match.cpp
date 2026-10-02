#include "../core/global.h"
#include "../core/fileutils.h"
#include "../core/makedir.h"
#include "../core/config_parser.h"
#include "../core/timer.h"
#include "../dataio/sgf.h"
#include "../search/asyncbot.h"
#include "../search/patternbonustable.h"
#include "../program/setup.h"
#include "../program/play.h"
#include "../command/commandline.h"
#include "../core/test.h"
#include "../main.h"

#include <csignal>

using namespace std;


//KataQuoridor: a fixed list of games (gameListFile), each with its bots and opening moves.
//Line format: <gameId> <blackBotIdx> <whiteBotIdx> [opening move ...], '#' starts a comment.
struct ListedGame {
  string id;
  int botB;
  int botW;
  vector<string> moveStrs;
  Sgf::PositionSample startPos;
};

static vector<ListedGame> readGameList(const string& file, int numBots) {
  vector<ListedGame> games;
  vector<string> lines = FileUtils::readFileLines(file,'\n');
  for(size_t i = 0; i<lines.size(); i++) {
    string line = Global::trim(Global::stripComments(lines[i]));
    if(line.length() <= 0)
      continue;
    vector<string> pieces = Global::split(line,' ');
    vector<string> words;
    for(const string& p: pieces)
      if(Global::trim(p).length() > 0)
        words.push_back(Global::trim(p));
    ListedGame g;
    if(words.size() < 3 || !Global::tryStringToInt(words[1],g.botB) || !Global::tryStringToInt(words[2],g.botW) ||
       g.botB < 0 || g.botB >= numBots || g.botW < 0 || g.botW >= numBots)
      throw StringError("gameListFile " + file + " line " + Global::uint64ToString(i+1) +
                        ": expected <gameId> <blackBotIdx> <whiteBotIdx> [moves...] with bot indices < numBots, got: " + line);
    g.id = words[0];
    g.moveStrs.assign(words.begin()+3,words.end());
    games.push_back(g);
  }
  if(games.size() <= 0)
    throw StringError("gameListFile " + file + " has no games");
  return games;
}

//Turn the opening moves into a start position, checking that they are legal under the configured rules.
static void setStartPositions(vector<ListedGame>& games, int xSize, int ySize, const Rules& rules) {
  for(ListedGame& g: games) {
    Board board(xSize,ySize);
    board.setFencesLeft(rules.blackInitialFences, rules.whiteInitialFences);
    g.startPos.board = board;
    g.startPos.nextPla = P_BLACK;
    g.startPos.initialTurnNumber = 0;
    g.startPos.hintLoc = Board::NULL_LOC;
    g.startPos.weight = 1.0;
    BoardHistory hist(board,P_BLACK,rules,0,BoardHistoryModes());
    Player pla = P_BLACK;
    for(const string& s: g.moveStrs) {
      Loc loc;
      if(!Location::tryOfString(s,board,pla,loc) || !hist.isLegal(board,loc,pla) || hist.isGameFinished)
        throw StringError("gameListFile: illegal opening move " + s + " in game " + g.id);
      g.startPos.moves.push_back(Move(loc,pla));
      hist.makeBoardMoveAssumeLegal(board,loc,pla,NULL);
      pla = getOpp(pla);
    }
  }
}

static string jsonEscape(const string& s) {
  string out;
  for(char c: s) {
    if(c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out;
}

static std::atomic<bool> sigReceived(false);
static std::atomic<bool> shouldStop(false);
static void signalHandler(int signal)
{
  if(signal == SIGINT || signal == SIGTERM) {
    sigReceived.store(true);
    shouldStop.store(true);
  }
}

int MainCmds::match(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();
  Rand seedRand;

  ConfigParser cfg;
  string logFile;
  string sgfOutputDir;
  try {
    KataGoCommandLine cmd("Play different nets against each other with different search settings in a match or tournament.");
    cmd.addConfigFileArg("","match_example.cfg");

    TCLAP::ValueArg<string> logFileArg("","log-file","Log file to output to",false,string(),"FILE");
    TCLAP::ValueArg<string> sgfOutputDirArg("","sgf-output-dir","Dir to output sgf files",false,string(),"DIR");

    cmd.add(logFileArg);
    cmd.add(sgfOutputDirArg);

    cmd.setShortUsageArgLimit();
    cmd.addOverrideConfigArg();

    cmd.parseArgs(args);

    logFile = logFileArg.getValue();
    sgfOutputDir = sgfOutputDirArg.getValue();

    cmd.getConfig(cfg);
  }
  catch (TCLAP::ArgException &e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }

  Logger logger(&cfg);
  logger.addFile(logFile);

  logger.write("Match Engine starting...");
  logger.write(string("Git revision: ") + Version::getGitRevision());

  //Load per-bot search config, first, which also tells us how many bots we're running
  vector<SearchParams> paramss = Setup::loadParams(cfg,Setup::SETUP_FOR_MATCH);
  assert(paramss.size() > 0);
  int numBots = (int)paramss.size();

  //Figure out all pairs of bots that will be playing.
  std::vector<std::pair<int,int>> matchupsPerRound;
  {
    //Load a filter on what bots we actually want to run. By default, include everything.
    vector<bool> includeBot(numBots);
    if(cfg.contains("includeBots")) {
      vector<int> includeBotIdxs = cfg.getInts("includeBots",0,Setup::MAX_BOT_PARAMS_FROM_CFG);
      for(int i = 0; i<numBots; i++) {
        if(contains(includeBotIdxs,i))
          includeBot[i] = true;
      }
    }
    else {
      for(int i = 0; i<numBots; i++) {
        includeBot[i] = true;
      }
    }

    std::vector<int> secondaryBotIdxs;
    if(cfg.contains("secondaryBots"))
      secondaryBotIdxs = cfg.getInts("secondaryBots",0,Setup::MAX_BOT_PARAMS_FROM_CFG);
    for(int i = 0; i<secondaryBotIdxs.size(); i++)
      if(secondaryBotIdxs[i] < 0 || secondaryBotIdxs[i] >= numBots)
        throw StringError("secondaryBots value " + Global::intToString(secondaryBotIdxs[i]) + " is out of range, numBots is " + Global::intToString(numBots));

    for(int i = 0; i<numBots; i++) {
      if(!includeBot[i])
        continue;
      for(int j = 0; j<numBots; j++) {
        if(!includeBot[j])
          continue;
        if(i < j && !(contains(secondaryBotIdxs,i) && contains(secondaryBotIdxs,j))) {
          matchupsPerRound.emplace_back(i,j);
          matchupsPerRound.emplace_back(j,i);
        }
      }
    }

    if(cfg.contains("extraPairs")) {
      std::vector<std::pair<int,int>> pairs = cfg.getNonNegativeIntDashedPairs("extraPairs",0,numBots-1);
      for(const std::pair<int,int>& pair: pairs) {
        int p0 = pair.first;
        int p1 = pair.second;
        if(cfg.contains("extraPairsAreOneSidedBW") && cfg.getBool("extraPairsAreOneSidedBW")) {
          matchupsPerRound.emplace_back(p0,p1);
        }
        else {
          matchupsPerRound.emplace_back(p0,p1);
          matchupsPerRound.emplace_back(p1,p0);
        }
      }
    }
  }

  //KataQuoridor: with gameListFile, play exactly the listed games (in order, each once, from its opening) instead of
  //random pairings; numGamesTotal and the pairing keys are then unused. gameResultsFile gets one JSON line per game.
  vector<ListedGame> listedGames;
  if(cfg.contains("gameListFile"))
    listedGames = readGameList(cfg.getString("gameListFile"),numBots);
  const string gameResultsFile = cfg.contains("gameResultsFile") ? cfg.getString("gameResultsFile") : string();

  //Load the names of the bots and which model each bot is using
  vector<string> nnModelFilesByBot(numBots);
  vector<string> botNames(numBots);
  for(int i = 0; i<numBots; i++) {
    string idxStr = Global::intToString(i);

    if(cfg.contains("botName"+idxStr))
      botNames[i] = cfg.getString("botName"+idxStr);
    else if(numBots == 1)
      botNames[i] = cfg.getString("botName");
    else
      throw StringError("If more than one bot, must specify botName0, botName1,... individually");

    if(cfg.contains("nnModelFile"+idxStr))
      nnModelFilesByBot[i] = cfg.getString("nnModelFile"+idxStr);
    else
      nnModelFilesByBot[i] = cfg.getString("nnModelFile");
  }

  vector<bool> botIsUsed(numBots);
  for(const std::pair<int,int>& pair : matchupsPerRound) {
    botIsUsed[pair.first] = true;
    botIsUsed[pair.second] = true;
  }
  if(listedGames.size() > 0) {
    std::fill(botIsUsed.begin(),botIsUsed.end(),false);
    for(const ListedGame& g: listedGames) {
      botIsUsed[g.botB] = true;
      botIsUsed[g.botW] = true;
    }
  }

  //Dedup and load each necessary model exactly once
  vector<string> nnModelFiles;
  vector<int> whichNNModel(numBots);
  for(int i = 0; i<numBots; i++) {
    if(!botIsUsed[i])
      continue;

    const string& desiredFile = nnModelFilesByBot[i];
    int alreadyFoundIdx = -1;
    for(int j = 0; j<nnModelFiles.size(); j++) {
      if(nnModelFiles[j] == desiredFile) {
        alreadyFoundIdx = j;
        break;
      }
    }
    if(alreadyFoundIdx != -1)
      whichNNModel[i] = alreadyFoundIdx;
    else {
      whichNNModel[i] = (int)nnModelFiles.size();
      nnModelFiles.push_back(desiredFile);
    }
  }

  //Load match runner settings
  int numGameThreads = cfg.getInt("numGameThreads",1,16384);
  const string gameSeedBase = Global::uint64ToHexString(seedRand.nextUInt64());

  //Work out an upper bound on how many concurrent nneval requests we could end up making.
  int expectedConcurrentEvals;
  {
    //Work out the max threads any one bot uses
    int maxBotThreads = 0;
    for(int i = 0; i<numBots; i++)
      if(paramss[i].numThreads > maxBotThreads)
        maxBotThreads = paramss[i].numThreads;
    //Mutiply by the number of concurrent games we could have
    expectedConcurrentEvals = maxBotThreads * numGameThreads;
  }

  //Initialize object for randomizing game settings and running games
  PlaySettings playSettings = PlaySettings::loadForMatch(cfg);
  GameRunner* gameRunner = new GameRunner(cfg, playSettings, logger);
  const int minBoardXSizeUsed = gameRunner->getGameInitializer()->getMinBoardXSize();
  const int minBoardYSizeUsed = gameRunner->getGameInitializer()->getMinBoardYSize();
  const int maxBoardXSizeUsed = gameRunner->getGameInitializer()->getMaxBoardXSize();
  const int maxBoardYSizeUsed = gameRunner->getGameInitializer()->getMaxBoardYSize();
  if(listedGames.size() > 0) {
    if(minBoardXSizeUsed != maxBoardXSizeUsed || minBoardYSizeUsed != maxBoardYSizeUsed)
      throw StringError("gameListFile needs a single board size");
    Rules listRules = Rules::getQuoridorRules();
    Setup::loadQuoridorRuleKeys(cfg, listRules);
    setStartPositions(listedGames, maxBoardXSizeUsed, maxBoardYSizeUsed, listRules);
    logger.write("Loaded " + Global::uint64ToString(listedGames.size()) + " games from gameListFile");
  }

  //Initialize neural net inference engine globals, and load models
  Setup::initializeSession(cfg);
  const vector<string>& nnModelNames = nnModelFiles;
  const bool defaultRequireExactNNLen = minBoardXSizeUsed == maxBoardXSizeUsed && minBoardYSizeUsed == maxBoardYSizeUsed;
  const bool disableFP16 = false;
  const vector<string> expectedSha256s;
  vector<NNEvaluator*> nnEvals = Setup::initializeNNEvaluators(
    nnModelNames,nnModelFiles,expectedSha256s,cfg,logger,seedRand,expectedConcurrentEvals,
    maxBoardXSizeUsed,maxBoardYSizeUsed,Setup::MaxBatchSizeRequest::requireFromConfig(),defaultRequireExactNNLen,disableFP16,
    Setup::SETUP_FOR_MATCH
  );
  logger.write("Loaded neural net");

  vector<NNEvaluator*> nnEvalsByBot(numBots);
  for(int i = 0; i<numBots; i++) {
    if(!botIsUsed[i])
      continue;
    nnEvalsByBot[i] = nnEvals[whichNNModel[i]];
  }

  std::vector<std::unique_ptr<PatternBonusTable>> patternBonusTables = Setup::loadAvoidSgfPatternBonusTables(cfg,logger);
  testAssert(patternBonusTables.size() == numBots);

  //Initialize object for randomly pairing bots
  int64_t numGamesTotal = listedGames.size() > 0 ? (int64_t)listedGames.size() : cfg.getInt64("numGamesTotal",1,((int64_t)1) << 62);
  MatchPairer* matchPairer = new MatchPairer(cfg,numBots,botNames,nnEvalsByBot,paramss,matchupsPerRound,numGamesTotal);

  //Check for unused config keys
  cfg.warnUnusedKeys(cerr,&logger);
  for(int i = 0; i<numBots; i++) {
    if(!botIsUsed[i])
      continue;
    Setup::maybeWarnHumanSLParams(paramss[i],nnEvalsByBot[i],NULL,cerr,&logger);
  }

  //Done loading!
  //------------------------------------------------------------------------------------
  logger.write("Loaded all config stuff, starting matches");
  if(!logger.isLoggingToStdout())
    cout << "Loaded all config stuff, starting matches" << endl;

  if(sgfOutputDir != string())
    MakeDir::make(sgfOutputDir);

  if(!std::atomic_is_lock_free(&shouldStop))
    throw StringError("shouldStop is not lock free, signal-quitting mechanism for terminating matches will NOT work!");
  std::signal(SIGINT, signalHandler);
  std::signal(SIGTERM, signalHandler);


  std::mutex statsMutex;
  int64_t gameCount = 0;
  std::map<string,double> timeUsedByBotMap;
  std::map<string,double> movesByBotMap;

  struct BotStats {
    int64_t wins = 0;
    int64_t losses = 0;
    int64_t draws = 0;
  };
  std::map<string,BotStats> botStatsMap;

  std::atomic<size_t> nextListedGame(0);
  ofstream* gameResultsOut = NULL;
  if(gameResultsFile != string()) {
    gameResultsOut = new ofstream();
    FileUtils::open(*gameResultsOut, gameResultsFile, std::ios::out | std::ios::app);
  }
  vector<NNEvaluator*> nnEvalsByBotForList = nnEvalsByBot;

  auto runMatchLoop = [
    &gameRunner,&matchPairer,&sgfOutputDir,&logger,&gameSeedBase,&patternBonusTables,
    &statsMutex, &gameCount, &timeUsedByBotMap, &movesByBotMap, &botStatsMap,
    &listedGames, &nextListedGame, &gameResultsOut, &nnEvalsByBotForList, &paramss, &botNames
  ](
    uint64_t threadHash
  ) {
    ofstream* sgfOut = NULL;
    if(sgfOutputDir.length() > 0) {
      sgfOut = new ofstream();
      FileUtils::open(*sgfOut, sgfOutputDir + "/" + Global::uint64ToHexString(threadHash) + ".sgfs");
    }
    auto shouldStopFunc = []() noexcept {
      return shouldStop.load();
    };
    WaitableFlag* shouldPause = nullptr;

    Rand thisLoopSeedRand;
    while(true) {
      if(shouldStop.load())
        break;

      FinishedGameData* gameData = NULL;

      MatchPairer::BotSpec botSpecB;
      MatchPairer::BotSpec botSpecW;
      const ListedGame* listedGame = NULL;
      bool haveMatchup;
      if(listedGames.size() > 0) {
        size_t idx = nextListedGame.fetch_add(1);
        haveMatchup = idx < listedGames.size();
        if(haveMatchup) {
          listedGame = &listedGames[idx];
          auto makeSpec = [&](int botIdx, MatchPairer::BotSpec& spec) {
            spec.botIdx = botIdx;
            spec.botName = botNames[botIdx];
            spec.nnEval = nnEvalsByBotForList[botIdx];
            spec.baseParams = paramss[botIdx];
          };
          makeSpec(listedGame->botB, botSpecB);
          makeSpec(listedGame->botW, botSpecW);
        }
      }
      else
        haveMatchup = matchPairer->getMatchup(botSpecB, botSpecW, logger);
      if(haveMatchup) {
        string seed = gameSeedBase + ":" + Global::uint64ToHexString(thisLoopSeedRand.nextUInt64());
        std::function<void(const MatchPairer::BotSpec&, Search*)> afterInitialization = [&patternBonusTables](const MatchPairer::BotSpec& spec, Search* search) {
          assert(spec.botIdx < patternBonusTables.size());
          search->setCopyOfExternalPatternBonusTable(patternBonusTables[spec.botIdx]);
        };
        gameData = gameRunner->runGame(
          seed, botSpecB, botSpecW, NULL, listedGame != NULL ? &listedGame->startPos : NULL, logger,
          shouldStopFunc, shouldPause, nullptr, afterInitialization, nullptr
        );
      }

      bool shouldContinue = gameData != NULL;
      if(gameData != NULL) {
        if(sgfOut != NULL) {
          if(listedGame != NULL)
            WriteSgf::writeSgf(*sgfOut,gameData->bName,gameData->wName,gameData->endHist,gameData,false,true,
                               std::numeric_limits<double>::quiet_NaN(),vector<string>({"gameId=" + listedGame->id}));
          else
            WriteSgf::writeSgf(*sgfOut,gameData->bName,gameData->wName,gameData->endHist,gameData,false,true);
          (*sgfOut) << endl;
        }

        {
          std::lock_guard<std::mutex> lock(statsMutex);
          if(gameResultsOut != NULL) {
            const BoardHistory& h = gameData->endHist;
            //A game cut off by a stop signal is not finished and is not recorded.
            if(h.isGameFinished) {
              string winner = h.winner == C_BLACK ? "\"b\"" : h.winner == C_WHITE ? "\"w\"" : "null";
              (*gameResultsOut)
                << "{\"id\":\"" << jsonEscape(listedGame != NULL ? listedGame->id : string()) << "\""
                << ",\"black\":\"" << jsonEscape(gameData->bName) << "\""
                << ",\"white\":\"" << jsonEscape(gameData->wName) << "\""
                << ",\"winner\":" << winner
                << ",\"result\":\"" << WriteSgf::gameResultNoSgfTag(h) << "\""
                << ",\"draw_reason\":\"" << WriteSgf::drawReason(h) << "\""
                << ",\"plies\":" << h.moveHistory.size()
                << ",\"opening_plies\":" << gameData->startHist.moveHistory.size()
                << ",\"rules\":\"" << jsonEscape(h.rules.toString()) << "\""
                << ",\"black_walls\":" << h.initialBoard.blackFences
                << ",\"white_walls\":" << h.initialBoard.whiteFences
                << ",\"b_time\":" << gameData->bTimeUsed
                << ",\"w_time\":" << gameData->wTimeUsed
                << "}" << endl;
              gameResultsOut->flush();
            }
          }
          gameCount += 1;
          timeUsedByBotMap[gameData->bName] += gameData->bTimeUsed;
          timeUsedByBotMap[gameData->wName] += gameData->wTimeUsed;
          movesByBotMap[gameData->bName] += (double)gameData->bMoveCount;
          movesByBotMap[gameData->wName] += (double)gameData->wMoveCount;

          string resultStr = WriteSgf::gameResultNoSgfTag(gameData->endHist);
          string outcomeDesc;
          if(gameData->endHist.winner == C_BLACK) {
            botStatsMap[gameData->bName].wins += 1;
            botStatsMap[gameData->wName].losses += 1;
            outcomeDesc = gameData->bName + " (Black) won";
          }
          else if(gameData->endHist.winner == C_WHITE) {
            botStatsMap[gameData->wName].wins += 1;
            botStatsMap[gameData->bName].losses += 1;
            outcomeDesc = gameData->wName + " (White) won";
          }
          else {
            botStatsMap[gameData->bName].draws += 1;
            botStatsMap[gameData->wName].draws += 1;
            outcomeDesc = gameData->hitTurnLimit ? "Draw (cutoff reached)" : "Draw";
          }

          string gameMsg = "Game " + Global::int64ToString(gameCount) + ": " +
            gameData->bName + " (B) vs " + gameData->wName + " (W) -> " +
            resultStr + " (" + outcomeDesc + "), " +
            Global::intToString(gameData->endHist.moveHistory.size()) + " moves";
          logger.write(gameMsg);
          if(!logger.isLoggingToStdout())
            cout << gameMsg << endl;

          int64_t x = gameCount;
          while(x % 2 == 0 && x > 1) x /= 2;
          if(x == 1 || x == 3 || x == 5) {
            for(auto& pair : timeUsedByBotMap) {
              logger.write(
                "Avg move time used by " + pair.first + " " +
                Global::doubleToString(pair.second / movesByBotMap[pair.first]) + " " +
                Global::doubleToString(movesByBotMap[pair.first]) + " moves"
              );
            }
            for(const auto& pair : botStatsMap) {
              string statsMsg = "Bot " + pair.first + " stats: " +
                Global::int64ToString(pair.second.wins) + " W / " +
                Global::int64ToString(pair.second.losses) + " L / " +
                Global::int64ToString(pair.second.draws) + " D";
              logger.write(statsMsg);
              if(!logger.isLoggingToStdout())
                cout << statsMsg << endl;
            }
          }
        }

        delete gameData;
      }

      if(shouldStop.load())
        break;
      if(!shouldContinue)
        break;
    }
    if(sgfOut != NULL) {
      sgfOut->close();
      delete sgfOut;
    }
    logger.write("Match loop thread terminating");
  };
  auto runMatchLoopProtected = [&logger,&runMatchLoop](uint64_t threadHash) {
    Logger::logThreadUncaught("match loop", &logger, [&](){ runMatchLoop(threadHash); });
  };


  Rand hashRand;
  vector<std::thread> threads;
  threads.reserve(numGameThreads);
  for(int i = 0; i<numGameThreads; i++) {
    threads.emplace_back(runMatchLoopProtected, hashRand.nextUInt64());
  }
  for(int i = 0; i<threads.size(); i++)
    threads[i].join();

  logger.write("--- Match Results ---");
  if(!logger.isLoggingToStdout())
    cout << "--- Match Results ---" << endl;
  for(const auto& pair : botStatsMap) {
    string finalMsg = "Bot " + pair.first + ": " +
      Global::int64ToString(pair.second.wins) + " Wins, " +
      Global::int64ToString(pair.second.losses) + " Losses, " +
      Global::int64ToString(pair.second.draws) + " Draws (Total " +
      Global::int64ToString(pair.second.wins + pair.second.losses + pair.second.draws) + " games)";
    logger.write(finalMsg);
    if(!logger.isLoggingToStdout())
      cout << finalMsg << endl;
  }

  if(gameResultsOut != NULL) {
    gameResultsOut->close();
    delete gameResultsOut;
  }
  delete matchPairer;
  delete gameRunner;

  nnEvalsByBot.clear();
  for(int i = 0; i<nnEvals.size(); i++) {
    if(nnEvals[i] != NULL) {
      logger.write(nnEvals[i]->getModelFileName());
      logger.write("NN rows: " + Global::int64ToString(nnEvals[i]->numRowsProcessed()));
      logger.write("NN batches: " + Global::int64ToString(nnEvals[i]->numBatchesProcessed()));
      logger.write("NN avg batch size: " + Global::doubleToString(nnEvals[i]->averageProcessedBatchSize()));
      delete nnEvals[i];
    }
  }
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();

  if(sigReceived.load())
    logger.write("Exited cleanly after signal");
  logger.write("All cleaned up, quitting");
  return 0;
}
