#include "../core/global.h"
#include "../core/rand.h"
#include "../external/nlohmann_json/json.hpp"
#include "../main.h"
#include "q4board.h"
#include "q4bots.h"
#include "q4history.h"
#include "q4record.h"
#include "q4rules.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

int MainCmds::q4match(const std::vector<std::string>& args) {
  std::vector<std::string> subArgs = args;
  if(!subArgs.empty() && subArgs[0] == "q4match")
    subArgs = std::vector<std::string>(subArgs.begin() + 1, subArgs.end());

  int numGames = 1000;
  std::vector<std::string> botTypes = {"random", "random", "random", "random"};
  std::string outputFile = "";
  uint64_t masterSeed = 12345;
  int maxPlies = 400;

  for(size_t i = 0; i < subArgs.size(); i++) {
    if(subArgs[i] == "-games" && i + 1 < subArgs.size()) {
      numGames = Global::stringToInt(subArgs[++i]);
    }
    else if(subArgs[i] == "-bots" && i + 1 < subArgs.size()) {
      botTypes = Global::split(subArgs[++i], ',');
    }
    else if(subArgs[i] == "-output" && i + 1 < subArgs.size()) {
      outputFile = subArgs[++i];
    }
    else if(subArgs[i] == "-seed" && i + 1 < subArgs.size()) {
      masterSeed = (uint64_t)Global::stringToInt64(subArgs[++i]);
    }
    else if(subArgs[i] == "-maxplies" && i + 1 < subArgs.size()) {
      maxPlies = Global::stringToInt(subArgs[++i]);
    }
  }

  while(botTypes.size() < 4) {
    botTypes.push_back("random");
  }

  std::cout << "=== KataQuoridor Q4 Match ===" << std::endl;
  std::cout << "Games: " << numGames << std::endl;
  std::cout << "Bots: P1=" << botTypes[0] << ", P2=" << botTypes[1]
            << ", P3=" << botTypes[2] << ", P4=" << botTypes[3] << std::endl;
  std::cout << "Max plies: " << maxPlies << std::endl;

  std::ofstream outStream;
  if(!outputFile.empty()) {
    outStream.open(outputFile);
    if(!outStream.is_open()) {
      std::cerr << "Error: Could not open output file " << outputFile << std::endl;
      return 1;
    }
  }

  int winCounts[4] = {0, 0, 0, 0};
  int drawCount = 0;
  int minPlies = 999999;
  int maxObservedPlies = 0;
  int64_t totalPlies = 0;

  Q4Rules rules;
  rules.maxPlies = maxPlies;

  Rand masterRand(masterSeed);

  for(int g = 0; g < numGames; g++) {
    // Instantiate 4 bots with unique seeds
    std::unique_ptr<Q4Bot> bots[4];
    for(int s = 0; s < 4; s++) {
      uint64_t bSeed = masterRand.nextUInt64();
      bots[s] = Q4Bots::makeBot(botTypes[s], bSeed, (s + 1) % 4);
    }

    Q4History history(rules);

    while(!history.isFinished) {
      int toMove = history.currentBoard.toMove;
      int act = bots[toMove]->getMove(history.currentBoard);
      if(act == Q4Board::NULL_ACTION)
        break;
      history.play(act);
    }

    if(history.winnerSeat >= 0 && history.winnerSeat < 4) {
      winCounts[history.winnerSeat]++;
    }
    else {
      drawCount++;
    }

    int pl = history.plies;
    totalPlies += pl;
    if(pl < minPlies) minPlies = pl;
    if(pl > maxObservedPlies) maxObservedPlies = pl;

    if(outStream.is_open()) {
      Q4Record rec;
      rec.rules = rules;
      rec.events = history.events;
      rec.result = history.getResultString();
      for(int s = 0; s < 4; s++) {
        rec.players[s].name = "P" + Global::intToString(s + 1) + "_" + botTypes[s];
        rec.players[s].type = botTypes[s];
      }
      outStream << rec.toJsonLine() << "\n";
    }

    if((g + 1) % 100 == 0 || g + 1 == numGames) {
      std::cout << "Played " << (g + 1) << "/" << numGames << " games..." << std::endl;
    }
  }

  std::cout << "\n--- Match Results (" << numGames << " games) ---" << std::endl;
  for(int s = 0; s < 4; s++) {
    double pct = 100.0 * winCounts[s] / numGames;
    std::cout << "Seat " << (s + 1) << " (" << botTypes[s] << "): "
              << winCounts[s] << " wins (" << std::fixed << std::setprecision(1) << pct << "%)" << std::endl;
  }
  double drawPct = 100.0 * drawCount / numGames;
  std::cout << "Draws: " << drawCount << " (" << std::fixed << std::setprecision(1) << drawPct << "%)" << std::endl;
  std::cout << "Game lengths (plies): min=" << minPlies
            << ", avg=" << std::setprecision(1) << ((double)totalPlies / numGames)
            << ", max=" << maxObservedPlies << std::endl;

  return 0;
}
