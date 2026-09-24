#ifndef GAME_BOARDHISTORY_H_
#define GAME_BOARDHISTORY_H_

#include "../core/global.h"
#include "../core/hash.h"
#include "../game/board.h"
#include "../game/rules.h"

#include <vector>
#include <iostream>

struct KoHashTable;

struct BoardHistoryModes {
  bool alwaysComputePassAliveUnderSuicideRules;
  bool excludeTerritoryAdjacentToAtari;

  BoardHistoryModes();
  BoardHistoryModes(bool alwaysComputePassAliveUnderSuicideRules, bool excludeTerritoryAdjacentToAtari);

  bool operator==(const BoardHistoryModes& other) const;
  bool operator!=(const BoardHistoryModes& other) const;
};

struct BoardHistory {
  Rules rules;

  // Chronological history of moves
  std::vector<Move> moveHistory;

  // Initial board and player
  Board initialBoard;
  Player initialPla;
  int64_t initialTurnNumber;
  bool assumeMultipleStartingBlackMovesAreHandicap;
  bool whiteHasMoved;
  int overrideNumHandicapStones;

  BoardHistoryModes modes;

  static const int NUM_RECENT_BOARDS = 6;
  Board recentBoards[NUM_RECENT_BOARDS];
  int currentRecentBoardIdx;
  Player presumedNextMovePla;

  bool isGameFinished;
  Player winner;
  bool isNoResult;
  bool isResignation;

  // Compatibility stubs for unchanged modules until Step 7-10
  int initialEncorePhase;
  int encorePhase;
  int numTurnsThisPhase;
  int numApproxValidTurnsThisPhase;
  int numConsecValidTurnsThisGame;
  int consecutiveEndingPasses;
  bool isPastNormalPhaseEnd;
  bool isScored;
  float finalWhiteMinusBlackScore;
  float whiteBonusScore;
  float whiteHandicapBonusScore;
  bool hasButton;

  bool wasEverOccupiedOrPlayed[Board::MAX_ARR_SIZE];
  bool superKoBanned[Board::MAX_ARR_SIZE];
  bool koRecapBlocked[Board::MAX_ARR_SIZE];
  Hash128 koRecapBlockHash;
  Color secondEncoreStartColors[Board::MAX_ARR_SIZE];

  std::vector<bool> preventEncoreHistory;
  std::vector<Hash128> koHashHistory;
  size_t firstTurnIdxWithKoHistory;
  std::vector<Hash128> hashesBeforeBlackPass;
  std::vector<Hash128> hashesBeforeWhitePass;

  STRUCT_NAMED_TRIPLE(Hash128,posHashBeforeMove,Loc,moveLoc,Player,movePla,EncoreKoCapture);
  std::vector<EncoreKoCapture> koCapturesInEncore;

  BoardHistory();
  ~BoardHistory();

  BoardHistory(const Board& board, Player pla);
  BoardHistory(const Board& board, Player pla, const Rules& rules, int encorePhase, const BoardHistoryModes& modes);

  BoardHistory(const BoardHistory& other);
  BoardHistory& operator=(const BoardHistory& other);

  BoardHistory(BoardHistory&& other) noexcept;
  BoardHistory& operator=(BoardHistory&& other) noexcept;

  // Clears all history and status, sets rules
  void clear(const Board& board, Player pla);
  void clear(const Board& board, Player pla, const Rules& rules, int encorePhase);
  void setKomi(float newKomi);
  void setInitialTurnNumber(int64_t n);
  void setAssumeMultipleStartingBlackMovesAreHandicap(bool b);
  void setOverrideNumHandicapStones(int n);
  void setModes(const BoardHistoryModes& modes);
  bool suicideLegalForPassAlive() const;

  BoardHistory copyToInitial() const;

  float whiteKomiAdjustmentForDraws(double drawEquivalentWinsForWhite) const;
  float currentSelfKomi(Player pla, double drawEquivalentWinsForWhite) const;

  const Board& getRecentBoard(int numMovesAgo) const;

  bool isLegal(const Board& board, Loc moveLoc, Player movePla) const;
  bool passWouldEndPhase(const Board& board, Player movePla) const;
  bool passWouldEndGame(const Board& board, Player movePla) const;
  bool shouldSuppressEndGameFromFriendlyPass(const Board& board, Player movePla) const;

  bool isFinalPhase() const;
  bool isPassForKo(const Board& board, Loc moveLoc, Player movePla) const;

  int64_t getCurrentTurnNumber() const;

  // Core Quoridor move execution and terminal check
  void makeBoardMoveAssumeLegal(Board& board, Loc moveLoc, Player movePla, const KoHashTable* rootKoHashTable, bool preventEncore = false);
  bool makeBoardMoveTolerant(Board& board, Loc moveLoc, Player movePla);
  bool makeBoardMoveTolerant(Board& board, Loc moveLoc, Player movePla, bool preventEncore);
  bool isLegalTolerant(const Board& board, Loc moveLoc, Player movePla) const;

  void endGameIfAllPassAlive(const Board& board);
  void endAndScoreGameNow(const Board& board);
  void endAndScoreGameNow(const Board& board, Color area[Board::MAX_ARR_SIZE]);
  void getAreaNow(const Board& board, Color area[Board::MAX_ARR_SIZE]) const;

  void setWinnerByResignation(Player pla);

  void printBasicInfo(std::ostream& out, const Board& board) const;
  void printDebugInfo(std::ostream& out, const Board& board) const;
  int numberOfKoHashOccurrencesInHistory(Hash128 koHash, const KoHashTable* rootKoHashTable) const;

  static int numHandicapStonesOnBoard(const Board& b);
  int computeNumHandicapStones() const;
  int computeWhiteHandicapBonus() const;

  bool hasBlackPassOrWhiteFirst() const;

  static Hash128 getSituationAndSimpleKoAndPrevPosHash(const Board& board, const BoardHistory& hist, Player nextPlayer);
  static Hash128 getSituationRulesAndKoHash(const Board& board, const BoardHistory& hist, Player nextPlayer, double drawEquivalentWinsForWhite);
  static Hash128 getSituationRulesAndKoHash(const Board& board, const BoardHistory& hist, Player nextPlayer, double drawEquivalentWinsForWhite, const BoardHistoryModes& modes);
};

struct KoHashTable {
  uint32_t* idxTable;
  std::vector<Hash128> koHashHistorySortedByLowBits;
  size_t firstTurnIdxWithKoHistory;

  static const int TABLE_SIZE = 1 << 10;
  static const uint64_t TABLE_MASK = TABLE_SIZE-1;

  KoHashTable();
  ~KoHashTable();

  KoHashTable(const KoHashTable& other) = delete;
  KoHashTable& operator=(const KoHashTable& other) = delete;

  size_t size() const;
  void recompute(const BoardHistory& history);
  bool containsHash(Hash128 hash) const;
  int numberOfOccurrencesOfHash(Hash128 hash) const;
};

#endif  // GAME_BOARDHISTORY_H_
