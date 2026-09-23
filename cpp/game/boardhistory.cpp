#include "../game/boardhistory.h"

#include <algorithm>
#include <cassert>
#include "../core/test.h"

using namespace std;

BoardHistoryModes::BoardHistoryModes()
  : alwaysComputePassAliveUnderSuicideRules(false),
    excludeTerritoryAdjacentToAtari(false)
{}

BoardHistoryModes::BoardHistoryModes(bool alwaysPassAliveSuicide, bool excludeTerritoryAdjAtari)
  : alwaysComputePassAliveUnderSuicideRules(alwaysPassAliveSuicide),
    excludeTerritoryAdjacentToAtari(excludeTerritoryAdjAtari)
{}

bool BoardHistoryModes::operator==(const BoardHistoryModes& other) const {
  return alwaysComputePassAliveUnderSuicideRules == other.alwaysComputePassAliveUnderSuicideRules
    && excludeTerritoryAdjacentToAtari == other.excludeTerritoryAdjacentToAtari;
}

bool BoardHistoryModes::operator!=(const BoardHistoryModes& other) const {
  return !(*this == other);
}

BoardHistory::BoardHistory()
  : rules(),
    moveHistory(),
    initialBoard(),
    initialPla(P_BLACK),
    initialTurnNumber(0),
    assumeMultipleStartingBlackMovesAreHandicap(false),
    whiteHasMoved(false),
    overrideNumHandicapStones(-1),
    modes(),
    recentBoards(),
    currentRecentBoardIdx(0),
    presumedNextMovePla(P_BLACK),
    isGameFinished(false),
    winner(C_EMPTY),
    isNoResult(false),
    isResignation(false),
    initialEncorePhase(0),
    encorePhase(0),
    numTurnsThisPhase(0),
    numApproxValidTurnsThisPhase(0),
    numConsecValidTurnsThisGame(0),
    consecutiveEndingPasses(0),
    isPastNormalPhaseEnd(false),
    isScored(false),
    finalWhiteMinusBlackScore(0.0f),
    whiteBonusScore(0.0f),
    whiteHandicapBonusScore(0.0f),
    hasButton(false),
    koRecapBlockHash(),
    preventEncoreHistory(),
    koHashHistory(),
    firstTurnIdxWithKoHistory(0),
    hashesBeforeBlackPass(),
    hashesBeforeWhitePass(),
    koCapturesInEncore()
{
  std::fill(wasEverOccupiedOrPlayed, wasEverOccupiedOrPlayed + Board::MAX_ARR_SIZE, false);
  std::fill(superKoBanned, superKoBanned + Board::MAX_ARR_SIZE, false);
  std::fill(koRecapBlocked, koRecapBlocked + Board::MAX_ARR_SIZE, false);
  std::fill(secondEncoreStartColors, secondEncoreStartColors + Board::MAX_ARR_SIZE, C_EMPTY);
  for(int i = 0; i < NUM_RECENT_BOARDS; i++)
    recentBoards[i] = initialBoard;
}

BoardHistory::~BoardHistory() {}

BoardHistory::BoardHistory(const Board& board, Player pla, const Rules& r, int ePhase, const BoardHistoryModes& modes_)
  : rules(r),
    moveHistory(),
    initialBoard(board),
    initialPla(pla),
    initialTurnNumber(0),
    assumeMultipleStartingBlackMovesAreHandicap(false),
    whiteHasMoved(false),
    overrideNumHandicapStones(-1),
    modes(modes_),
    recentBoards(),
    currentRecentBoardIdx(0),
    presumedNextMovePla(pla),
    isGameFinished(false),
    winner(C_EMPTY),
    isNoResult(false),
    isResignation(false),
    initialEncorePhase(ePhase),
    encorePhase(ePhase),
    numTurnsThisPhase(0),
    numApproxValidTurnsThisPhase(0),
    numConsecValidTurnsThisGame(0),
    consecutiveEndingPasses(0),
    isPastNormalPhaseEnd(false),
    isScored(false),
    finalWhiteMinusBlackScore(0.0f),
    whiteBonusScore(0.0f),
    whiteHandicapBonusScore(0.0f),
    hasButton(false),
    koRecapBlockHash(),
    preventEncoreHistory(),
    koHashHistory(),
    firstTurnIdxWithKoHistory(0),
    hashesBeforeBlackPass(),
    hashesBeforeWhitePass(),
    koCapturesInEncore()
{
  std::fill(wasEverOccupiedOrPlayed, wasEverOccupiedOrPlayed + Board::MAX_ARR_SIZE, false);
  std::fill(superKoBanned, superKoBanned + Board::MAX_ARR_SIZE, false);
  std::fill(koRecapBlocked, koRecapBlocked + Board::MAX_ARR_SIZE, false);
  std::fill(secondEncoreStartColors, secondEncoreStartColors + Board::MAX_ARR_SIZE, C_EMPTY);
  for(int i = 0; i < NUM_RECENT_BOARDS; i++)
    recentBoards[i] = board;
}

BoardHistory::BoardHistory(const BoardHistory& other) = default;
BoardHistory& BoardHistory::operator=(const BoardHistory& other) = default;
BoardHistory::BoardHistory(BoardHistory&& other) noexcept = default;
BoardHistory& BoardHistory::operator=(BoardHistory&& other) noexcept = default;

void BoardHistory::clear(const Board& board, Player pla, const Rules& r, int encorePhase_) {
  (void)encorePhase_;
  rules = r;
  moveHistory.clear();
  initialBoard = board;
  initialPla = pla;
  initialTurnNumber = 0;
  isGameFinished = false;
  winner = C_EMPTY;
  isNoResult = false;
  isResignation = false;
  presumedNextMovePla = pla;
  whiteHasMoved = false;

  currentRecentBoardIdx = 0;
  for(int i = 0; i < NUM_RECENT_BOARDS; i++)
    recentBoards[i] = board;

  // Reset compatibility stubs
  initialEncorePhase = 0;
  encorePhase = 0;
  numTurnsThisPhase = 0;
  numApproxValidTurnsThisPhase = 0;
  numConsecValidTurnsThisGame = 0;
  consecutiveEndingPasses = 0;
  isPastNormalPhaseEnd = false;
  isScored = false;
  finalWhiteMinusBlackScore = 0.0f;
  whiteBonusScore = 0.0f;
  whiteHandicapBonusScore = 0.0f;
  hasButton = false;
  koRecapBlockHash = Hash128();
  preventEncoreHistory.clear();
  koHashHistory.clear();
  firstTurnIdxWithKoHistory = 0;
  hashesBeforeBlackPass.clear();
  hashesBeforeWhitePass.clear();
  koCapturesInEncore.clear();
  std::fill(wasEverOccupiedOrPlayed, wasEverOccupiedOrPlayed + Board::MAX_ARR_SIZE, false);
  std::fill(superKoBanned, superKoBanned + Board::MAX_ARR_SIZE, false);
  std::fill(koRecapBlocked, koRecapBlocked + Board::MAX_ARR_SIZE, false);
  std::fill(secondEncoreStartColors, secondEncoreStartColors + Board::MAX_ARR_SIZE, C_EMPTY);
}

void BoardHistory::setKomi(float newKomi) {
  (void)newKomi;
}

void BoardHistory::setInitialTurnNumber(int64_t n) {
  initialTurnNumber = n;
}

void BoardHistory::setAssumeMultipleStartingBlackMovesAreHandicap(bool b) {
  assumeMultipleStartingBlackMovesAreHandicap = b;
}

void BoardHistory::setOverrideNumHandicapStones(int n) {
  overrideNumHandicapStones = n;
}

void BoardHistory::setModes(const BoardHistoryModes& m) {
  modes = m;
}

bool BoardHistory::suicideLegalForPassAlive() const {
  return false;
}

BoardHistory BoardHistory::copyToInitial() const {
  BoardHistory hist(initialBoard, initialPla, rules, initialEncorePhase, modes);
  hist.setInitialTurnNumber(initialTurnNumber);
  hist.setAssumeMultipleStartingBlackMovesAreHandicap(assumeMultipleStartingBlackMovesAreHandicap);
  hist.setOverrideNumHandicapStones(overrideNumHandicapStones);
  return hist;
}

float BoardHistory::whiteKomiAdjustmentForDraws(double drawEquivalentWinsForWhite) const {
  (void)drawEquivalentWinsForWhite;
  return 0.0f;
}

float BoardHistory::currentSelfKomi(Player pla, double drawEquivalentWinsForWhite) const {
  (void)pla;
  (void)drawEquivalentWinsForWhite;
  return 0.0f;
}

const Board& BoardHistory::getRecentBoard(int numMovesAgo) const {
  assert(numMovesAgo >= 0 && numMovesAgo < NUM_RECENT_BOARDS);
  int idx = (currentRecentBoardIdx - numMovesAgo + NUM_RECENT_BOARDS) % NUM_RECENT_BOARDS;
  return recentBoards[idx];
}

bool BoardHistory::isLegal(const Board& board, Loc moveLoc, Player movePla) const {
  if(isGameFinished)
    return false;
  if(movePla != presumedNextMovePla)
    return false;
  return board.isLegal(moveLoc, movePla);
}

bool BoardHistory::passWouldEndPhase(const Board&, Player) const {
  return false;
}

bool BoardHistory::passWouldEndGame(const Board&, Player) const {
  return false;
}

bool BoardHistory::shouldSuppressEndGameFromFriendlyPass(const Board&, Player) const {
  return false;
}

bool BoardHistory::isFinalPhase() const {
  return true;
}

bool BoardHistory::isPassForKo(const Board&, Loc, Player) const {
  return false;
}

int64_t BoardHistory::getCurrentTurnNumber() const {
  return initialTurnNumber + (int64_t)moveHistory.size();
}

void BoardHistory::makeBoardMoveAssumeLegal(
    Board& board, Loc moveLoc, Player movePla,
    const KoHashTable* rootKoHashTable, bool preventEncore) {
  (void)rootKoHashTable;
  (void)preventEncore;

  // 0. Reset any previous terminal state
  isGameFinished = false;
  winner = C_EMPTY;
  isNoResult = false;
  isResignation = false;

  // 1. Execute the move
  board.playMoveAssumeLegal(moveLoc, movePla);

  // 2. Update move history
  moveHistory.push_back({moveLoc, movePla});

  // 3. Update recentBoards ring buffer
  currentRecentBoardIdx = (currentRecentBoardIdx + 1) % NUM_RECENT_BOARDS;
  recentBoards[currentRecentBoardIdx] = board;

  presumedNextMovePla = getOpp(movePla);
  if(movePla == P_WHITE)
    whiteHasMoved = true;

  // 4. Terminal condition: victory check
  int blackY = Location::getY(board.blackPawnLoc, board.x_size);
  int whiteY = Location::getY(board.whitePawnLoc, board.x_size);
  if(blackY == 0) {
    isGameFinished = true;
    winner = P_BLACK;
    isNoResult = false;
    return;
  }
  if(whiteY == board.y_size - 1) {
    isGameFinished = true;
    winner = P_WHITE;
    isNoResult = false;
    return;
  }

  // 5. Terminal condition: 200-step draw
  if((int)moveHistory.size() >= rules.maxMovesPerGame) {
    isGameFinished = true;
    winner = C_EMPTY;
    isNoResult = true;
  }
}

bool BoardHistory::makeBoardMoveTolerant(Board& board, Loc moveLoc, Player movePla) {
  return makeBoardMoveTolerant(board, moveLoc, movePla, false);
}

bool BoardHistory::makeBoardMoveTolerant(Board& board, Loc moveLoc, Player movePla, bool preventEncore) {
  if(!isLegalTolerant(board, moveLoc, movePla))
    return false;
  makeBoardMoveAssumeLegal(board, moveLoc, movePla, NULL, preventEncore);
  return true;
}

bool BoardHistory::isLegalTolerant(const Board& board, Loc moveLoc, Player movePla) const {
  if(movePla != P_BLACK && movePla != P_WHITE)
    return false;
  return board.isLegal(moveLoc, movePla);
}

void BoardHistory::endGameIfAllPassAlive(const Board&) {}
void BoardHistory::endAndScoreGameNow(const Board&) {}
void BoardHistory::endAndScoreGameNow(const Board&, Color[Board::MAX_ARR_SIZE]) {}

void BoardHistory::getAreaNow(const Board&, Color area[Board::MAX_ARR_SIZE]) const {
  std::fill(area, area + Board::MAX_ARR_SIZE, C_EMPTY);
}

void BoardHistory::setWinnerByResignation(Player pla) {
  isGameFinished = true;
  winner = pla;
  isResignation = true;
  isNoResult = false;
}

void BoardHistory::printBasicInfo(ostream& out, const Board& board) const {
  Board::printBoard(out, board, Board::NULL_LOC, &moveHistory);
  out << "Next player: " << PlayerIO::playerToString(presumedNextMovePla) << endl;
  out << "Rules: " << rules.toJsonString() << endl;
  out << "Moves played: " << moveHistory.size() << " / " << rules.maxMovesPerGame << endl;
  if(isGameFinished) {
    out << "Game finished: winner = " << PlayerIO::playerToString(winner)
        << (isNoResult ? " (Draw/NoResult)" : "")
        << (isResignation ? " (Resignation)" : "") << endl;
  }
}

void BoardHistory::printDebugInfo(ostream& out, const Board& board) const {
  out << board << endl;
  out << "Initial pla " << PlayerIO::playerToString(initialPla) << endl;
  out << "Rules " << rules << endl;
  out << "Presumed next pla " << PlayerIO::playerToString(presumedNextMovePla) << endl;
  out << "Moves played: " << moveHistory.size() << " / " << rules.maxMovesPerGame << endl;
  out << "Game result " << isGameFinished << " " << PlayerIO::playerToString(winner)
      << " isNoResult=" << isNoResult << " isResignation=" << isResignation << endl;
  out << "Last moves ";
  for(size_t i = 0; i < moveHistory.size(); i++)
    out << Location::toString(moveHistory[i].loc, board) << " ";
  out << endl;
}

int BoardHistory::numberOfKoHashOccurrencesInHistory(Hash128, const KoHashTable*) const {
  return 0;
}

int BoardHistory::numHandicapStonesOnBoard(const Board&) {
  return 0;
}

int BoardHistory::computeNumHandicapStones() const {
  return 0;
}

int BoardHistory::computeWhiteHandicapBonus() const {
  return 0;
}

bool BoardHistory::hasBlackPassOrWhiteFirst() const {
  return false;
}

Hash128 BoardHistory::getSituationAndSimpleKoHash(const Board& board, Player nextPlayer) {
  return board.getSitHash(nextPlayer);
}

Hash128 BoardHistory::getSituationAndSimpleKoAndPrevPosHash(const Board& board, const BoardHistory&, Player nextPlayer) {
  return board.getSitHash(nextPlayer);
}

Hash128 BoardHistory::getSituationRulesAndKoHash(
    const Board& board, const BoardHistory&,
    Player nextPlayer, double) {
  return board.getSitHash(nextPlayer);
}

Hash128 BoardHistory::getSituationRulesAndKoHash(
    const Board& board, const BoardHistory&,
    Player nextPlayer, double, const BoardHistoryModes&) {
  return board.getSitHash(nextPlayer);
}

// ------------------------------------------------------------------------
// KoHashTable implementation
// ------------------------------------------------------------------------

KoHashTable::KoHashTable()
  : idxTable(NULL),
    koHashHistorySortedByLowBits(),
    firstTurnIdxWithKoHistory(0)
{}

KoHashTable::~KoHashTable() {
  if(idxTable)
    delete[] idxTable;
}

size_t KoHashTable::size() const {
  return 0;
}

void KoHashTable::recompute(const BoardHistory&) {}

bool KoHashTable::containsHash(Hash128) const {
  return false;
}

int KoHashTable::numberOfOccurrencesOfHash(Hash128) const {
  return 0;
}
