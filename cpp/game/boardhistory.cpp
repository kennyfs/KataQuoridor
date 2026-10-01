#include "../game/boardhistory.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
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
    isRepetitionDrawFlag(false),
    positionsSinceLastWall(),
    currentRepetitionCount(1),
    initialEncorePhase(0),
    encorePhase(0),
    numTurnsThisPhase(0),
    numApproxValidTurnsThisPhase(0),
    numConsecValidTurnsThisGame(0),
    consecutiveEndingPasses(0),
    isPastNormalPhaseEnd(false),
    isScored(false),
    finalWhiteMinusBlackScore(0.0f),
    finalWhiteLead(0.0f),
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
  resetRepetitions(initialBoard, initialPla);
}

BoardHistory::~BoardHistory() {}

BoardHistory::BoardHistory(const Board& board, Player pla)
  : BoardHistory(board, pla, Rules::getQuoridorRules(), 0, BoardHistoryModes())
{}

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
    isRepetitionDrawFlag(false),
    positionsSinceLastWall(),
    currentRepetitionCount(1),
    initialEncorePhase(ePhase),
    encorePhase(ePhase),
    numTurnsThisPhase(0),
    numApproxValidTurnsThisPhase(0),
    numConsecValidTurnsThisGame(0),
    consecutiveEndingPasses(0),
    isPastNormalPhaseEnd(false),
    isScored(false),
    finalWhiteMinusBlackScore(0.0f),
    finalWhiteLead(0.0f),
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
  resetRepetitions(board, pla);
}

BoardHistory::BoardHistory(const BoardHistory& other) = default;
BoardHistory& BoardHistory::operator=(const BoardHistory& other) = default;
BoardHistory::BoardHistory(BoardHistory&& other) noexcept = default;
BoardHistory& BoardHistory::operator=(BoardHistory&& other) noexcept = default;

void BoardHistory::clear(const Board& board, Player pla) {
  clear(board, pla, Rules::getQuoridorRules(), 0);
}

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
  isRepetitionDrawFlag = false;
  presumedNextMovePla = pla;
  whiteHasMoved = false;

  currentRecentBoardIdx = 0;
  for(int i = 0; i < NUM_RECENT_BOARDS; i++)
    recentBoards[i] = board;
  resetRepetitions(board, pla);

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
  finalWhiteLead = 0.0f;
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

//Which pawn is on its goal row, if any.
static Player pawnOnGoal(const Board& board) {
  if(Location::getY(board.blackPawnLoc, board.x_size) == 0)
    return P_BLACK;
  if(Location::getY(board.whitePawnLoc, board.x_size) == board.y_size - 1)
    return P_WHITE;
  return C_EMPTY;
}

void BoardHistory::setKomi(float newKomi) {
  if(!Rules::isValidKomi(newKomi))
    throw StringError(
      "Invalid komi " + Global::floatToString(newKomi) + ": must be a half-integer (n + 0.5) with |komi| <= " +
      Global::floatToString(Rules::MAX_KOMI)
    );
  rules.komi = newKomi;

  //Recompute the result of a game that ended at a goal, since the winner depends on komi.
  if(isGameFinished && isScored && !isResignation && !isNoResult) {
    const Board& board = getRecentBoard(0);
    Player arrived = pawnOnGoal(board);
    if(arrived != C_EMPTY)
      scoreGameEndedAtGoal(board, arrived);
  }
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

//As upstream. For Quoridor this is always 0: the score never decides a draw (gameResultWillBeInteger() is false),
//so a draw's utility goes only through ScoreValue::whiteWinsOfWinner.
float BoardHistory::whiteKomiAdjustmentForDraws(double drawEquivalentWinsForWhite) const {
  float drawAdjustment = rules.gameResultWillBeInteger() ? (float)(drawEquivalentWinsForWhite - 0.5) : 0.0f;
  return drawAdjustment;
}

//As upstream: komi from pla's perspective.
float BoardHistory::currentSelfKomi(Player pla, double drawEquivalentWinsForWhite) const {
  float whiteKomiAdjusted = whiteBonusScore + whiteHandicapBonusScore + rules.komi + whiteKomiAdjustmentForDraws(drawEquivalentWinsForWhite);

  if(pla == P_WHITE)
    return whiteKomiAdjusted;
  else if(pla == P_BLACK)
    return -whiteKomiAdjusted;
  else {
    ASSERT_UNREACHABLE;
  }
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

int64_t BoardHistory::pliesUntilDraw() const {
  return std::max((int64_t)0, (int64_t)rules.maxPlies - getCurrentTurnNumber());
}

bool BoardHistory::isDraw() const {
  return isGameFinished && !isNoResult && !isResignation && winner == C_EMPTY;
}

bool BoardHistory::isRepetitionDraw() const {
  return isDraw() && isRepetitionDrawFlag;
}

bool BoardHistory::isMaxPliesDraw() const {
  return isDraw() && !isRepetitionDrawFlag && getCurrentTurnNumber() >= rules.maxPlies;
}

void BoardHistory::resetRepetitions(const Board& board, Player pla) {
  positionsSinceLastWall.clear();
  positionsSinceLastWall.push_back(board.getSitHash(pla));
  currentRepetitionCount = 1;
}

void BoardHistory::endAsRuleDraw(bool byRepetition) {
  isGameFinished = true;
  winner = C_EMPTY;
  isNoResult = false;
  isResignation = false;
  isScored = true;
  isRepetitionDrawFlag = byRepetition;
  finalWhiteMinusBlackScore = 0.0f;
  finalWhiteLead = 0.0f;
}

void BoardHistory::scoreGameEndedAtGoal(const Board& board, Player arrived) {
  assert(arrived == P_BLACK || arrived == P_WHITE);
  double whiteMargin = board.whiteMarginWhenWonBy(arrived);
  double tempo = arrived == P_WHITE ? whiteMargin : whiteMargin + 1.0;
  double lead = tempo + rules.komi;
  //Komi is a half-integer, so the lead is never 0.
  assert(lead != 0.0);
  double bonus = (double)rules.timeBonusPerPly * (double)std::max((int64_t)0, (int64_t)rules.maxPlies - getCurrentTurnNumber());
  isGameFinished = true;
  isNoResult = false;
  isResignation = false;
  isScored = true;
  winner = lead > 0 ? P_WHITE : P_BLACK;
  finalWhiteLead = (float)lead;
  finalWhiteMinusBlackScore = (float)(lead > 0 ? lead + bonus : lead - bonus);
}

void BoardHistory::makeBoardMoveAssumeLegal(
    Board& board, Loc moveLoc, Player movePla,
    const KoHashTable* rootKoHashTable, bool preventEncore) {
  (void)rootKoHashTable;

  // 0. Reset any previous terminal state
  isGameFinished = false;
  winner = C_EMPTY;
  isNoResult = false;
  isResignation = false;
  isRepetitionDrawFlag = false;
  isScored = false;
  finalWhiteMinusBlackScore = 0.0f;
  finalWhiteLead = 0.0f;

  // 1. Execute the move
  board.playMoveAssumeLegal(moveLoc, movePla);

  // 2. Update move history
  moveHistory.push_back({moveLoc, movePla});
  preventEncoreHistory.push_back(preventEncore);
  numTurnsThisPhase += 1;
  numApproxValidTurnsThisPhase += 1;
  numConsecValidTurnsThisGame += 1;

  // 3. Update recentBoards ring buffer
  currentRecentBoardIdx = (currentRecentBoardIdx + 1) % NUM_RECENT_BOARDS;
  recentBoards[currentRecentBoardIdx] = board;

  presumedNextMovePla = getOpp(movePla);
  if(movePla == P_WHITE)
    whiteHasMoved = true;

  // 4. Repetition counts. A wall placement makes every earlier position unreachable (walls are never removed).
  Hash128 posHash = board.getSitHash(presumedNextMovePla);
  if(Location::isHWallLoc(moveLoc, board.x_size) || Location::isVWallLoc(moveLoc, board.x_size)) {
    positionsSinceLastWall.clear();
    currentRepetitionCount = 1;
  }
  else {
    int count = 1;
    for(const Hash128& h : positionsSinceLastWall) {
      if(h == posHash)
        count++;
    }
    currentRepetitionCount = count;
  }
  positionsSinceLastWall.push_back(posHash);

  // 5. Terminal conditions. A pawn on its goal ends the game (komi decides the winner, see Rules), even on the
  // last ply before the draw. Otherwise the repetitionDrawCount-th occurrence of a position, or reaching
  // rules.maxPlies plies, is a draw (a repetition on the last ply counts as a repetition draw).
  Player arrived = pawnOnGoal(board);
  if(arrived != C_EMPTY) {
    scoreGameEndedAtGoal(board, arrived);
    return;
  }
  if(rules.repetitionDrawCount > 0 && currentRepetitionCount >= rules.repetitionDrawCount) {
    endAsRuleDraw(true);
    return;
  }
  if(getCurrentTurnNumber() >= rules.maxPlies) {
    endAsRuleDraw(false);
    return;
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
void BoardHistory::endAndScoreGameNow(const Board&) {
  if(!isGameFinished)
    endAsRuleDraw(false);
}
void BoardHistory::endAndScoreGameNow(const Board& board, Color area[Board::MAX_ARR_SIZE]) {
  endAndScoreGameNow(board);
  getAreaNow(board, area);
}

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
  out << "Ply: " << getCurrentTurnNumber() << " (draw at " << rules.maxPlies << ", " << pliesUntilDraw() << " left)" << endl;
  if(rules.komi != Rules::DEFAULT_KOMI)
    out << "Komi: " << rules.komi << endl;
  if(rules.timeBonusPerPly != 0.0f)
    out << "Time bonus per ply: " << rules.timeBonusPerPly << endl;
  if(rules.repetitionDrawCount > 0)
    out << "Repetition draw at occurrence " << rules.repetitionDrawCount << " (current position: " << currentRepetitionCount << ")" << endl;
  if(isGameFinished) {
    if(winner == C_EMPTY) {
      // The repetition or maxPlies draw rule, or a game cut off by a controller (endAndScoreGameNow).
      if(isRepetitionDrawFlag)
        out << "Game finished: Draw (repetition, occurrence " << currentRepetitionCount << ")" << (isNoResult ? " (NoResult)" : "") << endl;
      else if(getCurrentTurnNumber() >= rules.maxPlies)
        out << "Game finished: Draw (" << rules.maxPlies << "-ply limit)" << (isNoResult ? " (NoResult)" : "") << endl;
      else
        out << "Game finished: Draw (move cutoff)" << (isNoResult ? " (NoResult)" : "") << endl;
    }
    else {
      out << "Game finished: winner = " << PlayerIO::playerToString(winner)
          << (isNoResult ? " (Draw/NoResult)" : "")
          << (isResignation ? " (Resignation)" : "");
      if(isScored && !isResignation && !isNoResult)
        out << " (" << (winner == P_WHITE ? "W+" : "B+") << std::fabs(finalWhiteLead) << ")";
      out << endl;
    }
  }
}

void BoardHistory::printDebugInfo(ostream& out, const Board& board) const {
  out << board << endl;
  out << "Initial pla " << PlayerIO::playerToString(initialPla) << endl;
  out << "Rules " << rules << endl;
  out << "Presumed next pla " << PlayerIO::playerToString(presumedNextMovePla) << endl;
  out << "Moves played: " << moveHistory.size() << endl;
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

Hash128 BoardHistory::getSituationAndSimpleKoAndPrevPosHash(const Board& board, const BoardHistory&, Player nextPlayer) {
  return board.getSitHash(nextPlayer);
}

static uint64_t floatBits(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  return bits;
}

//Besides the board and the side to move, the value of a Quoridor position depends on the rules that score it
//(komi, maxPlies, repetitionDrawCount, the time bonus) and on the ply count (the maxPlies draw and the time bonus). Pawn moves are
//reversible, so the same board recurs at different plies, and those must not share search nodes or (for nets
//that see the ply count or komi) NN evaluations. Both the graph search hash and the NN cache use this.
Hash128 BoardHistory::getSituationRulesAndKoHash(
    const Board& board, const BoardHistory& hist,
    Player nextPlayer, double) {
  Hash128 hash = board.getSitHash(nextPlayer);
  uint64_t h = Hash::splitMix64((uint64_t)hist.getCurrentTurnNumber());
  h = Hash::splitMix64(h ^ (uint64_t)(uint32_t)hist.rules.maxPlies);
  h = Hash::splitMix64(h ^ (uint64_t)(uint32_t)hist.rules.repetitionDrawCount);
  h = Hash::splitMix64(h ^ floatBits(hist.rules.komi));
  h = Hash::splitMix64(h ^ floatBits(hist.rules.timeBonusPerPly));
  hash.hash0 ^= h;
  hash.hash1 ^= Hash::nasam(h);
  return hash;
}

Hash128 BoardHistory::getSituationRulesAndKoHash(
    const Board& board, const BoardHistory& hist,
    Player nextPlayer, double drawEquivalentWinsForWhite, const BoardHistoryModes&) {
  return getSituationRulesAndKoHash(board, hist, nextPlayer, drawEquivalentWinsForWhite);
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
