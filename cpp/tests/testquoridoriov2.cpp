/*
 * testquoridoriov2.cpp
 * Tests for the Quoridor I/O v2 rules and scoring layer (docs/QuoridorIOv2.md, step 1):
 *  - tempo t, komi, lead s and the winner, for wins by either colour at several margins and komis
 *  - the utility score u with a time bonus
 *  - the draw at Rules::maxPlies, counted from the real game start, and a win on the last ply
 *  - undo across a finished game (win and draw)
 *  - the fence handicap (initial walls), including self-play game setup
 *  - Rules parsing / serialization and the self-play config checks
 *  - search: terminal nodes report the lead and the utility score separately, and see the draw
 *  - SGF: KM, WB/WW (initial walls), RU and RE round trip; older SGFs load as the standard game
 */

#include "../tests/tests.h"

#include "../core/config_parser.h"
#include "../dataio/sgf.h"
#include "../program/play.h"
#include "../program/playutils.h"
#include "../program/setup.h"
#include "../search/search.h"
#include "../tests/testsearchcommon.h"

using namespace std;

namespace {

static constexpr int SIZE = Board::DEFAULT_LEN;

static bool approxEqual(double x, double y) {
  return std::fabs(x - y) < 1e-4;
}

//A pawn-only game where each player's moves are given as a string of F(orward), L(eft), R(ight); Black moves first.
//Keeps every intermediate position, so tests can replay or undo.
struct ScriptedGame {
  Board initialBoard;
  BoardHistory initialHist;
  vector<Loc> moves;
  vector<Player> plas;
  Board board;
  BoardHistory hist;
};

static Loc scriptedStep(const Board& board, Player pla, char c) {
  Loc from = pla == P_BLACK ? board.blackPawnLoc : board.whitePawnLoc;
  int x = Location::getX(from, SIZE);
  int y = Location::getY(from, SIZE);
  if(c == 'F') y += pla == P_WHITE ? 2 : -2;
  else if(c == 'L') x -= 2;
  else if(c == 'R') x += 2;
  else testAssert(false);
  return Location::getLoc(x, y, SIZE);
}

//Plays the scripts until the game ends (or the scripts run out, if allowUnfinished).
static ScriptedGame playScripted(
  const string& blackScript, const string& whiteScript, const Rules& rules, int64_t initialTurnNumber = 0,
  bool allowUnfinished = false
) {
  ScriptedGame g;
  g.initialBoard = Board();
  g.initialBoard.setFencesLeft(rules.blackInitialFences, rules.whiteInitialFences);
  g.initialHist = BoardHistory(g.initialBoard, P_BLACK, rules, 0, BoardHistoryModes(false, false));
  g.initialHist.setInitialTurnNumber(initialTurnNumber);
  g.board = g.initialBoard;
  g.hist = g.initialHist;
  Player pla = P_BLACK;
  size_t bi = 0, wi = 0;
  while(!g.hist.isGameFinished) {
    const string& script = pla == P_BLACK ? blackScript : whiteScript;
    size_t& idx = pla == P_BLACK ? bi : wi;
    if(idx >= script.size() && allowUnfinished)
      break;
    testAssert(idx < script.size());
    Loc to = scriptedStep(g.board, pla, script[idx++]);
    testAssert(g.hist.isLegal(g.board, to, pla));
    g.moves.push_back(to);
    g.plas.push_back(pla);
    g.hist.makeBoardMoveAssumeLegal(g.board, to, pla, NULL);
    pla = getOpp(pla);
  }
  return g;
}

//Replays the first n moves of g from its initial position, the way QTP undo rebuilds the game.
static void replayFirst(const ScriptedGame& g, size_t n, Board& board, BoardHistory& hist) {
  board = g.initialBoard;
  hist = g.initialHist;
  for(size_t i = 0; i < n; i++) {
    testAssert(hist.isLegal(board, g.moves[i], g.plas[i]));
    hist.makeBoardMoveAssumeLegal(board, g.moves[i], g.plas[i], NULL);
  }
}

//Games with a known end. Black starts at e9 heading down, White at e1 heading up; the pawns take different columns.
struct KnownGame {
  string name;
  string black;
  string white;
  float whiteMargin;  //Board::whiteMarginWhenWonBy of the final position
  int tempo;          //t: margin if White arrived, margin + 1 if Black did
  int plies;          //T
};
static const vector<KnownGame> KNOWN_GAMES = {
  //Black wastes 1 move (L to column d), White walks up column e: White arrives on ply 16 with Black 1 away.
  {"W+1", "LFFFFFFF", "FFFFFFFF", 1.0f, 1, 16},
  //Black wastes 3 moves.
  {"W+3", "LRLFFFFF", "FFFFFFFF", 3.0f, 3, 16},
  //Both waste 1 move (Black to column d, White to column f): an equal race, Black arrives first on ply 17.
  {"B+1", "LFFFFFFFF", "RFFFFFFF", -1.0f, 0, 17},
  //White wastes 3 moves.
  {"B+3", "LFFFFFFFF", "RLRFFFFF", -3.0f, -2, 17},
};

static Rules rulesWithKomi(float komi) {
  Rules rules = Rules::getQuoridorRules();
  rules.komi = komi;
  return rules;
}

//------------------------------------------------------------------------------------------------

static void testTempoLeadAndWinner() {
  cout << "  Tempo, komi, lead and winner" << endl;
  Rules standard = Rules::getQuoridorRules();
  testAssert(standard.komi == -0.5f);
  testAssert(standard.maxPlies == 300);
  testAssert(standard.timeBonusPerPly == 0.0f);

  for(const KnownGame& kg : KNOWN_GAMES) {
    for(float komi : {-0.5f, 0.5f, 1.5f, -1.5f}) {
      ScriptedGame g = playScripted(kg.black, kg.white, rulesWithKomi(komi));
      const BoardHistory& h = g.hist;
      testAssert(h.isGameFinished && h.isScored && !h.isNoResult && !h.isResignation && !h.isDraw());
      testAssert(h.getCurrentTurnNumber() == kg.plies);
      testAssert(g.board.whiteMarginWhenWonBy(kg.whiteMargin > 0 ? P_WHITE : P_BLACK) == kg.whiteMargin);
      float lead = kg.tempo + komi;
      testAssert(h.finalWhiteLead == lead);
      //No time bonus by default, so the score is the lead.
      testAssert(h.finalWhiteMinusBlackScore == lead);
      testAssert(h.winner == (lead > 0 ? P_WHITE : P_BLACK));
      testAssert(WriteSgf::gameResultNoSgfTag(h) == (lead > 0 ? "W+" : "B+") + Global::floatToString(std::fabs(lead)));

      //Changing komi after the game re-scores it, like upstream.
      ScriptedGame gStd = playScripted(kg.black, kg.white, standard);
      BoardHistory rescored = gStd.hist;
      rescored.setKomi(komi);
      testAssert(rescored.winner == h.winner);
      testAssert(rescored.finalWhiteLead == h.finalWhiteLead);
      testAssert(rescored.finalWhiteMinusBlackScore == h.finalWhiteMinusBlackScore);
    }
  }

  //The table of docs/QuoridorIOv2.md: standard game W+1 -> +0.5, B+1 -> -0.5, B+3 -> -2.5; with komi the side
  //whose pawn arrived can lose.
  testAssert(playScripted("LFFFFFFF", "FFFFFFFF", standard).hist.finalWhiteLead == 0.5f);
  testAssert(playScripted("LFFFFFFFF", "RFFFFFFF", standard).hist.finalWhiteLead == -0.5f);
  testAssert(playScripted("LFFFFFFFF", "RLRFFFFF", standard).hist.finalWhiteLead == -2.5f);
  {
    ScriptedGame g = playScripted("LFFFFFFFF", "RFFFFFFF", rulesWithKomi(0.5f));  //B+1, komi +0.5
    testAssert(g.hist.winner == P_WHITE && g.hist.finalWhiteLead == 0.5f);
    testAssert(Location::getY(g.board.blackPawnLoc, SIZE) == 0);  //although Black's pawn is on its goal
  }
  {
    ScriptedGame g = playScripted("LFFFFFFF", "FFFFFFFF", rulesWithKomi(-1.5f));  //W+1, komi -1.5
    testAssert(g.hist.winner == P_BLACK && g.hist.finalWhiteLead == -0.5f);
  }

  //Komi must be a half-integer within range.
  testAssert(Rules::isValidKomi(-0.5f) && Rules::isValidKomi(0.5f) && Rules::isValidKomi(20.5f) && Rules::isValidKomi(-20.5f));
  testAssert(!Rules::isValidKomi(0.0f) && !Rules::isValidKomi(1.0f) && !Rules::isValidKomi(0.25f));
  testAssert(!Rules::isValidKomi(21.5f) && !Rules::isValidKomi(-21.5f) && !Rules::isValidKomi(std::nanf("")));
  testAssert(Rules::roundKomi(0.0) == 0.5f && Rules::roundKomi(-0.2) == -0.5f && Rules::roundKomi(1.9) == 1.5f);
  testAssert(Rules::roundKomi(100.0) == 20.5f && Rules::roundKomi(-100.0) == -20.5f);
  {
    BoardHistory h;
    bool threw = false;
    try { h.setKomi(0.0f); } catch(const StringError&) { threw = true; }
    testAssert(threw && h.rules.komi == -0.5f);
  }
  //Komi from a player's perspective (the planned v2 komi input), and no draw adjustment: draws come from maxPlies.
  {
    BoardHistory h(Board(), P_BLACK, rulesWithKomi(1.5f), 0, BoardHistoryModes());
    testAssert(h.currentSelfKomi(P_WHITE, 0.5) == 1.5f && h.currentSelfKomi(P_BLACK, 0.5) == -1.5f);
    testAssert(h.whiteKomiAdjustmentForDraws(1.0) == 0.0f);
    testAssert(!h.rules.gameResultWillBeInteger());
  }
}

//------------------------------------------------------------------------------------------------

static void testUtilityScore() {
  cout << "  Utility score with a time bonus" << endl;
  Rules rules = Rules::getQuoridorRules();
  rules.timeBonusPerPly = 0.05f;

  //u = s + sign(s) * lambda * (maxPlies - T)
  for(const KnownGame& kg : KNOWN_GAMES) {
    ScriptedGame g = playScripted(kg.black, kg.white, rules);
    float lead = kg.tempo - 0.5f;
    testAssert(g.hist.finalWhiteLead == lead);
    double bonus = 0.05 * (300 - kg.plies);
    testAssert(approxEqual(g.hist.finalWhiteMinusBlackScore, lead > 0 ? lead + bonus : lead - bonus));
  }
  ScriptedGame fast = playScripted("LFFFFFFF", "FFFFFFFF", rules);      //W+1 on ply 16
  ScriptedGame slow = playScripted("LRLFFFFFFF", "LRFFFFFFFF", rules);  //W+1 on ply 20
  testAssert(slow.hist.getCurrentTurnNumber() == 20 && slow.hist.finalWhiteLead == 0.5f);
  testAssert(approxEqual(fast.hist.finalWhiteMinusBlackScore, 0.5 + 0.05 * 284));
  testAssert(approxEqual(slow.hist.finalWhiteMinusBlackScore, 0.5 + 0.05 * 280));
  //The winner prefers winning earlier, the loser losing later.
  testAssert(fast.hist.finalWhiteMinusBlackScore > slow.hist.finalWhiteMinusBlackScore);
  ScriptedGame blackFast = playScripted("LFFFFFFFF", "RFFFFFFF", rules);        //B+1 on ply 17
  ScriptedGame blackSlow = playScripted("LRLFFFFFFFF", "RLRFFFFFFF", rules);    //B+1 on ply 21
  testAssert(blackSlow.hist.getCurrentTurnNumber() == 21 && blackSlow.hist.finalWhiteLead == -0.5f);
  testAssert(blackFast.hist.finalWhiteMinusBlackScore < blackSlow.hist.finalWhiteMinusBlackScore);

  //A win is always better than a draw, even on the last ply: |u| >= |s| >= 0.5.
  {
    ScriptedGame g = playScripted("LFFFFFFF", "FFFFFFFF", rules, 300 - 16);
    testAssert(g.hist.getCurrentTurnNumber() == 300 && g.hist.winner == P_WHITE);
    testAssert(g.hist.finalWhiteMinusBlackScore == 0.5f);
    ScriptedGame gb = playScripted("LFFFFFFFF", "RFFFFFFF", rules, 300 - 17);
    testAssert(gb.hist.winner == P_BLACK && gb.hist.finalWhiteMinusBlackScore == -0.5f);
  }

  //Terminal scores are exact: no Go-style gridding of the score variance, also off the 0.5 grid and at a draw.
  testAssert(ScoreValue::whiteScoreMeanSqOfScoreGridded(14.7, 0.5) == 14.7 * 14.7);
  testAssert(ScoreValue::whiteScoreMeanSqOfScoreGridded(-0.5, 0.5) == 0.25);
  testAssert(ScoreValue::whiteScoreMeanSqOfScoreGridded(0.0, 0.5) == 0.0);
  testAssert(ScoreValue::whiteScoreMeanSqOfScoreGridded(0.0, 1.0) == 0.0);
  BoardHistory h;
  testAssert(ScoreValue::whiteScoreDrawAdjust(12.5, 1.0, h) == 12.5);
}

//------------------------------------------------------------------------------------------------

static void testDrawAtMaxPlies() {
  cout << "  Draw at maxPlies" << endl;
  Rules rules = Rules::getQuoridorRules();
  rules.timeBonusPerPly = 0.1f;

  //White would arrive on ply 16 of this game, Black on ply 17.
  //Started at ply 284, White arrives on ply 300 = maxPlies: a win, not a draw.
  {
    ScriptedGame g = playScripted("LFFFFFFF", "FFFFFFFF", rules, 284);
    testAssert(g.hist.getCurrentTurnNumber() == 300);
    testAssert(g.hist.winner == P_WHITE && !g.hist.isDraw());
    testAssert(g.hist.finalWhiteLead == 0.5f && g.hist.finalWhiteMinusBlackScore == 0.5f);
  }
  //Started at ply 285, the game reaches ply 300 one move before White would arrive: a draw.
  {
    ScriptedGame g = playScripted("LFFFFFFF", "FFFFFFFF", rules, 285);
    testAssert(g.hist.getCurrentTurnNumber() == 300);
    testAssert(g.moves.size() == 15);
    testAssert(g.hist.isGameFinished && g.hist.isDraw() && g.hist.isScored);
    testAssert(g.hist.winner == C_EMPTY && !g.hist.isNoResult);
    testAssert(g.hist.finalWhiteLead == 0.0f && g.hist.finalWhiteMinusBlackScore == 0.0f);
    testAssert(g.hist.pliesUntilDraw() == 0);
    testAssert(WriteSgf::gameResultNoSgfTag(g.hist) == "0");
    //A draw is worth drawEquivalentWinsForWhite, 0.5 by default.
    testAssert(ScoreValue::whiteWinsOfWinner(g.hist.winner, 0.5) == 0.5);
    //Nothing is legal any more.
    testAssert(!g.hist.isLegal(g.board, scriptedStep(g.board, P_WHITE, 'F'), P_WHITE));
    //Undo across the draw: one ply back, the game is open again with 1 ply to go; redoing draws again.
    Board board;
    BoardHistory hist;
    replayFirst(g, g.moves.size() - 1, board, hist);
    testAssert(!hist.isGameFinished && hist.getCurrentTurnNumber() == 299 && hist.pliesUntilDraw() == 1);
    hist.makeBoardMoveAssumeLegal(board, g.moves.back(), g.plas.back(), NULL);
    testAssert(hist.isDraw());
  }
  //Other maxPlies. A smaller limit ends the same game earlier.
  {
    Rules r = rules;
    r.maxPlies = 10;
    ScriptedGame g = playScripted("LFFFFFFF", "FFFFFFFF", r);
    testAssert(g.hist.isDraw() && g.hist.getCurrentTurnNumber() == 10);
  }

  //pliesUntilDraw counts from the real game start: initialTurnNumber + moves played.
  {
    ScriptedGame g = playScripted("LFFF", "FFF", rules, 0, true);
    testAssert(!g.hist.isGameFinished && g.hist.getCurrentTurnNumber() == 7 && g.hist.pliesUntilDraw() == 293);
    ScriptedGame g2 = playScripted("LFFF", "FFF", rules, 100, true);
    testAssert(g2.hist.getCurrentTurnNumber() == 107 && g2.hist.pliesUntilDraw() == 193);
  }
  //A setup position counts plies from the walls on the board (QTP set_position and SGF loads use
  //Board::numStonesOnBoard as the initial turn number): 0 on an empty board.
  {
    Board board;
    testAssert(board.numStonesOnBoard() == 0);
    testAssert(board.setStones({}));
    BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
    hist.setInitialTurnNumber(board.numStonesOnBoard());
    testAssert(hist.getCurrentTurnNumber() == 0 && hist.pliesUntilDraw() == 300);
    hist.makeBoardMoveAssumeLegal(board, Location::ofString("e2h", board), P_BLACK, NULL);
    testAssert(board.numStonesOnBoard() == 1);
  }
}

//------------------------------------------------------------------------------------------------

static void testUndoAcrossWin() {
  cout << "  Undo across a finished game" << endl;
  Rules rules = Rules::getQuoridorRules();
  rules.timeBonusPerPly = 0.05f;
  ScriptedGame g = playScripted("LFFFFFFFF", "RFFFFFFF", rules);  //B+1 on ply 17
  testAssert(g.hist.winner == P_BLACK);
  Board board;
  BoardHistory hist;
  replayFirst(g, g.moves.size() - 1, board, hist);
  testAssert(!hist.isGameFinished && !hist.isScored && hist.winner == C_EMPTY);
  testAssert(hist.finalWhiteLead == 0.0f && hist.finalWhiteMinusBlackScore == 0.0f);
  testAssert(hist.getCurrentTurnNumber() == 16 && hist.pliesUntilDraw() == 284);
  //Black can still win, or play something else.
  testAssert(hist.isLegal(board, g.moves.back(), P_BLACK));
  testAssert(hist.isLegal(board, scriptedStep(board, P_BLACK, 'L'), P_BLACK));
  hist.makeBoardMoveAssumeLegal(board, g.moves.back(), P_BLACK, NULL);
  testAssert(hist.winner == P_BLACK);
  testAssert(hist.finalWhiteLead == g.hist.finalWhiteLead);
  testAssert(hist.finalWhiteMinusBlackScore == g.hist.finalWhiteMinusBlackScore);
}

//------------------------------------------------------------------------------------------------

static void testFenceHandicap() {
  cout << "  Fence handicap" << endl;
  Rules rules = Rules::getQuoridorRules();
  rules.blackInitialFences = 0;
  rules.whiteInitialFences = 3;

  Board board;
  board.setFencesLeft(rules.blackInitialFences, rules.whiteInitialFences);
  testAssert(board.blackFences == 0 && board.whiteFences == 3);
  board.checkConsistency();
  {
    //The hash matches a board that got there by setting fences on a fresh board in any order.
    Board other;
    other.setFencesLeft(5, 5);
    other.setFencesLeft(0, 3);
    testAssert(other.pos_hash == board.pos_hash);
    testAssert(Board().pos_hash != board.pos_hash);
  }
  BoardHistory hist(board, P_BLACK, rules, 0, BoardHistoryModes());
  Loc wall = Location::ofString("a2h", board);
  //Black has no walls: a wall is illegal, pawn moves are fine.
  testAssert(!hist.isLegal(board, wall, P_BLACK));
  Loc pawnMove = Location::ofString("e8", board);
  testAssert(hist.isLegal(board, pawnMove, P_BLACK));
  hist.makeBoardMoveAssumeLegal(board, pawnMove, P_BLACK, NULL);
  //White has 3.
  testAssert(hist.isLegal(board, wall, P_WHITE));
  hist.makeBoardMoveAssumeLegal(board, wall, P_WHITE, NULL);
  testAssert(board.whiteFences == 2 && board.blackFences == 0);
  board.checkConsistency();

  //Self-play game setup applies the config's initial walls.
  {
    istringstream in(
      "bSizes = 17\n"
      "bSizeRelProbs = 1\n"
      "blackInitialWalls = 7\n"
      "whiteInitialWalls = 10\n"
      "maxPlies = 250\n"
      "timeBonusPerPly = 0.05\n"
    );
    ConfigParser cfg(in);
    Logger logger(nullptr, false, false, false);
    GameInitializer gameInit(cfg, logger, "testquoridoriov2 fence handicap");
    Board b;
    Player pla;
    BoardHistory h;
    ExtraBlackAndKomi extraBlackAndKomi;
    OtherGameProperties otherGameProps;
    PlaySettings playSettings;
    gameInit.createGame(b, pla, h, extraBlackAndKomi, NULL, playSettings, otherGameProps, NULL, BoardHistoryModes());
    testAssert(b.blackFences == 7 && b.whiteFences == 10);
    testAssert(h.initialBoard.blackFences == 7);
    testAssert(h.rules.blackInitialFences == 7 && h.rules.whiteInitialFences == 10);
    testAssert(h.rules.maxPlies == 250 && h.rules.timeBonusPerPly == 0.05f);
    testAssert(h.rules.komi == -0.5f);
    testAssert(pla == P_BLACK);
  }
}

//------------------------------------------------------------------------------------------------

static void testRulesSerializationAndConfig() {
  cout << "  Rules serialization and config" << endl;
  Rules standard = Rules::getQuoridorRules();
  testAssert(standard.toString() == "Quoridor");
  testAssert(standard.toStringNoKomi() == "Quoridor");
  testAssert(Rules::parseRules("Quoridor") == standard);
  testAssert(standard.toJsonStringNoKomi() == "{\"blackInitialWalls\":10,\"maxPlies\":300,\"repetitionDrawCount\":0,\"timeBonusPerPly\":0.0,\"whiteInitialWalls\":10}");

  Rules r = standard;
  r.komi = 1.5f;
  r.maxPlies = 400;
  r.timeBonusPerPly = 0.05f;
  r.blackInitialFences = 7;
  testAssert(r != standard);
  testAssert(!r.equalsIgnoringKomi(standard));
  testAssert(r.toString() == "Quoridor:komi=1.5,maxPlies=400,timeBonusPerPly=0.05,blackInitialWalls=7");
  testAssert(r.toStringNoKomi() == "Quoridor:maxPlies=400,timeBonusPerPly=0.05,blackInitialWalls=7");
  testAssert(Rules::parseRules(r.toString()) == r);
  testAssert(Rules::parseRules(r.toJsonString()) == r);
  testAssert(Rules::parseRulesWithoutKomi(r.toStringNoKomi(), 1.5f) == r);
  testAssert(Rules::parseRules("quoridor:MAXPLIES=400, timebonusperply = 0.05,komi=1.5,blackInitialWalls=7") == r);
  {
    Rules k = standard;
    k.komi = 2.5f;
    testAssert(k.equalsIgnoringKomi(standard) && k != standard);
  }

  //Bad values and keys.
  Rules buf;
  testAssert(!Rules::tryParseRules("Quoridor:komi=0", buf));
  testAssert(!Rules::tryParseRules("Quoridor:maxPlies=0", buf));
  testAssert(!Rules::tryParseRules("Quoridor:timeBonusPerPly=-0.1", buf));
  testAssert(!Rules::tryParseRules("Quoridor:blackInitialWalls=11", buf));
  testAssert(!Rules::tryParseRules("Quoridor:whiteInitialWalls=-1", buf));
  testAssert(!Rules::tryParseRules("Quoridor:foo=1", buf));
  testAssert(!Rules::tryParseRules("Quoridor:maxPlies", buf));
  testAssert(!Rules::tryParseRules("{\"komi\": 1}", buf));
  testAssert(!Rules::tryParseRulesWithoutKomi("Quoridor", buf, 7.0f));

  //kata-set-rule
  testAssert(Rules::updateRules("maxPlies", "200", standard).maxPlies == 200);
  testAssert(Rules::updateRules("timeBonusPerPly", "0.1", standard).timeBonusPerPly == 0.1f);
  testAssert(Rules::updateRules("blackInitialWalls", "0", standard).blackInitialFences == 0);
  testAssert(Rules::updateRules("whiteInitialWalls", "9", standard).whiteInitialFences == 9);
  for(const pair<string,string>& kv : vector<pair<string,string>>{{"komi", "1.5"}, {"maxPlies", "x"}, {"maxPlies", "0"}, {"ko", "SIMPLE"}}) {
    bool threw = false;
    try { Rules::updateRules(kv.first, kv.second, standard); } catch(const StringError&) { threw = true; }
    testAssert(threw);
  }

  //maxMovesPerGame must agree with maxPlies.
  auto loadMax = [](const string& text, int& ret) {
    istringstream in(text);
    ConfigParser cfg(in);
    try { ret = Play::loadMaxMovesPerGame(cfg); } catch(const StringError&) { return false; }
    return true;
  };
  int maxMoves = -1;
  testAssert(loadMax("", maxMoves) && maxMoves == 300);
  testAssert(loadMax("maxPlies = 250\n", maxMoves) && maxMoves == 250);
  testAssert(loadMax("maxMovesPerGame = 300\n", maxMoves) && maxMoves == 300);
  testAssert(loadMax("maxPlies = 250\ncutoffMoves = 250\n", maxMoves) && maxMoves == 250);
  testAssert(!loadMax("maxMovesPerGame = 400\n", maxMoves));
  testAssert(!loadMax("maxPlies = 250\nmaxMovesPerGame = 300\n", maxMoves));
  testAssert(!loadMax("cutoffMoves = 250\n", maxMoves));

  //The self-play komi must be a Quoridor komi; 0.1.0's komiMean = 0 is rejected.
  auto initGame = [](const string& text) {
    istringstream in("bSizes = 17\nbSizeRelProbs = 1\n" + text);
    ConfigParser cfg(in);
    Logger logger(nullptr, false, false, false);
    try { GameInitializer gameInit(cfg, logger, "testquoridoriov2 komiMean"); } catch(const StringError&) { return false; }
    return true;
  };
  testAssert(initGame(""));
  testAssert(initGame("komiMean = 1.5\n"));
  testAssert(!initGame("komiMean = 0\n"));
  testAssert(!initGame("maxPlies = 0\n"));

  //Config rules for GTP and other tools.
  {
    istringstream in("maxPlies = 123\ntimeBonusPerPly = 0.25\nwhiteInitialWalls = 4\nkomi = 2.5\n");
    ConfigParser cfg(in);
    Rules noKomi = Setup::loadSingleRules(cfg, false);
    testAssert(noKomi.maxPlies == 123 && noKomi.timeBonusPerPly == 0.25f && noKomi.whiteInitialFences == 4);
    testAssert(noKomi.komi == -0.5f);
    testAssert(Setup::loadSingleRules(cfg, true).komi == 2.5f);
  }

  //Self-play komi randomization and PlayUtils::adjustKomiToEven stay on the Quoridor komi grid.
  {
    Rand rand("testquoridoriov2 komi noise");
    BoardHistory h;
    ExtraBlackAndKomi ebk;
    ebk.komiMean = -0.5f;
    ebk.komiStdev = 3.0f;
    for(int i = 0; i < 200; i++) {
      PlayUtils::setKomiWithNoise(ebk, h, rand);
      testAssert(Rules::isValidKomi(h.rules.komi));
    }
    ebk.komiMean = 0.0f;
    ebk.komiStdev = 0.0f;
    PlayUtils::setKomiWithoutNoise(ebk, h);
    testAssert(h.rules.komi == 0.5f);
    testAssert(PlayUtils::roundAndClipKomi(-7.0, Board()) == -6.5f);
  }
}

//------------------------------------------------------------------------------------------------

//A board with the pawns moved to the given pawn cells, no walls placed and none left (a fence handicap of 0/0), so
//that only pawn moves are legal and a small search with a random net surely tries them all.
static Board boardWithPawns(const string& white, const string& black) {
  Board start;
  nlohmann::json j = Board::toJson(start);
  vector<int> colors = j["colors"].get<vector<int>>();
  colors[start.whitePawnLoc] = C_EMPTY;
  colors[start.blackPawnLoc] = C_EMPTY;
  Loc w = Location::ofString(white, start);
  Loc b = Location::ofString(black, start);
  colors[w] = C_WHITE;
  colors[b] = C_BLACK;
  j["colors"] = colors;
  j["whitePawnLoc"] = w;
  j["blackPawnLoc"] = b;
  Board board = Board::ofJson(j);
  board.setFencesLeft(0, 0);
  return board;
}

//Search reports the lead s and the utility score u of a finished game separately at terminal nodes.
static void testSearchTerminalValues() {
  cout << "  Search terminal lead vs scoreMean" << endl;
  Logger logger(nullptr, false, false, false);
  //A random net, so the terminal values are all that the search knows about.
  NNEvaluator* nnEval = TestSearchCommon::startNNEval(
    "/dev/null", logger, "quoridorIOv2TerminalNN", NNPos::MAX_BOARD_LEN, NNPos::MAX_BOARD_LEN,
    0, false, false, false, true, false
  );
  SearchParams params;
  params.maxVisits = 400;
  params.numThreads = 1;
  //Random gaussian logits give a spiky policy; flatten it so the search tries the pawn moves.
  params.nnPolicyTemperature = 8.0f;
  params.staticScoreUtilityFactor = 0.10;
  params.dynamicScoreUtilityFactor = 0.30;

  Rules rules = Rules::getQuoridorRules();
  rules.timeBonusPerPly = 0.05f;
  rules.blackInitialFences = 0;
  rules.whiteInitialFences = 0;

  //White on e8, one step from its goal; Black on c5, 4 steps from its goal. At ply 101 White's e9 ends the game W+4:
  //lead 4 - 0.5 = 3.5, score 3.5 + 0.05 * (300 - 101) = 13.45.
  {
    Board board = boardWithPawns("e8", "c5");
    BoardHistory hist(board, P_WHITE, rules, 0, BoardHistoryModes());
    hist.setInitialTurnNumber(100);
    Search* search = new Search(params, nnEval, &logger, "quoridorIOv2TerminalSearch");
    search->setPosition(P_WHITE, board, hist);
    search->runWholeSearch(P_WHITE);
    vector<AnalysisData> data;
    search->getAnalysisData(data, 1, false, 2, false);
    Loc win = Location::ofString("e9", board);
    bool found = false;
    for(const AnalysisData& d : data) {
      if(d.move != win)
        continue;
      found = true;
      testAssert(d.numVisits > 0);
      testAssert(approxEqual(d.winLossValue, 1.0));
      testAssert(approxEqual(d.lead, 3.5));
      testAssert(approxEqual(d.scoreMean, 13.45));
    }
    testAssert(found);
    testAssert(search->getChosenMoveLoc() == win);
    delete search;
  }

  //At ply 299, every move but a goal move ends the game in a draw: lead, score and win/loss 0.
  {
    Board board = boardWithPawns("e7", "c5");
    BoardHistory hist(board, P_WHITE, rules, 0, BoardHistoryModes());
    hist.setInitialTurnNumber(299);
    testAssert(hist.pliesUntilDraw() == 1);
    Search* search = new Search(params, nnEval, &logger, "quoridorIOv2DrawSearch");
    search->setPosition(P_WHITE, board, hist);
    search->runWholeSearch(P_WHITE);
    vector<AnalysisData> data;
    search->getAnalysisData(data, 1, false, 2, false);
    int numVisitedChildren = 0;
    for(const AnalysisData& d : data) {
      if(d.numVisits <= 0)
        continue;
      numVisitedChildren++;
      testAssert(approxEqual(d.winLossValue, 0.0));
      testAssert(approxEqual(d.lead, 0.0));
      testAssert(approxEqual(d.scoreMean, 0.0));
    }
    testAssert(numVisitedChildren >= 3);
    delete search;
  }
  delete nnEval;
}

//------------------------------------------------------------------------------------------------

static bool contains(const string& s, const string& sub) {
  return s.find(sub) != string::npos;
}

static void testSgf() {
  cout << "  SGF KM, WB/WW, RU and RE" << endl;
  Rules rules = Rules::getQuoridorRules();
  rules.komi = 1.5f;
  rules.maxPlies = 400;
  rules.timeBonusPerPly = 0.05f;
  rules.blackInitialFences = 7;
  //B+3 on ply 17: tempo -2, lead -0.5 with komi 1.5, so Black still wins, by half a tempo.
  ScriptedGame g = playScripted("LFFFFFFFF", "RLRFFFFF", rules);
  testAssert(g.hist.winner == P_BLACK && g.hist.finalWhiteLead == -0.5f);

  ostringstream out;
  WriteSgf::writeSgf(out, "black", "white", g.hist, NULL, true, false);
  string sgf = out.str();
  testAssert(contains(sgf, "KM[1.5]"));
  testAssert(contains(sgf, "WB[7]WW[10]"));
  testAssert(contains(sgf, "RU[Quoridor:maxPlies=400,timeBonusPerPly=0.05,blackInitialWalls=7]"));
  //RE is the lead, not the utility score (which includes the time bonus).
  testAssert(contains(sgf, "RE[B+0.5]"));

  std::unique_ptr<CompactSgf> c = CompactSgf::parse(sgf);
  testAssert(c->getRulesOrFail() == rules);
  testAssert(c->getRulesOrFailAllowUnspecified(Rules::getQuoridorRules()) == rules);
  testAssert(c->getRulesOrWarn(Rules::getQuoridorRules(), [](const string&) { testAssert(false); }) == rules);
  testAssert(c->sgfWinner == P_BLACK);
  {
    Board board;
    Player pla;
    BoardHistory hist;
    c->setupBoardAndHistAssumeLegal(rules, board, pla, hist, (int64_t)c->moves.size(), BoardHistoryModes());
    testAssert(hist.initialBoard.blackFences == 7 && hist.initialBoard.whiteFences == 10);
    testAssert(hist.isGameFinished && hist.winner == P_BLACK);
    testAssert(hist.finalWhiteLead == g.hist.finalWhiteLead);
    testAssert(hist.finalWhiteMinusBlackScore == g.hist.finalWhiteMinusBlackScore);
    testAssert(approxEqual(hist.finalWhiteMinusBlackScore, -0.5 - 0.05 * (400 - 17)));
  }
  //Plies after an SGF load count from the start of the SGF (QTP loadsgf sets the initial turn number to
  //Board::numStonesOnBoard(), the walls on the initial board: 0).
  {
    Board board;
    Player pla;
    BoardHistory hist;
    c->setupInitialBoardAndHist(rules, board, pla, hist, BoardHistoryModes());
    hist.setInitialTurnNumber(board.numStonesOnBoard());
    c->playMovesAssumeLegal(board, pla, hist, 10);
    testAssert(!hist.isGameFinished && hist.getCurrentTurnNumber() == 10 && hist.pliesUntilDraw() == 390);
    testAssert(board.blackFences == 7);
  }

  //A draw: RE[0].
  {
    Rules r = Rules::getQuoridorRules();
    r.maxPlies = 10;
    ScriptedGame d = playScripted("LFFFFFFF", "FFFFFFFF", r);
    ostringstream dout;
    WriteSgf::writeSgf(dout, "black", "white", d.hist, NULL, true, false);
    testAssert(contains(dout.str(), "RE[0]"));
    testAssert(contains(dout.str(), "KM[-0.5]WB[10]WW[10]RU[Quoridor:maxPlies=10]"));
    std::unique_ptr<CompactSgf> dc = CompactSgf::parse(dout.str());
    Board board;
    Player pla;
    BoardHistory hist;
    dc->setupBoardAndHistAssumeLegal(dc->getRulesOrFail(), board, pla, hist, (int64_t)dc->moves.size(), BoardHistoryModes());
    testAssert(hist.isDraw());
  }

  //Older SGFs: KataQuoridor 0.1.0 wrote KM[0] and no WB/WW, for the standard game.
  {
    const string old = "(;FF[4]GM[1]SZ[17]PB[a]PW[b]HA[0]KM[0]RU[Quoridor]RE[B+R]AB[iq]AW[ia];B[io];W[ic])";
    std::unique_ptr<CompactSgf> oc = CompactSgf::parse(old);
    testAssert(oc->getRulesOrFail() == Rules::getQuoridorRules());
    Board board;
    Player pla;
    BoardHistory hist;
    oc->setupBoardAndHistAssumeLegal(oc->getRulesOrFail(), board, pla, hist, 2, BoardHistoryModes());
    testAssert(board.blackFences == 10 && board.whiteFences == 10 && hist.getCurrentTurnNumber() == 2);
  }
  //Without KM (and RU) the komi is the standard one, whatever the caller's default rules say.
  {
    const string noKomi = "(;FF[4]GM[1]SZ[17]AB[iq]AW[ia];B[io])";
    std::unique_ptr<CompactSgf> nc = CompactSgf::parse(noKomi);
    Rules defaults = Rules::getQuoridorRules();
    defaults.komi = 2.5f;
    testAssert(nc->getRulesOrFailAllowUnspecified(defaults).komi == -0.5f);
    testAssert(nc->getRulesOrWarn(defaults, [](const string&) {}).komi == -0.5f);
  }
  //Invalid KM and WB are rejected.
  for(const string& bad : {
    string("(;FF[4]GM[1]SZ[17]KM[1]RU[Quoridor];B[io])"),
    string("(;FF[4]GM[1]SZ[17]KM[30.5]RU[Quoridor];B[io])"),
    string("(;FF[4]GM[1]SZ[17]KM[-0.5]WB[11]RU[Quoridor];B[io])"),
  }) {
    bool threw = false;
    try { CompactSgf::parse(bad)->getRulesOrFail(); } catch(const StringError&) { threw = true; }
    testAssert(threw);
  }
}

}  // namespace

void Tests::runQuoridorIOv2Tests() {
  cout << "=== Running Quoridor I/O v2 rules and scoring tests ===" << endl;
  testTempoLeadAndWinner();
  testUtilityScore();
  testDrawAtMaxPlies();
  testUndoAcrossWin();
  testFenceHandicap();
  testRulesSerializationAndConfig();
  testSearchTerminalValues();
  testSgf();
  cout << "=== Quoridor I/O v2 rules and scoring tests passed ===" << endl;
}
