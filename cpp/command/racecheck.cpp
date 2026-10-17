/*
 * racecheck.cpp
 *   katago racecheck FILE.sgfs [-max N] [-std-only]
 * Replays self-play SGFs (one game per line, with the move comments of Play::runGame) and compares them with the exact
 * pawn race solver (docs/QuoridorRaceSolver.md, "Measurements"): in every position where both players have no walls, the
 * move played vs the solved value, and the search value of the comment vs the exact value; the same for the proven wins
 * of the only player that still has walls. CPU only.
 */

#include "../core/global.h"
#include "../core/fileutils.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../game/racesolver.h"
#include "../main.h"

#include <fstream>
#include <sstream>
#include <regex>

using namespace std;

namespace {

struct Tally {
  int64_t games = 0, gamesWithExact = 0, gamesWithOneSided = 0;
  int64_t plies = 0, pliesBothZero = 0, pliesExact = 0, pliesExactDraw = 0, pliesUnsolvedBothZero = 0, pliesOneSided = 0;
  int64_t pliesAfterFirstExact = 0;  // plies played from the first exactly solved position to the end
  int64_t moves = 0, movesLostClass = 0, movesLostLead = 0, movesKept = 0;
  int64_t movesWinToDraw = 0, movesWinToLoss = 0, movesDrawToLoss = 0;
  int64_t commentsExact = 0, disagreeExact = 0, commentsOneSided = 0, disagreeOneSided = 0;
  int64_t resultChecks = 0, resultMismatch = 0, resultMismatchWithLostMove = 0;
  int64_t oneSidedResultChecks = 0, oneSidedLost = 0, oneSidedDrawn = 0;
  int64_t gamesWithLostMove = 0;
  int64_t gamesStuck = 0;
  int64_t leadChecks = 0, leadMismatch = 0;
  double sumVisitsExact = 0, sumWeightExact = 0;
};

static string prop(const string& s, const string& key) {
  size_t p = 0;
  while((p = s.find(key + "[", p)) != string::npos) {
    if(p == 0 || !isalpha((unsigned char)s[p - 1])) {
      size_t e = s.find(']', p);
      return s.substr(p + key.size() + 1, e - p - key.size() - 1);
    }
    p += key.size();
  }
  return "";
}

static Loc sgfLoc(const string& s) { return Location::getLoc(s[0] - 'a', s[1] - 'a', 17); }

static void setPawn(Board& board, Player pla, Loc loc) {
  Loc& cur = pla == P_BLACK ? board.blackPawnLoc : board.whitePawnLoc;
  board.setStone(cur, C_EMPTY);
  cur = loc;
  board.setStone(cur, pla == P_BLACK ? C_BLACK : C_WHITE);
}

struct ParsedComment {
  bool ok = false;
  double winLossWhite = 0;
  double visits = 0, weight = 0;
};

static ParsedComment parseComment(const string& c) {
  ParsedComment r;
  istringstream in(c);
  double w, l, d;
  if(in >> w >> l >> d) {
    r.ok = true;
    r.winLossWhite = w - l;
    string tok;
    while(in >> tok) {
      if(tok.rfind("v=", 0) == 0) r.visits = stod(tok.substr(2));
      else if(tok.rfind("weight=", 0) == 0) r.weight = stod(tok.substr(7));
    }
  }
  return r;
}

static void processGame(const string& line, Tally& t) {
  size_t firstMove = line.find(";B[");
  size_t firstMoveW = line.find(";W[");
  if(firstMove == string::npos || (firstMoveW != string::npos && firstMoveW < firstMove))
    firstMove = firstMoveW;
  if(firstMove == string::npos)
    return;
  string head = line.substr(0, firstMove);
  Rules rules = Rules::getQuoridorRules();
  rules.komi = stof(prop(head, "KM"));
  string ru = prop(head, "RU");
  smatch m;
  if(regex_search(ru, m, regex("repetitionDrawCount=(\\d+)")))
    rules.repetitionDrawCount = stoi(m[1]);
  if(regex_search(ru, m, regex("maxPlies=(\\d+)")))
    rules.maxPlies = stoi(m[1]);
  int wb = prop(head, "WB").empty() ? 10 : stoi(prop(head, "WB"));
  int ww = prop(head, "WW").empty() ? 10 : stoi(prop(head, "WW"));
  rules.blackInitialFences = wb;
  rules.whiteInitialFences = ww;
  Board board;
  board.setFencesLeft(wb, ww);
  if(!prop(head, "AB").empty())
    setPawn(board, P_BLACK, sgfLoc(prop(head, "AB")));
  if(!prop(head, "AW").empty())
    setPawn(board, P_WHITE, sgfLoc(prop(head, "AW")));
  Player pla = line[firstMove + 1] == 'B' ? P_BLACK : P_WHITE;
  BoardHistory hist(board, pla, rules, 0, BoardHistoryModes());

  string re = prop(head, "RE");
  string tail = line.substr(firstMove);
  // The result is also written at the root; "0" is a draw.
  Player actualWinner = C_EMPTY;
  double actualLead = 0;
  if(!re.empty() && (re[0] == 'W' || re[0] == 'B')) {
    actualWinner = re[0] == 'W' ? P_WHITE : P_BLACK;
    actualLead = stod(re.substr(2)) * (re[0] == 'W' ? 1 : -1);
  }

  regex moveRe(";([BW])\\[([a-q]{2})\\](?:C\\[([^\\]]*)\\])?");
  bool sawExact = false, sawOneSided = false, lostMove = false, stuck = false;
  RaceResult firstExact, firstOneSided;
  Player oneSidedWinner = C_EMPTY;
  int64_t plyIdx = 0;
  for(sregex_iterator it(tail.begin(), tail.end(), moveRe), end; it != end; ++it) {
    Player mp = (*it)[1] == "B" ? P_BLACK : P_WHITE;
    Loc loc = sgfLoc((*it)[2]);
    if(mp != pla || hist.isGameFinished || !hist.isLegal(board, loc, pla))
      return;
    t.plies++;
    ParsedComment pc = parseComment((*it)[3]);
    if(board.blackFences == 0 && board.whiteFences == 0) {
      t.pliesBothZero++;
      RaceResult rr = RaceSolver::evaluate(board, hist, pla);
      if(rr.solved) {
        t.pliesExact++;
        if(!sawExact) {
          sawExact = true;
          firstExact = rr;
        }
        if(sawExact)
          t.pliesAfterFirstExact++;
        if(rr.winner == C_EMPTY)
          t.pliesExactDraw++;
        t.sumVisitsExact += pc.visits;
        t.sumWeightExact += pc.weight;
        double exactWL = rr.winner == P_WHITE ? 1 : (rr.winner == P_BLACK ? -1 : 0);
        if(pc.ok) {
          t.commentsExact++;
          if(std::fabs(pc.winLossWhite - exactWL) > 0.2)
            t.disagreeExact++;
        }
        // The move played vs the solved value.
        shared_ptr<const RaceSolution> sol = RaceSolver::getCached(board, rules.komi);
        int b = RaceSolution::sqOfLoc(board.blackPawnLoc, 17), w = RaceSolution::sqOfLoc(board.whitePawnLoc, 17);
        int v = sol->val2[RaceSolution::stateIndex(pla, b, w)];
        int to = RaceSolution::sqOfLoc(loc, 17);
        bool arrived = pla == P_BLACK ? to < 9 : to >= 72;
        int succ;
        if(arrived)
          succ = sol->terminalVal2(pla, pla == P_BLACK ? w : b);
        else
          succ = sol->val2[RaceSolution::stateIndex(getOpp(pla), pla == P_BLACK ? to : b, pla == P_BLACK ? w : to)];
        int mv = pla == P_WHITE ? v : -v, ms = pla == P_WHITE ? succ : -succ;
        t.moves++;
        if(ms == mv) {
          t.movesKept++;
        }
        else if((mv > 0) != (ms > 0) || (mv < 0) != (ms < 0)) {
          t.movesLostClass++;
          lostMove = true;
          if(mv > 0 && ms == 0) t.movesWinToDraw++;
          else if(mv > 0 && ms < 0) t.movesWinToLoss++;
          else if(mv == 0 && ms < 0) t.movesDrawToLoss++;
        }
        else {
          t.movesLostLead++;
          lostMove = true;
        }
      }
      else {
        t.pliesUnsolvedBothZero++;
        if(board.getLegalPawnDestinations(pla).empty()) stuck = true;
      }
    }
    else if(board.blackFences == 0 || board.whiteFences == 0) {
      RaceResult rr = RaceSolver::evaluate(board, hist, pla);
      if(rr.solved) {
        t.pliesOneSided++;
        if(!sawOneSided) {
          sawOneSided = true;
          firstOneSided = rr;
          oneSidedWinner = rr.winner;
        }
        if(pc.ok) {
          t.commentsOneSided++;
          if(std::fabs(pc.winLossWhite - (rr.winner == P_WHITE ? 1 : -1)) > 0.2)
            t.disagreeOneSided++;
        }
      }
    }
    hist.makeBoardMoveAssumeLegal(board, loc, pla, NULL);
    pla = getOpp(pla);
    plyIdx++;
  }
  t.games++;
  if(stuck)
    t.gamesStuck++;
  if(lostMove)
    t.gamesWithLostMove++;
  if(sawExact) {
    t.gamesWithExact++;
    t.resultChecks++;
    bool mismatch = firstExact.winner != actualWinner;
    if(mismatch) {
      t.resultMismatch++;
      if(lostMove)
        t.resultMismatchWithLostMove++;
    }
    else if(actualWinner != C_EMPTY) {
      t.leadChecks++;
      if(std::fabs(firstExact.lead2 / 2.0 - actualLead) > 1e-6)
        t.leadMismatch++;
    }
  }
  if(sawOneSided) {
    t.gamesWithOneSided++;
    t.oneSidedResultChecks++;
    if(actualWinner == C_EMPTY) t.oneSidedDrawn++;
    else if(actualWinner != oneSidedWinner) t.oneSidedLost++;
  }
}

static void print(const string& name, const Tally& t) {
  auto pct = [](int64_t a, int64_t b) { return b > 0 ? 100.0 * (double)a / (double)b : 0.0; };
  cout << "== " << name << " ==" << endl;
  cout << "games " << t.games << ", plies " << t.plies << endl;
  cout << "games with an exactly solved position (both players 0 walls, solver valid): " << t.gamesWithExact << " ("
       << pct(t.gamesWithExact, t.games) << "%)" << endl;
  cout << "plies with both players at 0 walls: " << t.pliesBothZero << " (" << pct(t.pliesBothZero, t.plies) << "% of plies); solved exactly: "
       << t.pliesExact << " (" << pct(t.pliesExact, t.plies) << "%), of them draws " << t.pliesExactDraw << ", not solved (horizon/repetition/stuck) "
       << t.pliesUnsolvedBothZero << ", games with a boxed-in pawn " << t.gamesStuck << endl;
  cout << "moves played in exact positions: " << t.moves << ": kept the value " << t.movesKept << ", lost the result class " << t.movesLostClass
       << " (win->draw " << t.movesWinToDraw << ", win->loss " << t.movesWinToLoss << ", draw->loss " << t.movesDrawToLoss << "), lost only lead "
       << t.movesLostLead << "; value-losing: " << pct(t.movesLostClass + t.movesLostLead, t.moves) << "% of moves, games with one: "
       << t.gamesWithLostMove << endl;
  cout << "exact result (first exact position) vs the game result: mismatches " << t.resultMismatch << " of " << t.resultChecks << " ("
       << pct(t.resultMismatch, t.resultChecks) << "%), of them with a value-losing move " << t.resultMismatchWithLostMove << "; lead mismatch among matching wins "
       << t.leadMismatch << " of " << t.leadChecks << endl;
  cout << "search win/loss value in the comments vs the exact value: |diff| > 0.2 in " << t.disagreeExact << " of " << t.commentsExact << " ("
       << pct(t.disagreeExact, t.commentsExact) << "%)" << endl;
  cout << "one-sided proven wins: plies " << t.pliesOneSided << " (" << pct(t.pliesOneSided, t.plies) << "%), games " << t.gamesWithOneSided
       << "; game lost by the proven winner " << t.oneSidedLost << " (" << pct(t.oneSidedLost, t.oneSidedResultChecks) << "%), drawn " << t.oneSidedDrawn
       << "; comment disagreement > 0.2: " << t.disagreeOneSided << " of " << t.commentsOneSided << " (" << pct(t.disagreeOneSided, t.commentsOneSided) << "%)" << endl;
  cout << "plies from the first exactly solved position to the end (the self-play finish plays these): " << t.pliesAfterFirstExact << " ("
       << pct(t.pliesAfterFirstExact, t.plies) << "% of plies); per 10000 games: " << (t.games > 0 ? t.pliesAfterFirstExact * 10000 / t.games : 0) << endl;
  cout << "search visits spent on exactly solved plies (upper bound of the NN evals saved): " << t.sumVisitsExact << " in total, per 10000 games "
       << (t.games > 0 ? t.sumVisitsExact * 10000 / (double)t.games : 0.0) << "; training row weight on those plies per 10000 games "
       << (t.games > 0 ? t.sumWeightExact * 10000 / (double)t.games : 0.0) << endl;
}

}  // namespace

int MainCmds::racecheck(const vector<string>& args) {
  Board::initHash();
  string file;
  int64_t maxGames = 1LL << 60;
  for(size_t i = 0; i < args.size(); i++) {
    if(args[i] == "-max" && i + 1 < args.size())
      maxGames = stoll(args[++i]);
    else if(file.empty() && !args[i].empty() && args[i][0] != '-' && args[i] != "racecheck")
      file = args[i];
  }
  if(file.empty()) {
    cerr << "usage: katago racecheck FILE.sgfs [-max N]" << endl;
    return 1;
  }
  ifstream in(file);
  string line;
  Tally all, standard;
  RaceSolver::resetCacheStats();
  while(getline(in, line) && all.games < maxGames) {
    if(line.empty())
      continue;
    Tally g;
    processGame(line, g);
    auto add = [&](Tally& a) {
      a.games += g.games; a.gamesWithExact += g.gamesWithExact; a.gamesWithOneSided += g.gamesWithOneSided;
      a.plies += g.plies; a.pliesBothZero += g.pliesBothZero; a.pliesExact += g.pliesExact; a.pliesExactDraw += g.pliesExactDraw;
      a.pliesUnsolvedBothZero += g.pliesUnsolvedBothZero; a.pliesOneSided += g.pliesOneSided; a.pliesAfterFirstExact += g.pliesAfterFirstExact;
      a.moves += g.moves; a.movesLostClass += g.movesLostClass; a.movesLostLead += g.movesLostLead; a.movesKept += g.movesKept;
      a.movesWinToDraw += g.movesWinToDraw; a.movesWinToLoss += g.movesWinToLoss; a.movesDrawToLoss += g.movesDrawToLoss;
      a.commentsExact += g.commentsExact; a.disagreeExact += g.disagreeExact; a.commentsOneSided += g.commentsOneSided; a.disagreeOneSided += g.disagreeOneSided;
      a.resultChecks += g.resultChecks; a.resultMismatch += g.resultMismatch; a.resultMismatchWithLostMove += g.resultMismatchWithLostMove;
      a.oneSidedResultChecks += g.oneSidedResultChecks; a.oneSidedLost += g.oneSidedLost; a.oneSidedDrawn += g.oneSidedDrawn;
      a.sumVisitsExact += g.sumVisitsExact; a.sumWeightExact += g.sumWeightExact;
      a.gamesWithLostMove += g.gamesWithLostMove; a.gamesStuck += g.gamesStuck; a.leadChecks += g.leadChecks; a.leadMismatch += g.leadMismatch;
    };
    add(all);
    if(line.find("KM[-0.5]") != string::npos)
      add(standard);
    if(all.games % 1000 == 0 && all.games > 0)
      cerr << all.games << " games" << endl;
  }
  print("all games", all);
  print("standard komi (-0.5)", standard);
  RaceSolver::CacheStats st = RaceSolver::getCacheStats();
  cout << "solver cache: " << st.hits << " hits, " << st.misses << " misses, " << st.solveMicros / 1000.0 << " ms solving ("
       << (st.misses > 0 ? st.solveMicros / 1000.0 / (double)st.misses : 0.0) << " ms per solve)" << endl;
  return 0;
}
