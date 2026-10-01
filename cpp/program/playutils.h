#ifndef PROGRAM_PLAY_UTILS_H_
#define PROGRAM_PLAY_UTILS_H_

#include "../core/config_parser.h"
#include "../program/play.h"
#include "../search/asyncbot.h"

//This is a grab-bag of various useful higher-level functions that select moves or evaluate the board in various ways.

namespace PlayUtils {
  //Use the given bot to play free handicap stones, modifying the board and hist in the process and setting the bot's position to it.
  //Does NOT switch the initial player of the board history to white
  void playExtraBlack(
    Search* bot,
    int numExtraBlack,
    Board& board,
    BoardHistory& hist,
    double temperature,
    Rand& gameRand
  );

  //Set board to empty and place fixed handicap stones, raising an exception if invalid
  void placeFixedHandicap(Board& board, int n);

  ExtraBlackAndKomi chooseExtraBlackAndKomi(
    float base, float stdev, double allowIntegerProb,
    double handicapProb, int numExtraBlackFixed,
    double bigStdevProb, float bigStdev,
    double biggerStdevProb, float biggerStdev,
    double sqrtBoardArea, Rand& rand
  );
  void setKomiWithoutNoise(const ExtraBlackAndKomi& extraBlackAndKomi, BoardHistory& hist); //Also ignores allowInteger
  void setKomiWithNoise(const ExtraBlackAndKomi& extraBlackAndKomi, BoardHistory& hist, Rand& rand);

  ReportedSearchValues getWhiteScoreValues(
    Search* bot,
    const Board& board,
    const BoardHistory& hist,
    Player pla,
    int64_t numVisits,
    const OtherGameProperties& otherGameProps
  );

  Loc chooseRandomLegalMove(const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, Loc banMove);
  int chooseRandomLegalMoves(const Board& board, const BoardHistory& hist, Player pla, Rand& gameRand, Loc* buf, int len);

  Loc chooseRandomPolicyMove(
    const NNOutput* nnOutput,
    const Board& board,
    const BoardHistory& hist,
    Player pla,
    Rand& gameRand,
    double temperature,
    bool allowPass,
    Loc banMove
  );

  Loc getGameInitializationMove(
    Search* botB, Search* botW, const Board& board, const BoardHistory& hist, Player pla, NNResultBuf& buf,
    Rand& gameRand, double temperature
  );
  void initializeGameUsingPolicy(
    Search* botB, Search* botW, Board& board, BoardHistory& hist, Player& pla,
    Rand& gameRand, bool doEndGameIfAllPassAlive,
    double proportionOfBoardArea, double policyInitGammaShape, double temperature
  );

  float roundAndClipKomi(double unrounded, const Board& board);

  //Quoridor I/O v2 step 3 (docs/QuoridorIOv2.md section 5). Nets of Quoridor I/O version >= 2 have a komi input.
  //I/O v1 nets don't: komi only reaches their search at terminal nodes, so finding a fair komi with them
  //(adjustKomiToEven) is meaningless, and Play::runGame skips komi compensation unless both nets see komi.
  constexpr int MIN_QUORIDOR_IO_VERSION_SEEING_KOMI = 2;
  bool nnEvalSeesKomi(const NNEvaluator* nnEval);

  //Match points of a game for White: 1 for a White win, 0 for a Black win, and exactly 0.5 for a game without a
  //winner (a draw by Rules::maxPlies, or a game cut off before its end). Used by the gatekeeper.
  double whitePointsOfGame(const BoardHistory& endHist);

  //The komi where White's expected result is even, given leadAndWinLossOfKomi(komi) = (white lead, white winLoss)
  //for valid Quoridor komis: a few steps along the lead from startKomi, then a binary search on winLoss over the
  //komi grid, interpolated between the two neighbouring komis. Not rounded. This is the search behind
  //adjustKomiToEven and computeLead.
  double findEvenKomi(const std::function<std::pair<double,double>(float)>& leadAndWinLossOfKomi, float startKomi);
  //Rounds to one of the two neighbouring valid komis, the nearer one more likely (linearly), clipped to the valid range.
  float roundKomiRandomly(double komi, Rand& rand);
  //Compensation (adjustKomiToEven) never moves the komi further than this from the standard komi (-0.5). A "fair"
  //komi outside that window comes from a net whose lead doesn't track komi yet (e.g. early in training): with it,
  //one side could practically only play for the maxPlies draw, so the game keeps its komi instead.
  constexpr double MAX_COMPENSATED_KOMI_DELTA = 10.0;
  //The komi a compensated game gets: newKomi randomly rounded, or oldKomi if newKomi is outside the window above.
  float compensatedKomiOrKeep(double newKomi, float oldKomi, Rand& rand);

  void adjustKomiToEven(
    Search* botB,
    Search* botW, //can be NULL if only one bot
    const Board& board,
    BoardHistory& hist,
    Player pla,
    int64_t numVisits,
    const OtherGameProperties& otherGameProps,
    Rand& rand
  );

  //Lead from WHITE's perspective
  float computeLead(
    Search* botB,
    Search* botW, //can be NULL if only one bot
    const Board& board,
    BoardHistory& hist,
    Player pla,
    int64_t numVisits,
    const OtherGameProperties& otherGameProps
  );

  double getSearchFactor(
    double searchFactorWhenWinningThreshold,
    double searchFactorWhenWinning,
    const SearchParams& params,
    const std::vector<double>& recentWinLossValues,
    Player pla
  );

  double getHackedLCBForWinrate(const Search* search, const AnalysisData& data, Player pla);

  std::vector<double> computeOwnership(
    Search* bot,
    const Board& board,
    const BoardHistory& hist,
    Player pla,
    int64_t numVisits
  );

  //Determine all living and dead stones, if the game were terminated right now and
  //the rules were interpreted naively and directly.
  //Returns a vector indexed by board Loc (length Board::MAX_ARR_SIZE).
  std::vector<bool> computeAnticipatedStatusesSimple(
    const Board& board,
    const BoardHistory& hist
  );

  //Determine all living and dead stones, trying to be clever and use the ownership prediction
  //of the neural net.
  //Returns a vector indexed by board Loc (length Board::MAX_ARR_SIZE).
  std::vector<bool> computeAnticipatedStatusesWithOwnership(
    Search* bot,
    const Board& board,
    const BoardHistory& hist,
    Player pla,
    int64_t numVisits,
    std::vector<double>& ownershipsBuf
  );


  struct BenchmarkResults {
    int numThreads = 0;
    int totalPositionsSearched = 0;
    int totalPositions = 0;
    int64_t totalVisits = 0;
    double totalSeconds = 0;
    int64_t numNNEvals = 0;
    int64_t numNNBatches = 0;
    double avgBatchSize = 0;

    std::string toStringNotDone() const;
    std::string toString() const;
    std::string toStringWithElo(const BenchmarkResults* baseline, double secondsPerGameMove) const;

    double computeEloEffect(double secondsPerGameMove) const;

    static void printEloComparison(const std::vector<BenchmarkResults>& results, double secondsPerGameMove);
  };

  //Run benchmark on sgf positions. ALSO prints to stdout the ongoing result as it benchmarks.
  BenchmarkResults benchmarkSearchOnPositionsAndPrint(
    const SearchParams& params,
    const CompactSgf& sgf,
    int numPositionsToUse,
    NNEvaluator* nnEval,
    const BenchmarkResults* baseline,
    double secondsPerGameMove,
    bool printElo
  );

  void printGenmoveLog(
    std::ostream& out,
    const Search* search,
    const NNEvaluator* nnEval,
    Loc moveLoc,
    double timeTaken,
    Player perspective,
    bool logSearchInfoForChosenMove
  );

  Rules genRandomRules(Rand& rand);

  Loc maybeCleanupBeforePass(
    enabled_t cleanupBeforePass,
    enabled_t friendlyPass,
    const Player pla,
    Loc moveLoc,
    const AsyncBot* bot
  );

  Loc maybeFriendlyPass(
    enabled_t cleanupBeforePass,
    enabled_t friendlyPass,
    const Player pla,
    Loc moveLoc,
    Search* bot,
    int64_t numVisits
  );

  std::shared_ptr<NNOutput> getFullSymmetryNNOutput(
    const Board& board, const BoardHistory& hist, Player pla, bool includeOwnerMap, const SGFMetadata* sgfMeta, NNEvaluator* nnEval
  );

}


#endif //PROGRAM_PLAY_UTILS_H_
