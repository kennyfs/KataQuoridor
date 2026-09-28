/*
 * nnparity.cpp
 * NN parity harness (docs/KataQuoridor_Review_and_Roadmap.md §5 Phase 0 step 4).
 *
 *   katago dumpnninputs -n 200 -seed parity -output rows.npz
 *     Plays random games and writes, for a sample of non-terminal positions, the V1 input rows exactly as
 *     QuoridorNN::fillRow produces them, the legal moves in search space, and the raw board state (so that
 *     Python can re-derive the features and the legal moves independently).
 *
 *   katago evalnnparity -model x.bin.gz -n 200 -seed parity -symmetry 0 -output out.npz [-config ...]
 *     Regenerates the same positions (same -n and -seed) and evaluates each one through NNEvaluator, which is
 *     the path the search uses, then writes the final policy (over the 17x17+1 search slots, after legal
 *     masking and softmax) and the value from White's perspective.
 *
 * python/tests/test_nn_parity.py drives both and compares against a PyTorch reference.
 */

#include "../core/global.h"
#include "../core/config_parser.h"
#include "../core/logger.h"
#include "../core/rand.h"
#include "../dataio/numpywrite.h"
#include "../game/board.h"
#include "../game/boardhistory.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/quoridornn.h"
#include "../program/setup.h"
#include "../command/commandline.h"
#include "../main.h"

using namespace std;

namespace {

struct ParityPosition {
  Board board;
  BoardHistory hist;
  Player pla;
};

static const int SEARCH_LEN = Board::DEFAULT_LEN;
static const int POLICY_SIZE = NNPos::MAX_NN_POLICY_SIZE; // 17*17+1

static bool isTerminal(const BoardHistory& hist) {
  return hist.isGameFinished;
}

// Deterministic in (numRows, seed): dumpnninputs and evalnnparity must see the same positions.
static vector<ParityPosition> generatePositions(int numRows, const string& seed) {
  Rand rand("nnparity:" + seed);
  vector<ParityPosition> out;
  const int maxPlies = 300;
  for(int game = 0; (int)out.size() < numRows; game++) {
    // Same mix as the rule fuzz test: 0 = uniform over legal moves, 1 = pawn move half the time,
    // 2 = mostly shortest-path pawn moves. This covers wall-heavy middlegames, jumps, and endgames.
    int style = game % 3;
    Board board;
    Player pla = P_BLACK;
    BoardHistory hist(board, pla, Rules::getQuoridorRules(), 0, BoardHistoryModes());
    for(int ply = 0; ply < maxPlies && !isTerminal(hist) && (int)out.size() < numRows; ply++) {
      // Once both players are out of walls games get long and repetitive, so sample those more sparsely.
      bool anyWallsLeft = board.blackFences > 0 || board.whiteFences > 0;
      if((game == 0 && ply == 0) || rand.nextBool(anyWallsLeft ? 0.5 : 0.05))
        out.push_back(ParityPosition{board, hist, pla});

      vector<Loc> moves;
      vector<Loc> pawnMoves;
      for(int pos = 0; pos < SEARCH_LEN * SEARCH_LEN; pos++) {
        Loc loc = NNPos::posToLoc(pos, SEARCH_LEN, SEARCH_LEN, SEARCH_LEN, SEARCH_LEN);
        if(hist.isLegal(board, loc, pla)) {
          moves.push_back(loc);
          if(Location::isPawnLoc(loc))
            pawnMoves.push_back(loc);
        }
      }
      testAssert(!moves.empty() && !pawnMoves.empty());

      Loc chosen;
      double u = rand.nextDouble();
      if(style == 2 && u < 0.8) {
        vector<Loc> path = board.findShortestPath(pla);
        chosen = (path.size() >= 2 && board.isLegal(path[1], pla)) ? path[1] : pawnMoves[rand.nextUInt((uint32_t)pawnMoves.size())];
      }
      else if(style >= 1 && u < 0.5)
        chosen = pawnMoves[rand.nextUInt((uint32_t)pawnMoves.size())];
      else
        chosen = moves[rand.nextUInt((uint32_t)moves.size())];

      hist.makeBoardMoveAssumeLegal(board, chosen, pla, NULL);
      pla = getOpp(pla);
    }
  }
  return out;
}

template <typename T>
static void writeArray(ZipFile& zip, const char* name, NumpyBuffer<T>& buf, int64_t numRows) {
  uint64_t numBytes = buf.prepareHeaderWithNumRows(numRows);
  zip.writeBuffer(name, buf.dataIncludingHeader, numBytes);
}

} // namespace

int MainCmds::dumpnninputs(const vector<string>& args) {
  Board::initHash();

  int numRows;
  string seed;
  string outputFile;
  try {
    KataGoCommandLine cmd("Dump Quoridor NN input rows for random positions, for the NN parity test.");
    TCLAP::ValueArg<int> numRowsArg("n", "num-rows", "Number of positions to dump", false, 200, "N");
    TCLAP::ValueArg<string> seedArg("", "seed", "Seed for the random games", false, "parity", "SEED");
    TCLAP::ValueArg<string> outputArg("", "output", "Output .npz file", true, string(), "FILE");
    cmd.add(numRowsArg);
    cmd.add(seedArg);
    cmd.add(outputArg);
    cmd.parseArgs(args);
    numRows = numRowsArg.getValue();
    seed = seedArg.getValue();
    outputFile = outputArg.getValue();
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  if(numRows <= 0) {
    cerr << "-n must be positive" << endl;
    return 1;
  }

  const int ioVersion = QuoridorNN::MAX_SUPPORTED_IO_VERSION;
  const int C = QuoridorNN::numSpatialFeatures(ioVersion);
  const int G = QuoridorNN::numGlobalFeatures(ioVersion);
  const int X = QuoridorNN::MODEL_LEN;
  const int Y = QuoridorNN::MODEL_LEN;

  vector<ParityPosition> positions = generatePositions(numRows, seed);

  NumpyBuffer<float> spatial({numRows, C, Y, X});
  NumpyBuffer<float> global({numRows, G});
  NumpyBuffer<uint8_t> legal({numRows, POLICY_SIZE});
  NumpyBuffer<int8_t> nextPlayer({numRows, 1}); // 1 = Black, 2 = White (NumpyBuffer cannot write 1-D shapes)
  NumpyBuffer<int8_t> pawns({numRows, 4});        // black c, black r, white c, white r (board coords)
  NumpyBuffer<int8_t> fencesLeft({numRows, 2});   // black, white
  NumpyBuffer<uint8_t> hWalls({numRows, 8, 8});   // [r][c] anchors, board coords
  NumpyBuffer<uint8_t> vWalls({numRows, 8, 8});

  MiscNNInputParams nnInputParams;
  for(int i = 0; i < numRows; i++) {
    const ParityPosition& p = positions[i];
    const Board& b = p.board;
    QuoridorNN::fillRow(b, p.hist, p.pla, nnInputParams, ioVersion, false, spatial.data + (size_t)i * C * X * Y, global.data + (size_t)i * G);

    for(int pos = 0; pos < POLICY_SIZE; pos++) {
      Loc loc = NNPos::posToLoc(pos, SEARCH_LEN, SEARCH_LEN, SEARCH_LEN, SEARCH_LEN);
      legal.data[(size_t)i * POLICY_SIZE + pos] = p.hist.isLegal(b, loc, p.pla) ? 1 : 0;
    }
    nextPlayer.data[i] = (int8_t)p.pla;
    pawns.data[i * 4 + 0] = (int8_t)(Location::getX(b.blackPawnLoc, SEARCH_LEN) / 2);
    pawns.data[i * 4 + 1] = (int8_t)(Location::getY(b.blackPawnLoc, SEARCH_LEN) / 2);
    pawns.data[i * 4 + 2] = (int8_t)(Location::getX(b.whitePawnLoc, SEARCH_LEN) / 2);
    pawns.data[i * 4 + 3] = (int8_t)(Location::getY(b.whitePawnLoc, SEARCH_LEN) / 2);
    fencesLeft.data[i * 2 + 0] = (int8_t)b.blackFences;
    fencesLeft.data[i * 2 + 1] = (int8_t)b.whiteFences;
    // Read wall anchors straight from Board's explicit wall arrays (the single source of truth;
    // see Board::vWalls/hWalls), rather than reconstructing from `colors`, which is ambiguous
    // when the arms of neighboring walls touch.
    for(int r = 0; r < 8; r++) {
      for(int c = 0; c < 8; c++) {
        hWalls.data[(size_t)i * 64 + r * 8 + c] = b.hWalls[c][r] ? 1 : 0;
        vWalls.data[(size_t)i * 64 + r * 8 + c] = b.vWalls[c][r] ? 1 : 0;
      }
    }
  }

  ZipFile zip(outputFile);
  writeArray(zip, "binaryInputNCHW", spatial, numRows);
  writeArray(zip, "globalInputNC", global, numRows);
  writeArray(zip, "legalMask", legal, numRows);
  writeArray(zip, "nextPlayer", nextPlayer, numRows);
  writeArray(zip, "pawns", pawns, numRows);
  writeArray(zip, "fencesLeft", fencesLeft, numRows);
  writeArray(zip, "hWalls", hWalls, numRows);
  writeArray(zip, "vWalls", vWalls, numRows);
  zip.close();
  cout << "Wrote " << numRows << " rows to " << outputFile << endl;
  return 0;
}

int MainCmds::evalnnparity(const vector<string>& args) {
  Board::initHash();
  ScoreValue::initTables();

  ConfigParser cfg;
  string modelFile;
  int numRows;
  string seed;
  int symmetry;
  string outputFile;
  try {
    KataGoCommandLine cmd("Evaluate the NN parity positions through NNEvaluator and write policy and value.");
    cmd.addModelFileArg();
    cmd.addConfigFileArg("", "", false);
    cmd.addOverrideConfigArg();
    TCLAP::ValueArg<int> numRowsArg("n", "num-rows", "Number of positions (must match dumpnninputs)", false, 200, "N");
    TCLAP::ValueArg<string> seedArg("", "seed", "Seed for the random games (must match dumpnninputs)", false, "parity", "SEED");
    TCLAP::ValueArg<int> symmetryArg("", "symmetry", "NN symmetry to evaluate at (0 or 1)", false, 0, "SYM");
    TCLAP::ValueArg<string> outputArg("", "output", "Output .npz file", true, string(), "FILE");
    cmd.add(numRowsArg);
    cmd.add(seedArg);
    cmd.add(symmetryArg);
    cmd.add(outputArg);
    cmd.parseArgs(args);
    modelFile = cmd.getModelFile();
    numRows = numRowsArg.getValue();
    seed = seedArg.getValue();
    symmetry = symmetryArg.getValue();
    outputFile = outputArg.getValue();
    cmd.getConfigAllowEmpty(cfg);
  }
  catch(TCLAP::ArgException& e) {
    cerr << "Error: " << e.error() << " for argument " << e.argId() << endl;
    return 1;
  }
  if(numRows <= 0 || symmetry < 0 || symmetry > 1) {
    cerr << "-n must be positive and -symmetry must be 0 or 1" << endl;
    return 1;
  }

  // Deterministic, uncached evaluation unless the config says otherwise.
  if(!cfg.contains("nnRandomize"))
    cfg.overrideKey("nnRandomize", "false");
  if(!cfg.contains("numNNServerThreadsPerModel"))
    cfg.overrideKey("numNNServerThreadsPerModel", "1");
  if(!cfg.contains("onnxProvider"))
    cfg.overrideKey("onnxProvider", "cpu");
  if(!cfg.contains("nnCacheSizePowerOfTwo"))
    cfg.overrideKey("nnCacheSizePowerOfTwo", "10");
  if(!cfg.contains("nnMutexPoolSizePowerOfTwo"))
    cfg.overrideKey("nnMutexPoolSizePowerOfTwo", "8");

  Logger logger(&cfg, false, true);
  Rand seedRand;
  // Same buffer size and exact-size setting that gtp uses for Quoridor.
  NNEvaluator* nnEval = Setup::initializeNNEvaluator(
    modelFile, modelFile, "", cfg, logger, seedRand, 1,
    SEARCH_LEN, SEARCH_LEN, Setup::MaxBatchSizeRequest::explicitSize(8), true, false,
    Setup::SETUP_FOR_OTHER
  );

  vector<ParityPosition> positions = generatePositions(numRows, seed);
  NumpyBuffer<float> policy({numRows, POLICY_SIZE});
  NumpyBuffer<float> value({numRows, 3}); // whiteWin, whiteLoss, whiteNoResult
  // whiteScoreMean, whiteLead, scoreStdev, varTimeLeft, shorttermWinlossError, shorttermScoreError
  NumpyBuffer<float> misc({numRows, 6});

  MiscNNInputParams nnInputParams;
  nnInputParams.symmetry = symmetry;
  for(int i = 0; i < numRows; i++) {
    const ParityPosition& p = positions[i];
    NNResultBuf buf;
    nnEval->evaluate(p.board, p.hist, p.pla, nnInputParams, buf, true, false);
    testAssert(buf.hasResult);
    const NNOutput& out = *buf.result;
    for(int pos = 0; pos < POLICY_SIZE; pos++)
      policy.data[(size_t)i * POLICY_SIZE + pos] = out.policyProbs[pos];
    value.data[i * 3 + 0] = out.whiteWinProb;
    value.data[i * 3 + 1] = out.whiteLossProb;
    value.data[i * 3 + 2] = out.whiteNoResultProb;
    misc.data[i * 6 + 0] = out.whiteScoreMean;
    misc.data[i * 6 + 1] = out.whiteLead;
    misc.data[i * 6 + 2] = (float)sqrt(std::max(0.0, (double)out.whiteScoreMeanSq - (double)out.whiteScoreMean * out.whiteScoreMean));
    misc.data[i * 6 + 3] = out.varTimeLeft;
    misc.data[i * 6 + 4] = out.shorttermWinlossError;
    misc.data[i * 6 + 5] = out.shorttermScoreError;
  }

  ZipFile zip(outputFile);
  writeArray(zip, "policy", policy, numRows);
  writeArray(zip, "value", value, numRows);
  writeArray(zip, "misc", misc, numRows);
  zip.close();
  cout << "Wrote " << numRows << " evaluations at symmetry " << symmetry << " to " << outputFile << endl;

  delete nnEval;
  NeuralNet::globalCleanup();
  ScoreValue::freeTables();
  return 0;
}
