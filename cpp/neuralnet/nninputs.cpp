#include "../neuralnet/nninputs.h"

#include "../core/test.h"

using namespace std;

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

const Hash128 MiscNNInputParams::ZOBRIST_CONSERVATIVE_PASS =
  Hash128(0x0c2b96f4b8ae2da9ULL, 0x5a14dee208fec0edULL);
const Hash128 MiscNNInputParams::ZOBRIST_FRIENDLY_PASS =
  Hash128(0xe750505a66f7c5c2ULL, 0x7a83139bf632d6c4ULL);
const Hash128 MiscNNInputParams::ZOBRIST_PASSING_HACKS =
  Hash128(0x9c89f4fd3ce5a92cULL, 0x268c9aff79c64d00ULL);
const Hash128 MiscNNInputParams::ZOBRIST_PLAYOUT_DOUBLINGS =
  Hash128(0xa5e6114d380bfc1dULL, 0x4160557f1222f4adULL);
const Hash128 MiscNNInputParams::ZOBRIST_NN_POLICY_TEMP =
  Hash128(0xebcbdfeec6f4334bULL, 0xb85e43ee243b5ad2ULL);
const Hash128 MiscNNInputParams::ZOBRIST_AVOID_MYTDAGGER_HACK =
  Hash128(0x612d22ec402ce054ULL, 0x0db915c49de527aeULL);
const Hash128 MiscNNInputParams::ZOBRIST_POLICY_OPTIMISM =
  Hash128(0x88415c85c2801955ULL, 0x39bdf76b2aaa5eb1ULL);
const Hash128 MiscNNInputParams::ZOBRIST_ZERO_HISTORY =
  Hash128(0x78f02afdd1aa4910ULL, 0xda78d550486fe978ULL);

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------

double ScoreValue::whiteWinsOfWinner(Player winner, double drawEquivalentWinsForWhite) {
  if(winner == P_WHITE)
    return 1.0;
  else if(winner == P_BLACK)
    return 0.0;

  testAssert(winner == C_EMPTY);
  return drawEquivalentWinsForWhite;
}

static const double twoOverPi = 0.63661977236758134308;
static const double piOverTwo = 1.57079632679489661923;

double ScoreValue::whiteScoreDrawAdjust(double finalWhiteMinusBlackScore, double drawEquivalentWinsForWhite, const BoardHistory& hist) {
  return finalWhiteMinusBlackScore + hist.whiteKomiAdjustmentForDraws(drawEquivalentWinsForWhite);
}

double ScoreValue::whiteScoreValueOfScoreSmooth(
  double finalWhiteMinusBlackScore,
  double center,
  double scale,
  double drawEquivalentWinsForWhite,
  double sqrtBoardArea,
  const BoardHistory& hist)
{
  double adjustedScore = finalWhiteMinusBlackScore + hist.whiteKomiAdjustmentForDraws(drawEquivalentWinsForWhite) - center;
  return atan(adjustedScore / (scale * sqrtBoardArea)) * twoOverPi;
}

double ScoreValue::whiteScoreValueOfScoreSmoothNoDrawAdjust(double finalWhiteMinusBlackScore, double center, double scale, double sqrtBoardArea) {
  double adjustedScore = finalWhiteMinusBlackScore - center;
  return atan(adjustedScore / (scale * sqrtBoardArea)) * twoOverPi;
}

double ScoreValue::whiteDScoreValueDScoreSmoothNoDrawAdjust(double finalWhiteMinusBlackScore, double center, double scale, double sqrtBoardArea) {
  double adjustedScore = finalWhiteMinusBlackScore - center;
  double scaleFactor;
  scaleFactor = scale * sqrtBoardArea;
  return scaleFactor / (scaleFactor * scaleFactor + adjustedScore * adjustedScore) * twoOverPi;
}

static double inverse_atan(double x) {
  if(x >= piOverTwo - 1e-6) return 1e6;
  if(x <= -piOverTwo + 1e-6) return -1e6;
  return tan(x);
}

double ScoreValue::approxWhiteScoreOfScoreValueSmooth(double scoreValue, double center, double scale, double sqrtBoardArea) {
  testAssert(scoreValue >= -1 && scoreValue <= 1);
  double scoreUnscaled = inverse_atan(scoreValue * piOverTwo);
  return scoreUnscaled * (scale * sqrtBoardArea) + center;
}

double ScoreValue::whiteScoreMeanSqOfScoreGridded(double finalWhiteMinusBlackScore, double drawEquivalentWinsForWhite) {
  testAssert((int)(finalWhiteMinusBlackScore * 2) == finalWhiteMinusBlackScore * 2);
  bool finalScoreIsInteger = ((int)finalWhiteMinusBlackScore == finalWhiteMinusBlackScore);
  if(!finalScoreIsInteger)
    return finalWhiteMinusBlackScore * finalWhiteMinusBlackScore;

  double lower = finalWhiteMinusBlackScore - 0.5;
  double upper = finalWhiteMinusBlackScore + 0.5;
  double lowerSq = lower * lower;
  double upperSq = upper * upper;

  return lowerSq + (upperSq - lowerSq) * drawEquivalentWinsForWhite;
}


static bool scoreValueTablesInitialized = false;
static double* expectedSVTable = NULL;
static const int svTableAssumedBSize = NNPos::MAX_BOARD_LEN;
static const int svTableMeanRadius = svTableAssumedBSize*svTableAssumedBSize + NNPos::EXTRA_SCORE_DISTR_RADIUS;
static const int svTableMeanLen = svTableMeanRadius*2;
static const int svTableStdevLen = svTableAssumedBSize*svTableAssumedBSize + NNPos::EXTRA_SCORE_DISTR_RADIUS;

void ScoreValue::freeTables() {
  if(scoreValueTablesInitialized) {
    delete[] expectedSVTable;
    expectedSVTable = NULL;
    scoreValueTablesInitialized = false;
  }
}

void ScoreValue::initTables() {
  testAssert(!scoreValueTablesInitialized);
  expectedSVTable = new double[svTableMeanLen*svTableStdevLen];

  //Precompute normal PDF
  const int stepsPerUnit = 10; //Must be divisible by 2. This is both the number of segments that we divide points into, and that we divide stdevs into
  const int boundStdevs = 5;
  int minStdevSteps = -boundStdevs*stepsPerUnit;
  int maxStdevSteps = boundStdevs*stepsPerUnit;
  double* normalPDF = new double[(maxStdevSteps-minStdevSteps)+1];
  for(int i = minStdevSteps; i <= maxStdevSteps; i++) {
    double xInStdevs = (double)i / stepsPerUnit;
    double w = exp(-0.5 * xInStdevs * xInStdevs);
    normalPDF[i-minStdevSteps] = w;
  }
  //Precompute scorevalue at increments of 1/stepsPerUnit points
  int minSVSteps = - (svTableMeanRadius*stepsPerUnit + stepsPerUnit/2 + boundStdevs * svTableStdevLen * stepsPerUnit);
  int maxSVSteps = -minSVSteps;
  double* svPrecomp = new double[(maxSVSteps-minSVSteps)+1];
  for(int i = minSVSteps; i <= maxSVSteps; i++) {
    double mean = (double)i / stepsPerUnit;
    double sv = whiteScoreValueOfScoreSmoothNoDrawAdjust(mean, 0.0, 1.0, svTableAssumedBSize);
    svPrecomp[i-minSVSteps] = sv;
  }

  //Perform numeric integration
  for(int meanIdx = 0; meanIdx < svTableMeanLen; meanIdx++) {
    int meanSteps = (meanIdx - svTableMeanRadius) * stepsPerUnit - stepsPerUnit/2;
    for(int stdevIdx = 0; stdevIdx < svTableStdevLen; stdevIdx++) {
      double wSum = 0.0;
      double wsvSum = 0.0;
      for(int i = minStdevSteps; i <= maxStdevSteps; i++) {
        int xSteps = meanSteps + stdevIdx * i;
        double w = normalPDF[i-minStdevSteps];
        assert(xSteps >= minSVSteps && xSteps <= maxSVSteps);
        double sv = svPrecomp[xSteps-minSVSteps];
        wSum += w;
        wsvSum += w*sv;
      }
      expectedSVTable[meanIdx*svTableStdevLen + stdevIdx] = wsvSum / wSum;
    }
  }

  delete[] normalPDF;
  delete[] svPrecomp;
  scoreValueTablesInitialized = true;
}

double ScoreValue::expectedWhiteScoreValue(double whiteScoreMean, double whiteScoreStdev, double center, double scale, double sqrtBoardArea) {
  testAssert(scoreValueTablesInitialized);

  double scaleFactor = (double)svTableAssumedBSize / (scale * sqrtBoardArea);

  double meanScaled = (whiteScoreMean - center) * scaleFactor;
  double stdevScaled = whiteScoreStdev * scaleFactor;

  double meanRounded = round(meanScaled);
  double stdevFloored = floor(stdevScaled);
  int meanIdx0 = (int)meanRounded + svTableMeanRadius;
  int stdevIdx0 = (int)stdevFloored;
  int meanIdx1 = meanIdx0+1;
  int stdevIdx1 = stdevIdx0+1;

  if(meanIdx0 < 0) { meanIdx0 = 0; meanIdx1 = 0; }
  if(meanIdx1 >= svTableMeanLen) { meanIdx0 = svTableMeanLen-1; meanIdx1 = svTableMeanLen-1; }
  testAssert(stdevIdx0 >= 0);
  if(stdevIdx1 >= svTableStdevLen) { stdevIdx0 = svTableStdevLen-1; stdevIdx1 = svTableStdevLen-1; }

  double lambdaMean = meanScaled - meanRounded + 0.5;
  double lambdaStdev = stdevScaled - stdevFloored;

  double a00 = expectedSVTable[meanIdx0*svTableStdevLen + stdevIdx0];
  double a01 = expectedSVTable[meanIdx0*svTableStdevLen + stdevIdx1];
  double a10 = expectedSVTable[meanIdx1*svTableStdevLen + stdevIdx0];
  double a11 = expectedSVTable[meanIdx1*svTableStdevLen + stdevIdx1];

  double b0 = a00 + lambdaStdev*(a01-a00);
  double b1 = a10 + lambdaStdev*(a11-a10);
  return b0 + lambdaMean*(b1-b0);
}

double ScoreValue::getScoreStdev(double scoreMean, double scoreMeanSq) {
  double variance = scoreMeanSq - scoreMean * scoreMean;
  if(variance <= 0.0)
    return 0.0;
  return sqrt(variance);
}

//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------



//-----------------------------------------------------------------------------------------------------------
//-----------------------------------------------------------------------------------------------------------


NNOutput::NNOutput()
  :whiteOwnerMap(NULL),noisedPolicyProbs(NULL)
{}
NNOutput::NNOutput(const NNOutput& other) {
  nnHash = other.nnHash;
  whiteWinProb = other.whiteWinProb;
  whiteLossProb = other.whiteLossProb;
  whiteNoResultProb = other.whiteNoResultProb;
  whiteScoreMean = other.whiteScoreMean;
  whiteScoreMeanSq = other.whiteScoreMeanSq;
  whiteLead = other.whiteLead;
  varTimeLeft = other.varTimeLeft;
  shorttermWinlossError = other.shorttermWinlossError;
  shorttermScoreError = other.shorttermScoreError;

  nnXLen = other.nnXLen;
  nnYLen = other.nnYLen;
  if(other.whiteOwnerMap != NULL) {
    whiteOwnerMap = new float[nnXLen * nnYLen];
    std::copy(other.whiteOwnerMap, other.whiteOwnerMap + nnXLen * nnYLen, whiteOwnerMap);
  }
  else
    whiteOwnerMap = NULL;

  if(other.noisedPolicyProbs != NULL) {
    noisedPolicyProbs = new float[NNPos::MAX_NN_POLICY_SIZE];
    std::copy(other.noisedPolicyProbs, other.noisedPolicyProbs + NNPos::MAX_NN_POLICY_SIZE, noisedPolicyProbs);
  }
  else
    noisedPolicyProbs = NULL;

  std::copy(other.policyProbs, other.policyProbs+NNPos::MAX_NN_POLICY_SIZE, policyProbs);
  policyOptimismUsed = other.policyOptimismUsed;
}

NNOutput::NNOutput(const vector<shared_ptr<NNOutput>>& others) {
  testAssert(others.size() < 1000000);
  int len = (int)others.size();
  float floatLen = (float)len;
  testAssert(len > 0);
  for(int i = 1; i<len; i++) {
    testAssert(others[i]->nnHash == others[0]->nnHash);
  }
  nnHash = others[0]->nnHash;

  whiteWinProb = 0.0f;
  whiteLossProb = 0.0f;
  whiteNoResultProb = 0.0f;
  whiteScoreMean = 0.0f;
  whiteScoreMeanSq = 0.0f;
  whiteLead = 0.0f;
  varTimeLeft = 0.0f;
  shorttermWinlossError = 0.0f;
  shorttermScoreError = 0.0f;
  for(int i = 0; i<len; i++) {
    const NNOutput& other = *(others[i]);
    whiteWinProb += other.whiteWinProb;
    whiteLossProb += other.whiteLossProb;
    whiteNoResultProb += other.whiteNoResultProb;
    whiteScoreMean += other.whiteScoreMean;
    whiteScoreMeanSq += other.whiteScoreMeanSq;
    whiteLead += other.whiteLead;
    varTimeLeft += other.varTimeLeft;
    shorttermWinlossError += other.shorttermWinlossError;
    shorttermScoreError += other.shorttermScoreError;
  }
  whiteWinProb /= floatLen;
  whiteLossProb /= floatLen;
  whiteNoResultProb /= floatLen;
  whiteScoreMean /= floatLen;
  whiteScoreMeanSq /= floatLen;
  whiteLead /= floatLen;
  varTimeLeft /= floatLen;
  shorttermWinlossError /= floatLen;
  shorttermScoreError /= floatLen;

  nnXLen = others[0]->nnXLen;
  nnYLen = others[0]->nnYLen;

  {
    float whiteOwnerMapCount = 0.0f;
    whiteOwnerMap = NULL;
    for(int i = 0; i<len; i++) {
      const NNOutput& other = *(others[i]);
      if(other.whiteOwnerMap != NULL) {
        if(whiteOwnerMap == NULL) {
          whiteOwnerMap = new float[nnXLen * nnYLen];
          std::fill(whiteOwnerMap, whiteOwnerMap + nnXLen * nnYLen, 0.0f);
        }
        whiteOwnerMapCount += 1.0f;
        for(int pos = 0; pos<nnXLen*nnYLen; pos++)
          whiteOwnerMap[pos] += other.whiteOwnerMap[pos];
      }
    }
    if(whiteOwnerMap != NULL) {
      testAssert(whiteOwnerMapCount > 0);
      for(int pos = 0; pos<nnXLen*nnYLen; pos++)
        whiteOwnerMap[pos] /= whiteOwnerMapCount;
    }
  }

  noisedPolicyProbs = NULL;

  //For technical correctness in case of impossibly rare hash collisions:
  //Just give up if they don't all match in move legality
  {
    bool mismatch = false;
    std::fill(policyProbs, policyProbs + NNPos::MAX_NN_POLICY_SIZE, 0.0f);
    for(int i = 0; i<len; i++) {
      const NNOutput& other = *(others[i]);
      for(int pos = 0; pos<NNPos::MAX_NN_POLICY_SIZE; pos++) {
        if(i > 0 && (policyProbs[pos] < 0) != (other.policyProbs[pos] < 0))
          mismatch = true;
        policyProbs[pos] += other.policyProbs[pos];
      }
    }
    //In case of mismatch, just take the first one
    //This should basically never happen, only on true hash collisions
    if(mismatch) {
      const NNOutput& other = *(others[0]);
      std::copy(other.policyProbs, other.policyProbs + NNPos::MAX_NN_POLICY_SIZE, policyProbs);
    }
    else {
      for(int pos = 0; pos<NNPos::MAX_NN_POLICY_SIZE; pos++)
        policyProbs[pos] /= floatLen;
    }
  }
  {
    bool allOptimismsMatch = true;
    for(int i = 1; i<len; i++) {
      if(others[i]->policyOptimismUsed != others[0]->policyOptimismUsed) {
        allOptimismsMatch = false;
        break;
      }
    }
    if(allOptimismsMatch) {
      policyOptimismUsed = others[0]->policyOptimismUsed;
    }
    else {
      policyOptimismUsed = 0.0;
      for(int i = 0; i<len; i++) {
        policyOptimismUsed += others[i]->policyOptimismUsed / (float)len;
      }
    }
  }
}

NNOutput& NNOutput::operator=(const NNOutput& other) {
  if(&other == this)
    return *this;
  nnHash = other.nnHash;
  whiteWinProb = other.whiteWinProb;
  whiteLossProb = other.whiteLossProb;
  whiteNoResultProb = other.whiteNoResultProb;
  whiteScoreMean = other.whiteScoreMean;
  whiteScoreMeanSq = other.whiteScoreMeanSq;
  whiteLead = other.whiteLead;
  varTimeLeft = other.varTimeLeft;
  shorttermWinlossError = other.shorttermWinlossError;
  shorttermScoreError = other.shorttermScoreError;

  nnXLen = other.nnXLen;
  nnYLen = other.nnYLen;
  if(whiteOwnerMap != NULL)
    delete[] whiteOwnerMap;
  if(other.whiteOwnerMap != NULL) {
    whiteOwnerMap = new float[nnXLen * nnYLen];
    std::copy(other.whiteOwnerMap, other.whiteOwnerMap + nnXLen * nnYLen, whiteOwnerMap);
  }
  else
    whiteOwnerMap = NULL;
  if(noisedPolicyProbs != NULL)
    delete[] noisedPolicyProbs;
  if(other.noisedPolicyProbs != NULL) {
    noisedPolicyProbs = new float[NNPos::MAX_NN_POLICY_SIZE];
    std::copy(other.noisedPolicyProbs, other.noisedPolicyProbs + NNPos::MAX_NN_POLICY_SIZE, noisedPolicyProbs);
  }
  else
    noisedPolicyProbs = NULL;

  std::copy(other.policyProbs, other.policyProbs+NNPos::MAX_NN_POLICY_SIZE, policyProbs);
  policyOptimismUsed = other.policyOptimismUsed;

  return *this;
}


NNOutput::~NNOutput() {
  if(whiteOwnerMap != NULL) {
    delete[] whiteOwnerMap;
    whiteOwnerMap = NULL;
  }
  if(noisedPolicyProbs != NULL) {
    delete[] noisedPolicyProbs;
    noisedPolicyProbs = NULL;
  }
}


void NNOutput::debugPrint(ostream& out, const Board& board) const {
  out << "Win " << Global::strprintf("%.2fc",whiteWinProb*100) << endl;
  out << "Loss " << Global::strprintf("%.2fc",whiteLossProb*100) << endl;
  out << "NoResult " << Global::strprintf("%.2fc",whiteNoResultProb*100) << endl;
  out << "ScoreMean " << Global::strprintf("%.2f",whiteScoreMean) << endl;
  out << "ScoreMeanSq " << Global::strprintf("%.1f",whiteScoreMeanSq) << endl;
  out << "Lead " << Global::strprintf("%.2f",whiteLead) << endl;
  out << "VarTimeLeft " << Global::strprintf("%.1f",varTimeLeft) << endl;
  out << "STWinlossError " << Global::strprintf("%.2fc",shorttermWinlossError*100) << endl;
  out << "STScoreError " << Global::strprintf("%.2f",shorttermScoreError) << endl;
  out << "OptimismUsed " << Global::strprintf("%.2f",policyOptimismUsed) << endl;

  out << "Policy" << endl;
  out << "Pass" << Global::strprintf("%4d ", (int)round(policyProbs[NNPos::getPassPos(nnXLen,nnYLen)] * 1000)) << endl;
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      int pos = NNPos::xyToPos(x,y,nnXLen);
      float prob = policyProbs[pos];
      if(prob < 0)
        out << "   - ";
      else
        out << Global::strprintf("%4d ", (int)round(prob * 1000));
    }
    out << endl;
  }

  if(whiteOwnerMap != NULL) {
    for(int y = 0; y<board.y_size; y++) {
      for(int x = 0; x<board.x_size; x++) {
        int pos = NNPos::xyToPos(x,y,nnXLen);
        float whiteOwn = whiteOwnerMap[pos];
        out << Global::strprintf("%5d ", (int)round(whiteOwn * 1000));
      }
      out << endl;
    }
    out << endl;
  }
}

//-------------------------------------------------------------------------------------------------------------

static void copyWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int cSize, bool useNHWC, int symmetry, bool reverse) {
  bool transpose = (symmetry & 0x4) != 0 && hSize == wSize;
  bool flipX = (symmetry & 0x2) != 0;
  bool flipY = (symmetry & 0x1) != 0;
  if(transpose && !reverse)
    std::swap(flipX,flipY);
  if(useNHWC) {
    int nStride = hSize * wSize * cSize;
    int hStride = wSize * cSize;
    int wStride = cSize;
    int hBaseNew = 0; int hStrideNew = hStride;
    int wBaseNew = 0; int wStrideNew = wStride;

    if(flipY) { hBaseNew = (hSize-1) * hStrideNew; hStrideNew = -hStrideNew; }
    if(flipX) { wBaseNew = (wSize-1) * wStrideNew; wStrideNew = -wStrideNew; }

    if(transpose)
      std::swap(hStrideNew,wStrideNew);

    for(int n = 0; n<nSize; n++) {
      for(int h = 0; h<hSize; h++) {
        int nhOld = n * nStride + h*hStride;
        int nhNew = n * nStride + hBaseNew + h*hStrideNew;
        for(int w = 0; w<wSize; w++) {
          int nhwOld = nhOld + w*wStride;
          int nhwNew = nhNew + wBaseNew + w*wStrideNew;
          for(int c = 0; c<cSize; c++) {
            dst[nhwNew + c] = src[nhwOld + c];
          }
        }
      }
    }
  }
  else {
    int ncSize = nSize * cSize;
    int ncStride = hSize * wSize;
    int hStride = wSize;
    int wStride = 1;
    int hBaseNew = 0; int hStrideNew = hStride;
    int wBaseNew = 0; int wStrideNew = wStride;

    if(flipY) { hBaseNew = (hSize-1) * hStrideNew; hStrideNew = -hStrideNew; }
    if(flipX) { wBaseNew = (wSize-1) * wStrideNew; wStrideNew = -wStrideNew; }

    if(transpose)
      std::swap(hStrideNew,wStrideNew);

    for(int nc = 0; nc<ncSize; nc++) {
      for(int h = 0; h<hSize; h++) {
        int nchOld = nc * ncStride + h*hStride;
        int nchNew = nc * ncStride + hBaseNew + h*hStrideNew;
        for(int w = 0; w<wSize; w++) {
          int nchwOld = nchOld + w*wStride;
          int nchwNew = nchNew + wBaseNew + w*wStrideNew;
          dst[nchwNew] = src[nchwOld];
        }
      }
    }
  }
}


void SymmetryHelpers::copyInputsWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int cSize, bool useNHWC, int symmetry) {
  copyWithSymmetry(src, dst, nSize, hSize, wSize, cSize, useNHWC, symmetry, false);
}

void SymmetryHelpers::copyOutputsWithSymmetry(const float* src, float* dst, int nSize, int hSize, int wSize, int symmetry) {
  copyWithSymmetry(src, dst, nSize, hSize, wSize, 1, false, symmetry, true);
}

int SymmetryHelpers::invert(int symmetry) {
  if(symmetry == 5)
    return 6;
  if(symmetry == 6)
    return 5;
  return symmetry;
}

int SymmetryHelpers::compose(int firstSymmetry, int nextSymmetry) {
  if(isTranspose(firstSymmetry))
    nextSymmetry = (nextSymmetry & 0x4) | ((nextSymmetry & 0x2) >> 1) | ((nextSymmetry & 0x1) << 1);
  return firstSymmetry ^ nextSymmetry;
}

int SymmetryHelpers::compose(int firstSymmetry, int nextSymmetry, int nextNextSymmetry) {
  return compose(compose(firstSymmetry,nextSymmetry),nextNextSymmetry);
}

Loc SymmetryHelpers::getSymLoc(int x, int y, int xSize, int ySize, int symmetry) {
  bool transpose = (symmetry & 0x4) != 0;
  bool flipX = (symmetry & 0x2) != 0;
  bool flipY = (symmetry & 0x1) != 0;
  if(flipX) { x = xSize - x - 1; }
  if(flipY) { y = ySize - y - 1; }

  if(transpose)
    std::swap(x,y);
  return Location::getLoc(x,y,transpose ? ySize : xSize);
}

Loc SymmetryHelpers::getSymLoc(int x, int y, const Board& board, int symmetry) {
  return getSymLoc(x,y,board.x_size,board.y_size,symmetry);
}

Loc SymmetryHelpers::getSymLoc(Loc loc, const Board& board, int symmetry) {
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  return getSymLoc(Location::getX(loc,board.x_size), Location::getY(loc,board.x_size), board, symmetry);
}

Loc SymmetryHelpers::getSymLoc(Loc loc, int xSize, int ySize, int symmetry) {
  if(loc == Board::NULL_LOC || loc == Board::PASS_LOC)
    return loc;
  return getSymLoc(Location::getX(loc,xSize), Location::getY(loc,xSize), xSize, ySize, symmetry);
}


Board SymmetryHelpers::getSymBoard(const Board& board, int symmetry) {
  bool transpose = (symmetry & 0x4) != 0;
  bool flipX = (symmetry & 0x2) != 0;
  bool flipY = (symmetry & 0x1) != 0;
  Board symBoard(
    transpose ? board.y_size : board.x_size,
    transpose ? board.x_size : board.y_size
  );
  Loc symKoLoc = Board::NULL_LOC;
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x,y,board.x_size);
      int symX = flipX ? board.x_size - x - 1 : x;
      int symY = flipY ? board.y_size - y - 1 : y;
      if(transpose)
        std::swap(symX,symY);
      Loc symLoc = Location::getLoc(symX,symY,symBoard.x_size);
      bool suc = symBoard.setStoneFailIfNoLibs(symLoc,board.colors[loc]);
      testAssert(suc);
      if(loc == board.ko_loc)
        symKoLoc = symLoc;
    }
  }
  //Set only at the end because otherwise setStoneFailIfNoLibs clears it.
  if(symKoLoc != Board::NULL_LOC)
    symBoard.setSimpleKoLoc(symKoLoc);
  return symBoard;
}

void SymmetryHelpers::markDuplicateMoveLocs(
  const Board& board,
  const BoardHistory& hist,
  const std::vector<int>* onlySymmetries,
  const std::vector<int>& avoidMoves,
  bool* isSymDupLoc,
  std::vector<int>& validSymmetries
) {
  std::fill(isSymDupLoc, isSymDupLoc + Board::MAX_ARR_SIZE, false);
  validSymmetries.clear();
  validSymmetries.reserve(SymmetryHelpers::NUM_SYMMETRIES);
  validSymmetries.push_back(0);

  //The board should never be considered symmetric if any moves are banned by ko or superko
  if(board.ko_loc != Board::NULL_LOC)
    return;
  for(int y = 0; y < board.y_size; y++) {
    for(int x = 0; x < board.x_size; x++) {
      if(hist.superKoBanned[Location::getLoc(x, y, board.x_size)])
        return;
    }
  }

  //If board has different sizes of x and y, we will not search symmetries involved with transpose.
  int symmetrySearchUpperBound = board.x_size == board.y_size ? SymmetryHelpers::NUM_SYMMETRIES : SymmetryHelpers::NUM_SYMMETRIES_WITHOUT_TRANSPOSE;

  for(int symmetry = 1; symmetry < symmetrySearchUpperBound; symmetry++) {
    if(onlySymmetries != NULL && !contains(*onlySymmetries,symmetry))
      continue;

    bool isBoardSym = true;
    for(int y = 0; y < board.y_size; y++) {
      for(int x = 0; x < board.x_size; x++) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        Loc symLoc = getSymLoc(x, y, board,symmetry);
        bool isStoneSym = (board.colors[loc] == board.colors[symLoc]);
        bool isKoRecapBlockedSym = hist.encorePhase > 0 ? hist.koRecapBlocked[loc] == hist.koRecapBlocked[symLoc] : true;
        bool isSecondEncoreStartColorsSym = hist.encorePhase == 2 ? hist.secondEncoreStartColors[loc] == hist.secondEncoreStartColors[symLoc] : true;
        if(!isStoneSym || !isKoRecapBlockedSym || !isSecondEncoreStartColorsSym) {
          isBoardSym = false;
          break;
        }
      }
      if(!isBoardSym)
        break;
    }
    if(isBoardSym)
      validSymmetries.push_back(symmetry);
  }

  //The way we iterate is to achieve https://senseis.xmp.net/?PlayingTheFirstMoveInTheUpperRightCorner%2FDiscussion
  //Reverse the iteration order for white, so that natural openings result in white on the left and black on the right
  //as is common now in SGFs
  if(hist.presumedNextMovePla == P_BLACK) {
    for(int x = board.x_size-1; x >= 0; x--) {
      for(int y = 0; y < board.y_size; y++) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        if(avoidMoves.size() > 0 && avoidMoves[loc] > 0)
          continue;
        for(int symmetry: validSymmetries) {
          if(symmetry == 0)
            continue;
          Loc symLoc = getSymLoc(x, y, board, symmetry);
          if(!isSymDupLoc[loc] && loc != symLoc)
            isSymDupLoc[symLoc] = true;
        }
      }
    }
  }
  else {
    for(int x = 0; x < board.x_size; x++) {
      for(int y = board.y_size-1; y >= 0; y--) {
        Loc loc = Location::getLoc(x, y, board.x_size);
        if(avoidMoves.size() > 0 && avoidMoves[loc] > 0)
          continue;
        for(int symmetry: validSymmetries) {
          if(symmetry == 0)
            continue;
          Loc symLoc = getSymLoc(x, y, board, symmetry);
          if(!isSymDupLoc[loc] && loc != symLoc)
            isSymDupLoc[symLoc] = true;
        }
      }
    }
  }
}

static double getSymmetryDifference(const Board& board, const Board& other, int symmetry, double maxDifferenceToReport) {
  double thisDifference = 0.0;
  for(int y = 0; y<board.y_size; y++) {
    for(int x = 0; x<board.x_size; x++) {
      Loc loc = Location::getLoc(x, y, board.x_size);
      Loc symLoc = SymmetryHelpers::getSymLoc(x, y, board, symmetry);
      // Difference!
      if(board.colors[loc] != other.colors[symLoc]) {
        // One of them was empty, the other was a stone
        if(board.colors[loc] == C_EMPTY || other.colors[symLoc] == C_EMPTY)
          thisDifference += 1.0;
        // Differing stones - triple the penalty
        else
          thisDifference += 3.0;

        if(thisDifference > maxDifferenceToReport)
          return maxDifferenceToReport;
      }
    }
  }
  return thisDifference;
}


// For each symmetry, return a metric about the "amount" of difference that board would have with other
// if symmetry were applied to board.
void SymmetryHelpers::getSymmetryDifferences(
  const Board& board, const Board& other, double maxDifferenceToReport, double symmetryDifferences[SymmetryHelpers::NUM_SYMMETRIES]
) {
  for(int symmetry = 0; symmetry<SymmetryHelpers::NUM_SYMMETRIES; symmetry++)
    symmetryDifferences[symmetry] = maxDifferenceToReport;

  // Don't bother handling ultra-fancy transpose logic
  if(board.x_size != other.x_size || board.y_size != other.y_size)
    return;

  int numSymmetries = SymmetryHelpers::NUM_SYMMETRIES;
  if(board.x_size != board.y_size)
    numSymmetries = SymmetryHelpers::NUM_SYMMETRIES_WITHOUT_TRANSPOSE;

  for(int symmetry = 0; symmetry<numSymmetries; symmetry++) {
    symmetryDifferences[symmetry] = getSymmetryDifference(board, other, symmetry, maxDifferenceToReport);
  }
}


//-------------------------------------------------------------------------------------------------------------

static void setRowBin(float* rowBin, int pos, int feature, float value, int posStride, int featureStride) {
  rowBin[pos * posStride + feature * featureStride] = value;
}


//Currently does NOT depend on history (except for marking ko-illegal spots)
Hash128 NNInputs::getHash(
  const Board& board, const BoardHistory& hist, Player nextPlayer,
  const MiscNNInputParams& nnInputParams
) {
  //Hash using the effective BoardHistoryModes for this eval, which are normally hist's own
  //but may be overridden per-query (e.g. for a secondary net whose declared featurization differs).
  Hash128 hash = BoardHistory::getSituationRulesAndKoHash(
    board, hist, nextPlayer, nnInputParams.drawEquivalentWinsForWhite,
    nnInputParams.getModes(hist)
  );

  //Fold in whether a pass ends this phase.
  if(hist.passWouldEndPhase(board,nextPlayer)) {
    hash ^= Board::ZOBRIST_PASS_ENDS_PHASE;
    //Technically some of the below only apply when passing ends the game, but it's pretty harmless to use the more
    //conservative hashing including them when the phase would end too.

    //And in the case that a pass ends the phase, conservativePass also affects the result for the root node
    if(nnInputParams.conservativePassAndIsRoot)
      hash ^= MiscNNInputParams::ZOBRIST_CONSERVATIVE_PASS;

    //If we're in a ruleset where passing without capturing all the stones is okay, and as a result are suppressing
    //the game end effect of a pass during search, hash this in.
    if(hist.shouldSuppressEndGameFromFriendlyPass(board,nextPlayer))
      hash ^= MiscNNInputParams::ZOBRIST_FRIENDLY_PASS;

    //Passing hacks can also affect things at game or phase end.
    if(nnInputParams.enablePassingHacks)
      hash ^= MiscNNInputParams::ZOBRIST_PASSING_HACKS;
  }
  //Fold in whether the game is over or not, since this affects how we compute input features
  //but is not a function necessarily of previous hashed values.
  //If the history is in a weird prolonged state, also treat it similarly.
  if(hist.isGameFinished || hist.isPastNormalPhaseEnd)
    hash ^= Board::ZOBRIST_GAME_IS_OVER;

  //Distinguish between forcing the history to always be empty and allow any nonempty amount
  //We already tolerate caching and reusing evals across distinct transpositions with different
  //history, for performance reasons, but the case of no history at all is probably distinct
  //enough that we should distinguish it.
  if(nnInputParams.maxHistory <= 0) {
    hash.hash0 += MiscNNInputParams::ZOBRIST_ZERO_HISTORY.hash0;
    hash.hash1 += MiscNNInputParams::ZOBRIST_ZERO_HISTORY.hash1;
  }

  //Fold in asymmetric playout indicator
  if(nnInputParams.playoutDoublingAdvantage != 0) {
    int64_t playoutDoublingsDiscretized = (int64_t)(nnInputParams.playoutDoublingAdvantage*256.0f);
    hash.hash0 += Hash::splitMix64((uint64_t)playoutDoublingsDiscretized);
    hash.hash1 += Hash::basicLCong((uint64_t)playoutDoublingsDiscretized);
    hash ^= MiscNNInputParams::ZOBRIST_PLAYOUT_DOUBLINGS;
  }

  //Fold in policy temperature
  if(nnInputParams.nnPolicyTemperature != 1.0f) {
    int64_t nnPolicyTemperatureDiscretized = (int64_t)(nnInputParams.nnPolicyTemperature*2048.0f);
    hash.hash0 ^= Hash::basicLCong2((uint64_t)nnPolicyTemperatureDiscretized);
    hash.hash1 = Hash::splitMix64(hash.hash1 + (uint64_t)nnPolicyTemperatureDiscretized);
    hash.hash0 += hash.hash1;
    hash ^= MiscNNInputParams::ZOBRIST_NN_POLICY_TEMP;
  }

  if(nnInputParams.avoidMYTDaggerHack)
    hash ^= MiscNNInputParams::ZOBRIST_AVOID_MYTDAGGER_HACK;

  //Fold in policy optimism
  if(nnInputParams.policyOptimism > 0) {
    hash ^= MiscNNInputParams::ZOBRIST_POLICY_OPTIMISM;
    int64_t policyOptimismDiscretized = (int64_t)(nnInputParams.policyOptimism*1024.0);
    hash.hash0 = Hash::rrmxmx(Hash::splitMix64(hash.hash0) + (uint64_t)policyOptimismDiscretized);
    hash.hash1 = Hash::rrmxmx(hash.hash1 + hash.hash0 + (uint64_t)policyOptimismDiscretized);
  }

  return hash;
}

bool MiscNNInputParams::getAlwaysComputePassAliveUnderSuicideRules(const BoardHistory& hist) const {
  if(passAliveSuicideRulesOverride >= 0)
    return passAliveSuicideRulesOverride != 0;
  return hist.modes.alwaysComputePassAliveUnderSuicideRules;
}

bool MiscNNInputParams::getExcludeTerritoryAdjacentToAtari(const BoardHistory& hist) const {
  if(excludeTerritoryAdjAtariOverride >= 0)
    return excludeTerritoryAdjAtariOverride != 0;
  return hist.modes.excludeTerritoryAdjacentToAtari;
}

BoardHistoryModes MiscNNInputParams::getModes(const BoardHistory& hist) const {
  return BoardHistoryModes(
    getAlwaysComputePassAliveUnderSuicideRules(hist),
    getExcludeTerritoryAdjacentToAtari(hist)
  );
}

bool MiscNNInputParams::getSuicideLegalForPassAlive(const BoardHistory& hist) const {
  return hist.rules.multiStoneSuicideLegal || getAlwaysComputePassAliveUnderSuicideRules(hist);
}


//===========================================================================================
// INPUTSVERSION 1 (Quoridor)
//===========================================================================================

void NNInputs::fillRowV1(
  const Board& board, const BoardHistory& boardHistory, Player nextPlayer,
  const MiscNNInputParams& nnInputParams, int nnXLen, int nnYLen, bool useNHWC, float* rowBin, float* rowGlobal
) {
  (void)nnInputParams;
  assert(nnXLen == NN_X_LEN);
  assert(nnYLen == NN_Y_LEN);
  assert(nextPlayer == P_BLACK || nextPlayer == P_WHITE);

  std::fill(rowBin, rowBin + NUM_FEATURES_SPATIAL_V1 * nnXLen * nnYLen, 0.0f);
  std::fill(rowGlobal, rowGlobal + NUM_FEATURES_GLOBAL_V1, 0.0f);

  int featureStride;
  int posStride;
  if(useNHWC) {
    featureStride = 1;
    posStride = NUM_FEATURES_SPATIAL_V1;
  }
  else {
    featureStride = nnXLen * nnYLen;
    posStride = 1;
  }

  // Current and opponent pawns
  Loc curPawnLoc = (nextPlayer == P_BLACK) ? board.blackPawnLoc : board.whitePawnLoc;
  Loc oppPawnLoc = (nextPlayer == P_BLACK) ? board.whitePawnLoc : board.blackPawnLoc;
  int curPawnC = Location::getX(curPawnLoc, board.x_size) / 2;
  int curPawnR = Location::getY(curPawnLoc, board.x_size) / 2;
  int oppPawnC = Location::getX(oppPawnLoc, board.x_size) / 2;
  int oppPawnR = Location::getY(oppPawnLoc, board.x_size) / 2;

  // Board edge blocks
  bool blockedN[9][9];
  bool blockedS[9][9];
  bool blockedE[9][9];
  bool blockedW[9][9];

  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      Loc loc = Location::pawnLoc(c, r, board.x_size);
      blockedN[c][r] = (r == 0) || !board.canPawnStep(loc, Location::pawnLoc(c, r - 1, board.x_size));
      blockedS[c][r] = (r == 8) || !board.canPawnStep(loc, Location::pawnLoc(c, r + 1, board.x_size));
      blockedE[c][r] = (c == 8) || !board.canPawnStep(loc, Location::pawnLoc(c + 1, r, board.x_size));
      blockedW[c][r] = (c == 0) || !board.canPawnStep(loc, Location::pawnLoc(c - 1, r, board.x_size));
    }
  }

  // Placed wall anchors (c, r in [0..7])
  bool hasVWall[8][8];
  bool hasHWall[8][8];
  for(int r = 0; r < 8; r++) {
    for(int c = 0; c < 8; c++) {
      hasVWall[c][r] = false;
      hasHWall[c][r] = false;
      Loc center = Location::hWallLoc(c, r, board.x_size); // (2c+1, 2r+1)
      if(board.colors[center] == C_FENCE) {
        Loc top = center + board.adj_offsets[0]; // (2c+1, 2r)
        if(board.colors[top] == C_FENCE)
          hasVWall[c][r] = true;
        Loc left = center + board.adj_offsets[1]; // (2c, 2r+1)
        if(board.colors[left] == C_FENCE)
          hasHWall[c][r] = true;
      }
    }
  }

  // BFS Distances
  auto runBFS = [&](const std::vector<std::pair<int,int>>& sources, int distMap[9][9]) {
    for(int r = 0; r < 9; r++) {
      for(int c = 0; c < 9; c++) {
        distMap[c][r] = -1;
      }
    }
    int qC[81];
    int qR[81];
    int qHead = 0;
    int qTail = 0;

    for(const auto& s : sources) {
      qC[qTail] = s.first;
      qR[qTail] = s.second;
      qTail++;
      distMap[s.first][s.second] = 0;
    }

    const int dc[4] = {0, 0, 1, -1};
    const int dr[4] = {-1, 1, 0, 0};

    while(qHead < qTail) {
      int c = qC[qHead];
      int r = qR[qHead];
      qHead++;
      int d = distMap[c][r];
      Loc currLoc = Location::pawnLoc(c, r, board.x_size);

      for(int i = 0; i < 4; i++) {
        int nc = c + dc[i];
        int nr = r + dr[i];
        if(nc >= 0 && nc < 9 && nr >= 0 && nr < 9 && distMap[nc][nr] == -1) {
          Loc nextLoc = Location::pawnLoc(nc, nr, board.x_size);
          if(board.canPawnStep(currLoc, nextLoc)) {
            distMap[nc][nr] = d + 1;
            qC[qTail] = nc;
            qR[qTail] = nr;
            qTail++;
          }
        }
      }
    }
  };

  // Black goal row on board is r=0; White goal row on board is r=8
  int curGoalR = (nextPlayer == P_BLACK) ? 0 : 8;
  int oppGoalR = (nextPlayer == P_BLACK) ? 8 : 0;

  int distToGoalCur[9][9];
  int distToGoalOpp[9][9];
  int distFromPawnCur[9][9];
  int distFromPawnOpp[9][9];

  std::vector<std::pair<int,int>> curGoalSources;
  std::vector<std::pair<int,int>> oppGoalSources;
  curGoalSources.reserve(9);
  oppGoalSources.reserve(9);
  for(int c = 0; c < 9; c++) {
    curGoalSources.push_back({c, curGoalR});
    oppGoalSources.push_back({c, oppGoalR});
  }

  runBFS(curGoalSources, distToGoalCur);
  runBFS(oppGoalSources, distToGoalOpp);
  runBFS({{curPawnC, curPawnR}}, distFromPawnCur);
  runBFS({{oppPawnC, oppPawnR}}, distFromPawnOpp);

  int shortestDistCur = distToGoalCur[curPawnC][curPawnR];
  int shortestDistOpp = distToGoalOpp[oppPawnC][oppPawnR];

  bool onPathCur[9][9];
  bool onPathOpp[9][9];
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      onPathCur[c][r] = (shortestDistCur >= 0 && distFromPawnCur[c][r] >= 0 && distToGoalCur[c][r] >= 0 &&
                         (distFromPawnCur[c][r] + distToGoalCur[c][r] == shortestDistCur));
      onPathOpp[c][r] = (shortestDistOpp >= 0 && distFromPawnOpp[c][r] >= 0 && distToGoalOpp[c][r] >= 0 &&
                         (distFromPawnOpp[c][r] + distToGoalOpp[c][r] == shortestDistOpp));
    }
  }

  // Populate 16 spatial channels in canonical perspective
  for(int rCanon = 0; rCanon < 9; rCanon++) {
    for(int c = 0; c < 9; c++) {
      int rBoard = (nextPlayer == P_WHITE) ? (8 - rCanon) : rCanon;
      int cBoard = c;
      int pos = NNPos::xyToPos(c, rCanon, nnXLen);

      // Ch 0: On-board mask
      setRowBin(rowBin, pos, 0, 1.0f, posStride, featureStride);

      // Ch 1: Current player pawn
      if(cBoard == curPawnC && rBoard == curPawnR)
        setRowBin(rowBin, pos, 1, 1.0f, posStride, featureStride);

      // Ch 2: Opponent player pawn
      if(cBoard == oppPawnC && rBoard == oppPawnR)
        setRowBin(rowBin, pos, 2, 1.0f, posStride, featureStride);

      // Ch 3: North-blocked edge (swapped with South if White)
      bool bN = (nextPlayer == P_WHITE) ? blockedS[cBoard][rBoard] : blockedN[cBoard][rBoard];
      if(bN)
        setRowBin(rowBin, pos, 3, 1.0f, posStride, featureStride);

      // Ch 4: South-blocked edge (swapped with North if White)
      bool bS = (nextPlayer == P_WHITE) ? blockedN[cBoard][rBoard] : blockedS[cBoard][rBoard];
      if(bS)
        setRowBin(rowBin, pos, 4, 1.0f, posStride, featureStride);

      // Ch 5: East-blocked edge
      if(blockedE[cBoard][rBoard])
        setRowBin(rowBin, pos, 5, 1.0f, posStride, featureStride);

      // Ch 6: West-blocked edge
      if(blockedW[cBoard][rBoard])
        setRowBin(rowBin, pos, 6, 1.0f, posStride, featureStride);

      // Ch 7: Goal row mask (always rCanon == 0)
      if(rCanon == 0)
        setRowBin(rowBin, pos, 7, 1.0f, posStride, featureStride);

      // Ch 8: Current player goal BFS distance / 32.0f
      int dCur = distToGoalCur[cBoard][rBoard];
      float fCur = (dCur < 0) ? 1.0f : std::min(1.0f, (float)dCur / 32.0f);
      setRowBin(rowBin, pos, 8, fCur, posStride, featureStride);

      // Ch 9: Opponent goal BFS distance / 32.0f
      int dOpp = distToGoalOpp[cBoard][rBoard];
      float fOpp = (dOpp < 0) ? 1.0f : std::min(1.0f, (float)dOpp / 32.0f);
      setRowBin(rowBin, pos, 9, fOpp, posStride, featureStride);

      // Ch 10: Current player pawn BFS distance / 32.0f
      int dpCur = distFromPawnCur[cBoard][rBoard];
      float fpCur = (dpCur < 0) ? 1.0f : std::min(1.0f, (float)dpCur / 32.0f);
      setRowBin(rowBin, pos, 10, fpCur, posStride, featureStride);

      // Ch 11: Opponent pawn BFS distance / 32.0f
      int dpOpp = distFromPawnOpp[cBoard][rBoard];
      float fpOpp = (dpOpp < 0) ? 1.0f : std::min(1.0f, (float)dpOpp / 32.0f);
      setRowBin(rowBin, pos, 11, fpOpp, posStride, featureStride);

      // Ch 12: Current player on-path mask
      if(onPathCur[cBoard][rBoard])
        setRowBin(rowBin, pos, 12, 1.0f, posStride, featureStride);

      // Ch 13: Opponent on-path mask
      if(onPathOpp[cBoard][rBoard])
        setRowBin(rowBin, pos, 13, 1.0f, posStride, featureStride);

      // Ch 14 & 15: Placed wall anchors
      if(c < 8 && rCanon < 8) {
        int rWallBoard = (nextPlayer == P_WHITE) ? (7 - rCanon) : rCanon;
        if(hasVWall[c][rWallBoard])
          setRowBin(rowBin, pos, 14, 1.0f, posStride, featureStride);
        if(hasHWall[c][rWallBoard])
          setRowBin(rowBin, pos, 15, 1.0f, posStride, featureStride);
      }
    }
  }

  // Populate 16 global features
  // Index 0: Next player is White
  rowGlobal[0] = (nextPlayer == P_WHITE) ? 1.0f : 0.0f;

  int myFences = (nextPlayer == P_WHITE) ? board.whiteFences : board.blackFences;
  int oppFences = (nextPlayer == P_WHITE) ? board.blackFences : board.whiteFences;

  // Index 1: My fence count remaining
  rowGlobal[1] = (float)myFences / 10.0f;

  // Index 2: Opponent fence count remaining
  rowGlobal[2] = (float)oppFences / 10.0f;

  // Index 3..6: My fence exponential encoding
  if(myFences > 0) {
    float d = (float)(myFences - 1);
    rowGlobal[3] = std::exp(-d / 1.0f);
    rowGlobal[4] = std::exp(-d / 2.0f);
    rowGlobal[5] = std::exp(-d / 4.0f);
    rowGlobal[6] = std::exp(-d / 8.0f);
  }

  // Index 7: Opponent fence count present
  rowGlobal[7] = (oppFences >= 1) ? 1.0f : 0.0f;

  // Index 8..11: Opponent fence exponential encoding
  if(oppFences > 0) {
    float d = (float)(oppFences - 1);
    rowGlobal[8] = std::exp(-d / 1.0f);
    rowGlobal[9] = std::exp(-d / 2.0f);
    rowGlobal[10] = std::exp(-d / 4.0f);
    rowGlobal[11] = std::exp(-d / 8.0f);
  }

  // Index 12: Action Parity (Jump Tempo)
  // Evaluates whether nextPlayer has the jump tempo if both pawns advance directly:
  // If Manhattan distance between pawns is odd, nextPlayer reaches adjacency on opponent's turn and jumps (+1.0f).
  // If Manhattan distance is even, opponent reaches adjacency on nextPlayer's turn and opponent jumps (-1.0f).
  // / 2 because Location::getX/Y gives coordination on 17x17 board
  int c1 = Location::getX(board.blackPawnLoc, board.x_size) / 2;
  int r1 = Location::getY(board.blackPawnLoc, board.x_size) / 2;
  int c2 = Location::getX(board.whitePawnLoc, board.x_size) / 2;
  int r2 = Location::getY(board.whitePawnLoc, board.x_size) / 2;
  int manhattanDist = std::abs(c1 - c2) + std::abs(r1 - r2);
  rowGlobal[12] = (manhattanDist % 2 != 0) ? 1.0f : -1.0f;
  int moveCount = !boardHistory.moveHistory.empty() ? (int)boardHistory.moveHistory.size() : board.movenum;

  // Index 13: Game progress
  rowGlobal[13] = std::min(1.0f, (float)moveCount / 200.0f);

  // Index 14: My shortest distance
  rowGlobal[14] = (shortestDistCur < 0) ? 1.0f : std::min(1.0f, (float)shortestDistCur / 32.0f);

  // Index 15: Opponent shortest distance
  rowGlobal[15] = (shortestDistOpp < 0) ? 1.0f : std::min(1.0f, (float)shortestDistOpp / 32.0f);
}

void NNInputs::applyPolicyMap(
  const float* rawPolicy243,
  Player nextPlayer,
  float* policyProbs290
) {
  for(int i = 0; i < NNPos::MAX_NN_POLICY_SIZE; i++) {
    policyProbs290[i] = -1e30f;
  }

  // Plane 0: Pawn moves to (c, r)
  for(int r = 0; r < 9; r++) {
    for(int c = 0; c < 9; c++) {
      int rCanon = (nextPlayer == P_WHITE) ? (8 - r) : r;
      int pos = NNPos::locToPos(Location::pawnLoc(c, r, 17), 17, 17, 17);
      policyProbs290[pos] = rawPolicy243[0 * 81 + rCanon * 9 + c];
    }
  }

  // Plane 1: Vertical walls at (c, r) [Upper arm]
  for(int r = 0; r < 8; r++) {
    for(int c = 0; c < 8; c++) {
      int rCanon = (nextPlayer == P_WHITE) ? (7 - r) : r;
      int pos = NNPos::locToPos(Location::vWallLoc(c, r, 17), 17, 17, 17);
      policyProbs290[pos] = rawPolicy243[1 * 81 + rCanon * 9 + c];
    }
  }

  // Plane 2: Horizontal walls at (c, r) [Center]
  for(int r = 0; r < 8; r++) {
    for(int c = 0; c < 8; c++) {
      int rCanon = (nextPlayer == P_WHITE) ? (7 - r) : r;
      int pos = NNPos::locToPos(Location::hWallLoc(c, r, 17), 17, 17, 17);
      policyProbs290[pos] = rawPolicy243[2 * 81 + rCanon * 9 + c];
    }
  }
}
