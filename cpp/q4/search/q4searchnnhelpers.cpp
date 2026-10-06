#include "q4search.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

#include "../../core/test.h"
#include "../nn/q4nn.h"
#include "q4searchnode.h"

namespace Q4S {

void Search::computeRootNNEvaluation(NNResultBuf& nnResultBuf) {
  (void)nnResultBuf;
}

bool Search::initNodeNNOutput(
  SearchThread& thread, SearchNode& node,
  bool isRoot, bool skipCache, bool isReInit
) {
  Q4NN::Eval eval;
  if(isRoot && searchParams.rootNumSymmetriesToSample > 1) {
    Q4NN::averageMultipleSymmetries(
      *nnEvaluator,
      thread.nnResultBuf,
      thread.state,
      thread.rand,
      searchParams.rootNumSymmetriesToSample,
      eval,
      (float)searchParams.nnPolicyTemperature
    );
  }
  else {
    int sym = -1;
    Q4NN::evaluate(
      *nnEvaluator,
      thread.nnResultBuf,
      thread.state,
      sym,
      skipCache,
      eval,
      &thread.rand,
      (float)searchParams.nnPolicyTemperature
    );
  }

  std::shared_ptr<NNOutput>* result = new std::shared_ptr<NNOutput>(new NNOutput());
  NNOutput* out = result->get();

  std::vector<int> legalActions;
  thread.state.getLegalActions(legalActions);
  std::vector<bool> isLegal(Q4Board::NUM_ACTIONS, false);
  for(int a : legalActions)
    isLegal[a] = true;

  for(int a = 0; a < Q4Board::NUM_ACTIONS; a++) {
    if(isLegal[a])
      out->policyProbs[a] = eval.policyProbs[0][a];
    else
      out->policyProbs[a] = -1.0f;
  }

  for(int i = 0; i < 5; i++)
    out->valueAbs[i] = eval.valueAbsMasked[i];

  out->shorttermWinlossError = eval.shorttermWinlossError;
  out->nnHash = Q4NN::getCacheHash(thread.state);

  assert(out->noisedPolicyProbs == NULL);
  std::shared_ptr<NNOutput>* noisedResult = maybeAddPolicyNoiseAndTemp(thread, isRoot, out);
  if(noisedResult != NULL) {
    std::shared_ptr<NNOutput>* tmp = result;
    result = noisedResult;
    delete tmp;
  }

  node.nodeAge.store(searchNodeAge, std::memory_order_release);

  if(isReInit) {
    bool wasNullBefore = node.storeNNOutput(result, thread);
    return wasNullBefore;
  }
  else {
    bool suc = node.storeNNOutputIfNull(result);
    if(!suc)
      delete result;
    else
      addCurrentNNOutputAsLeafValue(node, true);
    return suc;
  }
}

bool Search::maybeRecomputeExistingNNOutput(
  SearchThread& thread, SearchNode& node, bool isRoot
) {
  bool recomputeHappened = false;
  if(isRoot && node.nodeAge.load(std::memory_order_acquire) != searchNodeAge) {
    uint32_t oldAge = node.nodeAge.exchange(searchNodeAge, std::memory_order_acq_rel);
    if(oldAge < searchNodeAge) {
      NNOutput* nnOutput = node.getNNOutput();
      testAssert(nnOutput != NULL);

      if(searchParams.rootNumSymmetriesToSample > 1) {
        const bool skipCache = false;
        initNodeNNOutput(thread, node, isRoot, skipCache, true);
        recomputeHappened = true;
      }
      else {
        std::shared_ptr<NNOutput>* result = maybeAddPolicyNoiseAndTemp(thread, isRoot, nnOutput);
        if(result != NULL) {
          node.storeNNOutput(result, thread);
          recomputeHappened = true;
        }
      }
    }
  }
  return recomputeHappened;
}

}  // namespace Q4S
