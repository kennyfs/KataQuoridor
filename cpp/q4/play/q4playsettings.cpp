#include "q4playsettings.h"

#include "../../program/setup.h"

namespace Q4Play {

Q4PlaySettings::Q4PlaySettings()
  : initGamesWithPolicy(false),
    policyInitAreaProp(0.0),
    policyInitGammaShape(1.0),
    policyInitAreaTemperature(1.0),
    sidePositionProb(0.0),
    earlyForkGameProb(0.0),
    earlyForkGameExpectedMoveProp(0.0),
    forkGameProb(0.0),
    forkGameMinChoices(1),
    earlyForkGameMaxChoices(1),
    forkGameMaxChoices(1),
    cheapSearchProb(0.0),
    cheapSearchVisits(0),
    cheapSearchTargetWeight(0.0f),
    reduceVisits(false),
    reduceVisitsThreshold(100.0),
    reduceVisitsThresholdLookback(1),
    reducedVisitsMin(0),
    reducedVisitsWeight(1.0f),
    policySurpriseDataWeight(0.0),
    valueSurpriseDataWeight(0.0),
    useSearchValueSurprise(false),
    scaleDataWeight(1.0),
    noResolveTargetWeights(false),
    q4EliminationProb(0.03),
    opponentMode(Q4S::OPPONENT_MAXN),
    forSelfPlay(false),
    recordTimePerMove(false),
    logSearchTreeOptions(Setup::defaultLogSearchTreeOptions())
{}

Q4PlaySettings::~Q4PlaySettings()
{}

Q4PlaySettings Q4PlaySettings::loadForSelfplay(ConfigParser& cfg) {
  // Reject configs with removed features enabled (prompt B3)
  if(cfg.contains("allowResignation") && cfg.getBool("allowResignation"))
    throw StringError("allowResignation must be absent or false in Q4 self-play (resignation is removed)");
  if(cfg.contains("useReanalyze") && cfg.getBool("useReanalyze"))
    throw StringError("useReanalyze must be absent or false in Q4 self-play (reanalysis is removed)");
  if(cfg.contains("recordTreePositions") && cfg.getBool("recordTreePositions"))
    throw StringError("recordTreePositions must be absent or false in Q4 self-play (recordTreePositions is removed)");
  if(cfg.contains("estimateLeadProb") && cfg.getDouble("estimateLeadProb", 0.0, 1.0) > 0.0)
    throw StringError("estimateLeadProb must be absent or 0 in Q4 self-play (lead estimation is removed)");
  if(cfg.contains("sekiForkHackProb") && cfg.getDouble("sekiForkHackProb", 0.0, 1.0) > 0.0)
    throw StringError("sekiForkHackProb must be absent or 0 in Q4 self-play (seki hacks are removed)");
  if(cfg.contains("fancyKomiVarying") && cfg.getBool("fancyKomiVarying"))
    throw StringError("fancyKomiVarying must be absent or false in Q4 self-play (komi is removed)");
  if(cfg.contains("normalAsymmetricPlayoutProb") && cfg.getDouble("normalAsymmetricPlayoutProb", 0.0, 1.0) > 0.0)
    throw StringError("normalAsymmetricPlayoutProb must be absent or 0 in Q4 self-play (asymmetric playouts removed)");
  if(cfg.contains("handicapAsymmetricPlayoutProb") && cfg.getDouble("handicapAsymmetricPlayoutProb", 0.0, 1.0) > 0.0)
    throw StringError("handicapAsymmetricPlayoutProb must be absent or 0 in Q4 self-play (asymmetric playouts removed)");
  if(cfg.contains("quoridorKomiRandomProb") && cfg.getDouble("quoridorKomiRandomProb", 0.0, 1.0) > 0.0)
    throw StringError("quoridorKomiRandomProb must be absent or 0 in Q4 self-play (komi randomization removed)");
  if(cfg.contains("quoridorFenceHandicapProb") && cfg.getDouble("quoridorFenceHandicapProb", 0.0, 1.0) > 0.0)
    throw StringError("quoridorFenceHandicapProb must be absent or 0 in Q4 self-play (fence handicap removed)");

  Q4PlaySettings playSettings;
  playSettings.initGamesWithPolicy = cfg.getBool("initGamesWithPolicy");
  playSettings.policyInitAreaProp = cfg.contains("policyInitAreaProp") ? cfg.getDouble("policyInitAreaProp", 0.0, 1.0) : 0.04;
  playSettings.policyInitGammaShape = cfg.contains("policyInitGammaShape") ? cfg.getDouble("policyInitGammaShape", 0.5, 10.0) : 1.0;
  playSettings.sidePositionProb =
    (cfg.contains("forkSidePositionProb") && !cfg.contains("sidePositionProb")) ?
    cfg.getDouble("forkSidePositionProb", 0.0, 1.0) : cfg.getDouble("sidePositionProb", 0.0, 1.0);
  playSettings.policyInitAreaTemperature = cfg.contains("policyInitAreaTemperature") ? cfg.getDouble("policyInitAreaTemperature", 0.1, 5.0) : 1.0;

  playSettings.earlyForkGameProb = cfg.getDouble("earlyForkGameProb", 0.0, 1.0);
  playSettings.earlyForkGameExpectedMoveProp = cfg.getDouble("earlyForkGameExpectedMoveProp", 0.0, 1.0);
  playSettings.forkGameProb = cfg.getDouble("forkGameProb", 0.0, 1.0);
  playSettings.forkGameMinChoices = cfg.getInt("forkGameMinChoices", 1, 100);
  playSettings.earlyForkGameMaxChoices = cfg.getInt("earlyForkGameMaxChoices", 1, 100);
  playSettings.forkGameMaxChoices = cfg.getInt("forkGameMaxChoices", 1, 100);

  playSettings.cheapSearchProb = cfg.getDouble("cheapSearchProb", 0.0, 1.0);
  playSettings.cheapSearchVisits = cfg.getInt("cheapSearchVisits", 1, 10000000);
  playSettings.cheapSearchTargetWeight = cfg.getFloat("cheapSearchTargetWeight", 0.0f, 1.0f);

  playSettings.reduceVisits = cfg.getBool("reduceVisits");
  playSettings.reduceVisitsThreshold = cfg.getDouble("reduceVisitsThreshold", 0.0, 0.999999);
  playSettings.reduceVisitsThresholdLookback = cfg.getInt("reduceVisitsThresholdLookback", 0, 1000);
  playSettings.reducedVisitsMin = cfg.getInt("reducedVisitsMin", 1, 10000000);
  playSettings.reducedVisitsWeight = cfg.getFloat("reducedVisitsWeight", 0.0f, 1.0f);

  playSettings.policySurpriseDataWeight = cfg.getDouble("policySurpriseDataWeight", 0.0, 1.0);
  playSettings.valueSurpriseDataWeight = cfg.getDouble("valueSurpriseDataWeight", 0.0, 1.0);
  playSettings.useSearchValueSurprise = cfg.contains("useSearchValueSurprise") ? cfg.getBool("useSearchValueSurprise") : false;
  playSettings.scaleDataWeight = cfg.contains("scaleDataWeight") ? cfg.getDouble("scaleDataWeight", 0.01, 10.0) : 1.0;
  playSettings.noResolveTargetWeights = cfg.contains("noResolveTargetWeights") ? cfg.getBool("noResolveTargetWeights") : false;

  playSettings.q4EliminationProb = cfg.contains("q4EliminationProb") ? cfg.getDouble("q4EliminationProb", 0.0, 1.0) : 0.03;

  playSettings.population = Q4PopulationSettings::load(cfg);
  playSettings.opponentMode = Q4S::loadOpponentMode(cfg);

  playSettings.forSelfPlay = true;
  playSettings.recordTimePerMove = cfg.contains("recordTimePerMove") ? cfg.getBool("recordTimePerMove") : false;
  playSettings.logSearchTreeOptions = Setup::loadLogSearchTreeOptions(cfg);

  if(playSettings.policySurpriseDataWeight + playSettings.valueSurpriseDataWeight > 1.0)
    throw StringError("policySurpriseDataWeight + valueSurpriseDataWeight > 1.0");

  return playSettings;
}

}  // namespace Q4Play
