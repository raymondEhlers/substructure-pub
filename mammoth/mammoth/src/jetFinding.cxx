#include "mammoth/jetFinding.hpp"

namespace mammoth {

namespace detail {
// Base framework for working with fastjet selector.
// This is mostly copied directly from fastjet (ie. Selector.cc) since they don't make it part of the
// public interface, but it's quite useful as an example for implementing a new selector

/**
  * @brief Container for a quantity
  *
  */
class QuantityBase {
public:
  QuantityBase(double q) : _q(q){}
  virtual ~QuantityBase(){}
  virtual double operator()(const fastjet::PseudoJet & jet ) const =0;
  virtual std::string description() const =0;
  virtual bool is_geometric() const { return false;}
  virtual double comparison_value() const {return _q;}
  virtual double description_value() const {return comparison_value();}
protected:
  double _q;
};

/**
  * @brief Container for a squared quantity
  *
  */
class QuantitySquareBase : public QuantityBase {
public:
  QuantitySquareBase(double sqrtq) : QuantityBase(sqrtq*sqrtq), _sqrtq(sqrtq){}
  virtual double description_value() const {return _sqrtq;}
protected:
  double _sqrtq;
};

/**
  * @brief quantity >= minimum
  *
  * @tparam QuantityType Container for some quantity
  */
template<typename QuantityType>
class SW_QuantityMin : public fastjet::SelectorWorker {
public:
  /// detfault ctor (initialises the pt cut)
  SW_QuantityMin(double qmin) : _qmin(qmin) {}

  /// returns true is the given object passes the selection pt cut
  virtual bool pass(const fastjet::PseudoJet & jet) const {return _qmin(jet) >= _qmin.comparison_value();}

  /// returns a description of the worker
  virtual std::string description() const {
    return _qmin.description() + " >= " + std::to_string(_qmin.description_value());
  }

  virtual bool is_geometric() const { return _qmin.is_geometric();}

protected:
  QuantityType _qmin;     ///< the cut
};

/**
  * @brief quantity <= maximum
  *
  * @tparam QuantityType Container for some quantity
  */
template<typename QuantityType>
class SW_QuantityMax : public fastjet::SelectorWorker {
public:
  /// detfault ctor (initialises the pt cut)
  SW_QuantityMax(double qmax) : _qmax(qmax) {}

  /// returns true is the given object passes the selection pt cut
  virtual bool pass(const fastjet::PseudoJet & jet) const {return _qmax(jet) <= _qmax.comparison_value();}

  /// returns a description of the worker
  virtual std::string description() const {
    return _qmax.description() + " <= " + std::to_string(_qmax.description_value());
  }

  virtual bool is_geometric() const { return _qmax.is_geometric();}

protected:
  QuantityType _qmax;   ///< the cut
};


/**
  * @brief minimum <= quantity <= maximum
  *
  * @tparam QuantityType Container for some quantity
  */
template<typename QuantityType>
class SW_QuantityRange : public fastjet::SelectorWorker {
public:
  /// detfault ctor (initialises the pt cut)
  SW_QuantityRange(double qmin, double qmax) : _qmin(qmin), _qmax(qmax) {}

  /// returns true is the given object passes the selection pt cut
  virtual bool pass(const fastjet::PseudoJet & jet) const {
    double q = _qmin(jet); // we could identically use _qmax
    return (q >= _qmin.comparison_value()) && (q <= _qmax.comparison_value());
  }

  /// returns a description of the worker
  virtual std::string description() const {
    return std::to_string(_qmin.description_value()) + " <= " + _qmin.description() + " <= " + std::to_string(_qmax.description_value());
  }

  virtual bool is_geometric() const { return _qmin.is_geometric();}

protected:
  QuantityType _qmin;   // the lower cut
  QuantityType _qmax;   // the upper cut
};

/**
 * @brief Area selector
 *
 */
class QuantityArea : public detail::QuantityBase {
public:
  QuantityArea(double _area) : QuantityBase(_area){}
  virtual double operator()(const fastjet::PseudoJet & jet ) const { return jet.area();}
  virtual std::string description() const {return "area";}
};

/**
 * @brief Max constituent pt selector
 *
 * NOTE: This doesn't generalize quite as easily as the other SelectorWorker classes since
 *       we need to extra a single value. However, we're only likely to use a maximum here,
 *       so fine for now.
 */
class QuantityConstituentPtMax : public detail::QuantityBase {
public:
  QuantityConstituentPtMax(double _pt) : QuantityBase(_pt){}
  virtual double operator()(const fastjet::PseudoJet & jet ) const {
    double maxValue = 0;
    for (auto constituent : jet.constituents()) {
      if (constituent.pt() > maxValue) {
        maxValue = constituent.pt();
      }
    }
    return maxValue;
  }
  virtual std::string description() const {return "max constituent pt";}
};

} /* namespace detail */

fastjet::Selector SelectorAreaMin(double areaMin) {
  return fastjet::Selector(new detail::SW_QuantityMin<detail::QuantityArea>(areaMin));
}
fastjet::Selector SelectorAreaMax(double areaMax) {
  return fastjet::Selector(new detail::SW_QuantityMax<detail::QuantityArea>(areaMax));
}
fastjet::Selector SelectorAreaRange(double areaMin, double areaMax) {
  return fastjet::Selector(new detail::SW_QuantityRange<detail::QuantityArea>(areaMin, areaMax));
}

fastjet::Selector SelectorAreaPercentageMin(double jetParameter, double percentageMin) {
  double valueMin = percentageMin / 100. * M_PI * std::pow(jetParameter, 2);
  return fastjet::Selector(new detail::SW_QuantityMin<detail::QuantityArea>(valueMin));
}
fastjet::Selector SelectorAreaPercentageMax(double jetParameter, double percentageMax) {
  double valueMax = percentageMax / 100. * M_PI * std::pow(jetParameter, 2);
  return fastjet::Selector(new detail::SW_QuantityMax<detail::QuantityArea>(valueMax));
}
fastjet::Selector SelectorAreaPercentageRange(double jetParameter, double percentageMin, double percentageMax) {
  double valueMin = percentageMin / 100. * M_PI * std::pow(jetParameter, 2);
  double valueMax = percentageMax / 100. * M_PI * std::pow(jetParameter, 2);
  return fastjet::Selector(new detail::SW_QuantityRange<detail::QuantityArea>(valueMin, valueMax));
}

fastjet::Selector SelectorConstituentPtMax(double constituentPtMax) {
  return fastjet::Selector(new detail::SW_QuantityMax<detail::QuantityConstituentPtMax>(constituentPtMax));
}


const std::map<std::string, fastjet::AreaType> AreaSettings::areaTypes = {
  {"active_area", fastjet::AreaType::active_area},
  {"active_area_explicit_ghosts", fastjet::AreaType::active_area_explicit_ghosts},
  {"passive_area", fastjet::AreaType::passive_area},
};

fastjet::GhostedAreaSpec AreaSettings::ghostedAreaSpec() const {
  fastjet::GhostedAreaSpec ghostedAreaSpec(
    this->rapidityMax,
    this->repeatN,
    this->ghostArea,
    this->gridScatter,
    this->ktScatter,
    this->ktMean
  );
  // Use a fixed seed if it was provided
  if (this->randomSeed.size() > 0) {
    return ghostedAreaSpec.with_fixed_seed(this->randomSeed);
  }
  return ghostedAreaSpec;
}

fastjet::AreaDefinition AreaSettings::areaDefinition() const {
  fastjet::AreaDefinition areaDefinition(this->areaType(), this->ghostedAreaSpec());
  return areaDefinition;
}

std::string AreaSettings::to_string() const
{
  std::string randomSeedValues = "[";
  // Padding formatting based on https://stackoverflow.com/a/3498121/12907985
  const char * padding = "";
  for (const auto v : this->randomSeed) {
    randomSeedValues += (padding + std::to_string(v));
    padding = ", ";
  }
  randomSeedValues += "]";
  return "AreaSettings(area_type='" + this->areaTypeName + "'"
          + ", ghost_area=" + std::to_string(this->ghostArea)
          + ", rapidity_max=" + std::to_string(this->rapidityMax)
          + ", repeat_N_ghosts=" + std::to_string(this->repeatN)
          + ", grid_scatter=" + std::to_string(this->gridScatter)
          + ", kt_scatter=" + std::to_string(this->ktScatter)
          + ", kt_mean=" + std::to_string(this->ktMean)
          + ", random_seed=" + randomSeedValues
          + ")";
}

const std::map<std::string, fastjet::JetAlgorithm> JetFindingSettings::algorithms = {
  {"anti-kt", fastjet::JetAlgorithm::antikt_algorithm},
  {"anti_kt", fastjet::JetAlgorithm::antikt_algorithm},
  {"kt", fastjet::JetAlgorithm::kt_algorithm},
  {"CA", fastjet::JetAlgorithm::cambridge_algorithm},
};
const std::map<std::string, fastjet::RecombinationScheme> JetFindingSettings::recombinationSchemes = {
  {"BIpt2_scheme", fastjet::RecombinationScheme::BIpt2_scheme},
  {"BIpt_scheme", fastjet::RecombinationScheme::BIpt_scheme},
  {"E_scheme", fastjet::RecombinationScheme::E_scheme},
  {"Et2_scheme", fastjet::RecombinationScheme::Et2_scheme},
  {"Et_scheme", fastjet::RecombinationScheme::Et_scheme},
  {"external_scheme", fastjet::RecombinationScheme::external_scheme},
  {"pt2_scheme", fastjet::RecombinationScheme::pt2_scheme},
  {"pt_scheme", fastjet::RecombinationScheme::pt_scheme},
  {"WTA_modp_scheme", fastjet::RecombinationScheme::WTA_modp_scheme},
  {"WTA_pt_scheme", fastjet::RecombinationScheme::WTA_pt_scheme},
};
const std::map<std::string, fastjet::Strategy> JetFindingSettings::strategies = {
  {"Best", fastjet::Strategy::Best},
  {"BestFJ30", fastjet::Strategy::BestFJ30},
  {"plugin_strategy", fastjet::Strategy::plugin_strategy},
  // For convenience
  {"best", fastjet::Strategy::Best},
  {"bestFJ30", fastjet::Strategy::BestFJ30},
};

fastjet::Selector JetFindingSettings::selectorPtEtaNonGhost() const {
  const auto [jetPtMin, jetPtMax] = this->ptRange;
  const auto [jetEtaMin, jetEtaMax] = this->etaRange;

  fastjet::Selector selectJets = !fastjet::SelectorIsPureGhost()
                                * (
                                  fastjet::SelectorPtRange(jetPtMin, jetPtMax)
                                  && fastjet::SelectorEtaRange(jetEtaMin, jetEtaMax)
                                  );
  return selectJets;
}

std::unique_ptr<fastjet::ClusterSequence> JetFindingSettings::create(std::vector<fastjet::PseudoJet> particlePseudoJets) const {
  if (this->areaSettings) {
    return std::make_unique<fastjet::ClusterSequenceArea>(
      particlePseudoJets,
      this->definition(),
      this->areaSettings->areaDefinition()
    );
  }
  return std::make_unique<fastjet::ClusterSequence>(
    particlePseudoJets,
    this->definition()
  );
}

std::string JetFindingSettings::to_string() const
{
  std::string result = "JetFindingSettings(R=" + std::to_string(this->R)
    + ", algorithm='" + this->algorithmName + "'"
    + ", recombination_scheme='" + this->recombinationSchemeName + "'"
    + ", strategy='" + this->strategyName + "'"
    + ", pt=(" + std::to_string(std::get<0>(this->ptRange)) + ", " + std::to_string(std::get<1>(this->ptRange)) + ")"
    + ", eta=(" + std::to_string(std::get<0>(this->etaRange)) + ", " + std::to_string(std::get<1>(this->etaRange)) + ")";
  // Add area if it's defined.
  if (this->areaSettings) {
    result += ", area_settings=" + this->areaSettings->to_string();
  }
  result += ")";
  return result;
}


fastjet::Selector JetMedianBackgroundEstimator::selector() const {
  // Select jets for calculating the background
  // As of March 2022, this includes (ordered from top to bottom):
  // - Remove jets with a constituent with pt > 100
  // - pt and eta selections (eta should be fiducial)
  // - Remove the two hardest jets

  // NOTES:
  // - We want to apply the two hardest removal _after_ the acceptance cuts, so we use "*"
  //   Be aware that the order of the selectors really matters. It applies the **right** most first if we use "*"
  //   This is super important! Otherwise, you'll get unexpected selections!
  // - Including or removing the two hardest can have a big impact on the number of jets that are accepted.
  //   Removing them includes more jets (because the median background is smaller). We remove them because
  //   that's what is done in ALICE.
  // - We skip phi selection since we're looking at charged jets, so we take the full [0, 2pi).
  //   [0, 2pi) is the standard PseudoJet phi range. If you want to restrict the phi range, it should be
  //   applied at the right most with
  //   `* fastjet::SelectorPhiRange(backgroundJetPhiMin, backgroundJetPhiMax)`, or via the combined
  //   EtaPhi selector (see possible tweaks below).
  //
  // Some notes for possible tweaks (not saying that they necessarily should be done):
  // - If one goes back to applying phi cuts, one could use `* fastjet::SelectorRapPhiRange(backgroundJetEtaMin, backgroundJetEtaMax, backgroundJetPhiMin, backgroundJetPhiMax)`
  //   However, if using this combined selector, be careful about the difference between rapidity and eta!

  const auto [jetPtMin, jetPtMax] = this->settings.ptRange;
  const auto [jetEtaMin, jetEtaMax] = this->settings.etaRange;
  fastjet::Selector selRho = !fastjet::SelectorNHardest(this->excludeNHardestJets)
                              * (
                                fastjet::SelectorPtRange(jetPtMin, jetPtMax)
                                && fastjet::SelectorEtaRange(jetEtaMin, jetEtaMax)
                              )
                              * SelectorConstituentPtMax(this->constituentPtMax);
  return selRho;
}

std::unique_ptr<fastjet::BackgroundEstimatorBase> JetMedianBackgroundEstimator::create() const {
  // Validation
  // We have to handle the AreaSettings carefully because the JetFindingSettings don't require
  // the AreaSettings to be provided, but we _have_ to have an AreaSettings to be able to
  // estimate the background.
  const auto areaSettings = this->settings.areaSettings;
  if (!areaSettings) {
    throw std::runtime_error("Must specify area settings when using a background estimator!");
  }
  // Now, create the estimator
  auto backgroundEstimator = std::make_unique<fastjet::JetMedianBackgroundEstimator>(
    this->selector(), this->settings.definition(), areaSettings->areaDefinition()
  );
  // Ensure rho_m is calculated (should be by default, but just to be sure).
  // NOTE: The background estimator should calculate rho_m by default, but it's not used by default
  //       in the standard subtractor, so we explicitly enable it in subtractor (CS is handled separately)
  backgroundEstimator->set_compute_rho_m(this->computeRhoM);

  // According to the fj manual, taking the transverse component of the area four vector
  // provides a more accurate determination of rho. So we default to taking it here.
  // NOTE: ALICE also takes the transverse component.
  backgroundEstimator->set_use_area_4vector(this->useAreaFourVector);

  return backgroundEstimator;
}

std::string JetMedianBackgroundEstimator::to_string() const {
  std::stringstream ss;
  ss << std::boolalpha
     << "JetMedianBackgroundEstimator(compute_rho_M=" << this->computeRhoM
     << ", use_area_four_vector=" << this->useAreaFourVector
     << ", exclude_n_hardest_jets=" << this->excludeNHardestJets
     << ", constituent_pt_max=" << this->constituentPtMax
     << ", jet_finding_settings=" << this->settings.to_string()
     << ")";
  return ss.str();
}
std::string GridMedianBackgroundEstimator::to_string() const {
  std::stringstream ss;
  ss << std::boolalpha
     << "GridMedianBackgroundEstimator(rapidity_max=" << this->rapidityMax
     << ", grid_spacing=" << this->gridSpacing
     << ")";
  return ss.str();
}

std::string to_string(const BackgroundSubtraction_t & subtractionType) {
  const std::map<BackgroundSubtraction_t, std::string> subtractionTypes = {
    {BackgroundSubtraction_t::disabled, "Subtraction disabled"},
    {BackgroundSubtraction_t::rho, "Rho subtraction"},
    {BackgroundSubtraction_t::eventWiseCS, "Event-wise constituent subtraction"},
    {BackgroundSubtraction_t::jetWiseCS, "Jet-wise constituent subtraction"},
  };
  return subtractionTypes.at(subtractionType);
}

std::unique_ptr<fastjet::Transformer> RhoSubtractor::create(std::shared_ptr<fastjet::BackgroundEstimatorBase> backgroundEstimator) const {
  auto subtractor = std::make_unique<fastjet::Subtractor>(backgroundEstimator.get());
  // Use rho_m from the estimator.
  // NOTE: This causes the subtractor to subtract rhom in the four vector subtraction.
  //       This is done to account for nonzero hadron mass
  subtractor->set_use_rho_m(this->useRhoM);
  // Handle negative masses by adjusting the 4-vector to maintain the pt and phi, which leaves
  // the rapidity the same as the unsubtracted jet. The fj manual describes this as "a sensible
  // behavior" for most applications, so good enough for us.
  subtractor->set_safe_mass(this->useSafeMass);

  return subtractor;
}

std::string RhoSubtractor::to_string() const {
  std::stringstream ss;
  ss << std::boolalpha
     << "RhoSubtractor(use_rho_M=" << this->useRhoM
     << ", use_safe_mass=" << this->useSafeMass
     << ")";
  return ss.str();
}

const std::map<std::string, fastjet::contrib::ConstituentSubtractor::Distance> ConstituentSubtractor::distanceTypes = {
  {"deltaR", fastjet::contrib::ConstituentSubtractor::Distance::deltaR},
  {"angle", fastjet::contrib::ConstituentSubtractor::Distance::angle},
  // Alias is for convience
  {"delta_R", fastjet::contrib::ConstituentSubtractor::Distance::deltaR},
};

std::unique_ptr<fastjet::Transformer> ConstituentSubtractor::create(std::shared_ptr<fastjet::BackgroundEstimatorBase> backgroundEstimator) const {
  // Determine derived settings first
  fastjet::contrib::ConstituentSubtractor::Distance distanceType = this->distanceTypes.at(this->distanceMeasure);

  // Next, create the CS object
  auto constituentSubtractor = std::make_unique<fastjet::contrib::ConstituentSubtractor>();

  // Then configure
  constituentSubtractor->set_distance_type(distanceType);
  constituentSubtractor->set_max_distance(this->rMax);
  constituentSubtractor->set_alpha(this->alpha);
  // ALICE doesn't appear to set the ghost area, so we skip it here and use the default of 0.01
  //constituentSubtractor->set_ghost_area(areaSettings.ghostArea);
  // NOTE: Since this is event wise, the max eta should be the track eta, not the fiducial eta
  //constituentSubtractor->set_max_eta(etaMax);
  constituentSubtractor->set_max_eta(this->rapidityMax);
  constituentSubtractor->set_background_estimator(backgroundEstimator.get());
  // Use the same estimator for rho_m (by default, I think it won't be used in CS, but better
  // to provide it in case we change our mind later).
  // NOTE: This needs to be set after setting the background estimator.
  constituentSubtractor->set_common_bge_for_rho_and_rhom();
  // Apparently this is new and now required for event-wise CS.
  // From some tests, constituent subtraction gives some crazy results if it's not called!
  // NOTE: ALICE gets away with skipping this because we have the old call for event-wise
  //       subtraction where we pass the max rapidity. But better if we use the newer version here.
  constituentSubtractor->initialize();

  return constituentSubtractor;
}

std::string ConstituentSubtractor::to_string() const {
  std::stringstream ss;
  ss << std::boolalpha
     << "ConstituentSubtractor(r_max=" << this->rMax
     << ", alpha=" << this->alpha
     << ", rapidity_max=" << this->rapidityMax
     << ", distance_measure=" << this->distanceMeasure
     << ")";
  return ss.str();
}

std::string BackgroundSubtraction::to_string() const {
  std::stringstream ss;
  ss << std::boolalpha
     << "BackgroundSubtraction(type=" << this->type
     << ", estimator=";
  if (this->estimator) {
    ss << this->estimator->to_string();
  } else {
    ss << "nullptr";
  }
  ss << ", subtractor=";
  if (this->subtractor) {
    ss << this->subtractor->to_string();
  } else {
    ss << "nullptr";
  }
  ss << ")";
  return ss.str();
}

std::vector<std::vector<unsigned int>> constituentIndicesFromJets(
  const std::vector<fastjet::PseudoJet> & jets
)
{
  std::vector<std::vector<unsigned int>> indices;
  for (auto jet : jets) {
    std::vector<unsigned int> constituentIndicesInJet;
    for (auto constituent : jet.constituents()) {
      // We want to avoid ghosts, which have index -1
      if (constituent.user_index() != -1) {
        constituentIndicesInJet.push_back(constituent.user_index());
      }
    }
    indices.emplace_back(constituentIndicesInJet);
  }
  return indices;
}

std::vector<unsigned int> updateSubtractedConstituentIndices(
  std::vector<fastjet::PseudoJet> & pseudoJets
)
{
  std::vector<unsigned int> subtractedToUnsubtractedIndices;
  // NOTE: For event-wise CS, we don't need to account for ghost particles here (which have user index -1).
  //       However, if we use jet-wise CS, we should maintain the -1 label for when we use active ghosts
  for (unsigned int i = 0; i < pseudoJets.size(); ++i) {
    subtractedToUnsubtractedIndices.push_back(pseudoJets[i].user_index());
    // The indexing may be different due to the subtraction (for example, if a particle is entirely subtracted
    // away). Since we want the index to be continuous (up to ghost particle), we reassign it here to be certain.
    // If it's a ghost particle (with user_index == -1), we keep that assignment so we don't lose that identification.
    pseudoJets[i].set_user_index((pseudoJets[i].user_index() != -1) ? i : -1);
  }

  return subtractedToUnsubtractedIndices;
}

namespace JetSubstructure
{

/**
 * Subjets
 */

/**
 * Default constructor
 */
Subjets::Subjets():
  fSplittingNodeIndex{},
  fPartOfIterativeSplitting{},
  fConstituentIndices{}
{
  // Nothing more to be done.
}

/**
 * Copy constructor
 */
Subjets::Subjets(const Subjets& other)
 : fSplittingNodeIndex{other.fSplittingNodeIndex},
  fPartOfIterativeSplitting{other.fPartOfIterativeSplitting},
  fConstituentIndices{other.fConstituentIndices}
{
  // Nothing more to be done.
}

/**
 * Assignment operator. Note that we pass by _value_, so a copy is created and it is
 * fine to swap the values with the created object!
 */
Subjets& Subjets::operator=(Subjets other)
{
  swap(*this, other);
  return *this;
}

bool Subjets::Clear()
{
  fSplittingNodeIndex.clear();
  fPartOfIterativeSplitting.clear();
  fConstituentIndices.clear();
  return true;
}

std::tuple<unsigned short, bool, const std::vector<unsigned short>> Subjets::GetSubjet(int i) const
{
  return std::make_tuple(fSplittingNodeIndex.at(i), fPartOfIterativeSplitting.at(i), fConstituentIndices.at(i));
}

void Subjets::AddSubjet(const unsigned short splittingNodeIndex, const bool partOfIterativeSplitting, const std::vector<unsigned short> & constituentIndices)
{
  fSplittingNodeIndex.emplace_back(splittingNodeIndex);
  // NOTE: emplace_back isn't supported for std::vector<bool> until c++14.
  fPartOfIterativeSplitting.push_back(partOfIterativeSplitting);
  // Originally, we stored the constituent indices and their jagged indices separately to try to coax ROOT
  // into storing the nested vectors in a columnar format. However, even with that design, uproot can't
  // recreate the nested jagged array without a slow python loop. So we just store the indices directly
  // and wait for uproot 4. See: https://stackoverflow.com/q/60250877/12907985
  fConstituentIndices.emplace_back(constituentIndices);
}

/**
 * Prints information about the task.
 *
 * @return std::string containing information about the task.
 */
std::string Subjets::toString() const
{
  std::stringstream tempSS;
  tempSS << std::boolalpha;
  tempSS << "Subjets:\n";
  for (std::size_t i = 0; i < fSplittingNodeIndex.size(); i++)
  {
    tempSS << "#" << (i + 1) << ": Splitting Node: " << fSplittingNodeIndex.at(i)
        << ", part of iterative splitting = " << fPartOfIterativeSplitting.at(i)
        << ", number of jet constituents = " << fConstituentIndices.at(i).size() << "\n";
  }
  return tempSS.str();
}

/**
 * Print task information on an output stream using the string representation provided by
 * Subjets::toString. Used by operator<<
 * @param in output stream stream
 * @return reference to the output stream
 */
std::ostream& Subjets::Print(std::ostream& in) const
{
  in << toString();
  return in;
}

/**
 * Jet splittings
 */

/**
 * Default constructor.
 */
JetSplittings::JetSplittings():
  fKt{},
  fDeltaR{},
  fZ{},
  fParentIndex{}
{
  // Nothing more to be done.
}

/**
 * Copy constructor
 */
JetSplittings::JetSplittings(const JetSplittings& other)
 : fKt{other.fKt},
  fDeltaR{other.fDeltaR},
  fZ{other.fZ},
  fParentIndex{other.fParentIndex}
{
  // Nothing more to be done.
}

/**
 * Assignment operator. Note that we pass by _value_, so a copy is created and it is
 * fine to swap the values with the created object!
 */
JetSplittings& JetSplittings::operator=(JetSplittings other)
{
  swap(*this, other);
  return *this;
}

bool JetSplittings::Clear()
{
  fKt.clear();
  fDeltaR.clear();
  fZ.clear();
  fParentIndex.clear();
  return true;
}

void JetSplittings::AddSplitting(float kt, float deltaR, float z, short i)
{
  fKt.emplace_back(kt);
  fDeltaR.emplace_back(deltaR);
  fZ.emplace_back(z);
  fParentIndex.emplace_back(i);
}

std::tuple<float, float, float, short> JetSplittings::GetSplitting(int i) const
{
  return std::make_tuple(fKt.at(i), fDeltaR.at(i), fZ.at(i), fParentIndex.at(i));
}

/**
 * Prints information about the task.
 *
 * @return std::string containing information about the task.
 */
std::string JetSplittings::toString() const
{
  std::stringstream tempSS;
  tempSS << std::boolalpha;
  tempSS << "Jet splittings:\n";
  for (std::size_t i = 0; i < fKt.size(); i++)
  {
    tempSS << "#" << (i + 1) << ": kT = " << fKt.at(i)
        << ", deltaR = " << fDeltaR.at(i) << ", z = " << fZ.at(i)
        << ", parent = " << fParentIndex.at(i) << "\n";
  }
  return tempSS.str();
}

/**
 * Print task information on an output stream using the string representation provided by
 * JetSplittings::toString. Used by operator<<
 * @param in output stream stream
 * @return reference to the output stream
 */
std::ostream& JetSplittings::Print(std::ostream& in) const
{
  in << toString();
  return in;
}

/**
 * Jet substructure splittings container.
 */

/**
 * Default constructor.
 */
JetSubstructureSplittings::JetSubstructureSplittings():
  fJetSplittings{},
  fSubjets{}
{
  // Nothing more to be done.
}

/**
 * Copy constructor
 */
JetSubstructureSplittings::JetSubstructureSplittings(
 const JetSubstructureSplittings& other)
 : fJetSplittings{other.fJetSplittings},
  fSubjets{other.fSubjets}
{
}

/**
 * Assignment operator. Note that we pass by _value_, so a copy is created and it is
 * fine to swap the values with the created object!
 */
JetSubstructureSplittings& JetSubstructureSplittings::operator=(
 JetSubstructureSplittings other)
{
  swap(*this, other);
  return *this;
}

bool JetSubstructureSplittings::Clear()
{
  fJetSplittings.Clear();
  fSubjets.Clear();
  return true;
}

/**
 * Add a jet splitting to the object.
 *
 * @param[in] kt Kt of the splitting.
 * @param[in] deltaR Delta R between the subjets.
 * @param[in] z Momentum sharing between the subjets.
 */
void JetSubstructureSplittings::AddSplitting(float kt, float deltaR, float z, short parentIndex)
{
  fJetSplittings.AddSplitting(kt, deltaR, z, parentIndex);
}

/**
 * Add a subjet to the object.
 *
 * @param[in] part Constituent to be added.
 */
void JetSubstructureSplittings::AddSubjet(const unsigned short splittingNodeIndex, const bool partOfIterativeSplitting,
                     const std::vector<unsigned short>& constituentIndices)
{
  return fSubjets.AddSubjet(splittingNodeIndex, partOfIterativeSplitting, constituentIndices);
}

std::tuple<float, float, float, short> JetSubstructureSplittings::GetSplitting(int i) const
{
  return fJetSplittings.GetSplitting(i);
}

std::tuple<unsigned short, bool, const std::vector<unsigned short>> JetSubstructureSplittings::GetSubjet(int i) const
{
  return fSubjets.GetSubjet(i);
}

/**
 * Prints information about the task.
 *
 * @return std::string containing information about the task.
 */
std::string JetSubstructureSplittings::toString() const
{
  std::stringstream tempSS;
  tempSS << std::boolalpha;
  tempSS << "Splitting information: ";
  tempSS << fSubjets;
  tempSS << fJetSplittings;
  return tempSS.str();
}

/**
 * Print task information on an output stream using the string representation provided by
 * JetSubstructureSplittings::toString. Used by operator<<
 * @param in output stream stream
 * @return reference to the output stream
 */
std::ostream& JetSubstructureSplittings::Print(std::ostream& in) const
{
  in << toString();
  return in;
}

} /* namespace JetSubstructure */

void ExtractJetSplittings(
  JetSubstructure::JetSubstructureSplittings & jetSplittings,
  fastjet::PseudoJet & inputJet,
  int splittingNodeIndex,
  bool followingIterativeSplitting,
  const bool storeRecursiveSplittings
)
{
    fastjet::PseudoJet j1;
    fastjet::PseudoJet j2;
    if (inputJet.has_parents(j1, j2) == false) {
        // No parents, so we're done - just return.
        return;
    }
    std::cout << "j1 (" << j1.user_index() << "): " << j1.pt() << ", j2(" << j2.user_index() << "): " << j2.pt() << "\n";

    // j1 should always be the harder of the two subjets.
    if (j1.perp() < j2.perp()) {
        std::swap(j1, j2);
    }

    // We have a splitting. Record the properties.
    double z = j2.perp() / (j2.perp() + j1.perp());
    double delta_R = j1.delta_R(j2);
    double xkt = j2.perp() * std::sin(delta_R);
    std::cout << "delta_R=" << delta_R << ", kt=" << xkt << ", z=" << z << "\n";
    // Add the splitting node.
    jetSplittings.AddSplitting(xkt, delta_R, z, splittingNodeIndex);
    // Determine which splitting parent the subjets will point to (ie. the one that
    // we just stored). It's stored at the end of the splittings array. (which we offset
    // by -1 to stay within the array).
    splittingNodeIndex = jetSplittings.GetNumberOfSplittings() - 1;
    // Store the subjets
    std::vector<unsigned short> j1ConstituentIndices, j2ConstituentIndices;
    for (auto constituent: j1.constituents()) {
        j1ConstituentIndices.emplace_back(constituent.user_index());
    }
    for (auto constituent: j2.constituents()) {
        j2ConstituentIndices.emplace_back(constituent.user_index());
    }
    jetSplittings.AddSubjet(splittingNodeIndex, followingIterativeSplitting, j1ConstituentIndices);
    jetSplittings.AddSubjet(splittingNodeIndex, false, j2ConstituentIndices);

    // Recurse as necessary to get the rest of the splittings.
    ExtractJetSplittings(jetSplittings, j1, splittingNodeIndex, followingIterativeSplitting, storeRecursiveSplittings);
    if (storeRecursiveSplittings == true) {
        ExtractJetSplittings(jetSplittings, j2, splittingNodeIndex, false, storeRecursiveSplittings);
    }
}

} // namespace mammoth

/**
  * operator<< for printing objects.
  */

std::ostream& operator<<(std::ostream& in, const mammoth::JetSubstructure::Subjets& myTask)
{
  std::ostream& result = myTask.Print(in);
  return result;
}
std::ostream& operator<<(std::ostream& in, const mammoth::JetSubstructure::JetSplittings& myTask)
{
  std::ostream& result = myTask.Print(in);
  return result;
}
std::ostream& operator<<(std::ostream& in, const mammoth::JetSubstructure::JetSubstructureSplittings& myTask)
{
  std::ostream& result = myTask.Print(in);
  return result;
}

std::ostream& operator<<(std::ostream& in, const mammoth::AreaSettings & c) {
  in << c.to_string();
  return in;
}

std::ostream& operator<<(std::ostream& in, const mammoth::JetFindingSettings & c) {
  in << c.to_string();
  return in;
}

std::ostream& operator<<(std::ostream& in, const mammoth::BackgroundEstimator & c) {
  in << c.to_string();
  return in;
}
std::ostream& operator<<(std::ostream& in, const mammoth::JetMedianBackgroundEstimator & c) {
  in << c.to_string();
  return in;
}
std::ostream& operator<<(std::ostream& in, const mammoth::GridMedianBackgroundEstimator & c) {
  in << c.to_string();
  return in;
}

std::ostream& operator<<(std::ostream& in, const mammoth::BackgroundSubtractor & c) {
  in << c.to_string();
  return in;
}
std::ostream& operator<<(std::ostream& in, const mammoth::BackgroundSubtraction_t& c) {
  in << mammoth::to_string(c);
  return in;
}
std::ostream& operator<<(std::ostream& in, const mammoth::RhoSubtractor& c) {
  in << c.to_string();
  return in;
}
std::ostream& operator<<(std::ostream& in, const mammoth::ConstituentSubtractor& c) {
  in << c.to_string();
  return in;
}

std::ostream& operator<<(std::ostream& in, const mammoth::BackgroundSubtraction& c) {
  in << c.to_string();
  return in;
}

/**
 * Swap function for jet substructure tasks. Created using guide described here: https://stackoverflow.com/a/3279550.
 */
void swap(mammoth::JetSubstructure::Subjets& first,
     mammoth::JetSubstructure::Subjets& second) {
  using std::swap;

  // Same ordering as in the constructors (for consistency)
  swap(first.fSplittingNodeIndex, second.fSplittingNodeIndex);
  swap(first.fPartOfIterativeSplitting, second.fPartOfIterativeSplitting);
  swap(first.fConstituentIndices, second.fConstituentIndices);
}
void swap(mammoth::JetSubstructure::JetSplittings& first,
     mammoth::JetSubstructure::JetSplittings& second) {
  using std::swap;

  // Same ordering as in the constructors (for consistency)
  swap(first.fKt, second.fKt);
  swap(first.fDeltaR, second.fDeltaR);
  swap(first.fZ, second.fZ);
  swap(first.fParentIndex, second.fParentIndex);
}
void swap(mammoth::JetSubstructure::JetSubstructureSplittings& first,
     mammoth::JetSubstructure::JetSubstructureSplittings& second) {
  using std::swap;

  // Same ordering as in the constructors (for consistency)
  swap(first.fJetSplittings, second.fJetSplittings);
  swap(first.fSubjets, second.fSubjets);
}