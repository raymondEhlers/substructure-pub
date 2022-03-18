
#include "mammoth/aliceFastSim.hpp"

#include <cmath>
#include <iostream>

// Anonymous namespace to contain most of the implementation details
namespace {

/**
 * @brief Calculate Gaussian for given parameters
 *
 * Basically a part of `TMath::Gaus`.
 *
 * @param x Position to evaluate the Guassian.
 * @param mean Gaussian mean
 * @param sigma Gaussian width
 * @param normalized Whether to normalize the gaussian to a probability. Default: False.
 * @return double Gaussian evaluated at the given parameters.
 */
double gaussian(double x, double mean, double sigma, bool normalized = false)
{
    double val = std::exp(-0.5 * std::pow(x - mean, 2) / std::pow(sigma, 2));
    if (normalized) {
        val = val / (std::sqrt(2 * M_PI) * sigma);
    }
    return val;
}

// ******************************
// *** LHC11h parameters      ***
// ******************************
// LHC11h efficiency parameters for good runs
// 0-10% centrality
const double LHC11hParam_0_10[17] = {
  0.971679, 0.0767571, 1.13355,  -0.0274484, 0.856652,  0.00536795, 3.90795e-05, 1.06889, 0.011007,
  0.447046, -0.146626, 0.919777, 0.192601,   -0.268515, 1.00243,    0.00620849,  0.709477
};
// 10-30% centrality
const double LHC11hParam_10_30[17] = {
  0.97929,  0.0776039, 1.12213,  -0.0300645, 0.844722,  0.0134788, -0.0012333, 1.07955, 0.0116835,
  0.456608, -0.132743, 0.930964, 0.174175,   -0.267154, 0.993118,  0.00574892, 0.765256
};
// 30-50% centrality
const double LHC11hParam_30_50[17] = {
  0.997696, 0.0816769,  1.14341,  -0.0353734, 0.752151,  0.0744259, -0.0102926, 1.01561, 0.00713274,
  0.57203,  -0.0640248, 0.947747, 0.102007,   -0.194698, 0.999164,  0.00568476, 0.7237
};
// 50-90% centrality
const double LHC11hParam_50_90[17] = {
  0.97041, 0.0813559,  1.12151,  -0.0368797, 0.709327, 0.0701501, -0.00784043, 1.06276, 0.00676173,
  0.53607, -0.0703117, 0.982534, 0.0947881,  -0.18073, 1.03229,   0.00580109,  0.737801
};

// ******************************
// *** LHC15o parameters      ***
// ******************************
// For pt parameters, first 5 are low pt, next 5 are high pt
// For eta parameters, first 6 are eta =< -0.04 (eta left in Eliane's def), next 6 are => -0.04 (eta right
// in Eliane's def). The last parameter normalizes the eta values such that their maximum is 1. This was apparently
// part of their definition, but was implementing by normalizing a TF1 afterwards. My implementation approach here
// is more useful when not using a TF1.
// 0-10% centrality
const double LHC15oParam_0_10_pt[10] = { 0.8350,         0.0621,         0.0986, 0.2000,
                                    1.0124,         0.7568,         0.0277, -0.0034,
                                    0.1506 * 0.001, -0.0023 * 0.001 };
const double LHC15oParam_0_10_eta[13] = { 1.0086,  0.0074, 0.2404, -0.1230, -0.0107,
                                     0.0427,  0.8579, 0.0088, 0.4697,  0.0772,
                                     -0.0352, 0.0645, 0.7716 };
// 10-30% centrality
const double LHC15oParam_10_30_pt[10] = {
  0.8213, 0.0527, 0.0867, 0.1970, 1.1518, 0.7469, 0.0300, -0.0038, 0.1704 * 0.001, -0.0026 * 0.001
};
const double LHC15oParam_10_30_eta[13] = { 0.9726,  0.0066, 0.2543, -0.1167, -0.0113,
                                     0.0400,  0.8729, 0.0122, 0.4537,  0.0965,
                                     -0.0328, 0.0623, 0.7658 };
// 30-50% centrality
const double LHC15oParam_30_50_pt[10] = {
  0.8381, 0.0648, 0.1052, 0.1478, 1.0320, 0.7628, 0.0263, -0.0032, 0.1443 * 0.001, -0.0023 * 0.001
};
const double LHC15oParam_30_50_eta[13] = { 0.9076,  0.0065, 0.3216, -0.1130, -0.0107,
                                     0.0456,  0.8521, 0.0073, 0.4764,  0.0668,
                                     -0.0363, 0.0668, 0.7748 };
// 50-90% centrality
const double LHC15oParam_50_90_pt[10] = {
  0.8437, 0.0668, 0.1083, 0.2000, 0.9741, 0.7677, 0.0255, -0.0030, 0.1260 * 0.001, -0.0019 * 0.001
};
const double LHC15oParam_50_90_eta[13] = { 1.1259,  0.0105, 0.1961, -0.1330, -0.0103,
                                     0.0440,  0.8421, 0.0066, 0.5061,  0.0580,
                                     -0.0379, 0.0651, 0.7786 };


/**
 * Calculate the tracking efficiency for LHC11h - run 1 PbPb at 2.76 TeV. See Joel's jet-hadron analysis note.
 *
 * @param[in] trackPt Track pt
 * @param[in] trackEta Track eta
 * @param[in] eventActivity Centrality bin of the current event.
 * @param[in] taskName Name of the task which is calling this function (for logging purposes).
 * @returns The efficiency of measuring the given single track.
 */
double LHC11hTrackingEfficiency(const double trackPt, const double trackEta, const alice::fastsim::EventActivity_t eventActivity, const std::string & taskName)
{
  // Setup
  double etaAxis = 0;
  double ptAxis = 0;
  double efficiency = 1;

  // Assumes that the centrality bins follow (as defined in AliAnalysisTaskEmcal)
  // 0 = 0-10%
  // 1 = 10-30%
  // 2 = 30-50%
  // 3 = 50-90%
  switch (eventActivity) {
    case alice::fastsim::EventActivity_t::k0010:
      // Parameter values for GOOD TPC (LHC11h) runs (0-10%):
      ptAxis =
       (trackPt < 2.9) * (LHC11hParam_0_10[0] * exp(-pow(LHC11hParam_0_10[1] / trackPt, LHC11hParam_0_10[2])) +
                 LHC11hParam_0_10[3] * trackPt) +
       (trackPt >= 2.9) *
        (LHC11hParam_0_10[4] + LHC11hParam_0_10[5] * trackPt + LHC11hParam_0_10[6] * trackPt * trackPt);
      etaAxis =
       (trackEta < 0.0) *
        (LHC11hParam_0_10[7] * exp(-pow(LHC11hParam_0_10[8] / std::abs(trackEta + 0.91), LHC11hParam_0_10[9])) +
         LHC11hParam_0_10[10] * trackEta) +
       (trackEta >= 0.0 && trackEta <= 0.4) *
        (LHC11hParam_0_10[11] + LHC11hParam_0_10[12] * trackEta + LHC11hParam_0_10[13] * trackEta * trackEta) +
       (trackEta > 0.4) * (LHC11hParam_0_10[14] *
                 exp(-pow(LHC11hParam_0_10[15] / std::abs(-trackEta + 0.91), LHC11hParam_0_10[16])));
      efficiency = ptAxis * etaAxis;
      break;

    case alice::fastsim::EventActivity_t::k1030:
      // Parameter values for GOOD TPC (LHC11h) runs (10-30%):
      ptAxis = (trackPt < 2.9) *
            (LHC11hParam_10_30[0] * exp(-pow(LHC11hParam_10_30[1] / trackPt, LHC11hParam_10_30[2])) +
            LHC11hParam_10_30[3] * trackPt) +
           (trackPt >= 2.9) * (LHC11hParam_10_30[4] + LHC11hParam_10_30[5] * trackPt +
                     LHC11hParam_10_30[6] * trackPt * trackPt);
      etaAxis =
       (trackEta < 0.0) * (LHC11hParam_10_30[7] *
                  exp(-pow(LHC11hParam_10_30[8] / std::abs(trackEta + 0.91), LHC11hParam_10_30[9])) +
                 LHC11hParam_10_30[10] * trackEta) +
       (trackEta >= 0.0 && trackEta <= 0.4) * (LHC11hParam_10_30[11] + LHC11hParam_10_30[12] * trackEta +
                           LHC11hParam_10_30[13] * trackEta * trackEta) +
       (trackEta > 0.4) * (LHC11hParam_10_30[14] *
                 exp(-pow(LHC11hParam_10_30[15] / std::abs(-trackEta + 0.91), LHC11hParam_10_30[16])));
      efficiency = ptAxis * etaAxis;
      break;

    case alice::fastsim::EventActivity_t::k3050:
      // Parameter values for GOOD TPC (LHC11h) runs (30-50%):
      ptAxis = (trackPt < 2.9) *
            (LHC11hParam_30_50[0] * exp(-pow(LHC11hParam_30_50[1] / trackPt, LHC11hParam_30_50[2])) +
            LHC11hParam_30_50[3] * trackPt) +
           (trackPt >= 2.9) * (LHC11hParam_30_50[4] + LHC11hParam_30_50[5] * trackPt +
                     LHC11hParam_30_50[6] * trackPt * trackPt);
      etaAxis =
       (trackEta < 0.0) * (LHC11hParam_30_50[7] *
                  exp(-pow(LHC11hParam_30_50[8] / std::abs(trackEta + 0.91), LHC11hParam_30_50[9])) +
                 LHC11hParam_30_50[10] * trackEta) +
       (trackEta >= 0.0 && trackEta <= 0.4) * (LHC11hParam_30_50[11] + LHC11hParam_30_50[12] * trackEta +
                           LHC11hParam_30_50[13] * trackEta * trackEta) +
       (trackEta > 0.4) * (LHC11hParam_30_50[14] *
                 exp(-pow(LHC11hParam_30_50[15] / std::abs(-trackEta + 0.91), LHC11hParam_30_50[16])));
      efficiency = ptAxis * etaAxis;
      break;

    case alice::fastsim::EventActivity_t::k5090:
      // Parameter values for GOOD TPC (LHC11h) runs (50-90%):
      ptAxis = (trackPt < 2.9) *
            (LHC11hParam_50_90[0] * exp(-pow(LHC11hParam_50_90[1] / trackPt, LHC11hParam_50_90[2])) +
            LHC11hParam_50_90[3] * trackPt) +
           (trackPt >= 2.9) * (LHC11hParam_50_90[4] + LHC11hParam_50_90[5] * trackPt +
                     LHC11hParam_50_90[6] * trackPt * trackPt);
      etaAxis =
       (trackEta < 0.0) * (LHC11hParam_50_90[7] *
                  exp(-pow(LHC11hParam_50_90[8] / std::abs(trackEta + 0.91), LHC11hParam_50_90[9])) +
                 LHC11hParam_50_90[10] * trackEta) +
       (trackEta >= 0.0 && trackEta <= 0.4) * (LHC11hParam_50_90[11] + LHC11hParam_50_90[12] * trackEta +
                           LHC11hParam_50_90[13] * trackEta * trackEta) +
       (trackEta > 0.4) * (LHC11hParam_50_90[14] *
                 exp(-pow(LHC11hParam_50_90[15] / std::abs(-trackEta + 0.91), LHC11hParam_50_90[16])));
      efficiency = ptAxis * etaAxis;
      break;

    default:
      std::cerr << taskName << ": " << "Invalid centrality for determining LHC11h tracking efficiency.\n";
      throw alice::fastsim::Error_t::kInvalidCentrality;
  }

  return efficiency;
}

/**
 * Determine the pt efficiency axis for low pt tracks in LHC15o. Implementation function.
 *
 * @param[in] trackEta Track eta.
 * @param[in] params Parameters for use with the function.
 * @param[in] index Index where it should begin accessing the parameters.
 * @returns The efficiency associated with the eta parameterization.
 */
double LHC15oLowPtEfficiencyImpl(const double trackPt, const double params[10], const int index)
{
  return (params[index + 0] + -1.0 * params[index + 1] / trackPt) +
      params[index + 2] * gaussian(trackPt, params[index + 3], params[index + 4]);
}

/**
 * Determine the pt efficiency axis for high pt tracks in LHC15o. Implementation function.
 *
 * @param[in] trackEta Track eta.
 * @param[in] params Parameters for use with the function.
 * @param[in] index Index where it should begin accessing the parameters.
 * @returns The efficiency associated with the eta parameterization.
 */
double LHC15oHighPtEfficiencyImpl(const double trackPt, const double params[10], const int index)
{
  return params[index + 0] + params[index + 1] * trackPt + params[index + 2] * std::pow(trackPt, 2) +
      params[index + 3] * std::pow(trackPt, 3) + params[index + 4] * std::pow(trackPt, 4);
}

/**
 * Determine the pt efficiency axis for LHC15o. This is the main interface
 * for getting the efficiency.
 *
 * @param[in] trackEta Track eta.
 * @param[in] params Parameters for use with the function.
 * @returns The efficiency associated with the eta parameterization.
 */
double LHC15oPtEfficiency(const double trackPt, const double params[10])
{
  return ((trackPt <= 3.5) * LHC15oLowPtEfficiencyImpl(trackPt, params, 0) +
      (trackPt > 3.5) * LHC15oHighPtEfficiencyImpl(trackPt, params, 5));
}

/**
 * Determine the eta efficiency axis for LHC15o. Implementation function.
 *
 * @param[in] trackEta Track eta.
 * @param[in] params Parameters for use with the function.
 * @param[in] index Index where it should begin accessing the parameters.
 * @returns The efficiency associated with the eta parameterization.
 */
double LHC15oEtaEfficiencyImpl(const double trackEta, const double params[13],
                               const int index)
{
  // We need to multiply the track eta by -1 if we are looking at eta > 0 (which corresponds to
  // the second set of parameters, such that the index is greater than 0).
  int sign = index > 0 ? -1 : 1;
  return (params[index + 0] *
       std::exp(-1.0 * std::pow(params[index + 1] / std::abs(sign * trackEta + 0.91), params[index + 2])) +
      params[index + 3] * trackEta + params[index + 4] * gaussian(trackEta, -0.04, params[index + 5])) /
      params[12];
}

/**
 * Determine the eta efficiency axis for LHC15o.
 *
 * @param[in] trackEta Track eta.
 * @param[in] params Parameters for use with the function.
 * @returns The efficiency associated with the eta parameterization.
 */
double LHC15oEtaEfficiency(const double trackEta, const double params[13])
{
  // Just modify the arguments - the function is the same.
  return ((trackEta <= -0.04) * LHC15oEtaEfficiencyImpl(trackEta, params, 0) +
      (trackEta > -0.04) * LHC15oEtaEfficiencyImpl(trackEta, params, 6));
}

/**
 * Calculate the track efficiency for LHC15o - PbPb at 5.02 TeV. See the gamma-hadron analysis (from Eliane via Michael).
 *
 * @param[in] trackPt Track pt
 * @param[in] trackEta Track eta
 * @param[in] eventActivity Centrality bin of the current event.
 * @param[in] taskName Name of the task which is calling this function (for logging purposes).
 * @returns The efficiency of measuring the given single track.
 */
double LHC15oTrackingEfficiency(const double trackPt, const double trackEta, const alice::fastsim::EventActivity_t eventActivity, const std::string & taskName)
{
  // We use the switch to determine the parameters needed to call the functions.
  // Assumes that the centrality bins follow (as defined in AliAnalysisTaskEmcal)
  // 0 = 0-10%
  // 1 = 10-30%
  // 2 = 30-50%
  // 3 = 50-90%
  const double* ptParams = nullptr;
  const double* etaParams = nullptr;
  switch (eventActivity) {
    case alice::fastsim::EventActivity_t::k0010:
      ptParams = LHC15oParam_0_10_pt;
      etaParams = LHC15oParam_0_10_eta;
      break;
    case alice::fastsim::EventActivity_t::k1030:
      ptParams = LHC15oParam_10_30_pt;
      etaParams = LHC15oParam_10_30_eta;
      break;
    case alice::fastsim::EventActivity_t::k3050:
      ptParams = LHC15oParam_30_50_pt;
      etaParams = LHC15oParam_30_50_eta;
      break;
    case alice::fastsim::EventActivity_t::k5090:
      ptParams = LHC15oParam_50_90_pt;
      etaParams = LHC15oParam_50_90_eta;
      break;
    default:
      std::cerr << taskName << ": " << "Invalid centrality for determining LHC15o tracking efficiency.\n";
      throw alice::fastsim::Error_t::kInvalidCentrality;
  }

  // Calculate the efficiency using the parameters.
  double ptAxis = LHC15oPtEfficiency(trackPt, ptParams);
  double etaAxis = LHC15oEtaEfficiency(trackEta, etaParams);
  double efficiency = ptAxis * etaAxis;

  return efficiency;
}

/**
 * Calculate the track efficiency for LHC11a - pp at 2.76 TeV. Calculated using LHC12f1a. See the jet-hadron
 * analysis note for more details.
 *
 * @param[in] trackPt Track pt
 * @param[in] trackEta Track eta
 * @param[in] eventActivity Centrality bin of the current event.
 * @param[in] taskName Name of the task which is calling this function (for logging purposes).
 * @returns The efficiency of measuring the given single track.
 */
double LHC11aTrackingEfficiency(const double trackPt, const double trackEta, const alice::fastsim::EventActivity_t eventActivity, const std::string & taskName)
{
  // Validation
  if (eventActivity != alice::fastsim::EventActivity_t::kInclusive) {
      std::cerr << taskName << ": " << "Passed event activity other than inclusive to pp. Passed: " << static_cast<int>(eventActivity) << "\n";
      throw alice::fastsim::Error_t::kInvalidCentrality;
  }

  // Pt axis
  // If the trackPt > 6 GeV, then all we need is this coefficient
  double coefficient = 0.898052; // p6
  if (trackPt < 6) {
    coefficient = (1 + -0.442232 * trackPt              // p0
            + 0.501831 * std::pow(trackPt, 2)   // p1
            + -0.252024 * std::pow(trackPt, 3)  // p2
            + 0.062964 * std::pow(trackPt, 4)   // p3
            + -0.007681 * std::pow(trackPt, 5)  // p4
            + 0.000365 * std::pow(trackPt, 6)); // p5
  }

  // Eta axis
  double efficiency = coefficient * (1 + 0.402825 * std::abs(trackEta)            // p7
                    + -2.213152 * std::pow(trackEta, 2)          // p8
                    + 4.311098 * std::abs(std::pow(trackEta, 3)) // p9
                    + -2.778200 * std::pow(trackEta, 4));        // p10

  return efficiency;
}

}


namespace alice {
namespace fastsim {

double trackingEfficiencyByPeriod(
 const double trackPt, const double trackEta, const EventActivity_t eventActivity,
 const Period_t period)
{
    // NOTE: We can't pass this because it breaks the vectorization method.
    //       We could remove it, but it's easier just to leave it in place at this point.
    const std::string taskName = "mammoth-fastsim";

    // Efficiency is determined entirely based on the given efficiency period.
    double efficiency = 1;
    switch (period)
    {
    case Period_t::kDisabled:
        efficiency = 1;
        break;
    case Period_t::kLHC11h:
        efficiency = LHC11hTrackingEfficiency(trackPt, trackEta, eventActivity, taskName);
        break;
    case Period_t::kLHC15o:
        efficiency = LHC15oTrackingEfficiency(trackPt, trackEta, eventActivity, taskName);
        break;
    case Period_t::kLHC11a:
        efficiency = LHC11aTrackingEfficiency(trackPt, trackEta, eventActivity, taskName);
        break;
    case Period_t::kLHC18qr:
    case Period_t::kpA:
    case Period_t::kpp:
        // Intentionally fall through for LHC18, pA
        {
            // Need to find the period name for the error message
            // Put it in a scope for clarity
            std::string periodName = "";
            for (auto &&[k, v] : periodNameMap)
            {
                if (v == period)
                {
                    periodName = k;
                }
            }
            std::cerr << taskName << ": "
                      << "Tracking efficiency for period " << periodName << " is not yet implemented.\n";
        }
        throw Error_t::kPeriodNotImplemented;
        break;
    default:
        // No efficiency period option selected. Notify the user.
        std::cerr << taskName << ": "
                  << "No single track efficiency setting selected! Please select one.\n";
        throw Error_t::kInvalidPeriod;
    }

    return efficiency;
}

} /* namespace fastsim */
} /* namespace alice */
