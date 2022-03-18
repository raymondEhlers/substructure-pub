#if !(defined(__CINT__) || defined(__CLING__)) || defined(__ACLIC__)
#include <iostream>
#include <memory>
#include <vector>

#include <TTree.h>
#include <TChain.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TH2D.h>
#include <TLegend.h>
#include <TLine.h>
#include <TNtuple.h>
#include <TPostScript.h>
#include <TProfile.h>
#include <TROOT.h>
#include <TRandom.h>
#include <TString.h>
#include <TStyle.h>
#include <TSystem.h>

#include <RooUnfoldBayes.h>
#include <RooUnfoldResponse.h>
#endif

//==============================================================================
// Gaussian smearing, systematic translation, and variable inefficiency
//==============================================================================

TH2D* CorrelationHistShape(const TMatrixD& cov, const char* name, const char* title, Int_t na, Int_t nb, Int_t kbin)
{
  TH2D* h = new TH2D(name, title, nb, 0, nb, nb, 0, nb);

  for (int l = 0; l < nb; l++) {
    for (int n = 0; n < nb; n++) {
      int index1 = kbin + na * l;
      int index2 = kbin + na * n;
      Double_t Vv = cov(index1, index1) * cov(index2, index2);
      if (Vv > 0.0)
        h->SetBinContent(l + 1, n + 1, cov(index1, index2) / sqrt(Vv));
    }
  }
  return h;
}

TH2D* CorrelationHistPt(const TMatrixD& cov, const char* name, const char* title, Int_t na, Int_t nb, Int_t kbin)
{
  TH2D* h = new TH2D(name, title, na, 0, na, na, 0, na);

  for (int l = 0; l < na; l++) {
    for (int n = 0; n < na; n++) {
      int index1 = l + na * kbin;
      int index2 = n + na * kbin;
      Double_t Vv = cov(index1, index1) * cov(index2, index2);
      if (Vv > 0.0)
        h->SetBinContent(l + 1, n + 1, cov(index1, index2) / sqrt(Vv));
    }
  }
  return h;
}

TH2D* CorrelationHist(const TMatrixD& cov, const char* name, const char* title, Double_t lo, Double_t hi, Double_t lon,
           Double_t hin)
{
  Int_t nb = cov.GetNrows();
  Int_t na = cov.GetNcols();
  std::cout << nb << " " << na << "\n";
  TH2D* h = new TH2D(name, title, nb, 0, nb, na, 0, na);
  h->SetAxisRange(-1.0, 1.0, "Z");
  for (int i = 0; i < na; i++)
    for (int j = 0; j < nb; j++) {
      Double_t Viijj = cov(i, i) * cov(j, j);
      if (Viijj > 0.0)
        h->SetBinContent(i + 1, j + 1, cov(i, j) / sqrt(Viijj));
    }
  return h;
}

void Normalize2D(TH2* h)
{
  Int_t nbinsYtmp = h->GetNbinsY();
  const Int_t nbinsY = nbinsYtmp;
  Double_t norm[nbinsY];
  for (Int_t biny = 1; biny <= nbinsY; biny++) {
    norm[biny - 1] = 0;
    for (Int_t binx = 1; binx <= h->GetNbinsX(); binx++) {
      norm[biny - 1] += h->GetBinContent(binx, biny);
    }
  }

  for (Int_t biny = 1; biny <= nbinsY; biny++) {
    for (Int_t binx = 1; binx <= h->GetNbinsX(); binx++) {
      if (norm[biny - 1] == 0)
        continue;
      else {
        h->SetBinContent(binx, biny, h->GetBinContent(binx, biny) / norm[biny - 1]);
        h->SetBinError(binx, biny, h->GetBinError(binx, biny) / norm[biny - 1]);
      }
    }
  }
}

//==============================================================================
// Example Unfolding
//==============================================================================

/**
 * Perform the actual unfolding.
 *
 * We separate this functionality so that we can run the unfolding with different input spectra for the same
 * response, such as the when we used the hybrid as the smeared input as a trivial closure.
 *
 * @param[in] response Roounfold response.
 * @param[in] h2true True spectra. Only used for binning, so retrieved as a const.
 * @param[in] inputSpectra Input spectra to be unfolded (for example, data). Spectra could be 2D: pt + kt (or Rg, etc...)
 * @param[in] errorTreatment Roounfold error treatment.
 * @param[in] fout ROOT output file. It's never called explicitly, but I like to pass it because we're implicitly writing to it.
 * @param[in] tag Tag to be prepended to all histograms generated in the unfolding. Default: "".
 * @param[in] nIter Number of iterations. Default: 10.
 */
void Unfold(RooUnfoldResponse & response, TH2D & inputSpectra, RooUnfold::ErrorTreatment errorTreatment, TFile * fout, std::string tag = "", const int nIter = 10)
{
  std::cout << "\n=======================================================\n";
  std::cout << "Unfolding for tag \"" << tag << "\".\n";
  // Determine the tag. If we have a non-empty tag, we append it to all of the histograms.
  if (tag != "") {
    tag += "_";
  }

  for (int jar = 1; jar < nIter; jar++) {
    Int_t iter = jar;
    std::cout << "iteration" << iter << "\n";
    std::cout << "==============Unfold h1=====================\n";

    // Setup the response for unfolding, and then unfold.
    RooUnfoldBayes unfold(&response, &inputSpectra, iter);
    TH2D* hunf = dynamic_cast<TH2D*>(unfold.Hreco(errorTreatment));

    // FOLD BACK
    TH1* hfold = response.ApplyToTruth(hunf, "");

    TH2D* htempUnf = dynamic_cast<TH2D*>(hunf->Clone("htempUnf"));
    htempUnf->SetName(TString::Format("%sBayesian_Unfoldediter%d", tag.c_str(), iter));

    TH2D* htempFold = dynamic_cast<TH2D*>(hfold->Clone("htempFold"));
    htempFold->SetName(TString::Format("%sBayesian_Foldediter%d", tag.c_str(), iter));

    htempUnf->Write();
    htempFold->Write();
  }

  std::cout << "=======================================================\n";
}



/**
 * Determine pt hard scale factor of the currently open file.
 *
 * @param[in] f Current file.
 *
 * @returns Pt hard scale factor.
 */
double GetScaleFactor(TFile * f)
{
  // Retrieve the embedding helper to extract the cross section and ntrials.
  //auto task = dynamic_cast<TList*>(f->Get("AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_Raw_EventSub_Incl"));
  auto task = dynamic_cast<TList*>(f->Get("AliAnalysisTaskEmcalEmbeddingHelper_histos"));

  auto hcross = dynamic_cast<TProfile*>(task->FindObject("fHistXsection"));
  auto htrials = dynamic_cast<TH1D*>(task->FindObject("fHistTrials"));

  //std::cout << "hcross=" << hcross << ", htrials=" << htrials << "\n";

  int ptHardBin = 0;
  for (Int_t i = 1; i <= htrials->GetXaxis()->GetNbins(); i++) {
      if (htrials->GetBinContent(i) != 0) {
          ptHardBin = i;
      }
  }
  // Code adapted from Leticia.
  // The values don't seem to be correct...
  /*double scaleFactor = hcross->Integral(ptHardBin, ptHardBin) / htrials->Integral(ptHardBin, ptHardBin);
  // Needs a -1 to offset for the bin counting.
  std::cout << f->GetName() << ": ptHardBin=" << (ptHardBin - 1) << ", scaleFactor=" << scaleFactor;*/

  // These values seem more reasonable...
  double crossSection = hcross->GetBinContent(ptHardBin) * hcross->GetEntries();
  double nTrials = htrials->GetBinContent(ptHardBin);
  double scaleFactor = crossSection / nTrials;
  std::cout << f->GetName() << ": ptHardBin=" << (ptHardBin - 1) << ", scaleFactor=" << scaleFactor << "\n";

  return scaleFactor;
}

TH2D* getReweightedRatioForClosure(const std::string embeddedDatasetName, const std::string dataDatasetName, const std::string groomingMethod)
{
    auto embedded = TFile::Open(("output/embedPythia/RDF/" + embeddedDatasetName + "_" + groomingMethod + "_prefixes_hybrid_true_det_level_closure.root").c_str(), "READ");
    auto PbPb = TFile::Open(("output/PbPb/RDF/" + dataDatasetName + "_" + groomingMethod + "_prefixes_data_closure.root").c_str(), "READ");

    auto hEmbedded = dynamic_cast<TH2D*>(embedded->Get((groomingMethod + "_hybrid_kt_jet_pt").c_str()));
    auto hPbPb = dynamic_cast<TH2D*>(PbPb->Get((groomingMethod + "_data_kt_jet_pt").c_str()));

    auto hRatio = dynamic_cast<TH2D*>(hEmbedded->Clone("hRatio"));
    hRatio->Divide(hPbPb);
    hRatio->SetDirectory(0);

    embedded->Close();
    PbPb->Close();

    return hRatio;
}

int findReweightingBin(double value, const std::vector<double> & bins)
{
    auto bin = std::distance(bins.begin(), std::upper_bound(bins.begin(), bins.end(), value));
    if (bin == 0) {
        bin++;
    }
    if (bin >= bins.size()) {
        --bin;
    }

    return bin;
}

/**
 * Substructure unfolding variable.
 */
enum UnfoldingType_t {
  kt = 0,
  zg = 1,
  rg = 2
};

/**
 * Unfolding for a specified substructure variable. Most settings must be changed inside of the function...
 *
 */
void RunUnfolding(const std::string groomingMethod)
{
#ifdef __CINT__
  gSystem->Load("libRooUnfold");
#endif
  std::cout
   << "==================================== pick up the response matrix for background==========================\n";
  ///////////////////parameter setting

  // NOTE: These need to map to the branch names.
  std::map<UnfoldingType_t, std::string> unfoldingTypeNames = {
    std::make_pair(UnfoldingType_t::kt, "kt"),
    std::make_pair(UnfoldingType_t::zg, "z"),
    std::make_pair(UnfoldingType_t::rg, "delta_R")
  };

  // Setup
  // Unfolding type
  const UnfoldingType_t unfoldingType = UnfoldingType_t::kt;
  // If true, use pure matches
  const bool usePureMatches = true;
  // Unfolding settings
  RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;
  // Untagged bin below
  const bool untaggedBinBelowRange = true;
  // Determine the base output filename and directory
  std::string outputFilename = "unfolding";
  std::string outputDir = "output/PbPb/unfolding";
  // And an optional tag at the end...
  std::string tag = "closureSplitMC";

  //***************************************************

  // Define binning and tree branch names
  std::vector<double> smearedJetPtBins;
  std::vector<double> trueJetPtBins;
  std::vector<double> smearedSplittingVariableBins;
  std::vector<double> trueSplittingVariableBins;
  // Untagged bin properties
  double minSmearedSplittingVariable = 0.;
  double maxSmearedSplittingVariable = 100.;
  double smearedUntaggedBinValue = -0.025;
  std::string untaggedBinDescription = "";
  // Set to true if the untagged bin values are set explicitly. Must be done for all four values!
  bool explicitlySetUntaggedBinValues = false;

  double printFactor = 1;

  switch (unfoldingType) {
    case UnfoldingType_t::kt:
      smearedJetPtBins = {30, 40, 50, 60, 80, 100, 120};
      trueJetPtBins = {0, 30, 40, 60, 80, 100, 120, 160};
      smearedSplittingVariableBins = {1, 2, 3, 4, 5, 7, 10, 15};
      // NOTE: (-0.05, 0) is the untagged bin.
      trueSplittingVariableBins = {-0.05, 0, 2, 3, 4, 5, 7, 10, 15, 100};
      break;
    case UnfoldingType_t::zg:
      // This is for z_cut > 0.2
      smearedJetPtBins = {40, 50, 60, 70, 80, 100, 120};
      trueJetPtBins = {0, 20, 40, 60, 80, 100, 120, 140, 160};
      smearedSplittingVariableBins = {-0.05, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5};
      trueSplittingVariableBins = {0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5};
      break;
    case UnfoldingType_t::rg:
      smearedJetPtBins = {40, 50, 60, 70, 90, 120};
      trueJetPtBins = {0, 20, 40, 60, 80, 100, 120, 140, 160};
      smearedSplittingVariableBins = {-0.05, 0, 0.02, 0.04, 0.06, 0.1, 0.2, 0.35};
      trueSplittingVariableBins = {-0.05, 0, 0.02, 0.04, 0.06, 0.1, 0.2, 0.35, 0.6};
      printFactor = 1000;
      break;
    default:
      throw std::runtime_error("Must specify an unfolding type.");
      break;
  }

  // Should be set to true if they're explicitly set in the configuration above.
  if (explicitlySetUntaggedBinValues == false) {
    auto lastBin = smearedSplittingVariableBins.size() - 1;
    if (untaggedBinBelowRange) {
      minSmearedSplittingVariable = smearedSplittingVariableBins[1];
      maxSmearedSplittingVariable = smearedSplittingVariableBins[lastBin];
      smearedUntaggedBinValue = (smearedSplittingVariableBins[1] - smearedSplittingVariableBins[0]) / 2 + smearedSplittingVariableBins[0];
      untaggedBinDescription = std::to_string(static_cast<int>(smearedSplittingVariableBins[0] * printFactor)) + "_" + static_cast<int>(smearedSplittingVariableBins[1] * printFactor);
    }
    else {
      smearedUntaggedBinValue = (smearedSplittingVariableBins[lastBin] - smearedSplittingVariableBins[lastBin - 1]) / 2 + smearedSplittingVariableBins[lastBin - 1];
      minSmearedSplittingVariable = smearedSplittingVariableBins[0];
      maxSmearedSplittingVariable = smearedSplittingVariableBins[lastBin - 1];
      untaggedBinDescription = std::to_string(static_cast<int>(smearedSplittingVariableBins[lastBin - 1] * printFactor)) + "_" + static_cast<int>(smearedSplittingVariableBins[lastBin] * printFactor);
    }
  }

  // Determine the final configuration (which is based on the binning).
  std::string substructureVariableName = unfoldingTypeNames.at(unfoldingType);
  // Final determination and setup for the output directory and filename.
  gSystem->mkdir(outputDir.c_str(), true);
  // Determine the filename based on the options.
  outputFilename += "_" + substructureVariableName;
  outputFilename += "_grooming_method_";
  outputFilename += groomingMethod;
  // Binning information.
  // Used std::string and std::to_string at times to coerce the type to a string so we can keep adding.
  // kt
  outputFilename += "_smeared_" + substructureVariableName + "_" + static_cast<int>(minSmearedSplittingVariable * printFactor) + "_" + static_cast<int>(maxSmearedSplittingVariable * printFactor);
  // Untagged bin information.
  outputFilename += "_untagged_" + substructureVariableName + "_" + untaggedBinDescription;
  // pt. (use std::to_string to coerce the type to a string so we can keep adding).
  outputFilename += "_smeared_jetPt_" + std::to_string(static_cast<int>(smearedJetPtBins[0])) + "_" + static_cast<int>(smearedJetPtBins[smearedJetPtBins.size() - 1]);

  // Options
  if (usePureMatches == true) {
    outputFilename += "_pureMatches";
  }
  if (tag != "") {
    outputFilename += "_" + tag;
  }
  outputFilename = outputDir + "/" + outputFilename + ".root";

  // Print the configuration
  std::cout << "\n*********** Settings ***********\n" << std::boolalpha
       << "Unfolding for: " << substructureVariableName << "\n"
       << "Grooming method: " << groomingMethod << "\n"
       << "Untagged bin description: " << untaggedBinDescription << "\n"
       << "Use untagged bin below bins: " << untaggedBinBelowRange << "\n"
       << "Use pure matches in the response: " << usePureMatches << "\n"
       << "output filename: " << outputFilename << "\n"
       << "********************************\n\n";

  // Configuration (not totally clear if this actually does anything for this script...)
  ROOT::EnableImplicitMT();

  // the raw correlation (ie. data)
  TH2D h2raw("raw", "raw", smearedSplittingVariableBins.size() - 1, smearedSplittingVariableBins.data(), smearedJetPtBins.size() - 1, smearedJetPtBins.data());
  // detector measure level (ie. hybrid)
  TH2D h2smeared("smeared", "smeared", smearedSplittingVariableBins.size() - 1, smearedSplittingVariableBins.data(), smearedJetPtBins.size() - 1, smearedJetPtBins.data());
  // detector measure level no cuts (ie. hybrid, but no cuts).
  // NOTE: Strictly speaking, the y axis binning is at the hybrid level, but we want a wider range. So we use the trueJetPtBins.
  TH2D h2smearednocuts("smearednocuts", "smearednocuts", smearedSplittingVariableBins.size() - 1, smearedSplittingVariableBins.data(), trueJetPtBins.size() - 1, trueJetPtBins.data());
  // true correlations with measured cuts
  TH2D h2true("true", "true", trueSplittingVariableBins.size() - 1, trueSplittingVariableBins.data(), trueJetPtBins.size() - 1, trueJetPtBins.data());
  // full true correlation (without cuts)
  TH2D h2fulleff("truef", "truef", trueSplittingVariableBins.size() - 1, trueSplittingVariableBins.data(), trueJetPtBins.size() - 1, trueJetPtBins.data());
  // Correlation between the splitting variables at true and hybrid (with cuts).
  TH2D h2SplittingVariable("h2SplittingVariable", "h2SplittingVariable", smearedSplittingVariableBins.size() - 1, smearedSplittingVariableBins.data(), trueSplittingVariableBins.size() - 1, trueSplittingVariableBins.data());
  // PseudoData
  TH2D h2PseudoData("pseudoData", "pseudoData", smearedSplittingVariableBins.size() - 1, smearedSplittingVariableBins.data(), smearedJetPtBins.size() - 1, smearedJetPtBins.data());
  // PseudoTrue
  TH2D h2PseudoTrue("pseudoTrue", "pseudoTrue", trueSplittingVariableBins.size() - 1, trueSplittingVariableBins.data(), trueJetPtBins.size() - 1, trueJetPtBins.data());

  h2smeared.Sumw2();
  h2true.Sumw2();
  h2raw.Sumw2();
  h2fulleff.Sumw2();
  h2PseudoData.Sumw2();
  h2PseudoTrue.Sumw2();

  // Read the data and create the raw data hist.
  // First, setup the input data.
  //TChain dataChain("AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_Data_ConstSub_Incl");
  /*TChain dataChain("tree");
  //dataChain.Add("trains/PbPb/5987/*.root");
  dataChain.Add("trains/PbPb/5863/skim/*.root");
  // Print out for logs (and to mirror Leticia).
  //dataChain.ls();
  TTreeReader dataReader(&dataChain);
  //dataReader.Print();

  // Determines the type of data that we use. Usually, this is going to be "data" for raw data.
  std::string dataPrefix = "data";

  TTreeReaderValue<float> dataJetPt(dataReader, ("jet_pt_" + dataPrefix).c_str());
  TTreeReaderValue<float> dataSubstructureVariable(dataReader, (groomingMethod + "_" + dataPrefix + "_" + substructureVariableName).c_str());
  while (dataReader.Next()) {
    // Jet pt cut.
    if (*dataJetPt < smearedJetPtBins[0] || *dataJetPt > smearedJetPtBins[smearedJetPtBins.size() - 1]) {
      continue;
    }
    // Substructure variable cut.
    double dataSubstructureVariableValue = *dataSubstructureVariable;
    if (dataSubstructureVariableValue < 0) {
      // Assign to the untagged bin.
      dataSubstructureVariableValue = smearedUntaggedBinValue;
    }
    else {
      if (dataSubstructureVariableValue < minSmearedSplittingVariable || dataSubstructureVariableValue > maxSmearedSplittingVariable) {
        continue;
      }
    }
    h2raw.Fill(dataSubstructureVariableValue, *dataJetPt);
  }*/

  // Setup response tree.
  //TChain embeddedChain("AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl");
  TChain embeddedChain("tree");
  // We are specific on the filenames to avoid the friend trees.
  // It appears that it can only handle one * per call. So we have to enumerate each train.
  embeddedChain.Add("trains/embedPythia/5966/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5967/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5968/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5969/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5970/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5971/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5972/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5973/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5974/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5975/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5976/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5977/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5978/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5979/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5980/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5981/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5982/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5983/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5984/skim/*.root");
  embeddedChain.Add("trains/embedPythia/5985/skim/*.root");
  //embeddedChain.ls();
  //embeddedChain.Print();

  // Define the reader and process.
  //std::string truePrefix = "matched";
  std::string truePrefix = "true";
  //std::string hybridPrefix = "data";
  std::string hybridPrefix = "hybrid";
  std::string detLevelPrefix = "det_level";
  TTreeReader mcReader(&embeddedChain);
  TTreeReaderValue<double> scaleFactor(mcReader, "scale_factor");
  TTreeReaderValue<double> hybridJetPt(mcReader, (hybridPrefix + "_jet_pt").c_str());
  TTreeReaderValue<double> hybridSubstructureVariable(mcReader, (groomingMethod + "_" + hybridPrefix + "_" + substructureVariableName).c_str());
  TTreeReaderValue<double> trueJetPt(mcReader, (truePrefix + "_jet_pt").c_str());
  TTreeReaderValue<double> trueSubstructureVariable(mcReader, (groomingMethod + "_" + truePrefix + "_" + substructureVariableName).c_str());
  //TTreeReaderValue<long long> matchingLeading(mcReader, (groomingMethod + "_hybrid_det_level_matching_leading").c_str());
  TTreeReaderValue<long long> matchingLeading(mcReader, (groomingMethod + "_hybrid_det_level_matching_leading").c_str());
  TTreeReaderValue<long long> matchingSubleading(mcReader, (groomingMethod + "_hybrid_det_level_matching_subleading").c_str());
  // For the double counting cut.
  TTreeReaderValue<double> hybridUnsubLeadingTrackPt(mcReader, (hybridPrefix + "_leading_track_pt").c_str());
  TTreeReaderValue<double> detLevelLeadingTrackPt(mcReader, (detLevelPrefix + "_leading_track_pt").c_str());

  // Setup for the response
  RooUnfoldResponse response;
  RooUnfoldResponse responsenotrunc;
  response.Setup(&h2smeared, &h2true);
  responsenotrunc.Setup(&h2smearednocuts, &h2fulleff);
  //TH2D* hRatio = getReweightedRatioForClosure("LHC19f4_embedded_into_LHC18qr_5966_5985", "LHC18qr_5863", groomingMethod);

  TRandom3 random(0);

  int treeNumber = -1;
  //double scaleFactor = 0;
  while (mcReader.Next()) {
    // Check if the file changed.
    if (treeNumber < embeddedChain.GetTreeNumber()) {
      // File changed. Update the scale factor.
      //auto f = embeddedChain.GetFile();
      //scaleFactor = GetScaleFactor(f);
      // Update the tree number so we hold onto the scale factor until the next time we need to update.
      treeNumber = embeddedChain.GetTreeNumber();
    }
    // Ensure that we are in the right true pt and substructure variable range.
    if (*trueJetPt > trueJetPtBins[trueJetPtBins.size() - 1]) {
      continue;
    }
    if (*trueSubstructureVariable > trueSplittingVariableBins[trueSplittingVariableBins.size() - 1]) {
      continue;
    }
    // Double counting cut
    if (*hybridUnsubLeadingTrackPt > *detLevelLeadingTrackPt) {
      continue;
    }

    // Full efficiency hists.
    h2fulleff.Fill(*trueSubstructureVariable, *trueJetPt, *scaleFactor);
    h2smearednocuts.Fill(*hybridSubstructureVariable, *hybridJetPt, *scaleFactor);
    responsenotrunc.Fill(*hybridSubstructureVariable, *hybridJetPt, *trueSubstructureVariable, *trueJetPt, *scaleFactor);

    // Now start making cuts on the hybrid level.
    if (*hybridJetPt < smearedJetPtBins[0] || *hybridJetPt > smearedJetPtBins[smearedJetPtBins.size() - 1]) {
      continue;
    }
    // Also cut on hybrid substructure variable.
    double hybridSubstructureVariableValue = *hybridSubstructureVariable;
    if (hybridSubstructureVariableValue < 0) {
      // Assign to the untagged bin.
      hybridSubstructureVariableValue = smearedUntaggedBinValue;
    }
    else {
      if (hybridSubstructureVariableValue < minSmearedSplittingVariable || hybridSubstructureVariableValue > maxSmearedSplittingVariable) {
        continue;
      }
    }
    // Matching cuts: Requiring a pure match.
    if (usePureMatches && !(
                ((std::abs(*matchingLeading - 1) < 0.001) && (std::abs(*matchingSubleading - 1) < 0.001)) || (std::abs(hybridSubstructureVariableValue - smearedUntaggedBinValue) < 0.001)
            )) {
      continue;
    }
    h2smeared.Fill(hybridSubstructureVariableValue, *hybridJetPt, *scaleFactor);
    h2true.Fill(*trueSubstructureVariable, *trueJetPt, *scaleFactor);
    // So we can look at the substructure variable correlation.
    h2SplittingVariable.Fill(hybridSubstructureVariableValue, *trueSubstructureVariable, *scaleFactor);

    // Reweight
    // NOTE: We intentionally look at the true values even though it's binned at detector level.
    // We need to handle the binning carefully, so we use a dedicated function.
    /*int ktBin = findReweightingBin(*trueSubstructureVariable, smearedSplittingVariableBins);
    int jetPtBin = findReweightingBin(*trueJetPt, smearedJetPtBins);
    double reweightFactor = hRatio->GetBinContent(ktBin, jetPtBin);*/

    //std::cout << "ktBin: " << ktBin << ", trueSubstructureVariable: " << *trueSubstructureVariable << "\n";
    //std::cout << "jetPtBin: " << jetPtBin << ", trueJetPt: " << *trueJetPt << "\n";

    // Factor of 5 more stats in kt = 7-10
    // Factor of 3 more stats in kt = 10-15
    if (random.Rndm() < 0.75) {
        response.Fill(hybridSubstructureVariableValue, *hybridJetPt, *trueSubstructureVariable, *trueJetPt, *scaleFactor);
    }
    else {
        h2PseudoData.Fill(hybridSubstructureVariableValue, *hybridJetPt, *scaleFactor);
        h2PseudoTrue.Fill(*trueSubstructureVariable, *trueJetPt, *scaleFactor);
    }
  }

  TH1D* htrueptd = dynamic_cast<TH1D*>(h2fulleff.ProjectionX("trueptd", 1, -1));
  TH1D* htruept = dynamic_cast<TH1D*>(h2fulleff.ProjectionY("truept", 1, -1));

  //////////efficiencies done////////////////////////////////////
  TH1D* effok = dynamic_cast<TH1D*>(h2true.ProjectionX("effok", 2, 2));
  TH1D* effok1 = dynamic_cast<TH1D*>(h2fulleff.ProjectionX("effok2", 2, 2));
  effok->Divide(effok1);
  effok->SetName("correff20-40");

  TH1D* effok3 = dynamic_cast<TH1D*>(h2true.ProjectionX("effok3", 3, 3));
  TH1D* effok4 = dynamic_cast<TH1D*>(h2fulleff.ProjectionX("effok4", 3, 3));
  effok3->Divide(effok4);
  effok3->SetName("correff40-60");

  TH1D* effok5 = dynamic_cast<TH1D*>(h2true.ProjectionX("effok5", 4, 4));
  TH1D* effok6 = dynamic_cast<TH1D*>(h2fulleff.ProjectionX("effok6", 4, 4));
  effok5->Divide(effok6);
  effok5->SetName("correff60-80");

  TH1D* effok7 = dynamic_cast<TH1D*>(h2true.ProjectionX("effok7", 5, 6));
  TH1D* effok8 = dynamic_cast<TH1D*>(h2fulleff.ProjectionX("effok8", 5, 6));
  effok7->Divide(effok8);
  effok7->SetName("correff80-120");

  TFile* fout = new TFile(outputFilename.c_str(), "RECREATE");
  fout->cd();
  effok->Write();
  effok3->Write();
  effok5->Write();
  effok7->Write();
  h2raw.SetName("raw");
  h2raw.Write();
  h2smeared.SetName("smeared");
  h2smeared.Write();
  htrueptd->Write();
  h2true.SetName("true");
  h2true.Write();
  h2fulleff.Write();
  h2SplittingVariable.Write();
  h2PseudoData.Write();
  h2PseudoTrue.Write();

  // Unfold the standard spectra.
  int nIter = 20;
  // NOTE: This API isn't consistent with the standard unfolding...
  Unfold(response, h2PseudoData, errorTreatment, fout, "", nIter);
  // Unfold with the hybrid as the smeared input for a trivial closure.
  //Unfold(response, errorTreatment, fout, "hybridAsInput", nIter);

  // Cleanup
  fout->Close();
}

#ifndef __CINT__
int RooSimplePbPbClosureSplitMC()
{
  // Grooming method
  //const std::string groomingMethod = "leading_kt_z_cut_02";
  const std::string groomingMethod = "dynamical_kt";
  RunUnfolding(groomingMethod);
  return 0;
} // Main program when run stand-alone
#endif
