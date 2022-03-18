#if !(defined(__CINT__) || defined(__CLING__)) || defined(__ACLIC__)
#include <RooUnfoldBayes.h>
#include <RooUnfoldResponse.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TFile.h>
#include <TH1D.h>
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
#include <TTree.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include <iostream>
#include <memory>
#include <vector>
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
 * @param[in] inputSpectra Input spectra to be unfolded (for example, data). Spectra could be 2D: pt + kt (or Rg,
 * etc...)
 * @param[in] errorTreatment Roounfold error treatment.
 * @param[in] fout ROOT output file. It's never called explicitly, but I like to pass it because we're implicitly
 * writing to it.
 * @param[in] tag Tag to be prepended to all histograms generated in the unfolding. Default: "".
 * @param[in] nIter Number of iterations. Default: 10.
 */
void Unfold(RooUnfoldResponse& response, const TH1D& h1true, TH1D& inputSpectra,
      RooUnfold::ErrorTreatment errorTreatment, TFile* fout, std::string tag = "", const int nIter = 10)
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
    TH1D* hunf = dynamic_cast<TH1D*>(unfold.Hreco(errorTreatment));

    // FOLD BACK
    TH1* hfold = response.ApplyToTruth(hunf, "");

    TH1D* htempUnf = dynamic_cast<TH1D*>(hunf->Clone("htempUnf"));
    htempUnf->SetName(TString::Format("%sBayesian_Unfoldediter%d", tag.c_str(), iter));

    TH1D* htempFold = dynamic_cast<TH1D*>(hfold->Clone("htempFold"));
    htempFold->SetName(TString::Format("%sBayesian_Foldediter%d", tag.c_str(), iter));

    htempUnf->Write();
    htempFold->Write();

    /// HERE I GET THE COVARIANCE MATRIX/////

    // TODO: Look into this for 1D...
    /*if (iter == 8) {
      TMatrixD covmat = unfold.Ereco((RooUnfold::ErrorTreatment)RooUnfold::kCovariance);
      for (Int_t k = 0; k < h1true.GetNbinsX(); k++) {
        TH1D* hCorr = dynamic_cast<TH1D*>(
         CorrelationHistShape(covmat, TString::Format("%scorr%d", tag.c_str(), k), "Covariance matrix",
                    h1true.GetNbinsX(), h1true.GetNbinsY(), k));
        TH1D* covshape = dynamic_cast<TH1D*>(hCorr->Clone("covshape"));
        covshape->SetName(TString::Format("%spearsonmatrix_iter%d_binshape%d", tag.c_str(), iter, k));
        covshape->SetDrawOption("colz");
        covshape->Write();
      }

      for (Int_t k = 0; k < h1true.GetNbinsY(); k++) {
        TH1D* hCorr = dynamic_cast<TH1D*>(
         CorrelationHistPt(covmat, TString::Format("%scorr%dpt", tag.c_str(), k), "Covariance matrix",
                  h1true.GetNbinsX(), h1true.GetNbinsY(), k));
        TH1D* covpt = dynamic_cast<TH1D*>(hCorr->Clone("covpt"));
        covpt->SetName(TString::Format("%spearsonmatrix_iter%d_binpt%d", tag.c_str(), iter, k));
        covpt->SetDrawOption("colz");
        covpt->Write();
      }
    }*/
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
double GetScaleFactor(TFile* f)
{
  // Retrieve the embedding helper to extract the cross section and ntrials.
  // auto task =
  // dynamic_cast<TList*>(f->Get("AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_Raw_EventSub_Incl"));
  auto task = dynamic_cast<TList*>(f->Get("AliAnalysisTaskEmcalEmbeddingHelper_histos"));

  auto hcross = dynamic_cast<TProfile*>(task->FindObject("fHistXsection"));
  auto htrials = dynamic_cast<TH1D*>(task->FindObject("fHistTrials"));

  // std::cout << "hcross=" << hcross << ", htrials=" << htrials << "\n";

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

/**
 * Unfolding for a specified substructure variable. Most settings must be changed inside of the function...
 *
 */
void RunUnfolding()
{
#ifdef __CINT__
  gSystem->Load("libRooUnfold");
#endif
  std::cout
   << "==================================== pick up the response matrix for background==========================\n";
  ///////////////////parameter setting

  // Setup
  // Unfolding settings
  RooUnfold::ErrorTreatment errorTreatment = RooUnfold::kCovariance;
  // Determine the base output filename and directory
  std::string outputFilename = "unfolding";
  std::string outputDir = "output/PbPb/unfolding";
  // And an optional tag at the end...
  std::string tag = "";

  //***************************************************

  // Define binning and tree branch names
  std::vector<double> smearedJetPtBins;
  std::vector<double> trueJetPtBins;

  smearedJetPtBins = { 40, 50, 60, 80, 100, 120 };
  trueJetPtBins = { 0, 40, 60, 80, 100, 120, 160 };

  // Final determination and setup for the output directory and filename.
  gSystem->mkdir(outputDir.c_str(), true);
  // Binning information.
  // Used std::string and std::to_string at times to coerce the type to a string so we can keep adding.
  // pt. (use std::to_string to coerce the type to a string so we can keep adding).
  outputFilename += "_smeared_jetPt_" + std::to_string(static_cast<int>(smearedJetPtBins[0])) + "_" +
           static_cast<int>(smearedJetPtBins[smearedJetPtBins.size() - 1]);

  // Options
  if (tag != "") {
    outputFilename += "_" + tag;
  }
  outputFilename = outputDir + "/" + outputFilename + ".root";

  // Print the configuration
  std::cout << "\n*********** Settings ***********\n"
       << std::boolalpha << "output filename: " << outputFilename << "\n"
       << "********************************\n\n";

  // Configuration (not totally clear if this actually does anything for this script...)
  ROOT::EnableImplicitMT();

  // the raw correlation (ie. data)
  TH1D h1raw("r", "raw", smearedJetPtBins.size() - 1, smearedJetPtBins.data());
  // detector measure level (ie. hybrid)
  TH1D h1smeared("smeared", "smeared", smearedJetPtBins.size() - 1, smearedJetPtBins.data());
  // detector measure level no cuts (ie. hybrid, but no cuts).
  // NOTE: Strictly speaking, the y axis binning is at the hybrid level, but we want a wider range. So we use the
  // trueJetPtBins.
  TH1D h1smearednocuts("smearednocuts", "smearednocuts", trueJetPtBins.size() - 1, trueJetPtBins.data());
  // true correlations with measured cuts
  TH1D h1true("true", "true", trueJetPtBins.size() - 1, trueJetPtBins.data());
  // full true correlation (without cuts)
  TH1D h1fulleff("truef", "truef", trueJetPtBins.size() - 1, trueJetPtBins.data());
  // Correlation between the splitting variables at true and hybrid (with cuts).
  TH2D h2JetPt("h2JetPt", "h2JetPt", smearedJetPtBins.size() - 1, smearedJetPtBins.data(), trueJetPtBins.size() - 1,
         trueJetPtBins.data());

  TH1D* effnum = dynamic_cast<TH1D*>(h1fulleff.Clone("effnum"));
  TH1D* effdenom = dynamic_cast<TH1D*>(h1fulleff.Clone("effdenom"));

  effnum->Sumw2();
  effdenom->Sumw2();
  h1smeared.Sumw2();
  h1true.Sumw2();
  h1raw.Sumw2();
  h1fulleff.Sumw2();

  // Read the data and create the raw data hist.
  // First, setup the input data.
  TChain dataChain("tree");
  dataChain.Add("trains/PbPb/5863/skim/*.root");
  // Print out for logs.
  // dataChain.ls();
  TTreeReader dataReader(&dataChain);
  // dataReader.Print();

  // Determines the type of data that we use. Usually, this is going to be "data" for raw data.
  std::string dataPrefix = "data";

  TTreeReaderValue<double> dataJetPt(dataReader, (dataPrefix + "_jet_pt").c_str());
  while (dataReader.Next()) {
    // Jet pt cut.
    if (*dataJetPt < smearedJetPtBins[0] || *dataJetPt > smearedJetPtBins[smearedJetPtBins.size() - 1]) {
      continue;
    }
    h1raw.Fill(*dataJetPt);
  }

  // Setup response tree.
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
  // embeddedChain.ls();
  // embeddedChain.Print();

  // Define the reader and process.
  std::string truePrefix = "true";
  std::string hybridPrefix = "hybrid";
  std::string detLevelPrefix = "det_level";
  TTreeReader mcReader(&embeddedChain);
  TTreeReaderValue<double> scaleFactor(mcReader, "scale_factor");
  TTreeReaderValue<double> hybridJetPt(mcReader, (hybridPrefix + "_jet_pt").c_str());
  TTreeReaderValue<double> trueJetPt(mcReader, (truePrefix + "_jet_pt").c_str());
  // For the double counting cut.
  TTreeReaderValue<double> hybridUnsubLeadingTrackPt(mcReader, (hybridPrefix + "_leading_track_pt").c_str());
  TTreeReaderValue<double> detLevelLeadingTrackPt(mcReader, (detLevelPrefix + "_leading_track_pt").c_str());

  // Setup for the response
  RooUnfoldResponse response;
  RooUnfoldResponse responsenotrunc;
  response.Setup(&h1smeared, &h1true);
  responsenotrunc.Setup(&h1smearednocuts, &h1fulleff);

  int treeNumber = -1;
  // double scaleFactor = 0;
  while (mcReader.Next()) {
    // Check if the file changed.
    if (treeNumber < embeddedChain.GetTreeNumber()) {
      // File changed. Update the scale factor.
      // auto f = embeddedChain.GetFile();
      // scaleFactor = GetScaleFactor(f);
      // Update the tree number so we hold onto the scale factor until the next time we need to update.
      treeNumber = embeddedChain.GetTreeNumber();
    }
    // Ensure that we are in the right true jet pt range.
    if (*trueJetPt > trueJetPtBins[trueJetPtBins.size() - 1]) {
      continue;
    }
    // Double counting cut
    if (*hybridUnsubLeadingTrackPt > *detLevelLeadingTrackPt) {
      continue;
    }

    // Full efficiency hists.
    h1fulleff.Fill(*trueJetPt, *scaleFactor);
    h1smearednocuts.Fill(*hybridJetPt, *scaleFactor);
    responsenotrunc.Fill(*hybridJetPt, *trueJetPt, *scaleFactor);

    // Now start making cuts on the hybrid level.
    if (*hybridJetPt < smearedJetPtBins[0] || *hybridJetPt > smearedJetPtBins[smearedJetPtBins.size() - 1]) {
      continue;
    }
    h1smeared.Fill(*hybridJetPt, *scaleFactor);
    h1true.Fill(*trueJetPt, *scaleFactor);
    response.Fill(*hybridJetPt, *trueJetPt, *scaleFactor);
    // So we can look at the substructure variable correlation.
    h2JetPt.Fill(*hybridJetPt, *trueJetPt, *scaleFactor);
  }

  TH1D* htrueptd = dynamic_cast<TH1D*>(h1fulleff.Clone("trueptd"));

  TFile* fout = new TFile(outputFilename.c_str(), "RECREATE");
  fout->cd();
  h1raw.SetName("raw");
  h1raw.Write();
  h1smeared.SetName("smeared");
  h1smeared.Write();
  htrueptd->Write();
  h1true.SetName("true");
  h1true.Write();
  h1fulleff.Write();
  h2JetPt.Write();

  // Unfold the standard spectra.
  int nIter = 20;
  Unfold(response, h1true, h1raw, errorTreatment, fout, "", nIter);
  // Unfold with the hybrid as the smeared input for a trivial closure.
  Unfold(response, h1true, h1smeared, errorTreatment, fout, "hybridAsInput", nIter);

  // Cleanup
  fout->Close();
}

#ifndef __CINT__
int RooSimpleJetPt()
{
  RunUnfolding();
  return 0;
} // Main program when run stand-alone
#endif
