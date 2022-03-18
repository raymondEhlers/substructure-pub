
#include <string>
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TChain.h>
#include <TH1.h>
#include <TH2F.h>


void pythiaFallBack()
{
  // Setup
  std::string groomingMethod = "leading_kt";
  std::string groomingMethod = "leading_kt_z_cut_02";
  std::string groomingMethod = "dynamical_kt";
  //std::string groomingMethod = "dynamical_time";
  std::string detLevelPrefix = "data";
  std::string truePrefix = "matched";
  std::string substructureVariableName = "kt";
  std::string outputFilename = "pythia_";
  outputFilename += substructureVariableName;
  outputFilename += "_grooming_method_";
  outputFilename += groomingMethod;
  outputFilename += ".root";
  std::cout << "outputFilename: " << outputFilename << "\n";

  // Define hists
  // Response hists
  std::vector<TH1*> hists;
  TH2F hLundPlane("hLundPlane", "hLundPlane", 100, 0, 5, 100, -5, 5);
  hists.push_back(&hLundPlane);
  TH1D hNGroomedToSplit("hNGroomedToSplit", "hNGroomedToSplit", 20, -0.5, 19.5);
  hists.push_back(&hNGroomedToSplit);
  TH1D hNGroomedToSplitHighKt("hNGroomedToSplitHighKt", "hNGroomedToSplitHighKt", 20, -0.5, 19.5);
  hists.push_back(&hNGroomedToSplitHighKt);
  TH1D hNToSplit("hNToSplit", "hNToSplit", 20, -0.5, 19.5);
  hists.push_back(&hNToSplit);
  TH1D hNToSplitHighKt("hNToSplitHighKt", "hNToSplitHighKt", 20, -0.5, 19.5);
  hists.push_back(&hNToSplitHighKt);
  for (auto h : hists) {
    h->Sumw2();
  }

  // Setup response tree.
  TChain embeddedChain("tree");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/pythia/2110/run-by-run/FAST/skim/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/pythia/2110/run-by-run/cent_woSDD/skim/*.root");

  TTreeReader mcReader(&embeddedChain);
  TTreeReaderValue<float> scaleFactor(mcReader, "scale_factor");
  TTreeReaderValue<float> detLevelJetPt(mcReader, ("jet_pt_" + detLevelPrefix).c_str());
  TTreeReaderValue<float> trueJetPt(mcReader, ("jet_pt_" + truePrefix).c_str());
  TTreeReaderValue<float> detLevelSubstructureVariable(mcReader, (groomingMethod + "_" + detLevelPrefix + "_" + substructureVariableName).c_str());
  TTreeReaderValue<float> trueSubstructureVariable(mcReader, (groomingMethod + "_" + truePrefix + "_" + substructureVariableName).c_str());
  TTreeReaderValue<float> detLevelDeltaR(mcReader, (groomingMethod + "_" + detLevelPrefix + "_delta_R").c_str());
  TTreeReaderValue<float> trueDeltaR(mcReader, (groomingMethod + "_" + truePrefix + "_delta_R").c_str());
  TTreeReaderValue<long long> nGroomedToSplitDet(mcReader, (groomingMethod + "_" + detLevelPrefix + "_n_groomed_to_split").c_str());
  TTreeReaderValue<long long> nGroomedToSplitPart(mcReader, (groomingMethod + "_" + truePrefix + "_n_groomed_to_split").c_str());
  TTreeReaderValue<long long> nToSplitDet(mcReader, (groomingMethod + "_" + detLevelPrefix + "_n_to_split").c_str());
  TTreeReaderValue<long long> nToSplitPart(mcReader, (groomingMethod + "_" + truePrefix + "_n_to_split").c_str());

  int counter = 0;
  while (mcReader.Next()) {
    if (counter % 1000000 == 0) {
        std::cout << "Jet: " << counter << "\n";
    }
    counter++;
    // Ensure that we are in the right true pt and substructure variable range.
    /*if (*trueJetPt > 160) {
      continue;
    }
    if (*trueSubstructureVariable > 100) {
      continue;
    }*/

    //h2fulleff.Fill(*trueSubstructureVariable, *trueJetPt, *scaleFactor);
    //h2smearednocuts.Fill(*hybridSubstructureVariable, *hybridJetPt, *scaleFactor);
    //responsenotrunc.Fill(*hybridSubstructureVariable, *hybridJetPt, *trueSubstructureVariable, *trueJetPt, *scaleFactor);

    if (*trueJetPt < 60 || *trueJetPt > 80) {
        continue;
    }
    // n groomed to split
    hNGroomedToSplit.Fill(*nGroomedToSplitPart, *scaleFactor);
    hNToSplit.Fill(*nToSplitPart, *scaleFactor);
    // Fill lund plane
    hLundPlane.Fill(std::log(1.0 / *trueDeltaR), std::log(*trueSubstructureVariable), *scaleFactor);
    if (*trueSubstructureVariable > 5) {
        // n groomed to split
        hNGroomedToSplitHighKt.Fill(*nGroomedToSplitPart, *scaleFactor);
        hNToSplitHighKt.Fill(*nToSplitPart, *scaleFactor);
    }

    /*dynamical_z_data_n_groomed_to_split
    hists[f"{grooming_method}_{prefix}_lund_plane"].fill(
        masked_df[f"jet_pt_{prefix}"].to_numpy(),
        np.log(1.0 / masked_df[f"{grooming_method}_{prefix}_delta_R"].to_numpy()),
        np.log(masked_df[f"{grooming_method}_{prefix}_kt"].to_numpy()),
        weight=masked_df["scale_factor"].to_numpy(),
    )*/
  }

  TFile fOut(outputFilename.c_str(), "RECREATE");
  fOut.cd();
  for (auto h: hists) {
      h->Write();
  }

  // Cleanup
  fOut.Write();
  fOut.Close();
}
