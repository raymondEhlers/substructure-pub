
#include <string>
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <TTreeReader.h>
#include <TChain.h>
#include <TH1.h>
#include <TH2F.h>


void embeddingFallBack()
{
  // Setup
  std::string groomingMethod = "leading_kt";
  std::string hybridPrefix = "hybrid";
  std::string detLevelPrefix = "det_level";
  std::string truePrefix = "true";
  std::string substructureVariableName = "kt";
  std::string outputFilename = "embeddingResponse_";
  outputFilename += substructureVariableName;
  outputFilename += "_grooming_method_";
  outputFilename += groomingMethod;
  outputFilename += ".root";

  // Define hists
  // Response hists
  std::vector<TH1*> hists;
  TH2F hHybridTrueKtResponse("hHybridTrueKtResponse", "hHybridTrueKtResponse", 26, -1, 25, 26, -1, 25);
  hists.push_back(&hHybridTrueKtResponse);
  TH2F hHybridDetKtResponse("hHybridDetKtResponse", "hHybridDetKtResponse", 26, -1, 25, 26, -1, 25);
  hists.push_back(&hHybridDetKtResponse);
  TH2F hHybridTrueKtResponsePureMatches("hHybridTrueKtResponsePureMatches", "hHybridTrueKtResponsePureMatches", 26, -1, 25, 26, -1, 25);
  hists.push_back(&hHybridTrueKtResponsePureMatches);
  TH2F hHybridDetKtResponsePureMatches("hHybridDetKtResponsePureMatches", "hHybridDetKtResponsePureMatches", 26, -1, 25, 26, -1, 25);
  hists.push_back(&hHybridDetKtResponsePureMatches);
  // Matching hists
  TH1D hHybridDetMatchingAll("hHybridDetMatchingAll", "hHybridDetMatchingAll", 150, 0, 150);
  hists.push_back(&hHybridDetMatchingAll);
  TH1D hHybridDetMatchingPure("hHybridDetMatchingPure", "hHybridDetMatchingPure", 150, 0, 150);
  hists.push_back(&hHybridDetMatchingPure);
  TH1D hHybridDetMatchingLeadingUntaggedSubleadingCorrect("hHybridDetMatchingLeadingUntaggedSubleadingCorrect", "hHybridDetMatchingLeadingUntaggedSubleadingCorrect", 150, 0, 150);
  hists.push_back(&hHybridDetMatchingLeadingUntaggedSubleadingCorrect);
  TH1D hHybridDetMatchingLeadingCorrectSubleadingUntagged("hHybridDetMatchingLeadingCorrectSubleadingUntagged", "hHybridDetMatchingLeadingCorrectSubleadingUntagged", 150, 0, 150);
  hists.push_back(&hHybridDetMatchingLeadingCorrectSubleadingUntagged);
  TH1D hHybridDetMatchingLeadingUntaggedSubleadingMistag("hHybridDetMatchingLeadingUntaggedSubleadingMistag", "hHybridDetMatchingLeadingUntaggedSubleadingMistag", 150, 0, 150);
  hists.push_back(&hHybridDetMatchingLeadingUntaggedSubleadingMistag);
  TH1D hHybridDetMatchingLeadingMistagSubleadingUntagged("hHybridDetMatchingLeadingMistagSubleadingUntagged", "hHybridDetMatchingLeadingMistagSubleadingUntagged", 150, 0, 150);
  hists.push_back(&hHybridDetMatchingLeadingMistagSubleadingUntagged);
  TH1D hHybridDetMatchingSwap("hHybridDetMatchingSwap", "hHybridDetMatchingSwap", 150, 0, 150);
  hists.push_back(&hHybridDetMatchingSwap);
  TH1D hHybridDetMatchingBothUntagged("hHybridDetMatchingBothUntagged", "hHybridDetMatchingBothUntagged", 150, 0, 150);
  hists.push_back(&hHybridDetMatchingBothUntagged);

  for (auto h : hists) {
    h->Sumw2();
  }

  // Setup response tree.
  TChain embeddedChain("tree");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5884/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5885/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5886/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5887/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5888/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5889/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5890/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5891/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5892/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5893/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5894/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5895/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5896/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5897/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5898/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5898/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5900/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5901/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5902/skim/merged/*.root");
  embeddedChain.Add("/clusterfs4/rehlers/substructure/trains/embedPythia/5903/skim/merged/*.root");

  TTreeReader mcReader(&embeddedChain);
  TTreeReaderValue<float> scaleFactor(mcReader, "scale_factor");
  TTreeReaderValue<float> hybridJetPt(mcReader, ("jet_pt_" + hybridPrefix).c_str());
  TTreeReaderValue<float> detLevelJetPt(mcReader, ("jet_pt_" + detLevelPrefix).c_str());
  TTreeReaderValue<float> trueJetPt(mcReader, ("jet_pt_" + truePrefix).c_str());
  TTreeReaderValue<float> hybridSubstructureVariable(mcReader, (groomingMethod + "_" + hybridPrefix + "_" + substructureVariableName).c_str());
  TTreeReaderValue<float> detLevelSubstructureVariable(mcReader, (groomingMethod + "_" + detLevelPrefix + "_" + substructureVariableName).c_str());
  TTreeReaderValue<float> trueSubstructureVariable(mcReader, (groomingMethod + "_" + truePrefix + "_" + substructureVariableName).c_str());
  TTreeReaderValue<long long> matchingLeading(mcReader, (groomingMethod + "_hybrid_det_level_matching_leading").c_str());
  TTreeReaderValue<long long> matchingSubleading(mcReader, (groomingMethod + "_hybrid_det_level_matching_subleading").c_str());
  TTreeReaderValue<long long> matchingPartDetLeading(mcReader, (groomingMethod + "_det_level_true_matching_leading").c_str());
  TTreeReaderValue<long long> matchingPartDetSubleading(mcReader, (groomingMethod + "_det_level_true_matching_subleading").c_str());

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

    // Now start making cuts on the hybrid level.
    if (*hybridJetPt < 40 || *hybridJetPt > 120) {
      continue;
    }
    // Also cut on hybrid substructure variable.
    double hybridSubstructureVariableValue = *hybridSubstructureVariable;
    // TODO: This only works for kt!!!
    /*if (hybridSubstructureVariableValue < 0) {
      // Assign to the untagged bin.
      hybridSubstructureVariableValue = 0.5;
    }
    else {
      if (hybridSubstructureVariableValue < minSmearedSplittingVariable || hybridSubstructureVariableValue > smearedSplittingVariableBins[smearedSplittingVariableBins.size() - 1]) {
        continue;
      }
    }
    // Matching cuts: Requiring a pure match.
    if (usePureMatches && !(*matchingLeading == 1 && *matchingSubleading == 1)) {
      continue;
    }*/
    hHybridTrueKtResponse.Fill(hybridSubstructureVariableValue, *trueSubstructureVariable, *scaleFactor);
    hHybridDetKtResponse.Fill(hybridSubstructureVariableValue, *detLevelSubstructureVariable, *scaleFactor);
    //h2smeared.Fill(hybridSubstructureVariableValue, *hybridJetPt, *scaleFactor);
    //h2true.Fill(*trueSubstructureVariable, *trueJetPt, *scaleFactor);
    //response.Fill(hybridSubstructureVariableValue, *hybridJetPt, *trueSubstructureVariable, *trueJetPt, *scaleFactor);
    // Pure matches
    if (*matchingLeading == 1 && *matchingSubleading == 1) {
        hHybridTrueKtResponsePureMatches.Fill(hybridSubstructureVariableValue, *trueSubstructureVariable, *scaleFactor);
        hHybridDetKtResponsePureMatches.Fill(hybridSubstructureVariableValue, *detLevelSubstructureVariable, *scaleFactor);
    }

    // Matching
    hHybridDetMatchingAll.Fill(*detLevelJetPt);
    if (*matchingLeading == 0) {
        std::cout << "0\n";
    }
    if (*matchingLeading == 1 && *matchingSubleading == 1) {
        hHybridDetMatchingPure.Fill(*detLevelJetPt);
    }
    if (*matchingLeading == 3 && *matchingSubleading == 1) {
        hHybridDetMatchingLeadingUntaggedSubleadingCorrect.Fill(*detLevelJetPt);
    }
    if (*matchingLeading == 1 && *matchingSubleading == 3) {
        hHybridDetMatchingLeadingCorrectSubleadingUntagged.Fill(*detLevelJetPt);
    }
    if (*matchingLeading == 3 && *matchingSubleading == 2) {
        hHybridDetMatchingLeadingUntaggedSubleadingMistag.Fill(*detLevelJetPt);
    }
    if (*matchingLeading == 2 && *matchingSubleading == 3) {
        hHybridDetMatchingLeadingMistagSubleadingUntagged.Fill(*detLevelJetPt);
    }
    if (*matchingLeading == 2 && *matchingSubleading == 2) {
        hHybridDetMatchingSwap.Fill(*detLevelJetPt);
    }
    if (*matchingLeading == 3 && *matchingSubleading == 3) {
        hHybridDetMatchingBothUntagged.Fill(*detLevelJetPt);
    }
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
