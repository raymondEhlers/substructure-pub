""" Convert HERWIG model dependence outputs to the standard output file format.

Although this is just a rename operation, it makes life substantially easier,
so it's worth the effort.

Note:
    In this case, we only have the unfolded outputs. This means that we're
    be creating an imcomplete unfolding output object. But it's the best we can do.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path

from jet_substructure.analysis import plot_unfolding
from jet_substructure.base import helpers


def convert_file(
    input_path: Path, jet_R: str, grooming_method: str, n_iter: int, unfolding_output: plot_unfolding.UnfoldingOutput
) -> bool:
    # Delay import to avoid explicit dependence.
    import ROOT

    # Map to Leticia's filenames.
    # The unfolding outputs.
    input_filename = f"ppresult_dyn_{jet_R}.root"
    grooming_method_to_histogram_name = {
        "dynamical_core": "Dyn0",
        "dynamical_kt": "Dyn1",
        "dynamical_time": "Dyn2",
        "soft_drop_z_cut_02": "SD",
    }

    f = ROOT.TFile.Open(
        str(input_path / input_filename),
        "READ",
    )
    input_hist_names = [k.GetName() for k in f.GetListOfKeys()]

    # Create an output object for convenience to retrieve the identifier.
    # smeared_example_hist = binned_data.BinnedData.from_existing_data(f.Get("Bayesian_Foldediter2.root"))
    # unfolded_example_hist = binned_data.BinnedData.from_existing_data(f.Get("Bayesian_Unfoldediter2.root"))

    # f_out = ROOT.TFile.Open(str(unfolding_output./ f"{unfolding_output.identifier}.root"), "RECREATE")
    print(f"Writing to {unfolding_output.input_filename}")
    unfolding_output.input_filename.parent.mkdir(parents=True, exist_ok=True)
    f_out = ROOT.TFile.Open(str(unfolding_output.input_filename), "RECREATE")

    # print(f"hist_name: {hist_name}")
    f.cd()
    # NOTE: This is actually a TGraphAsymmErrors
    h_temp = f.Get(grooming_method_to_histogram_name[grooming_method])
    new_name = f"bayesian_unfolded_iter_{n_iter}"

    # We need to clean up this graph manually
    if grooming_method == "soft_drop_z_cut_02":
        # Bins edges should be: -0.05, 0, 0.25, 0.5, ...
        # Practically, we can drop the first two points
        #h_temp.RemovePoint(0)
        #h_temp.RemovePoint(0)

        #h_temp.SetPointX(0, -0.025)
        #h_temp.SetPoint
        #h_temp.SetPointEXlow(0, )
        pass

    f_out.cd()
    h_temp.SetName(new_name)
    h_temp.Write(new_name)

    return True


def convert(grooming_method: str) -> None:
    # Setup
    input_path = Path("output/comparison/models/herwig/modelDependence")
    output_dir = Path("output")
    collision_system = "pp"
    # output_path = Path("output/pp/unfolding/converted")
    # output_path.mkdir(parents=True, exist_ok=True)

    # NOTE: The outputs are missing the full "true" cut efficiency hist. So we can just have to take the
    #       projections which were stored.

    # Model dependence
    # R = 0.2
    smeared_untagged_var = helpers.KtRange(0, 0.25) if "soft_drop" in grooming_method else helpers.KtRange(0.25, 0.25)
    unfolding_output = plot_unfolding.UnfoldingOutput(
        grooming_method=grooming_method,
        substructure_variable="kt",
        smeared_var_range=helpers.KtRange(0.25, 6),
        smeared_untagged_var=smeared_untagged_var,
        smeared_jet_pt_range=helpers.JetPtRange(20, 85),
        collision_system=collision_system,
        base_dir=output_dir,
        suffix="pp_R02",
        label="model_dependence",
        # Pass empty hists so that it doesn't try to load the hists that don't yet exist...
        hists={"ignore": "this"},  # type: ignore
    )
    convert_file(
        input_path=input_path, jet_R="R02", grooming_method=grooming_method, n_iter=3, unfolding_output=unfolding_output
    )

    # R = 0.4
    # Previously, 8-9
    smeared_untagged_var = helpers.KtRange(0, 0.25) if "soft_drop" in grooming_method else helpers.KtRange(0.25, 0.25)
    unfolding_output = plot_unfolding.UnfoldingOutput(
        grooming_method=grooming_method,
        substructure_variable="kt",
        smeared_var_range=helpers.KtRange(0.25, 8),
        smeared_untagged_var=smeared_untagged_var,
        smeared_jet_pt_range=helpers.JetPtRange(20, 85),
        collision_system=collision_system,
        base_dir=output_dir,
        suffix="pp_R04",
        label="model_dependence",
        # Pass empty hists so that it doesn't try to load the hists that don't yet exist...
        hists={"ignore": "this"},  # type: ignore
    )
    convert_file(
        input_path=input_path, jet_R="R04", grooming_method=grooming_method, n_iter=5, unfolding_output=unfolding_output
    )


if __name__ == "__main__":
    convert("dynamical_core")
    convert("dynamical_kt")
    convert("dynamical_time")
    convert("soft_drop_z_cut_02")
