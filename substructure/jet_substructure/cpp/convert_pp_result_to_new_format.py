""" Convert existing pp results to the new file format.

Although this is just a rename operation, it makes life substantially easier,
so it's worth the effort.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path

from jet_substructure.analysis import plot_unfolding
from jet_substructure.base import helpers


def get_n_iter(hist_name: str) -> int:
    return int("".join(filter(str.isdigit, hist_name)))


def convert_file(input_path: Path, tag: str, unfolding_output: plot_unfolding.UnfoldingOutput) -> bool:
    # Delay import to avoid explicit dependence.
    import ROOT

    # Map to Leticia's filenames.
    # The unfolding outputs.
    grooming_method_unfolding_outputs_map = {
        "dynamical_kt": "UnfoldKgzg0.4ppDefaultRmax025{tag}Dyn",
        "dynamical_time": "UnfoldKgzg0.4ppDefaultRmax025{tag}Time",
        "leading_kt": "UnfoldKgzg0.4ppDefaultRmax025{tag}nocut",
        "leading_kt_z_cut_02": "UnfoldKgzg0.2ppDefaultRmax025{tag}",
        "leading_kt_z_cut_04": "UnfoldKgzg0.4ppDefaultRmax025{tag}",
    }
    # The final outputs (unused for now)
    grooming_method_results_map = {  # noqa: F841
        "dynamical_kt": "dynamickt",
        "dynamical_time": "dynamictf",
        "leading_kt": "leadingktnocut",
        "leading_kt_z_cut_02": "leadingktzcut02",
        "leading_kt_z_cut_04": "leadingktzcut04",
    }
    # Can be used with:
    # input_path / f"result_{grooming_method_results_map[grooming_method]}.root"

    f = ROOT.TFile.Open(
        str(
            input_path
            / f"{grooming_method_unfolding_outputs_map[unfolding_output.grooming_method].format(tag=tag)}.root"
        ),
        "READ",
    )
    hist_names = [k.GetName() for k in f.GetListOfKeys()]

    # Extract some properties.
    max_n_iter = 0
    for hist_name in hist_names:
        # We could equally use the folded.
        if "Bayesian_Unfoldediter" in hist_name:
            max_n_iter = max(max_n_iter, get_n_iter(hist_name))

    # Create an output object for convenience to retrieve the identifier.
    # smeared_example_hist = binned_data.BinnedData.from_existing_data(f.Get("Bayesian_Foldediter2.root"))
    # unfolded_example_hist = binned_data.BinnedData.from_existing_data(f.Get("Bayesian_Unfoldediter2.root"))

    # f_out = ROOT.TFile.Open(str(unfolding_output./ f"{unfolding_output.identifier}.root"), "RECREATE")
    print(f"Writing to {unfolding_output.input_filename}")
    unfolding_output.input_filename.parent.mkdir(parents=True, exist_ok=True)
    f_out = ROOT.TFile.Open(str(unfolding_output.input_filename), "RECREATE")

    for hist_name in hist_names:
        # print(f"hist_name: {hist_name}")
        f.cd()
        h_temp = f.Get(hist_name)
        new_name = h_temp.GetName()
        if "Bayesian_Unfolded" in hist_name:
            n_iter = get_n_iter(hist_name)
            new_name = f"bayesian_unfolded_iter_{n_iter}"
        elif "Bayesian_Folded" in hist_name:
            n_iter = get_n_iter(hist_name)
            new_name = f"bayesian_folded_iter_{n_iter}"

        f_out.cd()
        h_temp.SetName(new_name)
        h_temp.Write(new_name)

    return True


def convert(grooming_method: str) -> None:

    # Setup
    input_path = Path("output/pp/unfolding/leticia")
    output_dir = Path("output")
    collision_system = "pp"
    # output_path = Path("output/pp/unfolding/converted")
    # output_path.mkdir(parents=True, exist_ok=True)

    # NOTE: The outputs are missing the full "true" cut efficiency hist. So we can just have to take the
    #       projections which were stored.

    # Standard
    unfolding_output = plot_unfolding.UnfoldingOutput(
        grooming_method=grooming_method,
        substructure_variable="kt",
        smeared_var_range=helpers.KtRange(0.25, 8),
        smeared_untagged_var=helpers.KtRange(8, 9),
        smeared_jet_pt_range=helpers.JetPtRange(20, 85),
        collision_system=collision_system,
        base_dir=output_dir,
        suffix="pp_R04",
        # Pass empty hists so that it doesn't try to load the hists that don't yet exist...
        hists={"ignore": "this"},  # type: ignore
    )
    convert_file(input_path=input_path, tag="", unfolding_output=unfolding_output)

    # Tracking efficiency
    unfolding_output = plot_unfolding.UnfoldingOutput(
        grooming_method=grooming_method,
        substructure_variable="kt",
        smeared_var_range=helpers.KtRange(0.25, 8),
        smeared_untagged_var=helpers.KtRange(8, 9),
        smeared_jet_pt_range=helpers.JetPtRange(20, 85),
        collision_system=collision_system,
        base_dir=output_dir,
        suffix="pp_R04_tracking_efficiency",
        # Pass empty hists so that it doesn't try to load the hists that don't yet exist...
        hists={"ignore": "this"},  # type: ignore
    )
    convert_file(input_path=input_path, tag="Eff", unfolding_output=unfolding_output)

    # Truncation
    unfolding_output = plot_unfolding.UnfoldingOutput(
        grooming_method=grooming_method,
        substructure_variable="kt",
        smeared_var_range=helpers.KtRange(0.25, 8),
        smeared_untagged_var=helpers.KtRange(8, 9),
        smeared_jet_pt_range=helpers.JetPtRange(17, 85),
        collision_system=collision_system,
        base_dir=output_dir,
        suffix="pp_R04_truncation",
        # Pass empty hists so that it doesn't try to load the hists that don't yet exist...
        hists={"ignore": "this"},  # type: ignore
    )
    convert_file(input_path=input_path, tag="TruncLow", unfolding_output=unfolding_output)
    unfolding_output = plot_unfolding.UnfoldingOutput(
        grooming_method=grooming_method,
        substructure_variable="kt",
        smeared_var_range=helpers.KtRange(0.25, 8),
        smeared_untagged_var=helpers.KtRange(8, 9),
        smeared_jet_pt_range=helpers.JetPtRange(23, 85),
        collision_system=collision_system,
        base_dir=output_dir,
        suffix="pp_R04_truncation",
        # Pass empty hists so that it doesn't try to load the hists that don't yet exist...
        hists={"ignore": "this"},  # type: ignore
    )
    convert_file(input_path=input_path, tag="TruncHigh", unfolding_output=unfolding_output)

    # Random binning
    unfolding_output = plot_unfolding.UnfoldingOutput(
        grooming_method=grooming_method,
        substructure_variable="kt",
        smeared_var_range=helpers.KtRange(0.25, 8),
        smeared_untagged_var=helpers.KtRange(8, 9),
        smeared_jet_pt_range=helpers.JetPtRange(20, 85),
        collision_system=collision_system,
        base_dir=output_dir,
        suffix="pp_R04_random_binning",
        # Pass empty hists so that it doesn't try to load the hists that don't yet exist...
        hists={"ignore": "this"},  # type: ignore
    )
    convert_file(input_path=input_path, tag="Bin", unfolding_output=unfolding_output)

    # Untagged bin
    unfolding_output = plot_unfolding.UnfoldingOutput(
        grooming_method=grooming_method,
        substructure_variable="kt",
        smeared_var_range=helpers.KtRange(0.25, 8),
        smeared_untagged_var=helpers.KtRange(0, 0.25),
        smeared_jet_pt_range=helpers.JetPtRange(20, 85),
        collision_system=collision_system,
        base_dir=output_dir,
        suffix="pp_R04",
        # Pass empty hists so that it doesn't try to load the hists that don't yet exist...
        hists={"ignore": "this"},  # type: ignore
    )
    convert_file(input_path=input_path, tag="Displaced", unfolding_output=unfolding_output)

    # Reweight prior
    unfolding_output = plot_unfolding.UnfoldingOutput(
        grooming_method=grooming_method,
        substructure_variable="kt",
        smeared_var_range=helpers.KtRange(0.25, 8),
        smeared_untagged_var=helpers.KtRange(8, 9),
        smeared_jet_pt_range=helpers.JetPtRange(20, 85),
        collision_system=collision_system,
        base_dir=output_dir,
        suffix="pp_R04_reweight_prior",
        # Pass empty hists so that it doesn't try to load the hists that don't yet exist...
        hists={"ignore": "this"},  # type: ignore
    )
    convert_file(input_path=input_path, tag="Prior", unfolding_output=unfolding_output)


if __name__ == "__main__":
    convert("dynamical_kt")
    convert("dynamical_time")
    convert("leading_kt")
    convert("leading_kt_z_cut_02")
    convert("leading_kt_z_cut_04")
