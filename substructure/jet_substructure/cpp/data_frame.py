""" RDataFrame based analysis.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import attr
import numpy as np
import numpy.typing as npt
import uproot
from pachyderm import binned_data

from jet_substructure.base import helpers, skim_analysis_objects
from jet_substructure.base import unfolding as unfolding_base


logger = logging.getLogger(__name__)

# Typing helpers
RDF = Any
RootHist = Any


@attr.s(frozen=True)
class MatchingIndex:
    measured_like_prefix: str = attr.ib()
    generator_like_prefix: str = attr.ib()
    matching_type: str = attr.ib()

    @property
    def matching_level(self) -> str:
        return f"{self.measured_like_prefix}_{self.generator_like_prefix}"


def new_matching_hists(
    df: RDF,
    grooming_method: str,
    hist_suffix: str,
    matching_index: MatchingIndex,
    jet_pt_column_format: str,
    kt_axis: npt.NDArray[np.float32],
    jet_pt_axis: npt.NDArray[np.float32],
    kt_selection_axis: npt.NDArray[np.float32],
    matching_jet_pt_prefix: Optional[str] = None,
    create_generator_subjet_in_measured_jet_hists: bool = False,
) -> List[RootHist]:
    if matching_jet_pt_prefix is None:
        matching_jet_pt_prefix = matching_index.generator_like_prefix

    if hist_suffix:
        hist_suffix = f"_{hist_suffix}"

    # Setup
    hists = []

    # Matching
    name = f"{grooming_method}_matching_{matching_index.matching_level}_type_{matching_index.matching_type}_jet_pt_axis_{matching_jet_pt_prefix}{hist_suffix}"
    h_matching = df.Histo2D(
        (name, name, len(jet_pt_axis) - 1, jet_pt_axis, len(kt_selection_axis) - 1, kt_selection_axis),
        jet_pt_column_format.format(prefix=matching_jet_pt_prefix),
        # Always want to make selections in the smeared (hybrid) kt
        f"{grooming_method}_hybrid_kt",
        "scale_factor",
    )
    hists.append(h_matching)

    # Responses
    # Hybrid-true
    name = f"{grooming_method}_hybrid_true_kt_response_matching_{matching_index.matching_level}_type_{matching_index.matching_type}{hist_suffix}"
    kt_hybrid_true_response = df.Histo2D(
        (name, name, len(kt_axis) - 1, kt_axis, len(kt_axis) - 1, kt_axis),
        f"{grooming_method}_hybrid_kt",
        f"{grooming_method}_true_kt",
        "scale_factor",
    )
    hists.append(kt_hybrid_true_response)
    # Hybrid-det level
    if matching_index.measured_like_prefix == "hybrid":
        name = f"{grooming_method}_hybrid_det_level_kt_response_matching_{matching_index.matching_level}_type_{matching_index.matching_type}{hist_suffix}"
        kt_hybrid_det_level_response = df.Histo2D(
            (name, name, len(kt_axis) - 1, kt_axis, len(kt_axis) - 1, kt_axis),
            f"{grooming_method}_hybrid_kt",
            f"{grooming_method}_det_level_kt",
            "scale_factor",
        )
        hists.append(kt_hybrid_det_level_response)
    # Det-level true
    if matching_index.generator_like_prefix == "true":
        name = f"{grooming_method}_det_level_true_kt_response_matching_{matching_index.matching_level}_type_{matching_index.matching_type}{hist_suffix}"
        kt_det_level_true_response = df.Histo2D(
            (name, name, len(kt_axis) - 1, kt_axis, len(kt_axis) - 1, kt_axis),
            f"{grooming_method}_det_level_kt",
            f"{grooming_method}_true_kt",
            "scale_factor",
        )
        hists.append(kt_det_level_true_response)

    # Does the generator-like subjet stay in the measured-like jet?
    # As a concrete example, does the det level subjet stay in the hybrid jet?
    # In principle, we keep this switch in case we have to look at old productions which don't have this as an option.
    if create_generator_subjet_in_measured_jet_hists:
        for subjet_name in ["leading", "subleading"]:
            name = f"{grooming_method}_{subjet_name}_{matching_index.generator_like_prefix}_subjet_momentum_fraction_in_{matching_index.measured_like_prefix}_matching_{matching_index.matching_level}_type_{matching_index.matching_type}{hist_suffix}"
            h_subjet_pt_fraction = df.Histo1D(
                (
                    name,
                    name,
                    50,
                    0,
                    1,
                ),
                f"{grooming_method}_{matching_index.generator_like_prefix}_{subjet_name}_subjet_momentum_fraction_in_{matching_index.measured_like_prefix}_jet",
                "scale_factor",
            )
            hists.append(h_subjet_pt_fraction)

    return hists


def matching_hists(  # noqa: C901
    df: RDF,
    grooming_method: str,
    hist_suffix: str,
    general_selection: str,
    measured_like_prefix: str,
    generator_like_prefix: str,
    jet_pt_column_format: str,
    matching_jet_pt_prefix: Optional[str] = None,
    matching_jet_pt_axis: Optional[Tuple[int, float, float]] = None,
    create_generator_subjet_in_measured_jet_hists: bool = False,
) -> List[RootHist]:
    # Validation
    if matching_jet_pt_axis is None:
        matching_jet_pt_axis = (150, 0, 150)
    if matching_jet_pt_prefix is None:
        matching_jet_pt_prefix = generator_like_prefix

    # Setup
    hists = []
    matching_level = f"{measured_like_prefix}_{generator_like_prefix}"
    matching_map: Dict[str, str] = {
        "all": "",
        "pure": f"{grooming_method}_{matching_level}_matching_leading == 1"
        f" && {grooming_method}_{matching_level}_matching_subleading == 1",
        "leading_untagged_subleading_correct": f"{grooming_method}_{matching_level}_matching_leading == 3"
        f" && {grooming_method}_{matching_level}_matching_subleading == 1",
        "leading_correct_subleading_untagged": f"{grooming_method}_{matching_level}_matching_leading == 1"
        f" && {grooming_method}_{matching_level}_matching_subleading == 3",
        "leading_correct_subleading_mistag": f"{grooming_method}_{matching_level}_matching_leading == 1"
        f" && {grooming_method}_{matching_level}_matching_subleading == 2",
        "leading_mistag_subleading_correct": f"{grooming_method}_{matching_level}_matching_leading == 2"
        f" && {grooming_method}_{matching_level}_matching_subleading == 1",
        "leading_untagged_subleading_mistag": f"{grooming_method}_{matching_level}_matching_leading == 3"
        f" && {grooming_method}_{matching_level}_matching_subleading == 2",
        "leading_mistag_subleading_untagged": f"{grooming_method}_{matching_level}_matching_leading == 2"
        f" && {grooming_method}_{matching_level}_matching_subleading == 3",
        "swap": f"{grooming_method}_{matching_level}_matching_leading == 2"
        f" && {grooming_method}_{matching_level}_matching_subleading == 2",
        "both_untagged": f"{grooming_method}_{matching_level}_matching_leading == 3"
        f" && {grooming_method}_{matching_level}_matching_subleading == 3",
    }

    # First, apply the general selection
    if general_selection:
        df = df.Filter(general_selection)

    for matching_type, selection in matching_map.items():
        # Empty string will break the filter, so we need to only apply it if there is a valid selection.
        if selection:
            df_selection = df.Filter(selection)
        else:
            df_selection = df

        # Matching
        name = f"{grooming_method}_{matching_level}_matching_{matching_type}"
        if hist_suffix:
            name += f"_{hist_suffix}"
        h_matching = df_selection.Histo1D(
            (name, name, *matching_jet_pt_axis),
            jet_pt_column_format.format(prefix=matching_jet_pt_prefix),
            "scale_factor",
        )
        hists.append(h_matching)

        # Responses
        # Hybrid-true
        name = f"{grooming_method}_hybrid_true_kt_response_{matching_level}_matching_type_{matching_type}"
        if hist_suffix:
            name += f"_{hist_suffix}"
        kt_hybrid_true_response = df_selection.Histo2D(
            (
                name,
                name,
                26,
                -1,
                25,
                26,
                -1,
                25,
            ),
            f"{grooming_method}_hybrid_kt",
            f"{grooming_method}_true_kt",
            "scale_factor",
        )
        hists.append(kt_hybrid_true_response)
        # Hybrid-det level
        if measured_like_prefix == "hybrid":
            name = f"{grooming_method}_hybrid_det_level_kt_response_{matching_level}_matching_type_{matching_type}"
            if hist_suffix:
                name += f"_{hist_suffix}"
            kt_hybrid_det_level_response = df_selection.Histo2D(
                (
                    name,
                    name,
                    26,
                    -1,
                    25,
                    26,
                    -1,
                    25,
                ),
                f"{grooming_method}_hybrid_kt",
                f"{grooming_method}_det_level_kt",
                "scale_factor",
            )
            hists.append(kt_hybrid_det_level_response)
        # Det-level true
        if generator_like_prefix == "true":
            name = f"{grooming_method}_det_level_true_kt_response_{matching_level}_matching_type_{matching_type}"
            if hist_suffix:
                name += f"_{hist_suffix}"
            kt_det_level_true_response = df_selection.Histo2D(
                (
                    name,
                    name,
                    26,
                    -1,
                    25,
                    26,
                    -1,
                    25,
                ),
                f"{grooming_method}_det_level_kt",
                f"{grooming_method}_true_kt",
                "scale_factor",
            )
            hists.append(kt_det_level_true_response)

        # Does the generator-like subjet stay in the measured-like jet?
        # As a concrete example, does the det level subjet stay in the hybrid jet?
        # In principle, we keep this switch in case we have to look at old productions which don't have this as an option.
        if create_generator_subjet_in_measured_jet_hists:
            for subjet_name in ["leading", "subleading"]:
                name = f"{grooming_method}_{subjet_name}_{generator_like_prefix}_subjet_momentum_fraction_in_{measured_like_prefix}_{matching_level}_matching_{matching_type}"
                if hist_suffix:
                    name += f"_{hist_suffix}"
                h_subjet_pt_fraction = df_selection.Histo1D(
                    (
                        name,
                        name,
                        50,
                        0,
                        1,
                    ),
                    f"{grooming_method}_{generator_like_prefix}_{subjet_name}_subjet_momentum_fraction_in_{measured_like_prefix}_jet",
                    "scale_factor",
                )
                hists.append(h_subjet_pt_fraction)

    return hists


def _substructure_hists(
    df: RDF,
    jet_pt_column_format: str,
    jet_pt_axis: Tuple[int, float, float],
    jet_R: float,
    prefix: str,
    grooming_method: str,
    tag: str,
    jet_pt_prefix: Optional[str] = None,
    include_stats_hist: Optional[bool] = False,
) -> List[RootHist]:
    # Validation
    if jet_pt_prefix is None:
        jet_pt_prefix = prefix

    # Setup
    hists = []

    # Apply no additional pt cuts beyond those applied outside, but plot against the relevant jet pt.
    jet_pt_column = jet_pt_column_format.format(prefix=jet_pt_prefix)

    # kt binning should vary with jet R because the kt kinematic limits vary with jet_R
    kt_axis = np.linspace(-1, 25, 26 + 1, dtype=np.float64)
    if jet_R == 0.2:
        # NOTE: Lower edge varies to ensure that we only have below 0.
        kt_axis = np.linspace(-0.5, 12, 25 + 1, dtype=np.float64)

    kt = df.Histo2D(
        (
            f"{grooming_method}_{prefix}_kt{tag}",
            f"{grooming_method}_{prefix}_kt{tag}",
            *jet_pt_axis,
            len(kt_axis) - 1,
            kt_axis,
        ),
        jet_pt_column,
        f"{grooming_method}_{prefix}_kt",
        "scale_factor",
    )
    hists.append(kt)
    if include_stats_hist:
        # Determine the statistics that are available by not scaling according to the scale factor.
        # In data, this will be trivially the same, but for the response, this is quite helpful.
        kt_stats = df.Histo2D(
            (
                f"{grooming_method}_{prefix}_kt_stats{tag}",
                f"{grooming_method}_{prefix}_kt_stats{tag}",
                *jet_pt_axis,
                len(kt_axis) - 1,
                kt_axis,
            ),
            jet_pt_column,
            f"{grooming_method}_{prefix}_kt",
        )
        hists.append(kt_stats)
    # Use 0.02 for the bin width.
    n_bins_delta_R = round((jet_R + 0.02) / 0.02)
    delta_R = df.Histo2D(
        (
            f"{grooming_method}_{prefix}_delta_R{tag}",
            f"{grooming_method}_{prefix}_delta_R{tag}",
            *jet_pt_axis,
            n_bins_delta_R,
            -0.02,
            jet_R,
        ),
        jet_pt_column,
        f"{grooming_method}_{prefix}_delta_R",
        "scale_factor",
    )
    hists.append(delta_R)
    z = df.Histo2D(
        (f"{grooming_method}_{prefix}_z{tag}", f"{grooming_method}_{prefix}_z{tag}", *jet_pt_axis, 21, -0.025, 0.5),
        jet_pt_column,
        f"{grooming_method}_{prefix}_z",
        "scale_factor",
    )
    hists.append(z)
    n_to_split = df.Histo2D(
        (
            f"{grooming_method}_{prefix}_n_to_split{tag}",
            f"{grooming_method}_{prefix}_n_to_split{tag}",
            *jet_pt_axis,
            10,
            -0.5,
            9.5,
        ),
        jet_pt_column,
        f"{grooming_method}_{prefix}_n_to_split",
        "scale_factor",
    )
    hists.append(n_to_split)
    n_groomed_to_split = df.Histo2D(
        (
            f"{grooming_method}_{prefix}_n_groomed_to_split{tag}",
            f"{grooming_method}_{prefix}_n_groomed_to_split{tag}",
            *jet_pt_axis,
            10,
            -0.5,
            9.5,
        ),
        jet_pt_column,
        f"{grooming_method}_{prefix}_n_groomed_to_split",
        "scale_factor",
    )
    hists.append(n_groomed_to_split)
    n_passed_grooming = df.Histo2D(
        (
            f"{grooming_method}_{prefix}_n_passed_grooming{tag}",
            f"{grooming_method}_{prefix}_n_passed_grooming{tag}",
            *jet_pt_axis,
            10,
            -0.5,
            9.5,
        ),
        jet_pt_column,
        f"{grooming_method}_{prefix}_n_passed_grooming",
        "scale_factor",
    )
    hists.append(n_passed_grooming)
    df_lund_plane = df.Define("lund_plane_x_axis", f"log(1.0 / {grooming_method}_{prefix}_delta_R)").Define(
        f"{grooming_method}_{prefix}_log_kt", f"log({grooming_method}_{prefix}_kt)"
    )
    lund_plane = df_lund_plane.Histo2D(
        (
            f"{grooming_method}_{prefix}_lund_plane{tag}",
            f"{grooming_method}_{prefix}_lund_plane{tag}",
            100,
            0,
            5,
            100,
            -5.0,
            5.0,
        ),
        "lund_plane_x_axis",
        f"{grooming_method}_{prefix}_log_kt",
        "scale_factor",
    )
    hists.append(lund_plane)

    return hists


def run_embedded_pt_hard_scaling(  # noqa: C901
    collision_system: str,
    input_filenames: Sequence[Path],
    tree_name: str,
    prefixes: Sequence[str],
    grooming_method: str,
    jet_R: float,
    main_jet_pt_range: helpers.JetPtRange,
    output_filename: Path,
    jet_pt_prefix_first: bool = False,
    n_cores: int = 8,
    cross_check_task: bool = False,
) -> Tuple[bool, str]:
    # TODO: For now (Sept 2020), I just copy to move quickly. But it would be better to refactor the setup.

    # Setup
    smeared_cut_prefix = "hybrid" if collision_system == "embedPythia" else "data"
    # Parameters
    jet_pt_column_format = "{prefix}_jet_pt" if jet_pt_prefix_first else "jet_pt_{prefix}"

    # Delay ROOT import so we don't explicitly rely on it.
    import ROOT

    # Setup for ROOT
    # Enable multithreading
    ROOT.ROOT.EnableImplicitMT(n_cores)
    # Sumw2
    ROOT.TH1.SetDefaultSumw2(True)

    # Setup tree
    main_tree = ROOT.TChain(tree_name)
    for filename in input_filenames:
        main_tree.Add(str(filename))
    if cross_check_task:
        friend_tree = ROOT.TChain("tree")
        for filename in input_filenames:
            friend_tree.Add(str(filename.parent.parent / "scale_factor" / filename.name))
        # Add friends with scale factors
        main_tree.AddFriend(friend_tree)

    # Keep the fully original DF so we can see everything applied to it.
    df_true_original = ROOT.RDataFrame(main_tree)
    df_original = df_true_original

    if cross_check_task:
        # Add the aliases. This has to be done after the df is defined because apparently they don't carry over.
        renames = skim_analysis_objects.cross_check_task_branch_name_shim(
            grooming_method=grooming_method, input_branches=df_original.GetColumnNames()
        )
        for k, v in renames.items():
            df_original = df_original.Alias(k, v)

    # Add scale factor column with 1s if it doesn't exist yet.
    if "scale_factor" not in df_original.GetColumnNames():
        logger.info("Defining scale_factor column")
        df_original = df_original.Define("scale_factor", "1")

    # Apply general cuts.
    # Double counting must be applied for embedding.
    if collision_system == "embedPythia":
        double_counting_cut = f"(det_level_leading_track_pt >= hybrid_leading_track_pt) && ({jet_pt_column_format.format(prefix='true')} >= 10)"
        df_original = df_original.Filter(double_counting_cut)

    # Emulate the double counting cut
    if collision_system == "pythia":
        df_original = df_original.Filter("pt_hard >= 10")

    # Workaround for older pythia productions that we can't reskim so easily.
    # We can remove this eventually when train 2110 is replaced.
    true_prefix = "matched" if collision_system == "pythia" else "true"

    hists = []
    # We simply want the scaled true jet spectra.
    # No additional cuts.
    hists.append(
        df_original.Histo1D(
            ("true_pt_spectra", "true_pt_spectra", 200, 0, 200),
            f"{jet_pt_column_format.format(prefix=true_prefix)}",
            "scale_factor",
        )
    )
    # And with the hybrid pt cut.
    jet_pt_cut = f"{jet_pt_column_format.format(prefix=smeared_cut_prefix)} >= {main_jet_pt_range.min} && {jet_pt_column_format.format(prefix=smeared_cut_prefix)} < {main_jet_pt_range.max}"
    df = df_original.Filter(jet_pt_cut)
    hists.append(
        df.Histo1D(
            (
                f"true_pt_spectra_{jet_pt_column_format.format(prefix=smeared_cut_prefix)}_40_120",
                f"true_pt_spectra_{jet_pt_column_format.format(prefix=smeared_cut_prefix)}_40_120",
                200,
                0,
                200,
            ),
            f"{jet_pt_column_format.format(prefix=true_prefix)}",
            "scale_factor",
        )
    )

    # Add the plots as a function of hybrid pt (if we can)
    if collision_system == "embedPythia":
        hists.append(
            df_original.Histo1D(
                ("hybrid_pt_spectra", "hybrid_pt_spectra", 200, 0, 200),
                f"{jet_pt_column_format.format(prefix=smeared_cut_prefix)}",
                "scale_factor",
            )
        )

    # Calculate the DataFrame by forcing it determine a property.
    # Discard the result - we don't really care. We just need a meaningless property.
    logger.info("Calculating DF...")
    hists[0].GetEntries()

    logger.info(f"Creating output file for {collision_system}, {grooming_method}, {prefixes}")
    logger.info(f"Writing to {output_filename}")
    output = ROOT.TFile(str(output_filename), "RECREATE")
    output.cd()
    for h in hists:
        h.SetDirectory(output)
        # Why doesn't h.Write() work? Because ROOT. It fucking sucks.
        # h.Write()
    output.Write()
    # output.ls()
    output.Close()

    logger.info("Done!")

    return (True, "Processed")


def run_create_closure_ratio(  # noqa: C901
    collision_system: str,
    input_filenames: Sequence[Path],
    tree_name: str,
    prefixes: Sequence[str],
    grooming_method: str,
    jet_R: float,  # Intentionally ignored, but kept for uniform interface.
    main_jet_pt_range: helpers.JetPtRange,  # Intentionally ignored, but kept for uniform interface.
    output_filename: Path,
    # NOTE: This unfolding config and settings arguments are the only arguments which varies from
    #       the other run functions.
    base_unfolding_config: Mapping[str, Any],
    unfolding_settings: Mapping[str, Any],
    jet_pt_prefix_first: bool = False,
    n_cores: int = 8,
    cross_check_task: bool = False,
) -> Tuple[bool, str]:
    """Create the histogram necessary to create the closure ratio.

    This histogram is binned in the (smeared substructure variable, smeared jet pt) and by creating the ratio, we
    can reweight the response to match the data.

    The idea here is that for some unfolding variation (say, the random binning systematic), we want to have binning
    that matches that case, but we always want to use the nominal dataset data. The nominal dataset data must already
    be passed to this function, but retrieving the correct binning must be done in this function.

    Note:
        We are supposed to run this once for each dataset. So I ratio will require at least two calls to this function.
    """
    # Setup
    prefix_for_ratio = "hybrid" if collision_system == "embedPythia" else "data"
    # Parameters
    jet_pt_column_format = "{prefix}_jet_pt" if jet_pt_prefix_first else "jet_pt_{prefix}"
    # Retrieve the binning
    # We always want the binning that is relevant for this particular unfolding case so we don't have any mismatches.
    # NOTE: The reweighting data itself will always come from the nominal dataset.
    # NOTE: The np.unique accounts for the case where we want to untagged bin. In that case,
    #       we repeat the same bin edge. Usually, this is accounted for in the `SubstructureVariableSettings2D.from_binning(...)`
    #       but we can't take advantage of that here because we need to pass just the config. Therefore, to have valid
    #       binning, we need to only take unique values. unique sorts, but this is fine because they must be strictly
    #       increasing anyway.
    smeared_substructure_variable_bins = np.unique(  # type: ignore
        unfolding_base.get_binning(
            base_unfolding_config=base_unfolding_config,
            unfolding_settings=unfolding_settings,
            name="smeared_kt",
            grooming_method=grooming_method,
        )
    )
    smeared_jet_pt_bins = unfolding_base.get_binning(
        base_unfolding_config=base_unfolding_config,
        unfolding_settings=unfolding_settings,
        name="smeared_jet_pt",
        grooming_method=grooming_method,
    )
    # Ratio hist name
    # It encodes the binning so we can have multiple hists stored in a single file, which
    # makes variations in binning easier (since we don't have to start from scratch).
    # Since we use this in many places, it's better to define it immediately.
    hist_name = unfolding_base.hist_name_for_ratio_2D(
        grooming_method=grooming_method,
        prefix_for_ratio=prefix_for_ratio,
        smeared_substructure_variable_bins=smeared_substructure_variable_bins,
        smeared_jet_pt_bins=smeared_jet_pt_bins,
    )

    # Check for existing file. If it exists, check that the binning is the same.
    # We do this check early because it allows us to bail out it it already exists.
    if output_filename.exists():
        with uproot.open(output_filename) as f:
            # Check for an existing hist. Even if the binning is encoded in the name,
            # we check the binning explicitly to ensure we haven't overlooked anything.
            h_uproot = f.get(hist_name, None)
            if h_uproot:
                h_temp = binned_data.BinnedData.from_existing_data(h_uproot)

                # First, check that we can actually make the comparison.
                tests_for_same_binning = [
                    len(h_temp.axes[0].bin_edges) == len(smeared_substructure_variable_bins),
                    len(h_temp.axes[1].bin_edges) == len(smeared_jet_pt_bins),
                ]
                # If the lengths agree, we can then proceed to comparing the binning. If not, we definitely need to recalculate.
                if all(tests_for_same_binning):
                    tests_for_same_binning = [
                        np.allclose(h_temp.axes[0].bin_edges, smeared_substructure_variable_bins),
                        np.allclose(h_temp.axes[1].bin_edges, smeared_jet_pt_bins),
                    ]
                    # If output already exists, we can return immediately.
                    if all(tests_for_same_binning):
                        return (True, f"Same binning. Returning early. Name: {hist_name}")
            else:
                logger.info(f"Could not find hist {hist_name}. Creating...")

    # Delay ROOT import so we don't explicitly rely on it.
    import ROOT

    # Setup for ROOT
    # Enable multithreading
    ROOT.ROOT.EnableImplicitMT(n_cores)
    # Sumw2
    ROOT.TH1.SetDefaultSumw2(True)

    # Setup tree
    main_tree = ROOT.TChain(tree_name)
    for filename in input_filenames:
        main_tree.Add(str(filename))
    if cross_check_task:
        friend_tree = ROOT.TChain("tree")
        for filename in input_filenames:
            friend_tree.Add(str(filename.parent.parent / "scale_factor" / filename.name))
        # Add friends with scale factors
        main_tree.AddFriend(friend_tree)

    # Keep the fully original DF so we can see everything applied to it.
    df_true_original = ROOT.RDataFrame(main_tree)
    df_original = df_true_original

    if cross_check_task:
        # Add the aliases. This has to be done after the df is defined because apparently they don't carry over.
        renames = skim_analysis_objects.cross_check_task_branch_name_shim(
            grooming_method=grooming_method, input_branches=df_original.GetColumnNames()
        )
        for k, v in renames.items():
            df_original = df_original.Alias(k, v)

    # Add scale factor column with 1s if it doesn't exist yet.
    if "scale_factor" not in df_original.GetColumnNames():
        logger.info("Defining scale_factor column")
        df_original = df_original.Define("scale_factor", "1")

    # Apply general cuts.
    # Double counting must be applied for embedding.
    if collision_system == "embedPythia":
        double_counting_cut = f"(det_level_leading_track_pt >= hybrid_leading_track_pt) && ({jet_pt_column_format.format(prefix='true')} >= 10)"
        df_original = df_original.Filter(double_counting_cut)

    hists = []
    hists.append(
        df_original.Histo2D(
            (
                hist_name,
                hist_name,
                len(smeared_substructure_variable_bins) - 1,
                smeared_substructure_variable_bins,
                len(smeared_jet_pt_bins) - 1,
                smeared_jet_pt_bins,
            ),
            f"{grooming_method}_{prefix_for_ratio}_kt",
            f"{jet_pt_column_format.format(prefix=prefix_for_ratio)}",
            "scale_factor",
        )
    )

    # Calculate the DataFrame by forcing it determine a property.
    # Discard the result - we don't really care. We just need a meaningless property.
    logger.info("Calculating DF...")
    hists[0].GetEntries()

    logger.info(f"Creating ratio output file for {collision_system}, {grooming_method}, {prefixes}")
    logger.info(f"Writing to {output_filename}")
    output = ROOT.TFile(str(output_filename), "UPDATE")
    output.cd()
    for h in hists:
        h.SetDirectory(output)
        # Why doesn't h.Write() work? Because ROOT. It fucking sucks.
        # h.Write()
    output.Write()
    # output.ls()
    output.Close()

    logger.info("Done!")

    return (True, f"Processed, name: {hist_name}")


def run_response(  # noqa: C901
    collision_system: str,
    input_filenames: Sequence[Path],
    tree_name: str,
    prefixes: Sequence[str],
    grooming_method: str,
    jet_R: float,
    main_jet_pt_range: helpers.JetPtRange,
    output_filename: Path,
    jet_pt_prefix_first: bool = False,
    n_cores: int = 8,
    cross_check_task: bool = False,
) -> Tuple[bool, str]:
    # TODO: For now (Sept 2020), I just copy to move quickly. But it would be better to refactor the setup.

    # Delay ROOT import so we don't explicitly rely on it.
    import ROOT

    # Setup for ROOT
    # Enable multithreading
    ROOT.ROOT.EnableImplicitMT(n_cores)
    # Sumw2
    ROOT.TH1.SetDefaultSumw2(True)

    # Parameters
    jet_pt_column_format = "jet_pt_{prefix}"
    if jet_pt_prefix_first:
        jet_pt_column_format = "{prefix}_jet_pt"

    # Setup tree
    main_tree = ROOT.TChain(tree_name)
    for filename in input_filenames:
        main_tree.Add(str(filename))
    # if collision_system == "embedPythia":
    if cross_check_task:
        friend_tree = ROOT.TChain("tree")
        for filename in input_filenames:
            friend_tree.Add(str(filename.parent.parent / "scale_factor" / filename.name))
        # Add friends with scale factors
        main_tree.AddFriend(friend_tree)

        # Could add the aliases here. However, they don't seem to propagate to the df, so we wait.
        # renames = cross_check_task_renames(grooming_method=grooming_method, input_branches=[b.GetName() for b in main_tree.GetListOfBranches()])
        # for k, v in renames.items():
        #    if not main_tree.SetAlias(k, v):
        #        raise RuntimeError(f"wat? {k}, {v}")

    # Keep the fully original DF so we can see everything applied to it.
    df_true_original = ROOT.RDataFrame(main_tree)
    df_original = df_true_original
    # df = ROOT.RDataFrame("AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl", "")

    if cross_check_task:
        # Add the aliases. This has to be done after the df is defined because apparently they don't carry over.
        renames = skim_analysis_objects.cross_check_task_branch_name_shim(
            grooming_method=grooming_method, input_branches=df_original.GetColumnNames()
        )
        for k, v in renames.items():
            df_original = df_original.Alias(k, v)

    # Scale factors _must_ be defined here, so we don't provide a fall back.

    # Apply general cuts.
    # Double counting must be applied for embedding.
    if collision_system == "embedPythia":
        double_counting_cut = f"(det_level_leading_track_pt >= hybrid_leading_track_pt_sub) && ({jet_pt_column_format.format(prefix='true')} >= 10)"
        df_original = df_original.Filter(double_counting_cut)
        smeared_cut_prefix = "hybrid"
    else:
        smeared_cut_prefix = "data"

    # For the substructure variables, we also apply a jet pt cut at the measured level (ie. data or hybrid).
    jet_pt_cut = f"{jet_pt_column_format.format(prefix=smeared_cut_prefix)} >= {main_jet_pt_range.min} && {jet_pt_column_format.format(prefix=smeared_cut_prefix)} < {main_jet_pt_range.max}"
    df_measured_selections = df_original.Filter(jet_pt_cut)

    # kt binning should vary with jet R because the kt kinematic limits vary with jet_R
    kt_axis = np.linspace(-1, 25, 26 + 1, dtype=np.float64)
    if jet_R == 0.2:
        # NOTE: Lower edge varies to ensure that we only have below 0.
        kt_axis = np.linspace(-0.5, 12, 25 + 1, dtype=np.float64)

    hists = []
    # Responses
    # General responses.
    # Hybrid-det level
    kt_hybrid_det_level_response = df_measured_selections.Histo2D(
        (
            f"{grooming_method}_hybrid_det_level_kt_response_jet_pt_{smeared_cut_prefix}_40_120",
            f"{grooming_method}_hybrid_det_level_kt_response_jet_pt_{smeared_cut_prefix}_40_120",
            len(kt_axis) - 1,
            kt_axis,
            len(kt_axis) - 1,
            kt_axis,
        ),
        f"{grooming_method}_hybrid_kt",
        f"{grooming_method}_det_level_kt",
        "scale_factor",
    )
    hists.append(kt_hybrid_det_level_response)
    # Hybrid-true
    kt_hybrid_true_response = df_measured_selections.Histo2D(
        (
            f"{grooming_method}_hybrid_true_kt_response_jet_pt_{smeared_cut_prefix}_40_120",
            f"{grooming_method}_hybrid_true_kt_response_jet_pt_{smeared_cut_prefix}_40_120",
            len(kt_axis) - 1,
            kt_axis,
            len(kt_axis) - 1,
            kt_axis,
        ),
        f"{grooming_method}_hybrid_kt",
        f"{grooming_method}_true_kt",
        "scale_factor",
    )
    hists.append(kt_hybrid_true_response)
    # Det-level true
    kt_det_level_true_response = df_measured_selections.Histo2D(
        (
            f"{grooming_method}_det_level_true_kt_response_jet_pt_{smeared_cut_prefix}_40_120",
            f"{grooming_method}_det_level_true_kt_response_jet_pt_{smeared_cut_prefix}_40_120",
            len(kt_axis) - 1,
            kt_axis,
            len(kt_axis) - 1,
            kt_axis,
        ),
        f"{grooming_method}_det_level_kt",
        f"{grooming_method}_true_kt",
        "scale_factor",
    )
    hists.append(kt_det_level_true_response)

    # Debug code for the RDF Filtering.
    # We explicitly require splittings at both the det level and hybrid level.
    # From here, we require a splitting at det level.
    # df = df.Filter(f"{grooming_method}_det_level_n_passed_grooming > 0 && {grooming_method}_hybrid_n_passed_grooming > 0")
    # extra = df.Filter(f"{grooming_method}_hybrid_det_level_matching_leading == 1 && {grooming_method}_hybrid_det_level_matching_subleading == 2").Count()
    # logger.debug(f"Extra: {extra.GetValue()}")

    # Matrix of possible counts values.
    # counts = {}
    # for leading_value in range(-1, 4):
    #    for subleading_value in range(-1, 4):
    #        counts[
    #            f"{grooming_method}_hybrid_det_level_matching_leading == {leading_value}"
    #            f" && {grooming_method}_hybrid_det_level_matching_subleading == {subleading_value}"
    #        ] = 0
    # for selection in counts:
    #    counts[selection] = df.Filter(selection).Count()
    ## Get the values:
    # for selection, values in counts.items():
    #    logger.info(f"Selection: {selection}: {values.GetValue()}")

    # Matching and matching dependent responses.
    # First, setup the dataframes. We only want to apply this complex filtering once since
    # it takes so many filtering calls.
    matching_dfs = {}
    for measured_like_prefix, generator_like_prefix in [("hybrid", "det_level"), ("det_level", "true")]:
        # NOTE: We explicitly require splittings at both the det level (generator-like) and hybrid level (measured-like).
        #       This excludes matching_leading and matching_subleading == 0.
        require_splittings_filter = f"({grooming_method}_{measured_like_prefix}_n_passed_grooming > 0) && ({grooming_method}_{generator_like_prefix}_n_passed_grooming > 0)"
        df_require_splittings = df_original.Filter(require_splittings_filter)

        matching_map: Dict[str, str] = {
            "all": "",
            "pure": f"({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_leading == 1)"
            f" && ({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_subleading == 1)",
            "leading_untagged_subleading_correct": f"({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_leading == 3)"
            f" && ({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_subleading == 1)",
            "leading_correct_subleading_untagged": f"({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_leading == 1)"
            f" && ({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_subleading == 3)",
            "leading_correct_subleading_mistag": f"({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_leading == 1)"
            f" && ({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_subleading == 2)",
            "leading_mistag_subleading_correct": f"({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_leading == 2)"
            f" && ({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_subleading == 1)",
            "leading_untagged_subleading_mistag": f"({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_leading == 3)"
            f" && ({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_subleading == 2)",
            "leading_mistag_subleading_untagged": f"({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_leading == 2)"
            f" && ({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_subleading == 3)",
            "swap": f"({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_leading == 2)"
            f" && ({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_subleading == 2)",
            "both_untagged": f"({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_leading == 3)"
            f" && ({grooming_method}_{measured_like_prefix}_{generator_like_prefix}_matching_subleading == 3)",
        }
        for matching_type, selection in matching_map.items():
            # Empty string will break the filter, so we need to only apply it if there is a valid selection.
            matching_dfs[
                MatchingIndex(
                    measured_like_prefix=measured_like_prefix,
                    generator_like_prefix=generator_like_prefix,
                    matching_type=matching_type,
                )
            ] = (
                df_require_splittings.Filter(selection) if selection else df_require_splittings
            )

    for matching_index, df_base in matching_dfs.items():
        # Setup
        # As of 21 September 2020, this is only for hybrid <-> det level
        create_generator_subjet_in_measured_jet_hists = matching_index.measured_like_prefix == "hybrid"

        # We explicitly require splittings at both the det level (generator-like) and hybrid level (measured-like).
        # This excludes matching_leading and matching_subleading == 0.
        # require_splittings_filter = f"{grooming_method}_{measured_like_prefix}_n_passed_grooming > 0 && {grooming_method}_{generator_like_prefix}_n_passed_grooming > 0"
        # df_selection = df.Filter(require_splittings_filter)

        # We want to match without the hybrid jet pt cut as a function of hybrid jet pt. So do that first before we cut.
        # Matching
        jet_pt_axis = np.arange(0, 151, dtype=np.float64)
        kt_selection_axis = np.array([-1, 2, 3, 5, 25], dtype=np.float64)
        if jet_R == 0.2:
            kt_selection_axis = np.array([-1, 0.5, 1, 1.5, 2, 12], dtype=np.float64)
        name = f"{grooming_method}_matching_{matching_index.matching_level}_type_{matching_index.matching_type}_jet_pt_axis_hybrid"
        h_matching_hybrid_jet_pt = df_base.Histo2D(
            (name, name, len(jet_pt_axis) - 1, jet_pt_axis, len(kt_selection_axis) - 1, kt_selection_axis),
            jet_pt_column_format.format(prefix="hybrid"),
            # Always want to make selections in the smeared (hybrid) kt
            f"{grooming_method}_hybrid_kt",
            "scale_factor",
        )
        hists.append(h_matching_hybrid_jet_pt)

        # Apply hybrid jet pt cut.
        tag = f"jet_pt_{smeared_cut_prefix}_40_120"
        jet_pt_cut = f"{jet_pt_column_format.format(prefix=smeared_cut_prefix)} >= {main_jet_pt_range.min} && {jet_pt_column_format.format(prefix=smeared_cut_prefix)} < {main_jet_pt_range.max}"
        df_selection = df_base.Filter(jet_pt_cut)

        hists.extend(
            new_matching_hists(
                df=df_selection,
                grooming_method=grooming_method,
                hist_suffix=tag,
                matching_index=matching_index,
                jet_pt_column_format=jet_pt_column_format,
                kt_axis=kt_axis,
                jet_pt_axis=jet_pt_axis,
                kt_selection_axis=kt_selection_axis,
                create_generator_subjet_in_measured_jet_hists=create_generator_subjet_in_measured_jet_hists,
            )
        )

        # Measured kt selections
        # for measured_min_kt in [-1, 2, 3, 5]:
        #    # kt == -1 is the cause that includes the untagged. There, we don't want to include any kt tag.
        #    selection_tag = tag
        #    kt_smeared_selection_filter = f"{grooming_method}_{smeared_cut_prefix}_kt > {measured_min_kt}"
        #    kt_selection_tag = ""
        #    if measured_min_kt == -1:
        #        df = df_selection
        #    else:
        #        df = df_selection.Filter(kt_smeared_selection_filter)
        #        kt_selection_tag = f"min_smeared_kt_{measured_min_kt}"
        #    selection_tag += f"_{kt_selection_tag}"
        #    hists.extend(
        #        matching_hists(
        #            df=df_selection,
        #            grooming_method=grooming_method,
        #            hist_suffix=selection_tag,
        #            general_selection="",
        #            measured_like_prefix=measured_like_prefix,
        #            generator_like_prefix=generator_like_prefix,
        #            jet_pt_column_format=jet_pt_column_format,
        #            create_generator_subjet_in_measured_jet_hists=create_generator_subjet_in_measured_jet_hists,
        #        )
        #    )
        #    # Only do this in the case where we're looking at hybrid-det level matching
        #    if measured_like_prefix == "hybrid":
        #        # Since we're looking at as a function of hybrid jet pt, we don't want to apply the hybrid jet pt cut.
        #        df_hybrid_matching_selection = df_original.Filter(
        #            f"({kt_smeared_selection_filter}) && ({require_splittings_filter})"
        #        )
        #        selection_tag = kt_selection_tag
        #        hists.extend(
        #            matching_hists(
        #                df=df_hybrid_matching_selection,
        #                grooming_method=grooming_method,
        #                hist_suffix=selection_tag,
        #                general_selection="",
        #                measured_like_prefix=measured_like_prefix,
        #                generator_like_prefix=generator_like_prefix,
        #                jet_pt_column_format=jet_pt_column_format,
        #                matching_jet_pt_prefix="hybrid",
        #                create_generator_subjet_in_measured_jet_hists=create_generator_subjet_in_measured_jet_hists,
        #            )
        #        )

        # For now, we skip because it slows down RDF to have more hists...
        ## n_groomed_to_split > 1
        # hists.extend(
        #    matching_hists(
        #        df=df_selection, grooming_method=grooming_method, hist_suffix = f"{measured_like_prefix}_n_groomed_to_split_greater_than_1", general_selection = f"{grooming_method}_{measured_like_prefix}_n_groomed_to_split > 1",
        #        matching_level=matching_level,
        #    )
        # )
        ## n_groomed_to_split < 2
        # hists.extend(
        #    matching_hists(
        #        df=df_selection, grooming_method=grooming_method, hist_suffix = f"{measured_like_prefix}_n_groomed_to_split_less_than_2", general_selection = f"{grooming_method}_{measured_like_prefix}_n_groomed_to_split < 2",
        #        matching_level=matching_level,
        #    )
        # )

        # n_to_split > 4
        # hist_suffix = f"{tag}_{generator_like_prefix}_n_to_split_greater_than_4"
        # hists.extend(
        #    matching_hists(
        #        df=df_selection,
        #        grooming_method=grooming_method,
        #        hist_suffix=hist_suffix,
        #        general_selection=f"{grooming_method}_{generator_like_prefix}_n_to_split > 4",
        #        measured_like_prefix=measured_like_prefix,
        #        generator_like_prefix=generator_like_prefix,
        #        jet_pt_column_format=jet_pt_column_format,
        #        create_generator_subjet_in_measured_jet_hists=create_generator_subjet_in_measured_jet_hists,
        #    )
        # )
        ## n_to_split < 3
        # hist_suffix = f"{tag}_{generator_like_prefix}_n_to_split_less_than_3"
        # hists.extend(
        #    matching_hists(
        #        df=df_selection,
        #        grooming_method=grooming_method,
        #        hist_suffix=hist_suffix,
        #        general_selection=f"{grooming_method}_{generator_like_prefix}_n_to_split < 3",
        #        measured_like_prefix=measured_like_prefix,
        #        generator_like_prefix=generator_like_prefix,
        #        jet_pt_column_format=jet_pt_column_format,
        #        create_generator_subjet_in_measured_jet_hists=create_generator_subjet_in_measured_jet_hists,
        #    )
        # )

    # If we want to save the dot graph. Unfortunately, it won't really be so insightful because we create many branches for the histograms.
    # ROOT.RDF.SaveGraph(df_true_original, "graph.dot")

    # Calculate the DataFrame by forcing it determine a property.
    # Discard the result - we don't really care. We just need a meaningless property.
    logger.info("Calculating DF...")
    hists[0].GetEntries()

    logger.info(f"Creating response output file for {collision_system}, {grooming_method}, {prefixes}")
    logger.info(f"Writing to {output_filename}")
    output = ROOT.TFile(str(output_filename), "RECREATE")
    output.cd()
    for h in hists:
        h.SetDirectory(output)
        # Why doesn't h.Write() work? Because ROOT. It fucking sucks.
        # h.Write()
    output.Write()
    # output.ls()
    output.Close()

    logger.info("Done!")

    return (True, "Processed")


def run(  # noqa: C901
    collision_system: str,
    input_filenames: Sequence[Path],
    tree_name: str,
    prefixes: Sequence[str],
    grooming_method: str,
    jet_R: float,
    main_jet_pt_range: helpers.JetPtRange,
    output_filename: Path,
    jet_pt_prefix_first: bool = False,
    n_cores: int = 8,
    cross_check_task: bool = False,
) -> Tuple[bool, str]:
    # Delay ROOT import so we don't explicitly rely on it.
    import ROOT

    # Setup for ROOT
    # Enable multithreading
    ROOT.ROOT.EnableImplicitMT(n_cores)
    # Sumw2
    ROOT.TH1.SetDefaultSumw2(True)

    # Parameters
    jet_pt_column_format = "jet_pt_{prefix}"
    if jet_pt_prefix_first:
        jet_pt_column_format = "{prefix}_jet_pt"

    # Setup tree
    main_tree = ROOT.TChain(tree_name)
    for filename in input_filenames:
        main_tree.Add(str(filename))
    if cross_check_task:
        friend_tree = ROOT.TChain("tree")
        for filename in input_filenames:
            friend_tree.Add(str(filename.parent.parent / "scale_factor" / filename.name))
        # Add friends with scale factors
        main_tree.AddFriend(friend_tree)

    df_original = ROOT.RDataFrame(main_tree)
    # df = ROOT.RDataFrame("AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl", "")

    if cross_check_task:
        # Add the aliases. This has to be done after the df is defined because apparently they don't carry over.
        renames = skim_analysis_objects.cross_check_task_branch_name_shim(
            grooming_method=grooming_method, input_branches=df_original.GetColumnNames()
        )
        for k, v in renames.items():
            df_original = df_original.Alias(k, v)

    # Add scale factor column with 1s if it doesn't exist yet.
    if "scale_factor" not in df_original.GetColumnNames():
        logger.info("Defining scale_factor column")
        df_original = df_original.Define("scale_factor", "1")

    # Apply general cuts.
    # Double counting must be applied for embedding.
    if collision_system == "embedPythia":
        double_counting_cut = f"(det_level_leading_track_pt >= hybrid_leading_track_pt_sub) && ({jet_pt_column_format.format(prefix='true')} >= 10)"
        # double_counting_cut = f"{jet_pt_column_format.format(prefix='true')} >= 10"
        df_original = df_original.Filter(double_counting_cut)
        smeared_cut_prefix = "hybrid"
    else:
        smeared_cut_prefix = "data"

    # For the substructure variables, we also apply a jet pt cut at the measured level (ie. data or hybrid).
    jet_pt_cut = f"{jet_pt_column_format.format(prefix=smeared_cut_prefix)} >= {main_jet_pt_range.min} && {jet_pt_column_format.format(prefix=smeared_cut_prefix)} < {main_jet_pt_range.max}"
    df_measured_selections = df_original.Filter(jet_pt_cut)

    hists = []
    jet_pt_axis = (28, 0, 140)
    _measured_min_kt_values: List[float] = [-1, 2, 3, 5]
    if jet_R == 0.2:
        _measured_min_kt_values = [-1, 0.5, 1, 1.5, 2]

    for measured_min_kt in _measured_min_kt_values:
        tag = f"_jet_pt_{smeared_cut_prefix}_{main_jet_pt_range.min}_{main_jet_pt_range.max}"
        # kt == -1 is the cause that includes the untagged. There, we don't want to include any kt tag.
        if measured_min_kt == -1:
            df = df_measured_selections
            tag += ""
        else:
            df = df_measured_selections.Filter(f"{grooming_method}_{smeared_cut_prefix}_kt > {measured_min_kt}")
            tag += f"_min_smeared_kt_{measured_min_kt}"

        # General substructure histograms.
        for prefix in prefixes:
            hists.extend(
                _substructure_hists(
                    df=df,
                    jet_pt_column_format=jet_pt_column_format,
                    jet_pt_axis=jet_pt_axis,
                    jet_R=jet_R,
                    prefix=prefix,
                    grooming_method=grooming_method,
                    tag=tag,
                    jet_pt_prefix=prefix,
                    include_stats_hist=True,
                )
            )

    # Some more specialized substructure variables.
    if collision_system == "embedPythia":
        # Plot all substructure variables, but instead with a constant generator level pt cut.
        # Apply generator level jet pt cut
        jet_pt_cut = (
            f"{jet_pt_column_format.format(prefix='true')} >= 40 && {jet_pt_column_format.format(prefix='true')} < 140"
        )
        tag = "_jet_pt_true_40_140"
        df = df_original.Filter(jet_pt_cut)

        for prefix in prefixes:
            hists.extend(
                _substructure_hists(
                    df=df,
                    jet_pt_column_format=jet_pt_column_format,
                    jet_pt_axis=jet_pt_axis,
                    jet_R=jet_R,
                    prefix=prefix,
                    grooming_method=grooming_method,
                    tag=tag,
                    jet_pt_prefix="true",
                )
            )

    # If we want to save the dot graph. Unfortunately, it won't really be so insightful because we create many branches for the histograms.
    # ROOT.RDF.SaveGraph(df)

    # Calculate the DataFrame by forcing it determine a property.
    # Discard the result - we don't really care. We just need a meaningless property.
    logger.info("Calculating DF...")
    hists[0].GetEntries()

    logger.info(f"Creating output file for {collision_system}, {grooming_method}, {prefixes}")
    output = ROOT.TFile(str(output_filename), "RECREATE")
    output.cd()
    for h in hists:
        h.SetDirectory(output)
        # Why doesn't h.Write() work? Because ROOT. It fucking sucks.
        # h.Write()
    output.Write()
    # output.ls()
    output.Close()

    logger.info("Done!")

    return (True, "Processed")


def run_standalone(
    collision_system: str,
    train_numbers: Sequence[int],
    tree_name: str,
    prefixes: Sequence[str],
    grooming_method: str,
    jet_R: float,
    main_jet_pt_range: helpers.JetPtRange,
    jet_pt_prefix_first: bool = False,
    n_cores: int = 8,
    cross_check_task: bool = False,
) -> Tuple[bool, str]:
    # Determine the filenames based on the train numbers and predefined path here.
    #base_path = Path("trains/") / collision_system / "{train_number}/skim/AnalysisResults.*.root"
    base_path = Path("trains/") / collision_system / "00{train_number}/skim/*.root"
    filenames = helpers.expand_wildcards_in_filenames(
        [Path(str(base_path).format(train_number=train_number)) for train_number in train_numbers]
    )

    # Output filename
    # Add the train dir into the output path name if we're processing single pt hard bins for embed pythia.
    # It's frustrating that this is necessary, but so it's ROOT - what else is new?
    base_filename = Path("output") / collision_system / "RDF" / "standalone"
    if len(train_numbers) == 1 and collision_system == "embedPythia":
        base_filename = base_filename / str(train_numbers[0])
    base_filename.mkdir(parents=True, exist_ok=True)
    output_filename = base_filename / f"{grooming_method}_{'_'.join(prefixes)}_closure.root"

    return run_response(
        collision_system=collision_system,
        input_filenames=filenames,
        tree_name=tree_name,
        prefixes=prefixes,
        grooming_method=grooming_method,
        jet_R=jet_R,
        main_jet_pt_range=main_jet_pt_range,
        output_filename=output_filename,
        jet_pt_prefix_first=jet_pt_prefix_first,
        n_cores=n_cores,
        cross_check_task=cross_check_task,
    )


def embed_pythia_entry_point() -> None:
    """Allow processing one pt hard bin at a time.

    Why? Because RDF has awful performance for jitted filter statements. See: https://root-forum.cern.ch/t/rdataframe-is-very-slow-for-many-histograms/37875/15
    """
    helpers.setup_logging()

    parser = argparse.ArgumentParser(description="Skim cross-check task using ROOT RDF.")

    parser.add_argument("-t", "--trainNumber", type=int)
    parser.add_argument("-g", "--groomingMethod", type=str)
    args = parser.parse_args()

    run_standalone(
        collision_system="embedPythia",
        # train_numbers=[args.trainNumber],
        train_numbers=list(range(5966, 5986)),
        # tree_name="AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
        tree_name="tree",
        prefixes=["hybrid"],
        grooming_method=args.groomingMethod,
        jet_R=0.4,
        main_jet_pt_range=helpers.JetPtRange(40, 120),
    )


if __name__ == "__main__":
    helpers.setup_logging()
    prefixes = ["hybrid", "true", "det_level"]
    run_standalone(
        collision_system="embedPythia",
        # train_numbers=list(range(5791, 5792)),
        # train_numbers=list(range(5966, 5968)),
        # train_numbers=list(range(6338, 6339)),
        train_numbers=[61],
        # train_numbers=list(range(6017, 6018)),
        # train_numbers=list(range(5988, 5989)),
        # tree_name="AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR020_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
        tree_name="tree",
        # prefix="det_level",
        prefixes=prefixes,
        grooming_method="dynamical_core",
        jet_R=0.2,
        main_jet_pt_range=helpers.JetPtRange(40, 120),
        n_cores=2,
        jet_pt_prefix_first=True,
        cross_check_task=False,
    )
    # for grooming_method in ["leading_kt", "leading_kt_z_cut_02", "leading_kt_z_cut_04", "dynamical_z", "dynamical_kt", "dynamical_time", "soft_drop_z_cut_02", "soft_drop_z_cut_04"]:
    # run_standalone(
    #    collision_system="PbPb",
    #    train_numbers=[5537],
    #    #tree_name="AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_Data_ConstSub_Incl",
    #    tree_name="tree",
    #    prefixes=["data"],
    #    #grooming_method=grooming_method,
    #    grooming_method="leading_kt",
    #    jet_R=0.4,
    #    main_jet_pt_range=helpers.JetPtRange(40, 120),
    #    n_cores=6,
    #    jet_pt_prefix_first=True,
    # )
