#!/usr/bin/env python3

""" Skim train output to a flat tree.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>
"""

from __future__ import annotations

import argparse
import functools
import logging
import operator
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, Union, cast

import attr
import enlighten
import IPython
import numpy as np
import uproot3
from pachyderm import yaml
from pathos.multiprocessing import ProcessingPool as Pool

from jet_substructure.analysis import analyze_tree
from jet_substructure.base import analysis_objects, data_manager, helpers, substructure_methods
from jet_substructure.base.helpers import UprootArray


logger = logging.getLogger(__name__)


@attr.s
class Calculation:
    """Similar to `FillHistogramInput`, but adds the splittings indices.

    Note:
        The splitting indices are the overall indices of the input splittings within
        the entire splittings array. The indices are those of the splittings selected
        by the calculation.
    """

    input_jets: substructure_methods.SubstructureJetArray = attr.ib()
    input_splittings: substructure_methods.JetSplittingArray = attr.ib()
    input_splittings_indices: UprootArray[int] = attr.ib()
    values: UprootArray[float] = attr.ib()
    indices: UprootArray[int] = attr.ib()
    # If there's no additional grooming selection, then this will be identical to input_splittings_indices.
    possible_indices: UprootArray[int] = attr.ib()

    @property
    def splittings(self) -> substructure_methods.JetSplittingArray:
        try:
            return self._restricted_splittings
        except AttributeError:
            self._restricted_splittings: substructure_methods.JetSplittingArray = self.input_splittings[self.indices]
        return self._restricted_splittings

    @property
    def n_jets(self) -> int:
        """ Number of jets. """
        # We flatten the splittings because there may be jets (and consequently splittings) which aren't selected
        # at all due to the grooming (such as a z cut). Thus, we use the selected splittings directly.
        return len(self.splittings.flatten())

    def __getitem__(self, mask: np.ndarray) -> Calculation:
        """ Mask the stored values, returning a new object. """
        # Validation
        if len(self.input_jets) != len(mask):
            raise ValueError(
                f"Mask length is different than array lengths. mask length: {len(mask)}, array lengths: {len(self.input_jets)}"
            )

        # Return the masked arrays in a new object.
        return type(self)(
            # NOTE: It's super important to use the input variables. Otherwise, we'll try to apply the indices twice
            #       (which won't work for the masked object).
            input_jets=self.input_jets[mask],
            input_splittings=self.input_splittings[mask],
            input_splittings_indices=self.input_splittings_indices[mask],
            values=self.values[mask],
            indices=self.indices[mask],
            possible_indices=self.possible_indices[mask],
        )


@attr.s
class GroomingResultForTree:
    grooming_method: str = attr.ib()
    delta_R: np.ndarray = attr.ib()
    z: np.ndarray = attr.ib()
    kt: np.ndarray = attr.ib()
    n_to_split: np.ndarray = attr.ib()
    n_groomed_to_split: np.ndarray = attr.ib()
    # For SoftDrop, this is equivalent to n_sd.
    n_passed_grooming: np.ndarray = attr.ib()

    def asdict(self, prefix: str) -> Iterable[Tuple[str, np.ndarray]]:
        for k, v in attr.asdict(self, recurse=False).items():
            # Skip the label
            if isinstance(v, str):
                continue
            yield "_".join([self.grooming_method, prefix, k]), v


def _define_calculation_functions(
    dataset: analysis_objects.Dataset,
    iterative_splittings: bool,
) -> Dict[str, functools.partial[Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]]]:
    """Define the calculation functions of interest.

    Note:
        The type of the inclusive is different, but it takes and returns the same sets of arguments
        as the other functions.

    Args:
        dataset: Dataset properties necessary to fully specify the calculations.
    Returns:
        dynamical_z, dynamical_kt, dynamical_time, leading_kt, leading_kt z>0.2, leading_kt z>0.4, SD z>0.2, SD z>0.4
    """
    functions = {
        "dynamical_z": functools.partial(substructure_methods.JetSplittingArray.dynamical_z, R=dataset.settings.jet_R),
        "dynamical_kt": functools.partial(
            substructure_methods.JetSplittingArray.dynamical_kt, R=dataset.settings.jet_R
        ),
        "dynamical_time": functools.partial(
            substructure_methods.JetSplittingArray.dynamical_time, R=dataset.settings.jet_R
        ),
        "leading_kt": functools.partial(
            substructure_methods.JetSplittingArray.leading_kt,
        ),
        "leading_kt_z_cut_02": functools.partial(substructure_methods.JetSplittingArray.leading_kt, z_cutoff=0.2),
        "leading_kt_z_cut_04": functools.partial(substructure_methods.JetSplittingArray.leading_kt, z_cutoff=0.4),
    }
    # TODO: This currently only works for iterative splittings...
    #       Calculating recursive is way harder in any array-like manner.
    if iterative_splittings:
        functions["soft_drop_z_cut_02"] = functools.partial(
            substructure_methods.JetSplittingArray.soft_drop, z_cutoff=0.2
        )
        functions["soft_drop_z_cut_04"] = functools.partial(
            substructure_methods.JetSplittingArray.soft_drop, z_cutoff=0.4
        )
    return functions


def _select_and_retrieve_splittings(
    jets: substructure_methods.SubstructureJetArray, mask: UprootArray[bool], iterative_splittings: bool
) -> Tuple[substructure_methods.SubstructureJetArray, substructure_methods.JetSplittingArray, UprootArray[int]]:
    """Generalization of the function in analyze_tree to add the splitting index."""
    restricted_jets, restricted_splittings = analyze_tree._select_and_retrieve_splittings(
        jets, mask, iterative_splittings
    )
    # Add the indices.
    if iterative_splittings:
        restricted_splittings_indices = restricted_jets.subjets.iterative_splitting_index
    else:
        restricted_splittings_indices = restricted_jets.splittings.kt.localindex
    return restricted_jets, restricted_splittings, restricted_splittings_indices


def calculate_splitting_number(
    all_splittings: substructure_methods.JetSplittingArray,
    selected_splittings: substructure_methods.JetSplittingArray,
    restricted_splittings_indices: UprootArray[int],
    debug: bool = False,
) -> np.ndarray:
    # logger.debug("Calculating splitting number")
    # Setup
    # We need the parent index of all of the splittings and of those which we have selected.
    # The restricted splittings aren't enough on their own because they may not contain all of
    # the necessary splitting history to reconstruct the splitting.
    all_splittings_parent_index = all_splittings.parent_index
    parent_index = selected_splittings.parent_index
    counts = np.zeros_like(all_splittings_parent_index, dtype=np.int)

    # First, increment all which have a selected splitting, meaning that if the splitting is at
    # the origin, it is the considered the 1st splitting (so we're reserving 0 for the untagged).
    counts[selected_splittings.counts > 0] += 1

    # The general procedure is that we will mask as true all parent_index != -1
    # If those pass all of the cuts (including that it is in the restricted splittings)
    # then we increment the count. Once a parent index gets to -1, then we stop selecting it
    # in our mask, so it stops being updated.
    # NOTE: In general, we don't want to iterative with these type of arrays, but it's
    #       unavoidable here. And I don't think it should loop more than 30-40 times in the
    #       worst case (and often much less).
    while True:
        # First, we need to determine if we're done. If so, all parent_index values will be -1.
        mask = parent_index != -1
        # Need two all() calls because the mask is jagged (with dim one of the jagged axis).
        if (mask != True).all().all():  # noqa: E712
            break
        # Need to repeat the parent_index to be the same shape as the restricted splittings so we can
        # check if any are equal. If any are equal, then that splitting is in the restricted group.
        # NOTE: We fill padded values with -2 because that can't possibly be a splitting index.
        parent_repeated_to_be_same_shape = (
            restricted_splittings_indices.ones_like() * parent_index.pad(1).fillna(-2).flatten()
        )
        accept_mask = (parent_repeated_to_be_same_shape == restricted_splittings_indices).any()

        if debug:
            IPython.start_ipython(user_ns=locals())
        # In the case that the parent_index of our splitting is in the selected splittings,
        # and it hasn't gotten to the origin, we can finally increment our count.
        # NOTE: Need to pad, fill, and flatten to match the shape of the accept_mask (which is just an ndarray mask)
        counts[mask.pad(1).fillna(False).flatten() & accept_mask] += 1
        # We retrieve the parents, and then assign them for those which are not yet at the origin.
        parent_index[mask] = all_splittings_parent_index[parent_index][mask]

    # logger.debug("Finished splitting number calculation")
    return counts


# def calculate_soft_drop(
#    all_splittings: substructure_methods.JetSplittingArray,
#    restricted_splittings_indices: UprootArray[int],
#    z_cutoff: float,
# ) -> Tuple[np.ndarray, UprootArray[int]]:
#    """
#
#    """
#    # TODO: Move to the splittings object.
#    # Start with the origin (NOTE: the relevant origin is 0 because -1 is a dummy node to start the splittings)
#    parent_index = all_splittings.localindex.zeros_like()
#    # Initial value should be outside of the standard range.
#    values = np.ones(len(all_splittings)) * substructure_methods.UNFILLED_VALUE
#    indices = all_splittings.localindex.ones_like() * -1
#    # The idea here is to iterate over the generations, starting at the origin.
#    while True:
#        # Select splittings that we ca
#        splittings_from_parent_mask = (all_splittings.parent_index == parent_index)
#        splittings = all_splittings[splittings_from_parent_mask]
#        pass_cutoff_mask = splittings.kt > z_cutoff
#
#        splittings_indices = all_splittings.localindex[splittings_from_parent_mask]
#        splittings_indices_needed_to_be_the_same_shape = restricted_splittings_indices.ones_like() * splittings_indices
#        restricted_mask = splittings_indices_needed_to_be_the_same_shape == restricted_splittings_indices
#        values[pass_cutoff_mask & restricted_mask] = splittings.z
#
#        parent_index = all_splittings.localindex[splittings.splittings_from_parent_mask].parent_index
#
#    return values, indices

# def calculate_soft_drop(
#    all_splittings: substructure_methods.JetSplittingArray,
#    restricted_splittings_indices: UprootArray[int],
#    z_cutoff: float,
# ) -> Tuple[np.ndarray, UprootArray[int]]:
#    # Start with the origin (NOTE: the relevant origin is 0 because -1 is a dummy node to start the splittings)
#    parent_index = all_splittings.localindex.zeros_like()
#    # Initial value should be outside of the standard range.
#    values = np.ones(len(all_splittings)) * substructure_methods.UNFILLED_VALUE
#    indices = all_splittings.localindex.ones_like() * -2
#    # The idea is to step through the generations of splittings.
#    #while True:
#    #    splittings_contributing_to_parent =
#    #    ...


def prong_matching(
    measured_like_jets_calculation: Calculation,
    measured_like_jets_label: str,
    generator_like_jets_calculation: Calculation,
    generator_like_jets_label: str,
    grooming_method: str,
    match_using_distance: bool = True,
) -> Dict[str, np.ndarray]:
    """Performs prong matching for the provided collections.

    Note:
        0 is there were insufficient constituents to form a splitting, 1 is properly matched, 2 is mistagged
        (leading -> subleading or subleading -> leading), 3 is untagged (failed).

    Args:
        measured_like_jets_calculation: Grooming calculation for measured-like jets (hybrid for hybrid-det level matching).
        measured_like_jets_label: Label for measured jets (hybrid for hybrid-det level matching).
        generator_like_jets_calculation: Grooming calculation for generator-like jets (det level for hybrid-det level matching).
        generator_like_jets_label: Label for generator jets (det_level for hybrid-det level matching).
        grooming_method: Name of the grooming method.
        match_using_distance: If True, match using distance. Otherwise, match using the stored label.
    Returns:
        Matching and subleading matching values.
    """
    # We can only perform matching if there are selected splittings.
    # Need to mask for calculations which have no indices (ie didn't find any that met criteria.
    mask = (generator_like_jets_calculation.indices.counts != 0) & (measured_like_jets_calculation.indices.counts != 0)
    try:
        masked_generator_like_jets_calculation = generator_like_jets_calculation[mask]
        masked_measured_like_jets_calculation = measured_like_jets_calculation[mask]
    except IndexError as e:
        logger.warning(e)
        IPython.start_ipython(user_ns=locals())

    # Matching
    grooming_results = {}
    logger.info(f"Performing {measured_like_jets_label}-{generator_like_jets_label} matching for {grooming_method}")
    leading_matching, subleading_matching = analyze_tree.determine_matched_jets(
        hybrid_inputs=analysis_objects.FillHistogramInput(
            jets=masked_measured_like_jets_calculation.input_jets,
            splittings=masked_measured_like_jets_calculation.input_splittings,
            values=masked_measured_like_jets_calculation.values,
            indices=masked_measured_like_jets_calculation.indices,
        ),
        matched_inputs=analysis_objects.FillHistogramInput(
            jets=masked_generator_like_jets_calculation.input_jets,
            splittings=masked_generator_like_jets_calculation.input_splittings,
            values=masked_generator_like_jets_calculation.values,
            indices=masked_generator_like_jets_calculation.indices,
        ),
        match_using_distance=match_using_distance,
    )
    # Store leading, subleading matches
    for label, matching in [("leading", leading_matching), ("subleading", subleading_matching)]:
        # We'll store the output in an array, and then store that in the overall output with a mask
        # We need the additional mask because we can't perform matching for every jet (single particle jets, etc).
        output = np.zeros(len(generator_like_jets_calculation.input_jets), dtype=np.int)
        matching_output = np.zeros(len(matching.properly), dtype=np.int)
        matching_output[matching.properly] = 1
        matching_output[matching.mistag] = 2
        matching_output[matching.failed] = 3
        output[mask] = matching_output
        grooming_results[
            f"{grooming_method}_{measured_like_jets_label}_{generator_like_jets_label}_matching_{label}"
        ] = output

    return grooming_results


def calculate_and_skim_embedding(  # noqa: C901
    tree: data_manager.Tree,
    dataset: analysis_objects.Dataset,
    iterative_splittings: bool,
    create_friend_tree: bool = False,
    draw_example_splittings: bool = False,
) -> bool:
    """Determine the response and prong matching for jets substructure techniques.

    Why combine them together? Because then we only have to open and process a tree once.
    At a future date (beyond the start of April 2020), it would be better to refactor them more separately,
    such that we can enable or disable the different options and still have appropriate return values.
    But for now, we don't worry about it.
    """
    # Validation
    prefixes = ["matched", "detLevel", "data"]
    # Setup
    # Perhaps make these into arguments?
    iterative_splittings_label = "iterative" if iterative_splittings else "recursive"
    # TODO: Maybe convert to hdf5? But maybe not because of compression?
    output_dir = tree.filename.parent / "skim"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{tree.filename.stem}_{iterative_splittings_label}_splittings.root"

    # Use the train configuration to extract the train number and pt hard bin, which are used to get the scale factor.
    y = yaml.yaml()
    with open(tree.filename.parent / "config.yaml", "r") as f:
        train_config = y.load(f)
    train_number = train_config["number"]
    pt_hard_bin = train_config["pt_hard_bin"]
    logger.debug(f"Extracted train number: {train_number}, pt hard bin: {pt_hard_bin}")
    analysis_settings = cast(analysis_objects.PtHardAnalysisSettings, dataset.settings)
    scale_factor = analysis_settings.scale_factors[pt_hard_bin]

    # Actual setup.
    logger.info(f"Skimming tree from file {tree.filename}")
    successfully_accessed_data, all_jets = analyze_tree.load_jets_from_tree(tree=tree, prefixes=prefixes)
    true_jets, det_level_jets, hybrid_jets = all_jets
    if not successfully_accessed_data:
        return False

    ## Do the calculations
    # Do not mask on the number of constituents. This would prevent tagged <-> untagged migrations in the response.
    # mask = (
    #    (true_jets.constituents.counts > 1)
    #    & (det_level_jets.constituents.counts > 1)
    #    & (hybrid_jets.constituents.counts > 1)
    # )
    # Require that we have jets that aren't dominated by hybrid jets.
    # It's super important to be ">=". That allows the leading jet in the hybrid to be the same
    # as the leading jet in the true (which would be good - we've probably found the right jet).
    # NOTE: We already apply this cut at the analysis level, so it shouldn't really do anything here.
    #       We're just applying it again to be certain.
    # NOTE: As of 7 May 2020, we skip this cut at the analysis level, so it's super important to
    #       apply it here.
    # NOTE: As of 19 May 2019, we disable this cut event though it's not applied at the analysis level.
    #       This will allow L+L to study this at the analysis level.
    # mask = mask & (det_level_jets.constituents.max_pt >= hybrid_jets.constituents.max_pt)
    mask = np.ones_like(hybrid_jets.jet_pt) > 0

    # Mask the jets
    masked_true_jets, masked_true_jet_splittings, masked_true_jet_splittings_indices = _select_and_retrieve_splittings(
        true_jets, mask, iterative_splittings
    )
    (
        masked_det_level_jets,
        masked_det_level_jet_splittings,
        masked_det_level_jet_splittings_indices,
    ) = _select_and_retrieve_splittings(det_level_jets, mask, iterative_splittings)
    (
        masked_hybrid_jets,
        masked_hybrid_jet_splittings,
        masked_hybrid_jet_splittings_indices,
    ) = _select_and_retrieve_splittings(hybrid_jets, mask, iterative_splittings)

    grooming_results = {}
    if create_friend_tree:
        # Extract eta-phi of jets.
        output_filename = Path(str(output_filename.with_suffix("")) + "_friend.root")
        # As the skim is re-run, values are generally transitioned to the standard tree the next time it's generated.
    else:
        grooming_results["scale_factor"] = np.ones_like(true_jets.jet_pt[mask]) * scale_factor
        # Add jet pt for all prefixes.
        grooming_results["jet_pt_true"] = masked_true_jets.jet_pt
        grooming_results["jet_pt_det_level"] = masked_det_level_jets.jet_pt
        grooming_results["jet_pt_hybrid"] = masked_hybrid_jets.jet_pt
        # Add jet eta phi.
        for prefix, jets in zip(prefixes, [masked_true_jets, masked_det_level_jets, masked_hybrid_jets]):
            jet_four_vec = jets.constituents.four_vectors().sum()
            grooming_results[f"jet_eta_{prefix}"] = jet_four_vec.eta
            grooming_results[f"jet_phi_{prefix}"] = jet_four_vec.phi
        # Leading track
        grooming_results["leading_track_true"] = masked_true_jets.leading_track_pt
        grooming_results["leading_track_det_level"] = masked_det_level_jets.leading_track_pt
        grooming_results["leading_track_hybrid"] = masked_hybrid_jets.leading_track_pt

        # Perform our calculations.
        functions = _define_calculation_functions(dataset, iterative_splittings=iterative_splittings)
        for func_name, func in functions.items():
            true_jets_calculation = Calculation(
                masked_true_jets,
                masked_true_jet_splittings,
                masked_true_jet_splittings_indices,
                *func(masked_true_jet_splittings),
            )
            det_level_jets_calculation = Calculation(
                masked_det_level_jets,
                masked_det_level_jet_splittings,
                masked_det_level_jet_splittings_indices,
                *func(masked_det_level_jet_splittings),
            )
            hybrid_jets_calculation = Calculation(
                masked_hybrid_jets,
                masked_hybrid_jet_splittings,
                masked_hybrid_jet_splittings_indices,
                *func(masked_hybrid_jet_splittings),
            )

            for prefix, calculation in [
                ("true", true_jets_calculation),
                ("det_level", det_level_jets_calculation),
                ("hybrid", hybrid_jets_calculation),
            ]:
                # Calculate splitting number for the appropriate cases.
                groomed_splittings = calculation.splittings
                # Number of splittings until the selected splitting, irrespective of the grooming conditions.
                n_to_split = calculate_splitting_number(
                    all_splittings=calculation.input_jets.splittings,
                    selected_splittings=groomed_splittings,
                    # Need all splitting indices (unrestricted by any possible grooming selections).
                    restricted_splittings_indices=calculation.input_splittings_indices,
                )
                # Number of splittings which pass the grooming conditions until the selected splitting.
                n_groomed_to_split = calculate_splitting_number(
                    all_splittings=calculation.input_jets.splittings,
                    selected_splittings=groomed_splittings,
                    # Need the indices that correspond to the splittings that pass the grooming.
                    restricted_splittings_indices=calculation.possible_indices,
                )

                # We pad with the UNFILLED_VALUE constant to account for any calculations that don't find a splitting.
                grooming_result = GroomingResultForTree(
                    grooming_method=func_name,
                    delta_R=groomed_splittings.delta_R.pad(1).fillna(substructure_methods.UNFILLED_VALUE).flatten(),
                    z=groomed_splittings.z.pad(1).fillna(substructure_methods.UNFILLED_VALUE).flatten(),
                    kt=groomed_splittings.kt.pad(1).fillna(substructure_methods.UNFILLED_VALUE).flatten(),
                    # All of the numbers are already flattened. 0 means untagged.
                    n_to_split=n_to_split,
                    n_groomed_to_split=n_groomed_to_split,
                    # Number of splittings which pass the grooming condition. For SoftDrop, this is n_sd.
                    n_passed_grooming=calculation.possible_indices.counts,
                )
                grooming_results.update(grooming_result.asdict(prefix=prefix))

            # Hybrid-det level matching.
            # We match using distance here because the labels don't align anymore due to the subtraction mixing the labels.
            hybrid_det_level_matching_results = prong_matching(
                measured_like_jets_calculation=hybrid_jets_calculation,
                measured_like_jets_label="hybrid",
                generator_like_jets_calculation=det_level_jets_calculation,
                generator_like_jets_label="det_level",
                grooming_method=func_name,
                match_using_distance=False,
            )
            grooming_results.update(hybrid_det_level_matching_results)
            # Det level-true matching
            # We match using labels here because otherwise the reconstruction can cause the particles to move
            # enough that they may not match within a particular distance.
            det_level_true_matching_results = prong_matching(
                measured_like_jets_calculation=det_level_jets_calculation,
                measured_like_jets_label="det_level",
                generator_like_jets_calculation=true_jets_calculation,
                generator_like_jets_label="true",
                grooming_method=func_name,
                match_using_distance=False,
            )
            grooming_results.update(det_level_true_matching_results)

            # Look for leading kt just because it's easier to understand conceptually.
            hybrid_det_level_leading_matching = grooming_results[f"{func_name}_hybrid_det_level_matching_leading"]
            hybrid_det_level_subleading_matching = grooming_results[f"{func_name}_hybrid_det_level_matching_subleading"]
            if (
                draw_example_splittings
                and func_name == "leading_kt"
                and (hybrid_det_level_leading_matching.properly & hybrid_det_level_subleading_matching.failed).any()
            ):
                from jet_substructure.analysis import draw_splitting

                # Find a sufficiently interesting jet (ie high enough pt)
                mask_jets_of_interest = (
                    (hybrid_det_level_leading_matching.properly & hybrid_det_level_subleading_matching.failed)
                    & (masked_hybrid_jets.jet_pt > 80)
                    & (det_level_jets_calculation.splittings.kt > 10).flatten()
                )

                # Look at most the first 5 jets.
                for i, hybrid_jet in enumerate(masked_hybrid_jets[mask_jets_of_interest][:5]):
                    # Find the hybrid jet and splitting of interest.
                    # hybrid_jet = masked_hybrid_jets[mask_jets_of_interest][0]
                    # Take the index of the splitting of interest. We want the first jet, and then there must be one splitting index there.
                    hybrid_jet_selected_splitting_index = hybrid_jets_calculation.indices[mask_jets_of_interest][i][0]
                    # Same for det level.
                    det_level_jet = masked_det_level_jets[mask_jets_of_interest][i]
                    # Take the index of the splitting of interest. We want the first jet, and then there must be one splitting index there.
                    det_level_jet_selected_splitting_index = det_level_jets_calculation.indices[mask_jets_of_interest][
                        i
                    ][0]

                    # Draw the splittings
                    draw_splitting.splittings_graph(
                        jet=hybrid_jet,
                        path=dataset.output.parent / "leading_correct_subleading_failed/",
                        filename=f"{i}_hybrid_splittings_jet_pt_{hybrid_jet.jet_pt:.1f}GeV_selected_splitting_index_{hybrid_jet_selected_splitting_index}",
                        show_subjet_pt=True,
                        selected_splitting_index=hybrid_jet_selected_splitting_index,
                    )
                    draw_splitting.splittings_graph(
                        jet=det_level_jet,
                        path=dataset.output.parent / "leading_correct_subleading_failed/",
                        filename=f"{i}_det_level_splittings_jet_pt_{det_level_jet.jet_pt:.1f}GeV_selected_splitting_index_{det_level_jet_selected_splitting_index}",
                        show_subjet_pt=True,
                        selected_splitting_index=det_level_jet_selected_splitting_index,
                    )

    branches = {k: v.dtype for k, v in grooming_results.items()}
    logger.info(f"Writing skim to {output_filename}")
    with uproot3.recreate(output_filename) as output_file:
        output_file["tree"] = uproot3.newtree(branches)
        # Write all of the calculations
        output_file["tree"].extend(grooming_results)

    logger.info(f"Finished processing tree from file {tree.filename}")
    return True


def calculate_and_skim_data(
    tree: data_manager.Tree,
    dataset: analysis_objects.Dataset,
    iterative_splittings: bool,
    prefixes: Optional[Sequence[str]] = None,
) -> bool:
    # Validation
    if prefixes is None:
        prefixes = ["data"]

    # Setup
    # Perhaps make these into arguments?
    iterative_splittings_label = "iterative" if iterative_splittings else "recursive"
    output_dir = tree.filename.parent / "skim"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = output_dir / f"{tree.filename.stem}_{iterative_splittings_label}_splittings.root"

    # Actual setup.
    logger.info(f"Skimming tree from file {tree.filename}")
    successfully_accessed_data, all_jets = analyze_tree.load_jets_from_tree(tree=tree, prefixes=prefixes)
    if not successfully_accessed_data:
        return False
    # Only unpack if we've successfully accessed the data.

    # Dataset wide masks
    # Select everything by default.
    mask = np.ones_like(all_jets[0].jet_pt) > 0

    # Special selections for pythia.
    # Apparently I can get pt hard < 5. Which is bizarre. Filter these out when applicable.
    if dataset.collision_system == "pythia":
        mask = mask & (tree["ptHard"] >= 5.0)

    masked_jets: Dict[
        str, Tuple[substructure_methods.SubstructureJetArray, substructure_methods.JetSplittingArray, UprootArray[int]]
    ] = {}
    for prefix, input_jets in zip(prefixes, all_jets):
        masked_jets[prefix] = _select_and_retrieve_splittings(
            input_jets,
            mask,
            iterative_splittings=iterative_splittings,
        )

    # Results output
    grooming_results: Dict[str, np.ndarray] = {}
    # Add jet pt for all prefixes.
    for prefix, jets in masked_jets.items():
        grooming_results[f"jet_pt_{prefix}"] = jets[0].jet_pt

    # Add scale factors when appropriate (ie for pythia)
    if dataset.collision_system == "pythia":
        # Need to redo the pt hard bin ranges because we apparently get a pt hard larger than 1000!
        # So we do it again here.
        # pt_hard_bin_ranges = np.array([0, 5, 7, 9, 12, 16, 21, 28, 36, 45, 57, 70, 85, 99, 115, 132, 150, 169, 190, 212, 235, 2000])
        # Need to subtract 1 because our first bin is 1-indexed.
        # pt_hard_bins = np.searchsorted(pt_hard_bin_ranges, tree["ptHard"]) - 1
        # print(np.unique(pt_hard_bins))
        analysis_settings = cast(analysis_objects.PtHardAnalysisSettings, dataset.settings)
        scale_factors = dict(analysis_settings.scale_factors)
        # There is apparently a pt hard > 1000 in this dataset! This ends up with an entry in bin 21, which is weird.
        # So we copy the scale factor for pt hard bin 20 to 21 to cover it. It should be more or less correct.
        scale_factors[21] = scale_factors[20]

        pt_hard_bins = tree["ptHardBin"][mask]
        logger.debug(f"Pt hard bins contained in the file: {np.unique(pt_hard_bins)}")
        grooming_results.update(
            {
                "scale_factor": np.array([scale_factors[b] for b in pt_hard_bins], dtype=np.float32),
                "pt_hard_bin": pt_hard_bins,
                "pt_hard": tree["ptHard"][mask],
            }
        )

    # Perform our calculations.
    functions = _define_calculation_functions(dataset, iterative_splittings=iterative_splittings)
    for func_name, func in functions.items():
        for prefix, jets in masked_jets.items():
            # Setup
            debug = False
            # if func_name == "leading_kt_z_cut_02":
            #    debug = True

            # Perform the grooming calculation
            values, indices, possible_indices = func(jets[1])
            calculation = Calculation(jets[0], jets[1], jets[2], values, indices, possible_indices)

            # Calculate splitting number for the appropriate cases.
            groomed_splittings = calculation.splittings
            # Number of splittings until the selected splitting, irrespective of the grooming conditions.
            n_to_split = calculate_splitting_number(
                all_splittings=calculation.input_jets.splittings,
                selected_splittings=groomed_splittings,
                # Need all splitting indices (unrestricted by any possible grooming selections).
                restricted_splittings_indices=calculation.input_splittings_indices,
                debug=debug,
            )
            # Number of splittings which pass the grooming conditions until the selected splitting.
            n_groomed_to_split = calculate_splitting_number(
                all_splittings=calculation.input_jets.splittings,
                selected_splittings=groomed_splittings,
                # Need the indices that correspond to the splittings that pass the grooming.
                restricted_splittings_indices=calculation.possible_indices,
                debug=debug,
            )

            # We pad with the UNFILLED_VALUE constant to account for any calculations that don't find a splitting.
            grooming_result = GroomingResultForTree(
                grooming_method=func_name,
                delta_R=groomed_splittings.delta_R.pad(1).fillna(substructure_methods.UNFILLED_VALUE).flatten(),
                z=groomed_splittings.z.pad(1).fillna(substructure_methods.UNFILLED_VALUE).flatten(),
                kt=groomed_splittings.kt.pad(1).fillna(substructure_methods.UNFILLED_VALUE).flatten(),
                # All of the numbers are already flattened. 0 means untagged.
                n_to_split=n_to_split,
                n_groomed_to_split=n_groomed_to_split,
                # Number of splittings which pass the grooming condition. For SoftDrop, this is n_sd.
                n_passed_grooming=calculation.possible_indices.counts,
            )
            # Cross check
            mask = grooming_result.kt < 0
            logger.info(
                f"Tagged fraction for {prefix}: {func_name}: {1 - (len(grooming_result.kt[mask])/(len(grooming_result.kt)))}"
            )

            # Store the results.
            grooming_results.update(grooming_result.asdict(prefix=prefix))

            # if func_name == "leading_kt":
            #     IPython.embed()

    branches = {k: v.dtype for k, v in grooming_results.items()}
    logger.info(f"Writing skim to {output_filename}")
    with uproot3.recreate(output_filename) as output_file:
        output_file["tree"] = uproot3.newtree(branches)
        # Write all of the calculations
        output_file["tree"].extend(grooming_results)

    logger.info(f"Finished processing tree from file {tree.filename}")
    return True


def run(
    collision_system: str,
    iterative_splittings: bool,
    calculate_and_skim_func: Callable[[data_manager.Tree, analysis_objects.Dataset, bool], bool],
    number_of_cores: int,
    additional_kwargs_for_analysis: Optional[Mapping[str, Any]] = None,
    override_filenames: Optional[Sequence[Union[str, Path]]] = None,
) -> None:
    # Validation
    if additional_kwargs_for_analysis is None:
        additional_kwargs_for_analysis = {}
    if override_filenames is None:
        override_filenames = []

    # Setup
    settings_class_map: Mapping[str, Type[analysis_objects.AnalysisSettings]] = {
        "pythia": analysis_objects.PtHardAnalysisSettings,
        "embedPythia": analysis_objects.PtHardAnalysisSettings,
    }
    dataset = analysis_objects.Dataset.from_config_file(
        collision_system=collision_system,
        config_filename=Path("config") / "datasets.yaml",
        override_filenames=None,
        hists_filename_stem="IGNORE",
        output_base=Path("output"),
        settings_class=settings_class_map.get(collision_system, analysis_objects.AnalysisSettings),
        # NOTE: This value is irrelevant for the skim...
        z_cutoff=-5,
    )

    dm = data_manager.IterateTrees(
        filenames=(override_filenames if override_filenames else dataset.filenames),
        tree_name=dataset.tree_name,
        # Mypy is getting confused by Sequence[str] because str is an iterable, so we ignore the type...
        branches=dataset.branches,  # type: ignore
    )
    logger.info("Setup complete. Beginning processing of trees.")

    progress_manager = enlighten.get_manager()
    number_of_trees_processed = 0
    dm_iterator = dm.lazy_iteration(fully_lazy=(number_of_cores > 1))
    wrapper = functools.partial(
        calculate_and_skim_func,
        dataset=dataset,
        iterative_splittings=iterative_splittings,
        **additional_kwargs_for_analysis,
    )
    wrapper_multiprocessing = functools.partial(
        analyze_tree._wrap_multiprocessing,
        analysis_function=wrapper,
    )
    with progress_manager.counter(total=len(dm), desc="Skimming", unit="tree") as tree_counter:
        if number_of_cores > 1:
            with Pool(nodes=number_of_cores) as pool:
                number_of_trees_processed = functools.reduce(
                    operator.add,
                    tree_counter(pool.imap(wrapper_multiprocessing, dm_iterator)),
                )
        else:
            number_of_trees_processed = functools.reduce(
                operator.add,
                tree_counter(map(wrapper, dm_iterator)),
            )

    logger.info(f"Processed {number_of_trees_processed} out of {len(dm)} trees!")

    # Cleanup
    progress_manager.stop()


def parse_arguments() -> Tuple[str, List[Path], bool]:
    parser = argparse.ArgumentParser(description="Skim provided files in a given dataset.")

    parser.add_argument("-d", "--datasetName", type=str)
    parser.add_argument("-f", "--filenames", nargs="+", default=[])
    parser.add_argument("-r", "--recursiveSplittings", action="store_true", default=False)
    args = parser.parse_args()
    # Validation for filenames
    filenames = [Path(f) for f in args.filenames]
    return args.datasetName, filenames, args.recursiveSplittings


def skim_entry_point() -> None:
    helpers.setup_logging()
    dataset_name, override_filenames, use_recursive_splittings = parse_arguments()
    number_of_cores = 1
    iterative_splittings = not use_recursive_splittings

    logger.info(f"Processing {dataset_name} with filenames: {override_filenames}")

    if dataset_name == "embedPythia":
        run(
            collision_system=dataset_name,
            iterative_splittings=iterative_splittings,
            calculate_and_skim_func=calculate_and_skim_embedding,
            number_of_cores=number_of_cores,
            additional_kwargs_for_analysis={"draw_example_splittings": False},
            override_filenames=override_filenames,
        )
    else:
        additional_kwargs_for_analysis = {}
        if dataset_name == "pythia":
            additional_kwargs_for_analysis = {"prefixes": ["data", "matched"]}

        # Run PbPb, pp, pythia
        run(
            collision_system=dataset_name,
            iterative_splittings=iterative_splittings,
            calculate_and_skim_func=calculate_and_skim_data,
            number_of_cores=number_of_cores,
            additional_kwargs_for_analysis=additional_kwargs_for_analysis,
            override_filenames=override_filenames,
        )

    logger.info(f"Finished processing skim for dataset {dataset_name} with files: {override_filenames}")


if __name__ == "__main__":
    helpers.setup_logging()
    # Options
    iterative_splittings = True
    number_of_cores = 1

    # Run embedding
    # run(
    #    collision_system="embedPythia",
    #    iterative_splittings=iterative_splittings,
    #    calculate_and_skim_func=calculate_and_skim_embedding,
    #    number_of_cores=number_of_cores,
    #    additional_kwargs_for_analysis={"create_friend_tree": False, "draw_example_splittings": False},
    # )
    # Run pp
    # run(
    #    collision_system="pp",
    #    iterative_splittings=iterative_splittings,
    #    calculate_and_skim_func=calculate_and_skim_data,
    #    number_of_cores=number_of_cores,
    # )
    # Run PbPb
    # run(
    #    collision_system="PbPb",
    #    iterative_splittings=iterative_splittings,
    #    calculate_and_skim_func=calculate_and_skim_data,
    #    number_of_cores=number_of_cores,
    # )
    run(
        collision_system="pythia",
        iterative_splittings=iterative_splittings,
        # mypy apparently doesn't handle adding arguments, even with callable protocols...
        # We only get away with this because the prefixes are optional.
        calculate_and_skim_func=calculate_and_skim_data,
        number_of_cores=number_of_cores,
        additional_kwargs_for_analysis={"prefixes": ["data", "matched"]},
    )
