#!/usr/bin/env python3

""" Analyze the dynamical grooming tree.

.. codeauthor:: Raymnod Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import argparse
import functools
import gzip
import logging
import pickle
import zlib
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type, TypeVar, Union, cast

import attr
import awkward0 as ak0
import enlighten
import IPython
import numpy as np
from pachyderm import binned_data, yaml
from pathos.multiprocessing import ProcessingPool as Pool

from jet_substructure.base import analysis_objects, data_manager, helpers, substructure_methods
from jet_substructure.base.helpers import UprootArray


logger = logging.getLogger(__name__)

_T = TypeVar("_T", bound=analysis_objects.SubstructureHists)


@attr.s
class SubstructureResult:
    name: str = attr.ib()
    title: str = attr.ib()
    values: UprootArray[float] = attr.ib()
    indices: UprootArray[int] = attr.ib()
    subjet: substructure_methods.JetSplittingArray
    # TODO: Need to store iterative splitting information somehow!
    #       Perhaps just below...?

    @property
    def splitting_number(self) -> UprootArray[int]:
        try:
            return self._splitting_number
        except AttributeError:
            # +1 because splittings counts from 1, but indexing starts from 0.
            splitting_number = self.indices + 1
            # If there were no splittings, we want to set that to 0.
            splitting_number = splitting_number.pad(1).fillna(0)
            # Must flatten because the indices are still jagged.
            self._splitting_number: UprootArray[int] = splitting_number.flatten()

        return self._splitting_number

    @splitting_number.setter
    def splitting_number(self, value: UprootArray[int]) -> None:
        self._splitting_number = value


def setup_yaml() -> yaml.ruamel.yaml.YAML:
    return yaml.yaml(modules_to_register=[binned_data, analysis_objects, helpers])


def _convert_and_write_hists(
    hists: Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]],
    tree_filename: Path,
    yaml_filename: Path,
    y: yaml.ruamel.yaml.YAML,
) -> Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]]:
    # Convert to BinnedData and store the hists
    for h in hists.values():
        h.convert_boost_histograms_to_binned_data()
    with open(yaml_filename, "w") as f:
        logger.info(f"Writing hists of the tree {tree_filename} to {yaml_filename}")
        y.dump(hists, f)

    return hists


def _construct_jets_from_tree(
    prefix: str,
    tree: data_manager.Tree,
) -> substructure_methods.SubstructureJetArray:
    """Construct the substructure jet objects for data stored under a given prefix in a tree.

    Ideally, the object has already been created and stored. If not, it will be created and then
    stored in the tree for the future (where retrieving the created object from a file is far faster).

    Args:
        prefix: Prefix under which the data of interest is stored.
        tree: Tree where the data is stored.
    Returns:
        Constructed jet object.
    """
    constructed_name = f"{prefix}_constructed"
    if constructed_name in tree:
        logger.debug("Using fully constructed object")
        jets = cast(substructure_methods.SubstructureJetArray, tree[constructed_name])
        # Check whether the JaggedArrays which are stored in the file were constructed properly
        try:
            jets.constituents.max_pt
        except AttributeError:
            jets = substructure_methods.SubstructureJetArray._from_serialization(
                jet_pt=jets.jet_pt,
                jet_constituents=jets.constituents,
                subjets=jets.subjets,
                jet_splittings=jets.splittings,
            )
    else:
        logger.debug("Constructing object")
        jets = substructure_methods.SubstructureJetArray.from_tree(tree, prefix=prefix)

        # Save calculate columns so we don't need to re-calculate them every time.
        # NOTE: We always check if they already exist because HDF5 doesn't like us
        #       overwriting columns.
        # Calculated subjet constituents.
        name = f"{prefix}.fSubjets.constituents"
        if name not in tree:
            tree[name] = jets.subjets.constituents

        # Store the full treee in h5.
        # This provides a huge speed up in terms of processing speed!
        if constructed_name not in tree:
            tree[constructed_name] = jets

    # Flush the hdf5 portion of the tree to ensure that it's been written properly.
    # Otherwise, the file may end up corrupted.
    tree._hdf5_tree.flush()

    return jets


def load_jets_from_tree(
    tree: data_manager.Tree, prefixes: Sequence[str]
) -> Tuple[bool, Tuple[substructure_methods.SubstructureJetArray, ...]]:
    """Create jets from a tree with given prefix(es).

    Args:
        tree: Input tree.
        prefix: Prefix of branches under which the jets are stored in the tree. The jets
            are returned in the order of the prefixes.
    Returns:
        Successfully created the jets, (substructure jet arrays in the same order as the prefixes)
    """
    # Setup
    successfully_accessed_data = False
    return_jets = []
    logger.debug(f"Accessing data from the tree {tree.filename}.")

    # Attempt to retrieve and construct the requested jets.
    try:
        # If there are 0 entries, then just return - it won't work...
        if len(tree) > 0:
            for prefix in prefixes:
                logger.debug(f"Constructing {prefix} jets")
                return_jets.append(_construct_jets_from_tree(prefix=prefix, tree=tree))
            successfully_accessed_data = True
        else:
            logger.warning(f"No jets are in file {tree.filename}. Skipping")
    except zlib.error as e:
        logger.warning(f"Issue reading the data: {e}. Skipping")

    return successfully_accessed_data, tuple(return_jets)


def _calculate_inclusive(
    splittings: substructure_methods.JetSplittingArray,
) -> Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]:
    """Calculate the inclusive splittings.

    Note:
        This is a fake calculation to provide the same type of output as the actual calculations for compatibility.
        The values in the first element of the return tuple aren't really meaningful. They're just included to
        match the shape of the indices. The second set of values are the indices of all splittings.

    Args:
        splittings: The jet splittings to process.
    Returns:
        Ones in the same length as the indices, indices for all splittings.
    """
    return splittings.kt.ones_like().flatten(), splittings.localindex, splittings.localindex


def _define_calculation_funcs(
    dataset: analysis_objects.Dataset,
) -> Tuple[
    functools.partial[Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]],
    functools.partial[Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]],
    functools.partial[Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]],
    functools.partial[Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]],
    functools.partial[Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]],
    functools.partial[Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]],
]:
    """Define the calculation functions of interest.

    Note:
        The type of the inclusive is different, but it takes and returns the same sets of arguments
        as the other functions.

    Args:
        dataset: Dataset properties necessary to fully specify the calculations.
    Returns:
        Inclusive, dynamical_z, dynamical_kt, dynamical_time, leading_kt, leading_kt_hard_cutoff
    """
    inclusive = functools.partial(_calculate_inclusive)
    dynamical_z_func = functools.partial(substructure_methods.JetSplittingArray.dynamical_z, R=dataset.settings.jet_R)
    dynamical_kt_func = functools.partial(substructure_methods.JetSplittingArray.dynamical_kt, R=dataset.settings.jet_R)
    dynamical_time_func = functools.partial(
        substructure_methods.JetSplittingArray.dynamical_time, R=dataset.settings.jet_R
    )
    leading_kt_func = functools.partial(
        substructure_methods.JetSplittingArray.leading_kt,
    )
    leading_kt_hard_cutoff_func = functools.partial(
        substructure_methods.JetSplittingArray.leading_kt, z_cutoff=dataset.settings.z_cutoff
    )
    return (
        inclusive,
        dynamical_z_func,
        dynamical_kt_func,
        dynamical_time_func,
        leading_kt_func,
        leading_kt_hard_cutoff_func,
    )


def _fill_substructure_hists_with_calculation(
    calculation: functools.partial[Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]],
    fill_attr_name: str,
    restricted_jets: substructure_methods.SubstructureJetArray,
    restricted_jets_splittings: substructure_methods.JetSplittingArray,
    hists: analysis_objects.Hists[analysis_objects.SubstructureHists],
    jet_R: float,
    weight: float,
) -> None:
    # Calculate the inputs
    inputs = analysis_objects.FillHistogramInput(
        restricted_jets,
        restricted_jets_splittings,
        *calculation(restricted_jets_splittings)[:2],
    )
    # And fill the results.
    # NOTE: cast is to help out mypy.
    selected_hists = cast(analysis_objects.SubstructureHists, getattr(hists, fill_attr_name))
    selected_hists.fill(
        inputs=inputs,
        jet_R=jet_R,
        weight=weight,
    )


def analyze_single_tree(
    tree: data_manager.Tree,
    dataset: analysis_objects.Dataset,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    hists_filename_stem: str,
    force_reprocessing: bool = False,
) -> analysis_objects.SingleTreeResult:
    # Setup
    logger.info(f"Processing tree from file {tree.filename}")
    results = analysis_objects.SingleTreeResult()
    # If the output file already exist, skip processing the tree and just return the hists instead (which is way faster!)
    train_number = tree.filename.parent.name
    pkl_filename = dataset.output / f"{train_number}_{tree.filename.stem}_{hists_filename_stem}.pgz"
    if pkl_filename.exists() and not force_reprocessing:
        logger.info(f"Skipping processing of tree {tree.filename} by loading data from stored hists.")
        with gzip.GzipFile(pkl_filename, "r") as pkl_file:
            results = analysis_objects.SingleTreeResult(**pickle.load(pkl_file))  # type: ignore
            return results

    # Since we're actually processing, we setup the output hists
    iterables = {
        "iterative_splittings": [False, True],
        "jet_pt_bin": jet_pt_bins,
    }
    results.create_hists(dataset=dataset, **iterables)

    # Setup the substructure jets
    successfully_accessed_data, (jets,) = load_jets_from_tree(tree=tree, prefixes=["data"])
    # Catch all failed cases.
    if not successfully_accessed_data:
        # Return the empty hists. We can't process this data :-(
        return results

    # Sanity check using iterative splittings information stored with the splittings
    # This is used as a local testing cross check. We don't want to include it with the standard output
    # because it will increase the output size with redundant information
    try:
        iterative_splittings_mask = tree["data.fJetSplittings.fIterativeSplitting"]
        logger.debug("Checking iterative splittings are calculated correctly.")
        # The jet_pt_mask is just a hack for selecting everything.
        _, temp_iterative_splittings = _select_and_retrieve_splittings(
            jets,
            jet_pt_mask=np.ones_like(jets) > 0,
            iterative_splittings=True,
        )
        assert (jets.splittings[iterative_splittings_mask] == temp_iterative_splittings).all().all()
    except KeyError:
        ...

    # Define calculation functions
    (
        inclusive_func,
        dynamical_z_func,
        dynamical_kt_func,
        dynamical_time_func,
        leading_kt_func,
        leading_kt_hard_cutoff_func,
    ) = _define_calculation_funcs(dataset)

    # Loop over iterations (jet pt ranges, iterative splitting)
    progress_manager = enlighten.get_manager()
    with progress_manager.counter(
        total=len(results.hists), desc="Analyzing", unit="variation", leave=False
    ) as selections_counter:
        for identifier, h in selections_counter(results.hists.items()):
            # Restrict the jets
            restricted_jets, restricted_jets_splittings = _select_and_retrieve_splittings(
                jets,
                jet_pt_mask=identifier.jet_pt_bin.mask_array(jets.jet_pt),
                iterative_splittings=identifier.iterative_splittings,
            )

            # Fill the hists as appropriate
            # TODO: Does inclusive work??
            for func, attr_name in [
                (inclusive_func, "inclusive"),
                (dynamical_z_func, "dynamical_z"),
                (dynamical_kt_func, "dynamical_kt"),
                (dynamical_time_func, "dynamical_time"),
                (leading_kt_func, "leading_kt"),
                (leading_kt_hard_cutoff_func, "leading_kt_hard_cutoff"),
            ]:
                _fill_substructure_hists_with_calculation(
                    calculation=func,
                    fill_attr_name=attr_name,
                    restricted_jets=restricted_jets,
                    restricted_jets_splittings=restricted_jets_splittings,
                    hists=results.hists[identifier],
                    jet_R=dataset.settings.jet_R,
                    weight=1.0,
                )

    # Store hists with pickle because it takes too longer otherwise (and for consistency).
    with gzip.GzipFile(pkl_filename, "w") as pkl_file:
        pickle.dump(dict(results.items()), pkl_file)  # type: ignore

    logger.info(f"Finished processing tree from file {tree.filename}")
    return results


def _fill_toy_hists_with_calculation(
    calculation: functools.partial[Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]],
    fill_attr_name: str,
    restricted_data_jets: substructure_methods.SubstructureJetArray,
    restricted_data_jets_splittings: substructure_methods.JetSplittingArray,
    restricted_true_jets: substructure_methods.SubstructureJetArray,
    restricted_true_jets_splittings: substructure_methods.JetSplittingArray,
    hists: analysis_objects.Hists[analysis_objects.SubstructureToyHists],
    jet_R: float,
    weight: float,
) -> None:
    # Calculate the inputs
    data_inputs = analysis_objects.FillHistogramInput(
        restricted_data_jets,
        restricted_data_jets_splittings,
        *calculation(restricted_data_jets_splittings)[:2],
    )
    # TODO: We absolutely shouldn't be calculating the splitting properties here!
    # TODO: If we take the leading, we already know that it was only one splitting, and we already
    # TODO: know the values...
    true_inputs = analysis_objects.FillHistogramInput(
        restricted_true_jets,
        restricted_true_jets_splittings,
        *calculation(restricted_true_jets_splittings)[:2],
    )
    # NOTE: cast is to help out mypy.
    selected_toy_hists = cast(analysis_objects.SubstructureToyHists, getattr(hists, fill_attr_name))
    selected_toy_hists.fill(
        data_inputs=data_inputs,
        true_inputs=true_inputs,
        jet_R=jet_R,
        weight=weight,
    )


def analyze_single_tree_toy(
    tree: data_manager.Tree,
    dataset: analysis_objects.Dataset,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    hists_filename_stem: str,
    force_reprocessing: bool,
    data_prefix: str,
) -> analysis_objects.SingleTreeToyResult:
    # Validation
    if data_prefix == "hybrid":
        data_prefix = "data"
    # Setup
    logger.info(f"Processing tree from file {tree.filename}")
    results = analysis_objects.SingleTreeToyResult()
    # If the output file already exist, skip processing the tree and just return the hists instead (which is way faster!)
    train_number = tree.filename.parent.name
    pkl_filename = dataset.output / f"{train_number}_{tree.filename.stem}_{hists_filename_stem}.pgz"
    if pkl_filename.exists() and not force_reprocessing:
        logger.info(f"Skipping processing of tree {tree.filename} by loading data from stored hists.")
        with gzip.GzipFile(pkl_filename, "r") as pkl_file:
            results = analysis_objects.SingleTreeToyResult(**pickle.load(pkl_file))  # type: ignore
            return results

    # Since we're actually processing, we setup the output hists
    iterables = {
        "iterative_splittings": [False, True],
        "jet_pt_bin": jet_pt_bins,
    }
    results.create_hists(dataset=dataset, **iterables)

    # Setup the substructure jets
    successfully_accessed_data, (data_jets, true_jets) = load_jets_from_tree(tree=tree, prefixes=[data_prefix, "true"])
    # Catch all failed cases.
    if not successfully_accessed_data:
        # Return the empty hists. We can't process this data :-(
        return results

    # Define calculation functions
    (
        inclusive_func,
        dynamical_z_func,
        dynamical_kt_func,
        dynamical_time_func,
        leading_kt_func,
        leading_kt_hard_cutoff_func,
    ) = _define_calculation_funcs(dataset)

    # Loop over iterations (jet pt ranges, iterative splitting)
    progress_manager = enlighten.get_manager()
    with progress_manager.counter(
        total=len(results.hists), desc="Analyzing", unit="toy variation", leave=False
    ) as selections_counter:
        for identifier, h in selections_counter(results.hists.items()):
            # We want to restrict a constant hybrid jet pt range for both true and hybrid.
            # This will allow us to compare to measured jet pt ranges.
            jet_pt_mask = identifier.jet_pt_bin.mask_array(data_jets.jet_pt)
            # Add additional restrictions that we can't handle single constituent jets.
            jet_pt_mask = jet_pt_mask & (data_jets.subjets.counts > 2)

            # Then restrict our jets.
            restricted_data_jets, restricted_data_jets_splittings = _select_and_retrieve_splittings(
                data_jets, jet_pt_mask, identifier.iterative_splittings
            )
            restricted_true_jets, restricted_true_jets_splittings = _select_and_retrieve_splittings(
                true_jets, jet_pt_mask, identifier.iterative_splittings
            )

            # Fill the hists as appropriate
            # TODO: Does inclusive work??
            for func, attr_name in [
                (inclusive_func, "inclusive"),
                (dynamical_z_func, "dynamical_z"),
                (dynamical_kt_func, "dynamical_kt"),
                (dynamical_time_func, "dynamical_time"),
                (leading_kt_func, "leading_kt"),
                # (leading_kt_hard_cutoff_func, "leading_kt_hard_cutoff"),
            ]:
                _fill_toy_hists_with_calculation(
                    calculation=func,
                    fill_attr_name=attr_name,
                    restricted_data_jets=restricted_data_jets,
                    restricted_data_jets_splittings=restricted_data_jets_splittings,
                    restricted_true_jets=restricted_true_jets,
                    restricted_true_jets_splittings=restricted_true_jets_splittings,
                    hists=results.hists[identifier],
                    jet_R=dataset.settings.jet_R,
                    weight=1.0,
                )

    # Store hists with pickle because it takes too longer otherwise.
    with gzip.GzipFile(pkl_filename, "w") as pkl_file:
        pickle.dump(dict(results.items()), pkl_file)  # type: ignore

    logger.info(f"Finished processing tree from file {tree.filename}")
    return results


def _select_and_retrieve_splittings(
    jets: substructure_methods.SubstructureJetArray, jet_pt_mask: UprootArray[bool], iterative_splittings: bool
) -> Tuple[substructure_methods.SubstructureJetArray, substructure_methods.JetSplittingArray]:
    # Ensure that there are sufficient counts
    restricted_jets = jets[jet_pt_mask]
    if iterative_splittings:
        # Only keep iterative splittings.
        splittings = restricted_jets.splittings.iterative_splittings(restricted_jets.subjets)

        # Enable this test to determine if we've selected different sets of splittings with the
        # recursive vs iterative selections.
        # if (splittings.counts != restricted_jets.splittings.counts).any():
        #    logger.warning("Disagreement between number of inclusive and recursive splittings (as expected!)")
        #    IPython.embed()
    else:
        splittings = restricted_jets.splittings

    return restricted_jets, splittings


def _subjets_contributing_to_splittings(
    inputs: analysis_objects.FillHistogramInput,
) -> substructure_methods.SubjetArray:
    """Determine which subjets contribute to the selected splitting.

    We do this by looking for subjets with a parent splitting index that is equal to the selected index.

    Args:
        inputs: Jets and splittings selected by a particular algorithm.
    Returns:
        Subjets which contributed to the selected splittings. There will always be 2 subjets by definition.
    """
    # In order to compare to the subjets directly, we need to expand the indices to the same dimension as the subjets.
    selected_indices_mask = inputs.jets.subjets.parent_splitting_index.ones_like() * inputs.indices.flatten()
    matched_subjets_unsorted = inputs.jets.subjets[selected_indices_mask == inputs.jets.subjets.parent_splitting_index]
    return cast(substructure_methods.SubjetArray, matched_subjets_unsorted)


def _get_leading_and_subleading_subjets(
    subjets_unsorted: substructure_methods.SubjetArray,
) -> Tuple[substructure_methods.SubjetArray, substructure_methods.SubjetArray]:
    """Determine the leading and subleading subjets based on the sum of subjet constituents pt.

    Args:
        subjets_unsorted: Unsorted subjets of a given splitting. There are two subjets by definition.
    Returns:
        Leading subjets, subleading subjets.
    """
    # Sort the subjets such that 0 is always the leading subjet.
    # Coerces the bool into an integer by taking 1 - array.
    # The leading subjet will be have a 0, while the subleading will have a 1.
    # NOTE: We actually want to add the four vectors rather than just summing the constituent pt. It doesn't
    #       have a huge impact, but it's the right way to do it.
    unsorted_subjet_pt = subjets_unsorted.constituents.four_vectors().sum().pt
    subjets_pt_comparison = 1 - (unsorted_subjet_pt[:, 0] > unsorted_subjet_pt[:, 1])
    # For each subjet_pt_comparison, we want to take the index of the leading subjet and use that to extract the leading subjet.
    leading_indices = ak0.JaggedArray.fromoffsets(range(len(subjets_pt_comparison) + 1), subjets_pt_comparison)
    subjets_leading = subjets_unsorted[leading_indices].flatten()
    # Same idea for the subleading subjet (which is necessarily 1 - subjets_pt_comparison because there are only two subjets.
    subleading_indices = ak0.JaggedArray.fromoffsets(range(len(subjets_pt_comparison) + 1), 1 - subjets_pt_comparison)
    subjets_subleading = subjets_unsorted[subleading_indices].flatten()

    return subjets_leading, subjets_subleading


def _split_array(
    a: substructure_methods.SubjetArray, n: int
) -> Iterable[Tuple[substructure_methods.SubjetArray, slice]]:
    """Split an array into n chunks.

    Currently the typing suggests that it will only work for SubjetArray, but it should work for any array.

    From: https://stackoverflow.com/a/2135920/12907985
    """
    k, m = divmod(len(a), n)
    return (
        (
            a[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)],  # noqa: E203 . Conflicts with black...
            slice(i * k + min(i, m), (i + 1) * k + min(i + 1, m)),
        )
        for i in range(n)
    )


def _determine_matching_types(
    matched_subjets: substructure_methods.SubjetArray,
    hybrid_subjets: substructure_methods.SubjetArray,
    match_using_distance: bool = False,
) -> UprootArray[bool]:
    """Determine whether the given subjets match.

    Note:
        Matching by global index only works for det-hybrid matching. For part-det matching, matching must be performed
        using distance between constituents.

    Args:
        matched_subjets: Subjets from the matched jets.
        hybrid_subjets: Subjets from the hybrid jets.
        match_using_distance: If True, match using distance of the constituents,
            rather than global index.
    Returns:
        Mask indicating when these subjets matched.
    """
    # We split the array into chunks to keep memory usage to a more reasonable level.
    number_of_chunks = 6

    shared_constituents_pts = np.zeros(len(matched_subjets))
    # Can't use np.array_split (even though it would be nice and probably better tested) because the output ends up as
    # numpy array, and in this case, we don't want such a conversion.
    for (matched_subset, selected_range), (hybrid_subset, _) in zip(
        _split_array(matched_subjets.constituents, number_of_chunks),
        _split_array(hybrid_subjets.constituents, number_of_chunks),
    ):
        constituent_pairs = matched_subset.argcross(hybrid_subset)
        matched_leading_indices, hybrid_leading_indices = constituent_pairs.unzip()

        if match_using_distance:
            # Require delta eta and delta phi to be within 0.001
            delta = 0.01
            delta_eta_matching = (
                np.abs(matched_subset[matched_leading_indices].eta - hybrid_subset[hybrid_leading_indices].eta) < delta
            )
            delta_phi_matching = (
                np.abs(matched_subset[matched_leading_indices].phi - hybrid_subset[hybrid_leading_indices].phi) < delta
            )
            index_matching = delta_eta_matching & delta_phi_matching
        else:
            index_matching = (
                matched_subset[matched_leading_indices].global_index
                == hybrid_subset[hybrid_leading_indices].global_index
            )

        shared_constituents_pts[selected_range] = matched_subset[matched_leading_indices][index_matching].pt.sum()

    # Sanity check
    # NOTE: We use the sum of the constituents pt here because this is the convention.
    #       This is handled differently than for finding the leading subjet.
    if (shared_constituents_pts > matched_subjets.constituents.pt.sum()).any():
        mask_excess = shared_constituents_pts > matched_subjets.constituents.pt.sum()
        logger.warning(
            f"Constituent pts are greater than the subjet pts. Fraction: {np.count_nonzero(mask_excess) / len(mask_excess)}"
        )
        # IPython.embed()
        # raise ValueError("Constituent pts are greater than the subjet pts...")

    matched = (shared_constituents_pts / matched_subjets.constituents.pt.sum()) > 0.5
    return cast(UprootArray[bool], matched)


def determine_matched_jets(
    hybrid_inputs: analysis_objects.FillHistogramInput,
    matched_inputs: analysis_objects.FillHistogramInput,
    match_using_distance: bool = False,
) -> Tuple[analysis_objects.MatchingResult, analysis_objects.MatchingResult]:
    """Determine the matching between subjets.

    The passed jets need to have the selected indices already applied.
    We need to work with the indices applied to these jets.

    Args:
        hybrid_inputs: The selected hybrid jets and splittings.
        matched_inputs: The selected matched jets and splittings.
        match_using_distance: If True, match using distance of the constituents,
            rather than global index.
    Returns:
        Leading subjet matching results, subleading subjet matching results.
    """
    # Setup
    # Mask if one of the inputs doesn't have a selected splitting (appears to only matter at detector level if we have a z_cut)
    mask = (matched_inputs.indices.counts != 0) & (hybrid_inputs.indices.counts != 0)
    # try:
    restricted_matched_inputs = matched_inputs[mask]
    restricted_hybrid_inputs = hybrid_inputs[mask]
    # except IndexError as e:
    #    logger.warning(e)
    #    IPython.start_ipython(user_ns=locals())

    # Determine which subjets contribute to the selected splitting.
    matched_subjets_unsorted = _subjets_contributing_to_splittings(inputs=restricted_matched_inputs)
    hybrid_subjets_unsorted = _subjets_contributing_to_splittings(inputs=restricted_hybrid_inputs)

    # Sort the subjets such that 0 is always the leading subjet.
    matched_subjets_leading, matched_subjets_subleading = _get_leading_and_subleading_subjets(matched_subjets_unsorted)
    hybrid_subjets_leading, hybrid_subjets_subleading = _get_leading_and_subleading_subjets(hybrid_subjets_unsorted)

    # Now, determine the matching types based on the possible combinations of leading and subleading subjets.
    matched_leading_properly = _determine_matching_types(
        matched_subjets_leading, hybrid_subjets_leading, match_using_distance=match_using_distance
    )
    matched_leading_mistag = _determine_matching_types(
        matched_subjets_leading, hybrid_subjets_subleading, match_using_distance=match_using_distance
    )
    matched_subleading_properly = _determine_matching_types(
        matched_subjets_subleading, hybrid_subjets_subleading, match_using_distance=match_using_distance
    )
    matched_subleading_mistag = _determine_matching_types(
        matched_subjets_subleading, hybrid_subjets_leading, match_using_distance=match_using_distance
    )
    # Combine those cases to determine when the we failed to find the leading and subleading subjets.
    matched_leading_failed = ~matched_leading_properly & ~matched_leading_mistag
    matched_subleading_failed = ~matched_subleading_properly & ~matched_subleading_mistag

    return (
        analysis_objects.MatchingResult(matched_leading_properly, matched_leading_mistag, matched_leading_failed),
        analysis_objects.MatchingResult(
            matched_subleading_properly, matched_subleading_mistag, matched_subleading_failed
        ),
    )


def _fill_embedded_hists_with_calculation(
    calculation: functools.partial[Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]],
    fill_attr_name: str,
    identifier: analysis_objects.Identifier,
    restricted_true_jets: substructure_methods.SubstructureJetArray,
    restricted_true_jets_splittings: substructure_methods.JetSplittingArray,
    restricted_det_level_jets: substructure_methods.SubstructureJetArray,
    restricted_det_level_jets_splittings: substructure_methods.JetSplittingArray,
    restricted_hybrid_jets: substructure_methods.SubstructureJetArray,
    restricted_hybrid_jets_splittings: substructure_methods.JetSplittingArray,
    true_hists: analysis_objects.Hists[analysis_objects.SubstructureHists],
    det_level_hists: analysis_objects.Hists[analysis_objects.SubstructureHists],
    hybrid_hists: analysis_objects.Hists[analysis_objects.SubstructureHists],
    jet_R: float,
    weight: float,
    detector_particle_response_hists: Optional[
        analysis_objects.Hists[analysis_objects.SubstructureResponseHists]
    ] = None,
    hybrid_detector_response_hists: Optional[analysis_objects.Hists[analysis_objects.SubstructureResponseHists]] = None,
    hybrid_particle_response_hists: Optional[
        analysis_objects.Hists[analysis_objects.SubstructureResponseExtendedHists]
    ] = None,
) -> None:
    # Calculate the inputs
    true_inputs = analysis_objects.FillHistogramInput(
        restricted_true_jets,
        restricted_true_jets_splittings,
        *calculation(restricted_true_jets_splittings)[:2],
    )
    det_level_inputs = analysis_objects.FillHistogramInput(
        restricted_det_level_jets,
        restricted_det_level_jets_splittings,
        *calculation(restricted_det_level_jets_splittings)[:2],
    )
    hybrid_inputs = analysis_objects.FillHistogramInput(
        restricted_hybrid_jets,
        restricted_hybrid_jets_splittings,
        *calculation(restricted_hybrid_jets_splittings)[:2],
    )
    # And fill the results.
    # NOTE: casts are to help out mypy.
    selected_true_hists = cast(analysis_objects.SubstructureHists, getattr(true_hists, fill_attr_name))
    selected_true_hists.fill(
        inputs=true_inputs,
        jet_R=jet_R,
        weight=weight,
    )
    selected_det_hists = cast(analysis_objects.SubstructureHists, getattr(det_level_hists, fill_attr_name))
    selected_det_hists.fill(
        inputs=det_level_inputs,
        jet_R=jet_R,
        weight=weight,
    )
    selected_hybrid_hists = cast(analysis_objects.SubstructureHists, getattr(hybrid_hists, fill_attr_name))
    selected_hybrid_hists.fill(
        inputs=hybrid_inputs,
        jet_R=jet_R,
        weight=weight,
    )

    # Validation
    if (
        (detector_particle_response_hists is not None)
        != (hybrid_detector_response_hists is not None)
        != (hybrid_particle_response_hists is not None)
    ):
        # These should always be paired together, so something has gone wrong. Raise to notify.
        raise ValueError(
            f"Passed one of detector-particle, hybrid-detector, or hybrid-particle response hists, but not the other. Detector-particle: {detector_particle_response_hists}, hybrid-detector: {hybrid_detector_response_hists}, hybrid_particle: {hybrid_particle_response_hists}"
        )

    # Because of the validation, the time for filling the matching is equivalent to when we should fill the response hists.
    if hybrid_particle_response_hists is not None:
        logger.debug(f"Performing matching and filling response for {identifier}, {fill_attr_name}.")
        # Setup
        # Mask to ensure that there are selected splittings before we proceed to matching and filling.
        # We have to mask here (rather than at filling) because the matching will be masked, and we need
        # the inputs to be the right length to work the output from the matching.
        # Mask if one of the inputs doesn't have a selected splitting (that appears to only matter at detector
        # level if we have a z_cut).
        mask = (det_level_inputs.indices.counts != 0) & (hybrid_inputs.indices.counts != 0)
        try:
            restricted_det_level_inputs = det_level_inputs[mask]
            restricted_hybrid_inputs = hybrid_inputs[mask]
            # True inputs aren't necessarily appropriate here because we're matching between hybrid-det level,
            # but include it so we can include the matching in the hybrid-particle response.
            restricted_true_inputs = true_inputs[mask]
        except IndexError as e:
            logger.warning(e)
            IPython.start_ipython(user_ns=locals())

        # Perform the matching
        # TODO: Does this work with the leading cutoff?? Not yet.
        try:
            leading_matching, subleading_matching = determine_matched_jets(
                restricted_hybrid_inputs, restricted_det_level_inputs
            )
            matching_selections = analysis_objects.MatchingSelections(
                leading=leading_matching, subleading=subleading_matching
            )
        except ValueError as e:
            logger.warning(e)
            IPython.start_ipython(user_ns=locals())

        # TODO: These matching selections are wrong for particle-detector. Should be okay for hybrid-particle (because we're most interested in hybrid-det matching there).

        # Fill the responses (with matching dependence)
        # detector-particle
        # NOTE: casts are to help out mypy.
        selected_detector_particle_response_hists = cast(
            analysis_objects.SubstructureResponseHists, getattr(detector_particle_response_hists, fill_attr_name)
        )
        selected_detector_particle_response_hists.fill(
            measured_like_inputs=det_level_inputs,
            generator_like_inputs=true_inputs,
            matching_selections=matching_selections,
            jet_R=jet_R,
            weight=weight,
        )
        # Hybrid-detector
        selected_hybrid_detector_response_hists = cast(
            analysis_objects.SubstructureResponseHists, getattr(hybrid_detector_response_hists, fill_attr_name)
        )
        selected_hybrid_detector_response_hists.fill(
            measured_like_inputs=restricted_hybrid_inputs,
            generator_like_inputs=restricted_det_level_inputs,
            matching_selections=matching_selections,
            jet_R=jet_R,
            weight=weight,
        )
        # Hybrid-particle
        selected_hybrid_particle_response_hists = cast(
            analysis_objects.SubstructureResponseExtendedHists, getattr(hybrid_particle_response_hists, fill_attr_name)
        )
        selected_hybrid_particle_response_hists.fill(
            measured_like_inputs=restricted_hybrid_inputs,
            generator_like_inputs=restricted_true_inputs,
            matching_selections=matching_selections,
            jet_R=jet_R,
            weight=weight,
        )


def analyze_single_tree_embedding(  # noqa: C901
    tree: data_manager.Tree,
    dataset: analysis_objects.Dataset,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    hists_filename_stem: str,
    force_reprocessing: bool = False,
    scale_n_jets_when_loading_hists: bool = False,
) -> analysis_objects.SingleTreeEmbeddingResult:
    """Determine the response and prong matching for jets substructure techniques.

    Why combine them together? Because then we only have to open and process a tree once.
    At a future date (beyond the start of April 2020), it would be better to refactor them more separately,
    such that we can enable or disable the different options and still have appropriate return values.
    But for now, we don't worry about it.
    """
    # Setup
    logger.info(f"Processing tree from file {tree.filename}")
    # Help out mypy...
    assert isinstance(dataset.settings, analysis_objects.PtHardAnalysisSettings)
    results = analysis_objects.SingleTreeEmbeddingResult()
    # Determine scale factor
    # NOTE: This relies on the train_number being up a directory!
    train_number = tree.filename.parent.name
    pt_hard_bin = dataset.settings.train_number_to_pt_hard_bin[int(train_number)]
    scale_factor = dataset.settings.scale_factors[pt_hard_bin]

    # If the output file already exist, skip processing the tree and just return the hists instead (which is way faster!)
    pkl_filename = dataset.output / f"{train_number}_{tree.filename.stem}_{hists_filename_stem}.pgz"
    if pkl_filename.exists() and not force_reprocessing:
        logger.info(f"Skipping processing of tree {tree.filename} by loading data from stored hists.")
        with gzip.GzipFile(pkl_filename, "r") as pkl_file:
            results = analysis_objects.SingleTreeEmbeddingResult(**pickle.load(pkl_file))  # type: ignore
            # true_hists, det_level_hists, hybrid_hists, response_hists, matching_hists =   # type: ignore
            # NOTE: This is transient for loading files this way. However, it won't be transient if we load
            #       for just plotting (as it will be saved in the merge hists)
            if scale_n_jets_when_loading_hists:
                logger.warning(
                    "Rescaling n_jets by scale factor because it was forgotten during processing. This is transient for the individual files, but not for the merged!"
                )
                for result_hists in [results.true_hists, results.det_level_hists, results.hybrid_hists]:
                    for temp_hists in result_hists.values():
                        for technique, technique_hists in temp_hists:
                            technique_hists.n_jets *= scale_factor
            return results

    # Since we're actually processing, we setup the output hists
    iterables = {
        "iterative_splittings": [False, True],
        "jet_pt_bin": jet_pt_bins,
    }
    results.create_hists(dataset=dataset, **iterables)

    # Setup the substructure jets
    successfully_accessed_data, (hybrid_jets, true_jets, det_level_jets) = load_jets_from_tree(
        tree=tree, prefixes=["data", "matched", "detLevel"]
    )
    # Catch all failed cases.
    if not successfully_accessed_data:
        # Return the empty hists. We can't process this data :-(
        return results

    # Define calculation functions
    (
        inclusive_func,
        dynamical_z_func,
        dynamical_kt_func,
        dynamical_time_func,
        leading_kt_func,
        leading_kt_hard_cutoff_func,
    ) = _define_calculation_funcs(dataset)

    # Loop over iterations (jet pt ranges, iterative splitting)
    progress_manager = enlighten.get_manager()
    with progress_manager.counter(
        total=len(results.true_hists), desc="Analyzing", unit="embedded variation", leave=False
    ) as selections_counter:
        # NOTE: It's important to iterate with true_hists rather than response_hists because true_hists will have
        #       more selections, and is required to contain the response_hists selection(s).
        for identifier, h in selections_counter(results.true_hists.items()):
            # Jet quality selections.
            # We want to restrict a constant hybrid jet pt range for both true and hybrid.
            # This will allow us to compare to measured jet pt ranges.
            mask = identifier.jet_pt_bin.mask_array(hybrid_jets.jet_pt)
            # Ensure that we don't have single track jets because the splitting won't be defined for that case.
            # No actual jet pt range restrictions.
            mask = (
                mask
                & (hybrid_jets.constituents.counts > 1)
                & (true_jets.constituents.counts > 1)
                & (det_level_jets.constituents.counts > 1)
            )
            # Require that we have jets that aren't dominated by hybrid jets.
            # It's super important to be ">=". That allows the leading jet in the hybrid to be the same
            # as the leading jet in the true (which would be good - we've probably found the right jet).
            mask = mask & (true_jets.constituents.max_pt >= hybrid_jets.constituents.max_pt)

            # Then restrict our jets.
            restricted_true_jets, restricted_true_jets_splittings = _select_and_retrieve_splittings(
                true_jets, mask, identifier.iterative_splittings
            )
            restricted_det_level_jets, restricted_det_level_jets_splittings = _select_and_retrieve_splittings(
                det_level_jets, mask, identifier.iterative_splittings
            )
            restricted_hybrid_jets, restricted_hybrid_jets_splittings = _select_and_retrieve_splittings(
                hybrid_jets, mask, identifier.iterative_splittings
            )

            # Fill the hists as appropriate
            # TODO: Inclusive
            for func, attr_name in [
                (dynamical_z_func, "dynamical_z"),
                (dynamical_kt_func, "dynamical_kt"),
                (dynamical_time_func, "dynamical_time"),
                (leading_kt_func, "leading_kt"),
                (leading_kt_hard_cutoff_func, "leading_kt_hard_cutoff"),
            ]:
                _fill_embedded_hists_with_calculation(
                    calculation=func,
                    fill_attr_name=attr_name,
                    identifier=identifier,
                    restricted_true_jets=restricted_true_jets,
                    restricted_true_jets_splittings=restricted_true_jets_splittings,
                    restricted_det_level_jets=restricted_det_level_jets,
                    restricted_det_level_jets_splittings=restricted_det_level_jets_splittings,
                    restricted_hybrid_jets=restricted_hybrid_jets,
                    restricted_hybrid_jets_splittings=restricted_hybrid_jets_splittings,
                    true_hists=results.true_hists[identifier],
                    det_level_hists=results.det_level_hists[identifier],
                    hybrid_hists=results.hybrid_hists[identifier],
                    # We only fill the response for the widest jet pt selection so we don't store redundant information.
                    # If it is retrieved, we use it. If not, we'll skip filling it.
                    detector_particle_response_hists=results.detector_particle_response.get(identifier, None),
                    hybrid_detector_response_hists=results.hybrid_detector_response.get(identifier, None),
                    hybrid_particle_response_hists=results.hybrid_particle_response.get(identifier, None),
                    jet_R=dataset.settings.jet_R,
                    weight=scale_factor,
                )

    # Store the hists
    # Store hists with pickle because it takes too longer otherwise.
    # NOTE: We expand out the values when pickling in case the object changes.
    logger.debug("Done processing. Writing out results.")
    with gzip.GzipFile(pkl_filename, "w") as pkl_file:
        pickle.dump(dict(results.items()), pkl_file)  # type: ignore

    logger.info(f"Finished processing tree from file {tree.filename}")
    return results


# def _fill_matching_hists_with_calculation(
#    calculation: functools.partial[Tuple[UprootArray[float], UprootArray[int]]],
#    fill_attr_name: str,
#    restricted_hybrid_jets: substructure_methods.SubstructureJetArray,
#    restricted_hybrid_jets_splittings: substructure_methods.JetSplittingArray,
#    restricted_matched_jets: substructure_methods.SubstructureJetArray,
#    restricted_matched_jets_splittings: substructure_methods.JetSplittingArray,
#    matching_hists: Dict[
#        analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]
#    ],
#    identifier: analysis_objects.Identifier,
#    weight: float,
# ) -> None:
#    # Calculate the inputs
#    hybrid_inputs = analysis_objects.FillHistogramInput(
#        restricted_hybrid_jets, restricted_hybrid_jets_splittings, *calculation(restricted_hybrid_jets_splittings),
#    )
#    matched_inputs = analysis_objects.FillHistogramInput(
#        restricted_matched_jets, restricted_matched_jets_splittings, *calculation(restricted_matched_jets_splittings),
#    )
#    leading_matching, subleading_matching = determine_matched_jets(hybrid_inputs, matched_inputs)
#    # And fill the results.
#    temp_identifier = analysis_objects.Identifier(
#        iterative_splittings=identifier.iterative_splittings, jet_pt_bin=identifier.jet_pt_bin,
#    )
#    getattr(matching_hists[temp_identifier], fill_attr_name).fill(
#        matched_inputs=matched_inputs,
#        hybrid_inputs=hybrid_inputs,
#        leading=leading_matching,
#        subleading=subleading_matching,
#        weight=weight,
#    )


# def matching(
#    matching_hists: Dict[
#        analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]
#    ],
#    matched_jets: substructure_methods.SubstructureJetArray,
#    hybrid_jets: substructure_methods.SubstructureJetArray,
#    dataset: analysis_objects.Dataset,
#    scale_factor: float,
#    progress_manager: enlighten.Manager,
# ) -> Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]]:
#    """ Determine the prong matching for jets substructure techniques.
#
#    """
#    # Setup
#    # Define calculation functions
#    (
#        dynamical_z_func,
#        dynamical_kt_func,
#        dynamical_time_func,
#        leading_kt_func,
#        leading_kt_hard_cutoff_func,
#    ) = _define_calculation_funcs(dataset)
#
#    # Actually perform the matching
#    logger.info("Starting matching")
#    number_of_grooming_methods = 4
#    with progress_manager.counter(
#        total=len(matching_hists) * number_of_grooming_methods, desc="Analyzing", unit="matching variations", leave=False
#    ) as selections_counter:
#        for identifier, h in selections_counter(matching_hists.items()):
#            # Ensure that we don't have single track jets because the splitting won't be defined for that case.
#            # No actual jet pt range restrictions.
#            mask = (hybrid_jets.constituents.counts > 1) & (matched_jets.constituents.counts > 1)
#            # Require that we have jets that aren't dominated by hybrid jets.
#            # It's super important to be ">=". That allows the leading jet in the hybrid to be the same
#            # as the leading jet in the true (which would be good - we've probably found the right jet).
#            mask = mask & (matched_jets.constituents.max_pt >= hybrid_jets.constituents.max_pt)
#
#            restricted_hybrid_jets, restricted_hybrid_jets_splittings = _select_and_retrieve_splittings(
#                hybrid_jets, mask, identifier.iterative_splittings
#            )
#            restricted_matched_jets, restricted_matched_jets_splittings = _select_and_retrieve_splittings(
#                matched_jets, mask, identifier.iterative_splittings
#            )
#
#            # Scale factor to account for pt hard bin.
#            weight = scale_factor
#
#            # Fill the hists as appropriate
#            # TODO: Inclusive
#            # TODO: SD
#            for func, attr_name in [
#                (dynamical_z_func, "dynamical_z"),
#                (dynamical_kt_func, "dynamical_kt"),
#                (dynamical_time_func, "dynamical_time"),
#                (leading_kt_func, "leading_kt"),
#            ]:
#                _fill_matching_hists_with_calculation(
#                    calculation=func,
#                    fill_attr_name=attr_name,
#                    restricted_hybrid_jets=restricted_hybrid_jets,
#                    restricted_hybrid_jets_splittings=restricted_hybrid_jets_splittings,
#                    restricted_matched_jets=restricted_matched_jets,
#                    restricted_matched_jets_splittings=restricted_matched_jets_splittings,
#                    matching_hists=matching_hists,
#                    identifier=identifier,
#                    weight=weight,
#                )
#                selections_counter.update()
#
#    return matching_hists


def _wrap_multiprocessing(
    tree: Callable[[], data_manager.Tree],
    analysis_function: Callable[
        [data_manager.Tree],
        Sequence[Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]]],
    ],
) -> Sequence[Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]]]:
    """Wrap analysis function to instantiate the fully lazy tree.

    To be used in conjunction with multiprocessing (which is why we need to delay instantiating the tree).

    Args:
        tree: Tree to be instantiated.
        analysis_function: Analysis function to be called. All of the other arguments should be bound with partial.
    Returns:
        Executes the analysis function with the instantiated tree.
    """
    return analysis_function(tree())


# def merge_results(
#    existing: List[
#        Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]]
#    ], result: List[
#        Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]]
#    ]
# ) -> List[Dict[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.T_SubstructureHists]]]:
#    for hist_result in result:
#        for h in hist_result.values():
#            h.convert_boost_histograms_to_binned_data()
#
#    for i, hist_result in enumerate(existing):
#        for k in hist_result.keys():
#            existing[i][k] = existing[i][k] + result[i][k]
#
#    return existing


def merge_results(existing: _T_Result, new_result: _T_Result) -> _T_Result:
    for temp_result in [existing, new_result]:
        for hist_result in temp_result.values():
            for h in hist_result.values():
                h.convert_boost_histograms_to_binned_data()

    # TODO: ****Check****: Does this actually modify the object contained within??
    for (name, hist_result), (new_name, new_hist_result) in zip(existing.items(), new_result.items()):
        for k in hist_result.keys():
            hist_result[k] += new_hist_result[k]

    return existing


_T_Result = TypeVar("_T_Result", bound=analysis_objects.SingleTreeResultBase)


def run_shared(  # noqa: C901
    collision_system: str,
    analysis_function: Callable[
        [data_manager.Tree, analysis_objects.Dataset, Sequence[helpers.RangeSelector], str, bool],
        _T_Result,
    ],
    dataset_config_filename: Path,
    hists_filename: str,
    jet_pt_bins: Sequence[helpers.RangeSelector],
    z_cutoff: float = 0.2,
    output: Path = Path("output"),
    plot_only: bool = False,
    force_reprocessing: bool = False,
    number_of_cores: int = 1,
    override_filenames: Optional[Sequence[Union[str, Path]]] = None,
    additional_kwargs_for_analysis: Optional[Dict[str, str]] = None,
) -> Tuple[_T_Result, analysis_objects.Dataset]:
    """Run the given analysis function.

    Args:
        collision_system: Name of the collision system.
        analysis_function: Function to perform the desired analysis on a single tree.
        hists_filename: Filename to be used for the merged hists generated in this analysis.
        jet_pt_bins: Jet pt bins to be selected in the analysis.
        z_cutoff: Z cutoff. Default: 0.2.
        output: Output directory. Default: `Path("output")`.
        plot_only: Only plot using the stored, fully merged hists. Don't event try to access
            the underlying files. Default: False.
        force_reprocessing: Force the trees to be reprocessed regardless of whether they already
            have output histograms.
        number_of_cores: Number of cores to be used for processing. If more than 1, then use multiprocessing.
            Careful of memory usage!! Default: 1.
        override_filenames: Filenames to be used during the analysis, overriding those specified in the
            configuration. Default: None, in which cause the filenames in the configuration are used.
        additional_kwargs_for_analysis: Additional keyword arguments to pass on to the single tree analysis
            function. Default: {}.
    Returns:
        ((hists returned from the analysis, merged over all of the inputs file), dataset configuration)
    """
    # Validation
    if additional_kwargs_for_analysis is None:
        additional_kwargs_for_analysis = {}

    # Configuration
    # Only need to set options which vary from the default.
    settings_class_map: Mapping[str, Type[analysis_objects.AnalysisSettings]] = {
        "embedPythia": analysis_objects.PtHardAnalysisSettings,
    }
    dataset = analysis_objects.Dataset.from_config_file(
        collision_system=collision_system,
        config_filename=dataset_config_filename,
        override_filenames=override_filenames,
        hists_filename_stem=hists_filename,
        output_base=output,
        settings_class=settings_class_map.get(collision_system, analysis_objects.AnalysisSettings),
        z_cutoff=z_cutoff,
    )

    # Output hists
    output_hists: _T_Result

    # Have a special option if we're plotting only so we can just read the final files.
    # Even though merging isn't very hard, all of that I/O is still slow than reading them once.
    if plot_only:
        logger.info(
            f"Loading system {collision_system} with dataset {dataset.name}, hists: {hists_filename} for plotting only"
        )
        if dataset.hists_filename.exists():
            # Read the stored hists.
            # We don't use YAML because it would be super slow!
            with gzip.GzipFile(dataset.hists_filename, "r") as pkl_file:
                output_hists = pickle.load(pkl_file)  # type: ignore

            return output_hists, dataset

        # If the file doesn't exist, we still need to process
        logger.warning("Requested plotting only, but the hists aren't available. Continuing on to processing.")

    # Setup dataset
    dm = data_manager.IterateTrees(
        filenames=dataset.filenames,
        tree_name=dataset.tree_name,
        # Mypy is getting confused by Sequence[str] because str is an iterable, so we ignore the type...
        branches=dataset.branches,  # type: ignore
    )
    logger.info("Setup complete. Beginning processing of trees.")

    # Create the analysis functions
    # We bind them with partial so we can execute them using map (which enables multiprocessing).
    analyze_single_tree_func = functools.partial(
        analysis_function,
        dataset=dataset,
        jet_pt_bins=jet_pt_bins,
        hists_filename_stem=dataset.hists_filename.stem,
        force_reprocessing=force_reprocessing,
        **additional_kwargs_for_analysis,
    )
    analyze_single_tree_func_multiprocessing = functools.partial(
        _wrap_multiprocessing,
        analysis_function=analyze_single_tree_func,
    )

    # Iterate over trees.
    progress_manager = enlighten.get_manager()
    # We need to use fully lazy iteration if we're using multiprocessing. Otherwise, we run into problems
    # with pickling objects (which is necessary from them to be sent to the other processes).
    dm_iterator = dm.lazy_iteration(fully_lazy=(number_of_cores > 1))
    with progress_manager.counter(total=len(dm), desc="Analyzing", unit="tree") as tree_counter:
        if number_of_cores > 1:
            with Pool(nodes=number_of_cores) as pool:
                # merge_results if store_results else lambda x, y: None,
                output_hists = functools.reduce(
                    merge_results,
                    tree_counter(pool.imap(analyze_single_tree_func_multiprocessing, dm_iterator)),
                )
        else:
            output_hists = functools.reduce(
                merge_results,
                tree_counter(map(analyze_single_tree_func, dm_iterator)),
            )

    # Write out the merged hists
    # Write with pkl because yaml is super slow for hists that are this large.
    # NOTE: We don't expand out the values stored in the output hists here.
    #       Otherwise, we couldn't easily recreate the type when we load it for plotting only.
    with gzip.GzipFile(dataset.hists_filename, "w") as pkl_file:
        pickle.dump(output_hists, pkl_file)  # type: ignore

    progress_manager.stop()

    return output_hists, dataset


def parse_arguments(name: str) -> List[Path]:
    parser = argparse.ArgumentParser(description=f"Run {name}")

    parser.add_argument("-f", "--filenames", nargs="+", default=[])
    args = parser.parse_args()
    # Validation for filenames
    filenames = [Path(f) for f in args.filenames]
    return filenames


def embed_pythia_entry_point() -> None:
    helpers.setup_logging()
    filenames = parse_arguments(name="embed pythia")

    collision_system = "embedPythia"
    jet_pt_bins = [
        helpers.RangeSelector(min=0, max=120),
        helpers.RangeSelector(min=40, max=120),
        # Most likely where we will actually measure.
        helpers.RangeSelector(min=80, max=120),
        helpers.RangeSelector(min=60, max=80),
        helpers.RangeSelector(min=80, max=100),
        helpers.RangeSelector(min=100, max=120),
    ]

    embedding_hists, dataset = run_shared(
        collision_system=collision_system,
        analysis_function=analyze_single_tree_embedding,
        dataset_config_filename=Path("config") / "datasets.yaml",
        hists_filename="embedding_hists",
        jet_pt_bins=jet_pt_bins,
        z_cutoff=0.2,
        override_filenames=filenames,
    )
    logger.info(f"Finished processing embedPythia for: {filenames}")


if __name__ == "__main__":
    helpers.setup_logging()

    # Setup and run
    config_filename = Path("config") / "datasets.yaml"
    plot_only = False
    jet_pt_bins = [
        # Broadest range
        helpers.RangeSelector(min=0, max=140),
        # Main range of interest.
        helpers.RangeSelector(min=40, max=120),
        # Most likely where we will actually measure.
        helpers.RangeSelector(min=80, max=120),
        # Individual ranges.
        helpers.RangeSelector(min=40, max=60),
        helpers.RangeSelector(min=60, max=80),
        helpers.RangeSelector(min=80, max=100),
        helpers.RangeSelector(min=100, max=120),
    ]
    matching_hybrid_min_kt_values = [
        0.0,
        5.0,
        7.0,
    ]
    z_cutoff = 0.2
    # Standard analysis
    # data_hists, data_dataset = run_shared(
    #    collision_system="PbPb",
    #    analysis_function=analyze_single_tree,
    #    dataset_config_filename=config_filename,
    #    hists_filename="data_hists",
    #    jet_pt_bins=jet_pt_bins,
    #    z_cutoff=z_cutoff,
    #    plot_only=plot_only,
    #    force_reprocessing=False,
    #    number_of_cores=1,
    # )
    # plot_results.lund_plane(all_hists=data_hists, jet_type_label="det", path=data_dataset.output)
    # Toy
    # data_prefix = "hybrid"
    # collision_system = f"toy_true_{data_prefix}_splittings_iterative_allTrueSplittings_delta_R_040"
    # toy_hists, dataset = run_shared(
    #    collision_system=collision_system,
    #    analysis_function=analyze_single_tree_toy,
    #    dataset_config_filename=config_filename,
    #    hists_filename="toy_hists",
    #    jet_pt_bins=jet_pt_bins,
    #    z_cutoff=z_cutoff,
    #    plot_only=plot_only,
    #    number_of_cores=2,
    #    additional_kwargs_for_analysis=dict(
    #        data_prefix=data_prefix,
    #    )
    # )
    # plot_results.toy(all_toy_hists=toy_hists, data_prefix=data_prefix, path=dataset.output)
    # Embedding
    embedded_hists, embedded_dataset = run_shared(
        collision_system="embedPythia",
        analysis_function=analyze_single_tree_embedding,
        dataset_config_filename=config_filename,
        hists_filename="embedding_hists",
        jet_pt_bins=jet_pt_bins,
        z_cutoff=z_cutoff,
        plot_only=plot_only,
        force_reprocessing=True,
        number_of_cores=1,
        # additional_kwargs_for_analysis=dict(
        #    scale_n_jets_when_loading_hists=True,
        # )
    )
    ## True, hybrid hists
    # plot_results.lund_plane(all_hists=embedded_hists.true_hists, jet_type_label="true", path=embedded_dataset.output)
    # plot_results.lund_plane(all_hists=embedded_hists.det_level_hists, jet_type_label="det", path=embedded_dataset.output)
    # plot_results.lund_plane(all_hists=embedded_hists.hybrid_hists, jet_type_label="hybrid", path=embedded_dataset.output)
    ## Responses
    # plot_results.responses(all_response_hists=embedded_hists.response_hists, path=embedded_dataset.output)
    ## Matching
    # plot_results.matching(all_matching_hists=embedded_hists.matching_hists, hybrid_jet_pt_bins=jet_pt_bins, hybrid_min_kt_values=matching_hybrid_min_kt_values, path=embedded_dataset.output)

    # Comparison
    # plot_results.compare_kt(all_data_hists=data_hists.hists, all_embedded_hists=embedded_hists.hybrid_hists, data_dataset=data_dataset, embedded_dataset=embedded_dataset)

    IPython.start_ipython(user_ns=locals())
