#!/usr/bin/env python3

"""

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import functools
import logging
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import attr
import awkward as ak
from networkx.drawing.nx_pylab import draw
import numba as nb
import numpy as np
import numpy.typing as npt
import uproot


# We know already - nothing to be done...
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import uproot3

from pachyderm import yaml

from jet_substructure.base import new_methods, skim_analysis_objects
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

    input_jets: ak.Array = attr.ib()
    input_splittings: new_methods.JetSplittingArray = attr.ib()
    input_splittings_indices: UprootArray[int] = attr.ib()
    values: UprootArray[float] = attr.ib()
    indices: UprootArray[int] = attr.ib()
    # If there's no additional grooming selection, then this will be identical to input_splittings_indices.
    possible_indices: UprootArray[int] = attr.ib()

    @property
    def splittings(self) -> new_methods.JetSplittingArray:
        try:
            return self._restricted_splittings
        except AttributeError:
            self._restricted_splittings: new_methods.JetSplittingArray = self.input_splittings[self.indices]
        return self._restricted_splittings

    @property
    def n_jets(self) -> int:
        """Number of jets."""
        # We flatten the splittings because there may be jets (and consequently splittings) which aren't selected
        # at all due to the grooming (such as a z cut). Thus, we use the selected splittings directly.
        return len(self.splittings.flatten())

    def __getitem__(self, mask: np.ndarray) -> Calculation:
        """Mask the stored values, returning a new object."""
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
class MaskedJets:
    """Container for masked jets.

    This just provides a consistent named interface to keep track of everything.
    """

    jets: ak.Array = attr.ib()
    selected_splittings: new_methods.JetSplittingArray = attr.ib()
    selected_splittings_index: UprootArray[int] = attr.ib()


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
    jet_R: float,
    iterative_splittings: bool,
) -> Dict[str, functools.partial[Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]]]:
    """Define the calculation functions of interest.

    Note:
        The type of the inclusive is different, but it takes and returns the same sets of arguments
        as the other functions.

    Args:
        jet_R: Jet resolution parameter.
        iterative_splittings: Whether calculating iterative splittings or not.
    Returns:
        dynamical_core, dynamical_z, dynamical_kt, dynamical_time, leading_kt, leading_kt z>0.2, leading_kt z>0.4, SD z>0.2, SD z>0.4
    """
    functions = {
        "dynamical_core": functools.partial(new_methods.JetSplittingArray.dynamical_core, R=jet_R),
        "dynamical_z": functools.partial(new_methods.JetSplittingArray.dynamical_z, R=jet_R),
        "dynamical_kt": functools.partial(new_methods.JetSplittingArray.dynamical_kt, R=jet_R),
        "dynamical_time": functools.partial(new_methods.JetSplittingArray.dynamical_time, R=jet_R),
        "leading_kt": functools.partial(
            new_methods.JetSplittingArray.leading_kt,
        ),
        "leading_kt_z_cut_02": functools.partial(new_methods.JetSplittingArray.leading_kt, z_cutoff=0.2),
        "leading_kt_z_cut_04": functools.partial(new_methods.JetSplittingArray.leading_kt, z_cutoff=0.4),
    }
    # TODO: This currently only works for iterative splittings...
    #       Calculating recursive is way harder in any array-like manner.
    if iterative_splittings:
        functions["soft_drop_z_cut_02"] = functools.partial(new_methods.JetSplittingArray.soft_drop, z_cutoff=0.2)
        functions["soft_drop_z_cut_04"] = functools.partial(new_methods.JetSplittingArray.soft_drop, z_cutoff=0.4)
    return functions


def _select_and_retrieve_splittings(
    jets: ak.Array, mask: UprootArray[bool], iterative_splittings: bool
) -> Tuple[ak.Array, new_methods.JetSplittingArray, UprootArray[int]]:
    """Generalization of the function in analyze_tree to add the splitting index."""
    # Ensure that there are sufficient counts
    restricted_jets = jets[mask]

    # Add the splittings and indices.
    if iterative_splittings:
        # Only keep iterative splittings.
        restricted_splittings = restricted_jets.jet_splittings.iterative_splittings(restricted_jets.subjets)

        # Enable this test to determine if we've selected different sets of splittings with the
        # recursive vs iterative selections.
        # if (splittings.counts != restricted_jets.jet_splittings.counts).any():
        #    logger.warning("Disagreement between number of inclusive and recursive splittings (as expected!)")
        #    IPython.embed()
        restricted_splittings_indices = restricted_jets.subjets.iterative_splitting_index
    else:
        restricted_splittings = restricted_jets.jet_splittings
        restricted_splittings_indices = restricted_jets.jet_splittings.kt.layout.localindex()

    return restricted_jets, restricted_splittings, restricted_splittings_indices


@nb.njit  # noqa: C901
def calculate_splitting_number(  # noqa: C901
    all_splittings: new_methods.JetSplittingArray,
    selected_splittings: new_methods.JetSplittingArray,
    restricted_splittings_indices: UprootArray[int],
    debug: bool = False,
) -> np.ndarray:
    # NOTE: I would like to use uint8 here, but it's not implemented in uproot3's writing.
    #       However, int8 gives us enough range, so it's fine to use it instead.
    #       Further, root doesn't handle int8 correctly, so we have to use int16.
    output = np.zeros(len(selected_splittings), dtype=np.int16)

    for i, (selected_splitting, restricted_splitting_indices, available_splittings_parents) in enumerate(
        zip(selected_splittings, restricted_splittings_indices, all_splittings.parent_index)
    ):
        # restricted_splitting_indices = restricted_splittings_indices[i]
        # available_splittings_parents = all_splittings[i].parent_index

        parent_indices = selected_splitting.parent_index
        if len(parent_indices):
            # We have at least one splitting, so we add an entry for it.
            output[i] += 1

            parent_index = parent_indices[0]
            if debug:
                print("parent_index", parent_index, "restricted_splitting_indices", restricted_splitting_indices)
            # print("i", i, "parent_indices", parent_indices, "parent_index", parent_index, "restricted_splitting_indices", restricted_splitting_indices)
            # if i == 27:
            #    print("parent_indices", parent_indices, "parent_index", parent_index, "restricted_splitting_indices", restricted_splitting_indices)
            while parent_index != -1:
                # Apparently contains isn't implemented either. So we just implement by hand.
                # if parent_index in restricted_splitting_indices:
                for index in restricted_splitting_indices:
                    # print("parent_index: {parent_index}, index: {index}".format(parent_index=parent_index, index=index))
                    if debug:
                        print("parent_index", parent_index, "index", index)
                    # print("parent_index, index: %d, %d" % (parent_index, index))
                    # print("i", i, "parent_index", parent_index, "index", index)
                    if parent_index == index:
                        if debug:
                            print("Found parent index:", index)
                        output[i] += 1
                        # import IPython; IPython.embed()
                        parent_index = available_splittings_parents[parent_index]
                        if debug:
                            print("New parent index:", parent_index)
                        # print("Breaking...")
                        break
                else:
                    # We didn't find it, but we need to advance forward.
                    parent_index = available_splittings_parents[parent_index]

            if debug:
                print("output[i]", output[i])

    return output


@nb.njit  # type: ignore
def _find_contributing_subjets(input_jet: ak.Array, groomed_index: int) -> List[ak.Array]:
    """Find subjets which contribute to a given grooming index.

    Args:
        input_jet: Inputs jets.
        groomed_index: Selected grooming index (ie. splitting).
    Returns:
        Subjets contributing to the splitting.
    """
    # subjets = []
    # for sj in input_jet.subjets:
    #    if sj.parent_splitting_index == groomed_index:
    #        subjets.append(sj)
    # return subjets
    return [sj for sj in input_jet.subjets if sj.parent_splitting_index == groomed_index]


@nb.njit
def _sort_subjets(input_jet, input_subjets):
    pts = []
    for sj in input_subjets:
        px = 0
        py = 0
        for constituent_index in sj.constituent_indices:
            constituent = input_jet.jet_constituents[constituent_index]
            px += constituent.pt * np.cos(constituent.phi)
            py += constituent.pt * np.sin(constituent.phi)
        pts.append(np.sqrt(px ** 2 + py ** 2))

    leading = input_subjets[0]
    subleading = input_subjets[1]

    if pts[1] > pts[0]:
        leading, subleading = subleading, leading

    return leading, subleading


@nb.njit  # type: ignore
def _subjet_shared_momentum(
    generator_like_subjet,
    generator_like_jet,
    measured_like_subjet,
    measured_like_jet,
    match_using_distance: bool = False,
) -> float:
    sum_pt = 0
    delta = new_methods.DISTANCE_DELTA

    for generator_like_constituent_index in generator_like_subjet.constituent_indices:
        generator_like_constituent = generator_like_jet.jet_constituents[generator_like_constituent_index]
        for measured_like_constituent_index in measured_like_subjet.constituent_indices:
            measured_like_constituent = measured_like_jet.jet_constituents[measured_like_constituent_index]
            if match_using_distance:
                if np.abs(measured_like_constituent.eta - generator_like_constituent.eta) > delta:
                    continue
                if np.abs(measured_like_constituent.phi - generator_like_constituent.phi) > delta:
                    continue
            else:
                if generator_like_constituent.id != measured_like_constituent.id:
                    continue

            sum_pt += generator_like_constituent.pt
            # We've matched once - no need to match again.
            # Otherwise, the run the risk of summing a generator-like constituent pt twice.
            break

    return sum_pt


@nb.njit  # type: ignore
def subjet_pt(subjet, jet) -> float:
    px = 0
    py = 0
    # pt = 0
    for constituent_index in subjet.constituent_indices:
        constituent = jet.jet_constituents[constituent_index]
        px += constituent.pt * np.cos(constituent.phi)
        py += constituent.pt * np.sin(constituent.phi)
        # pt += constituent.pt
    # return pt
    return np.sqrt(px ** 2 + py ** 2)


@nb.njit  # type: ignore
def _subjet_contained_in_subjet(
    generator_like_subjet,
    generator_like_jet,
    measured_like_subjet,
    measured_like_jet,
    match_using_distance: bool = False,
) -> bool:
    return (
        _subjet_shared_momentum(
            generator_like_subjet=generator_like_subjet,
            generator_like_jet=generator_like_jet,
            measured_like_subjet=measured_like_subjet,
            measured_like_jet=measured_like_jet,
            match_using_distance=match_using_distance,
        )
        / subjet_pt(generator_like_subjet, generator_like_jet)
    ) > 0.5


@nb.njit
def determine_matched_jets_numba(
    generator_like_jets,
    generator_like_splittings,
    generator_like_groomed_values,
    generator_like_groomed_indices,
    measured_like_jets,
    measured_like_splittings,
    measured_like_groomed_values,
    measured_like_groomed_indices,
    match_using_distance: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    n_jets = len(measured_like_jets)
    leading_matching = np.full(n_jets, -1, dtype=np.int16)
    subleading_matching = np.full(n_jets, -1, dtype=np.int16)

    for (
        i,
        (
            generator_like_jet,
            generator_like_splitting,
            generator_like_groomed_value,
            generator_like_groomed_index_array,
            measured_like_jet,
            measured_like_splitting,
            measured_like_groomed_value,
            measured_like_groomed_index_array,
        ),
    ) in enumerate(
        zip(
            generator_like_jets,
            generator_like_splittings,
            generator_like_groomed_values,
            generator_like_groomed_indices,
            measured_like_jets,
            measured_like_splittings,
            measured_like_groomed_values,
            measured_like_groomed_indices,
        )
    ):
        # Find the selected index if it's available.
        if len(measured_like_groomed_index_array) > 0 and len(generator_like_groomed_index_array) > 0:
            # This is required. If not, we handle the other cases and continue.
            pass
        elif len(measured_like_groomed_index_array) > 0:
            # Assign 0 for this case and move on.
            leading_matching[i] = 0
            subleading_matching[i] = 0
            continue
        else:
            # Use the default values and continue
            continue

        # We maintain the singles structure per jet so that each index can be applied to each jet (ie. array entry)
        # (this also lets us keep empty cases accounted for). However, we've now already accounted for empty cases,
        # and it's much easier to work with the individual values, so we extract them. We know each one will have only
        # one entry because it's from an argmax call.
        generator_like_groomed_index = generator_like_groomed_index_array[0]
        measured_like_groomed_index = measured_like_groomed_index_array[0]

        # Find the contributing subjets
        generator_like_subjets = _find_contributing_subjets(generator_like_jet, generator_like_groomed_index)
        measured_like_subjets = _find_contributing_subjets(measured_like_jet, measured_like_groomed_index)
        # print(measured_like_subjets)
        # Sort
        generator_like_leading, generator_like_subleading = _sort_subjets(generator_like_jet, generator_like_subjets)
        measured_like_leading, measured_like_subleading = _sort_subjets(measured_like_jet, measured_like_subjets)

        # Compare
        if _subjet_contained_in_subjet(
            generator_like_subjet=generator_like_leading,
            generator_like_jet=generator_like_jet,
            measured_like_subjet=measured_like_leading,
            measured_like_jet=measured_like_jet,
            match_using_distance=match_using_distance,
        ):
            leading_matching[i] = 1
        elif _subjet_contained_in_subjet(
            generator_like_subjet=generator_like_leading,
            generator_like_jet=generator_like_jet,
            measured_like_subjet=measured_like_subleading,
            measured_like_jet=measured_like_jet,
            match_using_distance=match_using_distance,
        ):
            leading_matching[i] = 2
        else:
            leading_matching[i] = 3

        if _subjet_contained_in_subjet(
            generator_like_subjet=generator_like_subleading,
            generator_like_jet=generator_like_jet,
            measured_like_subjet=measured_like_subleading,
            measured_like_jet=measured_like_jet,
            match_using_distance=match_using_distance,
        ):
            subleading_matching[i] = 1
        elif _subjet_contained_in_subjet(
            generator_like_subjet=generator_like_subleading,
            generator_like_jet=generator_like_jet,
            measured_like_subjet=measured_like_leading,
            measured_like_jet=measured_like_jet,
            match_using_distance=match_using_distance,
        ):
            subleading_matching[i] = 2
        else:
            subleading_matching[i] = 3

    return leading_matching, subleading_matching


def prong_matching_numba_wrapper(
    measured_like_jets_calculation: Calculation,
    measured_like_jets_label: str,
    generator_like_jets_calculation: Calculation,
    generator_like_jets_label: str,
    grooming_method: str,
    match_using_distance: bool = False,
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
    # Matching
    grooming_results = {}
    logger.debug(f"Performing {measured_like_jets_label}-{generator_like_jets_label} matching for {grooming_method}")
    leading_matching, subleading_matching = determine_matched_jets_numba(
        generator_like_jets=generator_like_jets_calculation.input_jets,
        generator_like_splittings=generator_like_jets_calculation.input_splittings,
        generator_like_groomed_values=generator_like_jets_calculation.values,
        generator_like_groomed_indices=generator_like_jets_calculation.indices,
        measured_like_jets=measured_like_jets_calculation.input_jets,
        measured_like_splittings=measured_like_jets_calculation.input_splittings,
        measured_like_groomed_values=measured_like_jets_calculation.values,
        measured_like_groomed_indices=measured_like_jets_calculation.indices,
        match_using_distance=match_using_distance,
    )

    for label, matching in [("leading", leading_matching), ("subleading", subleading_matching)]:
        grooming_results[
            f"{grooming_method}_{measured_like_jets_label}_{generator_like_jets_label}_matching_{label}"
        ] = matching

    return grooming_results


@nb.njit
def _subjet_momentum_fraction_in_jet(
    generator_like_subjet,
    generator_like_jet,
    measured_like_jet,
    match_using_distance: bool = False,
) -> float:
    """Calculate subjet momentum fraction contained within another jet.

    Unfortunately, we can't blindly use the `_subjet_shared_momentum` function because
    the interfaces vary between jet constituents and subjet constituents. We could refactor them,
    but the code is simple enough that it's easier just to implement the different versions.
    """
    sum_pt = 0
    delta = new_methods.DISTANCE_DELTA

    for generator_like_constituent_index in generator_like_subjet.constituent_indices:
        generator_like_constituent = generator_like_jet.jet_constituents[generator_like_constituent_index]
        for measured_like_constituent in measured_like_jet.jet_constituents:
            if match_using_distance:
                if np.abs(measured_like_constituent.eta - generator_like_constituent.eta) > delta:
                    continue
                if np.abs(measured_like_constituent.phi - generator_like_constituent.phi) > delta:
                    continue
            else:
                if generator_like_constituent.id != measured_like_constituent.id:
                    continue

            sum_pt += generator_like_constituent.pt
            # We've matched once - no need to match again.
            # Otherwise, the run the risk of summing a generator-like constituent pt twice.
            break

    return sum_pt / subjet_pt(generator_like_subjet, generator_like_jet)


@nb.njit
def generator_subjet_momentum_fraction_in_measured_jet_numba(
    generator_like_jets,
    generator_like_splittings,
    generator_like_groomed_indices,
    measured_like_jets,
) -> Tuple[np.ndarray, np.ndarray]:
    """Determine the generator-like subjet momentum fraction stored in a measured-like jet.

    Note:
        This isn't looking at the measured-like subjet. It's about finding where subjets go,
        especially for those which aren't matched at the subjet level (they of course must have
        matched at the overall jet level).
    """
    n_jets = len(measured_like_jets)
    leading_momentum_fraction = np.zeros(n_jets, dtype=np.float32)
    subleading_momentum_fraction = np.zeros(n_jets, dtype=np.float32)

    for (
        i,
        (
            generator_like_jet,
            generator_like_splitting,
            generator_like_groomed_index_array,
            measured_like_jet,
        ),
    ) in enumerate(
        zip(
            generator_like_jets,
            generator_like_splittings,
            generator_like_groomed_indices,
            measured_like_jets,
        )
    ):
        # Find the selected index if it's available.
        if len(generator_like_groomed_index_array) > 0:
            # This is required. Otherwise, we just skip case.
            pass
        else:
            # Use the default values and continue
            continue

        # Retrieve the generator like subjet.
        # We know each one will have only one entry because it's from an argmax call, so we extract it.
        generator_like_groomed_index = generator_like_groomed_index_array[0]
        # Find the contributing subjets
        generator_like_subjets = _find_contributing_subjets(generator_like_jet, generator_like_groomed_index)
        # Sort
        generator_like_leading, generator_like_subleading = _sort_subjets(generator_like_jet, generator_like_subjets)

        leading_momentum_fraction[i] = _subjet_momentum_fraction_in_jet(
            generator_like_subjet=generator_like_leading,
            generator_like_jet=generator_like_jet,
            measured_like_jet=measured_like_jet,
        )
        subleading_momentum_fraction[i] = _subjet_momentum_fraction_in_jet(
            generator_like_subjet=generator_like_subleading,
            generator_like_jet=generator_like_jet,
            measured_like_jet=measured_like_jet,
        )

    return leading_momentum_fraction, subleading_momentum_fraction


def generator_subjet_momentum_fraction_in_measured_jet_numba_wrapper(
    measured_like_jets_calculation: Calculation,
    measured_like_jets_label: str,
    generator_like_jets_calculation: Calculation,
    generator_like_jets_label: str,
    grooming_method: str,
) -> Dict[str, np.ndarray]:
    grooming_results = {}

    leading_momentum_fraction, subleading_momentum_fraction = generator_subjet_momentum_fraction_in_measured_jet_numba(
        generator_like_jets=generator_like_jets_calculation.input_jets,
        generator_like_splittings=generator_like_jets_calculation.input_splittings,
        generator_like_groomed_indices=generator_like_jets_calculation.indices,
        measured_like_jets=measured_like_jets_calculation.input_jets,
    )

    for label, momentum_fraction in [
        ("leading", leading_momentum_fraction),
        ("subleading", subleading_momentum_fraction),
    ]:
        # groomingMethod + "_hybrid_det_level_matching_leading_pt_fraction_in_hybrid_jet"
        # NOTE: This is different than the name in the hardest kt cross check task. Since I had more time to think about it,
        #       this name makes more sense to me.
        grooming_results[
            f"{grooming_method}_{generator_like_jets_label}_{label}_subjet_momentum_fraction_in_{measured_like_jets_label}_jet"
        ] = momentum_fraction

    return grooming_results


def _calculate_jet_kinematics(
    constituents: new_methods.JetConstituentArray, float_type: Optional[npt.DTypeLike] = None
) -> Tuple[ak.Array, ak.Array]:
    """Calculate jet kinematics.

    Since `vector` isn't yet available, we perform the four vector calculations by hand.

    Args:
        constituents: Jet constituents.
        float_type: Float to be used for conversion. Default: None. This uses the ak default.
    Returns:
        eta, phi
    """
    # jet_four_vec = jets.jet_constituents.four_vectors().sum()
    # Since vector isn't ready yet, just do this by hand...
    px = ak.sum(constituents.pt * np.cos(constituents.phi), axis=1)  # type: ignore
    py = ak.sum(constituents.pt * np.sin(constituents.phi), axis=1)  # type: ignore
    pz = ak.sum(constituents.pt * np.sinh(constituents.eta), axis=1)  # type: ignore
    # Formulas just from inverting the above.
    eta = np.arcsinh(pz / np.sqrt(px ** 2 + py ** 2))
    phi = np.arctan2(py, px)
    if float_type is None:
        return eta, phi
    else:
        return ak.values_astype(eta, float_type), ak.values_astype(phi, float_type)


def calculate_embedding_skim_impl(  # noqa: C901
    all_jets: ak. Array,
    input_filename: Path,
    iterative_splittings: bool,
    prefixes: Mapping[str, str],
    # TODO: For full embedding with the track skim, this could be a problem.
    #       However, it will be fine to the thermal model, since we only embed one pt hard bin
    #scale_factors: Mapping[int, float],
    scale_factor: float,
    jet_R: float,
    output_filename: Path,
    output_tree_name: str = "tree",
    create_friend_tree: bool = False,
    draw_example_splittings: bool = False,
    write_feather: bool = False,
    write_parquet: bool = False,
) -> Tuple[bool, Path, str]:
    """Determine the response and prong matching for jets substructure techniques.

    Args:
        input_filename: Input file path.
        iterative_splittings: If True, we should only consider iterative splittings.
        prefixes: Mapping from our standard names to those which are used in the stored data.
        create_friend_tree: Create a friend tree instead of the standard tree. It contains
            supplemental information. See the code for precisely what it contains. Default: False.
        draw_example_splittings: If True, draw a few interesting splitting graphs. Default: False.
    """
    # Validation
    # Bail out early if the file already exists.
    if output_filename.exists():
        return True, output_filename, "already exists"

    # Output consistent types.
    float_type = np.float32
    to_float = functools.partial(ak.values_astype, to=np.float32)

    # Jets setup.
    logger.info(f"Skimming tree from file {input_filename}")
    # true_jets = all_jets["matched"]
    # det_level_jets = all_jets["detLevel"]
    # hybrid_jets = all_jets["data"]

    # Do the calculations
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
    mask = all_jets["hybrid"].jet_pt > 0

    # Mask the jets
    masked_jets: Dict[str, MaskedJets] = {}
    for prefix, input_jets in all_jets.items():
        masked_jets[prefix] = MaskedJets(
            *_select_and_retrieve_splittings(
                input_jets,
                mask,
                iterative_splittings=iterative_splittings,
            )
        )

    # Results output
    grooming_results = {}
    if create_friend_tree:
        # Extract eta-phi of jets.
        output_filename = Path(str(output_filename.with_suffix("")) + "_friend.root")
        # As the skim is re-run, values are generally transitioned to the standard tree the next time it's generated.
    else:
        grooming_results["scale_factor"] = to_float((masked_jets["true"].jets.jet_pt[mask] * 0) + scale_factor)

        for prefix, input_jets in masked_jets.items():
            # Add jet pt and general jet properties.
            # Jet kinematics
            grooming_results[f"{prefix}_jet_pt"] = to_float(input_jets.jets.jet_pt)
            grooming_results[f"{prefix}_jet_eta"], grooming_results[f"{prefix}_jet_phi"] = _calculate_jet_kinematics(
                input_jets.jets.jet_constituents,
                float_type=float_type,
            )
            # Leading track
            leading_track_name = f"{prefix}_leading_track_pt"
            if prefix == "hybrid":
                # First, store the unsubtracted (which we use for the double counting cut) as the normal leading track pt.
                if "leading_track_pt" in ak.fields(input_jets.jets):
                    grooming_results[leading_track_name] = to_float(input_jets.jets["leading_track_pt"])
                # Then update the name for the substracted constituents in data.
                leading_track_name = f"{prefix}_leading_track_pt_sub"
            grooming_results[leading_track_name] = to_float(ak.max(input_jets.jets.jet_constituents.pt, axis=1))

        # Perform our calculations.
        functions = _define_calculation_functions(jet_R=jet_R, iterative_splittings=iterative_splittings)
        for func_name, func in functions.items():
            logger.debug(f"func_name: {func_name}")
            calculations = {
                prefix: Calculation(
                    masked_jets[prefix].jets,
                    masked_jets[prefix].selected_splittings,
                    masked_jets[prefix].selected_splittings_index,
                    *func(masked_jets[prefix].selected_splittings),
                )
                for prefix in prefixes
            }

            for prefix, calculation in calculations.items():
                # Calculate splitting number for the appropriate cases.
                groomed_splittings = calculation.splittings
                # Number of splittings until the selected splitting, irrespective of the grooming conditions.
                n_to_split = calculate_splitting_number(
                    all_splittings=calculation.input_jets.jet_splittings,
                    selected_splittings=groomed_splittings,
                    # Need all splitting indices (unrestricted by any possible grooming selections).
                    restricted_splittings_indices=calculation.input_splittings_indices,
                )
                logger.debug(f"Done with first splitting calculation, {prefix}")
                # Number of splittings which pass the grooming conditions until the selected splitting.
                n_groomed_to_split = calculate_splitting_number(
                    all_splittings=calculation.input_jets.jet_splittings,
                    selected_splittings=groomed_splittings,
                    # Need the indices that correspond to the splittings that pass the grooming.
                    restricted_splittings_indices=calculation.possible_indices,
                    debug=False,
                )
                logger.debug(f"Done with second splitting calculation, {prefix}")

                # We pad with the UNFILLED_VALUE constant to account for any calculations that don't find a splitting.
                grooming_result = GroomingResultForTree(
                    grooming_method=func_name,
                    delta_R=to_float(
                        ak.flatten(ak.fill_none(ak.pad_none(groomed_splittings.delta_R, 1), new_methods.UNFILLED_VALUE))
                    ),
                    z=to_float(
                        ak.flatten(ak.fill_none(ak.pad_none(groomed_splittings.z, 1), new_methods.UNFILLED_VALUE))
                    ),
                    kt=to_float(
                        ak.flatten(ak.fill_none(ak.pad_none(groomed_splittings.kt, 1), new_methods.UNFILLED_VALUE))
                    ),
                    # All of the numbers are already flattened. 0 means untagged.
                    n_to_split=n_to_split,
                    n_groomed_to_split=n_groomed_to_split,
                    # Number of splittings which pass the grooming condition. For SoftDrop, this is n_sd.
                    # NOTE: I would like to use uint8 here, but it's not implemented in uproot3's writing.
                    #       However, int8 gives us enough range, so it's fine to use it instead.
                    #       Further, root doesn't handle int8 correctly, so we have to use int16.
                    n_passed_grooming=ak.values_astype(ak.num(calculation.possible_indices, axis=1), np.int16),
                )
                grooming_results.update(grooming_result.asdict(prefix=prefix))

            logger.debug("Before prong matching")
            # IPython.embed()
            # Hybrid-det level matching.
            # We match using distance here because the labels don't align anymore due to the subtraction mixing the labels.
            hybrid_det_level_matching_results = prong_matching_numba_wrapper(
                measured_like_jets_calculation=calculations["hybrid"],
                measured_like_jets_label="hybrid",
                generator_like_jets_calculation=calculations["det_level"],
                generator_like_jets_label="det_level",
                grooming_method=func_name,
                match_using_distance=False,
            )
            grooming_results.update(hybrid_det_level_matching_results)
            logger.debug("Done with first prong matching")
            # Det level-true matching
            # We match using labels here because otherwise the reconstruction can cause the particles to move
            # enough that they may not match within a particular distance.
            det_level_true_matching_results = prong_matching_numba_wrapper(
                measured_like_jets_calculation=calculations["det_level"],
                measured_like_jets_label="det_level",
                generator_like_jets_calculation=calculations["true"],
                generator_like_jets_label="true",
                grooming_method=func_name,
                match_using_distance=False,
            )
            grooming_results.update(det_level_true_matching_results)
            logger.debug("Done with second prong matching")
            # Subjet momentum fraction in hybrid
            grooming_results.update(
                generator_subjet_momentum_fraction_in_measured_jet_numba_wrapper(
                    measured_like_jets_calculation=calculations["hybrid"],
                    measured_like_jets_label="hybrid",
                    generator_like_jets_calculation=calculations["det_level"],
                    generator_like_jets_label="det_level",
                    grooming_method=func_name,
                )
            )
            logger.debug("Done with det level subjet momentum fraction in hybrid jets")

            # Look for leading kt just because it's easier to understand conceptually.
            hybrid_det_level_leading_matching = grooming_results[f"{func_name}_hybrid_det_level_matching_leading"]
            hybrid_det_level_subleading_matching = grooming_results[f"{func_name}_hybrid_det_level_matching_subleading"]
            if (
                draw_example_splittings
                and func_name == "leading_kt"
                and ak.any((hybrid_det_level_leading_matching == 1) & (hybrid_det_level_subleading_matching == 3))
            ):
                from jet_substructure.analysis import draw_splitting

                # Find a sufficiently interesting jet (ie high enough pt)
                mask_jets_of_interest = (
                    (hybrid_det_level_leading_matching.properly & hybrid_det_level_subleading_matching.failed)
                    & (masked_jets["hybrid"].jets.jet_pt > 80)
                    & (calculations["det_level"].splittings.kt > 10).flatten()
                )

                # Look at most the first 5 jets.
                for i, hybrid_jet in enumerate(masked_jets["hybrid"].jets[mask_jets_of_interest][:5]):
                    # Find the hybrid jet and splitting of interest.
                    # hybrid_jet = masked_hybrid_jets[mask_jets_of_interest][0]
                    # Take the index of the splitting of interest. We want the first jet, and then there must be one splitting index there.
                    hybrid_jet_selected_splitting_index = calculations["hybrid"].indices[mask_jets_of_interest][i][0]
                    # Same for det level.
                    det_level_jet = masked_jets["det_level"].jets[mask_jets_of_interest][i]
                    # Take the index of the splitting of interest. We want the first jet, and then there must be one splitting index there.
                    det_level_jet_selected_splitting_index = calculations["det_level"].indices[mask_jets_of_interest][
                        i
                    ][0]

                    splitting_graph_output_dir = output_filename.parent

                    # Draw the splittings
                    draw_splitting.splittings_graph(
                        jet=hybrid_jet,
                        path=splitting_graph_output_dir / "leading_correct_subleading_failed/",
                        filename=f"{i}_hybrid_splittings_jet_pt_{hybrid_jet.jet_pt:.1f}GeV_selected_splitting_index_{hybrid_jet_selected_splitting_index}",
                        show_subjet_pt=True,
                        selected_splitting_index=hybrid_jet_selected_splitting_index,
                    )
                    draw_splitting.splittings_graph(
                        jet=det_level_jet,
                        path=splitting_graph_output_dir / "leading_correct_subleading_failed/",
                        filename=f"{i}_det_level_splittings_jet_pt_{det_level_jet.jet_pt:.1f}GeV_selected_splitting_index_{det_level_jet_selected_splitting_index}",
                        show_subjet_pt=True,
                        selected_splitting_index=det_level_jet_selected_splitting_index,
                    )

            logger.debug(f"Completed {func_name}")

    # For extra safety
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    # First, convert to numpy since we want to write to an output tree.
    grooming_results_np = {k: np.asarray(v) for k, v in grooming_results.items()}
    #branches = {k: v.dtype for k, v in grooming_results_np.items()}
    logger.info(f"Writing embedding skim to {output_filename}")
    # Write with uproot
    #uproot3_filename = output_filename.parent / f"{output_filename.stem}_uproot3{output_filename.suffix}"
    #with uproot3.recreate(uproot3_filename) as output_file:
    #    output_file[output_tree_name] = uproot3.newtree(branches)
    #    # Write all of the calculations
    #    output_file[output_tree_name].extend(grooming_results_np)
    #uproot4_filename = output_filename.parent / f"{output_filename.stem}_uproot4{output_filename.suffix}"
    #with uproot.recreate(uproot4_filename) as output_file:
    with uproot.recreate(output_filename) as output_file:
        # Write all of the calculations
        output_file[output_tree_name] = grooming_results_np

    # Some alternative formats for other analysis techniques.
    if write_parquet:
        logger.info("Writing parquet...")
        ak.to_parquet(grooming_results, output_filename.with_suffix(".parquet"), compression="zstd")
    if write_feather:
        logger.info("Writing feather...")
        import pyarrow.feather

        pa_table = ak.to_arrow_table(grooming_results)
        pyarrow.feather.write_feather(pa_table, output_filename.with_suffix(".feather"), compression="zstd")

    logger.info(f"Finished processing tree from file {input_filename}")
    return True, output_filename, "processed"


def calculate_embedding_skim(  # noqa: C901
    input_filename: Path,
    iterative_splittings: bool,
    prefixes: Mapping[str, str],
    scale_factors: Mapping[int, float],
    train_directory: Path,
    jet_R: float,
    output_filename: Path,
    output_tree_name: str = "tree",
    create_friend_tree: bool = False,
    draw_example_splittings: bool = False,
    write_feather: bool = False,
    write_parquet: bool = False,
) -> Tuple[bool, Path, str]:
    # Validation
    # Bail out early if the file already exists.
    if output_filename.exists():
        return True, output_filename, "already exists"

    # Setup
    # Use the train configuration to extract the train number and pt hard bin, which are used to get the scale factor.
    y = yaml.yaml()
    with open(train_directory / "config.yaml", "r") as f:
        train_config = y.load(f)
    train_number = train_config["number"]
    pt_hard_bin = train_config["pt_hard_bin"]
    logger.debug(f"Extracted train number: {train_number}, pt hard bin: {pt_hard_bin}")
    scale_factor = scale_factors[pt_hard_bin]

    # Jets setup.
    logger.info(f"Skimming tree from file {input_filename}")
    all_jets = new_methods.parquet_to_substructure_analysis(filename=input_filename, prefixes=prefixes)

    return calculate_embedding_skim_impl(
        all_jets=all_jets,
        input_filename=input_filename,
        iterative_splittings=iterative_splittings,
        prefixes=prefixes,
        scale_factor=scale_factor,
        jet_R=jet_R,
        output_filename=output_filename,
        output_tree_name=output_tree_name,
        create_friend_tree=create_friend_tree,
        draw_example_splittings=draw_example_splittings,
        write_feather=write_feather,
        write_parquet=write_parquet,
    )


def calculate_data_skim_impl(  # noqa: C901
    all_jets: ak.Array,
    input_filename: Path,
    collision_system: str,
    iterative_splittings: bool,
    prefixes: Mapping[str, str],
    jet_R: float,
    output_filename: Path,
    output_tree_name: str = "tree",
    create_friend_tree: bool = False,
    scale_factors: Optional[Mapping[int, float]] = None,
    write_feather: bool = False,
    write_parquet: bool = False,
) -> Tuple[bool, Path, str]:
    # Setup
    # Output consistent types.
    float_type = np.float32
    to_float = functools.partial(ak.values_astype, to=np.float32)

    # Dataset wide masks
    # Select everything by default. We know that there must be at least one set of jets, so we're safe to select on 0.
    mask = all_jets[list(prefixes.keys())[0]].jet_pt > 0
    # Special selections for pythia.
    # Apparently I can get pt hard < 5. Which is bizarre, at least according to the binning...
    # Filter these out when applicable.
    if collision_system == "pythia" and "pt_hard" in all_jets:
        # The jets object will contain the pt hard bin if it's available.
        mask = mask & (all_jets["pt_hard"] >= 5.0)

    masked_jets: Dict[str, MaskedJets] = {}
    # for prefix, input_jets in all_jets.items():
    for prefix in prefixes:
        input_jets = all_jets[prefix]
        masked_jets[prefix] = MaskedJets(
            *_select_and_retrieve_splittings(
                input_jets,
                mask,
                iterative_splittings=iterative_splittings,
            )
        )

    # Results output
    grooming_results = {}
    # And start constructing the tree
    if create_friend_tree:
        # Extract eta-phi of jets.
        output_filename = Path(str(output_filename.with_suffix("")) + "_friend.root")
        # As the skim is re-run, values are generally transitioned to the standard tree the next time it's generated.
    else:
        for prefix, input_jets in masked_jets.items():
            # Add jet pt and general jet properties.
            # Jet kinematics
            grooming_results[f"{prefix}_jet_pt"] = to_float(input_jets.jets.jet_pt)
            grooming_results[f"{prefix}_jet_eta"], grooming_results[f"{prefix}_jet_phi"] = _calculate_jet_kinematics(
                input_jets.jets.jet_constituents,
                float_type=float_type,
            )
            # Leading track
            # NOTE: Since this is for data, it doesn't really matter, but better to always do the right thing.
            leading_track_name = f"{prefix}_leading_track_pt"
            # NOTE: We would include embedPythia here, but we don't run the embedding through this function, so we can ignore it.
            if prefix == "data" and collision_system == "PbPb":
                # First, store the unsubtracted (which we use for the double counting cut).
                if "leading_track_pt" in ak.fields(input_jets.jets):
                    grooming_results[leading_track_name] = to_float(input_jets.jets["leading_track_pt"])
                # Then update the name for the subtracted constituents in data.
                leading_track_name = f"{prefix}_leading_track_pt_sub"
            grooming_results[leading_track_name] = to_float(ak.max(input_jets.jets.jet_constituents.pt, axis=1))

            # Perform our calculations.
            functions = _define_calculation_functions(jet_R=jet_R, iterative_splittings=iterative_splittings)
            for func_name, func in functions.items():
                logger.debug(f"prefix: {prefix}, grooming function: {func_name}")
                calculation = Calculation(
                    input_jets.jets,
                    input_jets.selected_splittings,
                    input_jets.selected_splittings_index,
                    *func(input_jets.selected_splittings),
                )
                # Calculate splitting number for the appropriate cases.
                groomed_splittings = calculation.splittings
                # Number of splittings until the selected splitting, irrespective of the grooming conditions.
                n_to_split = calculate_splitting_number(
                    all_splittings=calculation.input_jets.jet_splittings,
                    selected_splittings=groomed_splittings,
                    # Need all splitting indices (unrestricted by any possible grooming selections).
                    restricted_splittings_indices=calculation.input_splittings_indices,
                )
                logger.debug("Done with first splitting calculation")
                # Number of splittings which pass the grooming conditions until the selected splitting.
                n_groomed_to_split = calculate_splitting_number(
                    all_splittings=calculation.input_jets.jet_splittings,
                    selected_splittings=groomed_splittings,
                    # Need the indices that correspond to the splittings that pass the grooming.
                    restricted_splittings_indices=calculation.possible_indices,
                    debug=False,
                )
                logger.debug("Done with second splitting calculation")

                # We pad with the UNFILLED_VALUE constant to account for any calculations that don't find a splitting.
                grooming_result = GroomingResultForTree(
                    grooming_method=func_name,
                    delta_R=to_float(
                        ak.flatten(ak.fill_none(ak.pad_none(groomed_splittings.delta_R, 1), new_methods.UNFILLED_VALUE))
                    ),
                    z=to_float(
                        ak.flatten(ak.fill_none(ak.pad_none(groomed_splittings.z, 1), new_methods.UNFILLED_VALUE))
                    ),
                    kt=to_float(
                        ak.flatten(ak.fill_none(ak.pad_none(groomed_splittings.kt, 1), new_methods.UNFILLED_VALUE))
                    ),
                    # All of the numbers are already flattened. 0 means untagged.
                    n_to_split=n_to_split,
                    n_groomed_to_split=n_groomed_to_split,
                    # Number of splittings which pass the grooming condition. For SoftDrop, this is n_sd.
                    # NOTE: I would like to use uint8 here, but it's not implemented in uproot3's writing.
                    #       However, int8 gives us enough range, so it's fine to use it instead.
                    #       Further, root doesn't handle int8 correctly, so we have to use int16.
                    n_passed_grooming=ak.values_astype(ak.num(calculation.possible_indices, axis=1), np.int16),
                )
                grooming_results.update(grooming_result.asdict(prefix=prefix))

        # Add scale factors when appropriate (ie for pythia)
        if collision_system == "pythia":
            # Help out mypy...
            assert scale_factors is not None
            # Validation. We make a copy to ensure that we don't modify the input.
            output_scale_factors = dict(scale_factors)

            # There is apparently a pt hard > 1000 in this dataset! This ends up with an entry in bin 21, which is weird.
            # So we copy the scale factor for pt hard bin 20 to 21 to cover it. It should be more or less correct.
            output_scale_factors[21] = output_scale_factors[20]

            # Need to mask because we didn't when masking the original jets.
            pt_hard_bins = np.array(all_jets["pt_hard_bin"][mask], dtype=np.int16)
            logger.debug(f"Pt hard bins contained in the file: {np.unique(pt_hard_bins)}")  # type: ignore
            pythia_specific_columns = {
                "scale_factor": np.array([output_scale_factors[b] for b in pt_hard_bins], dtype=np.float32),
                "pt_hard_bin": pt_hard_bins,
            }
            # The track skim doesn't have this info available, so we might have to skip it.
            if "pt_hard" in all_jets:
                pythia_specific_columns["pt_hard"] = to_float(all_jets["pt_hard"][mask])
            grooming_results.update(pythia_specific_columns)

    # For extra safety
    output_filename.parent.mkdir(parents=True, exist_ok=True)
    # First, convert to numpy since we want to write to an output tree.
    grooming_results_np = {k: np.asarray(v) for k, v in grooming_results.items()}
    branches = {k: v.dtype for k, v in grooming_results_np.items()}
    logger.info(f"Writing data skim to {output_filename}")
    # Write with uproot
    with uproot3.recreate(output_filename) as output_file:
        output_file[output_tree_name] = uproot3.newtree(branches)
        # Write all of the calculations
        output_file[output_tree_name].extend(grooming_results_np)

    # Some alternative formats for other analysis techniques.
    if write_parquet:
        logger.info("Writing parquet...")
        ak.to_parquet(grooming_results, output_filename.with_suffix(".parquet"), compression="zstd")
    if write_feather:
        logger.info("Writing feather...")
        import pyarrow.feather

        pa_table = ak.to_arrow_table(grooming_results)
        pyarrow.feather.write_feather(pa_table, output_filename.with_suffix(".feather"), compression="zstd")

    logger.info(f"Finished processing tree from file {input_filename}")
    return True, output_filename, "processed"


def calculate_data_skim(  # noqa: C901
    input_filename: Path,
    collision_system: str,
    iterative_splittings: bool,
    prefixes: Mapping[str, str],
    jet_R: float,
    output_filename: Path,
    output_tree_name: str = "tree",
    create_friend_tree: bool = False,
    scale_factors: Optional[Mapping[int, float]] = None,
    write_feather: bool = False,
    write_parquet: bool = False,
) -> Tuple[bool, Path, str]:
    # Validation
    if scale_factors is None and collision_system == "pythia":
        raise ValueError("Need scale factors for pythia to be provided externally.")
    # Bail out early if the file already exists.
    if output_filename.exists():
        return True, output_filename, "already exists"

    # Jets setup
    logger.info(f"Skimming tree from file {input_filename}")
    # Careful, this can return general columns, not just jets in prefixes (for example, the pt_hard in pythia)
    all_jets = new_methods.parquet_to_substructure_analysis(filename=input_filename, prefixes=prefixes)

    return calculate_data_skim_impl(
        all_jets=all_jets,
        input_filename=input_filename,
        collision_system=collision_system,
        iterative_splittings=iterative_splittings,
        prefixes=prefixes,
        jet_R=jet_R,
        output_filename=output_filename,
        output_tree_name=output_tree_name,
        create_friend_tree=create_friend_tree,
        scale_factors=scale_factors,
        write_feather=write_feather,
        write_parquet=write_parquet,
    )


def cross_check_task_names_to_export(
    grooming_method: str,
    prefixes: Mapping[str, str],
) -> Dict[str, npt.DTypeLike]:
    branch_names: Dict[str, npt.DTypeLike] = {}

    substructure_variables = [
        "{grooming_method}_{prefix}_delta_R",
        "{grooming_method}_{prefix}_kt",
        "{grooming_method}_{prefix}_z",
        "{grooming_method}_{prefix}_n_to_split",
        "{grooming_method}_{prefix}_n_groomed_to_split",
        "{grooming_method}_{prefix}_n_passed_grooming",
    ]

    # Contain 8 * 3 + 1 (scale_factor) + 1 (hybrid_leading_track_pt_sub)
    branch_names["scale_factor"] = np.float32
    for prefix in prefixes:
        # Jet properties
        for var_name in ["{prefix}_jet_pt", "{prefix}_leading_track_pt"]:
            branch_names[var_name.format(prefix=prefix)] = np.float32
        if prefix == "hybrid":
            branch_names["hybrid_leading_track_pt_sub"] = np.float32

        # Substructure properties
        for var_name in substructure_variables:
            branch_names[var_name.format(grooming_method=grooming_method, prefix=prefix)] = np.float32

    # Matching properties
    for measured_like, generator_like in [("det_level", "true"), ("hybrid", "det_level")]:
        for level in ["leading", "subleading"]:
            branch_names[f"{grooming_method}_{measured_like}_{generator_like}_matching_{level}"] = np.int16
            if measured_like == "hybrid":
                branch_names[
                    f"{grooming_method}_{generator_like}_{level}_subjet_momentum_fraction_in_{measured_like}_jet"
                ] = np.float32

    return branch_names


def skim_cross_check_task_to_uniform_output(
    input_filename: Path,
    grooming_method: str,
    input_tree_name: str,
    scale_factor: float,
    prefixes: Mapping[str, str],
    output_filename: Path,
    output_tree_name: str = "tree",
) -> bool:
    """Skim the cross-check task to uniform names and types.

    Args:
        input_filename: Input filename. Expected to be in the main directory.
        grooming_method: Name of the grooming method.
        input_tree_name: Name of the input tree.
        scale_factor: Scale factor to be written to the tree. We assume it is
            constant for the given file (ie. suitable for embedding trains).
        prefixes: Mapping from our standard names to those which are used in
            the stored data.
        output_filename: Output filename. It's expected to be stored in the
            "skim" directory.
        output_tree_name: Name of the output tree. Default: "tree".

    Returns:
        True if successful.
    """
    filename = f"{str(input_filename)}:{input_tree_name}"
    # Iterate over relatively small sizes to ensure that we don't blow up the memory usage.
    for i, array in enumerate(uproot.iterate(filename, step_size="200 MB")):
        renames = skim_analysis_objects.cross_check_task_branch_name_shim(
            grooming_method=grooming_method,
            input_branches=ak.fields(array),
        )
        # This shouldn't cause a copy (hopefully...)
        for k, v in renames.items():
            array[k] = array[v]

        # Use array[k] as a convenient way to access the length of the tree.
        array["scale_factor"] = np.full(len(array[k]), scale_factor, dtype=np.float32)

        # Determine the branch names to export. It's basically the same as in the
        # original files, but with uniform names and types.
        branch_names = cross_check_task_names_to_export(grooming_method=grooming_method, prefixes=prefixes)

        # For extra safety
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        # Remap from "00" -> "index number" (may still be "00")
        # NOTE: It's super important that we include the punctuation. Otherwise, it may pick
        #       up other instances of "00", such as in a train number!
        _output_filename = Path(str(output_filename).replace(".00_", f".{i:02d}_"))

        # Setup outputs and write the tree.
        outputs = {}
        for k, v in branch_names.items():
            outputs[k] = ak.values_astype(array[k], to=v)
        outputs_np = {k: np.asarray(v) for k, v in outputs.items()}
        branches = {k: v.dtype for k, v in outputs_np.items()}
        logger.info(f"Writing cross check task skim to {_output_filename}")
        # Write with uproot
        with uproot3.recreate(_output_filename) as output_file:
            output_file[output_tree_name] = uproot3.newtree(branches)
            # Write all of the calculations
            output_file[output_tree_name].extend(outputs_np)

    return True


if __name__ == "__main__":
    # An example for testing...
    from jet_substructure.base import helpers

    helpers.setup_logging()
    # res = calculate_embedding_skim(
    #     input_filename=Path(
    #         "trains/embedPythia/6650/parquet/events_per_job_100000/AnalysisResults.18q.repaired.00.parquet"
    #     ),
    #     iterative_splittings=True,
    #     prefixes={"hybrid": "data", "true": "matched", "det_level": "detLevel"},
    #     scale_factors={1: 16.0695},
    #     train_directory=Path("trains/embedPythia/6650/"),
    #     jet_R=0.4,
    #     output_filename=Path(
    #         "trains/embedPythia/6650/skim/test/AnalysisResults.18q.repaired.00_iterative_splittings.root"
    #     ),
    #     write_parquet=True,
    #     write_feather=True,
    # )
    # Skim data.
    collision_system = "pythia"
    train_number = 2461
    res = calculate_data_skim(
        input_filename=Path(
            f"trains/{collision_system}/{train_number}/parquet/events_per_job_200000/AnalysisResults.cent_woSDD.01.repaired.00.parquet"
        ),
        collision_system=collision_system,
        iterative_splittings=True,
        prefixes={
            "data": "data",
            "true": "matched",
        },
        # These are wrong, but we need to simulate all of them being available for testing.
        scale_factors={pt_hard_bin: 16.0695 for pt_hard_bin in range(1, 21)},
        jet_R=0.2,
        output_filename=Path(
            f"trains/{collision_system}/{train_number}/skim/test/AnalysisResults.cent_woSDD.01.repaired.00_iterative_splittings.root"
        ),
        write_parquet=True,
        write_feather=True,
    )
    # Skim cross check task.
    # res = skim_cross_check_task_to_uniform_output(
    #     # n_cores=2,
    #     # input_filenames=[Path("trains/embedPythia/6458/AnalysisResults.18q.root")],
    #     input_filename=Path("trains/embedPythia/6474/AnalysisResults.18q.root"),
    #     grooming_method="dynamical_core",
    #     scale_factor=16.1,
    #     input_tree_name="AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR020_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
    #     prefixes={"hybrid": "data", "true": "matched", "det_level": "det_level"},
    #     output_filename=Path("trains/embedPythia/6474/skim/AnalysisResults.18q.root"),
    # )
    logger.info(res)
    # import IPython; IPython.start_ipython(user_ns=locals())
