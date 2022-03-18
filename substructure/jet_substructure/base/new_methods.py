""" Uproot4 + awkward1 substructure methods.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import functools
import logging
import typing
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar, cast


try:
    from typing import Final  # type: ignore
except ImportError:
    from typing_extensions import Final

import awkward as ak
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import uproot
import vector

from jet_substructure.base.helpers import ArrayOrScalar, UprootArray


logger = logging.getLogger(__name__)


# Typing helpers
_T = TypeVar("_T")


# Constants
# This value corresponds to an unidentified splitting.
UNFILLED_VALUE: Final[float] = -0.005
DISTANCE_DELTA: Final[float] = 0.01


@typing.overload
def _dynamical_hardness_measure(
    delta_R: UprootArray[float], z: UprootArray[float], parent_pt: UprootArray[float], R: float, a: float
) -> UprootArray[float]:
    ...


@typing.overload
def _dynamical_hardness_measure(delta_R: float, z: float, parent_pt: float, R: float, a: float) -> float:
    ...


def _dynamical_hardness_measure(delta_R, z, parent_pt, R, a):  # type: ignore
    """Implements the dynamical hardness measure used in dynamical grooming.

    Args:
        delta_R: Splitting delta R.
        z: Splitting z.
        parent_pt: Pt of the parent of the splitting.
        R: Jet resolution parameter.
        a: Dynamical grooming parameter, a.
    Returns:
        The hardness of the splitting according to the measure.
    """
    return z * (1 - z) * parent_pt * (delta_R / R) ** a


dynamical_core = functools.partial(_dynamical_hardness_measure, a=0.5)
dynamical_core.__doc__ = """
Calculates dynamical core (a = 0.5).

Args:
    delta_R: Splitting delta R.
    z: Splitting z.
    parent_pt: Pt of the parent of the splitting.
    R: Jet resolution parameter.
Returns:
    The hardness of the splitting according to the measure.
"""

dynamical_z = functools.partial(_dynamical_hardness_measure, a=0.1)
dynamical_z.__doc__ = """
Calculates dynamical z (a = 0.1). Also known as zDrop.

Args:
    delta_R: Splitting delta R.
    z: Splitting z.
    parent_pt: Pt of the parent of the splitting.
    R: Jet resolution parameter.
Returns:
    The hardness of the splitting according to the measure.
"""

dynamical_kt = functools.partial(_dynamical_hardness_measure, a=1.0)
dynamical_kt.__doc__ = """
Calculates dynamical kt (a = 1). Also known as ktDrop.

Args:
    delta_R: Splitting delta R.
    z: Splitting z.
    parent_pt: Pt of the parent of the splitting.
    R: Jet resolution parameter.
Returns:
    The hardness of the splitting according to the measure.
"""

dynamical_time = functools.partial(_dynamical_hardness_measure, a=2.0)
dynamical_time.__doc__ = """
Calculates dynamical time (a = 2). Also known as timeDrop.

Args:
    delta_R: Splitting delta R.
    z: Splitting z.
    parent_pt: Pt of the parent of the splitting.
    R: Jet resolution parameter.
Returns:
    The hardness of the splitting according to the measure.
"""


def find_leading(values: UprootArray[_T]) -> Tuple[np.ndarray, UprootArray[int]]:
    """Calculate hardest value given a set of values.

    Used for dynamical grooming, hardest kt, etc.

    In the case that we don't find a viable max (ie. because there was no splitting), we pad
    to one entry and fill -0.005 (our UNFILLED_VALUE) before flattening. The corresponding index
    will be empty for that event. This way, we can just fill all values, regardless of whether
    the splittings were selected, and we automatically get the right normalization (as long as
    those values are included in the hist...).

    Returns:
        Leading value, index of value.
    """
    # As of August 2020, keepdims doesn't seem to play nice with applying to the values, so we restore the dimensions with ak.singletons.
    # As of February 2021, we still can't replace this with keepdims...
    # argmax with singletons gives empty lists when there is no max, while keepdims uses None.
    # It looks like the max_values agree, however.
    arg_max = ak.singletons(ak.argmax(values, axis=1))
    max_values = ak.fill_none(ak.pad_none(values[arg_max], 1), UNFILLED_VALUE)

    # Try with keepdims:
    # new_arg_max = ak.argmax(values, axis=1, keepdims=True)
    # new_max_values = ak.fill_none(ak.pad_none(values[new_arg_max], 1), UNFILLED_VALUE)
    # Cross check
    # assert ak.all(ak.flatten(arg_max == new_arg_max, axis=-1))
    # assert ak.all(ak.flatten(max_values == new_max_values, axis=-1))

    # return ak.flatten(new_max_values), new_arg_max
    return ak.flatten(max_values), arg_max


_T_JetConstituent = TypeVar("_T_JetConstituent", bound="JetConstituentCommon")


class JetConstituentCommon:
    offset: int = 2000000
    pt: ArrayOrScalar[float]
    eta: ArrayOrScalar[float]
    phi: ArrayOrScalar[float]
    ID: ArrayOrScalar[int]

    def delta_R(self: _T_JetConstituent, other: _T_JetConstituent) -> ArrayOrScalar[float]:
        """Separation between jet constituents."""
        return cast(ArrayOrScalar[float], np.sqrt((self.phi - other.phi) ** 2 + (self.eta - other.eta) ** 2))  # type: ignore


class JetConstituent(ak.Record, JetConstituentCommon):  # type: ignore
    """A single jet constituent.

    Args:
        pt: Jet constituent pt.
        eta: Jet constituent eta.
        phi: Jet constituent phi.
        id: Jet constituent identifier. MC label (via GetLabel()) or global index (with offset defined above).
    """

    pt: float
    eta: float
    phi: float
    id: int

    def four_vector(self, mass_hypothesis: float = 0.139) -> None:
        return vector.obj(pt=self.pt, eta=self.eta, phi=self.phi, m=mass_hypothesis)


class JetConstituentArray(ak.Array, JetConstituentCommon):  # type: ignore
    """Methods for operating on jet constituents arrays.

    These methods operate on externally stored arrays. This is solely a mixin.

    Note:
        Unfortunately, it doesn't appear to be possible to use a TypedDict here to specify the types of the
        fields stored in the dict-like base object, so we just have to cast the properties.
    """

    pt: UprootArray[float]
    eta: UprootArray[float]
    phi: UprootArray[float]
    id: UprootArray[int]

    @property
    def max_pt(self) -> ArrayOrScalar[float]:
        """Maximum pt of the stored constituent."""
        return cast(ArrayOrScalar[float], ak.max(self.pt, axis=-1))

    def four_vectors(self, mass_hypothesis: float = 0.139) -> ak.Array:
        return vector.Array(pt=self.pt, eta=self.eta, phi=self.phi, m=self.pt * 0 + mass_hypothesis)


# Register behavior
ak.behavior["JetConstituent"] = JetConstituent
ak.behavior["*", "JetConstituent"] = JetConstituentArray


class SubjetCommon:
    """Common subjet related methods."""

    part_of_iterative_splitting: ArrayOrScalar[bool]
    parent_splitting_index: ArrayOrScalar[int]
    constituents_indices: UprootArray[int]

    @typing.overload
    def parent_splitting(self, splittings: UprootArray[JetSplittingArray]) -> JetSplittingArray:
        ...

    @typing.overload
    def parent_splitting(self, splittings: JetSplittingArray) -> JetSplitting:
        ...

    def parent_splitting(self, splittings):  # type: ignore
        """Retrieve the parent splitting of this subjet.

        Args:
            splittings: All of the splittings from the overall jet.
        Returns:
            Splitting which led to this subjet.
        """
        return splittings[self.parent_splitting_index]


class Subjet(ak.Record, SubjetCommon):  # type: ignore
    """Single subjet."""

    part_of_iterative_splitting: bool
    parent_splitting_index: int
    constituents_indices: UprootArray[int]


class SubjetArray(ak.Array, SubjetCommon):  # type: ignore
    """Array of subjets."""

    part_of_iterative_splitting: UprootArray[bool]
    parent_splitting_index: UprootArray[int]
    constituents_indices: UprootArray[int]

    @property
    def iterative_splitting_index(self) -> UprootArray[int]:
        """Indices of splittings which were part of the iterative splitting chain."""
        return self.parent_splitting_index[self.part_of_iterative_splitting]


# Register behavior
ak.behavior["Subjet"] = Subjet
ak.behavior["*", "Subjet"] = SubjetArray


class JetSplittingCommon:
    """Common jet splitting related methods."""

    kt: ArrayOrScalar[float]
    delta_R: ArrayOrScalar[float]
    z: ArrayOrScalar[float]
    parent_index: ArrayOrScalar[int]

    @property
    def parent_pt(self) -> ArrayOrScalar[float]:
        """Pt of the (parent) subjets which lead to the splittings.

        The pt can be calculated from the splitting properties via:

        parent_pt = subleading / z = kt / sin(delta_R) / z

        Args:
            None.
        Returns:
            None.
        """
        # parent_pt = subleading / z = kt / sin(delta_R) / z
        return cast(UprootArray[float], self.kt / np.sin(self.delta_R) / self.z)  # type: ignore

    def theta(self, jet_R: float) -> ArrayOrScalar[float]:
        """Theta of the splitting.

        This is defined as delta_R normalized by the jet resolution parameter.

        Args:
            jet_R: Jet resolution parameter.
        Returns:
            Theta of the splitting.
        """
        return self.delta_R / jet_R


class JetSplitting(ak.Record, JetSplittingCommon):  # type: ignore
    """Single jet splitting."""

    kt: float
    delta_R: float
    z: float
    parent_index: int

    @property
    def parent_pt(self) -> float:
        """Pt of the (parent) subjet which lead to the splitting.

        The pt can be calculated from the splitting properties via:

        parent_pt = subleading / z = kt / sin(delta_R) / z

        Args:
            None.
        Returns:
            None.
        """
        # parent_pt = subleading / z = kt / sin(delta_R) / z
        return cast(float, self.kt / np.sin(self.delta_R) / self.z)

    def dynamical_core(self, R: float) -> float:
        """Dynamical core of the splitting.

        See the definition for further information.

        Args:
            R: Jet resolution parameter.
        Returns:
            Dynamical core of the splitting.
        """
        return dynamical_core(self.delta_R, self.z, self.parent_pt, R)  # type: ignore

    def dynamical_z(self, R: float) -> float:
        """Dynamical z of the splitting.

        See the definition for further information.

        Args:
            R: Jet resolution parameter.
        Returns:
            Dynamical z of the splitting.
        """
        return dynamical_z(self.delta_R, self.z, self.parent_pt, R)  # type: ignore

    def dynamical_kt(self, R: float) -> float:
        """Dynamical kt of the splitting.

        See the definition for further information.

        Args:
            R: Jet resolution parameter.
        Returns:
            Dynamical kt of the splitting.
        """
        return dynamical_kt(self.delta_R, self.z, self.parent_pt, R)  # type: ignore

    def dynamical_time(self, R: float) -> float:
        """Dynamical time of the splitting.

        See the definition for further information.

        Args:
            R: Jet resolution parameter.
        Returns:
            Dynamical time of the splitting.
        """
        return dynamical_time(self.delta_R, self.z, self.parent_pt, R)  # type: ignore


class JetSplittingArray(ak.Array, JetSplittingCommon):  # type: ignore
    """Array of jet splittings."""

    kt: UprootArray[float]
    delta_R: UprootArray[float]
    z: UprootArray[float]
    parent_index: UprootArray[int]

    def iterative_splittings(self, subjets: SubjetArray) -> SubjetArray:
        """Retrieve the iterative splittings.

        Args:
            subjets: Subjets of the jets which containing the iterative splitting information.
        Returns:
            The splittings which are part of the iterative splitting chain.
        """
        return cast(SubjetArray, self[subjets.iterative_splitting_index])

    def dynamical_core(self, R: float) -> Tuple[np.ndarray, UprootArray[int], UprootArray[int]]:
        """Dynamical core of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical core values, leading dynamical core indices, indices of all splittings.
        """
        values, indices = find_leading(dynamical_core(self.delta_R, self.z, self.parent_pt, R))
        return values, indices, ak.local_index(self.z, axis=-1)

    def dynamical_z(self, R: float) -> Tuple[np.ndarray, UprootArray[int], UprootArray[int]]:
        """Dynamical z of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical z values, leading dynamical z indices, indices of all splittings.
        """
        values, indices = find_leading(dynamical_z(self.delta_R, self.z, self.parent_pt, R))
        return values, indices, ak.local_index(self.z, axis=-1)

    def dynamical_kt(self, R: float) -> Tuple[np.ndarray, UprootArray[int], UprootArray[int]]:
        """Dynamical kt of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical kt values, leading dynamical kt indices, indices of all splittings.
        """
        values, indices = find_leading(dynamical_kt(self.delta_R, self.z, self.parent_pt, R))
        return values, indices, ak.local_index(self.z, axis=-1)

    def dynamical_time(self, R: float) -> Tuple[np.ndarray, UprootArray[int], UprootArray[int]]:
        """Dynamical time of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical time values, leading dynamical time indices, indices of all splittings.
        """
        values, indices = find_leading(dynamical_time(self.delta_R, self.z, self.parent_pt, R))
        return values, indices, ak.local_index(self.z, axis=-1)

    def leading_kt(self, z_cutoff: Optional[float] = None) -> Tuple[np.ndarray, UprootArray[int], UprootArray[int]]:
        """Leading kt of the jet splittings.

        Args:
            z_cutoff: Z cutoff to be applied before calculating the leading kt.
        Returns:
            Leading kt values, leading kt indices, indices of all splittings which pass the cutoff.
        """
        # Need to use the local index because we are going to mask z values. If we index from the masked
        # z values, it it is applied to the unmasked array later, it will give nonsense. So we mask the local index,
        # find the leading, and then apply that index back to the local index, which then gives us the leading index
        # in the unmasked array.
        indices_passing_cutoff = ak.local_index(self.z, axis=-1)
        if z_cutoff is not None:
            indices_passing_cutoff = ak.local_index(self.z, axis=-1)[self.z > z_cutoff]
        values, indices = find_leading(self.kt[indices_passing_cutoff])
        return values, indices_passing_cutoff[indices], indices_passing_cutoff

    def soft_drop(self, z_cutoff: float) -> Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]:
        """Calculate soft drop of the splittings.

        Note:
            z_g is filled with the `UNFILLED_VALUE` if a splitting wasn't selected. In that case, there is
            no index (ie. an empty JaggedArray entry), and n_sd = 0.

        Note:
            n_sd can be calculated by using `count_nonzero()` on the indices which pass the cutoff.

        Args:
            z_cutoff: Minimum z for Soft Drop.
        Returns:
            First z passing cutoff (z_g), index of z passing cutoff, indices of all splittings which pass the cutoff.
        """
        z_cutoff_mask = self.z > z_cutoff
        indices_passing_cutoff = ak.local_index(self.z, axis=-1)[z_cutoff_mask]
        # We use :1 because this maintains the jagged structure. That way, we can apply it to initial arrays.
        z_index = indices_passing_cutoff[:, :1]
        z_g = ak.flatten(ak.fill_none(ak.pad_none(self.z[z_index], 1), UNFILLED_VALUE))

        return z_g, z_index, indices_passing_cutoff


# Register behavior
ak.behavior["JetSplitting"] = JetSplitting
ak.behavior["*", "JetSplitting"] = JetSplittingArray


def _is_valid_parquet_file(
    filename: Path,
) -> bool:
    """Check if the given parquet file is valid by opening the file.

    It appears that if there is an issue with the file, it will immediately be obvious
    because opening it with parquet will fail.

    Args:
        filename: Filename of the parquet file.
    Returns:
        True if the file was opened successfully.
    """
    try:
        _ = pq.ParquetFile(filename)
    except pa.ArrowInvalid:
        return False
    return True


def _convert_tree_to_parquet(
    tree: Any,
    prefixes: Sequence[str],
    branches: Sequence[str],
    prefix_branches: Sequence[str],
    output_filename: Path,
    entries: Tuple[Optional[int], Optional[int]],
    verbose: bool = False,
) -> bool:
    """Convert open tree to parquet.

    The template of the branch names to include are defined here.

    Args:
        tree: Uproot TTree.
        prefixes: Prefixes of the branches to be stored in the parquet file. The template branches
            are already specified - these are just to fill them in.
        branches: Branches to be read from the tree. These are branches which shouldn't be formatted
            with the prefix.
        prefix_branches: Branches to be read from the tree. They will be formatted with the passed prefixes.
        output_filename: Name under which the parquet file should be saved. Default: None, which
            takes the root filename and replaces the `.root` extension with `.parquet`.
        entries: Selection of entries to be read. Default: None, which indicates no selection.
        verbose: If True, print verbose info, including the tree branches.
    Returns:
        True if the tree was successfully converted.
    """
    # Validation
    # Only create the parquet file if we haven't already made the conversion.
    if output_filename.exists():
        return True
    # Setup
    if not output_filename.parent.exists():
        output_filename.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        tree.show()

    # Determine branches.
    all_branches: List[str] = []
    all_branches.extend(branches)
    for prefix in prefixes:
        all_branches.extend([b.format(prefix=prefix) for b in prefix_branches])
    logger.debug(all_branches)

    # Extract arrays
    additional_kwargs = {}
    if entries:
        additional_kwargs.update({"entry_start": entries[0], "entry_stop": entries[1]})
    arrays = tree.arrays(all_branches, **additional_kwargs)

    # Write out to parquet.
    ak.to_parquet(arrays, output_filename, compression="zstd")

    # I don't think this will really help, but it's worth a try, since memory is
    # leaking somewhere (perhaps in awkward1 itself).
    del arrays

    return True


def convert_tree_to_parquet(
    filename: Path,
    tree_name: str,
    prefixes: Sequence[str],
    branches: Sequence[str],
    prefix_branches: Sequence[str],
    output_filename: Optional[Path] = None,
    entries: Optional[Tuple[Optional[int], Optional[int]]] = None,
) -> Tuple[bool, Path]:
    """Convert a ROOT tree to a parquet file using awkward.

    The main benefit is that it can open _much_ faster via parquet compared to uproot because it doesn't
    appear to need to use a python loop overly the doubly jagged arrays. This saves _huge_ amounts of time.

    Args:
        filename: Filename containing the tree.
        tree_name: Name of the tree stored in the file.
        prefixes: Prefixes of the branches to be stored in the parquet file. The template branches
            are already specified - these are just to fill them in.
        branches: Branches to be read from the tree. These are branches which shouldn't be formatted
            with the prefix.
        prefix_branches: Branches to be read from the tree. They will be formatted with the passed prefixes.
        output_filename: Name under which the parquet file should be saved. Default: None, which
            takes the root filename and replaces the `.root` extension with `.parquet`.
        entries: Selection of entries to be read. Default: None, which indicates no selection.
    Returns:
        True if the tree was successfully converted.
    """
    # Validation
    if output_filename is None:
        output_filename = filename.with_suffix(".parquet")
    if entries is None:
        entries = (None, None)

    # Bail out early if the file already exists.
    if output_filename.exists():
        if _is_valid_parquet_file(filename=output_filename):
            return True, output_filename
        # Remove the parquet file so we can try again.
        logger.info("Parquet is invalid. Try to convert again.")
        output_filename.unlink()

    with uproot.open(filename) as f:
        tree = f[tree_name]
        result = _convert_tree_to_parquet(
            tree=tree,
            prefixes=prefixes,
            branches=branches,
            prefix_branches=prefix_branches,
            output_filename=output_filename,
            entries=entries,
        )
    return result, output_filename


def parquet_to_substructure_analysis(filename: Path, prefixes: Mapping[str, str]) -> Dict[str, ak.Array]:
    """Convert an existing parquet file to arrays for substructure analysis.

    Note:
        We have implicitly built in the map of branches that we want to access into the
        ak.Array structure that we return.

    Args:
        filename: Filename of the parquet file.
        prefixes: Prefixes of the branches to be loaded from the parquet file. The template branches
            are already specified - these are just to fill them in. Each prefix will create a substructure array.
            We map from the desired prefixes to those which are used in storing the data.
    Returns:
        One substructure array per prefix, along with a few individual columns if available in the input data
            (related to pt hard info or unsubtracted leading track pt).
    """
    # Read all of the arrays from parquet.
    # NOTE: In principle, we could read fewer branches here. However, it doesn't seem to be necessary
    #       as of August 2020.
    arrays = ak.from_parquet(filename)

    # As of August 2020, there was an issue with loading data from parquet: All arrow data is nullable, so
    # the data types that we stored in the parquet file are not quite identical to those that we put in:
    # all types loaded from the file are now nullable (denoted by the "?" in awkward1). As of August 2020,
    # awkward1 seems to treat nullable data differently in many cases (I suspect bugs, but I'm not sure).
    # However, since we're not storing nulled data - at most, they have empty arrays, which don't count
    # as nullable - we can work around this issue by filling None with a throwaway value, which removes
    # the nullability.
    #
    # We use some very different value to make it clear if something ever goes wrong.
    # NOTE: It's important to do this before constructing our substructure array. Otherwise it will
    #       mess up the awkward1 behaviors.
    #
    # Update January 2021: awkward now preserves nullability when writing. Which is to say, if data is
    #       not null before writing, then it won't be after reading. To take advantage of this, we would
    #       have to redo the conversion from root to parquet, but that won't happen immediately. Therefore,
    #       we check for nullable arrays (denoted by "?" in the awkward type), and if we find it, we
    #       apply our hack. If not, then we can skip it. Eventually, once everything is reconverted,
    #       then this can be removed.
    # NOTE: This check probably isn't the most robust. I imagine we can actually iterate through the
    #       array and check for nullability on each type. But this check is quick and easy, so we'll
    #       stick with it.
    if "?" in str(arrays.type):
        fill_none_value = -9999
        arrays = ak.fill_none(arrays, fill_none_value)

    columns = {}
    # For most analysis task outputs
    if "ptHard" in ak.fields(arrays):
        columns.update(
            {
                "pt_hard": arrays["ptHard"],
                "pt_hard_bin": arrays["ptHardBin"],
            }
        )
    # For recent (2022) analysis task outputs
    if "pt_hard" in ak.fields(arrays):
        columns.update(
            {
                "pt_hard": arrays["pt_hard"],
                "pt_hard_bin": arrays["pt_hard_bin"],
            }
        )
    additional_columns = {}
    # Add unsubtracted leading track pt for data if it was stored.
    # We'll need to handle this later in the skim.
    # NOTE: This always has prefix "data" if it's included!
    if "data_leading_track_pt" in ak.fields(arrays):
        prefix_for_leading_track = "hybrid" if "hybrid" in list(prefixes.keys()) else "data"
        additional_columns[prefix_for_leading_track] = {
            "leading_track_pt": arrays["data_leading_track_pt"],
        }

    # We use a ton of gymnastics here because I'm not confident about memory ownership here and I want to
    # avoid copies as much as possible!
    # NOTE: Here, we translate from whatever the prefix was stored under (where we just blindly copied it),
    #       to standardized prefixes.
    return {
        **columns,
        **{
            output_prefix: ak.zip(
                {
                    "jet_pt": arrays[f"{input_prefix}.fJetPt"],
                    "jet_constituents": ak.zip(
                        {
                            "pt": arrays[f"{input_prefix}.fJetConstituents.fPt"],
                            "eta": arrays[f"{input_prefix}.fJetConstituents.fEta"],
                            "phi": arrays[f"{input_prefix}.fJetConstituents.fPhi"],
                            "id": arrays[f"{input_prefix}.fJetConstituents.fID"]
                            if f"{input_prefix}.fJetConstituents.fID" in ak.fields(arrays)
                            else arrays[f"{input_prefix}.fJetConstituents.fGlobalIndex"],
                        },
                        with_name="JetConstituent",
                        # We want to apply the behavior for each jet, and then for each constituent
                        # in the jet, so we use a depth limit of 2.
                        depth_limit=2,
                    ),
                    "jet_splittings": ak.zip(
                        {
                            "kt": arrays[f"{input_prefix}.fJetSplittings.fKt"],
                            "delta_R": arrays[f"{input_prefix}.fJetSplittings.fDeltaR"],
                            "z": arrays[f"{input_prefix}.fJetSplittings.fZ"],
                            "parent_index": arrays[f"{input_prefix}.fJetSplittings.fParentIndex"],
                        },
                        with_name="JetSplitting",
                        # We want to apply the behavior for each jet, and then for each splitting
                        # in the jet, so we use a depth limit of 2.
                        depth_limit=2,
                    ),
                    "subjets": ak.zip(
                        {
                            "part_of_iterative_splitting": arrays[f"{input_prefix}.fSubjets.fPartOfIterativeSplitting"],
                            "parent_splitting_index": arrays[f"{input_prefix}.fSubjets.fSplittingNodeIndex"],
                            "constituent_indices": arrays[f"{input_prefix}.fSubjets.fConstituentIndices"],
                        },
                        with_name="Subjet",
                        # We want to apply the behavior for each jet, and then for each subjet
                        # in the jet, so we use a depth limit of 2.
                        depth_limit=2,
                    ),
                    **additional_columns.get(output_prefix, {}),
                },
                # The structure of the jet pt and the other values is inherently different, so
                # we only put them together on the jet level with a depth limit of 1.
                depth_limit=1,
            )
            for output_prefix, input_prefix in prefixes.items()
        },
    }


if __name__ == "__main__":

    # import time

    # start = time.perf_counter()
    # convert_tree_to_parquet(
    #    filename=Path("trains/embedPythia/5966/AnalysisResults.18q.repaired.root"),
    #    tree_name="AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
    #    prefixes=["data", "matched", "detLevel"],
    # )
    # (arrays,) = parquet_to_substructure_analysis(
    #    filename=Path("trains/embedPythia/5966/AnalysisResults.18q.repaired.parquet"), prefixes=["data"]
    # )
    # finish = time.perf_counter()
    # print(f"Uproot4: Length: {ak.num(arrays, axis=0)}, time: {finish-start}")
    ## Sanity check if the fill_none is a problem
    # assert not ak.any(arrays == -9999)
    # import IPython

    # IPython.embed()

    # An example for testing.
    from jet_substructure.base import helpers

    helpers.setup_logging(level=logging.DEBUG)
    convert_tree_to_parquet(
        filename=Path("trains/embedPythia/6632/AnalysisResults.18r.repaired.root"),
        tree_name="AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
        prefixes=list({"hybrid": "data", "true": "matched", "det_level": "detLevel"}.values()),
        branches=[],
        prefix_branches=[
            "{prefix}.fJetPt",
            "{prefix}.fJetConstituents.fPt",
            "{prefix}.fJetConstituents.fEta",
            "{prefix}.fJetConstituents.fPhi",
            "{prefix}.fJetConstituents.fID",
            "{prefix}.fJetSplittings.fKt",
            "{prefix}.fJetSplittings.fDeltaR",
            "{prefix}.fJetSplittings.fZ",
            "{prefix}.fJetSplittings.fParentIndex",
            "{prefix}.fSubjets.fPartOfIterativeSplitting",
            "{prefix}.fSubjets.fSplittingNodeIndex",
            "{prefix}.fSubjets.fConstituentIndices",
        ],
        entries=(0, None),
        output_filename=Path(
            "trains/embedPythia/6632/parquet/events_per_job_100000/bak/AnalysisResults.18r.repaired.00.parquet"
        ),
    )

    # import uproot as uproot3
    # f = uproot3.open("trains/embedPythia/5966/AnalysisResults.18q.root")
    # tree = f["AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl"]
    # import time
    # start = time.perf_counter()
    # arrays = jet_substructure(tree, prefix="data")
    # finish = time.perf_counter()
    # print(f"Uproot3: Length: {len(arrays[b'data.fJetPt'])}, time: {finish-start}")
