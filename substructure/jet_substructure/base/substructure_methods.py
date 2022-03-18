#!/usr/bin/env python3

""" Tests for Jet Substructure interpretation for uproot.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import functools
import logging
import typing
from typing import TYPE_CHECKING, Callable, Optional, Tuple, Type, TypeVar, cast

import attr
import awkward0 as ak
import numpy as np
import uproot3_methods as uproot_methods
from typing_extensions import Final

from jet_substructure.base.helpers import ArrayOrScalar, UprootArray


logger = logging.getLogger(__name__)

# Typing helpers
T = TypeVar("T")


# Constants
UNFILLED_VALUE: Final[float] = -0.005


@typing.overload
def _dynamical_hardness_measure(
    delta_R: UprootArray[float], z: UprootArray[float], parent_pt: UprootArray[float], R: float, a: float
) -> UprootArray[float]:
    ...


@typing.overload
def _dynamical_hardness_measure(delta_R: float, z: float, parent_pt: float, R: float, a: float) -> float:
    ...


def _dynamical_hardness_measure(delta_R, z, parent_pt, R, a):  # type: ignore
    return z * (1 - z) * parent_pt * (delta_R / R) ** a


dynamical_z = functools.partial(_dynamical_hardness_measure, a=0.1)
dynamical_kt = functools.partial(_dynamical_hardness_measure, a=1.0)
dynamical_time = functools.partial(_dynamical_hardness_measure, a=2.0)


def find_leading(values: UprootArray[T]) -> Tuple[np.ndarray, UprootArray[int]]:
    """Calculate hardest value given a set of values.

    Used for dynamical grooming, hardest kt, etc.

    In the case that we don't find a viable max (ie. because there was no splitting), we pad
    to one entry and fill -0.01 (our UNFILLED_VALUE) before flattening. The corresponding index
    will be empty for that event. This way, we can just fill all values, regardless of whether
    the splittings were selected, and we automatically get the right normalization (as long as
    those values are included in the hist...).

    Returns:
        Leading value, index of value.
    """
    arg_max = values.argmax()
    return values[arg_max].pad(1).fillna(UNFILLED_VALUE).flatten(), arg_max


class ArrayMethods(ak.Methods):  # type: ignore
    """ Base class containing methods for use in awkward `ObjectArray`s. """

    # Seems to be required for creating JaggedArray elements within an Array.
    # Otherwise, it will create one object per event, with that object storing arrays of members.
    awkward = ak

    def _try_memo(self, name: str, function: Callable[..., T]) -> Callable[..., T]:
        """Try to memorize the result of a function so it doesn't need to be recalculated.

        It unwraps a layer of jaggedness, so some care is required for using it.

        Note:
            This is taken from `uproot-methods`.
        """
        memoname = "_memo_" + name
        wrap, (array,) = ak.util.unwrap_jagged(type(self), self.JaggedArray, (self,))
        if not hasattr(array, memoname):
            setattr(array, memoname, function(array))
        return cast(Callable[..., T], wrap(getattr(array, memoname)))


@attr.s
class JetConstituent:
    """A single jet constituent.

    Args:
        pt: Jet constituent pt.
        eta: Jet constituent eta.
        phi: Jet constituent phi.
        global_index: Global index assigned to the track during analysis. The index is unique
            for each event.
    """

    pt: float = attr.ib()
    eta: float = attr.ib()
    phi: float = attr.ib()
    _global_index: int = attr.ib()

    @property
    def index(self) -> int:
        return self._global_index

    def delta_R(self, other: "JetConstituent") -> float:
        return cast(float, np.sqrt((self.phi - other.phi) ** 2 + (self.eta - other.eta) ** 2))

    def four_vector(self, mass_hypothesis: float = 0.139) -> uproot_methods.TLorentzVector:
        return uproot_methods.TLorentzVector(
            self.pt,
            self.eta,
            self.phi,
            mass_hypothesis,
        )


class JetConstituentArrayMethods(ArrayMethods):
    """Methods for operating on jet constituents arrays.

    These methods operate on externally stored arrays. This is solely a mixin.

    Note:
        Unfortunately, it doesn't appear to be possible to use a TypedDict here to specify the types of the
        fields stored in the dict-like base object, so we just have to cast the properties.
    """

    def _init_object_array(self, table: ak.Table) -> None:
        """Create jet constituent views in a table.

        Args:
            table: Table where the constituents will be created.
        """
        self.awkward.ObjectArray.__init__(
            self, table, lambda row: JetConstituent(row["pt"], row["eta"], row["phi"], row["global_index"])
        )

    @property
    def pt(self) -> UprootArray[float]:
        """ Constituent pts. """
        return cast(UprootArray[float], self["pt"])

    @property
    def eta(self) -> UprootArray[float]:
        """ Constituent etas. """
        return cast(UprootArray[float], self["eta"])

    @property
    def phi(self) -> UprootArray[float]:
        """ Constituent phis. """
        return cast(UprootArray[float], self["phi"])

    @property
    def index(self) -> UprootArray[int]:
        """ Constituent global indices. """
        return cast(UprootArray[int], self["global_index"])

    @property
    def max_pt(self) -> ArrayOrScalar[float]:
        """ Maximum pt of the stored constituent. """
        return cast(ArrayOrScalar[float], self["pt"].max())

    def delta_R(self, other: "JetConstituentArray") -> UprootArray[float]:
        """ Delta R between one set of constituents and the others. """
        return cast(UprootArray[float], np.sqrt((self["phi"] - other["phi"]) ** 2 + (self["eta"] - other["eta"]) ** 2))

    def four_vectors(self, mass_hypothesis: float = 0.139) -> uproot_methods.TLorentzVectorArray:
        if isinstance(self.pt, np.ndarray):
            ones_like = np.ones_like(self.pt)
        else:
            ones_like = self.pt.ones_like()
        return uproot_methods.TLorentzVectorArray.from_ptetaphim(
            self.pt,
            self.eta,
            self.phi,
            ones_like * mass_hypothesis,
        )


# Adds in JaggedArray methods for constructing objects with jagged structure.
JaggedJetConstituentArrayMethods = JetConstituentArrayMethods.mixin(JetConstituentArrayMethods, ak.JaggedArray)


class JetConstituentArray(JetConstituentArrayMethods, ak.ObjectArray):  # type:ignore
    """Array of jet constituents.

    This effectively constructs a virtual object that can operate transparently on arrays.

    Args:
        pt: Array of constituent pt.
        eta: Array of constituent eta.
        phi: Array of constituent phi.
        global_index: Array of constituent global indices.
    """

    def __init__(
        self,
        pt: UprootArray[float],
        eta: UprootArray[float],
        phi: UprootArray[float],
        global_index: UprootArray[int],
    ) -> None:
        self._init_object_array(ak.Table())
        self["pt"] = pt
        self["eta"] = eta
        self["phi"] = phi
        self["global_index"] = global_index

    def __awkward_serialize__(self, serializer: ak.persist.Serializer) -> ak.persist.Serializer:
        """ Serialize to storage. """
        self._valid()
        pt, eta, phi, global_index = self.pt, self.eta, self.phi, self.global_index
        return serializer.encode_call(
            ["jet_substructure.base.substructure_methods", "JetConstituentArray", "from_jagged"],
            serializer(pt, "JetConstituentArray.pt"),
            serializer(eta, "JetConstituentArray.eta"),
            serializer(phi, "JetConstituentArray.phi"),
            serializer(global_index, "JetConstituentArray.global_index"),
        )

    @classmethod
    @ak.util.wrapjaggedmethod(JaggedJetConstituentArrayMethods)  # type: ignore
    def from_jagged(
        cls: Type[T],
        pt: UprootArray[UprootArray[float]],
        eta: UprootArray[UprootArray[float]],
        phi: UprootArray[UprootArray[float]],
        global_index: UprootArray[UprootArray[int]],
    ) -> T:
        """Creates a view of constituents with jagged structure.

        On it's own, this object can operate on arrays. However, by constructing with this method, it goes a step
        further in, constructing virtual objects in the jagged structure. Basically, it unwarps the first layer of
        jaggedness using a wrapper.

        Args:
            pt: Jagged constituent pt.
            eta: Jagged constituent eta.
            phi: Jagged constituent phi.
            global_index: Jagged constituent global index.

        Returns:
            Jet constituents array acting within the jagged array contents.
        """
        return cls(pt, eta, phi, global_index)  # type: ignore


@attr.s
class Subjet:
    """Subjet found within a jet.

    Args:
        part_of_iterative_splitting: True if the subjet is part of the iterative splitting.
        parent_splitting_index: Index of the splitting which lead to this subjet.
        constituents: Constituents which are contained within this subjet.
    """

    part_of_iterative_splitting: bool = attr.ib()
    _parent_splitting_index: int = attr.ib()
    _constituents: JetConstituentArray = attr.ib()

    @classmethod
    def from_constituents_indices(
        cls: Type["Subjet"],
        part_of_iterative_splitting: bool,
        parent_splitting_index: int,
        constituent_indices: UprootArray[int],
        jet_constituents: JetConstituentArray,
    ) -> "Subjet":
        """Construct the subjet from the constituent indices and jet constituents.

        Note:
            This is helpful for the case where the subjet constituents aren't directly available. This can be rather
            common because we don't save the entire constituents for each subjet, as this would be incredibly wasteful
            in terms of storage and retrieving from the grid. However, we can generate and store them later in analysis
            to save processing time during analysis when storage space is less critical.

        Args:
            part_of_iterative_splitting: True if the subjet is part of the iterative splitting.
            parent_splitting_index: Index of the splitting which lead to this subjet.
            constituents_indices: Indices of the constituents that are contained within this subjet.
                This is indexed by the number of constituents in a jet (i.e. it is not the global index!).
            jet_constituents: Jet constituents which are indexed by their number in the jet.
        """
        return cls(
            part_of_iterative_splitting,
            parent_splitting_index,
            jet_constituents[constituent_indices],
        )

    @property
    def parent_splitting_index(self) -> int:
        """ Index of the parent splitting which produced the subjet. """
        return self._parent_splitting_index

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
        return splittings[self._parent_splitting_index]

    @property
    def constituents(self) -> JetConstituentArray:
        """ Constituents of this subjet. """
        return self._constituents


def _convert_jagged_constituents_indices(
    constituents_indices: ak.JaggedArray, jagged_indices: ak.JaggedArray
) -> ak.JaggedArray:
    """Convert constituents indices and jagged indices into a doubly jagged constituents array.

    That array can then be used for determining which constituents belong to which subjets.

    Args:
        constituents_indices: Jagged array containing all of the constituents indices of the subjets.
        jagged_indices: Jagged array containing the offsets in the constituents_indices for each individual subjet.

    Returns:
        Doubly jagged indices specifying the constituents in each subjet.
    """
    return ak.fromiter(
        (ak.JaggedArray.fromoffsets(jagged, indices) for jagged, indices in zip(jagged_indices, constituents_indices))
    )


class SubjetArrayMethods(ArrayMethods):
    """Methods for operating on subjet arrays.

    These methods operate on externally stored arrays. This is solely a mixin.

    Note:
        Unfortunately, it doesn't appear to be possible to use a TypedDict here to specify the types of the
        fields stored in the dict-like base object, so we just have to cast the properties.
    """

    def _init_object_array(self, table: ak.Table) -> None:
        """Create jet constituent views in a table.

        Args:
            table: Table where the subjets will be created.
        Returns:
            None.
        """
        self.awkward.ObjectArray.__init__(
            self,
            table,
            lambda row: Subjet(row["part_of_iterative_splitting"], row["parent_splitting_index"], row["constituents"]),
        )

    def __awkward_serialize__(self, serializer: ak.persist.Serializer) -> ak.persist.Serializer:
        """ Serialize to storage. """
        self._valid()
        part_of_iterative_splitting, parent_splitting_index, constituents = (
            self.part_of_iterative_splitting,
            self.parent_splitting_index,
            self.constituents,
        )
        return serializer.encode_call(
            ["jet_substructure.base.substructure_methods", "SubjetArray", "_from_jagged_impl"],
            serializer(part_of_iterative_splitting, "SubjetArray.part_of_iterative_splitting"),
            serializer(parent_splitting_index, "SubjetArray.parent_splitting_index"),
            serializer(constituents, "SubjetArray.constituents"),
        )

    @property
    def part_of_iterative_splitting(self) -> UprootArray[bool]:
        """Whether subjets are part of the iterative splitting.

        Args:
            None.
        Returns:
            True if the subjets are part of the iterative splitting.
        """
        return cast(UprootArray[bool], self["part_of_iterative_splitting"])

    @property
    def parent_splitting_index(self) -> UprootArray[int]:
        """ Indices of the parent splittings. """
        return cast(UprootArray[int], self["parent_splitting_index"])

    @property
    def iterative_splitting_index(self) -> UprootArray[int]:
        """ Indices of splittings which were part of the iterative splitting chain. """
        return self.parent_splitting_index[self.part_of_iterative_splitting]

    def parent_splitting(self, splittings: UprootArray[JetSplittingArray]) -> UprootArray[JetSplittingArray]:
        """Retrieve the parent splittings of the subjets.

        Args:
            splittings: Splittings which may have produced the subjets.
        Returns:
            Parent splittings for the subjets.
        """
        return cast(UprootArray[JetSplittingArray], splittings[self["parent_splitting_index"]])


# Adds in JaggedArray methods for constructing objects with jagged structure.
JaggedSubjetArrayMethods = SubjetArrayMethods.mixin(SubjetArrayMethods, ak.JaggedArray)

_T_SubjetArray = TypeVar("_T_SubjetArray", bound="SubjetArray")


class SubjetArray(SubjetArrayMethods, ak.ObjectArray):  # type: ignore
    """Array of subjets.

    This effectively constructs a virtual object that can operate transparently on arrays.

    Args:
        part_of_iterative_splitting: True if the given subjet is part of the iterative splitting.
        parent_splitting_index: Indices of the parent splitting of the subjets.
        constituents: Constituents which are contained within the subjets.
    """

    def __init__(
        self,
        part_of_iterative_splitting: UprootArray[bool],
        parent_splitting_index: UprootArray[int],
        constituents: UprootArray[JetConstituentArray],
    ) -> None:
        self._init_object_array(ak.Table())
        self["part_of_iterative_splitting"] = part_of_iterative_splitting
        self["parent_splitting_index"] = parent_splitting_index
        self["constituents"] = constituents

    @classmethod
    def from_jagged(
        cls: Type[_T_SubjetArray],
        part_of_iterative_splitting: UprootArray[bool],
        parent_splitting_index: UprootArray[int],
        constituents_indices: UprootArray[int],
        subjet_constituents: Optional[UprootArray[JetConstituentArray]] = None,
        jet_constituents: Optional[UprootArray[JetConstituentArray]] = None,
        constituents_jagged_indices: Optional[UprootArray[int]] = None,
    ) -> _T_SubjetArray:
        """Creates a view of subjets with jagged structure.

        On it's own, this object can operate on arrays. However, by constructing with this method, it gives a step
        further in, constructing virtual objects in the jagged structure.

        Note:
            We need this additional wrapper because we can't construct doubly jagged arrays directly, and the jaggedness
            needs to be the same for all parts for of the array. We work around this by constructing the doubly jagged
            array before constructing the `ObjectArray`. See: https://github.com/scikit-hep/uproot/issues/452.

        Note:
            The caller may pass either the `subjet_constituents` or the `jet_constituents`. If the `subjet_constituents`
            are passed, they take precedence over everything else.

        Args:
            part_of_iterative_splitting: Jagged iterative splitting label.
            parent_splitting_index: Jagged parent splitting index.
            constituents_indices: Jagged constituents indices of the subjet.
            subjet_constituents: Subjet constituents. These are eventually constructed by this object, but
                can be saved afterwards to avoid having to reconstruct them again. If they're passed in,
                the rest of the constituents arguments are ignored and these constituents are used. Default: None.
            jet_constituents: Jet constituents. Used in conjunction withe the constituents_indices to determine
                which constituents belong in which subjets. Default: None.
            constituents_jagged_indices: Constituents jagged indices used to convert the constituents_indices into
                doubly jagged indices if they are not already. Default: None.
        Returns:
            Subjet array acting within the jagged array contents.
        """
        # Validation
        if subjet_constituents is None and constituents_indices is None:
            raise ValueError("Must pass subjet constituents or constituents indices.")
        logger.debug("Determining subjet constituents.")

        # We have three modes for creating the indices:
        # 1) The subjet constituents have already been determined in the past. Just use them.
        # 2) Construct the constituent indices using separately stored jagged indices.
        # 3) Construct the doubly jagged indices stored in the tree via fromiter(...)
        # In the case of 2 or 3, the subjets constituents are determine from the jet constituents.
        if subjet_constituents is not None:
            logger.debug("Using pre-calculated constituents.")
            pass
        else:
            if constituents_jagged_indices is not None:
                # Calculate the indices.
                logger.debug("Constructing constituents indices from manually stored jagged indices.")
                constituents_indices = _convert_jagged_constituents_indices(
                    constituents_indices, constituents_jagged_indices
                )
            else:
                # Construct the indices.
                logger.debug("Constructing constituents indices from doubly jagged indices.")
                constituents_indices = ak.fromiter(constituents_indices)

            # Help out mypy
            assert jet_constituents is not None

            # This doesn't seem super efficient, but I can't seem to broadcast it directly.
            logger.debug("Calculating subjets constituents from constituents indices.")
            subjet_constituents = ak.JaggedArray.fromoffsets(
                constituents_indices.offsets,
                ak.JaggedArray.fromoffsets(
                    constituents_indices.flatten().offsets,
                    jet_constituents[constituents_indices.flatten(axis=1)].flatten(),
                ),
            )
            # And convert back to JetConstituentArray. Without this, it will just appear as a JaggedArray.
            # Can't wait until I can use awkward1!
            # NOTE: May not be the most performant, but fine for now.
            # NOTE: All of the type ignores are because I don't wnat to deal with passing through the
            #       attributes through the UprrotArray type.
            subjet_constituents = JetConstituentArray.from_jagged(
                subjet_constituents.pt,  # type: ignore
                subjet_constituents.eta,  # type: ignore
                subjet_constituents.phi,  # type: ignore
                subjet_constituents.global_index,  # type: ignore
            )

        return cast(
            _T_SubjetArray,
            cls._from_jagged_impl(part_of_iterative_splitting, parent_splitting_index, subjet_constituents),
        )

    @classmethod
    @ak.util.wrapjaggedmethod(JaggedSubjetArrayMethods)  # type: ignore
    def _from_jagged_impl(
        cls: Type[_T_SubjetArray],
        part_of_iterative_splitting: UprootArray[bool],
        parent_splitting_index: UprootArray[int],
        constituents: UprootArray[JetConstituentArray],
    ) -> _T_SubjetArray:
        """Creates a view of subjets with jagged structure.

        On it's own, this object can operate on arrays. However, by constructing with this method, it gives a step
        further in, constructing virtual objects in the jagged structure.

        This is a bit of a pain to call directly, so it's hidden behind the wrapper defined above.

        Args:
            part_of_iterative_splitting: Jagged iterative splitting label.
            parent_splitting_index: Jagged parent splitting index.
            constituents: Constituents of the subjet.
        Returns:
            Subjet array acting within the jagged array contents.
        """
        return cls(part_of_iterative_splitting, parent_splitting_index, constituents)


@attr.s
class JetSplitting:
    """Properties of a jet splitting.

    Args:
        kt: Kt of the subjets.
        delta_R: Delta R between the subjets.
        z: Z of the softer subjet.
        parent_index: Index of the parent subjet.
    """

    kt: float = attr.ib()
    delta_R: float = attr.ib()
    z: float = attr.ib()
    _parent_index: int = attr.ib()

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

    def theta(self, jet_R: float) -> float:
        """Theta of the splitting.

        This is defined as delta_R normalized by the jet resolution parameter.

        Args:
            jet_R: Jet resolution parameter.
        Returns:
            Theta of the splitting.
        """
        return self.delta_R / jet_R

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


class JetSplittingArrayMethods(ArrayMethods):
    """Methods for operating on jet splittings arrays.

    These methods operate on externally stored arrays. This is solely a mixin.

    Note:
        Unfortunately, it doesn't appear to be possible to use a TypedDict here to specify the types of the
        fields stored in the dict-like base object, so we just have to cast the properties.
    """

    def _init_object_array(self, table: ak.Table) -> None:
        """Create jet splitting views in a table.

        Args:
            table: Table where the jet splittings will be created.
        Returns:
            None.
        """
        self.awkward.ObjectArray.__init__(
            self, table, lambda row: JetSplitting(row["kt"], row["delta_R"], row["z"], row["parent_index"])
        )

    def __awkward_serialize__(self, serializer: ak.persist.Serializer) -> ak.persist.Serializer:
        """ Serialize to storage. """
        self._valid()
        kt, delta_R, z, parent_index = self.kt, self.delta_R, self.z, self.parent_index
        return serializer.encode_call(
            ["jet_substructure.base.substructure_methods", "JetSplittingArray", "from_jagged"],
            serializer(kt, "JetSplittingArray.kt"),
            serializer(delta_R, "JetSplittingArray.delta_R"),
            serializer(z, "JetSplittingArray.z"),
            serializer(parent_index, "JetSplittingArray.parent_index"),
        )

    @property
    def kt(self) -> UprootArray[float]:
        """ Kt of the splittings. """
        return cast(UprootArray[float], self["kt"])

    @property
    def delta_R(self) -> UprootArray[float]:
        """ Delta R of the splittings. """
        return cast(UprootArray[float], self["delta_R"])

    @property
    def z(self) -> UprootArray[float]:
        """ z of the splitting. """
        return cast(UprootArray[float], self["z"])

    def iterative_splittings(self, subjets: SubjetArray) -> SubjetArray:
        """Retriieve the iterative splittings.

        Args:
            subjets: Subjets of the jets which containing the iterative splitting information.
        Returns:
            The splittings which are part of the iterative splitting chain.
        """
        return cast(SubjetArray, self[subjets.iterative_splitting_index])

    @property
    def parent_pt(self) -> UprootArray[float]:
        """Pt of the (parent) subjets which lead to the splittings.

        The pt can be calculated from the splitting properties via:

        parent_pt = subleading / z = kt / sin(delta_R) / z

        Args:
            None.
        Returns:
            None.
        """
        # parent_pt = subleading / z = kt / sin(delta_R) / z
        return cast(UprootArray[float], self.kt / np.sin(self.delta_R) / self.z)

    def theta(self, jet_R: float) -> UprootArray[float]:
        """Theta of the splittings.

        This is defined as delta_R normalized by the jet resolution parameter.

        Args:
            jet_R: Jet resolution parameter.
        Returns:
            Theta of the splittings.
        """
        return self.delta_R / jet_R

    def dynamical_z(self, R: float) -> Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]:
        """Dynamical z of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical z values, leading dynamical z indices, indices of all splittings.
        """
        values, indices = find_leading(dynamical_z(self.delta_R, self.z, self.parent_pt, R))
        return values, indices, self.z.localindex

    def dynamical_kt(self, R: float) -> Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]:
        """Dynamical kt of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical kt values, leading dynamical kt indices, indices of all splittings.
        """
        values, indices = find_leading(dynamical_kt(self.delta_R, self.z, self.parent_pt, R))
        return values, indices, self.z.localindex

    def dynamical_time(self, R: float) -> Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]:
        """Dynamical time of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical time values, leading dynamical time indices, indices of all splittings.
        """
        values, indices = find_leading(dynamical_time(self.delta_R, self.z, self.parent_pt, R))
        return values, indices, self.z.localindex

    def leading_kt(
        self, z_cutoff: Optional[float] = None
    ) -> Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]:
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
        indices_passing_cutoff = self.z.localindex
        if z_cutoff is not None:
            indices_passing_cutoff = self.z.localindex[self.z > z_cutoff]
        values, indices = find_leading(self.kt[indices_passing_cutoff])
        return values, indices_passing_cutoff[indices], indices_passing_cutoff

    def soft_drop(self, z_cutoff: float) -> Tuple[UprootArray[float], UprootArray[int], UprootArray[int]]:
        """Calculate soft drop of the splittings.

        Note:
            z_g is filled with the `UNFILLED_VALUE` if a splitting wasn't selected. In that case, there is
            no index (ie. an emptry JaggedArray entry), and n_sd = 0.

        Note:
            n_sd can be calculated by using `count_nonzero()` on the indices which pass the cutoff.

        Args:
            z_cutoff: Minimum z for Soft Drop.
        Returns:
            First z passing cutoff (z_g), index of z passing cutoff, indices of all splittings which pass the cutoff.
        """
        z_cutoff_mask = self.z > z_cutoff
        indices_passing_cutoff = self.z.localindex[z_cutoff_mask]
        # We use :1 because this maintains the jagged structure. That way, we can apply it to initial arrays.
        z_index = indices_passing_cutoff[:, :1]
        z_g = self.z[z_index].pad(1).fillna(UNFILLED_VALUE).flatten()

        return z_g, z_index, indices_passing_cutoff


# Adds in JaggedArray methods for constructing objects with jagged structure.
JaggedJetSplittingArrayMethods = JetSplittingArrayMethods.mixin(JetSplittingArrayMethods, ak.JaggedArray)


class JetSplittingArray(JetSplittingArrayMethods, ak.ObjectArray):  # type: ignore
    """Array of jet splittings.

    This effectively constructs a virtual object that can operate transparently on arrays.

    Args:
        kt: Kt of the jet splittings.
        delta_R: Delta R of the subjets.
        z: Momentum fraction of the softer subjet.
        parent_index: Index of the parent splitting.
    """

    def __init__(
        self,
        kt: UprootArray[float],
        delta_R: UprootArray[float],
        z: UprootArray[float],
        parent_index: UprootArray[int],
    ) -> None:
        self._init_object_array(ak.Table())
        self["kt"] = kt
        self["delta_R"] = delta_R
        self["z"] = z
        self["parent_index"] = parent_index

    @classmethod
    @ak.util.wrapjaggedmethod(JaggedJetSplittingArrayMethods)  # type: ignore
    def from_jagged(
        cls: Type[T],
        kt: UprootArray[float],
        delta_R: UprootArray[float],
        z: UprootArray[float],
        parent_index: UprootArray[int],
    ) -> T:
        """Creates a view of splittings with jagged structure.

        On it's own, this object can operate on arrays. However, by constructing with this method, it gives a step
        further in, constructing virtual objects in the jagged structure.

        Args:
            kt: Jagged kt.
            delta_R: Jagged delta R.
            z: Jagged z.
            parent_index: Jagged parent splitting index.
        Returns:
            Splittings array acting on the jagged array contents.
        """
        return cls(kt, delta_R, z, parent_index)  # type: ignore


class SubstructureJetCommonMethods:
    """Common methods for jet substructure methods.

    Note:
        These only work if properties have the same names in both the single and array classes.
    """

    if TYPE_CHECKING:
        _constituents: JetConstituentArray
        _subjets: SubjetArray
        _splittings: JetSplittingArray

    @property
    def constituents(self) -> JetConstituentArray:
        return self._constituents

    @property
    def subjets(self) -> SubjetArray:
        return self._subjets

    @property
    def splittings(self) -> JetSplittingArray:
        return self._splittings

    @property
    def leading_track_pt(self) -> ArrayOrScalar[float]:
        """ Leading track pt. """
        return self.constituents.max_pt

    def dynamical_z(self, R: float) -> Tuple[ArrayOrScalar[float], ArrayOrScalar[int], ArrayOrScalar[int]]:
        """Dynamical z of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical z value, leading dynamical z index.
        """
        return self.splittings.dynamical_z(R=R)

    def dynamical_kt(self, R: float) -> Tuple[ArrayOrScalar[float], ArrayOrScalar[int], ArrayOrScalar[int]]:
        """Dynamical kt of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical kt value, leading dynamical kt index.
        """
        return self.splittings.dynamical_kt(R=R)

    def dynamical_time(self, R: float) -> Tuple[ArrayOrScalar[float], ArrayOrScalar[int], ArrayOrScalar[int]]:
        """Dynamical time of the jet splittings.

        Args:
            R: Jet resolution parameter.
        Returns:
            Leading dynamical time value, leading dynamical time index.
        """
        return self.splittings.dynamical_time(R=R)

    def leading_kt(
        self, z_cutoff: Optional[float] = None
    ) -> Tuple[ArrayOrScalar[float], ArrayOrScalar[int], ArrayOrScalar[int]]:
        """Leading kt of the jet splittings.

        Args:
            z_cutoff: Z cutoff to be applied before calculating the leading kt.
        Returns:
            Leading kt values, leading kt indices.
        """
        return self.splittings.leading_kt(z_cutoff=z_cutoff)

    def soft_drop(self, z_cutoff: float) -> Tuple[ArrayOrScalar[float], ArrayOrScalar[int], ArrayOrScalar[int]]:
        """Calculate soft drop of the splittings.

        Args:
            z_cutoff: Minimum z for Soft Drop.
        Returns:
            First z passing cutoff (z_g), number of splittings passing SD (n_sd), index of z passing cutoff.
        """
        return self.splittings.soft_drop(z_cutoff=z_cutoff)


@attr.s
class SubstructureJet(SubstructureJetCommonMethods):
    """Substructure of a jet.

    Args:
        jet_pt: Jet pt.
        constituents: Jet constituents.
        subjets: Subjets.
        splittings: Jet splittings.
    """

    jet_pt: float = attr.ib()
    _constituents: JetConstituentArray = attr.ib()
    _subjets: SubjetArray = attr.ib()
    _splittings: JetSplittingArray = attr.ib()


class SubstructureJetArrayMethods(SubstructureJetCommonMethods, ArrayMethods):
    """Methods for operating on substructure jet arrays.

    These methods operate on externally stored arrays. This is solely a mixin.

    Note:
        Unfortunately, it doesn't appear to be possible to use a TypedDict here to specify the types of the
        fields stored in the dict-like base object, so we just have to cast the properties.
    """

    def _init_object_array(self, table: ak.Table) -> None:
        """Create substructure jet views in a table.

        Args:
            table: Table where the substructure jets will be created.
        Returns:
            None.
        """
        self.awkward.ObjectArray.__init__(
            self,
            table,
            lambda row: SubstructureJet(row["jet_pt"], row["constituents"], row["subjets"], row["splittings"]),
        )

    def __awkward_serialize__(self, serializer: ak.persist.Serializer) -> ak.persist.Serializer:
        """ Serialize to storage. """
        self._valid()
        jet_pt, constituents, subjets, splittings = self.jet_pt, self.constituents, self.subjets, self.splittings
        # NOTE: It doesn't appear that the second argument to the serializer is super meaningful...
        return serializer.encode_call(
            ["jet_substructure.base.substructure_methods", "SubstructureJetArray", "_from_serialization"],
            serializer(jet_pt, "SubstructureJetArray.jet_pt"),
            serializer(constituents, "SubstructureJetArray.constituents"),
            serializer(subjets, "SubstructureJetArray.subjets"),
            serializer(splittings, "SubstructureJetArray.splittings"),
        )

    @property
    def jet_pt(self) -> UprootArray[float]:
        """ Jet pt. """
        return cast(UprootArray[float], self["jet_pt"])

    @property
    def constituents(self) -> JetConstituentArray:
        """ Jet constituents. """
        return cast(JetConstituentArray, self["constituents"])

    @property
    def subjets(self) -> SubjetArray:
        """ Subjets. """
        return cast(SubjetArray, self["subjets"])

    @property
    def splittings(self) -> JetSplittingArray:
        """ Jet splittings. """
        return cast(JetSplittingArray, self["splittings"])


# Adds in JaggedArray methods for constructing objects with jagged structure.
JaggedSubstructureJetArrayMethods = SubstructureJetArrayMethods.mixin(SubstructureJetArrayMethods, ak.JaggedArray)


class SubstructureJetArray(SubstructureJetArrayMethods, ak.ObjectArray):  # type: ignore
    """Array of substructure jets.

    Note:
        This can't support a `from_jagged(...)` method because the contained arrays have different
        jaggedness. The overlay expanding into the jagged dimension only works if they have the same
        jaggedness. However, we can still create one object per element in the array (ie. for each jet).
    """

    def __init__(
        self,
        jet_pt: UprootArray[float],
        jet_constituents: UprootArray[JetConstituentArray],
        subjets: UprootArray[SubjetArray],
        jet_splittings: UprootArray[JetSplittingArray],
    ) -> None:
        self._init_object_array(ak.Table())
        self["jet_pt"] = jet_pt
        self["constituents"] = jet_constituents
        self["subjets"] = subjets
        self["splittings"] = jet_splittings

    @classmethod
    def from_tree(cls: Type[T], tree: ak.JaggedArray, prefix: str) -> T:
        """Construct from a tree.

        Args:
            tree: Tree containing the splittings.
            prefix: Prefix under which the branches are stored.
        Returns:
            Substructure jet array wrapping all of the arrays.
        """
        logger.debug("Creating substructure jet arrays.")
        constituent_index = tree.get(f"{prefix}.fJetConstituents.fID", None)
        if constituent_index is None:
            constituent_index = tree[f"{prefix}.fJetConstituents.fGlobalIndex"]
        constituents = JetConstituentArray.from_jagged(
            tree[f"{prefix}.fJetConstituents.fPt"],
            tree[f"{prefix}.fJetConstituents.fEta"],
            tree[f"{prefix}.fJetConstituents.fPhi"],
            constituent_index,
        )
        logger.debug("Done with constructing constituents")
        splittings = JetSplittingArray.from_jagged(
            tree[f"{prefix}.fJetSplittings.fKt"],
            tree[f"{prefix}.fJetSplittings.fDeltaR"],
            tree[f"{prefix}.fJetSplittings.fZ"],
            tree[f"{prefix}.fJetSplittings.fParentIndex"],
        )
        logger.debug("Done with constructing splittings")
        subjets = SubjetArray.from_jagged(
            tree[f"{prefix}.fSubjets.fPartOfIterativeSplitting"],
            tree[f"{prefix}.fSubjets.fSplittingNodeIndex"],
            tree[f"{prefix}.fSubjets.fConstituentIndices"],
            tree.get(f"{prefix}.fSubjets.constituents", None),
            constituents,
            tree.get(f"{prefix}.fSubjets.fConstituentJaggedIndices", None),
        )
        logger.debug("Done with constructing subjets.")
        logger.debug(f"Done with constructing jet inputs for {tree.filename}, {prefix}")

        # Construct substructure jets using the above
        return cls(  # type: ignore
            tree[f"{prefix}.fJetPt"],
            constituents,
            subjets,
            splittings,
        )

    @classmethod
    def _from_serialization(
        cls: Type[T],
        jet_pt: UprootArray[float],
        jet_constituents: JetConstituentArray,
        subjets: SubjetArray,
        jet_splittings: JetSplittingArray,
    ) -> T:
        """Serialization doesn't seem to receate these jagged members properly.

        Namely, it creates the object in the jagged array, but it doesn't wrap them up in the
        external objects properly. So we manually recreate the objects here. It may be that I'm
        doing something wrong in the serialization, but this seems like it will work too, and
        it's easy.

        Unfortunately, it loses a good deal of our speed up...
        """
        return cls(  # type: ignore
            jet_pt=jet_pt,
            jet_constituents=JetConstituentArray.from_jagged(
                jet_constituents.pt,
                jet_constituents.eta,
                jet_constituents.phi,
                jet_constituents.global_index,
            ),
            subjets=SubjetArray._from_jagged_impl(
                subjets.part_of_iterative_splitting,
                subjets.parent_splitting_index,
                # Ensure that the constituents are also constructed correctly.
                JetConstituentArray.from_jagged(
                    subjets.constituents.pt,
                    subjets.constituents.eta,
                    subjets.constituents.phi,
                    subjets.constituents.global_index,
                ),
            ),
            jet_splittings=JetSplittingArray.from_jagged(
                jet_splittings.kt,
                jet_splittings.delta_R,
                jet_splittings.z,
                jet_splittings.parent_index,
            ),
        )
