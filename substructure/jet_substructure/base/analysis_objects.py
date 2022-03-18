""" Main analysis objects.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import attr
import boost_histogram as bh
import numpy as np
from pachyderm import binned_data

from jet_substructure.base import helpers
from jet_substructure.base.helpers import UprootArray


if TYPE_CHECKING:
    from jet_substructure.base import substructure_methods

logger = logging.getLogger(__name__)

_T_Identifier = TypeVar("_T_Identifier", bound="Identifier")
_T_MatchingHybridIdentifier = TypeVar("_T_MatchingHybridIdentifier", bound="MatchingHybridIdentifier")


@attr.s(frozen=True)
class Identifier:
    iterative_splittings: bool = attr.ib()
    jet_pt_bin: helpers.RangeSelector = attr.ib()

    @property
    def iterative_splittings_label(self) -> str:
        return "iterative" if self.iterative_splittings else "recursive"

    def __str__(self) -> str:
        return f"jetPt_{self.jet_pt_bin.min}_{self.jet_pt_bin.max}_{self.iterative_splittings_label}_splittings"

    def display_str(self, jet_pt_label: str = "") -> str:
        return f"{self.iterative_splittings_label.capitalize()} splittings\n${self.jet_pt_bin.display_str(label=jet_pt_label)}$"

    @classmethod
    def from_existing(cls: Type[_T_Identifier], existing: _T_Identifier) -> _T_Identifier:
        return cls(
            iterative_splittings=existing.iterative_splittings,
            jet_pt_bin=existing.jet_pt_bin,
        )


@attr.s(frozen=True)
class MatchingHybridIdentifier(Identifier):
    """Identify hybrid cuts on the matching hists.

    Note:
        We only use this class for plotting! Not for identify the hists during processing!
    """

    min_kt: float = attr.ib(default=0)

    def __str__(self) -> str:
        base_str = super().__str__()
        if self.min_kt > 0:
            base_str = f"{base_str}_hybridMinKt_{self.min_kt}"
        return base_str

    def display_str(self, jet_pt_label: str = "") -> str:
        if jet_pt_label:
            logger.warning(f"We're ignoring the jet pt label {jet_pt_label}. Using 'hybrid'.")
        base_str = super().display_str(jet_pt_label="hybrid")
        if self.min_kt > 0:
            base_str += "\n" + fr"$k_{{\text{{T}}}}^{{\text{{hybrid}}}} > {self.min_kt}$"
        return base_str

    @classmethod
    def from_existing(
        cls: Type[_T_MatchingHybridIdentifier], existing: _T_MatchingHybridIdentifier
    ) -> _T_MatchingHybridIdentifier:
        return cls(
            iterative_splittings=existing.iterative_splittings, jet_pt_bin=existing.jet_pt_bin, min_kt=existing.min_kt
        )

    @classmethod
    def from_existing_identifier(
        cls: Type["MatchingHybridIdentifier"],
        existing: Identifier,
        hybrid_jet_pt_bin: helpers.RangeSelector,
        min_kt: float,
    ) -> "MatchingHybridIdentifier":
        return cls(
            iterative_splittings=existing.iterative_splittings,
            jet_pt_bin=hybrid_jet_pt_bin,
            min_kt=min_kt,
        )


@attr.s(frozen=True)
class AnalysisSettings:
    jet_R: float = attr.ib()
    z_cutoff: float = attr.ib()

    @classmethod
    def _extract_values_from_dataset_config(cls: Type["AnalysisSettings"], config: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "jet_R": config["jet_R"],
        }

    @classmethod
    def from_config(cls: Type["AnalysisSettings"], config: Mapping[str, Any], z_cutoff: float) -> "AnalysisSettings":
        return cls(
            z_cutoff=z_cutoff,
            **cls._extract_values_from_dataset_config(config),
        )


@attr.s(frozen=True)
class PtHardAnalysisSettings(AnalysisSettings):
    scale_factors: Mapping[int, float] = attr.ib()
    train_number_to_pt_hard_bin: Mapping[int, int] = attr.ib()

    def asdict(self) -> Dict[str, Any]:
        return attr.asdict(self, recurse=False)

    @classmethod
    def _extract_values_from_dataset_config(
        cls: Type["PtHardAnalysisSettings"], config: Mapping[str, Any]
    ) -> Dict[str, Any]:
        # Extract the base class values first, then add our additional values.
        values = super(PtHardAnalysisSettings, cls)._extract_values_from_dataset_config(config)
        values.update(
            {
                "scale_factors": config["scale_factors"],
                "train_number_to_pt_hard_bin": config["train_number_to_pt_hard_bin"],
            }
        )
        return values


@attr.s(frozen=True)
class Dataset:
    collision_system: str = attr.ib()
    name: str = attr.ib()
    filenames: Sequence[str] = attr.ib()
    tree_name: str = attr.ib()
    branches: Sequence[str] = attr.ib()
    settings: AnalysisSettings = attr.ib()
    _hists_filename: str = attr.ib()
    _output_base: Path = attr.ib()

    @property
    def output(self) -> Path:
        return self._output_base / self.collision_system / self.name

    @property
    def hists_filename(self) -> Path:
        return self.output / self._hists_filename

    def setup(self) -> bool:
        self.output.mkdir(parents=True, exist_ok=True)
        return True

    @classmethod
    def from_config_file(
        cls: Type["Dataset"],
        collision_system: str,
        config_filename: Path,
        hists_filename_stem: str,
        output_base: Path,
        settings_class: Type[AnalysisSettings],
        z_cutoff: float,
        override_filenames: Optional[Sequence[Union[str, Path]]] = None,
        # "pgz" = pickled gz file.
        hists_file_extension: str = "pgz",
    ) -> "Dataset":
        # Grab the configuration
        from pachyderm import yaml

        y = yaml.yaml()
        with open(config_filename, "r") as f:
            config = y.load(f)

        # Extract only the values from the config that we need to construct the object.
        _dataset_config = config["datasets"][collision_system]["dataset"]
        name = _dataset_config["name"]
        selected_dataset_config = config["available_datasets"][name]
        filenames = selected_dataset_config["files"] if override_filenames is None else override_filenames

        obj = cls(
            collision_system=collision_system,
            name=name,
            filenames=filenames,
            tree_name=selected_dataset_config["tree_name"],
            branches=_dataset_config["branches"],
            settings=settings_class.from_config(config=selected_dataset_config, z_cutoff=z_cutoff),
            hists_filename=f"{hists_filename_stem}.{hists_file_extension}",
            output_base=output_base,
        )
        # Complete setup
        obj.setup()

        return obj


@attr.s
class MatchingResult:
    properly: UprootArray[bool] = attr.ib()
    mistag: UprootArray[bool] = attr.ib()
    failed: UprootArray[bool] = attr.ib()

    def __getitem__(self, mask: UprootArray[bool]) -> "MatchingResult":
        return type(self)(
            properly=self.properly[mask],
            mistag=self.mistag[mask],
            failed=self.failed[mask],
        )


@attr.s
class FillHistogramInput:
    jets: "substructure_methods.SubstructureJetArray" = attr.ib()
    _splittings: "substructure_methods.JetSplittingArray" = attr.ib()
    values: UprootArray[float] = attr.ib()
    indices: UprootArray[int] = attr.ib()

    @property
    def splittings(self) -> "substructure_methods.JetSplittingArray":
        try:
            return self._restricted_splittings
        except AttributeError:
            self._restricted_splittings: "substructure_methods.JetSplittingArray" = self._splittings[self.indices]
        return self._restricted_splittings

    @property
    def n_jets(self) -> int:
        """Number of jets.

        Need to determine all jets which are accepted in the jet pt range.
        Otherwise, those which may fail (such as with a z_cutoff) may not get
        the proper normalization.
        """
        return len(self.jets)

    def __getitem__(self, mask: np.ndarray) -> FillHistogramInput:
        """ Mask the stored values, returning a new object. """
        # Validation
        if len(self.jets) != len(mask):
            raise ValueError(
                f"Mask length is different than array lengths. mask length: {len(mask)}, array lengths: {len(self.jets)}"
            )

        # Return the masked arrays in a new object.
        return type(self)(
            jets=self.jets[mask],
            # NOTE: It's super important to use the internal splittings. Otherwise, we'll try to apply the indices twice
            #       (which won't work for the masked object).
            splittings=self._splittings[mask],
            values=self.values[mask],
            indices=self.indices[mask],
        )


def _calculate_splitting_number(indices: UprootArray[int]) -> UprootArray[int]:
    # +1 because splittings counts from 1, but indexing starts from 0.
    splitting_number = indices + 1
    # If there were no splittings, we want to set that to 0.
    splitting_number = splitting_number.pad(1).fillna(0)
    # Must flatten because the indices are still jagged.
    splitting_number = splitting_number.flatten()
    return splitting_number


@attr.s
class SubstructureHistsBase:
    name: str = attr.ib()
    title: str = attr.ib()
    iterative_splittings: bool = attr.ib()

    def __add__(self, other: "SubstructureHistsBase") -> "SubstructureHistsBase":
        """ Handles a = b + c """
        new = copy.deepcopy(self)
        new += other
        return new

    def __radd__(self, other: SubstructureHistsBase) -> SubstructureHistsBase:
        """ For use with sum(...). """
        if other == 0:
            return self
        else:
            return self + other

    def __iadd__(self, other: SubstructureHistsBase) -> SubstructureHistsBase:
        raise NotImplementedError("Daughter classes must implement.")

    @property
    def attributes_to_skip(self) -> List[str]:
        return ["name", "title", "iterative_splittings"]

    def __iter__(self) -> Iterator[Tuple[str, Union[bh.Histogram, binned_data.BinnedData]]]:
        return iter(
            {k: v for k, v in attr.asdict(self, recurse=False).items() if k not in self.attributes_to_skip}.items()
        )

    def convert_boost_histograms_to_binned_data(self) -> None:
        # Check if we can return immediately.
        if all(isinstance(hist, binned_data.BinnedData) for _, hist in self):
            return

        # Sanity check.
        # Ensure that we're not somehow half converted. If so, something rather odd has occurred.
        if not all(isinstance(hist, bh.Histogram) for _, hist in self):
            types = {k: type(v) for k, v in self}
            raise ValueError(f"Not all hists are boost histograms! Cannot convert to binned data! Types: {types}")

        for k, v in self:
            setattr(self, k, binned_data.BinnedData.from_existing_data(v))


@attr.s
class SubstructureHists(SubstructureHistsBase):
    name: str = attr.ib()
    title: str = attr.ib()
    iterative_splittings: bool = attr.ib()
    n_jets: float = attr.ib()
    jet_pt: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    values: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    kt: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    z: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    delta_R: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    theta: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    splitting_number: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    splitting_number_perturbative: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    lund_plane: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    total_number_of_splittings: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()

    @property
    def attributes_to_skip(self) -> List[str]:
        attrs = super().attributes_to_skip
        attrs.extend(["n_jets"])
        return attrs

    def __iadd__(self, other: SubstructureHistsBase) -> "SubstructureHists":
        """ Handles a += b """
        # Validation
        if not isinstance(other, type(self)):
            raise TypeError(f"Must pass type {type(self)}. Passed: {type(other)}.")
        if self.iterative_splittings != other.iterative_splittings:
            raise TypeError(
                f"The types of splittings are different! self: {self.iterative_splittings}, other: {other.iterative_splittings}"
            )

        self.name = f"{self.name}_{other.name}" if self.name != other.name else self.name
        self.title = f"{self.title}_{other.title}" if self.title != other.title else self.title
        # Don't need to update iterative_splittings since they must be the same!
        self.n_jets += other.n_jets
        for (k, v), (k_other, v_other) in zip(self, other):
            v += v_other

        return self

    def __truediv__(self, other: "SubstructureHists") -> "SubstructureHists":
        data = []
        for (k, v), (k_other, v_other) in zip(self, other):
            # Sanity check
            if k != k_other:
                raise ValueError(f"Somehow keys mismatch. self key: {k}, other key: {k_other}")
            # First, normalize the hists by the number of jets.
            temp_v = v / self.n_jets
            temp_v_other = v_other / other.n_jets
            data.append(temp_v / temp_v_other)

        return type(self)(
            f"{self.name}_{other.name}",
            f"{self.title}_{other.title}",
            self.iterative_splittings and other.iterative_splittings,
            1,
            *data,
        )

    @classmethod
    def create_boost_histograms(
        cls: Type["SubstructureHists"], name: str, title: str, iterative_splittings: bool, values_axis: bh.Histogram
    ) -> "SubstructureHists":
        kt_axis = bh.axis.Regular(50, 0, 25)
        z_axis = bh.axis.Regular(20, 0, 0.5)
        delta_R_axis = bh.axis.Regular(20, 0, 0.4)
        theta_axis = bh.axis.Regular(20, 0, 1)
        splitting_number_axis = bh.axis.Regular(10, 0, 10)
        total_number_of_splittings_axis = bh.axis.Regular(50, 0, 50)
        lund_plane_axes = [bh.axis.Regular(100, 0, 5), bh.axis.Regular(100, -5.0, 5.0)]
        return cls(
            name=name,
            title=title,
            iterative_splittings=iterative_splittings,
            n_jets=0,
            jet_pt=bh.Histogram(bh.axis.Regular(150, 0, 150), storage=bh.storage.Weight()),
            values=bh.Histogram(values_axis, storage=bh.storage.Weight()),
            kt=bh.Histogram(kt_axis, storage=bh.storage.Weight()),
            z=bh.Histogram(z_axis, storage=bh.storage.Weight()),
            delta_R=bh.Histogram(delta_R_axis, storage=bh.storage.Weight()),
            theta=bh.Histogram(theta_axis, storage=bh.storage.Weight()),
            splitting_number=bh.Histogram(splitting_number_axis, storage=bh.storage.Weight()),
            splitting_number_perturbative=bh.Histogram(splitting_number_axis, storage=bh.storage.Weight()),
            total_number_of_splittings=bh.Histogram(total_number_of_splittings_axis, storage=bh.storage.Weight()),
            lund_plane=bh.Histogram(*lund_plane_axes, storage=bh.storage.Weight()),
        )

    def fill(
        self,
        inputs: FillHistogramInput,
        jet_R: float,
        splitting_number: Optional[UprootArray[int]] = None,
        weight: float = 1.0,
    ) -> None:
        # Validation
        # Give a useful error message
        if not all(isinstance(hist, bh.Histogram) for _, hist in self):
            raise ValueError("Not all hists are boost histograms! Cannot fill!")

        # And then help out mypy...
        assert (
            isinstance(self.values, bh.Histogram)
            and isinstance(self.jet_pt, bh.Histogram)
            and isinstance(self.kt, bh.Histogram)
            and isinstance(self.z, bh.Histogram)
            and isinstance(self.delta_R, bh.Histogram)
            and isinstance(self.theta, bh.Histogram)
            and isinstance(self.splitting_number, bh.Histogram)
            and isinstance(self.splitting_number_perturbative, bh.Histogram)
            and isinstance(self.total_number_of_splittings, bh.Histogram)
            and isinstance(self.lund_plane, bh.Histogram)
        )
        # Need to store the number of jets along the histograms.
        self.n_jets += inputs.n_jets * weight
        self.jet_pt.fill(inputs.jets.jet_pt, weight=weight)
        self.values.fill(inputs.values, weight=weight)
        self.kt.fill(inputs.splittings.kt.flatten(), weight=weight)
        self.z.fill(inputs.splittings.z.flatten(), weight=weight)
        self.delta_R.fill(inputs.splittings.delta_R.flatten(), weight=weight)
        self.theta.fill(inputs.splittings.theta(jet_R).flatten(), weight=weight)
        if splitting_number is None:
            splitting_number = _calculate_splitting_number(inputs.indices)
        self.splitting_number.fill(splitting_number, weight=weight)
        # Select only splittings with kt > 5.
        # +1 because splittings counts from 1, but indexing starts from 0.
        # NOTE: We aren't counting 0 here if it fails, so we aren't preserving counts!
        #       In this simpler case, we can just select directly on the indices.
        splitting_number_perturbative = (inputs.indices + 1)[inputs.splittings.kt > 5].flatten()
        self.splitting_number_perturbative.fill(splitting_number_perturbative, weight=weight)
        self.total_number_of_splittings.fill(inputs.splittings.counts, weight=weight)
        self.lund_plane.fill(
            np.log(1.0 / inputs.splittings.delta_R.flatten()), np.log(inputs.splittings.kt.flatten()), weight=weight
        )

        # Check the second peak in the z_cutoff recursive Lund Plane.
        if (np.log(1.0 / inputs.splittings.delta_R.flatten()) < 2).any() and (
            np.log(inputs.splittings.kt.flatten()) < -1.5
        ).any():
            # import IPython; IPython.embed()
            pass


@attr.s
class SubstructureToyHists(SubstructureHistsBase):
    name: str = attr.ib()
    title: str = attr.ib()
    iterative_splittings: bool = attr.ib()
    n_jets: float = attr.ib()
    values: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    kt: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    z: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    delta_R: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    theta: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()

    @property
    def attributes_to_skip(self) -> List[str]:
        attrs = super().attributes_to_skip
        attrs.extend(["n_jets"])
        return attrs

    def __iadd__(self, other: SubstructureHistsBase) -> "SubstructureToyHists":
        """ Handles a += b """
        # Validation
        if not isinstance(other, type(self)):
            raise TypeError(f"Must pass type {type(self)}. Passed: {type(other)}.")
        if self.iterative_splittings != other.iterative_splittings:
            raise TypeError(
                f"The types of splittings are different! self: {self.iterative_splittings}, other: {other.iterative_splittings}"
            )

        self.name = f"{self.name}_{other.name}" if self.name != other.name else self.name
        self.title = f"{self.title}_{other.title}" if self.title != other.title else self.title
        # Don't need to update iterative_splittings since they must be the same!
        self.n_jets += other.n_jets
        for (k, v), (k_other, v_other) in zip(self, other):
            v += v_other

        return self

    def __truediv__(self, other: "SubstructureToyHists") -> "SubstructureToyHists":
        data = []
        for (k, v), (k_other, v_other) in zip(self, other):
            # Sanity check
            if k != k_other:
                raise ValueError(f"Somehow keys mismatch. self key: {k}, other key: {k_other}")
            # First, normalize the hists by the number of jets.
            temp_v = v / self.n_jets
            temp_v_other = v_other / other.n_jets
            data.append(temp_v / temp_v_other)

        return type(self)(
            f"{self.name}_{other.name}",
            f"{self.title}_{other.title}",
            self.iterative_splittings and other.iterative_splittings,
            1,
            *data,
        )

    @classmethod
    def create_boost_histograms(
        cls: Type["SubstructureToyHists"], name: str, title: str, iterative_splittings: bool, values_axis: bh.Histogram
    ) -> "SubstructureToyHists":
        z_axis = bh.axis.Regular(20, 0, 0.5)
        delta_R_axis = bh.axis.Regular(20, 0, 0.4)
        theta_axis = bh.axis.Regular(20, 0, 1)
        return cls(
            name=name,
            title=title,
            iterative_splittings=iterative_splittings,
            n_jets=0,
            values=bh.Histogram(values_axis, values_axis, storage=bh.storage.Weight()),
            kt=bh.Histogram(bh.axis.Regular(100, -5, 5), bh.axis.Regular(100, -5, 5), storage=bh.storage.Weight()),
            z=bh.Histogram(z_axis, z_axis, storage=bh.storage.Weight()),
            delta_R=bh.Histogram(delta_R_axis, delta_R_axis, storage=bh.storage.Weight()),
            theta=bh.Histogram(theta_axis, theta_axis, storage=bh.storage.Weight()),
        )

    def fill(
        self,
        data_inputs: FillHistogramInput,
        true_inputs: FillHistogramInput,
        jet_R: float,
        weight: float,
    ) -> None:
        # Validation
        # Give a useful error message
        if not all(isinstance(hist, bh.Histogram) for _, hist in self):
            raise ValueError("Not all hists are boost histograms! Cannot fill!")

        # And then help out mypy...
        assert (
            isinstance(self.values, bh.Histogram)
            and isinstance(self.kt, bh.Histogram)
            and isinstance(self.z, bh.Histogram)
            and isinstance(self.delta_R, bh.Histogram)
            and isinstance(self.theta, bh.Histogram)
        )
        # Need to store the number of jets along the histograms.
        self.n_jets += data_inputs.n_jets * weight
        self.values.fill(true_inputs.values, data_inputs.values, weight=weight)
        self.kt.fill(
            np.log(true_inputs.splittings.kt.flatten()), np.log(data_inputs.splittings.kt.flatten()), weight=weight
        )
        self.z.fill(true_inputs.splittings.z.flatten(), data_inputs.splittings.z.flatten(), weight=weight)
        self.delta_R.fill(
            true_inputs.splittings.delta_R.flatten(), data_inputs.splittings.delta_R.flatten(), weight=weight
        )
        self.theta.fill(
            true_inputs.splittings.theta(jet_R).flatten(), data_inputs.splittings.theta(jet_R).flatten(), weight=weight
        )
        # self.kt.fill(data_inputs.splittings.kt.pad(1).fillna(0).flatten(), true_inputs.splittings.kt.pad(1).fillna(0).flatten())
        # self.z.fill(data_inputs.splittings.z.pad(1).fillna(0).flatten(), true_inputs.splittings.z.pad(1).fillna(0).flatten())
        # self.delta_R.fill(data_inputs.splittings.delta_R.pad(1).fillna(0).flatten(), true_inputs.splittings.delta_R.pad(1).fillna(0).flatten())
        # self.theta.fill(data_inputs.splittings.theta(jet_R).pad(1).fillna(0).flatten(), true_inputs.splittings.theta(jet_R).pad(1).fillna(0).flatten(),)


@attr.s
class MatchingSelections:
    leading: MatchingResult = attr.ib()
    subleading: MatchingResult = attr.ib()

    def __getitem__(self, name: str) -> np.ndarray:
        """ Helper to retrieve the masks. """
        return getattr(self, name)

    @property
    def all(self) -> np.ndarray:
        return (
            self.leading.properly
            | self.leading.mistag
            | self.leading.failed
            | self.subleading.properly
            | self.subleading.mistag
            | self.subleading.failed
        )

    @property
    def pure(self) -> np.ndarray:
        return self.leading.properly & self.subleading.properly

    @property
    def leading_untagged_subleading_correct(self) -> np.ndarray:
        return self.leading.failed & self.subleading.properly

    @property
    def leading_correct_subleading_untagged(self) -> np.ndarray:
        return self.leading.properly & self.subleading.failed

    @property
    def leading_untagged_subleading_mistag(self) -> np.ndarray:
        return self.leading.failed & self.subleading.mistag

    @property
    def leading_mistag_subleading_untagged(self) -> np.ndarray:
        return self.leading.mistag & self.subleading.failed

    @property
    def swap(self) -> np.ndarray:
        return self.leading.mistag & self.subleading.mistag

    @property
    def both_untagged(self) -> np.ndarray:
        return self.leading.failed & self.subleading.failed


_matching_name_to_axis_value: Dict[str, int] = {
    "all": 0,
    "pure": 1,
    "leading_untagged_subleading_correct": 2,
    "leading_correct_subleading_untagged": 3,
    "leading_untagged_subleading_mistag": 4,
    "leading_mistag_subleading_untagged": 5,
    "swap": 6,
    "both_untagged": 7,
}


@attr.s
class SubstructureResponseHists(SubstructureHistsBase):
    """

    Note:
        By convention, the first axis should be the closer to measured level, while the second axis should be closer
        to the generator level.

    """

    axis_map: Mapping[str, str] = attr.ib()
    use_matching_axis: bool = attr.ib()
    measured_like_n_jets: float = attr.ib()
    generator_like_n_jets: float = attr.ib()
    residuals_jet_pt: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    residuals_kt: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    response_kt: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    # NOTE: Intentionally not an attr.ib() property. We want it to be a ClassVar.
    matching_name_to_axis_value: ClassVar[Dict[str, int]] = _matching_name_to_axis_value

    @property
    def attributes_to_skip(self) -> List[str]:
        attrs = super().attributes_to_skip
        attrs.extend(["axis_map", "use_matching_axis", "measured_like_n_jets", "generator_like_n_jets"])
        return attrs

    def __iadd__(self, other: SubstructureHistsBase) -> "SubstructureResponseHists":
        """ Handles a += b """
        # Validation
        if not isinstance(other, type(self)):
            raise TypeError(f"Must pass type {type(self)}. Passed: {type(other)}.")
        if self.iterative_splittings != other.iterative_splittings:
            raise TypeError(
                f"The types of splittings are different! self: {self.iterative_splittings}, other: {other.iterative_splittings}"
            )

        self.name = f"{self.name}_{other.name}" if self.name != other.name else self.name
        self.title = f"{self.title}_{other.title}" if self.title != other.title else self.title
        # Don't need to update iterative_splittings since they must be the same!
        self.measured_like_n_jets += other.measured_like_n_jets
        self.generator_like_n_jets += other.generator_like_n_jets
        for (k, v), (k_other, v_other) in zip(self, other):
            v += v_other

        return self

    @classmethod
    def create_boost_histograms(
        cls: Type["SubstructureResponseHists"],
        name: str,
        title: str,
        iterative_splittings: bool,
        axis_map: Mapping[str, str],
        use_matching_axis: bool,
        measured_like_jet_pt_axis: bh.axis.Regular,
        generator_like_jet_pt_axis: bh.axis.Regular,
    ) -> "SubstructureResponseHists":
        kt_axis = bh.axis.Regular(25, 0, 25)
        if use_matching_axis:
            number_of_matching_axes = len(cls.matching_name_to_axis_value)
            matching_axis = bh.axis.Regular(number_of_matching_axes, 0, number_of_matching_axes)
        else:
            matching_axis = bh.axis.Regular(1, 0, 1)
        return cls(
            name=name,
            title=title,
            iterative_splittings=iterative_splittings,
            axis_map=axis_map,
            use_matching_axis=use_matching_axis,
            measured_like_n_jets=0,
            generator_like_n_jets=0,
            residuals_jet_pt=bh.Histogram(
                measured_like_jet_pt_axis,
                generator_like_jet_pt_axis,
                bh.axis.Regular(80, -2, 2),
                matching_axis,
                storage=bh.storage.Weight(),
            ),
            residuals_kt=bh.Histogram(
                measured_like_jet_pt_axis,
                # This should be for the measured-like kt values.
                kt_axis,
                generator_like_jet_pt_axis,
                bh.axis.Regular(80, -2, 2),
                matching_axis,
                storage=bh.storage.Weight(),
            ),
            response_kt=bh.Histogram(
                measured_like_jet_pt_axis,
                kt_axis,
                generator_like_jet_pt_axis,
                kt_axis,
                matching_axis,
                storage=bh.storage.Weight(),
            ),
        )

    def fill(
        self,
        measured_like_inputs: FillHistogramInput,
        generator_like_inputs: FillHistogramInput,
        matching_selections: MatchingSelections,
        weight: float,
        jet_R: float,
    ) -> None:
        # Validation
        # Give a useful error message
        if not all(isinstance(hist, bh.Histogram) for _, hist in self):
            raise ValueError("Not all hists are boost histograms! Cannot fill!")
        # And then help out mypy...
        assert (
            isinstance(self.residuals_jet_pt, bh.Histogram)
            and isinstance(self.residuals_kt, bh.Histogram)
            and isinstance(self.response_kt, bh.Histogram)
        )
        # Need to store the number of jets along the histograms.
        self.measured_like_n_jets += measured_like_inputs.n_jets * weight
        self.generator_like_n_jets += generator_like_inputs.n_jets * weight

        for matching_type, matching_axis_value in self.matching_name_to_axis_value.items():
            # Setup
            if not self.use_matching_axis:
                # We didn't provide the matching selections, so we just want to fill the 'all' bin,
                # and then stop filling.
                # So we skip if it's not all.
                if matching_type != "all":
                    continue
                # Create a true mask in the right shape.
                mask = np.ones(len(measured_like_inputs.jets.jet_pt), dtype=bool)
            else:
                mask = matching_selections[matching_type]
            measured_like_jet_pt = measured_like_inputs.jets.jet_pt[mask]
            generator_like_jet_pt = generator_like_inputs.jets.jet_pt[mask]

            # Jet pt residuals
            self.residuals_jet_pt.fill(
                measured_like_jet_pt,
                generator_like_jet_pt,
                (measured_like_jet_pt - generator_like_jet_pt) / generator_like_jet_pt,
                matching_axis_value,
                weight=weight,
            )

            # Store the kt residual and response
            # TODO: Can we do better than this pad and fillna hack??
            #       The length of those values can be shorter than the jet pt length due to
            #       the z_cutoff. Otherwise, they have no effect.
            measured_like_kt = measured_like_inputs.splittings.kt.pad(1).fillna(0).flatten()[mask]
            generator_like_kt = generator_like_inputs.splittings.kt.pad(1).fillna(0).flatten()[mask]
            self.residuals_kt.fill(
                measured_like_jet_pt,
                measured_like_kt,
                generator_like_jet_pt,
                (measured_like_kt - generator_like_kt) / generator_like_kt,
                matching_axis_value,
                weight=weight,
            )
            self.response_kt.fill(
                measured_like_jet_pt,
                measured_like_kt,
                generator_like_jet_pt,
                generator_like_kt,
                matching_axis_value,
                weight=weight,
            )


@attr.s
class SubstructureResponseExtendedHists(SubstructureResponseHists):
    response_z: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    response_delta_R: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    response_theta: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    response_splitting_number: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()

    @property
    def attributes_to_skip(self) -> List[str]:
        attrs = super().attributes_to_skip
        attrs.extend(["n_hybrid_jets", "n_true_jets"])
        return attrs

    @classmethod
    def create_boost_histograms(
        cls: Type["SubstructureResponseExtendedHists"],
        name: str,
        title: str,
        iterative_splittings: bool,
        axis_map: Mapping[str, str],
        use_matching_axis: bool,
        measured_like_jet_pt_axis: bh.axis.Regular,
        generator_like_jet_pt_axis: bh.axis.Regular,
    ) -> "SubstructureResponseExtendedHists":
        # Create a temporary object to handle create the hists that it already knows. We'll then assign those.
        # NOTE: Can't use super here because it will pass this object in the place of the super class.
        #       Usually this is what someone would want, but not here - we literally want to construct the
        #       parent object. So we just do it by hand (although it's not ideal).
        temp_response_hists_base = SubstructureResponseHists.create_boost_histograms(
            name=name,
            title=title,
            iterative_splittings=iterative_splittings,
            axis_map=axis_map,
            use_matching_axis=use_matching_axis,
            measured_like_jet_pt_axis=measured_like_jet_pt_axis,
            generator_like_jet_pt_axis=generator_like_jet_pt_axis,
        )
        # Setup the rest of the axes.
        z_axis = bh.axis.Regular(20, 0, 0.5)
        delta_R_axis = bh.axis.Regular(20, 0, 0.4)
        theta_axis = bh.axis.Regular(20, 0, 1)
        splitting_number_axis = bh.axis.Regular(10, 0, 10)
        if use_matching_axis:
            number_of_matching_axes = len(cls.matching_name_to_axis_value)
            matching_axis = bh.axis.Regular(number_of_matching_axes, 0, number_of_matching_axes)
        else:
            use_matching_axis = bh.axis.Regular(1, 0, 1)
        return cls(
            name=name,
            title=title,
            iterative_splittings=iterative_splittings,
            axis_map=axis_map,
            use_matching_axis=use_matching_axis,
            measured_like_n_jets=0,
            generator_like_n_jets=0,
            residuals_jet_pt=temp_response_hists_base.residuals_jet_pt,
            residuals_kt=temp_response_hists_base.residuals_kt,
            response_kt=temp_response_hists_base.response_kt,
            response_z=bh.Histogram(
                measured_like_jet_pt_axis,
                z_axis,
                generator_like_jet_pt_axis,
                z_axis,
                matching_axis,
                storage=bh.storage.Weight(),
            ),
            response_delta_R=bh.Histogram(
                measured_like_jet_pt_axis,
                delta_R_axis,
                generator_like_jet_pt_axis,
                delta_R_axis,
                matching_axis,
                storage=bh.storage.Weight(),
            ),
            response_theta=bh.Histogram(
                measured_like_jet_pt_axis,
                theta_axis,
                generator_like_jet_pt_axis,
                theta_axis,
                matching_axis,
                storage=bh.storage.Weight(),
            ),
            response_splitting_number=bh.Histogram(
                measured_like_jet_pt_axis,
                splitting_number_axis,
                generator_like_jet_pt_axis,
                splitting_number_axis,
                matching_axis,
                storage=bh.storage.Weight(),
            ),
        )

    def fill(
        self,
        measured_like_inputs: FillHistogramInput,
        generator_like_inputs: FillHistogramInput,
        matching_selections: MatchingSelections,
        weight: float,
        jet_R: float,
    ) -> None:
        super().fill(
            measured_like_inputs=measured_like_inputs,
            generator_like_inputs=generator_like_inputs,
            matching_selections=matching_selections,
            weight=weight,
            jet_R=jet_R,
        )

        # Validation
        # Give a useful error message
        if not all(isinstance(hist, bh.Histogram) for _, hist in self):
            raise ValueError("Not all hists are boost histograms! Cannot fill!")

        # And then help out mypy...
        assert (
            isinstance(self.response_z, bh.Histogram)
            and isinstance(self.response_delta_R, bh.Histogram)
            and isinstance(self.response_theta, bh.Histogram)
            and isinstance(self.response_splitting_number, bh.Histogram)
        )

        for matching_type, matching_axis_value in self.matching_name_to_axis_value.items():
            # Setup
            if not self.use_matching_axis:
                # We didn't provide the matching selections, so we just want to fill the 'all' bin,
                # and then stop filling.
                # So we skip if it's not all.
                if matching_type != "all":
                    continue
                # Create a true mask in the right shape.
                mask = np.ones(len(measured_like_inputs.jets.jet_pt), dtype=bool)
            else:
                mask = matching_selections[matching_type]
            measured_like_jet_pt = measured_like_inputs.jets.jet_pt[mask]
            generator_like_jet_pt = generator_like_inputs.jets.jet_pt[mask]

            self.response_z.fill(
                measured_like_jet_pt,
                measured_like_inputs.splittings.z.pad(1).fillna(0).flatten()[mask],
                generator_like_jet_pt,
                generator_like_inputs.splittings.z.pad(1).fillna(0).flatten()[mask],
                matching_axis_value,
                weight=weight,
            )
            self.response_delta_R.fill(
                measured_like_jet_pt,
                measured_like_inputs.splittings.delta_R.pad(1).fillna(0).flatten()[mask],
                generator_like_jet_pt,
                generator_like_inputs.splittings.delta_R.pad(1).fillna(0).flatten()[mask],
                matching_axis_value,
                weight=weight,
            )
            self.response_theta.fill(
                measured_like_jet_pt,
                measured_like_inputs.splittings.theta(jet_R).pad(1).fillna(0).flatten()[mask],
                generator_like_jet_pt,
                generator_like_inputs.splittings.theta(jet_R).pad(1).fillna(0).flatten()[mask],
                matching_axis_value,
                weight=weight,
            )
            self.response_splitting_number.fill(
                measured_like_jet_pt,
                _calculate_splitting_number(measured_like_inputs.indices)[mask],
                generator_like_jet_pt,
                _calculate_splitting_number(generator_like_inputs.indices)[mask],
                matching_axis_value,
                weight=weight,
            )


@attr.s
class SubstructureMatchingSubjetHists(SubstructureHistsBase):
    name: str = attr.ib()
    title: str = attr.ib()
    iterative_splittings: bool = attr.ib()
    all: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    pure: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    leading_untagged_subleading_correct: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    leading_correct_subleading_untagged: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    leading_untagged_subleading_mistag: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    leading_mistag_subleading_untagged: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    swap: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    both_untagged: Union[bh.Histogram, binned_data.BinnedData] = attr.ib()
    matching_name_to_axis_value: ClassVar[Dict[str, int]] = _matching_name_to_axis_value

    # @property
    # def attributes_to_skip(self) -> List[str]:
    #    attrs = super().attributes_to_skip
    #    attrs.extend(["n_jets"])
    #    return attrs

    def __iadd__(self, other: SubstructureHistsBase) -> "SubstructureMatchingSubjetHists":
        """ Handles a += b """
        # Validation
        if not isinstance(other, type(self)):
            raise TypeError(f"Must pass type {type(self)}. Passed: {type(other)}.")
        if self.iterative_splittings != other.iterative_splittings:
            raise TypeError(
                f"The types of splittings are different! self: {self.iterative_splittings}, other: {other.iterative_splittings}"
            )

        self.name = f"{self.name}_{other.name}" if self.name != other.name else self.name
        self.title = f"{self.title}_{other.title}" if self.title != other.title else self.title
        # Don't need to update iterative_splittings since they must be the same!
        # self.n_jets += self.n_jets
        for (k, v), (k_other, v_other) in zip(self, other):
            v += v_other

        return self

    @classmethod
    def create_boost_histograms(
        cls: Type["SubstructureMatchingSubjetHists"], name: str, title: str, iterative_splittings: bool
    ) -> "SubstructureMatchingSubjetHists":
        jet_pt_axis = bh.axis.Regular(75, 0, 150)
        kt_axis = bh.axis.Regular(25, 0, 25)
        hybrid_jet_pt_axis = bh.axis.Regular(8, 0, 160)
        return cls(
            name=name,
            title=title,
            iterative_splittings=iterative_splittings,
            all=bh.Histogram(jet_pt_axis, kt_axis, hybrid_jet_pt_axis, kt_axis, storage=bh.storage.Weight()),
            pure=bh.Histogram(jet_pt_axis, kt_axis, hybrid_jet_pt_axis, kt_axis, storage=bh.storage.Weight()),
            leading_untagged_subleading_correct=bh.Histogram(
                hybrid_jet_pt_axis, kt_axis, jet_pt_axis, kt_axis, storage=bh.storage.Weight()
            ),
            leading_correct_subleading_untagged=bh.Histogram(
                hybrid_jet_pt_axis, kt_axis, jet_pt_axis, kt_axis, storage=bh.storage.Weight()
            ),
            leading_untagged_subleading_mistag=bh.Histogram(
                hybrid_jet_pt_axis, kt_axis, jet_pt_axis, kt_axis, storage=bh.storage.Weight()
            ),
            leading_mistag_subleading_untagged=bh.Histogram(
                hybrid_jet_pt_axis, kt_axis, jet_pt_axis, kt_axis, storage=bh.storage.Weight()
            ),
            swap=bh.Histogram(hybrid_jet_pt_axis, kt_axis, jet_pt_axis, kt_axis, storage=bh.storage.Weight()),
            both_untagged=bh.Histogram(hybrid_jet_pt_axis, kt_axis, jet_pt_axis, kt_axis, storage=bh.storage.Weight()),
        )

    def fill(
        self,
        matched_inputs: FillHistogramInput,
        hybrid_inputs: FillHistogramInput,
        matching_selections: MatchingSelections,
        weight: float,
    ) -> None:
        # Validation
        # Give a useful error message
        if not all(isinstance(hist, bh.Histogram) for _, hist in self):
            raise ValueError("Not all hists are boost histograms! Cannot fill!")

        # Fill the matching hists.
        for matching_type, matching_hist in self:
            # Help out mypy
            assert isinstance(matching_hist, bh.Histogram)

            # Then make our selections and fill.
            selection = matching_selections[matching_type]
            matching_hist.fill(
                hybrid_inputs.jets.jet_pt[selection],
                hybrid_inputs.splittings.kt[selection],
                matched_inputs.jets.jet_pt[selection],
                matched_inputs.splittings.kt[selection],
                weight=weight,
            )


# NOTE: Typing here is super sketchy. I must be misunderstanding something about the mypy typing model,
#       because I can't seem to possibly find the right variations to make any sense.
#       I would have thought that binding to the base class would have been reasonable, but apparently not.
#       If I do, it just seems to cause more problems...
T_SubstructureHists = TypeVar(
    "T_SubstructureHists",
    SubstructureHists,
    SubstructureToyHists,
    SubstructureResponseHists,
    SubstructureResponseExtendedHists,
    SubstructureMatchingSubjetHists,
)


@attr.s
class Hists(Generic[T_SubstructureHists]):
    inclusive: T_SubstructureHists = attr.ib()
    dynamical_z: T_SubstructureHists = attr.ib()
    dynamical_kt: T_SubstructureHists = attr.ib()
    dynamical_time: T_SubstructureHists = attr.ib()
    leading_kt: T_SubstructureHists = attr.ib()
    leading_kt_hard_cutoff: T_SubstructureHists = attr.ib()

    def __iter__(self) -> Iterator[Tuple[str, T_SubstructureHists]]:
        # We don't want to recurse because we have to handle the dict conversion more careful
        # for the SubstructureHists
        return iter(attr.asdict(self, recurse=False).items())

    def __add__(self, other: "Hists[T_SubstructureHists]") -> "Hists[T_SubstructureHists]":
        """ Handles a = b + c. """
        new = copy.deepcopy(self)
        new += other
        return new

    def __iadd__(
        self: "Hists[T_SubstructureHists]", other: "Hists[T_SubstructureHists]"
    ) -> "Hists[T_SubstructureHists]":
        """ Handles a += b. """
        # Add the stored values together.
        for (k, v), (k_other, v_other) in zip(self, other):
            # Validation
            if k != k_other:
                raise ValueError(f"Somehow keys mismatch. self key: {k}, other key: {k_other}")

            # Assumes that they are passed by reference.
            v += v_other  # type: ignore

        return self

    def __radd__(self, other: "Hists[T_SubstructureHists]") -> "Hists[T_SubstructureHists]":
        """ For use with sum(...). """
        if other == 0:
            return self
        else:
            # Help out mypy
            assert not isinstance(other, int)
            return self + other

    def convert_boost_histograms_to_binned_data(self) -> None:
        for _, v in self:
            v.convert_boost_histograms_to_binned_data()


def create_substructure_hists(iterative_splittings: bool, z_cutoff: float) -> Hists[SubstructureHists]:
    kt_axis = bh.axis.Regular(50, 0, 25)
    inclusive = SubstructureHists.create_boost_histograms(
        name="inclusive",
        title="Inclusive",
        iterative_splittings=iterative_splittings,
        # This isn't really going to be meaningful for the inclusive case...
        values_axis=bh.axis.Regular(10, 0, 100),
    )
    dynamical_z = SubstructureHists.create_boost_histograms(
        name="dynamical_z",
        title="zDrop",
        iterative_splittings=iterative_splittings,
        values_axis=bh.axis.Regular(50, 0, 50),
    )
    dynamical_kt = SubstructureHists.create_boost_histograms(
        name="dynamical_kt",
        title="ktDrop",
        iterative_splittings=iterative_splittings,
        values_axis=kt_axis,
    )
    dynamical_time = SubstructureHists.create_boost_histograms(
        name="dynamical_time",
        title="timeDrop",
        iterative_splittings=iterative_splittings,
        values_axis=bh.axis.Regular(50, 0, 50),
    )
    leading_kt = SubstructureHists.create_boost_histograms(
        name="leading_kt",
        title=r"Leading $k_{\text{T}}$",
        iterative_splittings=iterative_splittings,
        values_axis=kt_axis,
    )
    leading_kt_hard_cutoff = SubstructureHists.create_boost_histograms(
        name="leading_kt_hard_cutoff",
        title=fr"$z > {z_cutoff}$ Leading $k_{{\text{{T}}}}$",
        iterative_splittings=iterative_splittings,
        values_axis=kt_axis,
    )

    # TODO: SD
    return Hists(
        inclusive=inclusive,
        dynamical_z=dynamical_z,
        dynamical_kt=dynamical_kt,
        dynamical_time=dynamical_time,
        leading_kt=leading_kt,
        leading_kt_hard_cutoff=leading_kt_hard_cutoff,
    )


def create_substructure_toy_hists(iterative_splittings: bool, z_cutoff: float) -> Hists[SubstructureToyHists]:
    kt_axis = bh.axis.Regular(50, 0, 25)
    inclusive = SubstructureToyHists.create_boost_histograms(
        name="inclusive",
        title="Inclusive",
        iterative_splittings=iterative_splittings,
        # This isn't really going to be meaningful for the inclusive case...
        values_axis=bh.axis.Regular(10, 0, 100),
    )
    dynamical_z = SubstructureToyHists.create_boost_histograms(
        name="dynamical_z",
        title="zDrop",
        iterative_splittings=iterative_splittings,
        values_axis=bh.axis.Regular(50, 0, 50),
    )
    dynamical_kt = SubstructureToyHists.create_boost_histograms(
        name="dynamical_kt",
        title="ktDrop",
        iterative_splittings=iterative_splittings,
        values_axis=kt_axis,
    )
    dynamical_time = SubstructureToyHists.create_boost_histograms(
        name="dynamical_time",
        title="timeDrop",
        iterative_splittings=iterative_splittings,
        values_axis=bh.axis.Regular(50, 0, 50),
    )
    leading_kt = SubstructureToyHists.create_boost_histograms(
        name="leading_kt",
        title=r"Leading $k_{\text{T}}$",
        iterative_splittings=iterative_splittings,
        values_axis=kt_axis,
    )
    leading_kt_hard_cutoff = SubstructureToyHists.create_boost_histograms(
        name="leading_kt_hard_cutoff",
        title=fr"$z > {z_cutoff}$ Leading $k_{{\text{{T}}}}$",
        iterative_splittings=iterative_splittings,
        values_axis=kt_axis,
    )

    # TODO: SD
    return Hists(
        inclusive=inclusive,
        dynamical_z=dynamical_z,
        dynamical_kt=dynamical_kt,
        dynamical_time=dynamical_time,
        leading_kt=leading_kt,
        leading_kt_hard_cutoff=leading_kt_hard_cutoff,
    )


_T_ResponseHists = TypeVar("_T_ResponseHists", SubstructureResponseHists, SubstructureResponseExtendedHists)


def _create_substructure_response_hists(
    response_hists_class: Type[_T_ResponseHists],
    iterative_splittings: bool,
    z_cutoff: float,
    axis_map: Mapping[str, str],
    use_matching_axis: bool,
    measured_like_jet_pt_axis: bh.axis.Regular,
    generator_like_jet_pt_axis: bh.axis.Regular,
) -> Hists[_T_ResponseHists]:
    inclusive = response_hists_class.create_boost_histograms(
        name="inclusive_response",
        title="Inclusive",
        iterative_splittings=iterative_splittings,
        axis_map=axis_map,
        use_matching_axis=use_matching_axis,
        measured_like_jet_pt_axis=measured_like_jet_pt_axis,
        generator_like_jet_pt_axis=generator_like_jet_pt_axis,
    )
    dynamical_z = response_hists_class.create_boost_histograms(
        name="dynamical_z_response",
        title="zDrop",
        iterative_splittings=iterative_splittings,
        axis_map=axis_map,
        use_matching_axis=use_matching_axis,
        measured_like_jet_pt_axis=measured_like_jet_pt_axis,
        generator_like_jet_pt_axis=generator_like_jet_pt_axis,
    )
    dynamical_kt = response_hists_class.create_boost_histograms(
        name="dynamical_kt_response",
        title="ktDrop",
        iterative_splittings=iterative_splittings,
        axis_map=axis_map,
        use_matching_axis=use_matching_axis,
        measured_like_jet_pt_axis=measured_like_jet_pt_axis,
        generator_like_jet_pt_axis=generator_like_jet_pt_axis,
    )
    dynamical_time = response_hists_class.create_boost_histograms(
        name="dynamical_time_response",
        title="timeDrop",
        iterative_splittings=iterative_splittings,
        axis_map=axis_map,
        use_matching_axis=use_matching_axis,
        measured_like_jet_pt_axis=measured_like_jet_pt_axis,
        generator_like_jet_pt_axis=generator_like_jet_pt_axis,
    )
    leading_kt = response_hists_class.create_boost_histograms(
        name="leading_kt_response",
        title=r"Leading $k_{\text{T}}$",
        iterative_splittings=iterative_splittings,
        axis_map=axis_map,
        use_matching_axis=use_matching_axis,
        measured_like_jet_pt_axis=measured_like_jet_pt_axis,
        generator_like_jet_pt_axis=generator_like_jet_pt_axis,
    )
    leading_kt_hard_cutoff = response_hists_class.create_boost_histograms(
        name="leading_kt_hard_cutoff_response",
        title=fr"$z > {z_cutoff}$ Leading $k_{{\text{{T}}}}$",
        iterative_splittings=iterative_splittings,
        axis_map=axis_map,
        use_matching_axis=use_matching_axis,
        measured_like_jet_pt_axis=measured_like_jet_pt_axis,
        generator_like_jet_pt_axis=generator_like_jet_pt_axis,
    )

    # TODO: SD
    return Hists(
        inclusive=inclusive,
        dynamical_z=dynamical_z,
        dynamical_kt=dynamical_kt,
        dynamical_time=dynamical_time,
        leading_kt=leading_kt,
        leading_kt_hard_cutoff=leading_kt_hard_cutoff,
    )


def create_substructure_response_hists(
    iterative_splittings: bool,
    z_cutoff: float,
    axis_map: Mapping[str, str],
    use_matching_axis: bool,
    measured_like_jet_pt_axis: bh.axis.Regular,
    generator_like_jet_pt_axis: bh.axis.Regular,
) -> Hists[SubstructureResponseHists]:
    return _create_substructure_response_hists(
        SubstructureResponseHists,
        iterative_splittings=iterative_splittings,
        z_cutoff=z_cutoff,
        axis_map=axis_map,
        use_matching_axis=use_matching_axis,
        measured_like_jet_pt_axis=measured_like_jet_pt_axis,
        generator_like_jet_pt_axis=generator_like_jet_pt_axis,
    )


def create_substructure_response_hists_extended(
    iterative_splittings: bool,
    z_cutoff: float,
    axis_map: Mapping[str, str],
    use_matching_axis: bool,
    measured_like_jet_pt_axis: bh.axis.Regular,
    generator_like_jet_pt_axis: bh.axis.Regular,
) -> Hists[SubstructureResponseExtendedHists]:
    return _create_substructure_response_hists(
        SubstructureResponseExtendedHists,
        iterative_splittings=iterative_splittings,
        z_cutoff=z_cutoff,
        axis_map=axis_map,
        use_matching_axis=use_matching_axis,
        measured_like_jet_pt_axis=measured_like_jet_pt_axis,
        generator_like_jet_pt_axis=generator_like_jet_pt_axis,
    )


def create_matching_hists(iterative_splittings: bool, z_cutoff: float) -> Hists[SubstructureMatchingSubjetHists]:
    """Matching subjets hists"""
    inclusive = SubstructureMatchingSubjetHists.create_boost_histograms(
        name="inclusive_response",
        title="Inclusive",
        iterative_splittings=iterative_splittings,
    )
    dynamical_z = SubstructureMatchingSubjetHists.create_boost_histograms(
        name="dynamical_z_response",
        title="zDrop",
        iterative_splittings=iterative_splittings,
    )
    dynamical_kt = SubstructureMatchingSubjetHists.create_boost_histograms(
        name="dynamical_kt_response", title="ktDrop", iterative_splittings=iterative_splittings
    )
    dynamical_time = SubstructureMatchingSubjetHists.create_boost_histograms(
        name="dynamical_time_response",
        title="timeDrop",
        iterative_splittings=iterative_splittings,
    )
    leading_kt = SubstructureMatchingSubjetHists.create_boost_histograms(
        name="leading_kt_response",
        title=r"Leading $k_{\text{T}}$",
        iterative_splittings=iterative_splittings,
    )
    leading_kt_hard_cutoff = SubstructureMatchingSubjetHists.create_boost_histograms(
        name="leading_kt_hard_cutoff_response",
        title=fr"$z > {z_cutoff}$ Leading $k_{{\text{{T}}}}$",
        iterative_splittings=iterative_splittings,
    )

    # TODO: SD
    return Hists(
        inclusive=inclusive,
        dynamical_z=dynamical_z,
        dynamical_kt=dynamical_kt,
        dynamical_time=dynamical_time,
        leading_kt=leading_kt,
        leading_kt_hard_cutoff=leading_kt_hard_cutoff,
    )


@attr.s
class SingleTreeResultBase:
    def keys(self) -> Iterator[str]:
        return iter(attr.fields_dict(type(self)).keys())

    def items(self) -> Iterator[Tuple[str, Dict[Identifier, Hists[T_SubstructureHists]]]]:
        return iter(attr.asdict(self, recurse=False).items())

    def values(self) -> Iterator[Dict[Identifier, Hists[T_SubstructureHists]]]:
        return iter(attr.astuple(self, recurse=False))

    def create_hists(self, dataset: Dataset, **selections: Any) -> bool:
        raise NotImplementedError("Needs to be implemented by the daughter class")


@attr.s
class SingleTreeResult(SingleTreeResultBase):
    hists: Dict[Identifier, Hists[SubstructureHists]] = attr.ib(factory=dict)

    def create_hists(self, dataset: Dataset, **selections: Any) -> bool:
        for kwargs in helpers.dict_product(selections):
            # We don't care about the jet pt bin for creating the hists - just the identifier. So we drop it here.
            create_hists_args = {k: v for k, v in kwargs.items() if k != "jet_pt_bin"}
            # And also add the z_cutoff from the dataset settings.
            create_hists_args["z_cutoff"] = dataset.settings.z_cutoff
            self.hists[Identifier(**kwargs)] = create_substructure_hists(**create_hists_args)

        return True


@attr.s
class SingleTreeToyResult(SingleTreeResultBase):
    hists: Dict[Identifier, Hists[SubstructureToyHists]] = attr.ib(factory=dict)

    def create_hists(self, dataset: Dataset, **selections: Any) -> bool:
        for kwargs in helpers.dict_product(selections):
            # We don't care about the jet pt bin for creating the hists - just the identifier. So we drop it here.
            create_hists_args = {k: v for k, v in kwargs.items() if k != "jet_pt_bin"}
            # And also add the z_cutoff from the dataset settings.
            create_hists_args["z_cutoff"] = dataset.settings.z_cutoff
            self.hists[Identifier(**kwargs)] = create_substructure_toy_hists(**create_hists_args)

        return True


@attr.s
class SingleTreeEmbeddingResult(SingleTreeResultBase):
    true_hists: Dict[Identifier, Hists[SubstructureHists]] = attr.ib(factory=dict)
    det_level_hists: Dict[Identifier, Hists[SubstructureHists]] = attr.ib(factory=dict)
    hybrid_hists: Dict[Identifier, Hists[SubstructureHists]] = attr.ib(factory=dict)
    detector_particle_response: Dict[Identifier, Hists[SubstructureResponseHists]] = attr.ib(factory=dict)
    hybrid_detector_response: Dict[Identifier, Hists[SubstructureResponseHists]] = attr.ib(factory=dict)
    hybrid_particle_response: Dict[Identifier, Hists[SubstructureResponseExtendedHists]] = attr.ib(factory=dict)

    def create_hists(self, dataset: Dataset, **selections: Any) -> bool:
        for kwargs in helpers.dict_product(selections):
            # We don't care about the jet pt bin for creating the hists - just the identifier. So we drop it here.
            create_hists_args = {k: v for k, v in kwargs.items() if k != "jet_pt_bin"}
            # And also add the z_cutoff from the dataset settings.
            create_hists_args["z_cutoff"] = dataset.settings.z_cutoff
            self.true_hists[Identifier(**kwargs)] = create_substructure_hists(**create_hists_args)
            self.det_level_hists[Identifier(**kwargs)] = create_substructure_hists(**create_hists_args)
            # Hybrid hists
            self.hybrid_hists[Identifier(**kwargs)] = create_substructure_hists(**create_hists_args)

        # Hybrid-particle and Detector-particle response are binned in jet pt, so we don't have to make a selection now.
        # Instead, we'll associate them with a maximal jet pt bin.
        maximal_jet_pt_bin = helpers.RangeSelector.full_range_over_selections(selections["jet_pt_bin"])
        selections_with_maximal_jet_pt = selections.copy()
        selections_with_maximal_jet_pt["jet_pt_bin"] = [maximal_jet_pt_bin]
        for kwargs in helpers.dict_product(selections_with_maximal_jet_pt):
            # We don't care about the jet pt bin for creating the hists - just the identifier. So we drop it here.
            create_hists_args = {k: v for k, v in kwargs.items() if k != "jet_pt_bin"}
            # Detector-particle response.
            self.detector_particle_response[Identifier(**kwargs)] = create_substructure_response_hists(
                z_cutoff=dataset.settings.z_cutoff,
                axis_map={"measured_like": "detector", "generator_like": "particle"},
                use_matching_axis=False,
                measured_like_jet_pt_axis=bh.axis.Regular(16, 0, 160),
                generator_like_jet_pt_axis=bh.axis.Regular(16, 0, 160),
                **create_hists_args,
            )
            # Hybrid-detector response.
            self.hybrid_detector_response[Identifier(**kwargs)] = create_substructure_response_hists(
                z_cutoff=dataset.settings.z_cutoff,
                axis_map={"measured_like": "hybrid", "generator_like": "detector"},
                use_matching_axis=True,
                measured_like_jet_pt_axis=bh.axis.Regular(16, 0, 160),
                generator_like_jet_pt_axis=bh.axis.Regular(16, 0, 160),
                **create_hists_args,
            )
            # Hybrid-particle response.
            self.hybrid_particle_response[Identifier(**kwargs)] = create_substructure_response_hists_extended(
                z_cutoff=dataset.settings.z_cutoff,
                axis_map={"measured_like": "hybrid", "generator_like": "particle"},
                use_matching_axis=True,
                measured_like_jet_pt_axis=bh.axis.Regular(16, 0, 160),
                generator_like_jet_pt_axis=bh.axis.Regular(16, 0, 160),
                **create_hists_args,
            )

            # Cross check that we have a reasonable jet_pt_bin
            true_hists_jet_pt_bins = [identifier.jet_pt_bin for identifier in self.true_hists.keys()]
            if kwargs["jet_pt_bin"] not in true_hists_jet_pt_bins:
                raise ValueError(
                    "Expected jet pt bin {kwargs['jet_pt_bin']} used for the response and matching to be in the true hists. However, the true_hists only contain: {true_hists_jet_pt_bins}"
                )

        return True
