""" Input sources

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import collections.abc
import itertools
import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Union,
)
from typing_extensions import Protocol

import attr
import awkward as ak
import numpy as np
import uproot

from mammoth.framework import models, utils

logger = logging.getLogger(__name__)


class Source(Protocol):
    """Data source.

    Attributes:
        metadata: Source metadata.
    """

    metadata: MutableMapping[str, Any]

    def __len__(self) -> int:
        """Number of entries in the source."""
        ...

    def data(self) -> ak.Array:
        """Return data from the source.

        Returns:
            Data in an awkward array.
        """
        ...


class SourceWithChunks(Source, Protocol):
    """A source that operates in chunks."""

    chunk_size: int


def _convert_range(entry_range: Union[utils.Range, Sequence[float]]) -> utils.Range:
    """Convert sequences to Range.

    Args:
        entry_range: Range of entries to be stored in a Range.
    Returns:
        Range
    """
    if isinstance(entry_range, utils.Range):
        return entry_range
    return utils.Range(*entry_range)


@attr.define
class UprootSource:
    _filename: Path = attr.field(converter=Path)
    _tree_name: str
    _columns: Sequence[str] = attr.Factory(list)
    _entry_range: utils.Range = attr.field(converter=_convert_range, default=utils.Range(None, None))
    metadata: MutableMapping[str, Any] = attr.Factory(dict)

    def __len__(self) -> int:
        if "n_entries" in self.metadata:
            return int(self.metadata["n_entries"])
        raise ValueError("N entries not yet available.")

    def data(self) -> ak.Array:
        with uproot.open(self._filename) as f:
            # Allow for star matching in tree_name
            if "*" in self._tree_name:
                logger.debug(f"Searching for tree name pattern {self._tree_name}")
                # Search for keys which contain the provided tree name. Very nicely, uproot already has this built-in
                _possible_tree_names = f.keys(cycle=False, filter_name=self._tree_name, filter_classname="TTree")
                if len(_possible_tree_names) != 1:
                    raise ValueError(
                        f"Ambiguous tree name '{self._tree_name}'. Please revise it as needed. Options: {_possible_tree_names}"
                    )
                # We're good - let's keep going
                self._tree_name = _possible_tree_names[0]

            tree = f[self._tree_name]

            # First, let's setup the arguments
            # Columns
            reading_kwargs: Dict[str, Any] = {
                "expressions": self._columns if self._columns else None,
            }
            # Add restricted start and stop entries if requested.
            # Only if we specify a start and stop do we actually pass it on to uproot.
            # Check explicitly for not none because min could be 0 and still a valid range.
            if self._entry_range.min is not None and self._entry_range.max is not None:
                reading_kwargs.update(
                    {
                        "entry_start": self._entry_range.min,
                        "entry_stop": self._entry_range.max,
                    }
                )

            # Add metadata
            self.metadata["entry_start"] = self._entry_range.min if self._entry_range.min is not None else 0
            self.metadata["entry_stop"] = (
                self._entry_range.max if self._entry_range.max is not None else tree.num_entries
            )
            self.metadata["n_entries"] = self.metadata["entry_stop"] - self.metadata["entry_start"]

            return tree.arrays(**reading_kwargs)


def chunked_uproot_source(
    filename: Path,
    tree_name: str,
    chunk_size: int,
    columns: Optional[Sequence[str]] = None,
) -> List[UprootSource]:
    """Create a set of uproot sources in chunks for a given filename.

    This is most likely to be the main interface.

    Returns:
        List of UprootSource configured with the provided properties.
    """
    sources = []
    if columns is None:
        columns = []
    with uproot.open(filename) as f:
        number_of_entries = f[tree_name].num_entries

        start = 0
        continue_iterating = True
        while continue_iterating:
            end = start + chunk_size
            # Ensure that we never ask for more entries than are in the file.
            if start + chunk_size > number_of_entries:
                end = number_of_entries
                continue_iterating = False
            # Store the start and stop for convenience.
            sources.append(
                UprootSource(
                    filename=filename,
                    tree_name=tree_name,
                    columns=columns,
                    entry_range=utils.Range(start, end),
                )
            )
            # Move up to the next iteration.
            start = end

    return sources


@attr.define
class ParquetSource:
    _filename: Path = attr.field(converter=Path)
    _columns: Sequence[str] = attr.Factory(list)
    metadata: MutableMapping[str, Any] = attr.Factory(dict)

    def __len__(self) -> int:
        if "n_entries" in self.metadata:
            return int(self.metadata["n_entries"])
        raise ValueError("N entries not yet available.")

    def data(self) -> ak.Array:
        arrays = ak.from_parquet(
            self._filename,
            columns=self._columns if self._columns else None,
        )

        # Extract metadata
        self.metadata["entry_start"] = 0
        self.metadata["entry_stop"] = len(arrays)
        self.metadata["n_entries"] = self.metadata["entry_stop"] - self.metadata["entry_start"]

        return arrays


@attr.define
class JetscapeSource(ParquetSource):
    """Jetscape source via Parquet file.

    Nothing needs to be done here.
    """

    ...


@attr.define
class PythiaSource:
    config: Path = attr.field(converter=Path)
    chunk_size: int
    metadata: MutableMapping[str, Any] = attr.Factory(dict)

    def __len__(self) -> int:
        return self.chunk_size

    def data(self) -> ak.Array:
        raise NotImplementedError("Working on it...")


@attr.define
class ThermalModelParameters:
    mean: float
    sigma: float
    pt_exponential_scale: float = attr.field(default=0.4)


THERMAL_MODEL_SETTINGS = {
    "central": ThermalModelParameters(mean=2500, sigma=500),
    "semi_central": ThermalModelParameters(mean=1000, sigma=40),
}


@attr.define
class ThermalModelExponential:
    """Thermal background model from Leticia

    Assume thermal particles are massless.
    pt = x*exp(-x/pt_exponential_scale), from 0 to 400, at least 40000 sampling points
    eta = flat from -1 to 1, at least 200 sampling points
    phi = flat from -pi to pi, at least 700 sampling points

    pt exponential scale is 0.4 by default.

    The number of thermal particles is determined by a Gaussian. The parameters are:
    - central: mean = 2500, sigma = 500
    - semi-central: mean = 1000, sigma = 40

    """

    chunk_size: int
    thermal_model_parameters: ThermalModelParameters
    metadata: MutableMapping[str, Any] = attr.Factory(dict)

    def __len__(self) -> int:
        return self.chunk_size

    def data(self) -> ak.Array:
        # Setup
        rng = np.random.default_rng()

        # Determine overall event parameters.
        # NOTE: This is effectively jagged, since the number of particles per event varies
        # NOTE: We round to integers because the number of particles must of course be an int.
        n_particles_per_event = np.rint(
            rng.normal(
                loc=self.thermal_model_parameters.mean,
                scale=self.thermal_model_parameters.sigma,
                size=self.chunk_size,
            ),
        ).astype(np.int32)
        # To help out with this effective jaggedness, we flatten everything, and then will unflatten with awkward.
        total_n_samples = int(np.sum(n_particles_per_event))

        # Sample the distributions.
        pt = models.sample_x_exp(
            n_samples=total_n_samples,
            scale=self.thermal_model_parameters.pt_exponential_scale,
            x_min=0,
            x_max=400,
        )
        #eta = rng.uniform(low=-1, high=1, size=total_n_samples)
        # We want to match the ALICE TPC acceptance
        eta = rng.uniform(low=-0.9, high=0.9, size=total_n_samples)
        phi = rng.uniform(low=-np.pi, high=np.pi, size=total_n_samples)

        # Need this as an intermediary, so calculate it first
        pz = pt * np.sinh(eta)

        # Finally, add the particle structure at the end.
        # NOTE: We return it wrapped in the "data" key because the framework
        #       expects that every source has some kind of particle column name.
        return ak.Array({"data": ak.unflatten(
            ak.Array(
                {
                    "px": pt * np.cos(phi),
                    "py": pt * np.sin(phi),
                    "pz": pz,
                    "E": np.sqrt(pt ** 2 + pz ** 2),
                }
            ),
            counts=n_particles_per_event,
        )})


@attr.define
class ALICEFastSimTrackingEfficiency:
    """ ALICE fast simulation based on tracking efficiency

    This is definitely a poor man's implementation, but it's fine for a first look.
    """
    particle_level_data: ak.Array
    fast_sim_parameters: models.ALICEFastSimParameters
    metadata: MutableMapping[str, Any] = attr.Factory(dict)

    def __len__(self) -> int:
        if "n_entries" in self.metadata:
            return int(self.metadata["n_entries"])
        raise ValueError("N entries not yet available.")

    def data(self) -> ak.Array:
        # Setup
        rng = np.random.default_rng()

        self.metadata["n_entries"] = len(self.particle_level_data)

        efficiencies = models.alice_fast_sim_tracking_efficiency(
            track_pt=np.asarray(ak.flatten(self.particle_level_data["part_level"].pt, axis=-1)),
            track_eta=np.asarray(ak.flatten(self.particle_level_data["part_level"].eta, axis=-1)),
            event_activity=self.fast_sim_parameters.event_activity,
            period=self.fast_sim_parameters.period,
        )

        n_particles_per_event = ak.num(self.particle_level_data["part_level"], axis=1)
        total_n_particles = ak.sum(n_particles_per_event)

        # Drop values that are higher than the tracking efficiency
        random_values = rng.uniform(low=0.0, high=1.0, size=total_n_particles)
        drop_particles_mask = random_values > efficiencies
        # Since True will keep the particle, we need to invert this
        drop_particles_mask = ~drop_particles_mask

        # Unflatten so we can apply the mask to the existing particles
        drop_particles_mask = ak.unflatten(drop_particles_mask, n_particles_per_event)

        # Finally, add the particle structure at the end.
        # NOTE: We return the fast sim wrapped in the "det_level" key because the framework
        #       expects that every source has some kind of particle column name.
        # NOTE: We also return the "part_level" because it's convenient to have both
        #       together, even if it's in principle available elsewhere. We also include the event
        #       level info for the same reason. I think we're violating the separation of concerns
        #       a little bit, but it seems to work, so good enough for now.
        return ak.Array(
            {
                "det_level": self.particle_level_data["part_level"][drop_particles_mask],
                "part_level": self.particle_level_data["part_level"],
                # Include the rest of the non particle related fields (ie. event level info)
                **{
                    k: v
                    for k, v in zip(ak.fields(self.particle_level_data), ak.unzip(self.particle_level_data))
                    if k not in ["det_level", "part_level"]
                },
            }
        )


def _sources_to_list(sources: Union[Source, Sequence[Source]]) -> Sequence[Source]:
    if not isinstance(sources, collections.abc.Iterable):
        return [sources]
    return sources


@attr.define
class ChunkSource:
    chunk_size: int
    sources: Sequence[Source] = attr.field(converter=_sources_to_list)
    repeat: bool = attr.field(default=False)
    metadata: MutableMapping[str, Any] = attr.Factory(dict)
    _iter_with_data_func: Iterator[ak.Array] = attr.field(init=False, default=None)

    def __len__(self) -> int:
        if "n_entries" in self.metadata:
            return int(self.metadata["n_entries"])
        raise ValueError("N entries not yet available.")

    def data(self) -> ak.Array:
        """Retrieve data to satisfy the given chunk size."""
        # We need to keep track of the iterator so that we can actually advance it.
        # To do this, we store an iter in a private class member, and then we call
        # next on that.
        if not self._iter_with_data_func:
            self._iter_with_data_func = iter(self.data_iter())
        return next(self._iter_with_data_func)

    def data_iter(self) -> Iterable[ak.Array]:
        if self.repeat:
            # See: https://stackoverflow.com/a/24225372/12907985
            source_iter = itertools.chain.from_iterable(itertools.repeat(self.sources))
        else:
            source_iter = iter(self.sources)
        remaining_data = None

        while True:
            if remaining_data is not None:
                _data = remaining_data
                remaining_data = None
            else:
                _data = next(source_iter).data()

            # Regardless of where we end up, the number of entries must be equal to the chunk size
            self.metadata["n_entries"] = self.chunk_size

            # Now, figure out how to get all of the required data.
            if len(_data) == self.chunk_size:
                yield _data
            elif len(_data) < self.chunk_size:
                additional_chunks = []
                remaining_n_events = self.chunk_size - len(_data)
                for _more_data_source in source_iter:
                    _more_data = _more_data_source.data()
                    remaining_n_events -= len(_more_data)
                    if remaining_n_events < 0:
                        # Slice the remaining data and store for the next iteration
                        additional_chunks.append(_more_data[:remaining_n_events])
                        remaining_data = _more_data[remaining_n_events:]
                        break
                    additional_chunks.append(_more_data)
                yield ak.concatenate(
                    [_data, *additional_chunks],
                    axis=0,
                )
            else:
                remaining_n_events = self.chunk_size - len(_data)
                remaining_data = _data[remaining_n_events:]
                yield _data[:remaining_n_events]


def _no_overlapping_keys(
    instance: "MultipleSources",
    attribute: attr.Attribute[Mapping[str, int]],
    value: Mapping[str, int],
) -> None:
    if set(instance._fixed_size_sources).intersection(set(instance._chunked_sources)):
        raise ValueError(
            f"Overlapping keys between fixed size and chunk sources. Fixed size sources: {list(instance._fixed_size_sources)}, chunked sources: {list(instance._chunked_sources)}."
        )


def _contains_signal_and_background(
    instance: "MultipleSources",
    attribute: attr.Attribute[Mapping[str, int]],
    value: Mapping[str, int],
) -> None:
    found_signal = False
    found_background = False
    for k in value.keys():
        if "signal" in k:
            found_signal = True
        if "background" in k:
            found_background = True
    if not found_signal:
        raise ValueError(f"Must contain at least one signal source. Found: {list(value.keys())}.")
    if not found_background:
        raise ValueError(f"Must contain at least one background source. Found: {list(value.keys())}.")


def _has_offset_per_source(
    instance: "MultipleSources",
    attribute: attr.Attribute[Mapping[str, int]],
    value: Mapping[str, int],
) -> None:
    if (set(instance._fixed_size_sources) | set(instance._chunked_sources)) != set(instance._source_index_identifiers):
        raise ValueError(
            f"Mismatch in sources and offsets. Fixed size sources: {list(instance._fixed_size_sources)}, chunked sources: {list(instance._chunked_sources)}, offsets: {list(instance._source_index_identifiers)}"
        )


@attr.define
class MultipleSources:
    """Combine multiple data sources together.

    Think: Embedding into data, embedding into thermal model, etc.

    Attributes:
        _fixed_size_sources: Sources which are of a fixed size. These sources determine the size
            of the chunk that will be provided.
        _chunked_sources: Sources which can provide chunks of data of a specified size. The size
            of these chunks is determined by the fixed sized sources and is set when retrieveing
            the data.
        _source_index_identifiers: Map containing an integer identifier for each source.
    """

    _fixed_size_sources: Mapping[str, Source] = attr.field(validator=[_no_overlapping_keys])
    _chunked_sources: Mapping[str, SourceWithChunks] = attr.field(validator=[_no_overlapping_keys])
    _source_index_identifiers: Mapping[str, int] = attr.field(
        factory=dict,
        validator=[_contains_signal_and_background, _has_offset_per_source],
    )
    metadata: MutableMapping[str, Any] = attr.Factory(dict)

    def __len__(self) -> int:
        if "n_entries" in self.metadata:
            return int(self.metadata["n_entries"])
        raise ValueError("N entries not yet available.")

    def data(self) -> ak.Array:
        # Grab the events from the fixed size sources first
        # NOTE: Sometimes these are already awkward arrays, so we explicitly check for this case for safety
        fixed_sized_data = {
            k: v if isinstance(v, ak.Array) else v.data()
            for k, v in self._fixed_size_sources.items()
        }

        # Cross check that we have the right sizes for all data sources
        lengths = [len(v) for v in fixed_sized_data.values()]
        if lengths.count(lengths[0]) != len(lengths):
            raise ValueError(f"Length of data doesn't match: {lengths}")

        # Set the length of the chunked source based on the size of the fixed size sources
        for v in self._chunked_sources.values():
            v.chunk_size = lengths[0]
        # Now that the chunked data source is well defined, extract the chunked data
        chunked_data = {k: v.data() for k, v in self._chunked_sources.items()}

        # Add metadata
        self.metadata["n_entries"] = lengths[0]

        # NOTE: We're safe to blindly combine these here because the class validates that there
        #       are no overlapping keys between the fixed size and chunked data.
        return ak.zip({**fixed_sized_data, **chunked_data}, depth_limit=1)
