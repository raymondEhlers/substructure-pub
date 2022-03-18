
""" Base analysis functionality

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import functools
import operator
from typing import Mapping, Optional, Sequence

import awkward as ak
import numba as nb
import numpy as np
import particle

@functools.lru_cache()
def _pdg_id_to_mass(pdg_id: int) -> float:
    """ Convert PDG ID to mass.

    We cache the result to speed it up.

    Args:
        pdg_id: PDG ID.
    Returns:
        Mass in GeV.
    """
    m = particle.Particle.from_pdgid(pdg_id).mass
    # Apparently neutrino mass returns None...
    if m is None:
        m = 0
    return m / 1000


@nb.njit  # type: ignore
def _determine_mass_from_PDG(arrays: ak.Array,
                             builder: ak.ArrayBuilder,
                             pdg_id_to_mass: Mapping[int, float],
                             particle_ID_column_name: str = "particle_ID") -> ak.Array:
    """ Determine the mass for each particle based on the PID.

    These masses are in the same shape as the given array so they can be stored alongside them.

    Args:
        arrays: Particles organized in event structure.
        builder: Array builder.
        pdg_id_to_mass: Map from PDG ID to mass. This mapping must be in a form that numba will accept!
        particle_ID_column_name: Name of the particle ID column. Default: "particle_ID"

    Returns:
        Masses calculated according to the PDG code, in the same shape as the input arrays.
    """
    for event in arrays:
        builder.begin_list()
        # The conditions for where we can iterate over the entire event vs only one column aren't yet
        # clear to me. It has something to do with record arrays...
        for particle_ID in event[particle_ID_column_name]:
            builder.append(pdg_id_to_mass[particle_ID])
        builder.end_list()

    return builder


def particle_masses_from_particle_ID(arrays: ak.Array, particle_ID_column_name: str = "particle_ID") -> ak.Array:
    """Determine particle masses based on PDG codes.

    Note:
        `arrays` is expected to be in an event-based format (ie. jagged arrays of particles).

    Args:
        arrays: Particle collection, over which we will determine the masses.
        particle_ID_column_name: Name of the particle ID column. Default: "particle_ID"

    Returns:
        Masses calculated according to the PDG code, in the same shape as the input arrays.
    """
    # Add the masses based on the PDG code.
    # First, determine all possible PDG codes, and then retrieve their masses for a lookup table.
    all_particle_IDs = np.unique(ak.to_numpy(ak.flatten(arrays[particle_ID_column_name])))  # type: ignore
    # NOTE: We need to use this special typed dict with numba.
    pdg_id_to_mass = nb.typed.Dict.empty(
        key_type=nb.core.types.int64,
        value_type=nb.core.types.float64,
    )
    # As far as I can tell, we can't fill the dict directly on initialization, so we have to loop over entires.
    for pdg_id in all_particle_IDs:
        pdg_id_to_mass[pdg_id] = _pdg_id_to_mass(pdg_id)
    # It's important that we snapshot it!
    return _determine_mass_from_PDG(arrays=arrays, builder=ak.ArrayBuilder(), pdg_id_to_mass=pdg_id_to_mass).snapshot()


def build_PID_selection_mask(arrays: ak.Array,
                             absolute_pids: Optional[Sequence[int]] = None,
                             single_pids: Optional[Sequence[int]] = None,
                             particle_ID_column_name: str = "particle_ID") -> ak.Array:
    """Build selection for particles based on particle ID.

    Args:
        arrays: Particle collection, over which we will build the mask.
        absolute_pids: PIDs which are selected using their absolute values.
        single_pids: PIDs which are selected only for the given values.
        particle_ID_column_name: Name of the particle ID column. Default: "particle_ID"

    Returns:
        Mask selecting the requested PIDs.
    """
    # Validation
    if absolute_pids is None:
        absolute_pids = []
    if single_pids is None:
        single_pids = []

    # Since it's unclear how to do an `in` test for a list in a vectorized way, and we
    # aren't likely to have _so_ many PIDS, we'll just take the hit and iterate over them.
    pid_cuts = [
        np.abs(arrays[particle_ID_column_name]) == pid for pid in absolute_pids
    ]
    pid_cuts.extend([
        arrays[particle_ID_column_name] == pid for pid in single_pids
    ])
    # To combine them, we want an OR here - if the particle is selected by any of the PIDs,
    # we want to keep it
    return functools.reduce(operator.or_, pid_cuts)
