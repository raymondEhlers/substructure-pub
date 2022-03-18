"""Convert JEWEL inputs from Laura + Raghav into expected awkward array format.

This is particularly focused on JEWEL w/ recoils simulations, which as of 5 December 2021,
is stored in: `/alf/data/laura/pc069/alice/thermal_ML/jewel_stuff`

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import awkward as ak

from mammoth.framework import sources

logger = logging.getLogger(__name__)


def jewel_to_awkward(
    filename: Path,
    entry_range: Optional[Tuple[int, int]] = None,
) -> ak.Array:
    # For JEWEL, these were the only meaningful columns
    event_level_columns = {
        "mcweight": "event_weight",
    }
    particle_columns = {
        "partpT": "pt",
        # This maps the rapidity to pseudorapidity. This is wrong, strictly speaking,
        # but good enough for these purposes (especially because mapping to the rapidity in
        # the vector constructors is apparently not so trivial...)
        "party": "eta",
        "partphi": "phi",
        "partm": "m",
        "partc": "charge",
        "parts": "scattering_center"
    }

    additional_uproot_source_kwargs = {}
    if entry_range is not None:
        additional_uproot_source_kwargs = {
            "entry_range": entry_range
        }

    data = sources.UprootSource(
        filename=filename,
        tree_name="ParticleTree",
        columns=list(event_level_columns) + list(particle_columns),
        **additional_uproot_source_kwargs,  # type: ignore
    ).data()

    return ak.Array({
        "part_level": ak.zip(
            dict(
                zip(
                    list(particle_columns.values()),
                    ak.unzip(data[particle_columns]),
                )
            )
        ),
        **dict(
            zip(
                list(event_level_columns.values()),
                ak.unzip(data[event_level_columns]),
            )
        ),
    })

