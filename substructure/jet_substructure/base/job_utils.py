"""Job utilities for running jet substructure analyses.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBNL
"""

from pathlib import Path
from typing import Dict

from pachyderm import yaml

from jet_substructure.base import skim_analysis_objects


def read_extracted_scale_factors(
    path: Path,
) -> Dict[int, float]:
    """Read extracted scale factors.

    Args:
        collision_system: Name of the collision system.
        dataset_name: Name of the dataset.

    Returns:
        Normalized scaled factors
    """
    y = yaml.yaml(classes_to_register=[skim_analysis_objects.ScaleFactor])
    with open(path, "r") as f:
        scale_factors: Dict[int, skim_analysis_objects.ScaleFactor] = y.load(f)

    return {pt_hard_bin: v.value() for pt_hard_bin, v in scale_factors.items()}