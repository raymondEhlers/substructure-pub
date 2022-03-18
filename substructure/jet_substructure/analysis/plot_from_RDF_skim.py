""" Plot from RDF skim

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import IPython
import uproot3
from pachyderm import binned_data

from jet_substructure.analysis import data_frame
from jet_substructure.base import helpers, skim_analysis_objects


logger = logging.getLogger(__name__)


def hists_from_file(
    filename: Path, grooming_methods: Optional[List[str]] = None
) -> Dict[str, Dict[str, binned_data.BinnedData]]:
    if grooming_methods is None:
        grooming_methods = ["leading_kt_z_cut_02"]

    # for grooming_method in grooming_methods:
    f = uproot3.open(filename)
    temp_hists = {
        k.decode("utf-8"): binned_data.BinnedData.from_existing_data(f[k]).to_boost_histogram() for k in f.keys()
    }
    hists = {}
    # Remove the cycle, which we don't care about.
    for k, v in temp_hists.items():
        k = k[: k.find(";")]
        hists[k] = v

    return hists


def plot_PbPb_embedded_comparison() -> None:
    # Settings
    grooming_methods = [
        "leading_kt_z_cut_02",
    ]
    _matching_name_to_axis_value: Dict[str, int] = {
        "all": 0,
        "pure": 1,
        "leading_untagged_subleading_correct": 2,
        "leading_correct_subleading_untagged": 3,
        "leading_correct_subleading_mistag": 4,
        "leading_mistag_subleading_correct": 5,
        "leading_untagged_subleading_mistag": 6,
        "leading_mistag_subleading_untagged": 7,
        "swap": 8,
        "both_untagged": 9,
    }
    response_types = [
        skim_analysis_objects.ResponseType(measured_like="hybrid", generator_like="det_level"),
        skim_analysis_objects.ResponseType(measured_like="hybrid", generator_like="true"),
        skim_analysis_objects.ResponseType(measured_like="det_level", generator_like="true"),
    ]

    PbPb_hists = hists_from_file(filename=Path("output/PbPb/RDF/leading_kt_z_cut_02_data.root"))
    embedded_hists = hists_from_file(filename=Path("output/embedPythia/RDF/leading_kt_z_cut_02_data.root"))

    # Add some helpful imports and definitions
    from importlib import reload  # noqa: F401

    try:
        # May not want to import if developing.
        import jet_substructure.analysis.plot_base as pb  # noqa: F401
        from jet_substructure.analysis import plot_from_skim  # noqa: F401
    except SyntaxError:
        logger.info("Couldn't load plot_from_skim due to syntax error. You need to load it.")

    user_ns = locals()
    user_ns.update({"output_dir_f": data_frame.output_dir_f, "Path": Path})
    IPython.start_ipython(user_ns=user_ns)


if __name__ == "__main__":
    helpers.setup_logging()

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_PbPb_embedded_comparison()
