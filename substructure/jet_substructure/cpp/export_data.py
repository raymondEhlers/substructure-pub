""" Convert data for export

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path

from pachyderm import binned_data

from jet_substructure.analysis import unfolding_base
from jet_substructure.base import helpers


def convert_measured_data_to_TGraphs(
    identifier: str, grooming_method: str, hist: binned_data.BinnedData, kt_range: helpers.KtRange
) -> None:
    # Setup
    output_dir = Path("output") / "export"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select the range
    hist = unfolding_base.select_hist_range(hist=hist, x_range=kt_range)

    # Delay import to avoid direct dependence.
    import ROOT

    stat_errors = ROOT.TGraphAsymmErrors(len(hist.axes[0].bin_centers))
    stat_errors.SetName("stat_errors")
    for i, (bin_center, val, error) in enumerate(zip(hist.axes[0].bin_centers, hist.values, hist.errors)):
        stat_errors.SetPoint(i, bin_center, val)
        stat_errors.SetPointError(i, 0, 0, error, error)
        # Trivial cross check...
        assert stat_errors.GetErrorYlow(i) == error
        assert stat_errors.GetErrorYhigh(i) == error

    sys_errors = ROOT.TGraphAsymmErrors(len(hist.axes[0].bin_centers))
    sys_errors.SetName("sys_errors")
    for i, (bin_center, val, error_low, error_high) in enumerate(
        zip(
            hist.axes[0].bin_centers,
            hist.values,
            hist.metadata["y_systematic"]["quadrature"].low,
            hist.metadata["y_systematic"]["quadrature"].high,
        )
    ):
        sys_errors.SetPoint(i, bin_center, val)
        sys_errors.SetPointError(i, 0, 0, error_low, error_high)
        # Trivial cross check...
        assert sys_errors.GetErrorYlow(i) == error_low
        assert sys_errors.GetErrorYhigh(i) == error_high

    f_out = ROOT.TFile.Open(str(output_dir / f"{grooming_method}_{identifier}.root"), "RECREATE")
    stat_errors.Write()
    sys_errors.Write()
    f_out.Close()
