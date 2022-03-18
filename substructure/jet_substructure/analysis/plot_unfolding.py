""" Functionality related to preparing unfolding outputs and plotting.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import itertools
import logging
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import attr
import boost_histogram as bh
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
import seaborn as sns
import uproot
from pachyderm import binned_data

from jet_substructure.analysis import plot_base as pb
from jet_substructure.analysis import unfolding_base
from jet_substructure.base import helpers


logger = logging.getLogger(__name__)

pachyderm.plot.configure()


def _efficiency_substructure_variable(
    hists: Mapping[str, binned_data.BinnedData], true_jet_pt_range: helpers.RangeSelector
) -> binned_data.BinnedData:
    """Efficiency for the substructure variable.

    Note:
        Since we need a set of hists, we just pass all of them.

    Args:
        hists: Input histograms.
        true_jet_pt_range: True jet pt range over which we will integrate.
    Returns:
        Efficiency hist for the substructure variable.
    """
    # Assign them for convenience
    try:
        bh_cut_efficiency = hists["true"].to_boost_histogram()
        bh_full_efficiency = hists["truef"].to_boost_histogram()

        # Select true pt range.
        selection = slice(bh.loc(true_jet_pt_range.min), bh.loc(true_jet_pt_range.max), bh.sum)
        cut = binned_data.BinnedData.from_existing_data(bh_cut_efficiency[:, selection])
        full = binned_data.BinnedData.from_existing_data(bh_full_efficiency[:, selection])

        return cut / full

    except KeyError:
        logger.warning(
            'Hist "true" was not found. Instead, trying to extract the efficiency directly from the projection.'
        )
        # This hist already has the efficiency applied, so we can return it directly!
        return binned_data.BinnedData.from_existing_data(
            hists[f"correff{int(true_jet_pt_range.min)}-{int(true_jet_pt_range.max)}"]
        )


def _project_substructure_variable(
    input_hist: binned_data.BinnedData, jet_pt_range: helpers.RangeSelector
) -> binned_data.BinnedData:
    """Project the hist to the substructure variable.

    Args:
        input_hist: Hist to be projected.
        jet_pt_range: True jet pt range over which we will integrate.
    Returns:
        The input hist projected onto the substructure variable axis.
    """
    # For convenience
    bh_hist = input_hist.to_boost_histogram()

    selection = slice(bh.loc(jet_pt_range.min), bh.loc(jet_pt_range.max), bh.sum)
    return binned_data.BinnedData.from_existing_data(bh_hist[:, selection])


def _efficiency_pt(
    hists: Mapping[str, binned_data.BinnedData], true_substructure_variable_range: helpers.RangeSelector
) -> binned_data.BinnedData:
    """Efficiency for the jet pt.

    Note:
        Since we need a set of hists, we just pass all of them.

    Args:
        hists: Input histograms.
        true_substructure_variable_range: True substructure variable range over which we will integrate.
    Returns:
        Efficiency hist for jet pt.
    """
    # For convenience
    bh_cut_efficiency = hists["true"].to_boost_histogram()
    bh_full_efficiency = hists["truef"].to_boost_histogram()

    # Select true pt range.
    selection = slice(
        bh.loc(true_substructure_variable_range.min), bh.loc(true_substructure_variable_range.max), bh.sum
    )
    cut = binned_data.BinnedData.from_existing_data(bh_cut_efficiency[selection, :])
    full = binned_data.BinnedData.from_existing_data(bh_full_efficiency[selection, :])

    return cut / full


def _project_jet_pt(
    input_hist: binned_data.BinnedData, substructure_variable_bin: helpers.RangeSelector
) -> binned_data.BinnedData:
    """Project the hist to the jet pt.

    Args:
        input_hist: Hist to be projected.
        substructure_variable_range: True substructure variable range over which we will integrate.
    Returns:
        The input hist projected onto the the jet pt axis.
    """
    bh_hist = input_hist.to_boost_histogram()

    selection = slice(bh.loc(substructure_variable_bin.min), bh.loc(substructure_variable_bin.max), bh.sum)
    return binned_data.BinnedData.from_existing_data(bh_hist[selection, :])


def _normalize_unfolded(hist: binned_data.BinnedData, efficiency: binned_data.BinnedData) -> binned_data.BinnedData:
    """Normalized unfolded hist.

    This involves applying the efficiency and then normalizing by the integral and the bin width.

    Args:
        hist: Histogram to be normalized.
        efficiency: Efficiency histogram with the same binning as the input hist.
    Returns:
        The normalized histogram.
    """
    # Apply the efficiency.
    hist /= efficiency
    # Then normalize by the integral (sum) and bin width.
    hist /= np.sum(hist.values)
    hist /= hist.axes[0].bin_widths
    return hist


def _normalize_refolded(hist: binned_data.BinnedData) -> binned_data.BinnedData:
    """Normalize refolded hist.

    This involves normalizing by the integral and the bin width.

    Args:
        hist: Histogram to be normalized.
    Returns:
        The normalized histogram.
    """
    hist /= np.sum(hist.values)
    hist /= hist.axes[0].bin_widths
    return hist


def _smeared(
    hists: Mapping[str, binned_data.BinnedData],
    hist_name: str,
    projection_func: Callable[[binned_data.BinnedData, helpers.RangeSelector], binned_data.BinnedData],
    smeared_range_to_integrate_over: helpers.RangeSelector,
) -> binned_data.BinnedData:
    """Helper function to get a smeared hist along a desired axis.

    Args:
        hists: Input hists.
        hist_name: Name of the smeared histogram to retrieve.
        projection_func: Function to project the histogram along the desired axis.
        smeared_range_to_integrate_over: Smeared range over which we will integrate.
    Returns:
        The desired smeared histogram.
    """
    hist = projection_func(hists[hist_name], smeared_range_to_integrate_over)
    return _normalize_refolded(hist=hist)


def _unfolded(
    hists: Mapping[str, binned_data.BinnedData],
    hist_name: str,
    projection_func: Callable[[binned_data.BinnedData, helpers.RangeSelector], binned_data.BinnedData],
    efficiency_func: Callable[[Mapping[str, binned_data.BinnedData], helpers.RangeSelector], binned_data.BinnedData],
    true_range_to_integrate_over: helpers.RangeSelector,
) -> binned_data.BinnedData:
    """Helper function to get an unfolded hist along a desired axis.

    Args:
        hists: Input hists.
        hist_name: Name of the unfolded histogram to retrieve.
        projection_func: Function to project the histogram along the desired axis.
        true_range_to_integrate_over: True range over which we will integrate.
    Returns:
        The desired unfolded histogram.
    """
    # efficiency = efficiency_func(hists, true_bin)
    ## For convenience in normalizing.
    # _normalize_hist = functools.partial(_normalize_unfolded, efficiency=efficiency)
    hist = projection_func(hists[hist_name], true_range_to_integrate_over)
    # hist = _normalize_hist(hist)
    efficiency = efficiency_func(hists, true_range_to_integrate_over)
    return _normalize_unfolded(hist=hist, efficiency=efficiency)


@attr.s
class UnfoldingOutput:
    substructure_variable: str = attr.ib()
    grooming_method: str = attr.ib()
    smeared_var_range: helpers.RangeSelector = attr.ib()
    smeared_untagged_var: helpers.RangeSelector = attr.ib()
    smeared_jet_pt_range: helpers.JetPtRange = attr.ib()
    collision_system: str = attr.ib()
    base_dir: Path = attr.ib(converter=Path)
    pure_matches: bool = attr.ib(default=False)
    suffix: str = attr.ib(default="")
    label: str = attr.ib(default="")
    n_iter_compare: int = attr.ib(default=4)
    raw_hist_name: str = attr.ib(default="raw")
    smeared_hist_name: str = attr.ib(default="smeared")
    true_hist_name: str = attr.ib(default="true")
    hists: MutableMapping[str, binned_data.BinnedData] = attr.ib(factory=dict)

    def __attrs_post_init__(self) -> None:
        # Fully setup base dir.
        # NOTE: Added "parsl" for the newer output results.
        # self.base_dir = self.base_dir / self.collision_system / "unfolding" / "parsl" / "feb2021_test"
        self.base_dir = self.base_dir / self.collision_system / "unfolding" / "parsl" / "2021-04"

        # Initialize the file if the histograms aren't specified.
        if not self.hists:
            f = uproot.open(self.input_filename)
            for k in f.keys(cycle=False):
                self.hists[k] = binned_data.BinnedData.from_existing_data(f[k])

    @property
    def identifier(self) -> str:
        name = f"{self.substructure_variable}_grooming_method_{self.grooming_method}"
        name += f"_smeared_{self.smeared_var_range}"
        name += f"_untagged_{self.smeared_untagged_var}"
        name += f"_smeared_{self.smeared_jet_pt_range}"
        if self.suffix:
            name += f"_{self.suffix}"
        if self.pure_matches:
            name += "_pure_matches"
        if self.label:
            name += f"_{self.label}"
        return name

    @property
    def max_n_iter(self) -> int:
        try:
            return self._max_n_iter
        except AttributeError:
            n = 1
            for hist_name in self.hists:
                # We could equally use the unfolded.
                if "bayesian_folded_iter_" in hist_name:
                    # We add a +1 so we can use it easily with range(...).
                    n = max(n, int(hist_name.split("_")[-1]) + 1)
            self._max_n_iter: int = n
        return self._max_n_iter

    def n_iter_range_to_plot(self) -> Iterable[int]:
        """Generate the n_iter range to plot.

        This lets us cut down on the iterations to plot when it would be too much to view comfortably.
        First, we return the first n, and then take every second from there.
        """
        change_to_sparse = 8
        if self.max_n_iter > change_to_sparse:
            return itertools.chain(range(1, change_to_sparse + 1), range(change_to_sparse + 1, self.max_n_iter, 2))
        return range(1, self.max_n_iter)

    @property
    def input_filename(self) -> Path:
        return self.base_dir / f"unfolding_{self.identifier}.root"

    @property
    def output_dir(self) -> Path:
        p = self.base_dir / self.substructure_variable / self.identifier
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def output_dir_png(self) -> Path:
        return self.output_dir / "png"

    @property
    def disabled_untagged_bin(self) -> bool:
        """If the untagged bin min and max are the same, the untagged bin was disabled."""
        return self.smeared_untagged_var.min == self.smeared_untagged_var.max

    def unfolded_substructure(self, n_iter: int, true_jet_pt_range: helpers.JetPtRange) -> binned_data.BinnedData:
        """ Helper to retrieve the unfolded substructure directly """
        return self.true_substructure(
            hist_name=f"bayesian_unfolded_iter_{n_iter}",
            true_jet_pt_range=true_jet_pt_range,
        )

    def true_substructure(self, hist_name: str, true_jet_pt_range: helpers.JetPtRange) -> binned_data.BinnedData:
        """ Retrieve a true level substructure hist. """
        return _unfolded(
            hists=self.hists,
            hist_name=hist_name,
            projection_func=_project_substructure_variable,
            efficiency_func=_efficiency_substructure_variable,
            true_range_to_integrate_over=true_jet_pt_range,
        )

    def unfolded_jet_pt(
        self, n_iter: int, true_substructure_variable_range: helpers.RangeSelector
    ) -> binned_data.BinnedData:
        return self.true_jet_pt(
            hist_name=f"bayesian_unfolded_iter_{n_iter}",
            true_substructure_variable_range=true_substructure_variable_range,
        )

    def true_jet_pt(
        self, hist_name: str, true_substructure_variable_range: helpers.RangeSelector
    ) -> binned_data.BinnedData:
        return _unfolded(
            hists=self.hists,
            hist_name=hist_name,
            projection_func=_project_jet_pt,
            efficiency_func=_efficiency_pt,
            true_range_to_integrate_over=true_substructure_variable_range,
        )

    def refolded_substructure(self, n_iter: int, smeared_jet_pt_range: helpers.JetPtRange) -> binned_data.BinnedData:
        """ Helper to retrieve the refolded substructure directly. """
        return self.smeared_substructure(
            hist_name=f"bayesian_folded_iter_{n_iter}",
            smeared_jet_pt_range=smeared_jet_pt_range,
        )

    def smeared_substructure(self, hist_name: str, smeared_jet_pt_range: helpers.JetPtRange) -> binned_data.BinnedData:
        """ Retrieve a smeared substructure hist. """
        return _smeared(
            hists=self.hists,
            hist_name=hist_name,
            projection_func=_project_substructure_variable,
            smeared_range_to_integrate_over=smeared_jet_pt_range,
        )

    def refolded_jet_pt(
        self, n_iter: int, smeared_substructure_variable_range: helpers.RangeSelector
    ) -> binned_data.BinnedData:
        """ Helper to retrieve the refolded jet pt directly. """
        return self.smeared_jet_pt(
            hist_name=f"bayesian_folded_iter_{n_iter}",
            smeared_substructure_variable_range=smeared_substructure_variable_range,
        )

    def smeared_jet_pt(
        self, hist_name: str, smeared_substructure_variable_range: helpers.RangeSelector
    ) -> binned_data.BinnedData:
        """ Retrieve a smeared jet pt hist. """
        return _smeared(
            hists=self.hists,
            hist_name=hist_name,
            projection_func=_project_jet_pt,
            smeared_range_to_integrate_over=smeared_substructure_variable_range,
        )


@attr.s
class SingleResult:
    """ Container for a single unfolding result. """

    data: binned_data.BinnedData = attr.ib()
    n_iter: int = attr.ib()
    ranges: Sequence[helpers.RangeSelector] = attr.ib(factory=list)


def plot_relative_individual_systematics(
    unfolded: SingleResult,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
) -> None:
    """Plot relative individual systematic errors."""
    import mplhep as hep

    # Setup
    logger.debug("Plotting systematic relative errors.")
    fig, ax = plt.subplots(figsize=(10, 7.5))

    for name, systematic in unfolded.data.metadata["y_systematic"].items():
        # Upper values
        extra_args = {}
        if name == "quadrature":
            extra_args = {
                "color": "black",
                "linewidth": 2,
            }
        p = hep.histplot(
            H=np.ones_like(unfolded.data.values) + (systematic.high / unfolded.data.values),
            bins=unfolded.data.axes[0].bin_edges,
            label=name.replace("_", " "),
            alpha=0.8,
            **extra_args,
        )
        # Lower values
        # Need to drop this - otherwise it will conflict with existing arguments.
        if name == "quadrature":
            extra_args.pop("color")
        hep.histplot(
            H=np.ones_like(unfolded.data.values) - (systematic.low / unfolded.data.values),
            bins=unfolded.data.axes[0].bin_edges,
            color=p[0].stairs.get_edgecolor(),
            alpha=0.8,
            **extra_args,
        )

    # For comparison, add the statistical too
    ax.errorbar(
        unfolded.data.axes[0].bin_centers,
        np.ones_like(unfolded.data.axes[0].bin_centers),
        yerr=unfolded.data.errors / unfolded.data.values,
        # color=style.color,
        marker="o",
        linestyle="",
        label="Statistical",
        # alpha=0.8,
    )

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)

    figure_name = f"{plot_config.name}"
    logger.info(f"Writing plot to {output_dir / figure_name}.pdf")
    fig.savefig(output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")

    plt.close(fig)


def _load_analytical_calculations(filename: Path, bin_edges: np.ndarray) -> binned_data.BinnedData:
    """Load analytical calcuations for a given jet R, as determined by the filename."""
    # May not be terribly efficient, but it works automatically and it's a small amount of data, so it's good enough.
    arr = np.loadtxt(filename)
    central_values = arr[:, 0]
    lower_bounds = arr[:, 1]
    upper_bounds = arr[:, 2]

    h = binned_data.BinnedData(
        axes=bin_edges,
        values=central_values,
        variances=np.zeros(len(central_values)),
    )
    h.metadata["y_systematic"] = {
        # The asymmetric errors are expected to be differences
        "quadrature": unfolding_base.AsymmetricErrors(
            low=central_values-lower_bounds,
            high=upper_bounds-central_values,
        )
    }
    return h


def load_analytical_calculations(
    path_to_calculations: Path, bin_edges: Dict[str, np.ndarray]
) -> Dict[str, Dict[str, binned_data.BinnedData]]:
    """Load analytical calcuations for a collection of jet R, as determined by the bin edges dict."""
    _grooming_methods_to_files = {
        "dynamical_kt": "1",
        "dynamical_time": "2",
    }
    output: Dict[str, Dict[str, binned_data.BinnedData]] = {}

    for jet_R_str, edges in bin_edges.items():
        output[jet_R_str] = {}
        for grooming_method, label in _grooming_methods_to_files.items():
            output[jet_R_str][grooming_method] = _load_analytical_calculations(
                filename=path_to_calculations / jet_R_str / f"ktg_a{label}.dat", bin_edges=edges
            )
    return output


def load_jetscape_data(filename: Path) -> Dict[str, Dict[str, binned_data.BinnedData]]:
    """Load jetscape predictions for all jet R.

    Include DyG core, kt, and time, as well as SD z > 0.2.
    """
    _dyg_values = {
        "005": "dynamical_core",
        "010": "dynamical_kt",
        "020": "dynamical_time",
    }
    output: Dict[str, Dict[str, binned_data.BinnedData]] = {}
    with uproot.open(filename) as f:
        for jet_R in ["02", "04", "05"]:
            output[f"R{jet_R}"] = {}
            for val, grooming_method in _dyg_values.items():
                output[f"R{jet_R}"][grooming_method] = binned_data.BinnedData.from_existing_data(
                    f[f"h_chjet_ktg_dyg_a_{val}_alice_R{jet_R}_pt0.0Scaled"]
                )
            # Soft Drop
            output[f"R{jet_R}"]["soft_drop_z_cut_02"] = binned_data.BinnedData.from_existing_data(
                f[f"h_chjet_ktg_soft_drop_z_cut_02_alice_R{jet_R}_pt0.0Scaled"]
            )

    return output


def calculate_jetscape_ratio(
    pp: Dict[str, Dict[str, binned_data.BinnedData]], PbPb: Dict[str, Dict[str, binned_data.BinnedData]]
) -> Dict[str, Dict[str, binned_data.BinnedData]]:
    """ Calculate jetscape predictions from the pp and PbPb kt spectra. """
    output: Dict[str, Dict[str, binned_data.BinnedData]] = {}
    for jet_R, pp_R in pp.items():
        output[jet_R] = {}
        # Retrieve by hand just in case they're not in the same order...
        PbPb_R = PbPb[jet_R]
        for grooming_method, pp_hist in pp_R.items():
            # Retrieve by hand just in case they're not in the same order...
            PbPb_hist = PbPb_R[grooming_method]
            ratio = PbPb_hist / pp_hist
            # Then normalize
            ratio /= np.sum(ratio.values)
            ratio /= ratio.axes[0].bin_widths
            output[jet_R][grooming_method] = ratio

    return output


def load_sherpa_predictions(
    filename: Path, jet_R_values: Union[float, Sequence[float]]
) -> Dict[str, Dict[str, binned_data.BinnedData]]:
    """Load sherpa predictions for a given jet R.

    Include DyG core, kt, and time, as well as SD z > 0.2.
    """
    if isinstance(jet_R_values, float):
        jet_R_values = [jet_R_values]
    _name_map = {
        "k0": "dynamical_core",
        "k1": "dynamical_kt",
        "k2": "dynamical_time",
        "ksd": "soft_drop_z_cut_02",
    }
    output: Dict[str, Dict[str, binned_data.BinnedData]] = {}
    with uproot.open(filename) as f:
        for jet_R in jet_R_values:
            jet_R_str = f"R{round(jet_R * 10):02}"
            output[jet_R_str] = {}
            for tag, grooming_method in _name_map.items():
                output[jet_R_str][grooming_method] = binned_data.BinnedData.from_existing_data(f[f"histo{tag}"])

    return output


#_model_palette = sns.color_palette("husl", n_colors=6)
#_model_palette = sns.color_palette("Accent", n_colors=10)
#_model_palette = sns.color_palette("colorblind", n_colors=10)
#_model_palette = sns.color_palette("dark", n_colors=6)

_model_palette = [
    (53, 73, 222),
    (170, 52, 222),
    (223, 82, 87),
    (225, 220, 103),
    (90, 224, 102),
    (57, 225, 215),
    (137, 185, 224),
    (89, 147, 223),
    (223, 34, 219),
    (216, 142, 224),
    (223, 124, 53),
    (108, 223, 41),
]
_model_palette = [
    (color[0] / 254, color[1] / 254, color[2] / 254)
    for color in _model_palette
]
_model_palette = _model_palette[1:] + [_model_palette[0]]

_models_styles = {
    "pythia": dict(
        label="PYTHIA8 Monash 2013",
        linewidth=3,
        linestyle="-",
        marker="s",
        #color=_model_palette[0],
        color=_model_palette[7],
        markerfacecolor="none",
        #markerfacecolor="white",
        markeredgewidth=3,
    ),
    "analytical": dict(
        label="Caucal et al.",
        linewidth=3,
        linestyle="-.",
        marker="P",
        #color=_model_palette[1],
        #color=_model_palette[5],
        #color=_model_palette[8],
        #color=_model_palette[4],
        color=_model_palette[3],
    ),
    "sherpa_lund": dict(
        label="SHERPA (Lund)",
        # NOTE: This will overlap with jetscape, but we currently (8 July 2021) can't compare them, so it's fine.
        #       To be resolved when the plotting plans are a bit clearer.
        linewidth=3,
        linestyle="--",
        marker="*",
        #color=_model_palette[2],
        color=_model_palette[1],
        #color=_model_palette[5],
        #color=_model_palette[2],
    ),
    "sherpa_ahadic": dict(
        label="SHERPA (AHADIC)",
        linewidth=3,
        linestyle=":",
        marker="X",
        #color=_model_palette[3],
        #color=_model_palette[7],
        #color=_model_palette[6],
        #color=_model_palette[3],
        color=_model_palette[6],
    ),
    "jetscape": dict(
        label="JETSCAPE PP19",
        linewidth=3,
        linestyle="--",
        marker="D",
        #color=_model_palette[4],
        #color=_model_palette[8],
        #color=_model_palette[4],
        color=_model_palette[3],
    ),
}


def _plot_data_model_comparison_for_single_system(
    hists: Mapping[str, SingleResult],
    models: Mapping[str, Mapping[str, binned_data.BinnedData]],
    grooming_methods: Sequence[str],
    set_zero_to_nan: bool,
    kt_range: Mapping[str, helpers.KtRange],
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    grooming_styling = pb.define_grooming_styles()

    with sns.color_palette("Set2"):
        # fig, ax = plt.subplots(figsize=(9, 10))
        # Size is specified to make it convenient to compare against Hard Probes plots.
        fig, (ax, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=(10, 10),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        for grooming_method in grooming_methods:
            plotting_last_method = grooming_method == grooming_methods[-1]

            # First, the data
            h = hists[grooming_method].data

            # Select range to display.
            h = unfolding_base.select_hist_range(h, kt_range[grooming_method])

            # Set 0s to NaN
            if set_zero_to_nan:
                h.errors[h.values == 0] = np.nan
                h.values[h.values == 0] = np.nan

            # Main data points
            p = ax.errorbar(
                h.axes[0].bin_centers,
                h.values,
                yerr=h.errors,
                xerr=h.axes[0].bin_widths / 2,
                marker="o",
                markersize=11,
                linestyle="",
                linewidth=3,
                label=grooming_styling[grooming_method].label,
            )

            # Systematic uncertainty
            pachyderm.plot.error_boxes(
                ax=ax,
                x_data=h.axes[0].bin_centers,
                y_data=h.values,
                x_errors=h.axes[0].bin_widths / 2,
                y_errors=np.array(
                    [
                        h.metadata["y_systematic"]["quadrature"].low,
                        h.metadata["y_systematic"]["quadrature"].high,
                    ]
                ),
                # y_errors=np.array([y_systematic_errors.low, y_systematic_errors.high]),
                # color=style.color,
                color=p[0].get_color(),
                linewidth=0,
            )

            for model_name, model_with_all_grooming_methods in models.items():
                model = model_with_all_grooming_methods.get(grooming_method, None)
                if not model:
                    logger.debug(
                        f"Skipping model {model_name}, grooming method: {grooming_method} because predictions aren't available"
                    )
                    continue

                # Then, plot the model
                model_style = grooming_styling[f"{grooming_method}_compare"]
                # Get the model for the reference.
                model = binned_data.BinnedData.from_existing_data(model)
                # TODO: Careful, pythia is already normalized, but jetscape wasn't. So we need to resolve this...
                #       Probably best to have some kind of "prepare model" function, which we can decide to use or not.
                # Then normalize
                model /= np.sum(model.values)
                model /= model.axes[0].bin_widths
                # And select the same range.
                # model = unfolding_base.select_hist_range(model, kt_range[grooming_method])

                # And plot
                # Make sure we copy the settings so we can modify them
                temp_kwargs = dict(_models_styles[model_name])
                temp_kwargs["label"] = temp_kwargs["label"] if plotting_last_method else None
                ax.errorbar(
                    model.axes[0].bin_centers,
                    model.values,
                    # yerr=model.errors,
                    # xerr=model.axes[0].bin_widths / 2,
                    color=grooming_styling[grooming_method].color,
                    # marker=style.marker,
                    # fillstyle=grooming_styling[grooming_method].fillstyle,
                    # linestyle="",
                    # label=_models_styles[model_name]["label"] if plotting_last_method else None,
                    zorder=model_style.zorder,
                    alpha=0.7,
                    **temp_kwargs,
                )

                # Ratio
                # Could move down here if you want to see the entire range
                model = unfolding_base.select_hist_range(model, kt_range[grooming_method])
                ratio = model / h

                # Ratio + statistical error bars
                ax_ratio.errorbar(
                    ratio.axes[0].bin_centers,
                    ratio.values,
                    yerr=ratio.errors,
                    xerr=ratio.axes[0].bin_widths / 2,
                    color=p[0].get_color(),
                    marker="o",
                    markersize=11,
                    linestyle="",
                    linewidth=3,
                )
                # Systematic errors.
                y_relative_error_low = unfolding_base.relative_error(
                    unfolding_base.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].low),
                )
                y_relative_error_high = unfolding_base.relative_error(
                    unfolding_base.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].high),
                )
                # From error prop, pythia has no systematic error, so we just convert the relative errors.
                ratio.metadata["y_systematic"] = {}
                ratio.metadata["y_systematic"]["quadrature"] = unfolding_base.AsymmetricErrors(
                    low=y_relative_error_low * ratio.values,
                    high=y_relative_error_high * ratio.values,
                )
                y_systematic = ratio.metadata["y_systematic"]["quadrature"]
                pachyderm.plot.error_boxes(
                    ax=ax_ratio,
                    x_data=ratio.axes[0].bin_centers,
                    y_data=ratio.values,
                    x_errors=ratio.axes[0].bin_widths / 2,
                    y_errors=np.array([y_systematic.low, y_systematic.high]),
                    color=p[0].get_color(),
                    linewidth=0,
                )

        # reference value for ratio
        ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Labeling and presentation
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])
    # A few additional tweaks.
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.2))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_grooming_model_comparisons_for_single_system(
    hists: Mapping[str, SingleResult],
    models: Mapping[str, Mapping[str, binned_data.BinnedData]],
    grooming_methods: Sequence[str],
    collision_system: str,
    collision_system_key: str,
    output_dir: Path,
    kt_range: Union[helpers.KtRange, Mapping[str, helpers.KtRange]],
    figure_kt_range: helpers.KtRange = helpers.KtRange(1.5, 15),
    jet_R_str: str = "R04",
) -> None:
    """Plot comparison of grooming methods for a single system."""

    # Validation
    if isinstance(kt_range, helpers.KtRange):
        kt_range = {grooming_method: kt_range for grooming_method in grooming_methods}

    # grooming_styling = pb.define_grooming_styles()
    jet_pt_bin = next(iter(hists.values())).ranges[0]

    text = pb.label_to_display_string["ALICE"]["work_in_progress"]
    text += "\n" + pb.label_to_display_string["collision_system"][collision_system_key]
    text += "\n" + pb.label_to_display_string["jets"]["general"]
    text += "\n" + pb.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
    _plot_data_model_comparison_for_single_system(
        hists=hists,
        models=models,
        grooming_methods=grooming_methods,
        set_zero_to_nan=False,
        kt_range=kt_range,
        plot_config=pb.PlotConfig(
            name=f"unfolded_kt_{collision_system}_model_comparison_{jet_R_str}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                            log=True,
                            range=(5e-3, 1),
                            font_size=22,
                        ),
                    ],
                    text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                    legend=pb.LegendConfig(location="lower left", font_size=22),
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=tuple(figure_kt_range), font_size=22),  # type: ignore
                        pb.AxisConfig(
                            "y",
                            label=r"$\frac{\text{Model}}{\text{Data}}$",
                            range=(0.45, 1.55),
                            font_size=22,
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.08)),
        ),
        output_dir=output_dir,
    )


def _plot_single_system_comparison(
    hists: Mapping[str, SingleResult],
    grooming_methods: Sequence[str],
    reference_grooming_method: str,
    set_zero_to_nan: bool,
    kt_range: Mapping[str, helpers.KtRange],
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    grooming_styling = pb.define_grooming_styles()

    with sns.color_palette("Set2"):
        # fig, ax = plt.subplots(figsize=(9, 10))
        # Size is specified to make it convenient to compare against Hard Probes plots.
        fig, (ax, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=(10, 10),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        # Use selected grooming method as a reference, but only in the range where the others are measured.
        ratio_reference_hist_unselected = hists[reference_grooming_method].data

        for grooming_method in grooming_methods:
            # Axes: jet_pt, attr_name
            h_input = hists[grooming_method].data

            # Select range to display.
            h = unfolding_base.select_hist_range(h_input, kt_range[grooming_method])

            # Set 0s to NaN
            if set_zero_to_nan:
                h.errors[h.values == 0] = np.nan
                h.values[h.values == 0] = np.nan

            # Main data points
            p = ax.errorbar(
                h.axes[0].bin_centers,
                h.values,
                yerr=h.errors,
                xerr=h.axes[0].bin_widths / 2,
                marker="o",
                markersize=11,
                linestyle="",
                linewidth=3,
                label=grooming_styling[grooming_method].label,
            )

            # Systematic uncertainty
            pachyderm.plot.error_boxes(
                ax=ax,
                x_data=h.axes[0].bin_centers,
                y_data=h.values,
                x_errors=h.axes[0].bin_widths / 2,
                y_errors=np.array(
                    [
                        h.metadata["y_systematic"]["quadrature"].low,
                        h.metadata["y_systematic"]["quadrature"].high,
                    ]
                ),
                # y_errors=np.array([y_systematic_errors.low, y_systematic_errors.high]),
                # color=style.color,
                color=p[0].get_color(),
                linewidth=0,
            )

            # Ratio
            # Skip pp because it's not meaningful.
            if grooming_method == reference_grooming_method:
                continue

            # Ensure the ratio is defined over the same range.
            # TODO: Refactor when more awake...
            kt_range_for_current_grooming_method = kt_range[grooming_method]
            kt_range_for_reference = kt_range[reference_grooming_method]
            kt_range_min, kt_range_max = tuple(kt_range_for_current_grooming_method)  # type: ignore
            if kt_range_min < kt_range_for_reference.min:
                kt_range_min = kt_range_for_reference.min
            if kt_range_max > kt_range_for_reference.max:
                kt_range_max = kt_range_for_reference.max
            kt_range_for_comparison = helpers.KtRange(kt_range_min, kt_range_max)
            logger.info(f"kt_range_for_comparison: {kt_range_for_comparison}")
            ratio_reference_hist = unfolding_base.select_hist_range(
                ratio_reference_hist_unselected,
                kt_range_for_comparison,
            )
            h = unfolding_base.select_hist_range(
                h_input,
                kt_range_for_comparison,
            )
            ratio = h / ratio_reference_hist
            # Ratio + statistical error bars
            ax_ratio.errorbar(
                ratio.axes[0].bin_centers,
                ratio.values,
                yerr=ratio.errors,
                xerr=ratio.axes[0].bin_widths / 2,
                color=p[0].get_color(),
                marker="o",
                markersize=11,
                linestyle="",
                linewidth=3,
            )
            # Systematic errors.
            y_relative_error_low = unfolding_base.relative_error(
                unfolding_base.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].low),
                unfolding_base.ErrorInput(
                    value=ratio_reference_hist.values,
                    error=ratio_reference_hist.metadata["y_systematic"]["quadrature"].low,
                ),
            )
            y_relative_error_high = unfolding_base.relative_error(
                unfolding_base.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].high),
                unfolding_base.ErrorInput(
                    value=ratio_reference_hist.values,
                    error=ratio_reference_hist.metadata["y_systematic"]["quadrature"].high,
                ),
            )
            # Sanity check
            # TODO: If this passes once, delete it. I've checked this a lot now...
            test_relative_y_error_low = np.sqrt(
                (h.metadata["y_systematic"]["quadrature"].low / h.values) ** 2
                + (ratio_reference_hist.metadata["y_systematic"]["quadrature"].low / ratio_reference_hist.values) ** 2
            )
            test_relative_y_error_high = np.sqrt(
                (h.metadata["y_systematic"]["quadrature"].high / h.values) ** 2
                + (ratio_reference_hist.metadata["y_systematic"]["quadrature"].high / ratio_reference_hist.values) ** 2
            )
            np.testing.assert_allclose(y_relative_error_low, test_relative_y_error_low)
            np.testing.assert_allclose(y_relative_error_high, test_relative_y_error_high)
            # Store the systematic.
            ratio.metadata["y_systematic"]["quadrature"] = unfolding_base.AsymmetricErrors(
                low=y_relative_error_low * ratio.values,
                high=y_relative_error_high * ratio.values,
            )
            y_systematic = ratio.metadata["y_systematic"]["quadrature"]
            pachyderm.plot.error_boxes(
                ax=ax_ratio,
                x_data=ratio.axes[0].bin_centers,
                y_data=ratio.values,
                x_errors=ratio.axes[0].bin_widths / 2,
                y_errors=np.array([y_systematic.low, y_systematic.high]),
                color=p[0].get_color(),
                linewidth=0,
            )

    # Labeling and presentation
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])
    # A few additional tweaks.
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.2))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_grooming_comparisons_for_single_system(
    hists: Mapping[str, SingleResult],
    grooming_methods: Sequence[str],
    reference_grooming_method: str,
    collision_system: str,
    collision_system_key: str,
    output_dir: Path,
    kt_range: Union[helpers.KtRange, Mapping[str, helpers.KtRange]],
    figure_kt_range: helpers.KtRange = helpers.KtRange(1.5, 15),
    jet_R_str: str = "R04",
) -> None:
    """Plot comparison of grooming methods for a single system."""

    # Validation
    if isinstance(kt_range, helpers.KtRange):
        kt_range = {grooming_method: kt_range for grooming_method in grooming_methods}

    grooming_styling = pb.define_grooming_styles()
    jet_pt_bin = next(iter(hists.values())).ranges[0]

    text = pb.label_to_display_string["ALICE"]["work_in_progress"]
    text += "\n" + pb.label_to_display_string["collision_system"][collision_system_key]
    text += "\n" + pb.label_to_display_string["jets"]["general"]
    text += "\n" + pb.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
    _plot_single_system_comparison(
        hists=hists,
        grooming_methods=grooming_methods,
        reference_grooming_method=reference_grooming_method,
        set_zero_to_nan=False,
        kt_range=kt_range,
        plot_config=pb.PlotConfig(
            name=f"unfolded_kt_{collision_system}_comparison_{jet_R_str}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                            log=True,
                            range=(5e-3, 1),
                            font_size=22,
                        ),
                    ],
                    text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                    legend=pb.LegendConfig(location="lower left", font_size=22),
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=tuple(figure_kt_range), font_size=22),  # type: ignore
                        pb.AxisConfig(
                            "y",
                            label=r"$\frac{\text{Method}}{\text{"
                            + grooming_styling[reference_grooming_method].label
                            + "}}$",
                            range=(0.45, 1.55),
                            font_size=22,
                        ),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.08)),
        ),
        output_dir=output_dir,
    )


def _plot_pp_PbPb_comparison(
    hists: Mapping[str, SingleResult],
    grooming_method: str,
    set_zero_to_nan: bool,
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    """Plot PbPb with systematics compared to pp with systematics for a set of grooming methods."""
    logger.info("Plotting grooming method comparison for kt with systematics")

    # Setup
    event_activity_label_map = {
        "pp": "pp",
        "central": r"0-10\% $\text{Pb--Pb}$",
        "semi_central": r"30-50\% $\text{Pb--Pb}$",
    }
    # NOTE: Probably should make this configurable at some point.
    # Based on kinematic eff and unfolding ranges
    event_activity_to_range = {
        # TEMP: Make this configurable...
        "pp": helpers.KtRange(0.25, 6),
        # "semi_central": helpers.KtRange(0.25, 6),
        # "pp": helpers.KtRange(0.5, 6),
        "semi_central": helpers.KtRange(2, 6),
        "central": helpers.KtRange(3, 6),
    }

    with sns.color_palette("Set2"):
        # fig, ax = plt.subplots(figsize=(9, 10))
        # Size is specified to make it convenient to compare against Hard Probes plots.
        fig, (ax, ax_ratio) = plt.subplots(
            2,
            1,
            figsize=(10, 10),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )

        # Use pp as reference, but only in the range where the others are measured.
        ratio_reference_hist_unselected = hists["pp"].data

        # Collision system is a bit misleading because it's really just a high label, but good enough for a quick look.
        for collision_system, hist in hists.items():
            # Axes: jet_pt, attr_name
            h = hist.data

            # Select range to display.
            h = unfolding_base.select_hist_range(h, event_activity_to_range[collision_system])

            # Set 0s to NaN
            if set_zero_to_nan:
                h.errors[h.values == 0] = np.nan
                h.values[h.values == 0] = np.nan

            # Main data points
            p = ax.errorbar(
                h.axes[0].bin_centers,
                h.values,
                yerr=h.errors,
                xerr=h.axes[0].bin_widths / 2,
                marker="o",
                markersize=11,
                linestyle="",
                linewidth=3,
                label=event_activity_label_map[collision_system],
            )

            # Systematic uncertainty
            pachyderm.plot.error_boxes(
                ax=ax,
                x_data=h.axes[0].bin_centers,
                y_data=h.values,
                x_errors=h.axes[0].bin_widths / 2,
                y_errors=np.array(
                    [
                        h.metadata["y_systematic"]["quadrature"].low,
                        h.metadata["y_systematic"]["quadrature"].high,
                    ]
                ),
                # y_errors=np.array([y_systematic_errors.low, y_systematic_errors.high]),
                # color=style.color,
                color=p[0].get_color(),
                linewidth=0,
            )

            # Ratio
            # Skip pp because it's not meaningful.
            if collision_system == "pp":
                continue

            # Ensure the ratio is defined over the same range.
            ratio_reference_hist = unfolding_base.select_hist_range(
                ratio_reference_hist_unselected, event_activity_to_range[collision_system]
            )
            logger.debug(f"h: {h.axes[0].bin_edges}")
            logger.debug(f"ratio_reference_hist: {ratio_reference_hist.axes[0].bin_edges}")
            ratio = h / ratio_reference_hist
            # Ratio + statistical error bars
            ax_ratio.errorbar(
                ratio.axes[0].bin_centers,
                ratio.values,
                yerr=ratio.errors,
                xerr=ratio.axes[0].bin_widths / 2,
                color=p[0].get_color(),
                marker="o",
                markersize=11,
                linestyle="",
                linewidth=3,
            )
            # Systematic errors.
            y_relative_error_low = unfolding_base.relative_error(
                unfolding_base.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].low),
                unfolding_base.ErrorInput(
                    value=ratio_reference_hist.values,
                    error=ratio_reference_hist.metadata["y_systematic"]["quadrature"].low,
                ),
            )
            y_relative_error_high = unfolding_base.relative_error(
                unfolding_base.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].high),
                unfolding_base.ErrorInput(
                    value=ratio_reference_hist.values,
                    error=ratio_reference_hist.metadata["y_systematic"]["quadrature"].high,
                ),
            )
            # Sanity check
            # TODO: If this passes once, delete it. I've checked this a lot now...
            test_relative_y_error_low = np.sqrt(
                (h.metadata["y_systematic"]["quadrature"].low / h.values) ** 2
                + (ratio_reference_hist.metadata["y_systematic"]["quadrature"].low / ratio_reference_hist.values) ** 2
            )
            test_relative_y_error_high = np.sqrt(
                (h.metadata["y_systematic"]["quadrature"].high / h.values) ** 2
                + (ratio_reference_hist.metadata["y_systematic"]["quadrature"].high / ratio_reference_hist.values) ** 2
            )
            np.testing.assert_allclose(y_relative_error_low, test_relative_y_error_low)
            np.testing.assert_allclose(y_relative_error_high, test_relative_y_error_high)
            # Store the systematic.
            ratio.metadata["y_systematic"]["quadrature"] = unfolding_base.AsymmetricErrors(
                low=y_relative_error_low * ratio.values,
                high=y_relative_error_high * ratio.values,
            )
            y_systematic = ratio.metadata["y_systematic"]["quadrature"]
            pachyderm.plot.error_boxes(
                ax=ax_ratio,
                x_data=ratio.axes[0].bin_centers,
                y_data=ratio.values,
                x_errors=ratio.axes[0].bin_widths / 2,
                y_errors=np.array([y_systematic.low, y_systematic.high]),
                color=p[0].get_color(),
                linewidth=0,
            )

    # Labeling and presentation
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])
    # A few additional tweaks.
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.2))

    filename = f"{plot_config.name}_{grooming_method}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_pp_PbPb_comparison(
    hists: Mapping[str, SingleResult],
    grooming_method: str,
    output_dir: Path,
    kt_range: Tuple[float, float] = (1.5, 15),
    jet_R_str: str = "R04",
) -> None:
    """Plot PbPb unfolded results with systematics."""
    jet_pt_bin = next(iter(hists.values())).ranges[0]
    grooming_styling = pb.define_grooming_styles()
    style = grooming_styling[grooming_method]

    text = pb.label_to_display_string["ALICE"]["work_in_progress"]
    text += "\n" + pb.label_to_display_string["collision_system"]["pp_PbPb_5TeV"]
    text += "\n" + pb.label_to_display_string["jets"]["general"]
    text += "\n" + pb.label_to_display_string["jets"][jet_R_str]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
    text += "\n" + fr"{style.label}"
    _plot_pp_PbPb_comparison(
        hists=hists,
        grooming_method=grooming_method,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"unfolded_kt_pp_PbPb_comparison_{jet_R_str}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                            log=True,
                            range=(5e-3, 1),
                            font_size=22,
                        ),
                    ],
                    text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                    legend=pb.LegendConfig(location="lower left", font_size=22),
                ),
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=kt_range, font_size=22),
                        pb.AxisConfig("y", label=r"$\frac{\text{Pb-Pb}}{\text{pp}}$", range=(0.45, 1.55), font_size=22),
                    ],
                ),
            ],
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.08)),
        ),
        output_dir=output_dir,
    )


def _plot_simple_kt_with_systematics(
    hists: Mapping[str, SingleResult],
    grooming_methods: Sequence[str],
    set_zero_to_nan: bool,
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    """Plot PbPb with systematics for a set of grooming methods."""
    logger.info("Plotting grooming method comparison for kt with systematics")

    # fig, ax = plt.subplots(figsize=(9, 10))
    # Size is specified to make it convenient to compare against Hard Probes plots.
    fig, ax = plt.subplots(figsize=(8, 4.5))

    grooming_styling = pb.define_grooming_styles()

    for grooming_method in grooming_methods:
        # Setup
        style = grooming_styling[grooming_method]

        # Axes: jet_pt, attr_name
        h = hists[grooming_method].data

        # Set 0s to NaN (for example, in z_g where have a good portion of the range cut off).
        if set_zero_to_nan:
            h.errors[h.values == 0] = np.nan
            h.values[h.values == 0] = np.nan

        # Plot options
        kwargs = {
            "markerfacecolor": "white" if style.fillstyle == "none" else style.color,
            "alpha": 1 if style.fillstyle == "none" else 0.8,
        }
        if style.fillstyle != "none":
            kwargs["markeredgewidth"] = 0

        # Main data points
        ax.errorbar(
            h.axes[0].bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axes[0].bin_widths / 2,
            color=style.color,
            marker=style.marker,
            fillstyle=style.fillstyle,
            linestyle="",
            label=style.label,
            zorder=style.zorder,
            **kwargs,
        )

        # Systematic uncertainty
        pachyderm.plot.error_boxes(
            ax=ax,
            x_data=h.axes[0].bin_centers,
            y_data=h.values,
            x_errors=h.axes[0].bin_widths / 2,
            y_errors=np.array(
                [
                    h.metadata["y_systematic"]["quadrature"].low,
                    h.metadata["y_systematic"]["quadrature"].high,
                ]
            ),
            # y_errors=np.array([y_systematic_errors.low, y_systematic_errors.high]),
            # color=style.color,
            # color=p[0].get_color(),
            color=style.color,
            linewidth=0,
        )

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)
    # A few additional tweaks.
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=1.0))
    # ax_ratio.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=0.2))

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_PbPb_systematics_simple(
    hists: Mapping[str, SingleResult],
    grooming_methods: Sequence[str],
    event_activity: str,
    output_dir: Path,
    kt_range: Tuple[float, float] = (1.5, 15),
    jet_R: str = "R04",
) -> None:
    """Plot PbPb unfolded results with systematics."""
    jet_pt_bin = hists[grooming_methods[0]].ranges[0]
    event_activity_map = {
        "central": r"0-10\%",
        "semi_central": r"30-50\%",
    }

    text = pb.label_to_display_string["ALICE"]["work_in_progress"]
    if event_activity != "pp":
        text += (
            "\n" + pb.label_to_display_string["collision_system"]["PbPb"] + f", {event_activity_map[event_activity]}"
        )
    else:
        text += "\n" + pb.label_to_display_string["collision_system"]["pp_5TeV"]
    text += "\n" + pb.label_to_display_string["jets"]["general"]
    text += "\n" + pb.label_to_display_string["jets"][jet_R]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
    _plot_simple_kt_with_systematics(
        hists=hists,
        grooming_methods=grooming_methods,
        set_zero_to_nan=False,
        plot_config=pb.PlotConfig(
            name=f"unfolded_kt_systematics_simple_{event_activity}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=kt_range),
                        pb.AxisConfig(
                            "y",
                            label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                            log=True,
                            range=(7e-3, 1),
                        ),
                    ],
                    text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                    legend=pb.LegendConfig(location="lower left", font_size=22),
                ),
            ],
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.06)),
        ),
        output_dir=output_dir,
    )


def _plot_compare_kt_with_systematics(
    hists: Mapping[str, SingleResult],
    reference: Mapping[str, binned_data.BinnedData],
    grooming_methods: Sequence[str],
    set_zero_to_nan: bool,
    plot_config: pb.PlotConfig,
    output_dir: Path,
) -> None:
    """Plot PbPb with systematics for a set of grooming methods."""
    logger.info("Plotting grooming method comparison for kt with systematics")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(9, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    ax, ax_ratio = axes

    grooming_styling = pb.define_grooming_styles()

    for grooming_method in grooming_methods:
        # Setup
        style = grooming_styling[grooming_method]

        # Axes: jet_pt, attr_name
        h = hists[grooming_method].data

        # Set 0s to NaN (for example, in z_g where have a good portion of the range cut off).
        if set_zero_to_nan:
            h.errors[h.values == 0] = np.nan
            h.values[h.values == 0] = np.nan

        # Plot options
        kwargs = {
            "markerfacecolor": "white" if style.fillstyle == "none" else style.color,
            "alpha": 1 if style.fillstyle == "none" else 0.8,
        }
        if style.fillstyle != "none":
            kwargs["markeredgewidth"] = 0

        # Main data points
        ax.errorbar(
            h.axes[0].bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axes[0].bin_widths / 2,
            color=style.color,
            marker=style.marker,
            fillstyle=style.fillstyle,
            linestyle="",
            label=style.label,
            zorder=style.zorder,
            **kwargs,
        )

        # Systematic uncertainty
        pachyderm.plot.error_boxes(
            ax=ax,
            x_data=h.axes[0].bin_centers,
            y_data=h.values,
            x_errors=h.axes[0].bin_widths / 2,
            y_errors=np.array(
                [
                    h.metadata["y_systematic"]["quadrature"].low,
                    h.metadata["y_systematic"]["quadrature"].high,
                ]
            ),
            # y_errors=np.array([y_systematic_errors.low, y_systematic_errors.high]),
            # color=style.color,
            # color=p[0].get_color(),
            color=style.color,
            linewidth=0,
        )

        # Ratio + statistical error bars from unfolding
        ratio = h / reference[grooming_method]
        ax_ratio.errorbar(
            ratio.axes[0].bin_centers,
            ratio.values,
            yerr=ratio.errors,
            xerr=ratio.axes[0].bin_widths / 2,
            color=style.color,
            marker=style.marker,
            fillstyle=style.fillstyle,
            linestyle="",
            zorder=style.zorder,
            **kwargs,
        )
        # Systematic errors.
        y_relative_error_low = unfolding_base.relative_error(
            unfolding_base.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].low)
        )
        y_relative_error_high = unfolding_base.relative_error(
            unfolding_base.ErrorInput(value=h.values, error=h.metadata["y_systematic"]["quadrature"].high)
        )
        # From error prop, pythia has no systematic error, so we just convert the relative errors.
        ratio.metadata["y_systematic"] = unfolding_base.AsymmetricErrors(
            low=y_relative_error_low * ratio.values,
            high=y_relative_error_high * ratio.values,
        )
        pachyderm.plot.error_boxes(
            ax=ax_ratio,
            x_data=ratio.axes[0].bin_centers,
            y_data=ratio.values,
            x_errors=ratio.axes[0].bin_widths / 2,
            y_errors=np.array([ratio.metadata["y_systematic"].low, ratio.metadata["y_systematic"].high]),
            color=style.color,
            linewidth=0,
            # label = "Background", color = plot_base.AnalysisColors.fit,
        )

    # Reference value for ratio
    ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Labeling and presentation
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])

    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def plot_PbPb_systematics(
    hists: Mapping[str, SingleResult],
    reference: Mapping[str, binned_data.BinnedData],
    grooming_methods: Sequence[str],
    event_activity: str,
    output_dir: Path,
    kt_range: Tuple[float, float] = (1.5, 15),
    jet_R: str = "R04",
) -> None:
    """Plot PbPb unfolded results with systematics."""
    jet_pt_bin = hists[grooming_methods[0]].ranges[0]
    event_activity_map = {
        "central": r"0-10\%",
        "semi_central": r"30-50\%",
    }

    text = pb.label_to_display_string["ALICE"]["work_in_progress"]
    if event_activity != "pp":
        text += (
            "\n" + pb.label_to_display_string["collision_system"]["PbPb"] + f", {event_activity_map[event_activity]}"
        )
    else:
        text += "\n" + pb.label_to_display_string["collision_system"]["pp_5TeV"]
    text += "\n" + pb.label_to_display_string["jets"]["general"]
    text += "\n" + pb.label_to_display_string["jets"][jet_R]
    text += "\n" + fr"${jet_pt_bin.display_str(label='')}\:\text{{GeV}}/c$"
    _plot_compare_kt_with_systematics(
        hists=hists,
        grooming_methods=grooming_methods,
        set_zero_to_nan=False,
        reference=reference,
        plot_config=pb.PlotConfig(
            name=f"unfolded_kt_systematics_{event_activity}",
            panels=[
                # Main panel
                pb.Panel(
                    axes=[
                        pb.AxisConfig(
                            "y",
                            label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                            log=True,
                            range=(1e-3, 0.3),
                        ),
                    ],
                    text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                    legend=pb.LegendConfig(location="lower left", font_size=22),
                ),
                # Ratio
                pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=kt_range),
                        pb.AxisConfig(
                            "y",
                            label=r"$\frac{\text{data}}{\text{PYTHIA}}$",
                            range=(0.55, 1.45),
                        ),
                    ]
                ),
            ],
            figure=pb.Figure(edge_padding=dict(left=0.12, bottom=0.06)),
        ),
        output_dir=output_dir,
    )


def setup_unfolding_closures(
    substructure_variable: str,
    grooming_method: str,
    smeared_var_range: helpers.KtRange,
    smeared_untagged_var: helpers.KtRange,
    smeared_jet_pt_range: helpers.JetPtRange,
    collision_system: str,
    n_iter_compare: int,
    suffix: str,
    output_dir: Path,
    pure_matches: bool = False,
) -> Dict[str, UnfoldingOutput]:
    # Setup the input files
    unfolding_outputs = {}
    unfolding_outputs["default"] = UnfoldingOutput(
        substructure_variable=substructure_variable,
        grooming_method=grooming_method,
        smeared_var_range=smeared_var_range,
        smeared_untagged_var=smeared_untagged_var,
        smeared_jet_pt_range=smeared_jet_pt_range,
        collision_system=collision_system,
        base_dir=output_dir,
        n_iter_compare=n_iter_compare,
        pure_matches=pure_matches,
        suffix=suffix,
    )

    # These should always exist.
    unfolding_outputs["trivial_closure"] = UnfoldingOutput(
        substructure_variable=substructure_variable,
        grooming_method=grooming_method,
        smeared_var_range=smeared_var_range,
        smeared_untagged_var=smeared_untagged_var,
        smeared_jet_pt_range=smeared_jet_pt_range,
        collision_system=collision_system,
        base_dir=output_dir,
        n_iter_compare=n_iter_compare,
        pure_matches=pure_matches,
        suffix=suffix,
        label="closure_trivial_hybrid_smeared_as_input",
        raw_hist_name="smeared",
    )

    unfolding_outputs["closure_later_iter"] = UnfoldingOutput(
        substructure_variable=substructure_variable,
        grooming_method=grooming_method,
        smeared_var_range=smeared_var_range,
        smeared_untagged_var=smeared_untagged_var,
        smeared_jet_pt_range=smeared_jet_pt_range,
        collision_system=collision_system,
        base_dir=output_dir,
        n_iter_compare=n_iter_compare,
        pure_matches=pure_matches,
        suffix=suffix,
        label="closure_5_iter_5",
    )

    try:
        unfolding_outputs["split_MC"] = UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            pure_matches=pure_matches,
            suffix=suffix,
            label="closure_split_MC",
            raw_hist_name="h2_pseudo_data",
            true_hist_name="h2_pseudo_true",
        )
    except FileNotFoundError:
        logger.debug("Skipping split MC because the output file doesn't exist.")

    try:
        unfolding_outputs["reweight_pseudo_data"] = UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            pure_matches=pure_matches,
            suffix=suffix,
            label="closure_reweight_pseudo_data",
            raw_hist_name="h2_pseudo_data",
            true_hist_name="h2_pseudo_true",
        )
    except FileNotFoundError:
        logger.debug("Skipping reweighted pseudo data because the output file doesn't exist.")

    try:
        unfolding_outputs["reweight_response"] = UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            pure_matches=pure_matches,
            suffix=suffix,
            label="closure_reweight_response",
            raw_hist_name="h2_pseudo_data",
            true_hist_name="h2_pseudo_true",
        )
    except FileNotFoundError:
        logger.debug("Skipping reweighted response because the output file doesn't exist.")

    return unfolding_outputs


def setup_unfolding_outputs(  # noqa: C901
    substructure_variable: str,
    grooming_method: str,
    smeared_var_range: helpers.KtRange,
    smeared_untagged_var: helpers.KtRange,
    smeared_jet_pt_range: helpers.JetPtRange,
    collision_system: str,
    n_iter_compare: int,
    suffix: str,
    output_dir: Path,
    truncation_shift: float = 5,
    displaced_untagged_above_range: bool = True,
    displaced_extremum: Optional[float] = None,
    skip_reweighted_prior_in_systematics: bool = False,
) -> Dict[str, UnfoldingOutput]:
    # Validation
    # Keep the truncation positive so we know how we've shifted.
    if truncation_shift < 0:
        truncation_shift = np.abs(truncation_shift)
    if displaced_extremum is None:
        # NOTE: We set 20 externally (in the unfolding configuration in parsl). But it should work fine
        #       because it encompasses all possible PbPb ranges used so far.
        displaced_extremum = 20

    # Setup the input files
    unfolding_outputs = {}
    unfolding_outputs["default"] = UnfoldingOutput(
        substructure_variable=substructure_variable,
        grooming_method=grooming_method,
        smeared_var_range=smeared_var_range,
        smeared_untagged_var=smeared_untagged_var,
        smeared_jet_pt_range=smeared_jet_pt_range,
        collision_system=collision_system,
        base_dir=output_dir,
        n_iter_compare=n_iter_compare,
        pure_matches=False,
        suffix=suffix,
    )
    logger.info(f"default: {unfolding_outputs['default'].identifier}")

    try:
        unfolding_outputs["tracking_efficiency"] = UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            pure_matches=False,
            suffix=suffix,
            label="tracking_efficiency",
        )
    except FileNotFoundError:
        logger.debug("Skipping tracking efficiency because the output file doesn't exist.")

    try:
        unfolding_outputs["truncation_low"] = UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=helpers.JetPtRange(
                smeared_jet_pt_range.min - truncation_shift, smeared_jet_pt_range.max
            ),
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            pure_matches=False,
            suffix=suffix,
            label="truncation",
        )
        unfolding_outputs["truncation_high"] = UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=helpers.JetPtRange(
                smeared_jet_pt_range.min + truncation_shift, smeared_jet_pt_range.max
            ),
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            pure_matches=False,
            suffix=suffix,
            label="truncation",
        )
    except FileNotFoundError:
        logger.debug("Skipping truncation because the output file doesn't exist.")

    try:
        unfolding_outputs["random_binning"] = UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            pure_matches=False,
            suffix=suffix,
            label="random_binning",
        )
    except FileNotFoundError:
        logger.debug("Skipping random binning because the output file doesn't exist.")

    try:
        # If the untagged bin is disabled, then skip this
        if not smeared_untagged_var.min == smeared_untagged_var.max:
            if displaced_untagged_above_range:
                displaced_untagged_var = helpers.KtRange(smeared_var_range.max, displaced_extremum)
            else:
                displaced_untagged_var = helpers.KtRange(displaced_extremum, smeared_var_range.min)

            unfolding_outputs["untagged_bin"] = UnfoldingOutput(
                substructure_variable=substructure_variable,
                grooming_method=grooming_method,
                smeared_var_range=smeared_var_range,
                smeared_untagged_var=displaced_untagged_var,
                smeared_jet_pt_range=smeared_jet_pt_range,
                collision_system=collision_system,
                base_dir=output_dir,
                n_iter_compare=n_iter_compare,
                pure_matches=False,
                suffix=suffix,
            )
            logger.debug(f"untagged_bin: {unfolding_outputs['untagged_bin'].identifier}")
        else:
            logger.info("Skipping untagged bin outputs because it is disabled")

    except FileNotFoundError:
        logger.debug("Skipping untagged bin location because the output file doesn't exist.")

    if not skip_reweighted_prior_in_systematics:
        try:
            unfolding_outputs["reweight_prior"] = UnfoldingOutput(
                substructure_variable=substructure_variable,
                grooming_method=grooming_method,
                smeared_var_range=smeared_var_range,
                smeared_untagged_var=smeared_untagged_var,
                smeared_jet_pt_range=smeared_jet_pt_range,
                collision_system=collision_system,
                base_dir=output_dir,
                n_iter_compare=n_iter_compare,
                pure_matches=False,
                suffix=suffix,
                label="reweight_prior",
            )
        except FileNotFoundError:
            logger.debug("Skipping reweighted prior because the output file doesn't exist.")
    else:
        logger.debug(
            "Skipping reweighted prior because it was requested (probably for pp, where we take a model dependence instead)."
        )

    # Model dependence
    try:
        # Careful here: the outputs in pp are not in the standard format. But this is a convenient fiction.
        unfolding_outputs["model_dependence"] = UnfoldingOutput(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            base_dir=output_dir,
            n_iter_compare=n_iter_compare,
            pure_matches=False,
            suffix=suffix,
            label="model_dependence",
        )
    except FileNotFoundError:
        logger.debug("Skipping model dependence because the output file doesn't exist.")

    # Background subtraction
    for background_setting in ["Rmax060", "Rmax005"]:
        try:
            unfolding_outputs[background_setting] = UnfoldingOutput(
                substructure_variable=substructure_variable,
                grooming_method=grooming_method,
                smeared_var_range=smeared_var_range,
                smeared_untagged_var=smeared_untagged_var,
                smeared_jet_pt_range=smeared_jet_pt_range,
                collision_system=collision_system,
                base_dir=output_dir,
                n_iter_compare=n_iter_compare,
                pure_matches=False,
                suffix=suffix,
                label=f"{background_setting}",
            )
        except FileNotFoundError:
            logger.debug(f"Skipping background setting {background_setting} because the output file doesn't exist.")

    return unfolding_outputs


def _load_unfolded_outputs(
    grooming_method: str,
    substructure_variable: str,
    smeared_var_range: helpers.KtRange,
    smeared_untagged_var: helpers.KtRange,
    smeared_jet_pt_range: helpers.JetPtRange,
    collision_system: str,
    event_activity: str,
    jet_R_str: str,
    n_iter_compare: int,
    truncation_shift: int,
    displaced_extremum: float,
    output_dir: Path,
    tag_after_suffix: str = "",
    displaced_untagged_above_range: bool = True,
    skip_reweighted_prior_in_systematics: bool = False,
) -> Tuple[Dict[str, UnfoldingOutput], Dict[str, UnfoldingOutput], Dict[str, UnfoldingOutput]]:
    # Validation
    suffix = f"{event_activity}_{jet_R_str}"
    if tag_after_suffix:
        suffix += f"_{tag_after_suffix}"

    unfolding_closure_outputs = setup_unfolding_closures(
        substructure_variable=substructure_variable,
        grooming_method=grooming_method,
        smeared_var_range=smeared_var_range,
        smeared_untagged_var=smeared_untagged_var,
        smeared_jet_pt_range=smeared_jet_pt_range,
        collision_system=collision_system,
        n_iter_compare=n_iter_compare,
        suffix=suffix,
        output_dir=output_dir,
    )
    try:
        unfolding_closure_pure_matches_outputs = setup_unfolding_closures(
            substructure_variable=substructure_variable,
            grooming_method=grooming_method,
            smeared_var_range=smeared_var_range,
            smeared_untagged_var=smeared_untagged_var,
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            n_iter_compare=n_iter_compare,
            suffix=suffix,
            output_dir=output_dir,
            pure_matches=True,
        )
    except KeyError as e:
        logger.warning(f"Could not find pure matches output '{e}'. Skipping")
        unfolding_closure_pure_matches_outputs = {}

    unfolding_systematics_outputs = setup_unfolding_outputs(
        substructure_variable=substructure_variable,
        grooming_method=grooming_method,
        smeared_var_range=smeared_var_range,
        smeared_untagged_var=smeared_untagged_var,
        smeared_jet_pt_range=smeared_jet_pt_range,
        collision_system=collision_system,
        n_iter_compare=n_iter_compare,
        suffix=suffix,
        output_dir=output_dir,
        truncation_shift=truncation_shift,
        displaced_untagged_above_range=displaced_untagged_above_range,
        displaced_extremum=displaced_extremum,
        skip_reweighted_prior_in_systematics=skip_reweighted_prior_in_systematics,
    )

    return unfolding_closure_outputs, unfolding_closure_pure_matches_outputs, unfolding_systematics_outputs


def load_unfolded_outputs(
    grooming_methods: Sequence[str],
    substructure_variable: str,
    smeared_var_range: Union[helpers.KtRange, Mapping[str, helpers.KtRange]],
    smeared_untagged_var: Union[helpers.KtRange, Mapping[str, helpers.KtRange]],
    smeared_jet_pt_range: helpers.JetPtRange,
    collision_system: str,
    event_activity: str,
    jet_R_str: str,
    n_iter_compare: Union[int, Mapping[str, int]],
    truncation_shift: int,
    displaced_extremum: float,
    output_dir: Path,
    tag_after_suffix: Union[str, Mapping[str, str]] = "",
    displaced_untagged_above_range: bool = True,
    skip_reweighted_prior_in_systematics: bool = False,
) -> Tuple[
    Dict[str, Dict[str, UnfoldingOutput]], Dict[str, Dict[str, UnfoldingOutput]], Dict[str, Dict[str, UnfoldingOutput]]
]:
    # Validation
    if isinstance(smeared_var_range, helpers.KtRange):
        # Copy for every grooming method
        smeared_var_range = {grooming_method: smeared_var_range for grooming_method in grooming_methods}
    if isinstance(smeared_untagged_var, helpers.KtRange):
        # Copy for every grooming method
        smeared_untagged_var = {grooming_method: smeared_untagged_var for grooming_method in grooming_methods}
    if isinstance(n_iter_compare, int):
        # Copy for every grooming method
        n_iter_compare = {grooming_method: n_iter_compare for grooming_method in grooming_methods}
    if isinstance(tag_after_suffix, str):
        # Copy for every grooming method
        tag_after_suffix = {grooming_method: tag_after_suffix for grooming_method in grooming_methods}
    unfolding_closure_outputs = {}
    unfolding_closure_pure_matches_outputs = {}
    unfolding_systematics_outputs = {}
    for grooming_method in grooming_methods:
        (
            unfolding_closure_outputs[grooming_method],
            unfolding_closure_pure_matches_outputs[grooming_method],
            unfolding_systematics_outputs[grooming_method],
        ) = _load_unfolded_outputs(
            grooming_method=grooming_method,
            substructure_variable=substructure_variable,
            smeared_var_range=smeared_var_range[grooming_method],
            smeared_untagged_var=smeared_untagged_var[grooming_method],
            smeared_jet_pt_range=smeared_jet_pt_range,
            collision_system=collision_system,
            event_activity=event_activity,
            jet_R_str=jet_R_str,
            n_iter_compare=n_iter_compare[grooming_method],
            truncation_shift=truncation_shift,
            displaced_extremum=displaced_extremum,
            output_dir=output_dir,
            tag_after_suffix=tag_after_suffix[grooming_method],
            displaced_untagged_above_range=displaced_untagged_above_range,
            skip_reweighted_prior_in_systematics=skip_reweighted_prior_in_systematics,
        )

    return (
        unfolding_closure_outputs,
        unfolding_closure_pure_matches_outputs,
        unfolding_systematics_outputs,
    )


def _unfolded_outputs_with_systematics(
    grooming_method: str,
    unfolding_systematics_outputs: Dict[str, Dict[str, UnfoldingOutput]],
    true_jet_pt_range: helpers.JetPtRange,
) -> Tuple[SingleResult, binned_data.BinnedData]:
    logger.info(f"Calculating systematics for {grooming_method}")
    unfolded = unfolded_substructure_results(
        unfolding_outputs=unfolding_systematics_outputs[grooming_method],
        true_jet_pt_range=true_jet_pt_range,
    )

    unfolded_with_systematics = calculate_systematics(
        unfolded=unfolded,
        unfolding_outputs=unfolding_systematics_outputs[grooming_method],
        true_jet_pt_range=true_jet_pt_range,
    )

    true_reference = unfolding_systematics_outputs[grooming_method]["default"].true_substructure(
        unfolding_systematics_outputs[grooming_method]["default"].true_hist_name, true_jet_pt_range=true_jet_pt_range
    )

    return unfolded_with_systematics, true_reference


def unfolded_outputs_with_systematics(
    grooming_methods: Sequence[str],
    unfolding_systematics_outputs: Dict[str, Dict[str, UnfoldingOutput]],
    true_jet_pt_range: helpers.JetPtRange,
) -> Tuple[Dict[str, SingleResult], Dict[str, binned_data.BinnedData]]:
    unfolded_with_systematics = {}
    true_reference = {}
    for grooming_method in grooming_methods:
        (
            unfolded_with_systematics[grooming_method],
            true_reference[grooming_method],
        ) = _unfolded_outputs_with_systematics(
            grooming_method=grooming_method,
            unfolding_systematics_outputs=unfolding_systematics_outputs,
            true_jet_pt_range=true_jet_pt_range,
        )

    return unfolded_with_systematics, true_reference


def unfolded_substructure_results(
    unfolding_outputs: Mapping[str, UnfoldingOutput], true_jet_pt_range: helpers.JetPtRange
) -> Dict[str, SingleResult]:
    """Convert unfolded results into individual unfolded substructure results (selecting a particular iteration).

    This is useful for working with substructure systematics.

    Note:
        We always select the n iter from the default unfolded result.

    Args:
        unfolding_output: All unfolded outputs.
        true_jet_pt_range: True jet pt range for the substructure result.
    Returns:
        Unfolded substructure results.
    """
    unfolded = {}
    for k, v in unfolding_outputs.items():
        if k == "model_dependence":
            # We have to handle this manually. See the systematics calculation.
            continue
        unfolded[k] = SingleResult(
            # NOTE: We want to match the iter of the default case.
            data=v.unfolded_substructure(
                n_iter=unfolding_outputs["default"].n_iter_compare,
                true_jet_pt_range=true_jet_pt_range,
            ),
            n_iter=unfolding_outputs["default"].n_iter_compare,
            ranges=[true_jet_pt_range],
        )
    return unfolded


def calculate_systematics(  # noqa: C901
    unfolded: Mapping[str, SingleResult],
    unfolding_outputs: Mapping[str, UnfoldingOutput],
    true_jet_pt_range: helpers.JetPtRange,
    truncation_iter: Optional[helpers.RangeSelector] = None,
) -> SingleResult:
    # Validation
    if truncation_iter is None:
        truncation_iter = helpers.RangeSelector(1, 1)
    if truncation_iter.min < 0:
        truncation_iter = helpers.RangeSelector(-1 * truncation_iter.min, truncation_iter.max)
    # Setup
    unfolded["default"].data.metadata["y_systematic"] = {}

    # Tracking efficiency
    # This is treated as a symmetric uncertainty.
    # However, we store it as asymmetric errors objects for consistency with everything else.
    try:
        # NOTE: Unlike the others, we take the abs and set the values here directly because
        #       we want them to be symmetric.
        tracking_efficiency_sym = np.abs(unfolded["tracking_efficiency"].data.values - unfolded["default"].data.values)
        unfolded["default"].data.metadata["y_systematic"]["tracking_efficiency"] = unfolding_base.AsymmetricErrors(
            tracking_efficiency_sym, tracking_efficiency_sym
        )
    except KeyError as e:
        logger.debug(f"Skipping tracking efficiency because of {e}")

    # Everything else is treated asymmetrically, potentially one-sided.
    # Truncation
    try:
        unfolded["default"].data.metadata["y_systematic"][
            "truncation"
        ] = unfolding_base.AsymmetricErrors.calculate_errors(
            unfolded["truncation_low"].data.values - unfolded["default"].data.values,
            unfolded["truncation_high"].data.values - unfolded["default"].data.values,
        )
    except KeyError as e:
        logger.debug(f"Skipping truncation because of {e}")

    # Regularization
    # +/- iterations
    unfolded["default"].data.metadata["y_systematic"][
        "regularization"
    ] = unfolding_base.AsymmetricErrors.calculate_errors(
        unfolded["default"].data.values
        - unfolding_outputs["default"]
        .unfolded_substructure(
            n_iter=unfolding_outputs["default"].n_iter_compare - truncation_iter.min,  # type: ignore
            true_jet_pt_range=true_jet_pt_range,
        )
        .values,
        unfolded["default"].data.values
        - unfolding_outputs["default"]
        .unfolded_substructure(
            n_iter=unfolding_outputs["default"].n_iter_compare + truncation_iter.max,  # type: ignore
            true_jet_pt_range=true_jet_pt_range,
        )
        .values,
    )

    # Random binning
    try:
        unfolded["default"].data.metadata["y_systematic"][
            "random_binning"
        ] = unfolding_base.AsymmetricErrors.calculate_errors(
            unfolded["random_binning"].data.values - unfolded["default"].data.values
        )
    except KeyError as e:
        logger.debug(f"Skipping random binning because of {e}")

    # Untagged bin location
    try:
        unfolded["default"].data.metadata["y_systematic"][
            "untagged_bin"
        ] = unfolding_base.AsymmetricErrors.calculate_errors(
            unfolded["untagged_bin"].data.values - unfolded["default"].data.values
        )
    except KeyError as e:
        logger.debug(f"Skipping untagged bin location because of {e}")

    # Reweight prior
    try:
        unfolded["default"].data.metadata["y_systematic"][
            "reweight_prior"
        ] = unfolding_base.AsymmetricErrors.calculate_errors(
            unfolded["reweight_prior"].data.values - unfolded["default"].data.values
        )
    except KeyError as e:
        logger.debug(f"Skipping reweighting prior because of {e}")

    # Background subtraction systematics.
    background_systematics = {}
    for background_setting in ["Rmax060", "Rmax005"]:
        try:
            background_systematics[background_setting] = (
                unfolded[background_setting].data.values - unfolded["default"].data.values
            )
        except KeyError as e:
            logger.debug(f"Skipping background systematic {background_setting} because of {e!r}")

    if len(background_systematics) > 0:
        first_background_sub = next(iter(background_systematics.values()))
        unfolded["default"].data.metadata["y_systematic"][
            "background_sub"
        ] = unfolding_base.AsymmetricErrors.calculate_errors(
            background_systematics.get("RMax005", first_background_sub),
            background_systematics.get("RMax060", first_background_sub),
        )
    else:
        logger.debug("Skipping background subtraction systematic because no values are available")

    # Non-closure
    # This is treated as a symmetric uncertainty.
    # However, we store it as asymmetric errors objects for consistency with everything else.
    try:
        # NOTE: Unlike the others, we take the abs and set the values here directly because
        #       we want them to be symmetric.
        # NOTE: The reference needs to be to the PseudoTrue, so we need to retrieve it here.
        # NOTE: We calculate this as a relative error because the scales could be (quite) different.
        #       We then scale the default values by this relative error to determine the non-closure.
        pseudo_true = unfolding_outputs["non_closure"].true_substructure(
            unfolding_outputs["non_closure"].true_hist_name, true_jet_pt_range=true_jet_pt_range
        )
        logger.info(f"true name: {unfolding_outputs['non_closure'].true_hist_name}")
        non_closure_sym_relative = np.abs(unfolded["non_closure"].data.values - pseudo_true.values) / pseudo_true.values
        logger.info(f"non_closure values: {unfolded['non_closure'].data.values}")
        logger.info(f"pseudo true: {pseudo_true.values}")
        # non_closure_sym = (1 - non_closure_sym_relative) * unfolded["default"].data.values
        logger.info(f"non_closure_sym_relative: {non_closure_sym_relative}")
        logger.info(f"non_closure bin edges: {unfolded['non_closure'].data.axes[0].bin_edges}")
        logger.info(f"pseudo_true bin edges: {pseudo_true.axes[0].bin_edges}")

        unfolded["default"].data.metadata["y_systematic"]["non_closure"] = unfolding_base.AsymmetricErrors(
            non_closure_sym_relative * unfolded["default"].data.values,
            non_closure_sym_relative * unfolded["default"].data.values,
        )
    except KeyError as e:
        logger.debug(f"Skipping non closure systematic because of {e}")

    # Model dependence.
    # The output should include _either_ the model dependence or the non-closure
    if "model_dependence" in unfolding_outputs:
        # First, extract the model dependence graph
        # NOTE: This output is quite different, so we just need to handle the graph (not hist!) directly.
        graph = unfolding_outputs["model_dependence"].hists[
            f'bayesian_unfolded_iter_{unfolding_outputs["model_dependence"].n_iter_compare}'
        ]

        # Then use the information
        relative_errors_on_model_dependence_low = graph.metadata["y_errors"]["low"] / graph.values
        relative_errors_on_model_dependence_high = graph.metadata["y_errors"]["high"] / graph.values

        logger.info(
            f"\nmodel_dependence bin_edges: {graph.axes[0].bin_edges}"
            f"\nnominal bin_edges: {unfolded['default'].data.axes[0].bin_edges}"
        )
        unfolded["default"].data.metadata["y_systematic"]["model_dependence"] = unfolding_base.AsymmetricErrors(
            relative_errors_on_model_dependence_low * unfolded["default"].data.values,
            relative_errors_on_model_dependence_high * unfolded["default"].data.values,
        )
        logger.info(
            f"\n\tlow: {relative_errors_on_model_dependence_low}"
            f"\n\thigh: {relative_errors_on_model_dependence_high}"
            f'\n\tmodel_dependence errors: {unfolded["default"].data.metadata["y_systematic"]["model_dependence"]}'
        )

    # Cross check to make sure that I haven't copied and pasted incorrectly.
    assert not any(
        [
            np.allclose(a.low, b.low)
            for k_a, a in unfolded["default"].data.metadata["y_systematic"].items()
            for k_b, b in unfolded["default"].data.metadata["y_systematic"].items()
            if k_a != k_b
        ]
    )
    assert not any(
        [
            np.allclose(a.high, b.high)
            for k_a, a in unfolded["default"].data.metadata["y_systematic"].items()
            for k_b, b in unfolded["default"].data.metadata["y_systematic"].items()
            if k_a != k_b
        ]
    )

    # Sum in quadrature
    # We protect against including quadrature in case we already calculated the systematics.
    unfolded["default"].data.metadata["y_systematic"]["quadrature"] = unfolding_base.AsymmetricErrors(
        low=np.sqrt(
            np.sum(
                [v.low ** 2 for k, v in unfolded["default"].data.metadata["y_systematic"].items() if k != "quadrature"],
                axis=0,
            )
        ),
        high=np.sqrt(
            np.sum(
                [
                    v.high ** 2
                    for k, v in unfolded["default"].data.metadata["y_systematic"].items()
                    if k != "quadrature"
                ],
                axis=0,
            )
        ),
    )

    # We could already retrieve this from the input, but return it for convenience.
    return unfolded["default"]


def plot_unfolded(
    unfolding_output: UnfoldingOutput,
    hist_true: binned_data.BinnedData,
    hist_n_iter_compare: binned_data.BinnedData,
    unfolded_hists: Mapping[int, binned_data.BinnedData],
    plot_config: pb.PlotConfig,
    plot_png: bool = False,
) -> None:
    """Plot unfolded."""
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")
    # Setup
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 12),
        gridspec_kw={"height_ratios": [4, 1, 1]},
        sharex=True,
    )
    ax_upper, ax_ratio_iter, ax_ratio_true = axes

    for i, hist in unfolded_hists.items():
        ax_upper.errorbar(
            hist.axes[0].bin_centers,
            hist.values,
            xerr=hist.axes[0].bin_widths / 2,
            yerr=hist.errors,
            label=f"Bayes {i}",
            marker="o",
            linestyle="",
            alpha=0.8,
        )

        # Plot ratio with selected iter (in principle could also be with true, but now it's
        # not necessary because we have another panel with the true).
        ratio = hist / hist_n_iter_compare
        ax_ratio_iter.errorbar(
            ratio.axes[0].bin_centers,
            ratio.values,
            xerr=ratio.axes[0].bin_widths / 2,
            yerr=ratio.errors,
            marker="o",
            linestyle="",
            alpha=0.8,
        )

        # Plot ratio with true
        ratio_true = hist / hist_true
        ax_ratio_true.errorbar(
            ratio_true.axes[0].bin_centers,
            ratio_true.values,
            xerr=ratio_true.axes[0].bin_widths / 2,
            yerr=ratio_true.errors,
            marker="o",
            linestyle="",
            alpha=0.8,
        )

    # Cross check.
    # Plot truth
    ax_upper.errorbar(
        hist_true.axes[0].bin_centers,
        hist_true.values,
        xerr=hist_true.axes[0].bin_widths / 2,
        yerr=hist_true.errors,
        label="True",
        marker="o",
        linestyle="",
        color="black",
        alpha=0.8,
    )
    ## And the ratio too
    # ratio = hist_true / h_ratio_denominator
    # ax_lower.errorbar(
    #    ratio.axes[0].bin_centers,
    #    ratio.values,
    #    xerr=ratio.axes[0].bin_widths / 2,
    #    yerr=ratio.errors,
    #    marker="o",
    #    linestyle="",
    #    color="black",
    #    alpha=0.8,
    # )

    # Plot truth and compare to the full efficient truth.
    ## Compare to the full efficiency to make sure that have the right shape...
    # full_eff_true = projection_func(hists["truef"], true_bin)
    ## Then normalize by the integral (sum) and bin width.
    ## Don't need to correct for the kinematic efficiency here because it's already fully efficient.
    # full_eff_true /= np.sum(full_eff_true.values)
    # full_eff_true /= full_eff_true.axes[0].bin_widths
    # ax_upper.errorbar(full_eff_true.axes[0].bin_centers, full_eff_true.values, xerr=full_eff_true.axes[0].bin_widths / 2, yerr=full_eff_true.errors, label = "True fully eff",
    #                  marker="o", linestyle="", alpha=0.8)
    ## Add ratio...
    # ratio = hist_true / full_eff_true
    # ax_lower.errorbar(
    #    ratio.axes[0].bin_centers,
    #    ratio.values,
    #    xerr=ratio.axes[0].bin_widths / 2,
    #    yerr=ratio.errors,
    #    marker="o",
    #    linestyle="",
    #    alpha=0.8,
    #    color="black",
    # )

    # Draw reference line for ratio
    ax_ratio_iter.axhline(y=1, color="black", linestyle="dashed", zorder=1)
    ax_ratio_true.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Label and layout
    # First, tweak the label for the ratio
    true_hist_name_to_ratio_label = {
        "true": "true",
        "h2_pseudo_true": "pseudo true",
    }
    plot_config.panels[2].axes[1].label = (
        plot_config.panels[2]
        .axes[1]
        .label.format(true_label=true_hist_name_to_ratio_label[unfolding_output.true_hist_name])
    )
    plot_config.apply(fig=fig, axes=[ax_upper, ax_ratio_iter, ax_ratio_true])

    figure_name = f"{plot_config.name}"
    logger.info(f"Writing plot to {unfolding_output.output_dir / figure_name}.pdf")
    fig.savefig(unfolding_output.output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = unfolding_output.output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")

    plt.close(fig)


def plot_refolded(
    unfolding_output: UnfoldingOutput,
    hist_raw: binned_data.BinnedData,
    hist_smeared: binned_data.BinnedData,
    refolded_hists: Mapping[int, binned_data.BinnedData],
    plot_config: pb.PlotConfig,
    plot_png: bool = False,
) -> None:
    """Plot refolded."""
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")
    # Setup
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 10),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    ax_upper, ax_lower = axes

    # Raw
    # Only plot if there's something meaningful to plot
    if hist_raw.values.any():
        ax_upper.errorbar(
            hist_raw.axes[0].bin_centers,
            hist_raw.values,
            xerr=hist_raw.axes[0].bin_widths / 2,
            yerr=hist_raw.errors,
            label="Raw",
            marker="o",
            linestyle="",
            color="red",
        )

    # Smeared
    ax_upper.errorbar(
        hist_smeared.axes[0].bin_centers,
        hist_smeared.values,
        xerr=hist_smeared.axes[0].bin_widths / 2,
        yerr=hist_smeared.errors,
        label="Smeared",
        marker="o",
        linestyle="",
        color="green",
    )

    raw_is_smeared = unfolding_output.raw_hist_name == "smeared"
    ratio_denominator = hist_smeared if raw_is_smeared else hist_raw
    for i, hist in refolded_hists.items():
        ax_upper.errorbar(
            hist.axes[0].bin_centers,
            hist.values,
            xerr=hist.axes[0].bin_widths / 2,
            yerr=hist.errors,
            label=f"Bayes {i}",
            marker="o",
            linestyle="",
            alpha=0.8,
        )

        ratio = hist / ratio_denominator
        ax_lower.errorbar(
            ratio.axes[0].bin_centers,
            ratio.values,
            xerr=ratio.axes[0].bin_widths / 2,
            yerr=ratio.errors,
            marker="o",
            linestyle="",
            alpha=0.8,
        )

    # Add smeared ratio in the right circumstances.
    if not raw_is_smeared:
        r = hist_smeared / ratio_denominator
        ax_lower.errorbar(
            r.axes[0].bin_centers,
            r.values,
            xerr=r.axes[0].bin_widths / 2,
            yerr=r.errors,
            marker="o",
            linestyle="",
            color="green",
            alpha=0.8,
        )

    # Draw reference line for ratio
    ax_lower.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Label and layout
    # First, tweak the label for the ratio
    raw_hist_name_to_ratio_label = {
        "raw": "data",
        "smeared": "smeared",
        "h2_pseudo_data": "pseudo data",
    }
    plot_config.panels[1].axes[1].label = (
        plot_config.panels[1]
        .axes[1]
        .label.format(refold_label=raw_hist_name_to_ratio_label[unfolding_output.raw_hist_name])
    )
    plot_config.apply(fig=fig, axes=[ax_upper, ax_lower])

    figure_name = f"{plot_config.name}"
    # if tag:
    #    figure_name = f"{tag}_{figure_name}"
    fig.savefig(unfolding_output.output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = unfolding_output.output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")

    plt.close(fig)


def plot_response(
    hists: Mapping[str, binned_data.BinnedData],
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
) -> None:
    # Setup
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")

    h = binned_data.BinnedData.from_existing_data(hists["h2_substructure_variable"])

    # Normalize the response.
    normalization_values = h.values.sum(axis=0, keepdims=True)
    h.values = np.divide(h.values, normalization_values, out=np.zeros_like(h.values), where=normalization_values != 0)

    # Finish setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine the normalization range
    z_axis_range = {
        # "vmin": h_proj.values[h_proj.values > 0].min(),
        "vmin": max(1e-4, h.values[h.values > 0].min()),
        # "vmax": h.values.max(),
        "vmax": 1,
    }

    # Plot
    mesh = ax.pcolormesh(
        h.axes[0].bin_edges.T,
        h.axes[1].bin_edges.T,
        h.values.T,
        norm=matplotlib.colors.LogNorm(**z_axis_range),
    )
    fig.colorbar(mesh, pad=0.02)

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)

    figure_name = f"{plot_config.name}"
    fig.savefig(output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")
    plt.close(fig)


def plot_jet_pt_vs_substructure(
    hists: Mapping[str, binned_data.BinnedData],
    hist_name: str,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
) -> None:
    # Setup
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")

    h = binned_data.BinnedData.from_existing_data(hists[hist_name])

    # Finish setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine the normalization range
    z_axis_range = {
        "vmin": h.values[h.values > 0].min(),
        "vmax": h.values.max(),
    }

    # Plot
    mesh = ax.pcolormesh(
        h.axes[0].bin_edges.T,
        h.axes[1].bin_edges.T,
        h.values.T,
        norm=matplotlib.colors.LogNorm(**z_axis_range),
    )
    fig.colorbar(mesh, pad=0.02)

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)

    figure_name = f"{plot_config.name}"
    fig.savefig(output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")
    plt.close(fig)


def plot_efficiency(
    hists: Mapping[str, binned_data.BinnedData],
    efficiency_func: Callable[[Mapping[str, binned_data.BinnedData], helpers.RangeSelector], binned_data.BinnedData],
    true_bins: Sequence[helpers.RangeSelector],
    true_bin_label: str,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
) -> None:
    """Plot kinematic efficiency."""
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")
    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))

    for true_bin in true_bins:
        # Project
        # We need the efficiency in the true bin that we actually want to measure.
        hist = efficiency_func(hists, true_bin)

        # Plot
        ax.errorbar(
            hist.axes[0].bin_centers,
            hist.values,
            xerr=hist.axes[0].bin_widths / 2,
            yerr=hist.errors,
            label=fr"${true_bin.min} < {true_bin_label}_{{\text{{T,jet}}}}^{{\text{{true}}}} < {true_bin.max}$",
            marker="o",
            linestyle="",
            alpha=0.8,
        )

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)

    fig.savefig(output_dir / f"{plot_config.name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{plot_config.name}.png")
    plt.close(fig)


def plot_select_iteration(
    unfolding_output: UnfoldingOutput,
    projection_func: Callable[[UnfoldingOutput, int, helpers.RangeSelector], binned_data.BinnedData],
    max_iter: int,
    true_bin: helpers.RangeSelector,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
    reweighted_prior_output: Optional[UnfoldingOutput] = None,
) -> None:
    """Plot selected iteration."""
    logger.debug(f"Plotting {plot_config.name.replace('_', ' ')}")
    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # -2 because we go two above, and then -2 because we start at 2
    n_bins = max_iter - 2 - 2
    hist_reg = binned_data.BinnedData(
        axes=[np.linspace(1.5, 1.5 + n_bins, n_bins + 1)],
        values=np.zeros(n_bins),
        variances=np.ones(n_bins),
    )
    hist_stat = binned_data.BinnedData(
        axes=[np.linspace(1.5, 1.5 + n_bins, n_bins + 1)],
        values=np.zeros(n_bins),
        variances=np.ones(n_bins),
    )
    hist_prior = binned_data.BinnedData(
        axes=[np.linspace(1.5, 1.5 + n_bins, n_bins + 1)],
        values=np.zeros(n_bins),
        variances=np.ones(n_bins),
    )
    hist_total = binned_data.BinnedData(
        axes=[np.linspace(1.5, 1.5 + n_bins, n_bins + 1)],
        values=np.zeros(n_bins),
        variances=np.ones(n_bins),
    )

    for i, iter in enumerate(range(2, max_iter - 2)):
        # Current iteration
        current_iter_hist = projection_func(unfolding_output, iter, true_bin)
        # Previous iter hist
        previous_iter_hist = projection_func(unfolding_output, iter - 1, true_bin)
        # Iter + 2 hist
        forward_iter_hist = projection_func(unfolding_output, iter + 2, true_bin)

        # Calculate and store regularization error
        regularization_value = np.sum(
            (
                np.divide(
                    np.maximum(
                        np.abs(previous_iter_hist.values - current_iter_hist.values),
                        np.abs(forward_iter_hist.values - current_iter_hist.values),
                    ),
                    # TEMP: Try excluding the untagged bin.
                    # / current_iter_hist.values)[1:]
                    current_iter_hist.values,
                    out=np.zeros_like(current_iter_hist.values),
                    where=current_iter_hist.values != 0,
                )
            )
        )
        hist_reg.values[i] = regularization_value
        # Calculate and store stat error
        # Skip the untagged since it tends to blow up the stat error in a way that's not meaningful
        lower_edge = None if unfolding_output.disabled_untagged_bin else 1
        stat_value = np.sum(
            np.divide(
                current_iter_hist.errors[lower_edge:],
                current_iter_hist.values[lower_edge:],
                out=np.zeros_like(current_iter_hist.values[lower_edge:]),
                where=current_iter_hist.values[lower_edge:] != 0,
            )
        )
        hist_stat.values[i] = stat_value
        # If prior is provided, calculate.
        prior_value = 0
        if reweighted_prior_output:
            prior = projection_func(reweighted_prior_output, iter, true_bin)
            prior_value = np.sum(
                np.divide(
                    np.abs(current_iter_hist.values - prior.values),
                    current_iter_hist.values,
                    out=np.zeros_like(current_iter_hist.values),
                    where=current_iter_hist.values != 0,
                )
            )
            hist_prior.values[i] = prior_value

        # Total
        hist_total.values[i] = np.sqrt(regularization_value ** 2 + stat_value ** 2 + prior_value ** 2)

    # Plot the total errors
    ax.errorbar(
        hist_total.axes[0].bin_centers,
        hist_total.values,
        xerr=hist_total.axes[0].bin_widths / 2,
        label="Total",
        marker="o",
        linestyle="",
    )
    # The regularization errors
    ax.errorbar(
        hist_reg.axes[0].bin_centers,
        hist_reg.values,
        xerr=hist_reg.axes[0].bin_widths / 2,
        label="Regularization",
        marker="o",
        linestyle="",
    )
    # Plot the stat errors
    ax.errorbar(
        hist_stat.axes[0].bin_centers,
        hist_stat.values,
        xerr=hist_stat.axes[0].bin_widths / 2,
        label="Statistical",
        marker="o",
        linestyle="",
    )
    # And the prior values, if they were provided
    if reweighted_prior_output:
        ax.errorbar(
            hist_prior.axes[0].bin_centers,
            hist_prior.values,
            xerr=hist_prior.axes[0].bin_widths / 2,
            label="Prior",
            marker="o",
            linestyle="",
        )

    # Label and layout
    plot_config.apply(fig=fig, ax=ax)
    # Additional tweaks
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=2.0))

    figure_name = f"{plot_config.name}"
    fig.savefig(output_dir / f"{figure_name}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{figure_name}.png")
    plt.close(fig)


def plot_kt_unfolding(
    unfolding_output: UnfoldingOutput,
    plot_png: bool = False,
    reweighted_prior_output: Optional[UnfoldingOutput] = None,
    unfolding_kt_display_range: Optional[Tuple[float, float]] = None,
) -> Path:
    if unfolding_kt_display_range is None:
        unfolding_kt_display_range = (-0.5, unfolding_output.smeared_var_range.max)
    logger.info(f"Plotting {unfolding_output.identifier}")
    # with sns.color_palette("GnBu_d", n_colors=11):
    with sns.color_palette("Paired", n_colors=unfolding_output.max_n_iter):
        # Main unfolded plot.
        true_jet_pt_range = helpers.JetPtRange(60, 80)
        text = f"${true_jet_pt_range.display_str(label='true')}$"
        plot_unfolded(
            unfolding_output=unfolding_output,
            hist_true=unfolding_output.true_substructure(
                unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
            ),
            hist_n_iter_compare=unfolding_output.unfolded_substructure(
                unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
            ),
            unfolded_hists={
                n_iter: unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
                for n_iter in unfolding_output.n_iter_range_to_plot()
                # for n_iter in range(1, unfolding_output.n_iter_compare + 5)
            },
            plot_config=pb.PlotConfig(
                name=f"unfolded_{unfolding_output.substructure_variable}_true_{str(true_jet_pt_range)}",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"$\text{{d}}N/\text{{d}}k_{{\text{{T}}}}\:(\text{{GeV}}/c)^{{-1}}$",  # noqa: F541
                                log=True,
                                range=(8e-4, None),
                            )
                        ],
                        legend=pb.LegendConfig(location="lower left", ncol=2),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                    pb.Panel(
                        axes=[
                            # pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=(-0.5, 15)),
                            # Take advantage of the smeared and true level substructure var being the same range.
                            pb.AxisConfig(
                                "x",
                                label=r"$k_{\text{T}}\:(\text{GeV}/c)$",
                                range=unfolding_kt_display_range,
                            ),
                            pb.AxisConfig(
                                "y",
                                label="Ratio to {true_label}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding=dict(bottom=0.06)),
            ),
            plot_png=plot_png,
        )
        # Check a broader true jet pt range: 40-120
        true_jet_pt_range = helpers.JetPtRange(40, 120)
        text = f"${true_jet_pt_range.display_str(label='true')}$"
        plot_unfolded(
            unfolding_output=unfolding_output,
            hist_true=unfolding_output.true_substructure(
                unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
            ),
            hist_n_iter_compare=unfolding_output.unfolded_substructure(
                unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
            ),
            unfolded_hists={
                n_iter: unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
                # for n_iter in unfolding_output.n_iter_range_to_plot()
                for n_iter in range(1, unfolding_output.n_iter_compare + 5)
            },
            plot_config=pb.PlotConfig(
                name=f"unfolded_{unfolding_output.substructure_variable}_true_{str(true_jet_pt_range)}",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"$\text{{d}}N/\text{{d}}k_{{\text{{T}}}}\:(\text{{GeV}}/c)^{{-1}}$",  # noqa: F541
                                log=True,
                                range=(1e-4, None),
                            )
                        ],
                        legend=pb.LegendConfig(location="lower left", ncol=2),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                    pb.Panel(
                        axes=[
                            # Take advantage of the smeared and true level substructure var being the same range.
                            pb.AxisConfig(
                                "x",
                                label=r"$k_{\text{T}}\:(\text{GeV}/c)$",
                                range=unfolding_kt_display_range,
                            ),
                            pb.AxisConfig(
                                "y",
                                label="Ratio to {true_label}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding=dict(bottom=0.06)),
            ),
            plot_png=plot_png,
        )
        # Check a higher jet pt bin: 80-100
        true_jet_pt_range = helpers.JetPtRange(80, 100)
        text = f"${true_jet_pt_range.display_str(label='true')}$"
        plot_unfolded(
            unfolding_output=unfolding_output,
            hist_true=unfolding_output.true_substructure(
                unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
            ),
            hist_n_iter_compare=unfolding_output.unfolded_substructure(
                unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
            ),
            unfolded_hists={
                n_iter: unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
                for n_iter in unfolding_output.n_iter_range_to_plot()
                # for n_iter in range(1, unfolding_output.n_iter_compare + 5)
            },
            plot_config=pb.PlotConfig(
                name=f"unfolded_{unfolding_output.substructure_variable}_true_{str(true_jet_pt_range)}",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"$\text{{d}}N/\text{{d}}k_{{\text{{T}}}}\:(\text{{GeV}}/c)^{{-1}}$",  # noqa: F541
                                log=True,
                                range=(1e-3, None),
                            )
                        ],
                        legend=pb.LegendConfig(location="lower left", ncol=2),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                    pb.Panel(
                        axes=[
                            # Take advantage of the smeared and true level substructure var being the same range.
                            pb.AxisConfig(
                                "x",
                                label=r"$k_{\text{T}}\:(\text{GeV}/c)$",
                                range=unfolding_kt_display_range,
                            ),
                            pb.AxisConfig(
                                "y",
                                label="Ratio to {true_label}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding=dict(bottom=0.06)),
            ),
            plot_png=plot_png,
        )
        # Unfolded jet pt
        # First, over the full kt range.
        true_substructure_variable_range = helpers.KtRange(-1, 100)
        text = f"${true_substructure_variable_range.display_str(label='true')}$"
        plot_unfolded(
            unfolding_output=unfolding_output,
            hist_true=unfolding_output.true_jet_pt(
                unfolding_output.true_hist_name, true_substructure_variable_range=true_substructure_variable_range
            ),
            hist_n_iter_compare=unfolding_output.unfolded_jet_pt(
                unfolding_output.n_iter_compare, true_substructure_variable_range=true_substructure_variable_range
            ),
            unfolded_hists={
                n_iter: unfolding_output.unfolded_jet_pt(
                    n_iter=n_iter, true_substructure_variable_range=true_substructure_variable_range
                )
                # for n_iter in unfolding_output.n_iter_range_to_plot()
                for n_iter in range(1, unfolding_output.n_iter_compare + 5)
            },
            plot_config=pb.PlotConfig(
                name="unfolded_pt",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                        ],
                        legend=pb.LegendConfig(location="lower left", ncol=2),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                            pb.AxisConfig(
                                "y",
                                label="Ratio to {true_label}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding=dict(bottom=0.06)),
            ),
            plot_png=plot_png,
        )
        # Since our smeared and true kt ranges usually match, we'll restrict it here.
        # NOTE: Careful here, this doesn't actually apply for the main semi-central and central ranges...
        true_substructure_variable_range = unfolding_output.smeared_var_range  # type: ignore
        text = f"${true_substructure_variable_range.display_str(label='true')}$"
        plot_unfolded(
            unfolding_output=unfolding_output,
            hist_true=unfolding_output.true_jet_pt(
                unfolding_output.true_hist_name, true_substructure_variable_range=true_substructure_variable_range
            ),
            hist_n_iter_compare=unfolding_output.unfolded_jet_pt(
                unfolding_output.n_iter_compare, true_substructure_variable_range=true_substructure_variable_range
            ),
            unfolded_hists={
                n_iter: unfolding_output.unfolded_jet_pt(
                    n_iter=n_iter, true_substructure_variable_range=true_substructure_variable_range
                )
                # for n_iter in unfolding_output.n_iter_range_to_plot()
                for n_iter in range(1, unfolding_output.n_iter_compare + 5)
            },
            plot_config=pb.PlotConfig(
                # Display with f"unfolded_pt_true_{unfolding_output.smeared_var_range}"
                name=f"unfolded_pt_true_{str(true_substructure_variable_range)}",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                        ],
                        legend=pb.LegendConfig(location="lower left", ncol=2),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                            pb.AxisConfig(
                                "y",
                                label="Ratio to {true_label}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding=dict(bottom=0.06)),
            ),
            plot_png=plot_png,
        )

        # Now, on to the refolded.
        text = f"${unfolding_output.smeared_jet_pt_range.display_str(label='data')}$"
        plot_refolded(
            unfolding_output=unfolding_output,
            hist_raw=unfolding_output.smeared_substructure(
                hist_name=unfolding_output.raw_hist_name, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
            ),
            hist_smeared=unfolding_output.smeared_substructure(
                hist_name=unfolding_output.smeared_hist_name, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
            ),
            refolded_hists={
                n_iter: unfolding_output.refolded_substructure(
                    n_iter=n_iter, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
                )
                for n_iter in unfolding_output.n_iter_range_to_plot()
            },
            plot_config=pb.PlotConfig(
                name=f"refolded_{unfolding_output.substructure_variable}",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("y", label=r"$\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                        ],
                        legend=pb.LegendConfig(location="lower left", ncol=2, anchor=(0.15, 0.025)),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$"),
                            # y label is set in the function.
                            pb.AxisConfig(
                                "y",
                                label="Ratio to {refold_label}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding=dict(bottom=0.06)),
            ),
            plot_png=plot_png,
        )
        # Jet pt
        text = f"${unfolding_output.smeared_var_range.display_str(label='data')}$"
        plot_refolded(
            unfolding_output=unfolding_output,
            hist_raw=unfolding_output.smeared_jet_pt(
                hist_name=unfolding_output.raw_hist_name,
                smeared_substructure_variable_range=unfolding_output.smeared_var_range,
            ),
            hist_smeared=unfolding_output.smeared_jet_pt(
                hist_name=unfolding_output.smeared_hist_name,
                smeared_substructure_variable_range=unfolding_output.smeared_var_range,
            ),
            refolded_hists={
                n_iter: unfolding_output.refolded_jet_pt(
                    n_iter=n_iter, smeared_substructure_variable_range=unfolding_output.smeared_var_range
                )
                for n_iter in unfolding_output.n_iter_range_to_plot()
            },
            plot_config=pb.PlotConfig(
                name="refolded_pt",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                        ],
                        legend=pb.LegendConfig(location="upper right", ncol=2, anchor=(0.975, 0.90)),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                            # y label is set in the function.
                            pb.AxisConfig(
                                "y",
                                label="Ratio to {refold_label}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding=dict(bottom=0.06)),
            ),
            plot_png=plot_png,
        )

        # Slice the refolded in jet pt just to get a sense of what they look like.
        if unfolding_output.smeared_jet_pt_range.min > 20:
            # Effectively, a proxy for PbPb
            _small_jet_pt_bins = np.array([30, 40, 60, 80, 100, 120])
            if unfolding_output.smeared_jet_pt_range.min > 30:
                # Drop the lowest bin, since it's outside of our smeared jet pt range.
                _small_jet_pt_bins = _small_jet_pt_bins[1:]
                # Set the lowest bin lower edge to the smallest smeared value. This way,
                # it will work for tuncation systematics.
                # _small_jet_pt_bins[0] = unfolding_output.smeared_jet_pt_range.min
        else:
            # Effectively, a proxy for pp
            _small_jet_pt_bins = np.array([20, 30, 40, 50, 60, 85])
        for _low, _high in zip(_small_jet_pt_bins[:-1], _small_jet_pt_bins[1:]):
            _small_jet_pt_range = helpers.JetPtRange(_low, _high)
            text = f"${_small_jet_pt_range.display_str(label='data')}$"
            plot_refolded(
                unfolding_output=unfolding_output,
                hist_raw=unfolding_output.smeared_substructure(
                    hist_name=unfolding_output.raw_hist_name, smeared_jet_pt_range=_small_jet_pt_range
                ),
                hist_smeared=unfolding_output.smeared_substructure(
                    hist_name=unfolding_output.smeared_hist_name, smeared_jet_pt_range=_small_jet_pt_range
                ),
                refolded_hists={
                    n_iter: unfolding_output.refolded_substructure(
                        n_iter=n_iter, smeared_jet_pt_range=_small_jet_pt_range
                    )
                    for n_iter in unfolding_output.n_iter_range_to_plot()
                },
                plot_config=pb.PlotConfig(
                    name=f"refolded_{unfolding_output.substructure_variable}_{_small_jet_pt_range.histogram_str(label='smeared')}",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y", label=r"$\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True
                                )
                            ],
                            legend=pb.LegendConfig(location="lower left", ncol=2, anchor=(0.15, 0.025)),
                            text=pb.TextConfig(text, 0.97, 0.97),
                        ),
                        # Ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$"),
                                # y label is set in the function.
                                pb.AxisConfig(
                                    "y",
                                    label="Ratio to {refold_label}",
                                    range=(0.5, 1.5),
                                ),
                            ],
                        ),
                    ],
                    figure=pb.Figure(edge_padding=dict(bottom=0.06)),
                ),
                plot_png=plot_png,
            )

    # Plot the response
    if "h2_substructure_variable" in unfolding_output.hists:
        text = f"${unfolding_output.smeared_jet_pt_range.display_str(label='hybrid')}$"
        plot_response(
            hists=unfolding_output.hists,
            plot_config=pb.PlotConfig(
                name=f"response_{unfolding_output.substructure_variable}_hybrid_{unfolding_output.smeared_jet_pt_range}",
                panels=pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$k_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$"),
                        # Use the smeared variable max value as a proxy for the max true value of interest.
                        pb.AxisConfig(
                            "y",
                            label=r"$k_{\text{T}}^{\text{true}}\:(\text{GeV}/c)$",
                            range=(0, unfolding_output.smeared_var_range.max),
                        ),
                    ],
                    text=pb.TextConfig(text, 0.97, 0.03),
                ),
            ),
            output_dir=unfolding_output.output_dir,
            plot_png=plot_png,
        )

    # Plot kt vs jet pt
    plot_jet_pt_vs_substructure(
        hists=unfolding_output.hists,
        hist_name="smeared",
        plot_config=pb.PlotConfig(
            name=f"{unfolding_output.substructure_variable}_vs_jet_pt_hybrid",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$k_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$"),
                    pb.AxisConfig("y", label=r"$p_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$"),
                ],
                text=pb.TextConfig(text, 0.97, 0.03),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )
    # True
    plot_jet_pt_vs_substructure(
        hists=unfolding_output.hists,
        hist_name="true",
        plot_config=pb.PlotConfig(
            name=f"{unfolding_output.substructure_variable}_vs_jet_pt_true",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$k_{\text{T}}^{\text{true}}\:(\text{GeV}/c)$", range=(None, 20)),
                    pb.AxisConfig("y", label=r"$p_{\text{T}}^{\text{true}}\:(\text{GeV}/c)$"),
                ],
                text=pb.TextConfig(text, 0.97, 0.03),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )

    # Select the n_iter iteration
    for true_jet_pt_range in [helpers.JetPtRange(60, 80), helpers.JetPtRange(80, 100)]:
        text = f"${true_jet_pt_range.display_str(label='true')}$"
        plot_select_iteration(
            unfolding_output=unfolding_output,
            projection_func=UnfoldingOutput.unfolded_substructure,  # type: ignore
            max_iter=unfolding_output.max_n_iter,
            true_bin=true_jet_pt_range,
            plot_config=pb.PlotConfig(
                name=f"select_iteration_{unfolding_output.substructure_variable}_true_{str(true_jet_pt_range)}",
                panels=pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label="Iteration"),
                        pb.AxisConfig("y", label="Summed Error", range=(0, None)),
                    ],
                    legend=pb.LegendConfig(location="center right"),
                    text=pb.TextConfig(text, 0.03, 0.03),
                ),
            ),
            output_dir=unfolding_output.output_dir,
            plot_png=plot_png,
            reweighted_prior_output=reweighted_prior_output,
        )

    # Efficiency
    plot_efficiency(
        hists=unfolding_output.hists,
        efficiency_func=_efficiency_substructure_variable,
        true_bins=[
            helpers.JetPtRange(40, 120),
            helpers.JetPtRange(40, 60),
            helpers.JetPtRange(60, 80),
            helpers.JetPtRange(80, 100),
            helpers.JetPtRange(80, 120),
        ],
        true_bin_label="p",
        plot_config=pb.PlotConfig(
            name=f"efficiency_{unfolding_output.substructure_variable}",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", log=True),
                    pb.AxisConfig("y", label="Efficiency"),
                ],
                legend=pb.LegendConfig(location="lower left"),
                # text=pb.TextConfig(text, 0.97, 0.97),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )
    # Cleaned up kt efficiency, focused on the ranges that we will measure.
    plot_efficiency(
        hists=unfolding_output.hists,
        efficiency_func=_efficiency_substructure_variable,
        true_bins=[
            helpers.JetPtRange(60, 80),
        ],
        true_bin_label="p",
        plot_config=pb.PlotConfig(
            name=f"efficiency_{unfolding_output.substructure_variable}_true_pt_60_80",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$k_{\text{T}}\:(\text{GeV}/c)$", range=unfolding_kt_display_range),
                    pb.AxisConfig("y", label="Efficiency"),
                ],
                legend=pb.LegendConfig(location="lower left"),
                # text=pb.TextConfig(text, 0.97, 0.97),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )
    plot_efficiency(
        hists=unfolding_output.hists,
        efficiency_func=_efficiency_pt,
        true_bins=[
            unfolding_output.smeared_var_range,
            # helpers.RangeSelector(unfolding_output.smeared_var_range.min, unfolding_output.smeared_var_range.max),
            # helpers.RangeSelector(1, 15),
            # helpers.RangeSelector(2, 13),
            # helpers.RangeSelector(2, 15),
        ],
        true_bin_label="k",
        plot_config=pb.PlotConfig(
            name="efficiency_pt",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                    pb.AxisConfig("y", label="Efficiency"),
                ],
                legend=pb.LegendConfig(location="lower right"),
                # text=pb.TextConfig(text, 0.97, 0.97),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )

    # plot_spectra_comparison(hists, output_dir)
    # plot_spectra_comparison_fine_binned(hists, output_dir)
    # plot_response_matrix(hists["responseUnscaled"], "response", output_dir)

    return unfolding_output.output_dir


def run(collision_system: str) -> None:
    base_dir = Path("output")
    for unfolding_output in [
        ###################### kt smeared = 2-10 ##########################
        ## 2-10, 1-2, 30-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(1, 2),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=7,
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(1, 2),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=7,
        #    smeared_input=True,
        # ),
        ## 2-10, 1-2, 40-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(1, 2),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(1, 2),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        ## 2-10, 10-13, 30-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=7,
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=7,
        #    smeared_input=True,
        # ),
        ## 2-10, 10-13, 40-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(2, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        ###################### kt smeared = 3-10 ##########################
        ## 3-10, 2-3, 30-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        #    smeared_input=True,
        # ),
        # 3-10, 2-3, 40-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        ## 3-10, 10-13, 30-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        #    smeared_input=True,
        # ),
        ## 3-10, 10-13, 40-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(10, 13),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        # 3-10, 2-3, 40-120, pure matches
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    pure_matches=True,
        #    n_iter_compare=11,
        #    max_iter=15,
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    pure_matches=True,
        #    n_iter_compare=11,
        #    max_iter=15,
        #    smeared_input=True,
        # ),
        ###################### kt smeared = 3-10, broad true bins ##########################
        ## 3-10, 2-3, 30-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        #    suffix="broadTrueBins",
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    n_iter_compare=4,
        #    suffix="broadTrueBins",
        #    smeared_input=True,
        # ),
        ## 3-10, 2-3, 40-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    suffix="broadTrueBins",
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 10),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    suffix="broadTrueBins",
        #    smeared_input=True,
        # ),
        ## 3-11, 30-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 11),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 11),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    smeared_input=True,
        # ),
        ## 3-11, 40-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 11),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 11),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    smeared_input=True,
        # ),
        ## 3-15, 30-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 15),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 15),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(30, 120),
        #    smeared_input=True,
        # ),
        ## 3-15, 40-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 15),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    pure_matches=True,
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(3, 15),
        #    smeared_untagged_var=helpers.KtRange(2, 3),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    pure_matches=True,
        #    smeared_input=True,
        # ),
        ###################### kt smeared = 5-15 ##########################
        ## 4-15, 3-4, 40-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(4, 15),
        #    smeared_untagged_var=helpers.KtRange(3, 4),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(4, 15),
        #    smeared_untagged_var=helpers.KtRange(3, 4),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        ## 5-15, 4-5, 40-120
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(5, 15),
        #    smeared_untagged_var=helpers.KtRange(4, 5),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        # ),
        # UnfoldingOutput(
        #    "kt",
        #    "leading_kt_z_cut_02",
        #    smeared_var_range=helpers.KtRange(5, 15),
        #    smeared_untagged_var=helpers.KtRange(4, 5),
        #    smeared_jet_pt_range=helpers.JetPtRange(40, 120),
        #    n_iter_compare=3,
        #    smeared_input=True,
        # ),
        ####### Dynamical kt ##########
        # 3-15, 2-3, 40-120
        UnfoldingOutput(
            "kt",
            "dynamical_kt",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        UnfoldingOutput(
            "kt",
            "dynamical_kt",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            raw_hist_name="smeared",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        # 2-15, 1-2, 30-120
        UnfoldingOutput(
            "kt",
            "dynamical_kt",
            smeared_var_range=helpers.KtRange(2, 15),
            smeared_untagged_var=helpers.KtRange(1, 2),
            smeared_jet_pt_range=helpers.JetPtRange(30, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        UnfoldingOutput(
            "kt",
            "dynamical_kt",
            smeared_var_range=helpers.KtRange(2, 15),
            smeared_untagged_var=helpers.KtRange(1, 2),
            smeared_jet_pt_range=helpers.JetPtRange(30, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            raw_hist_name="smeared",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        ####### Dynamical time ##########
        # 3-15, 2-3, 40-120
        UnfoldingOutput(
            "kt",
            "dynamical_time",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        UnfoldingOutput(
            "kt",
            "dynamical_time",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            raw_hist_name="smeared",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        ####### Leading kt ##########
        # 3-15, 2-3, 40-120
        UnfoldingOutput(
            "kt",
            "leading_kt",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        UnfoldingOutput(
            "kt",
            "leading_kt",
            smeared_var_range=helpers.KtRange(3, 15),
            smeared_untagged_var=helpers.KtRange(2, 3),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            n_iter_compare=3,
            suffix="broadTrueBins",
            raw_hist_name="smeared",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
    ]:
        plot_kt_unfolding(unfolding_output=unfolding_output)


def plot_delta_R_unfolding(unfolding_output: UnfoldingOutput, plot_png: bool = False) -> Path:
    # with sns.color_palette("GnBu_d", n_colors=11):
    with sns.color_palette("Paired", n_colors=unfolding_output.max_n_iter):
        # Main unfolded plot.
        true_jet_pt_range = helpers.JetPtRange(60, 80)
        text = f"${true_jet_pt_range.display_str(label='true')}$"
        plot_unfolded(
            unfolding_output=unfolding_output,
            hist_true=unfolding_output.true_substructure(
                unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
            ),
            hist_n_iter_compare=unfolding_output.unfolded_substructure(
                unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
            ),
            unfolded_hists={
                n_iter: unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
                for n_iter in unfolding_output.n_iter_range_to_plot()
            },
            plot_config=pb.PlotConfig(
                name=f"unfolded_{unfolding_output.substructure_variable}_true_pt_60_80",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=r"$\text{d}N/\text{d}\Delta R$",
                                log=True,
                            )
                        ],
                        # legend=pb.LegendConfig(location="lower left"),
                        legend=pb.LegendConfig(location="center right"),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                            )
                        ],
                    ),
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$\Delta R$"),
                            pb.AxisConfig(
                                "y",
                                label="Ratio to {true_label}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
            ),
            plot_png=plot_png,
        )
        # Check a broader true jet pt range: 40-120
        true_jet_pt_range = helpers.JetPtRange(40, 120)
        text = f"${true_jet_pt_range.display_str(label='true')}$"
        plot_unfolded(
            unfolding_output=unfolding_output,
            hist_true=unfolding_output.true_substructure(
                unfolding_output.true_hist_name, true_jet_pt_range=true_jet_pt_range
            ),
            hist_n_iter_compare=unfolding_output.unfolded_substructure(
                unfolding_output.n_iter_compare, true_jet_pt_range=true_jet_pt_range
            ),
            unfolded_hists={
                n_iter: unfolding_output.unfolded_substructure(n_iter=n_iter, true_jet_pt_range=true_jet_pt_range)
                for n_iter in unfolding_output.n_iter_range_to_plot()
            },
            plot_config=pb.PlotConfig(
                name=f"unfolded_{unfolding_output.substructure_variable}_true_pt_40_120",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=r"$\text{d}N/\text{d}\Delta R$",
                                log=True,
                            )
                        ],
                        legend=pb.LegendConfig(location="center right"),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                            )
                        ],
                    ),
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$\Delta R$"),
                            pb.AxisConfig(
                                "y",
                                label="Ratio to {true_label}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
            ),
            plot_png=plot_png,
        )
        # Unfolded jet pt
        true_substructure_variable_range = helpers.RgRange(-1, 100)
        text = f"${true_substructure_variable_range.display_str(label='true')}$"
        plot_unfolded(
            unfolding_output=unfolding_output,
            hist_true=unfolding_output.true_jet_pt(
                unfolding_output.true_hist_name, true_substructure_variable_range=true_substructure_variable_range
            ),
            hist_n_iter_compare=unfolding_output.unfolded_jet_pt(
                unfolding_output.n_iter_compare, true_substructure_variable_range=true_substructure_variable_range
            ),
            unfolded_hists={
                n_iter: unfolding_output.unfolded_jet_pt(
                    n_iter=n_iter, true_substructure_variable_range=true_substructure_variable_range
                )
                for n_iter in unfolding_output.n_iter_range_to_plot()
            },
            plot_config=pb.PlotConfig(
                name="unfolded_pt",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                        ],
                        legend=pb.LegendConfig(location="lower left"),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label=fr"Ratio to iter {unfolding_output.n_iter_compare}",
                            )
                        ],
                    ),
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                            pb.AxisConfig(
                                "y",
                                label="Ratio to {true_label}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
            ),
            plot_png=plot_png,
        )

        # Now, on to the refolded.
        text = f"${unfolding_output.smeared_jet_pt_range.display_str(label='data')}$"
        plot_refolded(
            unfolding_output=unfolding_output,
            hist_raw=unfolding_output.smeared_substructure(
                hist_name=unfolding_output.raw_hist_name, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
            ),
            hist_smeared=unfolding_output.smeared_substructure(
                hist_name=unfolding_output.smeared_hist_name, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
            ),
            refolded_hists={
                n_iter: unfolding_output.refolded_substructure(
                    n_iter=n_iter, smeared_jet_pt_range=unfolding_output.smeared_jet_pt_range
                )
                for n_iter in unfolding_output.n_iter_range_to_plot()
            },
            plot_config=pb.PlotConfig(
                name=f"refolded_{unfolding_output.substructure_variable}",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[pb.AxisConfig("y", label=r"$\text{d}N/\text{d}\Delta R$", log=True)],
                        legend=pb.LegendConfig(location="lower center", ncol=2),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$\Delta R$"),
                            # y label is set in the function.
                            pb.AxisConfig(
                                "y",
                                label="Ratio to {refold_label}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
            ),
            plot_png=plot_png,
        )
        # Jet pt
        text = f"${unfolding_output.smeared_var_range.display_str(label='data')}$"
        plot_refolded(
            unfolding_output=unfolding_output,
            hist_raw=unfolding_output.smeared_jet_pt(
                hist_name=unfolding_output.raw_hist_name,
                smeared_substructure_variable_range=unfolding_output.smeared_var_range,
            ),
            hist_smeared=unfolding_output.smeared_jet_pt(
                hist_name=unfolding_output.smeared_hist_name,
                smeared_substructure_variable_range=unfolding_output.smeared_var_range,
            ),
            refolded_hists={
                n_iter: unfolding_output.refolded_jet_pt(
                    n_iter=n_iter, smeared_substructure_variable_range=unfolding_output.smeared_var_range
                )
                for n_iter in unfolding_output.n_iter_range_to_plot()
            },
            plot_config=pb.PlotConfig(
                name="refolded_pt",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("y", label=r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$", log=True)
                        ],
                        legend=pb.LegendConfig(location="upper right", ncol=2, anchor=(0.975, 0.90)),
                        text=pb.TextConfig(text, 0.97, 0.97),
                    ),
                    # Ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig("x", label=r"$p_{\text{T}}\:(\text{GeV}/c)$"),
                            # y label is set in the function.
                            pb.AxisConfig(
                                "y",
                                label="Ratio to {refold_label}",
                                range=(0.5, 1.5),
                            ),
                        ],
                    ),
                ],
            ),
            plot_png=plot_png,
        )

    # Plot the response
    if "h2_substructure_variable" in unfolding_output.hists:
        jet_pt_for_text = helpers.JetPtRange(40, 120)
        text = f"${jet_pt_for_text.display_str(label='hybrid')}$"
        plot_response(
            hists=unfolding_output.hists,
            plot_config=pb.PlotConfig(
                name=f"response_{unfolding_output.substructure_variable}_hybrid_40_120",
                panels=pb.Panel(
                    axes=[
                        pb.AxisConfig("x", label=r"$\Delta R^{\text{hybrid}}\:(\text{GeV}/c)$"),
                        pb.AxisConfig("y", label=r"$\Delta R^{\text{true}}\:(\text{GeV}/c)$", range=(0, 0.4)),
                    ],
                    text=pb.TextConfig(text, 0.97, 0.03),
                ),
                figure=pb.Figure(edge_padding={"left": 0.11, "bottom": 0.10}),
            ),
            output_dir=unfolding_output.output_dir,
            plot_png=plot_png,
        )

    # Select the n_iter iteration
    jet_pt_for_text = helpers.JetPtRange(60, 80)
    text = f"${jet_pt_for_text.display_str(label='true')}$"
    plot_select_iteration(
        unfolding_output=unfolding_output,
        projection_func=UnfoldingOutput.unfolded_substructure,  # type: ignore
        max_iter=unfolding_output.max_n_iter,
        true_bin=helpers.JetPtRange(60, 80),
        plot_config=pb.PlotConfig(
            name=f"select_iteration_{unfolding_output.substructure_variable}_true_pt_60_80",
            panels=pb.Panel(
                axes=[
                    pb.AxisConfig("x", label="Iteration"),
                    pb.AxisConfig("y", label="Summed Error", range=(0, None)),
                ],
                legend=pb.LegendConfig(location="center right"),
                text=pb.TextConfig(text, 0.03, 0.03),
            ),
        ),
        output_dir=unfolding_output.output_dir,
        plot_png=plot_png,
    )

    return unfolding_output.output_dir


def run_delta_R(collision_system: str) -> None:
    base_dir = Path("output")
    for unfolding_output in [
        UnfoldingOutput(
            "delta_R",
            "leading_kt_z_cut_02",
            # Hack until the labeling is fixed...
            smeared_var_range=helpers.RgRange(0, 350),
            smeared_untagged_var=helpers.RgRange(-50, 0),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            collision_system=collision_system,
            base_dir=base_dir,
        ),
        UnfoldingOutput(
            "delta_R",
            "leading_kt_z_cut_02",
            # Hack until the labeling is fixed...
            smeared_var_range=helpers.RgRange(0, 350),
            smeared_untagged_var=helpers.RgRange(-50, 0),
            smeared_jet_pt_range=helpers.JetPtRange(40, 120),
            raw_hist_name="smeared",
            collision_system=collision_system,
            base_dir=base_dir,
        ),
    ]:
        plot_delta_R_unfolding(unfolding_output=unfolding_output)


if __name__ == "__main__":
    # Setup
    helpers.setup_logging()
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("pachyderm.histogram").setLevel(logging.INFO)
    collision_system = "PbPb"

    # Enable ticks on all sides
    # Unfortunately, some of this is overriding the pachyderm plotting style.
    # That will have to be updated eventually...
    # matplotlib.rcParams["xtick.top"] = True
    # matplotlib.rcParams["xtick.minor.top"] = True
    # matplotlib.rcParams["ytick.right"] = True
    # matplotlib.rcParams["ytick.minor.right"] = True

    run(collision_system=collision_system)
    # run_delta_R(collision_system=collision_system)
