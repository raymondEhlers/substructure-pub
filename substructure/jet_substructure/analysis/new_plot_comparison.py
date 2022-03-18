""" New methods for plotting substructure comparison from the skim output.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Optional, Sequence

import boost_histogram as bh
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
from pachyderm import binned_data

import jet_substructure.analysis.plot_base as pb
from jet_substructure.analysis import plot_from_skim
from jet_substructure.base import helpers


logger = logging.getLogger(__name__)

pachyderm.plot.configure()
# Enable ticks on all sides
# Unfortunately, some of this is overriding the pachyderm plotting style.
# That will have to be updated eventually...
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["xtick.minor.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["ytick.minor.right"] = True


def plot_compare_grooming_methods_for_attribute(
    hists: Mapping[str, bh.Histogram],
    grooming_methods: Sequence[str],
    attr_name: str,
    prefix: str,
    tag: str,
    jet_pt_bin: helpers.JetPtRange,
    set_zero_to_nan: bool,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    fig: Optional[matplotlib.figure.Figure] = None,
    ax: Optional[matplotlib.axes.Axes] = None,
    plot_png: Optional[bool] = False,
) -> str:
    """Plot comparison between grooming methods for a given substructure variable (attribute) and given prefix.

    It can be for a single or multiple grooming methods.

    """
    logger.info(f"Plotting grooming method comparison for {attr_name}")

    passed_mpl_fig = True
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        passed_mpl_fig = False

    grooming_styling = pb.define_grooming_styles()

    for grooming_method in grooming_methods:
        # Setup
        style = grooming_styling[grooming_method]

        # Axes: jet_pt, attr_name
        logger.debug(f"Looking at hist: {grooming_method}_{prefix}_{attr_name}_{tag}")
        bh_hist = hists[f"{grooming_method}_{prefix}_{attr_name}_{tag}"]
        # Need to project to just the attr of interest.
        h = binned_data.BinnedData.from_existing_data(
            bh_hist[bh.loc(jet_pt_bin.min) : bh.loc(jet_pt_bin.max) : bh.sum, :]  # noqa: E203
        )

        # Normalize
        # Normalize by the sum of the values to get the n_jets values.
        # Then, we still need to normalize by the bin widths.
        h /= np.sum(h.values)
        h /= h.axes[0].bin_widths

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

    # Labeling and presentation
    plot_config.apply(fig=fig, ax=ax)

    # Store and cleanup
    # It's expected that the attr_name is already included in the `plot_config.name`.
    # Sanity check to make sure we don't get that wrong!
    # if attr_name not in plot_config.name:
    #    raise ValueError(
    #        f"PlotConfig name must contain the attr name! attr_name: {attr_name}, name: {plot_config.name}"
    #    )

    filename = f"{plot_config.name}_{jet_pt_bin}_iterative_splittings"
    fig.savefig(output_dir / f"{filename}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{filename}.png")

    if not passed_mpl_fig:
        plt.close(fig)

    return filename


def plot_compare_grooming_methods_for_prefix(
    hists: Sequence[plot_from_skim.PlotHists],
    attr_name: str,
    tag: str,
    grooming_methods: Sequence[str],
    jet_pt_bin: helpers.RangeSelector,
    set_zero_to_nan: bool,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    plot_png: bool = False,
) -> None:
    """Plot comparison between prefixes for a given substructure variable (attribute) and grooming method(s).

    It can be for a single or multiple grooming methods.
    """
    # Setup
    display_labels_vs = " vs. ".join([obj.display_label for obj in hists])
    logger.info(
        f"Plotting grooming method comparison for {attr_name}, {display_labels_vs}, grooming_methods: {grooming_methods}"
    )
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )

    grooming_styles = pb.define_grooming_styles()

    for grooming_method in grooming_methods:
        main_hist = None
        for hists_obj in hists:
            # Setup and project
            # Axes: jet_pt, attr_name
            h = plot_from_skim._project_and_prepare_grooming_variable_hist(
                bh_hist=hists_obj.hists[f"{grooming_method}_{hists_obj.prefix}_{attr_name}_{tag}"],
                jet_pt_bin=jet_pt_bin,
                set_zero_to_nan=set_zero_to_nan,
            )

            # Setup
            # And then the label
            grooming_method_display = grooming_styles[grooming_method].label
            label = f"{hists_obj.display_label}, {grooming_method_display}"

            ax.errorbar(
                h.axes[0].bin_centers,
                h.values,
                yerr=h.errors,
                xerr=h.axes[0].bin_widths / 2,
                linestyle="",
                label=label,
            )

            # We've plotted the main obj, so now we store that hist, and we will treat the rest as comparisons.
            # For those comparisons, we want to create ratios.
            if main_hist is None:
                main_hist = h

            # Plot every time so the colors line up. This is lazy, but it's fine for now.
            ratio = main_hist / h
            ax_ratio.errorbar(
                ratio.axes[0].bin_centers,
                ratio.values,
                yerr=ratio.errors,
                xerr=ratio.axes[0].bin_widths / 2,
                linestyle="",
            )

    # Reference value for ratio
    ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Apply the PlotConfig
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])

    # Store and cleanup
    # It's expected that the attr_name is already included in the `plot_config.name`.
    # Sanity check to make sure we don't get that wrong!
    # if attr_name not in plot_config.name:
    #    raise ValueError(
    #        f"PlotConfig name must contain the attr name! attr_name: {attr_name}, name: {plot_config.name}"
    #    )

    grooming_methods_filename_label = ""
    if len(grooming_methods) == 1:
        grooming_methods_filename_label = f"_{grooming_methods[0]}"
    identifiers = "_".join([obj.identifier for obj in hists])
    filename = f"{plot_config.name}_{jet_pt_bin}{grooming_methods_filename_label}_{identifiers}_iterative_splittings"
    fig.savefig(output_dir / f"{filename}.pdf")
    if plot_png:
        output_dir_png = output_dir / "png"
        output_dir_png.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir_png / f"{filename}.png")
    plt.close(fig)
