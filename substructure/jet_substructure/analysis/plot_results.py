#!/usr/bin/env python3

""" Plotting for the jet substructure analysis.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import logging
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import attr
import boost_histogram as bh
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
from pachyderm import binned_data

# from jet_substructure.analysis import analysis_methods
from jet_substructure.base import analysis_objects, helpers


logger = logging.getLogger(__name__)

pachyderm.plot.configure()
# Enable ticks on all sides
# Unfortunately, some of this is overriding the pachyderm plotting style.
# That will have to be updated eventually...
matplotlib.rcParams["xtick.top"] = True
matplotlib.rcParams["xtick.minor.top"] = True
matplotlib.rcParams["ytick.right"] = True
matplotlib.rcParams["ytick.minor.right"] = True


@attr.s
class PlotConfig:
    name: str = attr.ib()
    x_label: str = attr.ib()
    y_label: str = attr.ib()
    legend_location: str = attr.ib(default="center right")
    log_y: bool = attr.ib(default=True)


def _plot_distribution(  # noqa: C901
    attribute_name: str,
    hists: analysis_objects.Hists[analysis_objects.SubstructureHists],
    identifier: analysis_objects.Identifier,
    jet_type_label: str,
    plot_config: PlotConfig,
    path: Path,
    ratio_denominator_hists: Optional[analysis_objects.Hists[analysis_objects.SubstructureHists]] = None,
) -> None:
    # Setup
    if ratio_denominator_hists:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=(8, 6),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        ax, ax_ratio = axes
    else:
        fig, ax = plt.subplots(figsize=(8, 6))
    logger.info(f"Plotting {attribute_name}, {identifier}{', ratio' if ratio_denominator_hists else ''}")

    for technique, technique_hists in hists:
        # It will fail for 0 jets, so skip it
        if technique_hists.n_jets == 0:
            logger.warning(f"No jets within {identifier}_{technique}. Skipping bin!")
            continue
        # We don't want to include inclusive in the comparison at the moment.
        if technique == "inclusive":
            continue

        h: Union[bh.Histogram, binned_data.BinnedData] = getattr(technique_hists, attribute_name)
        h = binned_data.BinnedData.from_existing_data(h)

        # Rebin by a factor of 2 for the kt.
        if attribute_name == "kt":
            rebin_factor = 2
            bh_hist = h.to_boost_histogram()
            bh_hist = bh_hist[:: bh.rebin(rebin_factor)]
            h = binned_data.BinnedData.from_existing_data(bh_hist)
            h /= rebin_factor

        # Scale by bin widths and number of jets
        h /= h.axis.bin_widths
        h /= technique_hists.n_jets

        ax.errorbar(
            h.axis.bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axis.bin_widths / 2,
            marker=".",
            linestyle="",
            label=technique_hists.title,
        )

        if ratio_denominator_hists:
            ratio_denominator = getattr(ratio_denominator_hists, technique)

            h_denominator: Union[bh.Histogram, binned_data.BinnedData] = getattr(ratio_denominator, attribute_name)
            h_denominator = binned_data.BinnedData.from_existing_data(h_denominator)

            # Rebin by a factor of 2 for the kt.
            if attribute_name == "kt":
                rebin_factor = 2
                bh_hist_denominator = h_denominator.to_boost_histogram()
                bh_hist_denominator = bh_hist_denominator[:: bh.rebin(rebin_factor)]
                h_denominator = binned_data.BinnedData.from_existing_data(bh_hist_denominator)
                h_denominator /= rebin_factor

            h_denominator /= ratio_denominator.n_jets
            h_denominator /= h_denominator.axis.bin_widths

            # Don't apply any further normalization! We want the direct ratio of the values!
            h_ratio = h / h_denominator

            logger.warning(f"Ratio integral: {np.sum(h_ratio.values * h_ratio.axis.bin_widths)}")

            # Plot the ratio
            ax_ratio.errorbar(
                h_ratio.axis.bin_centers,
                h_ratio.values,
                yerr=h_ratio.errors,
                xerr=h_ratio.axis.bin_widths / 2,
                marker=".",
                linestyle="",
                alpha=0.6,
            )

    # Labeling
    text = identifier.display_str(jet_pt_label=jet_type_label)
    ax.text(
        0.95,
        0.95,
        text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
    )

    # Presentation
    ax.legend(frameon=False, loc=plot_config.legend_location)
    if plot_config.log_y:
        ax.set_yscale("log")
    ax.set_ylabel(plot_config.y_label)
    if ratio_denominator_hists:
        ax_ratio.set_xlabel(plot_config.x_label)
        ax_ratio.set_ylabel("Recur./Iter.")
        # As standard for a ratio.
        ax_ratio.set_ylim([0, 2])
    else:
        ax.set_xlabel(plot_config.x_label)
    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.12,
        bottom=0.11,
        right=0.98,
        top=0.98,
    )
    fig.align_ylabels()

    # Store and cleanup
    filename = f"{plot_config.name}_{str(identifier)}"
    if ratio_denominator_hists:
        filename = f"{filename}_ratio"
    fig.savefig(path / f"{filename}.pdf")
    plt.close(fig)


def _plot_total_number_of_splittings(
    masked_recursive_hists: analysis_objects.Hists[analysis_objects.SubstructureHists],
    masked_iterative_hists: analysis_objects.Hists[analysis_objects.SubstructureHists],
    identifier: analysis_objects.Identifier,
    jet_type_label: str,
    plot_config: PlotConfig,
    path: Path,
) -> None:
    # Setup
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    ax, ax_ratio = axes
    logger.info(f"Plotting total_number_of_splittings, {identifier} with ratio")

    recursive_hists = masked_recursive_hists.inclusive
    h: Union[bh.Histogram, binned_data.BinnedData] = recursive_hists.total_number_of_splittings
    h = binned_data.BinnedData.from_existing_data(h)

    # Scale by bin widths and number of jets
    h /= h.axis.bin_widths
    h /= recursive_hists.n_jets

    ax.errorbar(
        h.axis.bin_centers,
        h.values,
        yerr=h.errors,
        xerr=h.axis.bin_widths / 2,
        marker=".",
        linestyle="",
        label="Recursive",
    )

    # Plot iterative splittings
    iterative_hists = masked_iterative_hists.inclusive
    h_iterative: Union[bh.Histogram, binned_data.BinnedData] = iterative_hists.total_number_of_splittings
    h_iterative = binned_data.BinnedData.from_existing_data(h_iterative)

    # Scale by bin widths and number of jets
    h_iterative /= h_iterative.axis.bin_widths
    h_iterative /= iterative_hists.n_jets

    ax.errorbar(
        h_iterative.axis.bin_centers,
        h_iterative.values,
        yerr=h_iterative.errors,
        xerr=h_iterative.axis.bin_widths / 2,
        marker=".",
        linestyle="",
        label="Iterative",
    )

    # Ratio
    # Don't apply any further normalization! We want the direct ratio of the values!
    h_ratio = h / h_iterative

    logger.warning(f"Ratio integral: {np.sum(h_ratio.values * h_ratio.axis.bin_widths)}")

    # Plot the ratio
    ax_ratio.errorbar(
        h_ratio.axis.bin_centers,
        h_ratio.values,
        yerr=h_ratio.errors,
        xerr=h_ratio.axis.bin_widths / 2,
        marker=".",
        linestyle="",
        alpha=0.6,
    )

    # Labeling
    text = identifier.display_str(jet_pt_label=jet_type_label)
    ax.text(
        0.95,
        0.95,
        text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
    )

    # Presentation
    ax.legend(frameon=False, loc=plot_config.legend_location)
    if plot_config.log_y:
        ax.set_yscale("log")
    ax.set_ylabel(plot_config.y_label)
    ax_ratio.set_xlabel(plot_config.x_label)
    ax_ratio.set_ylabel("Recur./Iter.")
    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.12,
        bottom=0.11,
        right=0.98,
        top=0.98,
    )
    fig.align_ylabels()

    # Store and cleanup
    filename = f"{plot_config.name}_inclusive_{str(identifier)}_ratio"
    fig.savefig(path / f"{filename}.pdf")
    plt.close(fig)


def _plot_distribution_in_different_pt_bins(
    all_hists: Mapping[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]],
    jet_type_label: str,
    attribute_name: str,
    selected_technique: str,
    plot_config: PlotConfig,
    path: Path,
) -> None:
    for iterative_splittings in [False, True]:
        fig, ax = plt.subplots(figsize=(8, 6))
        for identifier, selected_hists in all_hists.items():
            if identifier.iterative_splittings != iterative_splittings:
                continue
            splittings_label = f"{identifier.iterative_splittings_label}_splittings"

            technique_hists = getattr(selected_hists, selected_technique)
            h: Union[bh.histogram, binned_data.BinnedData] = getattr(technique_hists, attribute_name)
            if isinstance(h, bh.Histogram):
                h = binned_data.BinnedData.from_existing_data(h)

            if technique_hists.n_jets == 0:
                logger.warning(f"No jets within {identifier}_{selected_technique}. Skipping bin!")
                continue

            # Scale by bin widths and number of jets
            h /= h.axes[0].bin_widths
            h /= technique_hists.n_jets

            logger.debug(
                f"Plotting {identifier}, {selected_technique}, {attribute_name}, n_jets: {technique_hists.n_jets}"
            )
            # Plot
            ax.errorbar(
                h.axis.bin_centers,
                h.values,
                yerr=h.errors,
                xerr=h.axis.bin_widths / 2,
                marker=".",
                linestyle="",
                label=f"${identifier.jet_pt_bin.display_str(label=jet_type_label)}$",
            )

        # Labeling
        # Make it into (for example), Iterative splittings
        text = " ".join(splittings_label.split("_")).capitalize()
        text += "\n" + technique_hists.title
        ax.text(
            0.95,
            0.95,
            text,
            transform=ax.transAxes,
            horizontalalignment="right",
            verticalalignment="top",
            multialignment="right",
        )

        # Presentation
        ax.legend(frameon=False, loc=plot_config.legend_location)
        if plot_config.log_y:
            ax.set_yscale("log")
        ax.set_xlabel(plot_config.x_label)
        ax.set_ylabel(plot_config.y_label)
        fig.tight_layout()
        fig.subplots_adjust(
            # Reduce spacing between subplots
            hspace=0,
            wspace=0,
            # Reduce external spacing
            left=0.13,
            bottom=0.11,
            right=0.99,
            top=0.98,
        )

        # Store and reset
        filename = path / f"{attribute_name}_jet_pt_comparison_{selected_technique}_{splittings_label}.pdf"
        logger.debug(f"Saving to filename {filename}")
        fig.savefig(filename)
        plt.close(fig)


def _plot_lund_plane(
    technique: str,
    identifier: analysis_objects.Identifier,
    jet_type_label: str,
    hists: analysis_objects.SubstructureHists,
    path: Path,
) -> None:
    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))
    logger.info(f"Plotting lund plane for {technique}, {identifier}")

    h: Union[bh.Histogram, binned_data.BinnedData] = hists.lund_plane
    if isinstance(h, bh.Histogram):
        h = binned_data.BinnedData.from_existing_data(h)

    # Scale by bin width
    x_bin_widths, y_bin_widths = np.meshgrid(*h.axes.bin_widths)
    bin_widths = x_bin_widths * y_bin_widths
    # print(f"x_bin_widths: {x_bin_widths.size}")
    # print(f"y_bin_widths: {y_bin_widths.size}")
    # print(f"bin_widths size: {bin_widths.size}")
    h /= bin_widths
    # Scale by njets.
    h /= hists.n_jets

    # Determine the normalization range
    z_axis_range = {
        "vmin": h.values[h.values > 0].min(),
        "vmax": h.values.max(),
    }
    if technique == "inclusive":
        z_axis_range = {
            "vmin": 10e-3,
            "vmax": 5,
        }

    # Make the plot
    mesh = ax.pcolormesh(
        h.axes[0].bin_edges.T,
        h.axes[1].bin_edges.T,
        h.values.T,
        norm=matplotlib.colors.LogNorm(**z_axis_range),
    )
    fig.colorbar(mesh, pad=0.02)

    # Labeling
    text = identifier.display_str(jet_pt_label=jet_type_label)
    text += "\n" + hists.title
    ax.text(
        0.95,
        0.95,
        text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
    )

    # Presentation
    ax.set_xlabel(r"$\log{(1/\Delta R)}$")
    ax.set_ylabel(r"$\log{(k_{\text{T}})}$")
    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.10,
        bottom=0.11,
        right=0.99,
        top=0.98,
    )

    # Store and reset
    fig.savefig(path / f"lund_plane_{str(identifier)}_{technique}.pdf")
    plt.close(fig)


def lund_plane(  # noqa: C901
    all_hists: Mapping[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]],
    jet_type_label: str,
    path: Path,
) -> None:
    # Validation
    path = path / jet_type_label
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Storing plots in {path}")

    # Plot labels
    kt_label = PlotConfig(
        name="kt",
        x_label=r"$k_{\text{T}}\:(\text{GeV}/c)$",
        y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
        legend_location="lower left",
    )
    z_label = PlotConfig(
        name="z",
        x_label=r"$z$",
        y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}z$",
        legend_location="lower right",
    )
    delta_R_label = PlotConfig(
        name="delta_R",
        x_label=r"$R$",
        y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}R$",
        legend_location="lower right",
    )
    theta_label = PlotConfig(
        name="theta",
        x_label=r"$\theta$",
        y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}\theta$",
        legend_location="lower right",
    )
    splitting_number_label = PlotConfig(
        name="splitting_number",
        x_label=r"$n$",
        y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n$",
        legend_location="center right",
        log_y=False,
    )
    # NOTE: Assumes fixed value of kt > 5!!
    splitting_number_perturbative_label = PlotConfig(
        name="splitting_number_kt_greater_than_5",
        x_label=r"$n$",
        y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n$ ($k_{\text{T}} > 5$)",
        legend_location="center right",
        log_y=False,
    )
    total_number_of_splittings_label = PlotConfig(
        name="total_number_of_splittings",
        x_label=r"$n_{\text{total}}$",
        y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}n_{\text{total}}$",
        legend_location="center right",
        log_y=False,
    )

    distributions: List[Tuple[str, PlotConfig]] = [
        ("kt", kt_label),
        ("z", z_label),
        ("delta_R", delta_R_label),
        ("theta", theta_label),
        ("splitting_number", splitting_number_label),
        ("splitting_number_perturbative", splitting_number_perturbative_label),
        # total number of splittings is intentionally _not_ included because we only case about
        # the inclusive case.
        # ("total_number_of_splittings", total_number_of_splittings_label),
    ]
    # Keep track of this for the final plots
    techniques = set()
    for identifier, masked_hists in all_hists.items():
        for attribute_name, plot_config in distributions:
            _plot_distribution(
                attribute_name=attribute_name,
                hists=masked_hists,
                identifier=identifier,
                jet_type_label=jet_type_label,
                plot_config=plot_config,
                path=path,
            )
            if not identifier.iterative_splittings:
                # Plot the ratio of recursive to iterative
                _plot_distribution(
                    attribute_name=attribute_name,
                    hists=masked_hists,
                    identifier=identifier,
                    jet_type_label=jet_type_label,
                    plot_config=plot_config,
                    path=path,
                    ratio_denominator_hists=all_hists[
                        analysis_objects.Identifier(iterative_splittings=True, jet_pt_bin=identifier.jet_pt_bin)
                    ],
                )

        # Plot the inclusive case for total number of splittings.
        # We directly compare the iterative and recursive cases, so we only plot once and show both sets of values.
        if not identifier.iterative_splittings:
            # It will fail for 0 jets, so skip it
            if masked_hists.inclusive.n_jets != 0:
                _plot_total_number_of_splittings(
                    masked_recursive_hists=masked_hists,
                    masked_iterative_hists=all_hists[
                        analysis_objects.Identifier(iterative_splittings=True, jet_pt_bin=identifier.jet_pt_bin)
                    ],
                    identifier=identifier,
                    jet_type_label=jet_type_label,
                    plot_config=total_number_of_splittings_label,
                    path=path,
                )
            else:
                logger.warning(f"No jets within {identifier}_total_number_of_splittings. Skipping bin!")

        for technique, hists in masked_hists:
            # Store each technique
            techniques.add(technique)
            if hists.n_jets == 0:
                logger.warning(f"No jets within {identifier}_{technique}. Skipping bin!")
                continue
            # Plot Lund Plane
            _plot_lund_plane(
                technique=technique, identifier=identifier, jet_type_label=jet_type_label, hists=hists, path=path
            )

    # Plot the same parameter, for different jet pt bins parameters.
    for attribute_name, plot_config in distributions:
        for technique in techniques:
            # Skip inclusive for now.
            if technique == "inclusive":
                continue
            _plot_distribution_in_different_pt_bins(
                all_hists=all_hists,
                jet_type_label=jet_type_label,
                attribute_name=attribute_name,
                selected_technique=technique,
                plot_config=plot_config,
                path=path,
            )


def _project_matching(
    input_hist: binned_data.BinnedData, axis_to_keep: int, identifier: analysis_objects.MatchingHybridIdentifier
) -> binned_data.BinnedData:
    # Setup
    # Axes: 0 = matched_jet_pt, 1 = matched_kt, 2 = hybrid_jet_pt, 3 = hybrid_kt
    # Determine the range for hybrid jet pt.
    epsilon = 0.00001
    # NOTE: We don't have an epsilon on the upper range because slices aren't inclusive on the upper edge (unlike ROOT projections)
    hybrid_jet_pt_range = slice(
        input_hist.axes[2].find_bin(identifier.jet_pt_bin.min + epsilon),
        input_hist.axes[2].find_bin(identifier.jet_pt_bin.max),
    )
    # We only want to select on the minimum value.
    hybrid_kt_range = slice(input_hist.axes[3].find_bin(identifier.min_kt + epsilon), None)
    slices = (slice(None, None), slice(None, None), hybrid_jet_pt_range, hybrid_kt_range)
    axes_to_sum = tuple([i for i in range(len(input_hist.axes)) if i != axis_to_keep])

    # logger.debug(f"axis_to_keep: {axis_to_keep}, axes_to_sum: {axes_to_sum}, hybrid jet pt range: {hybrid_jet_pt_range}, hybrid kt range: {hybrid_kt_range}")

    return_hist = binned_data.BinnedData(
        axes=input_hist.axes[axis_to_keep],
        values=np.sum(input_hist.values[slices], axis=axes_to_sum),
        variances=np.sum(input_hist.variances[slices], axis=axes_to_sum),
    )

    # Sanity check
    if axis_to_keep == 0:
        import boost_histogram as bh

        bh_hist = input_hist.to_boost_histogram()
        logger.debug(f"bh_hist shape: {bh_hist.view().value.shape}")
        bh_slices = (
            slice(None, None),
            slice(slices[1].start, slices[1].stop, bh.sum),
            slice(slices[2].start, slices[2].stop, bh.sum),
            slice(slices[3].start, slices[3].stop, bh.sum),
        )
        logger.debug(bh_slices)
        bh_return_hist = bh_hist[tuple(bh_slices)]
        assert np.allclose(return_hist.values, bh_return_hist.view().value)
        assert np.allclose(return_hist.variances, bh_return_hist.view().variance)

    return return_hist


def _plot_matching(
    technique: str,
    identifier: analysis_objects.MatchingHybridIdentifier,
    axis_parameter: str,
    hists: analysis_objects.SubstructureMatchingSubjetHists,
    path: Path,
) -> None:
    # Setup
    # Maps from the name that we want to keep to the number of the axis that we need to sum out.
    axis_name_to_axis_to_keep_map = {"pt": 0, "kt": 1}
    axis_to_keep = axis_name_to_axis_to_keep_map[axis_parameter]
    # Figures
    fig, axes = plt.subplots(3, 3, figsize=(10, 10), sharex=True, sharey=True)
    fig_single, ax = plt.subplots(figsize=(10, 8))
    logger.info(f"Plotting matching hist for {technique}, {identifier}")

    # NOTE: We convert the hists here to ensure that we're working with copies!
    normalization = _project_matching(hists.all, axis_to_keep=axis_to_keep, identifier=identifier)

    # Both correctly tagged goes in the upper left
    pure = _project_matching(hists.pure, axis_to_keep=axis_to_keep, identifier=identifier)
    pure /= normalization
    axes[0, 0].errorbar(
        pure.axes[0].bin_centers,
        pure.values,
        yerr=pure.errors,
        xerr=pure.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        # label="",
    )
    ax.errorbar(
        pure.axes[0].bin_centers,
        pure.values,
        yerr=pure.errors,
        xerr=pure.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        label="Pure matches",
    )
    # Skip this panel
    # axes[0, 1].set_visible(False)
    # Leading wasn't tagged, but subleading was correctly tagged as subleading.
    leading_untagged_subleading_correct = _project_matching(
        hists.leading_untagged_subleading_correct,
        axis_to_keep=axis_to_keep,
        identifier=identifier,
    )
    leading_untagged_subleading_correct /= normalization
    axes[0, 2].errorbar(
        leading_untagged_subleading_correct.axes[0].bin_centers,
        leading_untagged_subleading_correct.values,
        yerr=leading_untagged_subleading_correct.errors,
        xerr=leading_untagged_subleading_correct.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        # label="",
    )
    ax.errorbar(
        leading_untagged_subleading_correct.axes[0].bin_centers,
        leading_untagged_subleading_correct.values,
        yerr=leading_untagged_subleading_correct.errors,
        xerr=leading_untagged_subleading_correct.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        label="Leading unmatched, subleading matched",
    )
    # Skip this panel
    # axes[1, 0].set_visible(False)
    # Swapped (ie. swap)
    swap = _project_matching(hists.swap, axis_to_keep=axis_to_keep, identifier=identifier)
    swap /= normalization
    axes[1, 1].errorbar(
        swap.axes[0].bin_centers,
        swap.values,
        yerr=swap.errors,
        xerr=swap.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        # label="",
    )
    ax.errorbar(
        swap.axes[0].bin_centers,
        swap.values,
        yerr=swap.errors,
        xerr=swap.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        label="Swaps",
    )
    # Leading untagged, subleading mistag
    leading_untagged_subleading_mistag = _project_matching(
        hists.leading_untagged_subleading_mistag, axis_to_keep=axis_to_keep, identifier=identifier
    )
    leading_untagged_subleading_mistag /= normalization
    axes[1, 2].errorbar(
        leading_untagged_subleading_mistag.axes[0].bin_centers,
        leading_untagged_subleading_mistag.values,
        yerr=leading_untagged_subleading_mistag.errors,
        xerr=leading_untagged_subleading_mistag.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        # label="",
    )
    ax.errorbar(
        leading_untagged_subleading_mistag.axes[0].bin_centers,
        leading_untagged_subleading_mistag.values,
        yerr=leading_untagged_subleading_mistag.errors,
        xerr=leading_untagged_subleading_mistag.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        label="Leading unmatched, subleading in leading",
    )
    # Leading correct, subleading untagged
    leading_correct_subleading_untagged = _project_matching(
        hists.leading_correct_subleading_untagged, axis_to_keep=axis_to_keep, identifier=identifier
    )
    leading_correct_subleading_untagged /= normalization
    axes[2, 0].errorbar(
        leading_correct_subleading_untagged.axes[0].bin_centers,
        leading_correct_subleading_untagged.values,
        yerr=leading_correct_subleading_untagged.errors,
        xerr=leading_correct_subleading_untagged.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        # label="",
    )
    ax.errorbar(
        leading_correct_subleading_untagged.axes[0].bin_centers,
        leading_correct_subleading_untagged.values,
        yerr=leading_correct_subleading_untagged.errors,
        xerr=leading_correct_subleading_untagged.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        label="Leading matched, subleading unmatched",
    )
    # Leading mistag, subleading untagged
    leading_mistag_subleading_untagged = _project_matching(
        hists.leading_mistag_subleading_untagged, axis_to_keep=axis_to_keep, identifier=identifier
    )
    leading_mistag_subleading_untagged /= normalization
    axes[2, 1].errorbar(
        leading_mistag_subleading_untagged.axes[0].bin_centers,
        leading_mistag_subleading_untagged.values,
        yerr=leading_mistag_subleading_untagged.errors,
        xerr=leading_mistag_subleading_untagged.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        # label="",
    )
    ax.errorbar(
        leading_mistag_subleading_untagged.axes[0].bin_centers,
        leading_mistag_subleading_untagged.values,
        yerr=leading_mistag_subleading_untagged.errors,
        xerr=leading_mistag_subleading_untagged.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        label="Leading in subleading, subleading unmatched",
    )
    # Both untagged
    both_untagged = _project_matching(hists.both_untagged, axis_to_keep=axis_to_keep, identifier=identifier)
    both_untagged /= normalization
    axes[2, 2].errorbar(
        both_untagged.axes[0].bin_centers,
        both_untagged.values,
        yerr=both_untagged.errors,
        xerr=both_untagged.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        # label="",
    )
    ax.errorbar(
        both_untagged.axes[0].bin_centers,
        both_untagged.values,
        yerr=both_untagged.errors,
        xerr=both_untagged.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        label="Leading, subleading unmatched",
    )

    # Set range
    for ax_temp in axes.flat:
        ax_temp.set_ylim([0, 1.2])

    # Labeling
    text = identifier.display_str()
    text += "\n" + hists.title
    axes[0, 1].text(
        0.5,
        0.9,
        text,
        transform=axes[0, 1].transAxes,
        horizontalalignment="center",
        verticalalignment="top",
        multialignment="center",
    )

    # Presentation
    # Axis labels
    x_axis_label = fr"${axis_parameter[0]}" + r"_{\text{T}}^{\text{det}}\:(\text{GeV}/c)$"
    axes[0, 0].set_ylabel("Subleading correct")
    axes[1, 0].set_ylabel("Subleading in leading")
    axes[2, 0].set_ylabel("Subleading in no prong")
    axes[0, 0].set_title("Leading correct", size=18)
    axes[0, 1].set_title("Leading in subleading", size=18)
    axes[0, 2].set_title("Leading in no prong", size=18)
    axes[2, 0].set_xlabel(x_axis_label)
    axes[2, 1].set_xlabel(x_axis_label)
    axes[2, 2].set_xlabel(x_axis_label)
    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.10,
        bottom=0.08,
        right=0.99,
        top=0.96,
    )

    # Store and reset
    fig.savefig(path / f"subjet_matching_{axis_parameter}_{technique}_{str(identifier)}.pdf")
    plt.close(fig)

    # Labeling
    text = identifier.display_str()
    text += "\n" + hists.title
    ax.text(
        0.975,
        0.55,
        text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="center",
        multialignment="right",
    )

    # Presentation
    # Axis labels
    ax.set_ylabel("Tagging fraction")
    ax.set_xlabel(x_axis_label)
    ax.set_ylim([1e-3, 10])
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="upper left", ncol=2, fontsize=14)
    fig_single.tight_layout()
    fig_single.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.10,
        bottom=0.08,
        right=0.99,
        top=0.96,
    )

    # Store and reset
    fig_single.savefig(path / f"subjet_matching_{axis_parameter}_{technique}_{str(identifier)}_single_figure.pdf")
    plt.close(fig_single)


def _plot_matching_response_pt(
    technique: str,
    identifier: analysis_objects.Identifier,
    hists: analysis_objects.SubstructureMatchingSubjetHists,
    plot_config: PlotConfig,
    path: Path,
) -> None:
    # Setup
    logger.info(f"Plotting det-hybrid jet pt response for {technique}, {identifier}")

    # Define the hybrid range of interest: 40-120 GeV
    identifier = analysis_objects.Identifier(
        iterative_splittings=identifier.iterative_splittings, jet_pt_bin=helpers.RangeSelector(min=40, max=120)
    )

    for matching_type, h in hists:
        # Setup
        # Axes: 0 = matched_jet_pt, 1 = matched_kt, 2 = hybrid_jet_pt, 3 = hybrid_kt
        # Determine the range for hybrid jet pt.
        epsilon = 0.00001
        hybrid_jet_pt_range = slice(
            h.axes[2].find_bin(identifier.jet_pt_bin.min + epsilon),
            h.axes[2].find_bin(identifier.jet_pt_bin.max),
        )
        hybrid_jet_pt_axis_range = slice(
            h.axes[2].find_bin(identifier.jet_pt_bin.min + epsilon),
            h.axes[2].find_bin(identifier.jet_pt_bin.max) + 1,
        )

        # Project into our axes of interest (namely, the attribute at hybrid and true level).
        h_proj = binned_data.BinnedData(
            axes=[h.axes[0], h.axes[2].bin_edges[hybrid_jet_pt_axis_range]],
            values=np.sum(h.values[:, :, hybrid_jet_pt_range, :], axis=(1, 3)),
            variances=np.sum(h.variances[:, :, hybrid_jet_pt_range, :], axis=(1, 3)),
        )

        # If there aren't counts, we  need to stop here.
        if len(h_proj.values[h_proj.values > 0]) == 0:
            logger.warning(f"No values left for {technique}, {identifier}, {matching_type}. Skipping")
            return

        # Rebin the matching axis because the granularity is super high there (much more so than the hybrid)
        rebin_factor = 5
        bh_proj = h_proj.to_boost_histogram()
        bh_proj = bh_proj[:: bh.rebin(rebin_factor), :]
        h_proj = binned_data.BinnedData.from_existing_data(bh_proj)
        h_proj /= rebin_factor

        # Normalize the response.
        normalization_values = h_proj.values.sum(axis=0, keepdims=True)
        h_proj.values = np.divide(
            h_proj.values, normalization_values, out=np.zeros_like(h_proj.values), where=normalization_values != 0
        )

        # Finish setup
        fig, ax = plt.subplots(figsize=(8, 6))

        # Determine the normalization range
        z_axis_range = {
            # "vmin": h_proj.values[h_proj.values > 0].min(),
            "vmin": 1e-4,
            "vmax": h_proj.values.max(),
        }
        if technique == "inclusive":
            z_axis_range = {
                "vmin": 10e-3,
                "vmax": 5,
            }

        # Make the plot
        # NOTE: We have transposed the axis and not transposed the values (as we usually would) because we want hybrid on the x-axis.
        mesh = ax.pcolormesh(
            h_proj.axes[1].bin_edges.T,
            h_proj.axes[0].bin_edges.T,
            h_proj.values,
            norm=matplotlib.colors.LogNorm(**z_axis_range),
        )
        fig.colorbar(mesh, pad=0.02)

        # Labeling
        matches_label = " ".join(matching_type.split("_"))
        text = identifier.display_str(jet_pt_label="hybrid")
        text += "\n" + hists.title
        text += "\n" + f"{matches_label} matches"
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            multialignment="left",
        )

        # Presentation
        ax.set_xlabel(plot_config.x_label)
        ax.set_ylabel(plot_config.y_label)
        fig.tight_layout()
        fig.subplots_adjust(
            # Reduce spacing between subplots
            hspace=0,
            wspace=0,
            # Reduce external spacing
            left=0.10,
            bottom=0.11,
            right=0.99,
            top=0.98,
        )

        # Store and reset
        fig.savefig(
            path / f"matching_response_pt_det_hybrid_{str(identifier)}_{technique}_matchingType_{matching_type}.pdf"
        )
        plt.close(fig)


def _plot_matching_response_kt(
    technique: str,
    identifier: analysis_objects.Identifier,
    hists: analysis_objects.SubstructureMatchingSubjetHists,
    plot_config: PlotConfig,
    path: Path,
) -> None:
    # Setup
    logger.info(f"Plotting det-hybrid jet pt response for {technique}, {identifier}")

    # Define the hybrid range of interest: 40-120 GeV
    identifier = analysis_objects.Identifier(
        iterative_splittings=identifier.iterative_splittings, jet_pt_bin=helpers.RangeSelector(min=40, max=120)
    )

    for matching_type, h in hists:
        # Setup
        # Axes: 0 = matched_jet_pt, 1 = matched_kt, 2 = hybrid_jet_pt, 3 = hybrid_kt
        # Determine the range for hybrid jet pt.
        epsilon = 0.00001
        hybrid_jet_pt_range = slice(
            h.axes[2].find_bin(identifier.jet_pt_bin.min + epsilon),
            h.axes[2].find_bin(identifier.jet_pt_bin.max),
        )
        # hybrid_jet_pt_axis_range = slice(
        #    h.axes[2].find_bin(identifier.jet_pt_bin.min + epsilon), h.axes[2].find_bin(identifier.jet_pt_bin.max) + 1,
        # )

        # Project into our axes of interest (namely, the attribute at hybrid and true level).
        h_proj = binned_data.BinnedData(
            axes=[h.axes[1], h.axes[3]],
            values=np.sum(h.values[:, :, hybrid_jet_pt_range, :], axis=(0, 2)),
            variances=np.sum(h.variances[:, :, hybrid_jet_pt_range, :], axis=(0, 2)),
        )

        # If there aren't counts, we  need to stop here.
        if len(h_proj.values[h_proj.values > 0]) == 0:
            logger.warning(f"No values left for {technique}, {identifier}, {matching_type}. Skipping")
            return

        # Normalize the response.
        normalization_values = h_proj.values.sum(axis=0, keepdims=True)
        h_proj.values = np.divide(
            h_proj.values, normalization_values, out=np.zeros_like(h_proj.values), where=normalization_values != 0
        )

        # Finish setup
        fig, ax = plt.subplots(figsize=(8, 6))

        # Determine the normalization range
        z_axis_range = {
            # "vmin": h_proj.values[h_proj.values > 0].min(),
            "vmin": 1e-4,
            "vmax": h_proj.values.max(),
        }
        if technique == "inclusive":
            z_axis_range = {
                "vmin": 10e-3,
                "vmax": 5,
            }

        # Make the plot
        # NOTE: We have transposed the axis and not transposed the values (as we usually would) because we want hybrid on the x-axis.
        mesh = ax.pcolormesh(
            h_proj.axes[1].bin_edges.T,
            h_proj.axes[0].bin_edges.T,
            h_proj.values,
            norm=matplotlib.colors.LogNorm(**z_axis_range),
        )
        fig.colorbar(mesh, pad=0.02)

        # Labeling
        matches_label = " ".join(matching_type.split("_"))
        text = identifier.display_str(jet_pt_label="hybrid")
        text += "\n" + hists.title
        text += "\n" + f"{matches_label} matches"
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            multialignment="left",
        )

        # Presentation
        ax.set_xlabel(plot_config.x_label)
        ax.set_ylabel(plot_config.y_label)
        fig.tight_layout()
        fig.subplots_adjust(
            # Reduce spacing between subplots
            hspace=0,
            wspace=0,
            # Reduce external spacing
            left=0.10,
            bottom=0.11,
            right=0.99,
            top=0.98,
        )

        # Store and reset
        fig.savefig(
            path / f"matching_response_kt_det_hybrid_{str(identifier)}_{technique}_matchingType_{matching_type}.pdf"
        )
        plt.close(fig)


def matching(
    all_matching_hists: Mapping[
        analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureMatchingSubjetHists]
    ],
    hybrid_min_kt_values: Sequence[float],
    path: Path,
) -> None:
    # Validation
    path.mkdir(parents=True, exist_ok=True)
    # Setup
    # We only care about this set of bins...
    hybrid_jet_pt_bins = [
        helpers.RangeSelector(min=40, max=120),
    ]

    for axis_parameter in ["pt", "kt"]:
        for identifier, matching_hists in all_matching_hists.items():
            for technique, hists in matching_hists:
                for hybrid_jet_pt_bin in hybrid_jet_pt_bins:
                    for hybrid_min_kt in hybrid_min_kt_values:
                        # Use the matching identifier to fully specify our values of interest.
                        matching_identifier = analysis_objects.MatchingHybridIdentifier.from_existing_identifier(
                            identifier, hybrid_jet_pt_bin=hybrid_jet_pt_bin, min_kt=hybrid_min_kt
                        )
                        # Plot matching distributions
                        _plot_matching(
                            technique=technique,
                            identifier=matching_identifier,
                            axis_parameter=axis_parameter,
                            hists=hists,
                            path=path,
                        )

    for identifier, matching_hists in all_matching_hists.items():
        for technique, hists in matching_hists:
            _plot_matching_response_pt(
                technique=technique,
                identifier=identifier,
                hists=hists,
                plot_config=PlotConfig(
                    name="matching_response_pt",
                    x_label=r"$p_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$",
                    y_label=r"$p_{\text{T}}^{\text{det}}\:(\text{GeV}/c)$",
                ),
                path=path,
            )
            _plot_matching_response_kt(
                technique=technique,
                identifier=identifier,
                hists=hists,
                plot_config=PlotConfig(
                    name="matching_response_kt",
                    x_label=r"$k_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$",
                    y_label=r"$k_{\text{T}}^{\text{det}}\:(\text{GeV}/c)$",
                ),
                path=path,
            )


def _plot_toy(
    technique: str,
    identifier: analysis_objects.Identifier,
    attribute_name: str,
    hists: analysis_objects.SubstructureToyHists,
    plot_config: PlotConfig,
    data_label: str,
    path: Path,
) -> None:
    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))
    logger.info(f"Plotting toy hist for {technique}, {identifier}, {attribute_name}")

    h: Union[bh.Histogram, binned_data.BinnedData] = getattr(hists, attribute_name)
    if isinstance(h, bh.Histogram):
        h = binned_data.BinnedData.from_existing_data(h)

    # Scale by bin width
    x_bin_widths, y_bin_widths = np.meshgrid(*h.axes.bin_widths)
    # bin_widths = x_bin_widths * y_bin_widths
    # print(f"x_bin_widths: {x_bin_widths.size}")
    # print(f"y_bin_widths: {y_bin_widths.size}")
    # print(f"bin_widths size: {bin_widths.size}")
    # h /= bin_widths

    # Determine the normalization range
    z_axis_range = {
        "vmin": h.values[h.values > 0].min(),
        "vmax": h.values.max(),
    }
    # z_axis_range = {
    #    "vmin": 1e-2,
    #    "vmax": h.values.max(),
    # }
    # if technique == "inclusive":
    #    z_axis_range = {
    #        "vmin": 10e-3,
    #        "vmax": 5,
    #    }

    # Make the plot
    mesh = ax.pcolormesh(
        h.axes[0].bin_edges.T,
        h.axes[1].bin_edges.T,
        h.values.T,
        norm=matplotlib.colors.LogNorm(**z_axis_range),
    )
    fig.colorbar(mesh, pad=0.02)

    # Labeling
    text = identifier.display_str(jet_pt_label=data_label)
    text += "\n" + hists.title
    ax.text(
        0.05,
        0.05,
        text,
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="bottom",
        multialignment="left",
    )

    # Presentation
    ax.set_xlabel(plot_config.x_label)
    ax.set_ylabel(plot_config.y_label)
    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.10,
        bottom=0.11,
        right=0.99,
        top=0.98,
    )

    # Store and reset
    fig.savefig(path / f"{attribute_name}_{str(identifier)}_{technique}.pdf")
    plt.close(fig)


def toy(
    all_toy_hists: Mapping[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureToyHists]],
    data_prefix: str,
    path: Path,
) -> None:
    # Validation
    path.mkdir(parents=True, exist_ok=True)

    # Plot labels
    label_map: Dict[str, str] = {
        "hybrid": "hybrid",
        "pythia": "pythia",
    }
    label = label_map[data_prefix]
    kt_label = PlotConfig(
        name="kt",
        x_label=r"$\log{k_{\text{T}}^{\text{hardest graph}}\:(\text{GeV}/c)}$",
        y_label=r"$\log{k_{\text{T}}^{\text{" + label + r"}}\:(\text{GeV}/c)}$",
    )
    z_label = PlotConfig(
        name="z",
        x_label=r"$z^{\text{hardest graph}}\:(\text{GeV}/c)$",
        y_label=r"$z^{\text{" + label + r"}}\:(\text{GeV}/c)$",
    )
    delta_R_label = PlotConfig(
        name="delta_R",
        x_label=r"$R^{\text{hardest graph}}\:(\text{GeV}/c)$",
        y_label=r"$R^{\text{" + label + r"}}\:(\text{GeV}/c)$",
    )
    theta_label = PlotConfig(
        name="theta",
        x_label=r"$\theta^{\text{hardest graph}}\:(\text{GeV}/c)$",
        y_label=r"$\theta^{\text{" + label + r"}}\:(\text{GeV}/c)$",
    )

    distributions: List[Tuple[str, PlotConfig]] = [
        ("kt", kt_label),
        ("z", z_label),
        ("delta_R", delta_R_label),
        ("theta", theta_label),
    ]

    for identifier, toy_hists in all_toy_hists.items():
        for technique, hists in toy_hists:
            if hists.n_jets == 0:
                logger.warning(f"No jets within {identifier}_{technique}. Skipping bin!")
                continue
            for attribute_name, plot_config in distributions:
                # Plot toy distributions
                _plot_toy(
                    technique=technique,
                    identifier=identifier,
                    attribute_name=attribute_name,
                    hists=hists,
                    plot_config=plot_config,
                    data_label=label,
                    path=path,
                )


def _plot_response(
    technique: str,
    identifier: analysis_objects.Identifier,
    attribute_name: str,
    hists: analysis_objects.SubstructureResponseHists,
    plot_config: PlotConfig,
    path: Path,
) -> None:
    # Setup
    logger.info(f"Plotting response hist for {technique}, {identifier}, {attribute_name}")
    h: Union[bh.Histogram, binned_data.BinnedData] = getattr(hists, f"response_{attribute_name}")
    if isinstance(h, bh.Histogram):
        h = binned_data.BinnedData.from_existing_data(h)

    # Define the hybrid range of interest: 40-120 GeV
    identifier = analysis_objects.Identifier(
        iterative_splittings=identifier.iterative_splittings, jet_pt_bin=helpers.RangeSelector(min=40, max=120)
    )
    # Use this to select the right hybrid range.
    # epsilon = 0.00001
    # hybrid_jet_pt_range = slice(
    #    h.axes[0].find_bin(identifier.jet_pt_bin.min + epsilon),
    #    h.axes[0].find_bin(identifier.jet_pt_bin.max),
    # )
    response_with_matching_types = binned_data.BinnedData(
        axes=[h.axes[1], h.axes[3], h.axes[4]],
        values=np.sum(h.values[:, :, :, :, :], axis=(0, 2)),
        variances=np.sum(h.variances[:, :, :, :, :], axis=(0, 2)),
    )

    # response_with_matching_types = binned_data.BinnedData(
    #    axes=[h.axes[1], h.axes[3], h.axes[4]], values=np.sum(h.values, axis=(0, 2)), variances=np.sum(h.variances, axis=(0, 2))
    # )

    for matching_type, matching_type_value in hists.matching_name_to_axis_value.items():
        # import IPython; IPython.embed()
        h_proj = binned_data.BinnedData(
            axes=[response_with_matching_types.axes[0], response_with_matching_types.axes[1]],
            values=response_with_matching_types.values[:, :, matching_type_value],
            variances=response_with_matching_types.variances[:, :, matching_type_value],
        )

        # Project into our axes of interest (namely, the attribute at hybrid and true level).
        # h_proj = binned_data.BinnedData(
        #    axes=[h.axes[1], h.axes[3]], values=np.sum(h.values, axis=(0, 2)), variances=np.sum(h.variances, axis=(0, 2)),
        # )

        # If there aren't counts, we  need to stop here.
        if len(h_proj.values[h_proj.values > 0]) == 0:
            logger.warning(
                f"No values left for {technique}, {identifier}, {attribute_name}, matching_type {matching_type}. Skipping"
            )
            return

        # Normalize the response.
        normalization_values = h_proj.values.sum(axis=0, keepdims=True)
        h_proj.values = np.divide(
            h_proj.values, normalization_values, out=np.zeros_like(h_proj.values), where=normalization_values != 0
        )

        # Finish setup
        fig, ax = plt.subplots(figsize=(8, 6))

        # Determine the normalization range
        z_axis_range = {
            # "vmin": h_proj.values[h_proj.values > 0].min(),
            "vmin": 1e-4,
            "vmax": h_proj.values.max(),
        }
        if technique == "inclusive":
            z_axis_range = {
                "vmin": 10e-3,
                "vmax": 5,
            }

        # Make the plot
        mesh = ax.pcolormesh(
            h_proj.axes[0].bin_edges.T,
            h_proj.axes[1].bin_edges.T,
            h_proj.values.T,
            norm=matplotlib.colors.LogNorm(**z_axis_range),
        )
        fig.colorbar(mesh, pad=0.02)

        # Labeling
        matches_label = " ".join(matching_type.split("_"))
        text = identifier.display_str(jet_pt_label="hybrid")
        text += "\n" + hists.title
        text += "\n" + f"{matches_label} matches"
        ax.text(
            0.05,
            0.95,
            text,
            transform=ax.transAxes,
            horizontalalignment="left",
            verticalalignment="top",
            multialignment="left",
        )

        # Presentation
        ax.set_xlabel(plot_config.x_label)
        ax.set_ylabel(plot_config.y_label)
        fig.tight_layout()
        fig.subplots_adjust(
            # Reduce spacing between subplots
            hspace=0,
            wspace=0,
            # Reduce external spacing
            left=0.10,
            bottom=0.11,
            right=0.99,
            top=0.98,
        )

        # Store and reset
        fig.savefig(path / f"response_{attribute_name}_{str(identifier)}_{technique}_matchingType_{matching_type}.pdf")
        plt.close(fig)


def _plot_response_pt(
    technique: str,
    identifier: analysis_objects.Identifier,
    attribute_name: str,
    hists: analysis_objects.SubstructureResponseHists,
    plot_config: PlotConfig,
    path: Path,
) -> None:
    # Setup
    logger.info(f"Plotting jet pt response hist for {technique}, {identifier}, {attribute_name}")
    h: Union[bh.Histogram, binned_data.BinnedData] = getattr(hists, f"response_{attribute_name}")
    if isinstance(h, bh.Histogram):
        h = binned_data.BinnedData.from_existing_data(h)

    # Define the hybrid range of interest: 40-120 GeV
    identifier = analysis_objects.Identifier(
        iterative_splittings=identifier.iterative_splittings, jet_pt_bin=helpers.RangeSelector(min=40, max=120)
    )
    # Use this to select the right hybrid range.
    # epsilon = 0.00001
    # hybrid_jet_pt_range = slice(
    #    h.axes[0].find_bin(identifier.jet_pt_bin.min + epsilon),
    #    h.axes[0].find_bin(identifier.jet_pt_bin.max),
    # )

    # import IPython; IPython.embed()
    # Project into our axes of interest (namely, the attribute at hybrid and true level).
    h_proj = binned_data.BinnedData(
        axes=[h.axes[0], h.axes[2]],
        values=np.sum(h.values[:, :, :, :, :], axis=(1, 3, 4)),
        variances=np.sum(h.variances[:, :, :, :, :], axis=(1, 3, 4)),
    )

    # If there aren't counts, we  need to stop here.
    if len(h_proj.values[h_proj.values > 0]) == 0:
        logger.warning(f"No values left for {technique}, {identifier}, {attribute_name}. Skipping")
        return

    # Normalize the response.
    normalization_values = h_proj.values.sum(axis=0, keepdims=True)
    h_proj.values = np.divide(
        h_proj.values, normalization_values, out=np.zeros_like(h_proj.values), where=normalization_values != 0
    )

    # Finish setup
    fig, ax = plt.subplots(figsize=(8, 6))

    # Determine the normalization range
    z_axis_range = {
        # "vmin": h_proj.values[h_proj.values > 0].min(),
        "vmin": 1e-4,
        "vmax": h_proj.values.max(),
    }
    if technique == "inclusive":
        z_axis_range = {
            "vmin": 10e-3,
            "vmax": 5,
        }

    # Make the plot
    mesh = ax.pcolormesh(
        h.axes[0].bin_edges.T,
        h.axes[2].bin_edges.T,
        h_proj.values.T,
        norm=matplotlib.colors.LogNorm(**z_axis_range),
    )
    fig.colorbar(mesh, pad=0.02)

    # Labeling
    text = identifier.display_str(jet_pt_label="hybrid")
    text += "\n" + hists.title
    ax.text(
        0.05,
        0.95,
        text,
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        multialignment="left",
    )

    # Presentation
    ax.set_xlabel(plot_config.x_label)
    ax.set_ylabel(plot_config.y_label)
    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.10,
        bottom=0.11,
        right=0.99,
        top=0.98,
    )

    # Store and reset
    fig.savefig(path / f"response_pt_{attribute_name}_{str(identifier)}_{technique}.pdf")
    plt.close(fig)


def _plot_response_jet_spectra(
    technique: str,
    identifier: analysis_objects.Identifier,
    attribute_name: str,
    hists: analysis_objects.SubstructureResponseHists,
    plot_config: PlotConfig,
    path: Path,
) -> None:
    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))
    logger.info(f"Plotting response jet spectra for {technique}, {identifier}, {attribute_name}")

    h: Union[bh.Histogram, binned_data.BinnedData] = getattr(hists, f"response_{attribute_name}")
    if isinstance(h, bh.Histogram):
        h = binned_data.BinnedData.from_existing_data(h)

    # Project into our axes of interest (namely, the spectra at hybrid and true level).
    # Now get the individual spectra.
    h_part = binned_data.BinnedData(
        axes=h.axes[2], values=np.sum(h.values, axis=(0, 1, 3)), variances=np.sum(h.variances, axis=(0, 1, 3))
    )
    h_hybrid = binned_data.BinnedData(
        axes=h.axes[0], values=np.sum(h.values, axis=(1, 2, 3)), variances=np.sum(h.variances, axis=(1, 2, 3))
    )

    # Normalization
    # Scale by bin widths and number of jets
    h_part /= hists.generator_like_n_jets
    h_part /= h_part.axes[0].bin_widths
    h_hybrid /= hists.measured_like_n_jets
    h_hybrid /= h_hybrid.axes[0].bin_widths

    # Plot
    ax.errorbar(
        h_part.axes[0].bin_centers,
        h_part.values,
        yerr=h_part.errors,
        xerr=h_part.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        label="Particle level",
    )
    # Plot
    ax.errorbar(
        h_hybrid.axes[0].bin_centers,
        h_hybrid.values,
        yerr=h_hybrid.errors,
        xerr=h_hybrid.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        label="Hybrid level",
    )

    # Labeling
    text = identifier.display_str(jet_pt_label="hybrid")
    text += "\n" + hists.title
    ax.text(
        0.95,
        0.95,
        text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
    )

    # Presentation
    ax.set_xlabel(plot_config.x_label)
    ax.set_ylabel(plot_config.y_label)
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.12,
        bottom=0.11,
        right=0.99,
        top=0.98,
    )

    # Store and reset
    fig.savefig(path / f"response_spectra_{attribute_name}_{str(identifier)}_{technique}.pdf")
    plt.close(fig)


def _plot_response_kt_distributions(
    technique: str,
    identifier: analysis_objects.Identifier,
    attribute_name: str,
    hists: analysis_objects.SubstructureResponseHists,
    plot_config: PlotConfig,
    path: Path,
) -> None:
    # Setup
    fig, ax = plt.subplots(figsize=(8, 6))
    logger.info(f"Plotting response kt spectra for {technique}, {identifier}, {attribute_name}")

    h: Union[bh.Histogram, binned_data.BinnedData] = getattr(hists, f"response_{attribute_name}")
    if isinstance(h, bh.Histogram):
        h = binned_data.BinnedData.from_existing_data(h)

    # Project into our axes of interest (namely, the spectra at hybrid and true level).
    # Now get the individual spectra.
    h_part = binned_data.BinnedData(
        axes=h.axes[3], values=np.sum(h.values, axis=(0, 1, 2, 4)), variances=np.sum(h.variances, axis=(0, 1, 2, 4))
    )
    h_hybrid = binned_data.BinnedData(
        axes=h.axes[1], values=np.sum(h.values, axis=(0, 2, 3, 4)), variances=np.sum(h.variances, axis=(0, 2, 3, 4))
    )

    # Normalization
    # Scale by bin widths and number of jets
    h_part /= hists.generator_like_n_jets
    h_part /= h_part.axes[0].bin_widths
    h_hybrid /= hists.measured_like_n_jets
    h_hybrid /= h_hybrid.axes[0].bin_widths

    # Plot
    ax.errorbar(
        h_part.axes[0].bin_centers,
        h_part.values,
        yerr=h_part.errors,
        xerr=h_part.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        label="Particle level",
    )
    # Plot
    ax.errorbar(
        h_hybrid.axes[0].bin_centers,
        h_hybrid.values,
        yerr=h_hybrid.errors,
        xerr=h_hybrid.axes[0].bin_widths / 2,
        marker=".",
        linestyle="",
        label="Hybrid level",
    )

    # Labeling
    text = identifier.display_str(jet_pt_label="hybrid")
    text += "\n" + hists.title
    ax.text(
        0.95,
        0.95,
        text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
    )

    # Presentation
    ax.set_xlabel(plot_config.x_label)
    ax.set_ylabel(plot_config.y_label)
    ax.legend(frameon=False, loc="lower left")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.14,
        bottom=0.11,
        right=0.99,
        top=0.98,
    )

    # Store and reset
    fig.savefig(path / f"response_kt_spectra_{attribute_name}_{str(identifier)}_{technique}.pdf")
    plt.close(fig)


def responses(
    all_response_hists: Mapping[
        analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureResponseHists]
    ],
    path: Path,
) -> None:
    # Validation
    path.mkdir(parents=True, exist_ok=True)

    kt_label = PlotConfig(
        name="kt",
        x_label=r"$k_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$",
        y_label=r"$k_{\text{T}}^{\text{part}}\:(\text{GeV}/c)$",
    )
    z_label = PlotConfig(
        name="z",
        x_label=r"$z^{\text{hybrid}}$",
        y_label=r"$z^{\text{part}}$",
    )
    delta_R_label = PlotConfig(
        name="delta_R",
        x_label=r"$R^{\text{hybrid}}$",
        y_label=r"$R^{\text{part}}$",
    )
    theta_label = PlotConfig(
        name="theta",
        x_label=r"$\theta^{\text{hybrid}}$",
        y_label=r"$\theta^{\text{part}}$",
    )
    splitting_number_label = PlotConfig(
        name="splitting_number",
        x_label=r"$n^{\text{hybrid}}$",
        y_label=r"$n^{\text{part}}$",
    )

    distributions: List[Tuple[str, PlotConfig]] = [
        ("kt", kt_label),
        ("z", z_label),
        ("delta_R", delta_R_label),
        ("theta", theta_label),
        ("splitting_number", splitting_number_label),
    ]

    for identifier, response_hists in all_response_hists.items():
        for technique, hists in response_hists:
            if hists.measured_like_n_jets == 0 or hists.generator_like_n_jets == 0:
                logger.warning(f"No jets within {identifier}_{technique}. Skipping bin!")
                continue
            for attribute_name, plot_config in distributions:
                # Plot response matrices
                _plot_response(
                    technique=technique,
                    identifier=identifier,
                    attribute_name=attribute_name,
                    hists=hists,
                    plot_config=plot_config,
                    path=path,
                )
            # Only plot once. It's redundant otherwise
            if identifier.jet_pt_bin in [
                # TODO: Restore this range to 40-120
                helpers.RangeSelector(min=20, max=140),
                helpers.RangeSelector(min=80, max=120),
            ]:
                _plot_response_jet_spectra(
                    technique=technique,
                    identifier=identifier,
                    attribute_name="kt",
                    hists=hists,
                    plot_config=PlotConfig(
                        name="response_spectra",
                        x_label=r"$p_{\text{T}}\:(\text{GeV}/c)$",
                        y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                    ),
                    path=path,
                )
                _plot_response_kt_distributions(
                    technique=technique,
                    identifier=identifier,
                    attribute_name="kt",
                    hists=hists,
                    plot_config=PlotConfig(
                        name="response_spectra",
                        x_label=r"$k_{\text{T}}\:(\text{GeV}/c)$",
                        y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                    ),
                    path=path,
                )
                _plot_response_pt(
                    technique=technique,
                    identifier=identifier,
                    attribute_name="kt",
                    hists=hists,
                    plot_config=PlotConfig(
                        name="response_pt",
                        x_label=r"$p_{\text{T}}^{\text{hybrid}}\:(\text{GeV}/c)$",
                        y_label=r"$p_{\text{T}}^{\text{part}}\:(\text{GeV}/c)$",
                    ),
                    path=path,
                )


def _plot_compare_kt(
    technique: str,
    identifier: analysis_objects.Identifier,
    data_hists: analysis_objects.SubstructureHists,
    embedded_hists: analysis_objects.SubstructureHists,
    det_hists: analysis_objects.SubstructureHists,
    true_hists: analysis_objects.SubstructureHists,
    plot_config: PlotConfig,
    path: Path,
) -> None:
    # Setup
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    logger.info(f"Plotting response kt spectra for {technique}, {identifier}, kt")

    h_data = binned_data.BinnedData.from_existing_data(data_hists.kt)
    h_embed = binned_data.BinnedData.from_existing_data(embedded_hists.kt)
    h_det = binned_data.BinnedData.from_existing_data(det_hists.kt)
    h_true = binned_data.BinnedData.from_existing_data(true_hists.kt)

    # Convert to boost_histogram and rebin (and scale by rebin factor to maintain the normalization).
    rebin_factor = 2
    # Data
    bh_data = h_data.to_boost_histogram()
    bh_data = bh_data[:: bh.rebin(rebin_factor)]
    h_data = binned_data.BinnedData.from_existing_data(bh_data)
    h_data /= rebin_factor
    # Embed
    bh_embed = h_embed.to_boost_histogram()
    bh_embed = bh_embed[:: bh.rebin(rebin_factor)]
    h_embed = binned_data.BinnedData.from_existing_data(bh_embed)
    h_embed /= rebin_factor
    # Det
    bh_det = h_det.to_boost_histogram()
    bh_det = bh_det[:: bh.rebin(rebin_factor)]
    h_det = binned_data.BinnedData.from_existing_data(bh_det)
    h_det /= rebin_factor
    # True
    bh_true = h_true.to_boost_histogram()
    bh_true = bh_true[:: bh.rebin(rebin_factor)]
    h_true = binned_data.BinnedData.from_existing_data(bh_true)
    h_true /= rebin_factor

    # Normalization
    # Scale by bin widths and number of jets
    # Data
    h_data /= data_hists.n_jets
    h_data /= h_data.axes[0].bin_widths
    # Embed
    h_embed /= embedded_hists.n_jets
    h_embed /= h_embed.axes[0].bin_widths
    # Det
    h_det /= det_hists.n_jets
    h_det /= h_det.axes[0].bin_widths
    # True
    h_true /= true_hists.n_jets
    h_true /= h_true.axes[0].bin_widths

    # Plot
    # for h, label in [(h_data, "Data"), (h_embed, "Hybrid"), (h_det, "Det"), (h_true, "True")]:
    for h, label in [(h_data, "Data"), (h_embed, "Hybrid")]:
        p = ax.errorbar(
            h.axes[0].bin_centers,
            h.values,
            yerr=h.errors,
            xerr=h.axes[0].bin_widths / 2,
            marker=".",
            linestyle="",
            label=label,
        )

        ## Fit to power law.
        # fit = analysis_methods.fit_kt_spectrum(h)
        # fit_hist = binned_data.BinnedData(
        #    axes = h.axes[0].bin_edges,
        #    values = fit(fit.fit_result.x),
        #    variances = fit.fit_result.errors ** 2,
        # )

        ## Complete lael with fit info:
        # label = label + fr": $kt^{{({fit.fit_result.values_at_minimum['p']:.02} \pm {fit.fit_result.errors_on_parameters['p']:.02})}}$"

        # plot = ax.plot(fit_hist.axes[0].bin_centers, fit_hist.values, label = label)
        ## Plot the fit errors.
        ## We need to ensure that the errors are copied so we don't accidentally modify the fit result.
        # ax.fill_between(
        #    fit_hist.axes[0].bin_centers,
        #    fit_hist.values - fit_hist.errors, fit_hist.values + fit_hist.errors,
        #    facecolor = plot[0].get_color(), alpha = 0.8
        # )

        if label != "Data":
            h_ratio = h_data / h
            # If we get 0, we don't want to show that point.
            h_ratio.values[h_ratio.values == 0] = np.nan

            # Plot the ratio
            ax_ratio.errorbar(
                h_ratio.axes[0].bin_centers,
                h_ratio.values,
                yerr=h_ratio.errors,
                xerr=h_ratio.axes[0].bin_widths / 2,
                marker=".",
                linestyle="",
                color=p[0].get_color(),
            )

    # Reference value
    ax_ratio.axhline(y=1, color="black", linestyle="dashed", zorder=1)

    # Labeling
    text = identifier.display_str(jet_pt_label="hybrid")
    text += "\n" + data_hists.title
    ax.text(
        0.95,
        0.95,
        text,
        transform=ax.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        multialignment="right",
    )

    # Presentation
    ax_ratio.set_xlabel(plot_config.x_label)
    ax.set_ylabel(plot_config.y_label)
    ax_ratio.set_ylabel("PbPb/Ref")
    # As standard for a ratio.
    ax_ratio.set_ylim([0, 5])
    # Rest of labeling.
    ax.legend(frameon=False, loc="lower left")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.subplots_adjust(
        # Reduce spacing between subplots
        hspace=0,
        wspace=0,
        # Reduce external spacing
        left=0.14,
        bottom=0.11,
        right=0.99,
        top=0.98,
    )
    fig.align_ylabels()

    # Store and reset
    fig.savefig(path / f"kt_spectra_comparison_{str(identifier)}_{technique}.pdf")
    plt.close(fig)


def compare_kt(
    all_data_hists: Mapping[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]],
    all_embedded_hists: Mapping[
        analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]
    ],
    all_det_hists: Mapping[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]],
    all_true_hists: Mapping[analysis_objects.Identifier, analysis_objects.Hists[analysis_objects.SubstructureHists]],
    data_dataset: analysis_objects.Dataset,
    embedded_dataset: analysis_objects.Dataset,
) -> None:
    for (
        (data_identifier, data_hists),
        (embedded_identifier, embedded_hists),
        (det_identifier, det_hists),
        (true_identifier, true_hists),
    ) in zip(all_data_hists.items(), all_embedded_hists.items(), all_det_hists.items(), all_true_hists.items()):
        for (technique, d_hists), (_, e_hists), (_, pythia_d_hists), (_, t_hists) in zip(
            data_hists, embedded_hists, det_hists, true_hists
        ):
            if d_hists.n_jets == 0 or e_hists.n_jets == 0 or pythia_d_hists.n_jets == 0 or t_hists.n_jets == 0:
                logger.warning(f"No jets within {data_identifier}_{technique}. Skipping bin!")
                logger.warning(
                    f"data n_jets: {d_hists.n_jets}, hybrid n_jets: {e_hists.n_jets}, det n_jets: {pythia_d_hists.n_jets}, true n_jets: {t_hists.n_jets}"
                )
                continue

            _plot_compare_kt(
                technique=technique,
                identifier=data_identifier,
                data_hists=d_hists,
                embedded_hists=e_hists,
                det_hists=pythia_d_hists,
                true_hists=t_hists,
                plot_config=PlotConfig(
                    name="kt_spectra",
                    x_label=r"$k_{\text{T}}\:(\text{GeV}/c)$",
                    y_label=r"$1/N_{\text{jets}}\:\text{d}N/\text{d}k_{\text{T}}\:(\text{GeV}/c)^{-1}$",
                ),
                path=data_dataset.output,
            )
