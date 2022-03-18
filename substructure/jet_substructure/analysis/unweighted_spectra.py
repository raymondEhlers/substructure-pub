#!/usr/bin/env python

import functools
import gzip
import pickle
from pathlib import Path
from typing import Iterator, Mapping, Tuple, Type

import boost_histogram as bh
import matplotlib.pyplot as plt
import pachyderm.plot
import seaborn as sns
from pachyderm import binned_data

from jet_substructure.analysis import analyze_tree
from jet_substructure.base import analysis_objects, helpers


pachyderm.plot.configure()


def rebin(h: binned_data.BinnedData, rebin_factor: int) -> binned_data.BinnedData:
    bh_hist = h.to_boost_histogram()
    bh_hist = bh_hist[:: bh.rebin(rebin_factor)]
    h_return = binned_data.BinnedData.from_existing_data(bh_hist)
    h_return /= rebin_factor
    return h_return


def get_hists_for_train_number(
    dataset: analysis_objects.Dataset, train_number: int
) -> Iterator[analysis_objects.SingleTreeEmbeddingResult]:
    for pkl_filename in dataset.hists_filename.parent.glob(f"{train_number}_*_{dataset.hists_filename.name}"):
        # May be useful later, so we'll just keep commented for now.
        # pt_hard_bin = dataset.settings.train_number_to_pt_hard_bin[int(train_number)]
        # scale_factor = dataset.settings.scale_factors[pt_hard_bin]

        # If the output file already exist, skip processing the tree and just return the hists instead (which is way faster!)
        if pkl_filename.exists():
            print(f"Loading pkl file {pkl_filename}")
            with gzip.GzipFile(pkl_filename, "r") as pkl_file:
                result = analysis_objects.SingleTreeEmbeddingResult(**pickle.load(pkl_file))  # type: ignore

        yield result


def get_hists(
    dataset: analysis_objects.Dataset, identifier: analysis_objects.Identifier
) -> Iterator[
    Tuple[
        int,
        analysis_objects.Hists[analysis_objects.SubstructureHists],
        analysis_objects.Hists[analysis_objects.SubstructureHists],
    ]
]:
    for train_number in range(5517, 5537):
        print(f"Processing train number {train_number}")
        # Merge all hists from a given train together.
        result = functools.reduce(analyze_tree.merge_results, get_hists_for_train_number(dataset, train_number))

        # We only want to take the true and hybrid hists so we don't consume too much memory while keeping them all available.
        yield train_number, result.true_hists[identifier], result.hybrid_hists[identifier]


def scaling_per_pt_hard_bin() -> None:
    settings_class_map: Mapping[str, Type[analysis_objects.AnalysisSettings]] = {
        "embedPythia": analysis_objects.PtHardAnalysisSettings,
    }
    dataset = analysis_objects.Dataset.from_config_file(
        collision_system="embedPythia",
        config_filename=Path("config") / "datasets.yaml",
        override_filenames=None,
        hists_filename_stem="embedding_hists",
        output_base=Path("output"),
        settings_class=settings_class_map.get("embedPythia", analysis_objects.AnalysisSettings),
        z_cutoff=0.2,
    )

    # results_true: List[analysis_objects.Hists[analysis_objects.SubstructureHists]] = []
    # results_hybrid: List[analysis_objects.Hists[analysis_objects.SubstructureHists]] = []
    identifier = analysis_objects.Identifier(
        iterative_splittings=True, jet_pt_bin=helpers.RangeSelector(min=20, max=140)
    )
    rebin_factor = 2
    with sns.color_palette("GnBu_d", n_colors=20):
        # Need to be inside of the with statement to have the color settings applied
        fig_true, ax_true = plt.subplots(figsize=(8, 6))
        fig_hybrid, ax_hybrid = plt.subplots(figsize=(8, 6))
        for pt_hard_bin, (train_number, true_hists, hybrid_hists) in enumerate(get_hists(dataset, identifier), start=1):
            # results_true.append(true_hists)
            # results_hybrid.append(hybrid_hists)
            # pt_hard_bin = dataset.settings.train_number_to_pt_hard_bin[int(train_number)]
            # scale_factor = dataset.settings.scale_factors[pt_hard_bin]

            # Plot each contribution...
            # For now, just use leading kt
            true_jet_pt = binned_data.BinnedData.from_existing_data(true_hists.leading_kt.jet_pt)
            # Rebin
            true_jet_pt = rebin(true_jet_pt, rebin_factor)
            # true_jet_pt /= (true_hists.leading_kt.n_jets / scale_factor)
            true_jet_pt /= true_jet_pt.axes[0].bin_widths
            ax_true.errorbar(
                true_jet_pt.axes[0].bin_centers,
                true_jet_pt.values,
                yerr=true_jet_pt.variances,
                marker=".",
                linestyle="",
                label=fr"$p_{{\text{{T}}}}^{{\text{{hard}}}} = {pt_hard_bin}$",
            )

            # Same for hybrid
            hybrid_jet_pt = binned_data.BinnedData.from_existing_data(hybrid_hists.leading_kt.jet_pt)
            # Rebin
            hybrid_jet_pt = rebin(hybrid_jet_pt, rebin_factor)
            # hybrid_jet_pt /= (hybrid_hists.leading_kt.n_jets / scale_factor)
            hybrid_jet_pt /= hybrid_jet_pt.axes[0].bin_widths
            ax_hybrid.errorbar(
                hybrid_jet_pt.axes[0].bin_centers,
                hybrid_jet_pt.values,
                yerr=hybrid_jet_pt.variances,
                marker=".",
                linestyle="",
                label=fr"$p_{{\text{{T}}}}^{{\text{{hard}}}} = {pt_hard_bin}$",
            )

    # Plot merged true
    with gzip.GzipFile(Path("output/embedPythia/LHC19f4_embedded_into_LHC18qr/embedding_hists.pgz"), "r") as pkl_file:
        merged_result = pickle.load(pkl_file)  # type: ignore
    merged_true_hists = merged_result.true_hists[identifier]
    merged_true_jet_pt = merged_true_hists.leading_kt.jet_pt
    # Rebin
    merged_true_jet_pt = rebin(merged_true_jet_pt, rebin_factor)
    # merged_true_jet_pt /= true_hists.leading_kt.n_jets
    merged_true_jet_pt /= merged_true_jet_pt.axes[0].bin_widths
    ax_true.errorbar(
        merged_true_jet_pt.axes[0].bin_centers,
        merged_true_jet_pt.values,
        yerr=merged_true_jet_pt.variances,
        marker=".",
        linestyle="",
        label=r"Merged",
        color="black",
    )

    # Plot merged hybrid
    merged_hybrid_hists = merged_result.hybrid_hists[identifier]
    merged_hybrid_jet_pt = merged_hybrid_hists.leading_kt.jet_pt
    # Rebin
    merged_hybrid_jet_pt = rebin(merged_hybrid_jet_pt, rebin_factor)
    # merged_hybrid_jet_pt /= hybrid_hists.leading_kt.n_jets
    merged_hybrid_jet_pt /= merged_hybrid_jet_pt.axes[0].bin_widths
    ax_hybrid.errorbar(
        merged_hybrid_jet_pt.axes[0].bin_centers,
        merged_hybrid_jet_pt.values,
        yerr=merged_hybrid_jet_pt.variances,
        marker=".",
        linestyle="",
        label=r"Merged",
        color="black",
    )

    # Final styling, labeling, etc.
    for fig, ax, label in [(fig_true, ax_true, "true"), (fig_hybrid, ax_hybrid, "hybrid")]:
        # Labeling
        text = identifier.display_str(jet_pt_label="hybrid")
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
        ax.set_xlabel(r"$p_{\text{T}}\:(\text{GeV}/c)$")
        ax.set_ylabel(r"$\text{d}N/\text{d}p_{\text{T}}\:(\text{GeV}/c)^{-1}$")
        ax.set_yscale("log")
        ax.legend(frameon=False, loc="lower left", fontsize=10)
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
        fig.savefig(dataset.output / f"jet_spectra_{label}_{str(identifier)}.pdf")
        plt.close(fig)


if __name__ == "__main__":
    scaling_per_pt_hard_bin()
