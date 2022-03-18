#!/usr/bin/env python3

""" Attempt analysis just using data frames.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import argparse
import functools
import gzip
import itertools
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import attr
import boost_histogram as bh
import dill
import enlighten
import IPython
import numpy as np
import pandas as pd
import uproot3

from jet_substructure.base import analysis_objects, helpers, skim_analysis_objects


logger = logging.getLogger(__name__)


_matching_name_to_axis_value: Dict[str, int] = {
    "all": 0,
    "pure": 1,
    "leading_untagged_subleading_correct": 2,
    "leading_correct_subleading_untagged": 3,
    "leading_untagged_subleading_mistag": 4,
    "leading_mistag_subleading_untagged": 5,
    "swap": 6,
    "both_untagged": 7,
}

## NOTE: Order is changed here to match from before!!
# _matching_name_to_axis_value: Dict[str, int] = {
#    "all": 0,
#    "pure": 1,
#    "leading_untagged_subleading_correct": 2,
#    "swap": 6,
#    "leading_untagged_subleading_mistag": 4,
#    "leading_correct_subleading_untagged": 3,
#    "leading_mistag_subleading_untagged": 5,
#    "both_untagged": 7,
# }


def _set_output_filename(instance: "SkimDataset", attribute: attr.Attribute[str], value: str) -> None:
    if value == "":
        value = instance.collision_system
    if instance.prefix:
        value = f"{value}_{instance.prefix}"
    setattr(instance, attribute.name, value)


@attr.s
class SkimDataset:
    collision_system: str = attr.ib()
    train_numbers: List[int] = attr.ib()
    prefix: str = attr.ib(default="")
    hists: Dict[str, bh.Histogram] = attr.ib(factory=dict)
    _output_filename_identifier: str = attr.ib(default="", validator=_set_output_filename)
    merged_skim: bool = attr.ib(default=False)
    _path_list: List[Path] = attr.ib(factory=list)

    @property
    def path_list(self) -> List[Path]:
        if self._path_list:
            input_path_list = self._path_list
        else:
            # base_path = Path("trains") / self.collision_system / "{train_number}" / "skim"
            base_path = Path("trains") / self.collision_system / "{train_number}"
            if self.merged_skim:
                base_path = base_path / "merged" / "*.root"
            else:
                base_path = base_path / "AnalysisResults*.root"
                # base_path = base_path / "*_iterative_splittings*.root"
            input_path_list = [
                Path(str(base_path).format(train_number=train_number)) for train_number in self.train_numbers
            ]
        # logger.debug(f"input_path_list: {input_path_list}")
        path_list = helpers.ensure_and_expand_paths(
            input_path_list,
            # [
            #    #Path("trains/embedPythia/5903/skim/merged/*.root")
            #    Path("trains/embedPythia/5903/skim/merged/AnalysisResults.merged.01.root")
            # ]
        )
        return path_list

    @property
    def output_path(self) -> Path:
        base = Path("output") / self.collision_system / "skim"
        if len(self.train_numbers) == 1:
            base = base / str(self.train_numbers[0])
        return base

    @property
    def output_filename(self) -> Path:
        return self.output_path / f"{self._output_filename_identifier}.pgz"

    def load_hists(self) -> bool:
        with gzip.GzipFile(self.output_filename, "r") as f:
            self.hists = dill.load(f)
        return True


def _merge_hists(a: Dict[str, bh.Histogram], b: Dict[str, bh.Histogram]) -> Dict[str, bh.Histogram]:
    """Merge hists stored in a file."""
    for k in b:
        a[k] += b[k]
    return a


def dask_df_from_file() -> None:
    df = uproot3.tree.daskframe(
        path=[
            "temp_cache/embedPythia/55*/skim/*_iterative_splittings.root",
            "trains/embedPythia/55*/skim/*_iterative_splittings.root",
        ],
        treepath="tree",
        namedecode="utf-8",
        branches=["scale_factor", "*det_level*", "*hybrid*"],
    )

    IPython.start_ipython(user_ns=locals())


def dask_df_from_delayed() -> None:
    # From: https://stackoverflow.com/q/60189433/12907985
    import dask.dataframe as dd
    from dask import delayed

    @delayed  # type: ignore
    def get_df(file: Path, treepath: str, branches: Sequence[str]) -> pd.DataFrame:
        tree = uproot3.open(file)[treepath]
        return tree.pandas.df(branches=branches)

    path_list = helpers.ensure_and_expand_paths(
        [
            Path("temp_cache/embedPythia/55*/skim/*_iterative_splittings.root"),
            Path("trains/embedPythia/55*/skim/*_iterative_splittings.root"),
        ]
    )

    dfs = [get_df(path, "tree", branches=["scale_factor", "*det_level*", "*hybrid*"]) for path in path_list]
    daskframe = dd.from_delayed(dfs)

    IPython.start_ipython(user_ns=locals())


# def df_from_file(filenames: Sequence[Path], branches: Sequence[str]):
def df_from_file_embedding(dataset: SkimDataset, path_list_friends: Sequence[Path]) -> None:  # noqa: 901
    # It's dumb to reimport, but we need to do  it here for it to be available immediately in IPython.
    from pathlib import Path  # noqa: F401

    data_frames = uproot3.pandas.iterate(
        path=dataset.path_list,
        treepath="tree",
        namedecode="utf-8",
        # Apparently I forgot to rename the prefixes for eta, data, so I account for that here and when I access the values.
        branches=[
            "scale_factor",
            "*true*",
            "*det_level*",
            "*hybrid*",
            "jet_eta_data",
            "jet_phi_data",
            "jet_eta_detLevel",
            "jet_phi_detLevel",
        ],
        reportpath=True,
        # Otherwise, we can't really count how many steps it's going to take...
        # entrysteps=float("inf"),
    )
    # NOTE: One needs to be careful here if iterating over many files. The friends may not match up in the entries!!
    # data_frames_friends = uproot3.pandas.iterate(
    #    path=path_list_friends,
    #    treepath="tree",
    #    namedecode="utf-8",
    #    branches=["*matched*", "*detLevel*", "*data*"],
    #    reportpath=True,
    # )

    # NOPE! Still too big...
    # df = pd.concat(data_frames, axis=1, copy=False)

    # TODO: Define grooming methods better?
    grooming_methods = [
        "dynamical_z",
        "dynamical_kt",
        "dynamical_time",
        "leading_kt",
        "leading_kt_z_cut_02",
        "leading_kt_z_cut_04",
        "soft_drop_z_cut_02",
        "soft_drop_z_cut_04",
    ]

    # Define hists.
    response_types = [
        skim_analysis_objects.ResponseType(measured_like="hybrid", generator_like="det_level"),
        skim_analysis_objects.ResponseType(measured_like="hybrid", generator_like="true"),
        skim_analysis_objects.ResponseType(measured_like="det_level", generator_like="true"),
    ]
    hists = {}
    for grooming_method in grooming_methods:
        for matching_type in _matching_name_to_axis_value:
            #
            # Residuals by matching type
            #
            # jet_pt residual
            # Axes: hybrid_level_jet_pt, det_level_jet_pt, residual
            hists[f"{grooming_method}_hybrid_det_level_jet_pt_residuals_matching_type_{matching_type}"] = bh.Histogram(
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(80, -2, 2),
                storage=bh.storage.Weight(),
            )
            # kt residual
            # Axes: hybrid_level_jet_kt, det_level_jet_kt, residual
            hists[f"{grooming_method}_hybrid_det_level_kt_residuals_matching_type_{matching_type}"] = bh.Histogram(
                bh.axis.Regular(25, 0, 25),
                bh.axis.Regular(25, 0, 25),
                bh.axis.Regular(80, -2, 2),
                storage=bh.storage.Weight(),
            )
            #
            # Responses
            #
            # Example axes: hybrid_pt, hybrid_kt, det_level_pt, det_level_kt
            # Generally: measured_pt, measured_kt, generator_pt, generator_kt
            for response_type in response_types:
                # kt response
                hists[
                    f"{grooming_method}_{str(response_type)}_kt_response_matching_type_{matching_type}"
                ] = bh.Histogram(
                    bh.axis.Regular(28, 0, 140),
                    bh.axis.Regular(26, -1, 25),
                    bh.axis.Regular(28, 0, 140),
                    bh.axis.Regular(26, -1, 25),
                    storage=bh.storage.Weight(),
                )
                # Delta R response
                # Axes: measured_pt, measured_R, generator_pt, generator_R
                hists[
                    f"{grooming_method}_{str(response_type)}_delta_R_response_matching_type_{matching_type}"
                ] = bh.Histogram(
                    bh.axis.Regular(28, 0, 140),
                    bh.axis.Regular(21, -0.02, 0.4),
                    bh.axis.Regular(28, 0, 140),
                    bh.axis.Regular(21, -0.02, 0.4),
                    storage=bh.storage.Weight(),
                )

            # Where we restrict the matching distance.
            # kt response
            hists[
                f"{grooming_method}_hybrid_det_level_kt_response_matching_type_leading_correct_subleading_untagged_distance_less_than_005"
            ] = bh.Histogram(
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(26, -1, 25),
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(26, -1, 25),
                storage=bh.storage.Weight(),
            )
            # Delta R response
            # Axes: measured_pt, measured_R, generator_pt, generator_R
            hists[
                f"{grooming_method}_hybrid_det_level_delta_R_response_matching_type_leading_correct_subleading_untagged_distance_less_than_005"
            ] = bh.Histogram(
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(21, -0.02, 0.4),
                bh.axis.Regular(28, 0, 140),
                bh.axis.Regular(21, -0.02, 0.4),
                storage=bh.storage.Weight(),
            )

        #
        # Residuals
        #
        # Jet pt residual mean: JES
        # Error is the width: JER
        # We normalize the width by the true jet pt afterwards, so we have to collect it separately.
        # NOTE: Ideally, we'd extract both values from one profile histogram which is binned in true and
        #       hybrid jet pt, but projecting profiles doesn't seem to work quite as expected with bh, and
        #       it's not worth investigating further at the moment.
        for hybrid_jet_pt_bin in [helpers.RangeSelector(40, 120), helpers.RangeSelector(20, 200)]:
            hists[
                f"{grooming_method}_hybrid_det_level_jet_pt_residual_mean_hybrid_{str(hybrid_jet_pt_bin)}"
            ] = bh.Histogram(
                bh.axis.Regular(25, 0, 250),
                storage=bh.storage.WeightedMean(),
            )
            hists[f"{grooming_method}_hybrid_true_jet_pt_residual_mean_hybrid_{str(hybrid_jet_pt_bin)}"] = bh.Histogram(
                bh.axis.Regular(25, 0, 250),
                storage=bh.storage.WeightedMean(),
            )
            hists[f"{grooming_method}_det_true_jet_pt_residual_mean_hybrid_{str(hybrid_jet_pt_bin)}"] = bh.Histogram(
                bh.axis.Regular(25, 0, 250),
                storage=bh.storage.WeightedMean(),
            )
        # Residual so we can see the entire distribution.
        # We intentionally don't select the hybrid jet pt range here.
        hists[f"{grooming_method}_hybrid_det_level_jet_pt_residual_distribution"] = bh.Histogram(
            bh.axis.Regular(15, 0, 150),
            bh.axis.Regular(150, -1.5, 1.5),
            storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_hybrid_true_jet_pt_residual_distribution"] = bh.Histogram(
            bh.axis.Regular(15, 0, 150),
            bh.axis.Regular(150, -1.5, 1.5),
            storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_det_true_jet_pt_residual_distribution"] = bh.Histogram(
            bh.axis.Regular(15, 0, 150),
            bh.axis.Regular(150, -1.5, 1.5),
            storage=bh.storage.Weight(),
        )

        # Distance comparison by matching type
        for matching_type in _matching_name_to_axis_value:
            hists[f"{grooming_method}_hybrid_det_level_distance_matching_type_{matching_type}"] = bh.Histogram(
                bh.axis.Regular(20, 0, 0.5),
                storage=bh.storage.Weight(),
            )

    progress_manager = enlighten.Manager()
    # NOTE: Careful with this counter! It may not be correct if we are iterating over chunks in a file.
    with progress_manager.counter(
        total=len(dataset.path_list), desc="Analyzing", unit="tree", leave=True
    ) as tree_counter:
        # for (df_path, df), (df_friend_path, df_friend) in tree_counter(zip(data_frames, data_frames_friends)):
        for df_path, df in tree_counter(data_frames):
            logger.debug(f"Processing df from {df_path}")
            # Merge the friends together.
            # Rename friends columns because I forgot to rename earlier.
            # df_friend = df_friend.rename(
            #    columns=lambda s: s.replace("matched", "true")
            #    .replace("data", "hybrid")
            #    .replace("detLevel", "det_level")
            # )
            # df = pd.concat([df, df_friend], axis=1)

            # Setup
            hybrid_jet_pt_mask = (df["jet_pt_hybrid"] > 40) & (df["jet_pt_hybrid"] < 120)
            # Add in the double counting cut into the jet pt mask (because we always want to apply it
            # along side the jet pt cut)
            hybrid_jet_pt_mask = hybrid_jet_pt_mask & (df["leading_track_det_level"] >= df["leading_track_hybrid"])
            # And finally process
            for grooming_method in grooming_methods:

                matching_leading = df[f"{grooming_method}_hybrid_det_level_matching_leading"]
                matching_subleading = df[f"{grooming_method}_hybrid_det_level_matching_subleading"]

                matching_selections = analysis_objects.MatchingSelections(
                    leading=analysis_objects.MatchingResult(
                        properly=(matching_leading == 1),
                        mistag=(matching_leading == 2),
                        failed=(matching_leading == 3),
                    ),
                    subleading=analysis_objects.MatchingResult(
                        properly=(matching_subleading == 1),
                        mistag=(matching_subleading == 2),
                        failed=(matching_subleading == 3),
                    ),
                )

                # Residuals (without caring about matching types, so we just take all)
                mask = matching_selections["all"]
                for temp_hybrid_jet_pt_bin in [helpers.RangeSelector(40, 120), helpers.RangeSelector(20, 200)]:
                    # NOTE: Add temp_ onto the names to avoid interfering with the general hybrid jet pt mask
                    #       which is used all over the place.
                    temp_hybrid_jet_pt_mask = temp_hybrid_jet_pt_bin.mask_array(df["jet_pt_hybrid"])
                    masked_df = df[mask & temp_hybrid_jet_pt_mask]
                    # Residual mean and width
                    # We store the unnnormalized residual so we can use this profile hist to extract both the
                    # mean and the width. Note that it's slightly less accurate because then normalize by the bin center
                    # rather than the exact true jet pt per fill, but it's close enough for our purposes.
                    hists[
                        f"{grooming_method}_hybrid_det_level_jet_pt_residual_mean_hybrid_{str(temp_hybrid_jet_pt_bin)}"
                    ].fill(
                        masked_df["jet_pt_true"].to_numpy(),
                        sample=(masked_df["jet_pt_hybrid"] - masked_df["jet_pt_det_level"]).to_numpy(),
                        weight=masked_df["scale_factor"].to_numpy(),
                    )
                    hists[
                        f"{grooming_method}_hybrid_true_jet_pt_residual_mean_hybrid_{str(temp_hybrid_jet_pt_bin)}"
                    ].fill(
                        masked_df["jet_pt_true"].to_numpy(),
                        sample=(masked_df["jet_pt_hybrid"] - masked_df["jet_pt_true"]).to_numpy(),
                        weight=masked_df["scale_factor"].to_numpy(),
                    )
                    hists[f"{grooming_method}_det_true_jet_pt_residual_mean_hybrid_{str(temp_hybrid_jet_pt_bin)}"].fill(
                        masked_df["jet_pt_true"].to_numpy(),
                        sample=(masked_df["jet_pt_det_level"] - masked_df["jet_pt_true"]).to_numpy(),
                        weight=masked_df["scale_factor"].to_numpy(),
                    )
                # Full residual as a function of true pt. We can select the true jet pt range when plotting.
                # NOTE: We intentionally didn't apply a hybrid jet pt cut. And our true jet pt selection will
                #       be applied when plotting.
                masked_df = df[mask]
                hists[f"{grooming_method}_hybrid_det_level_jet_pt_residual_distribution"].fill(
                    masked_df["jet_pt_true"].to_numpy(),
                    (
                        (masked_df["jet_pt_hybrid"] - masked_df["jet_pt_det_level"]) / masked_df["jet_pt_det_level"]
                    ).to_numpy(),
                    weight=masked_df["scale_factor"].to_numpy(),
                )
                hists[f"{grooming_method}_hybrid_true_jet_pt_residual_distribution"].fill(
                    masked_df["jet_pt_true"].to_numpy(),
                    ((masked_df["jet_pt_hybrid"] - masked_df["jet_pt_true"]) / masked_df["jet_pt_true"]).to_numpy(),
                    weight=masked_df["scale_factor"].to_numpy(),
                )
                hists[f"{grooming_method}_det_true_jet_pt_residual_distribution"].fill(
                    masked_df["jet_pt_true"].to_numpy(),
                    ((masked_df["jet_pt_det_level"] - masked_df["jet_pt_true"]) / masked_df["jet_pt_true"]).to_numpy(),
                    weight=masked_df["scale_factor"].to_numpy(),
                )
                # Matching distance
                masked_df = df[hybrid_jet_pt_mask]
                # We convert from pd.Series to ndarray because the pd.Series conversions seem a bit odd at times.
                distances = np.sqrt(
                    (masked_df["jet_eta_data"] - masked_df["jet_eta_detLevel"]) ** 2
                    + (masked_df["jet_phi_data"] - masked_df["jet_phi_detLevel"]) ** 2
                ).to_numpy()
                for matching_type in _matching_name_to_axis_value:
                    mask = matching_selections[matching_type]
                    masked_df = df[mask & hybrid_jet_pt_mask]
                    # Axes: hybrid_level_jet_pt, det_level_jet_pt, residual
                    hists[f"{grooming_method}_hybrid_det_level_jet_pt_residuals_matching_type_{matching_type}"].fill(
                        masked_df["jet_pt_hybrid"].to_numpy(),
                        masked_df["jet_pt_det_level"].to_numpy(),
                        (
                            (masked_df["jet_pt_hybrid"] - masked_df["jet_pt_det_level"]) / masked_df["jet_pt_det_level"]
                        ).to_numpy(),
                        weight=masked_df["scale_factor"].to_numpy(),
                    )
                    # Axes: hybrid_level_jet_kt, det_level_jet_kt, residual
                    hists[f"{grooming_method}_hybrid_det_level_kt_residuals_matching_type_{matching_type}"].fill(
                        masked_df[f"{grooming_method}_hybrid_kt"].to_numpy(),
                        masked_df[f"{grooming_method}_det_level_kt"].to_numpy(),
                        (
                            (masked_df[f"{grooming_method}_hybrid_kt"] - masked_df[f"{grooming_method}_det_level_kt"])
                            / masked_df[f"{grooming_method}_det_level_kt"]
                        ).to_numpy(),
                        weight=masked_df["scale_factor"].to_numpy(),
                    )

                    # Matching distances
                    # Distances were only calculated for hybrid jets, so we don't need to reapply the hybrid jet pt cut here.
                    # Just apply the matching mask to the distances.
                    hists[f"{grooming_method}_hybrid_det_level_distance_matching_type_{matching_type}"].fill(
                        distances[mask[hybrid_jet_pt_mask]], weight=masked_df["scale_factor"].to_numpy()
                    )

                    for response_type in response_types:
                        # Axes: measured_like_pt, measured_like_kt, generator_like_pt, generator_like_kt
                        hists[f"{grooming_method}_{str(response_type)}_kt_response_matching_type_{matching_type}"].fill(
                            masked_df[f"jet_pt_{response_type.measured_like}"].to_numpy(),
                            masked_df[f"{grooming_method}_{response_type.measured_like}_kt"].to_numpy(),
                            masked_df[f"jet_pt_{response_type.generator_like}"].to_numpy(),
                            masked_df[f"{grooming_method}_{response_type.generator_like}_kt"].to_numpy(),
                            weight=masked_df["scale_factor"].to_numpy(),
                        )
                        # Axes: measured_like_pt, measured_like_R, generator_like_pt, generator_like_R
                        hists[
                            f"{grooming_method}_{str(response_type)}_delta_R_response_matching_type_{matching_type}"
                        ].fill(
                            masked_df[f"jet_pt_{response_type.measured_like}"].to_numpy(),
                            masked_df[f"{grooming_method}_{response_type.measured_like}_delta_R"].to_numpy(),
                            masked_df[f"jet_pt_{response_type.generator_like}"].to_numpy(),
                            masked_df[f"{grooming_method}_{response_type.generator_like}_delta_R"].to_numpy(),
                            weight=masked_df["scale_factor"].to_numpy(),
                        )
                    # Response with matching distance for leading_correct_subleading_untagged
                    if matching_type == "leading_correct_subleading_untagged":
                        distance_mask = distances[mask[hybrid_jet_pt_mask]] < 0.05
                        # kt
                        hists[
                            f"{grooming_method}_hybrid_det_level_kt_response_matching_type_leading_correct_subleading_untagged_distance_less_than_005"
                        ].fill(
                            masked_df[distance_mask][f"jet_pt_{response_type.measured_like}"].to_numpy(),
                            masked_df[distance_mask][f"{grooming_method}_{response_type.measured_like}_kt"].to_numpy(),
                            masked_df[distance_mask][f"jet_pt_{response_type.generator_like}"].to_numpy(),
                            masked_df[distance_mask][f"{grooming_method}_{response_type.generator_like}_kt"].to_numpy(),
                            weight=masked_df[distance_mask]["scale_factor"].to_numpy(),
                        )
                        # Delta R
                        hists[
                            f"{grooming_method}_hybrid_det_level_delta_R_response_matching_type_leading_correct_subleading_untagged_distance_less_than_005"
                        ].fill(
                            masked_df[distance_mask][f"jet_pt_{response_type.measured_like}"].to_numpy(),
                            masked_df[distance_mask][
                                f"{grooming_method}_{response_type.measured_like}_delta_R"
                            ].to_numpy(),
                            masked_df[distance_mask][f"jet_pt_{response_type.generator_like}"].to_numpy(),
                            masked_df[distance_mask][
                                f"{grooming_method}_{response_type.generator_like}_delta_R"
                            ].to_numpy(),
                            weight=masked_df[distance_mask]["scale_factor"].to_numpy(),
                        )

    progress_manager.stop()

    # Write the hists
    dataset.output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving hists to {dataset.output_filename}")
    with gzip.GzipFile(dataset.output_filename, "w") as pkl_file:
        dill.dump(hists, pkl_file)


def map_reduce_pandas_concat() -> None:
    data_frames = uproot3.pandas.iterate(
        path=[
            "temp_cache/embedPythia/55*/skim/*_iterative_splittings.root",
            "trains/embedPythia/55*/skim/*_iterative_splittings.root",
        ],
        treepath="tree",
        namedecode="utf-8",
        branches=["scale_factor", "*det_level*", "*hybrid*"],
    )

    # NOPE! Still too big...
    logger.debug("Reducing")
    df = functools.reduce(lambda x, y: pd.concat([x, y], copy=False, ignore_index=True), data_frames)
    logger.debug("Finished")
    # NOPE! Still too big...
    # logger.debug("Starting concat")
    # df = pd.concat(data_frames, copy=False)

    IPython.start_ipython(user_ns=locals())


def _fill_grooming_hists(
    masked_df: pd.DataFrame,
    grooming_method: str,
    hists: Mapping[str, bh.Histogram],
    prefix: str,
    suffix: Optional[str] = None,
) -> None:
    """Fill grooming hists using the df.

    This is in a separate function so the DataFrame can be masked.

    Args:
        masked_df: Masked DataFrame to be used for filling.
        grooming_method: Grooming method to be filled.
        hists: Hists to be filled.
        prefix: Prefix specifying the data type, such as "data", "matched", "true", etc.
        suffix: Suffix to additional identify the hists.
    Returns:
        None. The hists stored in the hists dict are filled.
    """
    if suffix is None:
        suffix = ""
    else:
        if not suffix.startswith("_"):
            suffix = f"_{suffix}"

    # Handle the case of the first split.
    # If "_first_split" isn't included in the grooming method, then nothing is replaced.
    grooming_method_for_df = grooming_method.replace("_first_split", "")

    hists[f"{grooming_method}_{prefix}_kt{suffix}"].fill(
        masked_df[f"{prefix}_jet_pt"].to_numpy(),
        masked_df[f"{grooming_method_for_df}_{prefix}_kt"].to_numpy(),
        weight=masked_df["scale_factor"].to_numpy(),
    )
    hists[f"{grooming_method}_{prefix}_delta_R{suffix}"].fill(
        masked_df[f"{prefix}_jet_pt"].to_numpy(),
        masked_df[f"{grooming_method_for_df}_{prefix}_delta_R"].to_numpy(),
        weight=masked_df["scale_factor"].to_numpy(),
    )
    hists[f"{grooming_method}_{prefix}_z{suffix}"].fill(
        masked_df[f"{prefix}_jet_pt"].to_numpy(),
        masked_df[f"{grooming_method_for_df}_{prefix}_z"].to_numpy(),
        weight=masked_df["scale_factor"].to_numpy(),
    )
    hists[f"{grooming_method}_{prefix}_n_to_split{suffix}"].fill(
        masked_df[f"{prefix}_jet_pt"].to_numpy(),
        masked_df[f"{grooming_method_for_df}_{prefix}_n_to_split"].to_numpy(),
        weight=masked_df["scale_factor"].to_numpy(),
    )
    hists[f"{grooming_method}_{prefix}_n_groomed_to_split{suffix}"].fill(
        masked_df[f"{prefix}_jet_pt"].to_numpy(),
        masked_df[f"{grooming_method_for_df}_{prefix}_n_groomed_to_split"].to_numpy(),
        weight=masked_df["scale_factor"].to_numpy(),
    )
    # Number of splittings which pass the grooming condition. For SoftDrop, this is n_sd.
    hists[f"{grooming_method}_{prefix}_n_passed_grooming{suffix}"].fill(
        masked_df[f"{prefix}_jet_pt"].to_numpy(),
        masked_df[f"{grooming_method_for_df}_{prefix}_n_passed_grooming"].to_numpy(),
        weight=masked_df["scale_factor"].to_numpy(),
    )


def df_from_file_data(dataset: SkimDataset) -> None:  # noqa: 901
    # Setup
    jet_R = 0.4
    prefix = dataset.prefix
    branches = [f"*{prefix}*"]
    if dataset.collision_system in ["pythia", "embedPythia"]:
        # TODO: Extract properly and re-enable...
        # branches.append("scale_factor")
        pass
    if dataset.collision_system == "embedPythia":
        branches.append("det_level_leading_track_pt")
        branches.append("data_leading_track_pt")
    # data_frames = uproot3.pandas.iterate(
    # Using entrysteps=float("inf") is imperative for uproot.pandas.iterate to get reasonable performance!
    # TODO: Make tree name configurable...
    data_frames = uproot3.tree.iterate(
        outputtype=pd.DataFrame,
        # path=dataset.path_list, treepath="tree", namedecode="utf-8", branches=branches, reportpath=True,
        # path=dataset.path_list, treepath="AliAnalysisTaskJetHardestKt_Jet_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_Data_ConstSub_Incl",
        path=dataset.path_list,
        treepath="AliAnalysisTaskJetHardestKt_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
        namedecode="utf-8",
        branches=branches,
        reportpath=True,  # entrysteps=float("inf"),
        entrysteps="500MB",
    )
    logger.debug(f"Path list: {dataset.path_list}")
    logger.debug(f"Branches: {branches}")

    # TODO: Define grooming methods better?
    grooming_methods = [
        # "dynamical_z",
        # "dynamical_kt",
        # "dynamical_time",
        # "leading_kt",
        "leading_kt_z_cut_02",
        # "leading_kt_z_cut_04",
        # "soft_drop_z_cut_02",
        # "soft_drop_z_cut_04",
    ]
    direct_comparison_grooming_methods: List[str] = [
        # "leading_kt_z_cut_02_first_split",
        # "leading_kt_z_cut_04_first_split",
    ]

    # Define hists.
    hists = {}
    for grooming_method in itertools.chain(grooming_methods, direct_comparison_grooming_methods):
        # Standard
        jet_pt_axis = bh.axis.Regular(28, 0, 140)
        hists[f"{grooming_method}_{prefix}_kt"] = bh.Histogram(
            jet_pt_axis,
            bh.axis.Regular(26, -1, 25),
            storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_delta_R"] = bh.Histogram(
            jet_pt_axis,
            bh.axis.Regular(21, -0.02, jet_R),
            storage=bh.storage.Weight(),
        )
        # hists[f"{grooming_method}_{prefix}_theta"] = bh.Histogram(
        #    jet_pt_axis, bh.axis.Regular(21, -0.05, 1.0), storage=bh.storage.Weight(),
        # )
        hists[f"{grooming_method}_{prefix}_z"] = bh.Histogram(
            jet_pt_axis,
            bh.axis.Regular(21, -0.025, 0.5),
            storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_n_to_split"] = bh.Histogram(
            jet_pt_axis,
            bh.axis.Regular(10, -0.5, 9.5),
            storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_n_groomed_to_split"] = bh.Histogram(
            jet_pt_axis,
            bh.axis.Regular(10, -0.5, 9.5),
            storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_n_passed_grooming"] = bh.Histogram(
            jet_pt_axis,
            bh.axis.Regular(10, -0.5, 9.5),
            storage=bh.storage.Weight(),
        )
        # Lund plane
        hists[f"{grooming_method}_{prefix}_lund_plane"] = bh.Histogram(
            jet_pt_axis,
            bh.axis.Regular(100, 0, 5),
            bh.axis.Regular(100, -5.0, 5.0),
            storage=bh.storage.Weight(),
        )
        # High kt
        hists[f"{grooming_method}_{prefix}_kt_high_kt"] = bh.Histogram(
            jet_pt_axis,
            bh.axis.Regular(26, -1, 25),
            storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_delta_R_high_kt"] = bh.Histogram(
            jet_pt_axis,
            bh.axis.Regular(21, -0.02, jet_R),
            storage=bh.storage.Weight(),
        )
        # hists[f"{grooming_method}_{prefix}_theta_high_kt"] = bh.Histogram(
        #    jet_pt_axis, bh.axis.Regular(21, -0.05, 1.0), storage=bh.storage.Weight(),
        # )
        hists[f"{grooming_method}_{prefix}_z_high_kt"] = bh.Histogram(
            jet_pt_axis,
            bh.axis.Regular(21, -0.025, 0.5),
            storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_n_to_split_high_kt"] = bh.Histogram(
            jet_pt_axis,
            bh.axis.Regular(10, -0.5, 9.5),
            storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_n_groomed_to_split_high_kt"] = bh.Histogram(
            jet_pt_axis,
            bh.axis.Regular(10, -0.5, 9.5),
            storage=bh.storage.Weight(),
        )
        hists[f"{grooming_method}_{prefix}_n_passed_grooming_high_kt"] = bh.Histogram(
            jet_pt_axis,
            bh.axis.Regular(10, -0.5, 9.5),
            storage=bh.storage.Weight(),
        )

    progress_manager = enlighten.Manager()
    logger.debug("About to start processing...")
    with progress_manager.counter(
        total=len(dataset.path_list), desc="Analyzing", unit="tree", leave=True
    ) as tree_counter:
        for df_path, df in tree_counter(data_frames):
            logger.info(f"Processing df from {df_path}")
            # Setup
            # Add scale_factor weight as 1 if it's not included. This way, we can always weight from the
            # scale factor of the masked_df.
            if "scale_factor" not in df:
                df = df.assign(scale_factor=np.ones_like(df[f"{prefix}_jet_pt"]))
            # Jet pt bin
            jet_pt_bin = helpers.RangeSelector(min=40, max=120)
            jet_pt_mask = jet_pt_bin.mask_array(df[f"{prefix}_jet_pt"])
            # Add double counting cut for embedPythia
            if dataset.collision_system == "embedPythia":
                jet_pt_mask = jet_pt_mask & (df["det_level_leading_track_pt"] >= df["data_leading_track_pt"])

            # Standard grooming method plots
            masked_df = df[jet_pt_mask]
            for grooming_method in grooming_methods:
                _fill_grooming_hists(masked_df=masked_df, grooming_method=grooming_method, hists=hists, prefix=prefix)

            # Lund plane
            hists[f"{grooming_method}_{prefix}_lund_plane"].fill(
                masked_df[f"{prefix}_jet_pt"].to_numpy(),
                np.log(1.0 / masked_df[f"{grooming_method}_{prefix}_delta_R"].to_numpy()),
                np.log(masked_df[f"{grooming_method}_{prefix}_kt"].to_numpy()),
                weight=masked_df["scale_factor"].to_numpy(),
            )

            # Direct comparison plots
            mask = jet_pt_mask & (df[f"{grooming_method}_{prefix}_n_passed_grooming"] <= 1)
            masked_df = df[mask]
            for grooming_method in direct_comparison_grooming_methods:
                _fill_grooming_hists(masked_df=masked_df, grooming_method=grooming_method, hists=hists, prefix=prefix)

            # High kt grooming plots.
            for grooming_method in grooming_methods:
                mask = jet_pt_mask & (df[f"{grooming_method}_{prefix}_kt"] > 10)
                _fill_grooming_hists(
                    masked_df=df[mask], grooming_method=grooming_method, hists=hists, prefix=prefix, suffix="_high_kt"
                )

    progress_manager.stop()

    # Write the hists
    dataset.output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving hists to {dataset.output_filename}")
    with gzip.GzipFile(dataset.output_filename, "w") as pkl_file:
        dill.dump(hists, pkl_file)


# def run_embed_pythia(run_response: bool = True) -> None:
#    collision_system = "embedPythia"
#    path_list = helpers.ensure_and_expand_paths(
#        [
#            Path("trains/embedPythia/588*/skim/*_iterative_splittings.root"),
#            Path("trains/embedPythia/589*/skim/*_iterative_splittings.root"),
#            Path("trains/embedPythia/590*/skim/*_iterative_splittings.root"),
#        ]
#    )
#    path_list_friends = helpers.ensure_and_expand_paths(
#        [
#            #Path("temp_cache/embedPythia/55*/skim/*_iterative_splittings_friend.root"),
#            #Path("trains/embedPythia/55*/skim/*_iterative_splittings_friend.root"),
#        ]
#    )
#    if run_response:
#        for train_number in range(5903, 5904):
#            logger.info(f"Processing train number {train_number}")
#            path_list = helpers.ensure_and_expand_paths(
#                [
#                    #Path("trains/embedPythia/5903/skim/merged/*.root")
#                    Path("trains/embedPythia/5903/skim/merged/AnalysisResults.merged.01.root")
#                ]
#            )
#            print(path_list)
#            df_from_file_embedding(path_list=path_list, path_list_friends=path_list_friends, output_dir=Path(f"output/{collision_system}/skim/{train_number}"))
#
#        # Marge and write the data hists
#        embedding_hists = functools.reduce(merge_hists, [dill.load(gzip.GzipFile(f"output/{collision_system}/skim/{train_number}/embedded.pgz", "r")) for train_number in range(5884, 5904)])
#        pkl_filename = Path(f"output/{collision_system}/skim/embedded.pgz")
#        logger.info(f"Saving hists to {pkl_filename}")
#        with gzip.GzipFile(pkl_filename, "w") as pkl_file:
#            dill.dump(embedding_hists, pkl_file)
#
#    for train_number in range(5904, 5904):
#        logger.info(f"Processing train number {train_number}")
#        path_list = helpers.ensure_and_expand_paths(
#            [
#                Path(f"trains/embedPythia/{train_number}/skim/*_iterative_splittings.root"),
#            ]
#        )
#        df_from_file_data(
#            collision_system=collision_system, path_list=path_list, prefix="hybrid", output_dir=Path(f"output/{collision_system}/skim/{train_number}")
#        )
#
#    # Marge and write the data hists
#    hists = functools.reduce(merge_hists, [pickle.load(gzip.GzipFile(f"output/{collision_system}/skim/{train_number}/{collision_system}.pgz", "r")) for train_number in range(5884, 5904)])  # type: ignore
#    pkl_filename = Path(f"output/{collision_system}/skim/{collision_system}.pgz")
#    logger.info(f"Saving hists to {pkl_filename}")
#    with gzip.GzipFile(pkl_filename, "w") as pkl_file:
#        pickle.dump(hists, pkl_file)  # type: ignore


def merge_output(train_numbers: Sequence[int], output_filename: Path, output_path: Path, prefix: str = "") -> Path:
    filename = output_filename.with_suffix("").name
    # Have to remore the prefix if it's there because I made a typo when creating the files
    logger.debug(f"prefix: {prefix}")
    search_filename = filename
    if prefix:
        search_filename = search_filename.replace(f"_{prefix}", "")
    # embedding_hists = functools.reduce(merge_hists, [dill.load(gzip.GzipFile(f"output/{collision_system}/skim/{train_number}/embedded.pgz", "r")) for train_number in train_numbers])
    files = []
    for train_number in train_numbers:
        files.extend(list((Path(output_path) / str(train_number)).glob(f"{search_filename}_*.pgz")))
    logger.info(f"filename: {filename}")
    logger.info(f"files: {files}")
    first_filename = files[0]
    with gzip.GzipFile(first_filename, "r") as f_gz:
        hists = dill.load(f_gz)
    for f in files[1:3]:
        logger.debug(f"Handling file {f}")
        with gzip.GzipFile(f, "r") as f_gz:
            temp_hists = dill.load(f_gz)
            for k, v in temp_hists.items():
                hists[k] += v
        del temp_hists
    # hists = functools.reduce(
    #    _merge_hists,
    #    (
    #        dill.load(gzip.GzipFile(f, "r"))
    #        for f in files
    #    ),
    # )
    pkl_filename = output_path / f"{filename}.pgz"
    logger.info(f"Saving hists to {pkl_filename}")
    with gzip.GzipFile(pkl_filename, "w") as pkl_file:
        dill.dump(hists, pkl_file)

    return pkl_filename


def output_dir_f(output_dir: Path, identifier: str) -> Path:
    """Format an output_dir path with a given identifier.

    Also ensures that the directory exists.

    Args:
        output_dir: Output dir containing a format identifier, `{identifier}`.
        identifier: Identifier to include in the path. Usually, it's the collision system,
            but it doesn't have to be.
    Returns:
        Output path formatted with the identifier.
    """
    p = Path(str(output_dir).format(identifier=identifier))
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_all() -> None:
    # TODO: Consolidate
    grooming_methods = [
        # "dynamical_z",
        # "dynamical_kt",
        # "dynamical_time",
        # "leading_kt",
        "leading_kt_z_cut_02",
        # "leading_kt_z_cut_04",
        # "soft_drop_z_cut_02",
        # "soft_drop_z_cut_04",
    ]
    direct_comparison_grooming_methods: List[str] = [
        # "leading_kt_z_cut_02_first_split",
        # "leading_kt_z_cut_04_first_split",
    ]

    # NOTE: Intentionally skipping the f-string here. We want to format it later!
    base_dir = Path("output/{identifier}/skim")

    logger.info("Loading embedded response data")
    pkl_filename = Path("output") / "embedPythia" / "skim" / "embedded.pgz"
    with gzip.GzipFile(pkl_filename, "r") as pkl_file:
        response_hists = pickle.load(pkl_file)  # type: ignore

    logger.info("Loading embedPythia data")
    pkl_filename = Path("output") / "embedPythia" / "skim" / "embedPythia.pgz"
    with gzip.GzipFile(pkl_filename, "r") as pkl_file:
        embed_pythia_hists = pickle.load(pkl_file)  # type: ignore

    logger.info("Loading PbPb data")
    pkl_filename = Path("output") / "PbPb" / "skim" / "PbPb.pgz"
    with gzip.GzipFile(pkl_filename, "r") as pkl_file:
        PbPb_hists = pickle.load(pkl_file)  # type: ignore

    # Add some helpful imports and definitions
    from importlib import reload  # noqa: F401

    try:
        # May not want to import if developing.
        from jet_substructure.analysis import plot_from_skim  # noqa: F401
    except SyntaxError:
        logger.info("Couldn't load plot_from_skim due to syntax error. You need to load it.")

    user_ns = locals()
    user_ns.update({"output_dir_f": output_dir_f, "Path": Path})
    IPython.start_ipython(user_ns=user_ns)


def run_plot(datasets: Mapping[str, SkimDataset], remerge: bool = False) -> None:
    # Setup
    # TODO: Consolidate
    grooming_methods = [
        "dynamical_z",
        "dynamical_kt",
        "dynamical_time",
        "leading_kt",
        "leading_kt_z_cut_02",
        "leading_kt_z_cut_04",
        "soft_drop_z_cut_02",
        "soft_drop_z_cut_04",
    ]
    direct_comparison_grooming_methods = [
        "leading_kt_z_cut_02_first_split",
        "leading_kt_z_cut_04_first_split",
    ]

    # Load datasets (and merge if necessary)
    for dataset in datasets.values():
        if not dataset.output_filename.exists() or remerge:
            merge_output(
                train_numbers=dataset.train_numbers,
                output_filename=dataset.output_filename,
                output_path=dataset.output_path,
                prefix=dataset.prefix,
            )
        dataset.load_hists()

    # Add some helpful imports and definitions
    from importlib import reload  # noqa: F401

    try:
        # May not want to import if developing.
        from jet_substructure.analysis import plot_from_skim  # noqa: F401
    except SyntaxError:
        logger.info("Couldn't load plot_from_skim due to syntax error. You need to load it.")

    user_ns = locals()
    user_ns.update({"output_dir_f": output_dir_f})
    IPython.start_ipython(user_ns=user_ns)

    # Plotting
    # plot_from_skim.plot_residuals_by_matching_type(
    #     hists=hists, grooming_methods=grooming_methods, matching_types=list(_matching_name_to_axis_value.keys()), output_dir=output_dir
    # )
    # plot_from_skim.plot_residuals(hists=hists, grooming_methods=grooming_methods, output_dir=output_dir)
    # plot_from_skim.plot_response_by_matching_type(
    #     hists=hists, grooming_methods=grooming_methods, matching_types=list(_matching_name_to_axis_value.keys()), output_dir=output_dir,
    # )
    # plot_from_skim.plot_compare_kt(hists=hists, data_hists=data_hists[0], grooming_methods=grooming_methods, output_dir=output_dir)


def define_embedding_datasets(
    output_identifier: str = "",
    train_number: Optional[int] = None,
    train_numbers: Optional[Sequence[int]] = None,
    merged_skim: bool = False,
) -> Dict[str, SkimDataset]:
    if train_number is None and train_numbers is None:
        raise ValueError("Must pass either train number or train_numbers")
    if train_numbers is None:
        # Help out mypy
        assert train_number is not None
        train_numbers = [train_number]
    # Validation
    train_numbers = list(train_numbers)

    output_identifier_full = "embedding"
    if output_identifier:
        output_identifier_full = output_identifier_full + f"_{output_identifier}"
    datasets = {
        "embedPythia_response": SkimDataset(
            collision_system="embedPythia",
            train_numbers=train_numbers,
            merged_skim=merged_skim,
            output_filename_identifier=output_identifier_full,
        ),
    }
    # Could analyze any of these prefixes for the embedded.
    output_identifier_full = "embedPythia"
    if output_identifier:
        output_identifier_full = output_identifier_full + f"_{output_identifier}"
    # for prefix in ["hybrid", "det_level", "true"]:
    for prefix in ["data", "det_level", "matched"]:
        datasets[f"embedPythia_{prefix}"] = SkimDataset(
            collision_system="embedPythia",
            train_numbers=train_numbers,
            merged_skim=merged_skim,
            prefix=prefix,
            output_filename_identifier=output_identifier_full,
        )
    return datasets


def process_embedding_skim_entry_point() -> None:
    """Entry point for processing the skim.

    Args:
        None. It can be configured through command line arguments.

    Returns:
        None.
    """
    helpers.setup_logging()
    parser = argparse.ArgumentParser(description="Processed the skimmed dataset.")
    parser.add_argument("-t", "--trainNumber", required=True, type=int, help="Embedding train number to process.")
    parser.add_argument("-f", "--filename", required=True, type=str, help="Filename to process.")
    args = parser.parse_args()

    embedding_datasets = define_embedding_datasets(
        train_number=args.trainNumber, output_identifier=Path(args.filename).with_suffix("").name
    )
    # This is a hack, but it's easy right now...
    for dataset in embedding_datasets.values():
        dataset._path_list = [Path(args.filename)]

    # Process
    # Response
    df_from_file_embedding(
        dataset=embedding_datasets["embedPythia_response"],
        path_list_friends=[],
    )
    # Hybrid
    df_from_file_data(dataset=embedding_datasets["embedPythia_hybrid"])

    logger.info("Done!")


def run() -> None:
    helpers.setup_logging()
    plot_only = True
    # Define possible datasets
    datasets = {
        "PbPb": SkimDataset(
            collision_system="PbPb",
            # TODO: improve this when time allows to accept single values....
            train_numbers=[5987],
            prefix="data",
            path_list=[Path("trains") / "PbPb" / "5987" / "AnalysisResults.*.root"],
        ),
        # "pythia_data": SkimDataset(
        #    collision_system="pythia",
        #    # TODO: improve this when time allows to accept single values....
        #    train_numbers=[2110],
        #    prefix="data",
        # ),
        # "pythia_matched": SkimDataset(
        #    collision_system="pythia",
        #    # TODO: improve this when time allows to accept single values....
        #    train_numbers=[2110],
        #    prefix="matched",
        # ),
    }
    # datasets.update(define_embedding_datasets(train_numbers=list(range(5988, 6008))))
    datasets.update(define_embedding_datasets(train_numbers=list(range(6007, 6008))))

    if not plot_only:
        # df_from_file_data(dataset=datasets["PbPb"],)
        df_from_file_data(dataset=datasets["embedPythia_data"])
        # run_embed_pythia(run_response=True)
        # df_from_file_data(dataset=datasets["embedPythia_hybrid"],)
        # df_from_file_embedding(
        #    dataset=datasets["embedPythia_response"], path_list_friends=[],
        # )
        # df_from_file_data(dataset=datasets["pythia_matched"],)
        # df_from_file_data(dataset=datasets["pythia_data"],)
        # dask_df_from_file()
        # dask_df_from_delayed()
        # map_reduce_pandas_concat()

    run_plot(datasets=datasets)


if __name__ == "__main__":
    run()
