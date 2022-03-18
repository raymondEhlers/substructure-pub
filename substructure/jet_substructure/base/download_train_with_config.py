#!/usr/bin/env python3

""" Download files based on the config.yaml

"""

import argparse
import logging
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Sequence

import pachyderm.alice.download as alice_dl
import pachyderm.alice.utils as alice_utils
from pachyderm import yaml

from jet_substructure.base import helpers


logger = logging.getLogger(__name__)


_possible_merging_stages = [
    "merged",
    "manual",
    "single_run_manual",
    "run_by_run",
    "Stage_1",
    "Stage_2",
    "Stage_3",
    "Stage_4",
    "Stage_5",
]


def year_from_dataset(dataset: str) -> int:
    """Extract the year from the dataset.

    Args:
        dataset: Dataset (period) name.
    Returns:
        Year of the given dataset.
    """
    if not dataset.startswith("LHC"):
        raise ValueError(f"Invalid dataset name: {dataset}")
    return int(dataset[3:5]) + 2000


def add_files_from_xml_file(
    alien_xml_file: Path, local_xml_file: Path, local_train_dir: Path, child_label: str, additional_label: str = ""
) -> Dict[str, str]:
    output = {}
    # Download the XML file...
    logger.info(f"Downloading {alien_xml_file.name} file: {alien_xml_file} to file://{local_xml_file}")
    subprocess.run(
        ["alien_cp", str(alien_xml_file), f"file://{str(local_xml_file)}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Open local XML file
    tree = ET.parse(str(local_xml_file))
    root = tree.getroot()
    collection = root[0]
    # Extract the filenames
    for node in collection:
        try:
            lfn = node[0].attrib["lfn"]
            alien_file = lfn.replace("root_archive.zip", "AnalysisResults.root")
            i = int(node.attrib["name"])
            if additional_label:
                label = f"{additional_label}.{i:03}"
            else:
                label = f"{i:03}"
            local_file = local_train_dir / f"AnalysisResults.{child_label}.{label}.root"
            # print(f"Adding alien://{alien_file} : {local_file}")
            output[str(alien_file)] = str(local_file)
            # print(f"Downloading alien://{alien_file} to {local_file}")
            # process = subprocess.run(
            #    ["alien_cp", f"alien://{str(alien_file)}", str(local_file)],
            #    stdout = subprocess.PIPE, stderr = subprocess.PIPE
            # )
        except IndexError:
            pass

    return output


def download(trains: Sequence[int]) -> None:  # noqa: C901

    y = yaml.yaml()

    output = {}
    for train_number in trains:
        logger.info(f"Processing train {train_number}")
        local_train_dir = Path(str(train_number))
        config_filename = local_train_dir / "config.yaml"
        with open(config_filename, "r") as f:
            config = y.load(f)

        # Sanity check
        # If this fails, it means I probably forgot to update the run number in the YAML config.
        assert int(config["number"]) == train_number

        # Determine train properties.
        base_alien_path = Path("/alice/cern.ch/user/a/alitrain/")
        PWG = config.get("PWG", "PWGJE")
        train_name = config.get("train")
        base_alien_path = base_alien_path / PWG / train_name
        train_directories_on_alien = alice_utils.list_alien_dir(base_alien_path)
        possible_directories = [dir for dir in train_directories_on_alien if dir.startswith(str(train_number))]
        # NOTE: There could be more than 1 directory if there are children. We only have to check for whether it's empty.
        logger.debug(f"base_alien_path: {base_alien_path}")
        if len(possible_directories) == 0:
            logger.warning(f"Can't find any directories for train. Skipping {train_number}.")
            continue

        alien_output_info = config["alien_output_info"]
        for child_name, child_info in alien_output_info.items():
            # Validation
            child_name = child_name.lower()

            # _ is equivalent to "pass"
            if child_name != "_":
                # Determine the corresponding AliEn directory.
                possible_train_or_child_directories = [dir for dir in possible_directories if dir.endswith(child_name)]
                if len(possible_train_or_child_directories) != 1:
                    logger.debug(f"Could not find train directory corresponding to child {child_name}. Continuing")
                    continue
            else:
                possible_train_or_child_directories = [
                    dir for dir in possible_directories if dir.startswith(str(train_number))
                ]
                if len(possible_train_or_child_directories) != 1:
                    logger.debug(
                        f"Could not find train directory corresponding to train number {train_number}. Continuing"
                    )
                    continue

            # Up until now, the possible directories have been relative to the base_alien_path.
            # Since we're getting close to downloading or saving, we add it back it so we have an absolute path.
            likely_child_directory = possible_train_or_child_directories[0]
            alien_dir = base_alien_path / Path(likely_child_directory) / "merge"

            # Extract values from the config needed to finally determine the files to download.
            # Last successful merging stage, which we will use to determine what to download.
            # We force the user to record the stage (rather than determine it automatically) so the record will be saved.
            stage_to_download = child_info["stage_to_download"]
            # Validation
            if stage_to_download not in _possible_merging_stages:
                raise ValueError(
                    f"Invalid last successful merging stage. Provided: {stage_to_download}. Possible values: {_possible_merging_stages}"
                )
            # Child label (such as LHC18q)
            child_label = child_info.get("name", child_name)
            # Validation
            child_label = child_label.replace("LHC", "").replace("lhc", "")

            if stage_to_download == "merged":
                # We just want the final merge files.
                local_file = local_train_dir / f"AnalysisResults.{child_label}.root"
                # So we just save it.
                output[str(alien_dir / "AnalysisResults.root")] = str(local_file)
            elif stage_to_download == "manual":
                manual_config = child_info["manual"]
                logger.warning("Relying on LHC18qr specific info. Careful!")
                for run_number, manual_stage_to_download in manual_config.items():
                    # Validation
                    if manual_stage_to_download not in _possible_merging_stages:
                        raise ValueError(
                            f"Invalid last successful merging stage. Provided: {manual_stage_to_download}. Possible values: {_possible_merging_stages}"
                        )
                    # NOTE: This is LHC18{q,r} specific
                    # Example: /alice/data/2018/LHC18r/000296934/pass1/PWGJE/Jets_EMC_PbPb/5902_20200515-1910_child_1
                    # We default to pass1 because we didn't specify this in the past.
                    pass_default = 1

                    # Dataset agnostic
                    # Determining pass and AOD shouldn't be super specific to LHC18{q,r}
                    pass_value = Path(f"pass{config.get('pass', pass_default)}")
                    aod_value = config.get("AOD", None)
                    if aod_value:
                        pass_value /= f"AOD{aod_value}"
                    dataset_name = f"LHC{child_label}"
                    is_data = alice_dl.does_period_contain_data(dataset_name)
                    run_prefix = "000" if is_data else ""
                    data_or_sim_str = "data" if is_data else "sim"

                    # Back to (somewhat) LHC18{q,r} specific.
                    manual_dir: Path = (
                        Path(f"/alice/{data_or_sim_str}/{year_from_dataset(dataset_name)}/")
                        / dataset_name
                        / f"{run_prefix}{run_number}"
                        / pass_value
                        / PWG
                        / train_name
                    )
                    if likely_child_directory:
                        manual_dir = manual_dir / likely_child_directory
                    if manual_stage_to_download == "merged":
                        local_file = local_train_dir / f"AnalysisResults.{child_label}.{run_number}.root"
                        output[str(manual_dir / "AnalysisResults.root")] = str(local_file)
                    elif "Stage" in manual_stage_to_download:
                        # Use a stage of the merging.
                        alien_xml_file = manual_dir / f"{manual_stage_to_download}.xml"
                        local_xml_file = local_train_dir / f"{run_number}_{manual_stage_to_download}_{child_name}.xml"
                        result = add_files_from_xml_file(
                            alien_xml_file=alien_xml_file,
                            local_xml_file=local_xml_file,
                            local_train_dir=local_train_dir,
                            child_label=child_label,
                            additional_label=str(run_number),
                        )
                        output.update(result)
                    elif manual_stage_to_download == "single_run_manual":
                        logger.info(f"Processing manual single run {run_number} for child {child_label}")
                        # Didn't even get to a stage of the merging. Take whatever is there...
                        _directories_with_output_files = alice_utils.list_alien_dir(manual_dir)
                        directories_with_output_files = [manual_dir / d for d in _directories_with_output_files]
                        _additional_label = str(run_number)
                        for d in directories_with_output_files:
                            # Use having a suffix as a proxy for a file, and without a suffix as a directory.
                            # (I think we can query AliEn, but that seems like overkill just for determining directories...
                            if d.suffix:
                                continue

                            # AliEn filename
                            alien_file = d / "AnalysisResults.root"
                            # Local filename
                            i = int(d.name)
                            if _additional_label:
                                label = f"{_additional_label}.{i:03}"
                            else:
                                label = f"{i:03}"
                            local_file = local_train_dir / f"AnalysisResults.{child_label}.{label}.root"
                            logger.debug(f"Adding alien://{alien_file} : {local_file}")
                            output[str(alien_file)] = str(local_file)
                    else:
                        raise ValueError(
                            f"Invalid manual stage to download {manual_stage_to_download} for run number {run_number}"
                        )

            elif stage_to_download == "run_by_run":
                # Setup
                dataset_name = f"LHC{child_label}"
                # To start, we need to figure out the dataset.
                # Dataset agnostic
                # NOTE: May be run2 specific
                # First, look at a possible child specific pass
                # This potentially lets us hijack the value to add in more info, such as the partition (eg. "FAST")
                _pass_value = child_info.get("pass", None)
                if _pass_value is None:
                    _pass_value = str(config.get("pass", ""))
                    # Add "pass" to the front of the name. Only if it's a general dataset pass
                    if _pass_value != "":
                        _pass_value = f"pass{str(_pass_value)}"
                if _pass_value is None:
                    _pass_value = ""
                pass_value = Path(_pass_value)
                aod_value = config.get("AOD", None)
                if aod_value:
                    pass_value /= f"AOD{aod_value}"
                is_data = alice_dl.does_period_contain_data(dataset_name)
                run_prefix = "000" if is_data else ""
                data_or_sim_str = "data" if is_data else "sim"

                # Determine the local run_by_run dir
                # If the pass value is "" or "pass1", nothing is added. But if there is more,
                # such as "pass1_CENT_woSDD", the _last_identifier will be (for example) "LHC17p_CENT_woSDD"
                _last_identifier = "_".join([dataset_name, *_pass_value.split("_")[1:]])
                local_run_by_run_dir = local_train_dir / "run_by_run" / _last_identifier

                # We need to get all runs possible runs to check for outputs of interest.
                base_dataset_path = Path(f"/alice/{data_or_sim_str}/{year_from_dataset(dataset_name)}/") / dataset_name
                # Add pt hard bins if requested
                n_pt_hard_bins = config.get("n_pt_hard_bins", None)
                if n_pt_hard_bins:
                    pt_hard_bin_generator = range(1, n_pt_hard_bins + 1)
                else:
                    pt_hard_bin_generator = range(0, 1)

                for possible_pt_hard_bin in pt_hard_bin_generator:
                    dataset_path = base_dataset_path
                    # If the pt hard bin is > 0, then it's valid.
                    if possible_pt_hard_bin > 0:
                        dataset_path = base_dataset_path / str(possible_pt_hard_bin)

                    # NOTE: run_numbers contain the run_prefix by default because we're actually querying alien.
                    run_numbers = [r for r in alice_utils.list_alien_dir(str(dataset_path)) if r.isdigit()]

                    for run_number in run_numbers:
                        # If train output exists for this run number, we expect it to be at this path.
                        # Example dir: `/alice/data/2018/LHC18q/000296934/pass3/AOD252/PWGJE/Jets_EMC_PbPb/6989_...`
                        # MC example: `/alice/sim/2020/LHC20g4/1/295612/PWGHF/HF_TreeCreator/568_20210122-0859/`
                        _possible_train_output_dir_for_run = dataset_path / run_number
                        if pass_value != "Path":
                            _possible_train_output_dir_for_run /= pass_value
                        _possible_train_output_dir_for_run /= Path(PWG) / train_name
                        if likely_child_directory:
                            _possible_train_output_dir_for_run = (
                                _possible_train_output_dir_for_run / likely_child_directory
                            )

                        logger.debug(f"_possible_train_output_dir_for_run: {_possible_train_output_dir_for_run}")

                        _directories_with_data = [
                            d for d in alice_utils.list_alien_dir(_possible_train_output_dir_for_run) if d.isdigit()
                        ]
                        if _directories_with_data:
                            logger.info(f"Found output for run number: {run_number}")
                            logger.debug(f"run_number: {run_number}, _directories_with_data: {_directories_with_data}")
                            for _directory_with_data in _directories_with_data:
                                # Setup
                                _directory_with_data_full_path = (
                                    _possible_train_output_dir_for_run / _directory_with_data
                                )

                                # Check for AnalysisResults.root
                                # Skip this check for the sake of efficiency. It seems quite rare for the directory to be created,
                                # but not to contain AnalysisResults.root . Perhaps it practically never happens.
                                if False:
                                    _dir_contents = alice_utils.list_alien_dir(_directory_with_data_full_path)
                                    if "AnalysisResults.root" not in _dir_contents:
                                        logger.debug(
                                            f"Run number: {run_number}, directory {_directory_with_data} doesn't contain an AnalysisResults.root. Skipping..."
                                        )
                                        continue

                                # Add to the queue
                                alien_file = _directory_with_data_full_path / "AnalysisResults.root"
                                # Local filename
                                # We use the integer defined in grabbing the train contents.
                                # NOTE: These numbers don't have to be continuous, especially if there was a low success rate for the train.
                                i = int(_directory_with_data_full_path.name)
                                # label = f"{int(run_number)}.{i:03}"
                                label = f"{i:03}"
                                # We use a different directory structure than standard because we're most likely to merge these afterwards.
                                local_file = local_run_by_run_dir / str(int(run_number))
                                if possible_pt_hard_bin > 0:
                                    local_file /= str(possible_pt_hard_bin)
                                local_file /= f"AnalysisResults.{child_label}.{label}.root"
                                logger.debug(f"Adding alien://{alien_file} : {local_file}")
                                output[str(alien_file)] = str(local_file)
                        else:
                            logger.debug(f"No train output found for run {run_number}")

            else:
                alien_xml_file = alien_dir / f"{stage_to_download}.xml"
                local_xml_output_filename = stage_to_download
                if child_name != "_":
                    local_xml_output_filename = f"{stage_to_download}_{child_name}"
                local_xml_file = local_train_dir / f"{local_xml_output_filename}.xml"
                result = add_files_from_xml_file(
                    alien_xml_file=alien_xml_file,
                    local_xml_file=local_xml_file,
                    local_train_dir=local_train_dir,
                    child_label=child_label,
                )
                output.update(result)

                ## Download the XML file...
                # print(f"Downloading {stage_to_download}.xml file: alien://{alien_xml_file} to {local_xml_file}")
                # process = subprocess.run(
                #    ["alien_cp", f"alien://{str(alien_xml_file)}", str(local_xml_file)],
                #    stdout=subprocess.PIPE,
                #    stderr=subprocess.PIPE,
                # )

                ## Open local XML file
                # tree = ET.parse(str(local_xml_file))
                # root = tree.getroot()
                # collection = root[0]
                ## Extract the filenames
                # for node in collection:
                #    try:
                #        lfn = node[0].attrib["lfn"]
                #        alien_file = lfn.replace("root_archive.zip", "AnalysisResults.root")
                #        label = int(node.attrib["name"])
                #        local_file = local_train_dir / f"AnalysisResults.{child_label}.{label:03}.root"
                #        # print(f"Adding alien://{alien_file} : {local_file}")
                #        output[str(alien_file)] = str(local_file)
                #        # print(f"Downloading alien://{alien_file} to {local_file}")
                #        # process = subprocess.run(
                #        #    ["alien_cp", f"alien://{str(alien_file)}", str(local_file)],
                #        #    stdout = subprocess.PIPE, stderr = subprocess.PIPE
                #        # )
                #    except IndexError:
                #        pass

    # Write out the files
    with open("files_to_download.yaml", "w") as f:
        y.dump(output, f)


def entry_point() -> None:
    helpers.setup_logging(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Download LEGO train outputs.")
    parser.add_argument(
        "--train",
        type=int,
        help="Single train to process, or first train to process.",
    )
    parser.add_argument(
        "--maxTrain",
        type=int,
        default=0,
        help="Max train number. It will include all trains between train and maxTrain (ie. upper limit is inclusive).",
    )
    args = parser.parse_args()

    # Determine what to download
    trains = []
    if not args.maxTrain:
        trains = [args.train]
    else:
        # NOTE: We add +1 so that we're inclusive on this upper limit. I think this will be more intuitive.
        trains = list(range(args.train, args.maxTrain + 1))

    logger.info(f"Downloading trains: {trains}")

    download(trains=trains)


if __name__ == "__main__":
    entry_point()
