"""Run mammoth skimming and analysis tasks via parsl

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import datetime
import enum
import functools
import logging
import secrets
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

import IPython
import attrs
from mammoth import helpers, job_utils
from mammoth.framework import sources
from pachyderm import yaml
from parsl.app.app import python_app
from parsl.data_provider.files import File
from parsl.dataflow.futures import AppFuture
from rich.progress import Progress

import jet_substructure.base.helpers
from jet_substructure.base import job_utils as substructure_job_utils, skim_analysis_objects


logger = logging.getLogger(__name__)


def _git_hash_from_module(module: Any) -> str:
    """Retrieve the git hash from a particular module

    Adapted from: https://stackoverflow.com/a/21901260/12907985

    Note:
        This assumes it is stored in a git repository, and it doesn't check.

    Args:
        module: Module to retrieve the git hist for.
    Returns:
        The git hash associated with the module.
    """
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=Path(module.__file__).parent.parent,
        capture_output=True
    ).stdout.decode("ascii").strip()


def _installed_python_software() -> List[str]:
    """Extract all installed python software via `pip freeze`

    Adapted from: https://stackoverflow.com/a/58013217/12907985

    NOTE:
        This doesn't really work as expected with poetry - it just points to local packages.
        However, we also have poetry.lock, so as long as we have the has of the repo, we'll
        know the versions of all of the software.

    Args:
        None
    Returns:
        List of str, which each entry specifying a package + version.
    """
    import sys
    return subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], capture_output=True
    ).stdout.decode("ascii").strip("\n").split("\n")


def _describe_production_software(production_config: Mapping[str, Any]) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    output["software"] = {}

    # We want to store the git hash of:
    # - pachyderm
    # - mammoth
    # - jet_substructure
    # To determine the location, we do something kind of lazy and import the file to determine the
    # location of the git repo
    output["software"]["hashes"] = {}
    import importlib
    for module_name in ["pachyderm", "mammoth", "jet_substructure"]:
        _m = importlib.import_module(module_name)
        output["software"]["hashes"][module_name] = _git_hash_from_module(_m)

    # We also want a full pip freeze. We'll store each package as an entry in a list
    output["software"]["packages"] = _installed_python_software()

    return output


def read_full_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Read full YAML configuration file.

    Args:
        config_path: Path to the configuration file. Default: "config/new_config.yaml".
    Returns:
        Full YAML configuration. Requires some interpretation.
    """
    if config_path is None:
        config_path = Path("config/track_skim_config.yaml")

    # Here, we use ruamel.yaml directly with the "safe" type because we the roundtrip
    # types that we usually use don't play so nicely when we rewrite a subset of the data
    # (eg. anchors don't seem to resolve correctly because we only rewrite the subset, there
    # are some stray comments that we don't really want to keep, etc)
    import ruamel.yaml
    y = ruamel.yaml.YAML(typ="safe")
    with open(config_path, "r") as f:
        full_config: Dict[str, Any] = y.load(f)

    return full_config


class SplittingsSelection(enum.Enum):
    recursive = 0
    iterative = 1

    def __str__(self) -> str:
        return f"{self.name}_splittings"


def _validate_collision_system(instance: "ProductionSettings", attribute: attrs.Attribute[str], value: str) -> None:
    _possible_collision_systems = [
        "pp",
        "pythia",
        "PbPb",
        "embedPythia",
        "embed_thermal_model",
    ]
    if value not in _possible_collision_systems:
        raise ValueError(f"Invalid collisions system. Provided: {value}")


_collision_systems_with_scale_factors = ["pythia", "embed_thermal_model", "embedPythia"]

@attrs.frozen(slots=False)
class ProductionSettings:
    collision_system: str = attrs.field(validator=_validate_collision_system)
    number: int
    config: Dict[str, Any]

    @functools.cached_property
    def formatted_number(self) -> str:
        # Put some leading 0s for consistency in sorting, etc
        return f"{self.number:04}"

    @functools.cached_property
    def identifier(self) -> str:
        name = ""
        # First, handle the case of possible embedding
        signal_dataset = self.config["metadata"].get("signal_dataset")
        if signal_dataset:
            name += f"{self.config['metadata']['signal_dataset']['name']}_embedded_into"
        # Then, the production name
        name = f"{self.config['metadata']['dataset']['name']}"
        # The label
        extra_label = self.config.get("label")
        if extra_label:
            name += f"_{extra_label}"
        # New section: the analysis parameters
        # First, we want to denote a new section with an extra "__"
        name += "__"
        _analysis_settings = self.config["settings"]
        # We want particular handling for some, so we do those by hand. The rest are included automatically
        _manual_analysis_parameter_keys = ["jet_R", "splittings_selection", "min_jet_pt", "background_subtraction"]
        # Jet R
        jet_R_value = _analysis_settings["jet_R"]
        name += f"_jet_R{round(jet_R_value * 100):03}"
        # Selection of splittings
        splittings_selection_value = SplittingsSelection[_analysis_settings["splittings_selection"]]
        name += f"_{str(splittings_selection_value)}"
        # Min jet pt
        name += "_min_jet_pt"
        for k, v in _analysis_settings["min_jet_pt"].items():
            name += f"_{k}_{round(v)}"
        # Background subtraction
        name += "_background_subtraction"
        for k, v in _analysis_settings["background_subtraction"].items():
            name += f"_{k}_{str(v)}"
        # And then all the rest
        for k, v in _analysis_settings.items():
            if k in _manual_analysis_parameter_keys:
                continue
            name += f"_{k}_{str(v)}"
        # And finally, the production details
        # First, we want to denote a new section with an extra "__"
        name += "__"
        # The production number itself
        name += f"_production_{self.number}"
        # The date for good measure
        name += f"_{datetime.datetime.utcnow().strftime('%Y_%m_%d')}"
        return name

    def input_files(self) -> List[Path]:
        n_pt_hat_bins = self.config["metadata"]["dataset"].get("n_pt_hat_bins")
        if n_pt_hat_bins is not None:
            # Handle pt hat binned production
            _files_per_pt_hat = self.input_files_per_pt_hat()
            _files = []
            for _files_in_single_pt_hat in _files_per_pt_hat.values():
                _files.extend(_files_in_single_pt_hat)
            return _files

        # Otherwise, we just can blindly expand
        return jet_substructure.base.helpers.ensure_and_expand_paths(
            self.config["metadata"]["dataset"]["files"]
        )

    def input_files_per_pt_hat(self) -> Dict[int, List[Path]]:
        if self.collision_system not in ["pythia", "embedPythia", "embed_thermal_model"]:
            raise ValueError(f"Asking for input files per pt hat doesn't make sense for collision system {self.collision_system}")

        # Will be signal_dataset if embedded, but otherwise will be the standard "dataset" key
        dataset_key = "signal_dataset" if "signal_dataset" in self.config["metadata"] else "dataset"

        # +1 due to pt hat bins being 1-indexed
        _files = {}
        for pt_hat_bin in range(1, self.config["metadata"][dataset_key]["n_pt_hat_bins"] + 1):
            _files[pt_hat_bin] = jet_substructure.base.helpers.ensure_and_expand_paths(
                [
                    Path(s.format(pt_hat_bin=pt_hat_bin)) for s in
                    self.config["metadata"][dataset_key]["files"]
                ]
            )

        return _files

    @property
    def has_scale_factors(self) -> bool:
        return (
            "signal_dataset" in self.config["metadata"]
            or "n_pt_hat_bins" in self.config["metadata"]["dataset"]
        )

    @property
    def scale_factors_filename(self) -> Path:
        # Validation
        if self.collision_system not in _collision_systems_with_scale_factors:
            raise ValueError(f"Invalid collision system for extracting scale factors: {self.collision_system}")

        dataset_key = "signal_dataset" if "signal_dataset" in self.config["metadata"] else "dataset"
        # Need to go up twice to get back to the "trains" directory because the collision system
        # stored in the production config may not be the same as the dataset that we're actually
        # extracting the scale factors from (ie. embedPythia != pythia)
        return (
            self.output_dir.parent.parent
            # NOTE: The values from the config are wrapped in a string to help out mypy.
            #       Otherwise, it can't determine the type for some reason...
            / str(self.config["metadata"][dataset_key]["collision_system"])
            / str(self.config["metadata"][dataset_key]["name"])
            / "scale_factors.yaml"
        )

    def scale_factors(self) -> Dict[int, float]:
        # Validation
        if self.collision_system not in _collision_systems_with_scale_factors:
            raise ValueError(f"Invalid collision system for extracting scale factors: {self.collision_system}")

        scale_factors = substructure_job_utils.read_extracted_scale_factors(
            self.scale_factors_filename
        )
        return scale_factors

    @functools.cached_property
    def output_dir(self) -> Path:
        output_dir = Path("trains") / self.collision_system / self.formatted_number
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    @functools.cached_property
    def tasks_to_execute(self) -> List[str]:
        # Could in principle be multiple tasks.
        _tasks = []

        # Need scale factors
        if self.collision_system in ["pythia", "embedPythia"]:
            _tasks.append("extract_scale_factors")

        # Skim task
        _base_name = "calculate_{label}_skim"
        _label_map = {
            "pp": "data",
            "pythia": "data",
            "PbPb": "data",
            "embedPythia": "embed_pythia",
            "embed_thermal_model": "embed_thermal_model",
        }
        _tasks.append(
            _base_name.format(label=_label_map[self.collision_system])
        )
        return _tasks

    def store_production_parameters(
        self
    ) -> None:
        output: Dict[str, Any] = {}
        output["identifier"] = self.identifier
        output["date"] = datetime.datetime.utcnow().strftime('%Y-%m-%d')
        output["config"] = dict(self.config)
        output["input_filenames"] = [
            str(p) for p in self.input_files()
        ]
        if "signal_dataset" in self.config["metadata"]:
            output["signal_filenames"] = [
                str(_filename)
                for filenames in self.input_files_per_pt_hat().values()
                for _filename in filenames
            ]
        # Add description of the software
        output.update(
            _describe_production_software(production_config=self.config)
        )

        # If we've already run this production before, we don't want to overwrite the existing production.yaml
        # Instead, we want to add a new production file with the new parameters (which should be the same as before,
        # except for the production date).
        # In order to avoid overwriting, we try adding an additional index to the filename.
        # 100 is arbitrarily selected, but I see it as highly unlikely that we would have 100 productions...
        for _additional_production_number in range(0, 100):
            _production_filename = self.output_dir / "production.yaml"
            # No need for an index for the first file.
            if _additional_production_number > 0:
                _production_filename = _production_filename.parent / f"production_{_additional_production_number}.yaml"

            if _production_filename.exists():
                # Don't overwrite the production file
                continue
            else:
                y = yaml.yaml()
                with open(_production_filename, "w") as f:
                    y.dump(output, f)

                # We've written, so no need to loop anymore
                break

    @classmethod
    def read_config(cls, collision_system: str, number: int, track_skim_config_filename: Optional[Path] = None) -> "ProductionSettings":
        track_skim_config = read_full_config(track_skim_config_filename)
        config = track_skim_config["productions"][collision_system][number]

        return cls(
            collision_system=collision_system,
            number=number,
            config=config,
        )


def safe_output_filename_from_relative_path(filename: Path, output_dir: Path) -> str:
    """Safe and identifiable name for naming output files based on the relative path.

    Converts: "2111/run_by_run/LHC17p_CENT_woSDD/282341/AnalysisResults.17p.001.root"
           -> "2111__run_by_run__LHC17p_CENT_woSDD__282341__AnalysisResults_17p_001"

    Returns:
        Filename that is safe for using as the output filename.
    """
    # NOTE: We use the grandparent of the output dir because the input filename is going to be a different train
    #       than our output. For the case of embedding trains, we might not even share the collision system.
    #       So by going to the grandparent (ie `trains`), we end up with a shared path
    return str(
        filename.relative_to(output_dir.parent.parent).with_suffix("")
    ).replace("/", "__").replace(".", "_")


@python_app  # type: ignore
def _extract_scale_factors_from_hists(
    list_name: str,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
) -> AppFuture:
    """
    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.
    """
    from pathlib import Path

    from jet_substructure.base import skim_analysis_objects
    from jet_substructure.cpp import scale_factors as sf

    res = skim_analysis_objects.ScaleFactor.from_hists(
        *sf.scale_factor_ROOT(
            filenames=[Path(i.filepath) for i in inputs], list_name=list_name
        )
    )
    return res


def setup_extract_scale_factors(
    production: ProductionSettings,
) -> Dict[int, AppFuture]:
    """Extract scale factors from embedding or pythia hists.

    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.

    Note:
        This is surprisingly fast.
    """
    # Setup
    scale_factors = {}
    logger.info("Determining input files for extracting scale factors.")
    input_files_per_pt_hat_bin = production.input_files_per_pt_hat()

    dataset_key = "signal_dataset" if "signal_dataset" in production.config["metadata"] else "dataset"
    for pt_hat_bin, input_files in input_files_per_pt_hat_bin.items():
        logger.debug(f"pt_hat_bin: {pt_hat_bin}, filenames: {input_files}")
        if input_files:
            scale_factors[pt_hat_bin] = _extract_scale_factors_from_hists(
                inputs=[File(str(fname)) for fname in input_files],
                list_name=production.config["metadata"][dataset_key]["list_name"],
            )

    return scale_factors


@python_app  # type: ignore
def _write_scale_factors_to_yaml(
    scale_factors: Mapping[int, skim_analysis_objects.ScaleFactor],
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> AppFuture:
    """
    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.
    """
    from pathlib import Path

    from pachyderm import yaml

    from jet_substructure.base import skim_analysis_objects

    # Write them to YAML for later.
    y = yaml.yaml(classes_to_register=[skim_analysis_objects.ScaleFactor])
    output_dir = Path(outputs[0].filepath)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(output_dir, "w") as f:
        y.dump(scale_factors, f)

    return True


def setup_write_scale_factors(
    production: ProductionSettings,
    scale_factors: Mapping[int, AppFuture],
) -> AppFuture:
    """
    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.
    """
    """Write scale factors to YAML and to trees if necessary."""
    # First, we write to YAML.
    # We want to do this regardless of potentially writing the scale factor trees.
    logger.info("Writing scale factors to YAML. Jobs are executing, so this will take a minute...")
    output_filename = production.scale_factors_filename
    parsl_output_file = File(str(output_filename))
    # NOTE: I'm guessing passing this is a problem because it's a class that's imported in an app, and then
    #       we're trying to pass the result into another app. I think we can go one direction or the other,
    #       but not both. So we just take the result.
    yaml_result = _write_scale_factors_to_yaml(
        scale_factors={k: v.result() for k, v in scale_factors.items()},
        outputs=[parsl_output_file],
    )

    return yaml_result


@python_app  # type: ignore
def _extract_pt_hat_spectra(
    scale_factors: Mapping[int, float],
    offsets: Mapping[int, int],
    list_name: str,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
    stdout: Optional[str] = None,
    stderr: Optional[str] = None,
) -> AppFuture:
    """
    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.

    Args:
        scale_factors: pt_hat_bin to scale factor.
        offsets: pt_hat_bin to index where files in that pt hard bin start.
    """
    from pathlib import Path

    from jet_substructure.cpp import scale_factors as sf

    # Convert back from parsl inputs
    offsets_values = list(offsets.values())
    filenames = {
        pt_hat_bin: [
            Path(f.filepath) for f in inputs[sum(offsets_values[:i]) : sum(offsets_values[: i + 1])]  # noqa: E203
        ]
        for i, pt_hat_bin in enumerate(offsets)
    }

    res = sf.pt_hard_spectra_from_hists(
        filenames=filenames,
        scale_factors=scale_factors,
        list_name=list_name,
        output_filename=Path(outputs[0].filepath),
    )
    return res


def setup_check_pt_hat_spectra(
    production: ProductionSettings,
    input_results: Sequence[AppFuture],
) -> AppFuture:
    """
    Copied from jet_substructure.analysis.parsl. The interface is slightly modified,
    but the main functionality is the same.
    """
    logger.info("Checking pt hat spectra")
    # Input files
    input_files_per_pt_hat_bin = production.input_files_per_pt_hat()

    # Need a hard dependency on the writing of the yaml output, so we ask for the result here.
    # We don't actually care about the result, but it avoids a race condition.
    _ = input_results[0].result()
    # Must read the scale factors from file to get the properly scaled values.
    scale_factors = production.scale_factors()

    # Convert inputs to Parsl files.
    # Needs to be a list, so flatten them, and then unflatten in the App.
    parsl_files = []
    offsets = {}
    for pt_hat_bin, list_of_files in input_files_per_pt_hat_bin.items():
        converted_filenames = [File(str(f)) for f in list_of_files]
        offsets[pt_hat_bin] = len(converted_filenames)
        parsl_files.extend(converted_filenames)
    # Add the dependency. We won't actually open the file in the task, but this will provide explicit dependence.
    parsl_files.extend([i.outputs[0] for i in input_results])

    dataset_key = "signal_dataset" if "signal_dataset" in production.config["metadata"] else "dataset"
    # We want to store it in the same directory as the scale factors, so it's easiest to just grab that filename.
    output_filename = production.scale_factors_filename.parent / "pt_hat_spectra.yaml"
    results = _extract_pt_hat_spectra(
        scale_factors=scale_factors,
        offsets=offsets,
        list_name=production.config["metadata"][dataset_key]["list_name"],
        inputs=parsl_files,
        outputs=[File(str(output_filename))],
    )

    return results


def steer_extract_scale_factors(
    production: ProductionSettings,
) -> List[AppFuture]:
    # Validation
    if production.collision_system not in _collision_systems_with_scale_factors:
        raise ValueError(f"Invalid collision system for extracting scale factors: {production.collision_system}")

    # Attempt to bail out early if it's already been extracted
    scale_factors_filename = production.scale_factors_filename
    if scale_factors_filename.exists():
        scale_factors = production.scale_factors()
        # We check if it's non-zero to avoid the case where it's accidentally empty
        if scale_factors:
            logger.debug("Scale factors already exist. Skipping extracting them again!")
            return []
    logger.info("Extracting scale factors...")

    # First, we need to extract the scale factors and keep track of the results
    all_results: List[AppFuture] = []
    scale_factors = setup_extract_scale_factors(production=production)
    all_results.extend(list(scale_factors.values()))
    # Then, we need to write them
    all_results.append(
        setup_write_scale_factors(production=production, scale_factors=scale_factors)
    )
    # And then create the spectra (and plot them) to cross check the extraction
    all_results.append(
        setup_check_pt_hat_spectra(
            production=production,
            # Need the future which writes the YAML
            input_results=[all_results[-1]],
        )
    )

    if all_results[-1].result() != True:
        logger.warning("Some issue with the scale factor extraction! Check on them!")
        IPython.start_ipython(user_ns={**locals(), **globals()})
        # We want to stop here, so help ourselves out by raising the exception.
        raise ValueError("Some issue with the scale factor extraction!")

    # NOTE: We don't want to scale_factor futures because they contain the actual scale factors.
    #       We'll just return the futures associated with writing to the file and the pt hat spectra cross check.
    return all_results[-2:]


@python_app  # type: ignore
def _run_data_skim(
    collision_system: str,
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    iterative_splittings: bool,
    background_subtraction: Mapping[str, Any],
    loading_data_rename_prefix: Mapping[str, str],
    convert_data_format_prefixes: Mapping[str, str],
    scale_factors: Mapping[int, float],
    pt_hat_bin: int,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> AppFuture:
    import traceback
    from pathlib import Path

    from jet_substructure.analysis import track_skim_adapter

    try:
        result = track_skim_adapter.hardest_kt_data_skim(
            input_filename=Path(inputs[0].filepath),
            collision_system=collision_system,
            jet_R=jet_R,
            min_jet_pt=min_jet_pt,
            iterative_splittings=iterative_splittings,
            background_subtraction=background_subtraction,
            loading_data_rename_prefix=loading_data_rename_prefix,
            convert_data_format_prefixes=convert_data_format_prefixes,
            scale_factors=scale_factors,
            pt_hat_bin=pt_hat_bin,
            output_filename=Path(outputs[0].filepath),
        )
    except Exception:
        result = (
            False,
            f"failure for {collision_system}, R={jet_R}, {inputs[0].filepath} with: \n{traceback.format_exc()}",
        )
    return result


def setup_calculate_data_skim(
    production: ProductionSettings,
) -> List[AppFuture]:
    """Create futures to produce hardest kt data skim"""
    # Setup input and output
    # Need to handle pt hat bin productions differently than standard productions
    # since we need to keep track of the pt hat bin
    if "n_pt_hat_bins" in production.config["metadata"]["dataset"]:
        input_files: Dict[int, List[Path]] = production.input_files_per_pt_hat()
    else:
        input_files = {-1: production.input_files()}
    output_dir = production.output_dir / "skim"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup for analysis and dataset settings
    _metadata_config = production.config["metadata"]
    _analysis_config = production.config["settings"]
    # Splitting selection (iterative vs recursive)
    splittings_selection = SplittingsSelection[_analysis_config["splittings_selection"]]
    # Scale factors
    scale_factors = None
    if production.has_scale_factors:
        scale_factors = production.scale_factors()

    results = []
    _file_counter = 0
    for pt_hat_bin, input_filenames in input_files.items():
        for input_filename in input_filenames:
            if _file_counter % 500 == 0:
                logger.info(f"Adding {input_filename} for analysis")

            # For testing...
            #if _file_counter > 1:
            #    break
            # END

            # Setup file I/O
            # Converts: "2111/run_by_run/LHC17p_CENT_woSDD/282341/AnalysisResults.17p.001.root"
            #        -> "2111__run_by_run__LHC17p_CENT_woSDD__282341__AnalysisResults_17p_001"
            output_identifier = safe_output_filename_from_relative_path(filename=input_filename,
                                                                        output_dir=production.output_dir)
            output_filename = output_dir / f"{output_identifier}_{str(splittings_selection)}.root"
            # And create the tasks
            results.append(
                _run_data_skim(
                    collision_system=production.collision_system,
                    jet_R=_analysis_config["jet_R"],
                    min_jet_pt=_analysis_config["min_jet_pt"],
                    iterative_splittings=splittings_selection == SplittingsSelection.iterative,
                    background_subtraction=_analysis_config.get("background_subtraction", {}),
                    loading_data_rename_prefix=_metadata_config["loading_data_rename_prefix"],
                    convert_data_format_prefixes=_metadata_config["convert_data_format_prefixes"],
                    inputs=[File(str(input_filename))],
                    outputs=[File(str(output_filename))],
                    pt_hat_bin=pt_hat_bin,
                    scale_factors=scale_factors,
                )
            )

            _file_counter += 1

    return results


@python_app  # type: ignore
def _run_embedding_skim(
    collision_system: str,
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    iterative_splittings: bool,
    background_subtraction: Mapping[str, Any],
    det_level_artificial_tracking_efficiency: float,
    convert_data_format_prefixes: Mapping[str, str],
    scale_factor: float,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> AppFuture:
    import traceback
    from pathlib import Path

    from jet_substructure.analysis import track_skim_adapter

    try:
        result = track_skim_adapter.hardest_kt_embedding_skim(
            collision_system=collision_system,
            signal_input=[
                Path(_input_file.filepath)
                for _input_file in inputs[:-1]
            ],
            background_input_filename=Path(inputs[-1].filepath),
            convert_data_format_prefixes=convert_data_format_prefixes,
            jet_R=jet_R,
            min_jet_pt=min_jet_pt,
            iterative_splittings=iterative_splittings,
            background_subtraction=background_subtraction,
            det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
            output_filename=Path(outputs[0].filepath),
            scale_factor=scale_factor,
        )
    except Exception:
        result = (
            False,
            f"failure for {collision_system}, R={jet_R}, signal={[_f.filepath for _f in inputs[:-1]]}, background={inputs[1].filepath} with: \n{traceback.format_exc()}",
        )
    return result


def setup_calculate_embed_pythia_skim(
    production: ProductionSettings,
) -> List[AppFuture]:
    """Create futures to produce hardest kt embedded pythia skim"""
    # Setup input and output
    output_dir = production.output_dir / "skim"
    output_dir.mkdir(parents=True, exist_ok=True)
    input_files = production.input_files()
    # We store the signal input files in a few different formats to enable sampling different ways.
    # We can sample pt hat bins equally by sampling the pt hat bin, and then taking a random file
    # from that bin. In this case, the pythia files _are not_ sampled equally.
    signal_input_files_per_pt_hat = production.input_files_per_pt_hat()
    # And since we would sample the pt hat bins, it's better to keep track of them directly.
    pt_hat_bins = list(signal_input_files_per_pt_hat)
    # Or alternatively, we sample the pythia files directly. In this case, the PYTHIA files are sampled
    # evenly, while the pt hat bins are not (we will sample the higher pt hat bins more because there
    # are more statistics).
    # To do this sampling, we need to flatten out the list of signal input files.
    # We also store the pt hat bin since we need that for the grabbing the right scale factor.
    signal_input_files_flat = [
        (_pt_hat_bin, _signal_path)
        for _pt_hat_bin, _signal_paths in signal_input_files_per_pt_hat.items()
        for _signal_path in _signal_paths
    ]

    # Setup for analysis and dataset settings
    _metadata_config = production.config["metadata"]
    # Sample the pt hat equally, or directly sample the signal_input_files
    sample_each_pt_hat_bin_equally = _metadata_config["sample_each_pt_hat_bin_equally"]
    n_signal_files_to_provide = _metadata_config["n_signal_files_to_provide"]
    _analysis_config = production.config["settings"]

    # Splitting selection (iterative vs recursive)
    splittings_selection = SplittingsSelection[_analysis_config["splittings_selection"]]
    # Scale factors
    scale_factors = None
    if production.has_scale_factors:
        scale_factors = production.scale_factors()
    else:
        raise ValueError("Check the embedding config - you need a signal dataset.")

    # Cross check
    if set(scale_factors) != set(pt_hat_bins):
        raise ValueError(
            f"Mismatch between the pt hat bins in the scale factors ({set(scale_factors)}) and the pt hat bins ({set(pt_hat_bins)})"
        )

    results = []
    _embedding_file_pairs = {}
    for _file_counter, input_filename in enumerate(input_files):
        if _file_counter % 500 == 0:
            logger.info(f"Adding {input_filename} for analysis")

        # For testing...
        #if _file_counter > 1:
        #    break
        # END

        # Randomly select (in some manner) an input file to match up with the background input file.
        # NOTE: The signal input file will repeat if there are more background events.
        #       So far, this doesn't seem to be terribly common, but even if it was, it
        #       would be perfectly fine as long as it doesn't happen too often.
        signal_input = []
        if sample_each_pt_hat_bin_equally:
            # Each pt hat bin will be equally likely, and then we select the file from
            # those which are available.
            # NOTE: This doesn't mean that the embedded statistics will still be the same in the end.
            #       For example, if I compare a low and high pt hat bin, there are just going to be
            #       more accepted jets in the high pt hat sample.
            pt_hat_bin = secrets.choice(pt_hat_bins)
            signal_input = [
                secrets.choice(signal_input_files_per_pt_hat[pt_hat_bin])
                for _ in range(n_signal_files_to_provide)
            ]
        else:
            # Directly sample the files. This probes the generator stats because
            # the number of files is directly proportional to the generated statistics.
            pt_hat_bin, _signal_input_filename = secrets.choice(signal_input_files_flat)
            signal_input = [_signal_input_filename]
            # Since we want to keep the same pt hat bin, use the pt hat ban to randomly select additional files
            signal_input.extend([
                secrets.choice(signal_input_files_per_pt_hat[pt_hat_bin])
                # -1 since we already have a filename
                for _ in range(n_signal_files_to_provide - 1)
            ])

        # Setup file I/O
        # We want to identify as: "{signal_identifier}__embedded_into__{background_identifier}"
        # Take the first signal filename as the main identifier to the path.
        # Otherwise, the filename could become indefinitely long... (apparently there are file length limits in unix...)
        output_identifier = safe_output_filename_from_relative_path(filename=signal_input[0], output_dir=production.output_dir)
        output_identifier += "__embedded_into__"
        output_identifier += safe_output_filename_from_relative_path(filename=input_filename, output_dir=production.output_dir)
        #logger.info(f"output_identifier: {output_identifier}")
        output_filename = output_dir / f"{output_identifier}_{str(splittings_selection)}.root"

        # Store the file pairs for our records
        # The output identifier contains the first signal filename, as well as the background filename.
        # We use it here rather than _just_ the background filename because we may embed into data multiple times
        # NOTE: In principle, if we're unlucky, we could overwrite entries here. However, it should be quite unlikely
        #       given the large number of files that are embedded. So I won't worry about it for now (as of March 2022).
        _embedding_file_pairs[output_identifier] = [str(_filename) for _filename in signal_input]

        # And create the tasks
        results.append(
            _run_embedding_skim(
                collision_system=production.collision_system,
                jet_R=_analysis_config["jet_R"],
                min_jet_pt=_analysis_config["min_jet_pt"],
                iterative_splittings=splittings_selection == SplittingsSelection.iterative,
                background_subtraction=_analysis_config["background_subtraction"],
                det_level_artificial_tracking_efficiency=_analysis_config["det_level_artificial_tracking_efficiency"],
                convert_data_format_prefixes=_metadata_config["convert_data_format_prefixes"],
                inputs=[
                    *[File(str(_filename)) for _filename in signal_input],
                    File(str(input_filename)),
                ],
                outputs=[
                    File(str(output_filename))
                ],
                scale_factor=scale_factors[pt_hat_bin],
            )
        )

    # And write the file pairs, again for our records
    y = yaml.yaml()
    embedding_file_pairs_filename = production.output_dir / "embedding_file_pairs.yaml"
    _existing_embedding_file_pairs = {}
    if embedding_file_pairs_filename.exists():
        with open(embedding_file_pairs_filename, "r") as f:
            _existing_embedding_file_pairs = y.load(f)
    # Add back in the existing file pairs if we've read them
    if _existing_embedding_file_pairs:
        _embedding_file_pairs.update(_existing_embedding_file_pairs)
    # And then (re)write the file pairs
    with open(embedding_file_pairs_filename, "w") as f:
        y.dump(_embedding_file_pairs, f)

    return results

@python_app  # type: ignore
def _run_embed_thermal_model_skim(
    collision_system: str,
    jet_R: float,
    min_jet_pt: Mapping[str, float],
    iterative_splittings: bool,
    background_subtraction: Mapping[str, Any],
    det_level_artificial_tracking_efficiency: float,
    thermal_model_parameters: sources.ThermalModelParameters,
    convert_data_format_prefixes: Mapping[str, str],
    scale_factor: float,
    inputs: Sequence[File] = [],
    outputs: Sequence[File] = [],
) -> AppFuture:
    import traceback
    from pathlib import Path

    from jet_substructure.analysis import track_skim_adapter

    try:
        result = track_skim_adapter.hardest_kt_embed_thermal_model_skim(
            collision_system=collision_system,
            signal_input=[
                Path(_input_file.filepath)
                for _input_file in inputs
            ],
            convert_data_format_prefixes=convert_data_format_prefixes,
            jet_R=jet_R,
            min_jet_pt=min_jet_pt,
            iterative_splittings=iterative_splittings,
            background_subtraction=background_subtraction,
            det_level_artificial_tracking_efficiency=det_level_artificial_tracking_efficiency,
            thermal_model_parameters=thermal_model_parameters,
            output_filename=Path(outputs[0].filepath),
            scale_factor=scale_factor,
        )
    except Exception:
        result = (
            False,
            f"failure for {collision_system}, R={jet_R}, signal={[_f.filepath for _f in inputs]} with: \n{traceback.format_exc()}",
        )
    return result


def setup_calculate_embed_thermal_model_skim(
    production: ProductionSettings,
) -> List[AppFuture]:
    """Create futures to produce hardest kt embedded pythia skim"""
    # Setup input and output
    # Need to handle pt hat bin productions differently than standard productions
    # since we need to keep track of the pt hat bin
    if "n_pt_hat_bins" in production.config["metadata"]["dataset"]:
        input_files: Dict[int, List[Path]] = production.input_files_per_pt_hat()
    else:
        input_files = {-1: production.input_files()}
        raise RuntimeError("Need pt hat production for embedding into a thermal model...")
    output_dir = production.output_dir / "skim"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup for analysis and dataset settings
    _metadata_config = production.config["metadata"]
    _analysis_config = production.config["settings"]
    # Splitting selection (iterative vs recursive)
    splittings_selection = SplittingsSelection[_analysis_config["splittings_selection"]]
    thermal_model_parameters = sources.THERMAL_MODEL_SETTINGS[_analysis_config["event_activity"]]
    # Scale factors
    scale_factors = None
    if production.has_scale_factors:
        scale_factors = production.scale_factors()
    else:
        raise ValueError("Check the thermal model config - you need a signal dataset.")

    # Cross check
    if set(scale_factors) != set(list(input_files)):
        raise ValueError(
            f"Mismatch between the pt hat bins in the scale factors ({set(scale_factors)}) and the pt hat bins ({set(list(input_files))})"
        )

    results = []
    _file_counter = 0
    # Reversed because the higher pt hard bins are of more importance to get done sooner.
    for pt_hat_bin, input_filenames in reversed(input_files.items()):
        for input_filename in input_filenames:
            if _file_counter % 500 == 0:
                logger.info(f"Adding {input_filename} for analysis")

            # For testing...
            #if _file_counter > 1:
            #    break
            # END

            # Setup file I/O
            # Converts: "2111/run_by_run/LHC17p_CENT_woSDD/282341/AnalysisResults.17p.001.root"
            #        -> "2111__run_by_run__LHC17p_CENT_woSDD__282341__AnalysisResults_17p_001"
            output_identifier = safe_output_filename_from_relative_path(filename=input_filename,
                                                                        output_dir=production.output_dir)
            output_filename = output_dir / f"{output_identifier}_{str(splittings_selection)}.root"
            # And create the tasks
            results.append(
                _run_embed_thermal_model_skim(
                    collision_system=production.collision_system,
                    jet_R=_analysis_config["jet_R"],
                    min_jet_pt=_analysis_config["min_jet_pt"],
                    iterative_splittings=splittings_selection == SplittingsSelection.iterative,
                    background_subtraction=_analysis_config["background_subtraction"],
                    det_level_artificial_tracking_efficiency=_analysis_config["det_level_artificial_tracking_efficiency"],
                    thermal_model_parameters=thermal_model_parameters,
                    convert_data_format_prefixes=_metadata_config["convert_data_format_prefixes"],
                    inputs=[
                        File(str(input_filename)),
                    ],
                    outputs=[
                        File(str(output_filename))
                    ],
                    scale_factor=scale_factors[pt_hat_bin],
                )
            )

            _file_counter += 1

    return results


#def setup_calculate_thermal_model_skim(
#    collision_system: str,
#    min_jet_pt: Union[float, Mapping[str, float]],
#    jet_R_values: Sequence[float],
#    iterative_splittings: bool,
#    convert_data_format_prefixes: Mapping[str, str],
#    input_path: Path,
#    event_activity: str,
#    scale_factors_dataset: str,
#    r_max: float = 0.25,
#    n_repeat_file: int = 1,
#) -> List[AppFuture]:
#    """Analyze and skim hardest kt embedding"""
#    # NOTE: These are pythia input files. They include an extra "*" to account for the pt hard bin...
#    input_files = sorted(input_path.glob("*/*/*/*.root"))
#
#    # TEMP for testing
#    #input_files = input_files[:1]
#    #input_files = input_files[:4]
#    #input_files = input_files[:10]
#    # ENDTEMP
#
#    # NOTE: Delayed import since this comes with _a lot_ of other things. Should be refactored...
#    import jet_substructure.analysis.parsl
#    scale_factors = jet_substructure.analysis.parsl.read_extracted_scale_factors(
#        # TODO: Unclear if this should be hard coded
#        collision_system="embedPythia",
#        dataset_name=scale_factors_dataset,
#    )
#
#    thermal_model_parameters = sources.THERMAL_MODEL_SETTINGS[event_activity]
#
#    results = []
#    for i, input_filename in enumerate(input_files):
#        if i % 500 == 0:
#            logger.info(f"Adding {input_filename} for analysis")
#
#        # The input_file is in trains/collision_system/train_number/run_by_run/period/run_number/filename.root
#        # So to get the train directory, we need to take the parent 4 times.
#        train_directory = input_filename.parent.parent.parent.parent
#        run_dir = input_filename.parent.name
#        pt_hat_bin = -1
#        # However, if we're looking at pythia, we also need to account for the pt hard bin
#        #if collision_system == "pythia":
#        if True:
#            # In this case, the input_file is in trains/collision_system/train_number/run_by_run/period/run_number/pt_hat_bin/filename.root
#            # At least for LHC20g4 and LHC18b8
#            # Thus, to get the train directory, we need to take the parent 5 times
#            train_directory = input_filename.parent.parent.parent.parent.parent
#            # The pt hard bin is the parent dir
#            pt_hat_bin = int(str(input_filename.parent.name))
#            # For the run dir, we want to include the period, run, and pt hat bin
#            # The period is 3 parents up and the run number is 2 parents up
#            run_dir = str(Path(input_filename.parent.parent.parent.name) / input_filename.parent.parent.name / str(pt_hat_bin))
#
#        # Further setup
#        iterative_splittings_label = "iterative" if iterative_splittings else "recursive"
#
#        for jet_R in jet_R_values:
#            # Setup file I/O
#            output_dir = train_directory / "skim" / collision_system / event_activity / f"R{round(jet_R * 10):02}" / run_dir
#
#            if not output_dir.exists():
#                output_dir.mkdir(parents=True, exist_ok=True)
#            for i in range(n_repeat_file):
#                # NOTE: This includes the period, run, and pt hat
#                output_filename = output_dir / f"{input_filename.stem}_{iterative_splittings_label}_splittings_{i:02}.root"
#                results.append(
#                    _run_embedding_skim(
#                        collision_system=collision_system,
#                        jet_R=jet_R,
#                        min_jet_pt=min_jet_pt,
#                        iterative_splittings=iterative_splittings,
#                        thermal_model_parameters=thermal_model_parameters,
#                        convert_data_format_prefixes=convert_data_format_prefixes,
#                        scale_factor=scale_factors[pt_hat_bin],
#                        r_max=r_max,
#                        inputs=[File(str(input_filename))],
#                        outputs=[File(str(output_filename))],
#                    )
#                )
#
#    return results


def run_old() -> None:
    # Basic setup
    iterative_splittings = True
    jet_R_values = [0.2]
    min_jet_pt = {
        "pp": 5,
        "pythia": {"det_level": 5},
        "embedPythia": 20,
        "PbPb": 20,
        "embed_thermal_model": {"hybrid": 20},
    }
    #collision_systems_to_process = ["embed_thermal_model"]
    collision_systems_to_process = ["pp"]
    dataset_name = "LHC18qr_central_642"
    event_activity = "central"

    # NOTE: Need to glob in the task
    # TODO: Moved to config...
    input_paths = {
        "pp": Path("trains/pp/2110/"),
        # TODO: This really needs to be updated!
        "pythia": Path("trains/pythia/2619/run_by_run/LHC18b8_fast/"),
        "PbPb": Path("trains/PbPb/642/run_by_run/"),
        "embed_thermal_model": Path("trains/pythia/641/run_by_run/"),
    }
    loading_data_rename_prefix = {
        "pp": {"data": "data"},
        # It has a separate function, so this isn't super meaningful.
        # However, it could be if we just wanted to look at one prefix, since we could
        # use the data loading function.
        # "pythia": {"data": "det_level", "true": "part_level"},
        "PbPb": {"data": "data"},
    }
    convert_data_format_prefixes = {
        "pp": {"data": "data"},
        # The loading data rename prefix won't apply any mapping for pythia,
        # so we have to handle it here.
        "pythia": {"det_level": "data", "part_level": "true"},
        "PbPb": {"data": "data"},
        "embed_thermal_model": {"hybrid": "hybrid", "det_level": "det_level", "part_level": "true"},
    }

    # Job execution parameters
    task_name = "hardest_kt_mammoth"
    tasks_to_execute = [
        # "calculate_data_skim"
        # "calculate_embedding_skim",
        "calculate_embed_thermal_model_skim",
    ]

    # Job execution configuration
    task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=1)
    # n_cores_to_allocate = 120
    # walltime = "1:59:00"
    n_cores_to_allocate = 80
    walltime = "24:00:00"
    n_cores_to_allocate = 2
    #n_cores_to_allocate = 10

    # Validation
    # Collision system
    _possible_collision_systems = [
        "pp",
        "pythia",
        "PbPb",
        "embedPythia",
        "embed_thermal_model",
    ]
    if not set(collision_systems_to_process).issubset(_possible_collision_systems):
        raise ValueError(f"Invalid collisions system(s) to process. Provided: {collision_systems_to_process}")

    # Basic setup: logging and parsl.
    # NOTE: Parsl's logger setup is broken, so we have to set it up before starting logging. Otherwise,
    #       it's super verbose and a huge pain to turn off. Note that by passing on the storage messages,
    #       we don't actually lose any info.
    config, facility_config, stored_messages = job_utils.config(
        facility="ORNL_b587_long",
        # facility="ORNL_b587_short",
        task_config=task_config,
        n_tasks=n_cores_to_allocate,
        walltime=walltime,
        enable_monitoring=True,
    )
    # Keep track of the dfk to keep parsl alive
    dfk = helpers.setup_logging_and_parsl(
        parsl_config=config,
        level=logging.INFO,
        stored_messages=stored_messages,
    )

    all_results = []
    for collision_system in collision_systems_to_process:
        # Collision system dependent
        scale_factors_dataset = "LHC18b8_pythia_R04_2520" if collision_system == "pythia" else ""

        # Setup tasks
        system_results = []
        if "calculate_data_skim" in tasks_to_execute:
            system_results.extend(
                setup_calculate_data_skim(
                    collision_system=collision_system,
                    event_activity=event_activity,
                    min_jet_pt=min_jet_pt[collision_system],  # type: ignore
                    jet_R_values=jet_R_values,
                    iterative_splittings=iterative_splittings,
                    loading_data_rename_prefix=loading_data_rename_prefix.get(collision_system, {}),
                    convert_data_format_prefixes=convert_data_format_prefixes[collision_system],
                    scale_factors_dataset=scale_factors_dataset,
                    input_path=input_paths[collision_system],
                )
            )
        if "calculate_thermal_model_skim" in tasks_to_execute:
            if event_activity == "central":
                scale_factors_dataset = "LHC20g4_embedded_into_LHC18qr_central_R02_6982_7001"
            elif event_activity == "semi_central":
                scale_factors_dataset = "LHC20g4_embedded_into_LHC18qr_semi_central_R02_6932_6951"
            else:
                raise RuntimeError(f"Dunno what to do with {event_activity} for the scale factors dataset. Check this...")

            #system_results.extend(
            #    setup_calculate_thermal_model_skim(
            #        collision_system=collision_system,
            #        event_activity=event_activity,
            #        min_jet_pt=min_jet_pt[collision_system],  # type: ignore
            #        jet_R_values=jet_R_values,
            #        iterative_splittings=iterative_splittings,
            #        convert_data_format_prefixes=convert_data_format_prefixes[collision_system],
            #        scale_factors_dataset=scale_factors_dataset,
            #        input_path=input_paths[collision_system],
            #        n_repeat_file=5,
            #    )
            #)

        all_results.extend(system_results)
        logger.info(f"Accumulated {len(system_results)} futures for {collision_system}")

    logger.info(f"Accumulated {len(all_results)} total futures")

    # Process the futures, showing processing progress
    # Since it returns the results, we can actually use this to accumulate results.
    gen_results = job_utils.provide_results_as_completed(all_results, running_with_parsl=True)

    # In order to support writing histograms from multiple systems, we need to index the output histograms
    # by the collision system + centrality.
    output_hists: Dict[str, Dict[Any, Any]] = {k: {} for k in collision_systems_to_process}
    with Progress(console=helpers.rich_console, refresh_per_second=1, speed_estimate_period=300) as progress:
        track_results = progress.add_task(total=len(all_results), description="Processing results...")
        # for a in all_results:
        for result in gen_results:
            # r = a.result()
            logger.info(f"result: {result[:2]}")
            if result[0] and len(result) == 4 and isinstance(result[3], dict):
                k = result[2]
                logger.info(f"Found result for key {k}")
                output_hists[k] = job_utils.merge_results(output_hists[k], result[3])
            logger.info(f"output_hists: {output_hists}")
            progress.update(track_results, advance=1)

    # Save hists to uproot (if needed)
    for system, hists in output_hists.items():
        if hists:
            import uproot

            split_system_name = system.split("_")
            # Either "pp" or "PbPb"
            collision_system = split_system_name[0]
            # Additional label for centrality when appropriate
            # NOTE: If the list is of length 1, it will be empty
            file_label = "_".join(split_system_name[1:])
            if file_label:
                file_label = f"_{file_label}"

            output_hist_filename = Path("output") / collision_system / f"hardest_kt_{file_label}.root"
            output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing output_hists to {output_hist_filename} for system {system}")
            with uproot.recreate(output_hist_filename) as f:
                helpers.write_hists_to_file(hists=hists, f=f)

    # By embedded here, we can inspect results, etc in the meantime.
    # NOTE: This may be commented out sometimes when I have long running processes and wil
    #       probably forget to close it.
    IPython.start_ipython(user_ns={**locals(), **globals()})

    # In case we close IPython early, wait for all apps to complete
    # Also allows for a summary at the end.
    # By taking only the first two, it just tells use the status and a quick message.
    # Otherwise, we can overwhelm with trying to print large objects
    res = [r.result()[:2] for r in all_results]
    logger.info(res)


def determine_additional_worker_init(productions: Sequence[ProductionSettings],
                                     tasks_requiring_root: Sequence[str],
                                     tasks_requiring_aliphysics: Sequence[str]) -> str:
    _software_to_load = []
    _additional_worker_init_script = ""
    if any(
            (
                task in tasks_requiring_root
                for prod in productions
                for task in prod.tasks_to_execute
            )
        ):
        _software_to_load.append("ROOT/latest")
    if any(
            (
                task in tasks_requiring_aliphysics
                for prod in productions
                for task in prod.tasks_to_execute
            )
        ):
        # This is a little unconventional to redefine the list here, but ROOT is already
        # a dependency of AliPhysics, so we redefine the list to remove ROOT.
        _software_to_load = [s for s in _software_to_load if s != "ROOT/latest"]
        # And then include AliPhysics
        _software_to_load.append("AliPhysics/latest")

    # If there is anything to load, add the initialization
    if _software_to_load:
        _additional_worker_init_script = f"eval `/usr/bin/alienv -w /software/rehlers/alice/sw --no-refresh printenv {','.join(_software_to_load)}`"

    return _additional_worker_init_script


def define_productions() -> List[ProductionSettings]:
    # We want to provide the opportunity to run multiple productions at once.
    # We'll do so by defining each production below and then iterating over them below
    productions = []

    # Create and store production information
    productions.extend(
        [
            # ProductionSettings.read_config(
            #     collision_system="embedPythia", number=62,
            # ),
            # ProductionSettings.read_config(
            #     collision_system="embedPythia", number=63,
            # ),
            ProductionSettings.read_config(
                collision_system="embed_thermal_model", number=60,
            ),
        ]
    )

    # Write out the production settings
    for production_settings in productions:
        production_settings.store_production_parameters()

    return productions


def run() -> None:
    # Job execution parameters
    productions = define_productions()
    task_name = "hardest_kt_mammoth"

    # Job execution configuration
    task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=2)
    # n_cores_to_allocate = 120
    #n_cores_to_allocate = 110
    n_cores_to_allocate = 50
    walltime = "24:00:00"
    #n_cores_to_allocate = 2
    #walltime = "1:59:00"
    #n_cores_to_allocate = 10

    # Basic setup: logging and parsl.
    # First, need to figure out if we need additional environments such as ROOT
    _additional_worker_init_script = determine_additional_worker_init(
        productions=productions,
        tasks_requiring_root=["extract_scale_factors"],
        tasks_requiring_aliphysics=[],
    )
    # NOTE: Parsl's logger setup is broken, so we have to set it up before starting logging. Otherwise,
    #       it's super verbose and a huge pain to turn off. Note that by passing on the storage messages,
    #       we don't actually lose any info.
    config, facility_config, stored_messages = job_utils.config(
        facility="ORNL_b587_long",
        # facility="ORNL_b587_short",
        task_config=task_config,
        n_tasks=n_cores_to_allocate,
        walltime=walltime,
        enable_monitoring=True,
        additional_worker_init_script=_additional_worker_init_script,
    )
    # Keep track of the dfk to keep parsl alive
    dfk = helpers.setup_logging_and_parsl(
        parsl_config=config,
        level=logging.INFO,
        stored_messages=stored_messages,
    )

    all_results = []
    for production in productions:
        tasks_to_execute = production.tasks_to_execute
        # Collision system dependent
        #scale_factors_dataset = "LHC18b8_pythia_R04_2520" if collision_system == "pythia" else ""

        # Setup tasks
        system_results = []
        if "extract_scale_factors" in tasks_to_execute:
            # NOTE: This will block on the result since it needs to be done before anything can proceed
            system_results.extend(
                steer_extract_scale_factors(
                    production=production,
                )
            )
        if "calculate_data_skim" in tasks_to_execute:
            system_results.extend(
                setup_calculate_data_skim(
                    production=production,
                )
            )
        if "calculate_thermal_model_skim" in tasks_to_execute:
            #system_results.extend(
            #    setup_calculate_thermal_model_skim(
            #        production=production,
            #        # collision_system=collision_system,
            #        # event_activity=event_activity,
            #        # min_jet_pt=min_jet_pt[collision_system],  # type: ignore
            #        # jet_R_values=jet_R_values,
            #        # iterative_splittings=iterative_splittings,
            #        # convert_data_format_prefixes=convert_data_format_prefixes[collision_system],
            #        # scale_factors_dataset=scale_factors_dataset,
            #        # input_path=input_paths[collision_system],
            #        # n_repeat_file=5,
            #    )
            #)
            ...
        #    if event_activity == "central":
        #        scale_factors_dataset = "LHC20g4_embedded_into_LHC18qr_central_R02_6982_7001"
        #    elif event_activity == "semi_central":
        #        scale_factors_dataset = "LHC20g4_embedded_into_LHC18qr_semi_central_R02_6932_6951"
        #    else:
        #        raise RuntimeError(f"Dunno what to do with {event_activity} for the scale factors dataset. Check this...")

        #    system_results.extend(
        #        setup_calculate_thermal_model_skim(
        #            collision_system=collision_system,
        #            event_activity=event_activity,
        #            min_jet_pt=min_jet_pt[collision_system],  # type: ignore
        #            jet_R_values=jet_R_values,
        #            iterative_splittings=iterative_splittings,
        #            convert_data_format_prefixes=convert_data_format_prefixes[collision_system],
        #            scale_factors_dataset=scale_factors_dataset,
        #            input_path=input_paths[collision_system],
        #            n_repeat_file=5,
        #        )
        #    )

        if "calculate_embed_thermal_model_skim" in tasks_to_execute:
            system_results.extend(
                setup_calculate_embed_thermal_model_skim(
                    production=production,
                )
            )
        if "calculate_embed_pythia_skim" in tasks_to_execute:
            system_results.extend(
                setup_calculate_embed_pythia_skim(
                    production=production,
                )
            )

        all_results.extend(system_results)
        logger.info(f"Accumulated {len(system_results)} futures for {production.collision_system}")

    logger.info(f"Accumulated {len(all_results)} total futures")

    # Process the futures, showing processing progress
    # Since it returns the results, we can actually use this to accumulate results.
    gen_results = job_utils.provide_results_as_completed(all_results, running_with_parsl=True)

    # In order to support writing histograms from multiple systems, we need to index the output histograms
    # by the collision system + centrality.
    output_hists: Dict[str, Dict[Any, Any]] = {_p.collision_system: {} for _p in productions}
    with Progress(console=helpers.rich_console, refresh_per_second=1, speed_estimate_period=300) as progress:
        track_results = progress.add_task(total=len(all_results), description="Processing results...")
        # for a in all_results:
        for result in gen_results:
            # r = a.result()
            logger.info(f"result: {result[:2]}")
            if result[0] and len(result) == 4 and isinstance(result[3], dict):
                k = result[2]
                logger.info(f"Found result for key {k}")
                output_hists[k] = job_utils.merge_results(output_hists[k], result[3])
            logger.info(f"output_hists: {output_hists}")
            progress.update(track_results, advance=1)

    # Save hists to uproot (if needed)
    for system, hists in output_hists.items():
        if hists:
            import uproot

            split_system_name = system.split("_")
            # Either "pp" or "PbPb"
            collision_system = split_system_name[0]
            # Additional label for centrality when appropriate
            # NOTE: If the list is of length 1, it will be empty
            file_label = "_".join(split_system_name[1:])
            if file_label:
                file_label = f"_{file_label}"

            output_hist_filename = Path("output") / collision_system / f"hardest_kt_{file_label}.root"
            output_hist_filename.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing output_hists to {output_hist_filename} for system {system}")
            with uproot.recreate(output_hist_filename) as f:
                helpers.write_hists_to_file(hists=hists, f=f)

    # By embedded here, we can inspect results, etc in the meantime.
    # NOTE: This may be commented out sometimes when I have long running processes and wil
    #       probably forget to close it.
    IPython.start_ipython(user_ns={**locals(), **globals()})

    # In case we close IPython early, wait for all apps to complete
    # Also allows for a summary at the end.
    # By taking only the first two, it just tells use the status and a quick message.
    # Otherwise, we can overwhelm with trying to print large objects
    res = [r.result()[:2] for r in all_results]
    logger.info(res)


if __name__ == "__main__":
    run()
