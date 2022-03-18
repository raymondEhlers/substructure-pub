""" Extract scale factors from all repaired files.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import fnmatch
import logging
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple

import numpy as np
import uproot
from pachyderm import binned_data, yaml


# We know already - nothing to be done...
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    import uproot3

from jet_substructure.base import helpers, skim_analysis_objects


logger = logging.getLogger(__name__)


def scale_factor_ROOT_wrapper(base_path: Path, train_number: int) -> Tuple[int, int, Any, Any]:
    # Setup
    filenames = helpers.ensure_and_expand_paths([Path(str(base_path).format(train_number=train_number))])

    return scale_factor_ROOT(filenames)


def scale_factor_ROOT(filenames: Sequence[Path], list_name: str = "") -> Tuple[int, int, Any, Any]:
    """Calculate the scale factor for a given train.

    Args:
        filenames: Filenames for output from a given train.
        list_name: Name of the list from which we will retrieve the hists.
    Returns:
        n_accepted_events, n_entries, cross_section, n_trials
    """
    # Validation
    if not list_name:
        list_name = "*DynamicalGrooming*"
    # Delay import to avoid direct dependence
    import ROOT

    cross_section_hists = []
    n_trials_hists = []
    n_entries = 0
    n_accepted_events = 0
    for filename in filenames:
        f = ROOT.TFile(str(filename), "READ")

        # Retrieve the output list. First try the embedding hists. If not available, then
        # try to find the task output.
        hists = f.Get("AliAnalysisTaskEmcalEmbeddingHelper_histos")
        # If the embedding helper isn't available, let's try to get the task output.
        if not hists:
            # If the embedding helper isn't available, try to move on to an analysis task.
            # We will search through the keys using a glob.
            task_hists_name = fnmatch.filter([k.GetName() for k in f.GetListOfKeys()], list_name)
            # And then require that "Tree" is not in the name (otherwise we're likely to put up the TTree)
            task_hists_name = [
                name for name in task_hists_name if "Tree" not in name
            ]
            if len(task_hists_name) != 1:
                raise RuntimeError(f"Cannot find unique task name. Names: {task_hists_name}. Skipping!")
            else:
                hists = f.Get(task_hists_name[0])
                if not hists:
                    raise RuntimeError(
                        f"Cannot find a task output list. Tried: {task_hists_name[0]}. Keys: {list(f.GetListOfKeys())}"
                    )

        # This list is usually an AliEmcalList. Although we don't care about any of the AliEmcalList functionality
        # here, this still requires an AliPhysics installation, which may not always be so convenient.
        # Since all we want is the TList base class, we can get ROOT to cast it into a TList.
        # NOTE: Apparently I can't do a standard isinstance check, so this will have to do...
        if "AliEmcalList" in str(type(hists)):
            # This is basically a c++ cast
            hists = ROOT.bind_object(ROOT.addressof(hists), "TList")

        cross_section_hists.append(hists.FindObject("fHistXsection"))
        cross_section_hists[-1].SetDirectory(0)
        n_entries += cross_section_hists[-1].GetEntries()
        n_trials_hists.append(hists.FindObject("fHistTrials"))
        n_trials_hists[-1].SetDirectory(0)

        # Keep track of accepted events for normalizing the scale factors later.
        n_events_hist = hists.FindObject("fHistEventCount")
        n_accepted_events += n_events_hist.GetBinContent(1)

        f.Close()

    cross_section = cross_section_hists[0]
    # Add the rest...
    [cross_section.Add(other) for other in cross_section_hists[1:]]
    n_trials = n_trials_hists[0]
    # Add the rest...
    [n_trials.Add(other) for other in n_trials_hists[1:]]

    return n_accepted_events, n_entries, cross_section, n_trials


def scale_factor_uproot_wrapper(
    base_path: Path, train_number: int, run_despite_issues: bool = False
) -> Tuple[int, int, Any, Any]:
    # Setup
    filenames = helpers.ensure_and_expand_paths([Path(str(base_path).format(train_number=train_number))])

    return scale_factor_uproot(filenames=filenames, run_despite_issues=run_despite_issues)


def scale_factor_uproot(filenames: Sequence[Path], run_despite_issues: bool = False) -> Tuple[int, int, Any, Any]:
    # Validation
    if not run_despite_issues:
        raise RuntimeError("Pachyderm binned data doesn't add profile histograms correctly...")

    # NOTE: This code is from a previous piece of code to extract scale factors. This may only work
    #       for uproot3. For now, it's not worth looking into, but I keep this around for posterity
    #       in case it is useful later.
    if False:
        # To make this code appear valid, this line is just a hack and should be ignored when looking
        # at the code as a reference.
        filename = filenames[0]

        # Setup
        input_file = uproot3.open(filename)

        # Retrieve the embedding helper to extract the cross section and ntrials.
        embedding_hists = input_file["AliAnalysisTaskEmcalEmbeddingHelper_histos"]
        h_cross_section_uproot = [h for h in embedding_hists if hasattr(h, "name") and h.name == b"fHistXsection"][0]
        h_cross_section = binned_data.BinnedData.from_existing_data(h_cross_section_uproot)
        h_n_trials = binned_data.BinnedData.from_existing_data(
            [h for h in embedding_hists if hasattr(h, "name") and h.name == b"fHistTrials"][0]
        )
        # Find the first non-zero values bin.
        # argmax will return the index of the first instance of True.
        pt_hard_bin = (h_cross_section.values != 0).argmax(axis=0)

        # The cross section is a profile hist, but we just read the raw values with uproot + binned_data. Consequently, the values
        # aren't scaled down by the number of entries in that bin (as already performed by ROOT), so we just take the
        # cross section / n_trials
        scale_factor = h_cross_section.values[pt_hard_bin] / h_n_trials.values[pt_hard_bin]
        logger.debug(f"Scale factor: {scale_factor}")

    cross_section_hists = []
    n_trials_hists = []
    n_entries_list = []
    n_accepted_events = []
    for filename in filenames:
        with uproot.open(filename) as input_file:
            # Retrieve the embedding helper to extract the cross section and ntrials.
            embedding_hists = input_file["AliAnalysisTaskEmcalEmbeddingHelper_histos"]
            cross_section_hist = [
                h for h in embedding_hists if h.has_member("fName") and h.member("fName") == "fHistXsection"
            ][0]
            n_entries_list.append(cross_section_hist.effective_entries())
            cross_section_hists.append(binned_data.BinnedData.from_existing_data(cross_section_hist))
            n_trials_hists.append(
                binned_data.BinnedData.from_existing_data(
                    [h for h in embedding_hists if h.has_member("fName") and h.member("fName") == "fHistTrials"][0]
                )
            )

            # Keep track of accepted events for normalizing the scale factors later.
            n_events_hist = binned_data.BinnedData.from_existing_data(
                [h for h in embedding_hists if h.has_member("fName") and h.member("fName") == "fHistEventCount"][0]
            )
            n_accepted_events.append(n_events_hist.values[0])

    # Convert to numpy array for help in finding first empty bin.
    n_entries = np.array(n_entries_list)

    # Take the first non-zero value of n_entries (there should only be 1)
    return (
        sum(n_accepted_events),
        n_entries[(n_entries != 0).argmax(axis=0)],
        sum(cross_section_hists),
        sum(n_trials_hists),
    )


def create_scale_factor_tree_for_cross_check_task_output(
    filename: Path,
    scale_factor: float,
) -> bool:
    """Create scale factor for a single embedded output for the cross check task.

    As of May 2021, this is deprecated, but we keep it around as an example
    """
    # Get number of entries in the tree to determine
    with uproot.open(filename) as f:
        # This should usually get us the tree name, regardless of what task actually generated it.
        # NOTE: Adding a suffix will yield "Raw{grooming_method}Tree", so instead we search for "tree"
        #       and one of the task names.
        tree_name = [k for k in f.keys() if "RawTree" in k and ("HardestKt" in k or "DynamicalGrooming" in k)][0]
        n_entries = f[tree_name].num_entries
        logger.debug(f"n entries: {n_entries}")

    # We want the scale_factor directory to be in the main train directory.
    base_dir = filename.parent
    if base_dir.name == "skim":
        # If we're in the skim dir, we need to move up one more level.
        base_dir = base_dir.parent
    output_filename = base_dir / "scale_factor" / filename.name
    output_filename.parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"Writing scale_factor to {output_filename}")
    branches = {"scale_factor": np.float32}
    with uproot3.recreate(output_filename) as output_file:
        output_file["tree"] = uproot3.newtree(branches)
        # Write all of the calculations
        output_file["tree"].extend({"scale_factor": np.full(n_entries, scale_factor, dtype=np.float32)})

    return True


def pt_hard_spectra_from_hists(
    filenames: Mapping[int, Sequence[Path]], scale_factors: Mapping[int, float], output_filename: Path,
    list_name: str = "",
) -> bool:
    """Extract and save pt hard spectra from embedding or pythia.

    This functionality is exceptional because we only have the histograms, not the tree.

    Note:
        I write to yaml using binned_data because I'm not sure errors, etc would be handled properly
        when writing the hist with uproot3.

    Args:
        filenames: Filenames as a function of pt hard bin.
        scale_factors: Pt hard scale factors.
        output_filename: Where the spectra should be saved (in yaml).
    Returns:
        True if successful.
    """
    # Validation
    if not list_name:
        list_name = "*DynamicalGrooming*"

    pt_hard_spectra = []
    for pt_hard_bin, pt_hard_filenames in filenames.items():
        single_bin_pt_hard_spectra = []
        for filename in pt_hard_filenames:
            with uproot.open(filename) as f:
                hists = f.get("AliAnalysisTaskEmcalEmbeddingHelper_histos", None)
                if not hists:
                    # If not the embedding helper, look for the analysis task output.
                    logger.debug(f"Searching for task hists with the name pattern '{list_name}'")
                    # Search for keys which contain the provided tree name. Very nicely, uproot already has this built-in
                    _possible_task_hists_names = f.keys(
                        cycle=False, filter_name=list_name, filter_classname=["AliEmcalList", "TList"]
                    )
                    if len(_possible_task_hists_names) != 1:
                        raise ValueError(
                            f"Ambiguous list name '{list_name}'. Please revise it as needed. Options: {_possible_task_hists_names}"
                        )
                    # We're good - let's keep going
                    hists = f.get(_possible_task_hists_names[0], None)

                if not isinstance(hists, uproot.models.TList.Model_TList):
                    # Grab the underlying TList rather than the AliEmcalList...
                    hists = hists.bases[0]
                single_bin_pt_hard_spectra.append(
                    binned_data.BinnedData.from_existing_data(
                        [h for h in hists if h.has_member("fName") and h.member("fName") == "fHistPtHard"][0]
                    )
                )
        h_temp = sum(single_bin_pt_hard_spectra)
        # The scale factor may not be defined if (for example) working with a test production without all pt hard bins
        # If it's not available, don't bother trying to append the spectra - it doesn't gain anything, and it's likely
        # to cause additional issues.
        _scale_factor = scale_factors.get(pt_hard_bin)
        if _scale_factor:
            pt_hard_spectra.append(h_temp * _scale_factor)

    final_spectra = sum(pt_hard_spectra)

    output_filename.parent.mkdir(exist_ok=True, parents=True)
    y = yaml.yaml(modules_to_register=[binned_data])
    with open(output_filename, "w") as f_out:
        y.dump([final_spectra, {i: p for i, p in enumerate(pt_hard_spectra, start=1)}], f_out)

    return True


def scale_factor_from_hists(
    n_accepted_events: int, n_entries: int, cross_section: Any, n_trials: Any
) -> skim_analysis_objects.ScaleFactor:
    scale_factor = skim_analysis_objects.ScaleFactor.from_hists(
        n_accepted_events=n_accepted_events,
        cross_section=cross_section,
        n_trials=n_trials,
        n_entries=n_entries,
    )

    return scale_factor


def test() -> None:
    scale_factors_ROOT = {}
    # scale_factors_uproot = {}
    # train_numbers = list(range(6316, 6318))
    train_numbers = [6650, 6659]

    base_path = Path("trains/embedPythia/{train_number}/AnalysisResults.*.repaired.root")
    for train_number in train_numbers:
        logger.info(f"train_number: {train_number}")
        scale_factors_ROOT[train_number] = scale_factor_from_hists(
            *scale_factor_ROOT_wrapper(base_path=base_path, train_number=train_number)
        )
        # scale_factors_uproot[train_number] = scale_factor_from_hists(
        #    *scale_factor_uproot_wrapper(base_path=base_path, train_number=train_number, run_despite_issues=True)
        # )
        # res_ROOT = scale_factor_ROOT(base_path=base_path, train_number=train_number)
        # res_uproot = scale_factor_uproot(base_path=base_path, train_number=train_number)

    y = yaml.yaml(classes_to_register=[skim_analysis_objects.ScaleFactor])
    with open("test.yaml", "w") as f:
        y.dump(scale_factors_ROOT, f)

    print(f"scale_factors_ROOT: {scale_factors_ROOT}")
    # print(f"scale_factors_uproot: {scale_factors_uproot}")
    import IPython

    IPython.start_ipython(user_ns=locals())


if __name__ == "__main__":
    helpers.setup_logging()
    test()
