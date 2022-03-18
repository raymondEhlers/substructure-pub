""" Basic shared functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import argparse
import itertools
import logging
import typing
from pathlib import Path
from typing import (
    Any,
    Collection,
    Dict,
    Iterable,
    List,
    Mapping,
    NoReturn,
    Optional,
    Sequence,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import attr
import numpy as np
import numpy.typing as npt


logger = logging.getLogger(__name__)

# Typing helpers
T = TypeVar("T")


def assert_never(value: NoReturn) -> NoReturn:
    """Typing helper to check that all values have been exhausted.

    Can be used for checking Enum and Literals.

    From: https://hakibenita.com/python-mypy-exhaustive-checking

    Args:
        value: Value to check for exhaustiveness.
    """
    assert False, f"Unhandled value: {value} ({type(value).__name__})"


class UprootArray(Collection[T]):
    """Effectively a protocol for the UprootArray type.

    The main advantage is that it allows us to keep track of the types. I don't believe
    that they're closely checked, but if nothing else, they're useful as sanity checks
    for me.

    These definitely _aren't_ comprehensive, but they're a good start.
    """

    @typing.overload
    def __getitem__(self, key: UprootArray[bool]) -> UprootArray[T]:
        ...

    @typing.overload
    def __getitem__(self, key: UprootArray[int]) -> UprootArray[T]:
        ...

    @typing.overload
    def __getitem__(self, key: Tuple[slice, slice]) -> UprootArray[T]:
        ...

    @typing.overload
    def __getitem__(self, key: npt.NDArray[np.bool_]) -> UprootArray[T]:
        ...

    @typing.overload
    def __getitem__(self, key: bool) -> T:
        ...

    @typing.overload
    def __getitem__(self, key: int) -> T:
        ...

    def __getitem__(self, key):  # type: ignore
        raise NotImplementedError("Just typing information.")

    def __add__(self, other: Union[UprootArray[T], int, float]) -> UprootArray[T]:
        raise NotImplementedError("Just typing information.")

    def __radd__(self, other: Union[UprootArray[T], int, float]) -> UprootArray[T]:
        raise NotImplementedError("Just typing information.")

    def __sub__(self, other: Union[UprootArray[T], int, float]) -> UprootArray[T]:
        raise NotImplementedError("Just typing information.")

    def __rsub__(self, other: Union[UprootArray[T], int, float]) -> UprootArray[T]:
        raise NotImplementedError("Just typing information.")

    def __mul__(self, other: Union[UprootArray[T], int, float]) -> UprootArray[T]:
        raise NotImplementedError("Just typing information.")

    def __rmul__(self, other: Union[UprootArray[T], int, float]) -> UprootArray[T]:
        raise NotImplementedError("Just typing information.")

    def __truediv__(self, other: Union[float, UprootArray[T]]) -> UprootArray[T]:
        raise NotImplementedError("Just typing information.")

    def __pow__(self, p: float) -> UprootArray[T]:
        raise NotImplementedError("Just typing information.")

    def argmax(self) -> UprootArray[int]:
        raise NotImplementedError("Just typing information.")

    def offsets(self) -> npt.NDArray[np.int64]:
        raise NotImplementedError("Just typing information.")

    def flatten(self, axis: Optional[int] = ...) -> npt.NDArray[Any]:
        raise NotImplementedError("Just typing information.")

    @property
    def localindex(self) -> UprootArray[int]:
        raise NotImplementedError("Just typing information.")

    def count_nonzero(self) -> UprootArray[int]:
        raise NotImplementedError("Just typing information.")

    def counts(self) -> UprootArray[int]:
        raise NotImplementedError("Just typing information.")

    def __lt__(self, other: Union[UprootArray[T], float]) -> UprootArray[bool]:
        raise NotImplementedError("Just typing information.")

    def __le__(self, other: Union[UprootArray[T], float]) -> UprootArray[bool]:
        raise NotImplementedError("Just typing information.")

    def __gt__(self, other: Union[UprootArray[T], float]) -> UprootArray[bool]:
        raise NotImplementedError("Just typing information.")

    def __ge__(self, other: Union[UprootArray[T], float]) -> UprootArray[bool]:
        raise NotImplementedError("Just typing information.")

    def __and__(self, other: UprootArray[bool]) -> UprootArray[bool]:
        ...

    def __or__(self, other: UprootArray[bool]) -> UprootArray[bool]:
        raise NotImplementedError("Just typing information.")

    def __invert__(self) -> UprootArray[bool]:
        raise NotImplementedError("Just typing information.")

    def pad(self, value: int) -> UprootArray[T]:
        raise NotImplementedError("Just typing information.")

    def fillna(self, value: Any) -> UprootArray[T]:
        """Fill na values with the given values.

        Note:
            This is a bit of a white lie. The types in the array can be a Union[T, Type[value]],
            but including such types makes it a good deal more complicated. So we just don't
            mention the other values. In practice, they will usually be the same type.
        """
        raise NotImplementedError("Just typing information.")

    def ones_like(self) -> UprootArray[int]:
        raise NotImplementedError("Just typing information.")


# Additional typing helpers
ArrayOrScalar = Union[UprootArray[T], T]
UprootArrays = Mapping[str, UprootArray[Any]]


def setup_logging(level: int = logging.DEBUG) -> None:
    # Delayed import since we may not want this as a hard dependency in such a base module.
    import coloredlogs

    # Basic setup
    coloredlogs.install(level=level, fmt="%(asctime)s %(name)s:%(lineno)d %(levelname)s %(message)s")
    # Quiet down the matplotlib logging
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    # For sanity when using IPython
    logging.getLogger("parso").setLevel(logging.INFO)
    logging.getLogger("blib2to3").setLevel(logging.INFO)
    logging.getLogger("asyncio").setLevel(logging.INFO)
    # Quiet down BinnedData copy warnings
    logging.getLogger("pachyderm.binned_data").setLevel(logging.INFO)
    # Quiet down numba
    logging.getLogger("numba").setLevel(logging.INFO)


def pretty_print_tree(d: Mapping[int, Any], indent: int = 0) -> None:
    """Convenience function for pretty printing the splitting tree.

    From: https://stackoverflow.com/a/3229493.

    Args:
        d: Dictionary containing the splittings.
        indent: How far to indent (effectively how far we are into the recursion).

    Returns:
        None.
    """
    for key, value in d.items():
        print("\t" * indent + str(key))
        if isinstance(value, Mapping):
            pretty_print_tree(value, indent + 1)
        else:
            print("\t" * (indent + 1) + str(value))


def convert_flat_to_tree(parent_label: int, relationships: Sequence[Tuple[int, int]]) -> Dict[int, Any]:
    """Convert the flat array to the tree.

    Slightly modified from: https://stackoverflow.com/a/43728268

    Args:
        parent_label: Label of the root parent (usually -1).
        relationships: Relationships from child to parent. Of the form (child index, parent index).
    Returns:
        Tree representing these relationships.
    """
    return {
        p: convert_flat_to_tree(p, relationships)
        for p in [index for index, parent in relationships if parent == parent_label]
    }


def dict_product(input_dict: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    """Like `itertools.product`, but with a dictionary containing lists.

    By way of example:

    >>> list(product_dict({"a": [1, 2], "b": [3], "c": [4, 5]}))
    [{'a': 1, 'b': 3, 'c': 4}, {'a': 1, 'b': 3, 'c': 5}, {'a': 2, 'b': 3, 'c': 4}, {'a': 2, 'b': 3, 'c': 5}]

    It will give us all possible combinations of the list values with their associated keys.

    From: https://stackoverflow.com/a/40623158/12907985

    Args:
        kwargs: Dictionary for the product.
    Returns:
        Product of the dict keys and values.
    """
    return (dict(zip(input_dict.keys(), values)) for values in itertools.product(*input_dict.values()))


@attr.s(frozen=True)
class RangeSelector:
    min: float = attr.ib()
    max: float = attr.ib()
    _variable_name = "jet_pt"
    _display_name = r"p_{\text{T,ch jet}}"

    def mask_attribute(self, df: UprootArrays, attribute_name: str) -> UprootArray[bool]:
        """Create a mask to given attribute to the provided range.

        Args:
            df: Data to be used to define the mask. May be a pandas DataFrame or output from loading arrays
                via uproot.
            attribute_name: Name of the attribute (column) to be used in the mask.
        Returns:
            Mask of the df for the attribute values within the stored range.
        """
        # Range defined by what is shown in the paper.
        return self.mask_array(df[attribute_name])

    def mask_array(self, array: UprootArray[T]) -> UprootArray[bool]:
        return (array >= self.min) & (array < self.max)

    @classmethod
    def full_range_over_selections(cls: Type["RangeSelector"], selections: Sequence[RangeSelector]) -> "RangeSelector":
        """Extract the min and max range value over all of the selections.

        The DataFrames can be reduced to only contain these values.

        One could be more efficient in the case that there are gaps in the selections, but this
        is sufficient for our purposes.

        Args:
            selections: Selections to be applied.

        Returns:
            Minimum and maximum values.
        """
        return cls(
            min=min(selections, key=lambda v: v.min).min,
            max=max(selections, key=lambda v: v.max).max,
        )

    def __str__(self) -> str:
        return f"{self._variable_name}_{self.min:g}_{self.max:g}"

    def histogram_str(self, label: str = "") -> str:
        if label:
            label = f"_{label}"
        return fr"{self._variable_name}{label}_{self.min}_{self.max}"

    def zero_padded_str(self, n_zeros: int = 0) -> str:
        return f"{self._variable_name}_{int(self.min * 10 ** n_zeros)}_{int(self.max * 10 ** n_zeros)}"

    def display_str(self, label: str = "") -> str:
        return fr"{self.min} < {self._display_name}^{{\text{{{label}}}}} < {self.max}"

    def __iter__(self) -> Iterable[float]:
        yield self.min
        yield self.max


@attr.s(frozen=True)
class JetPtRange(RangeSelector):
    ...


@attr.s(frozen=True)
class KtRange(RangeSelector):
    _variable_name = "kt"
    _display_name = r"k_{\text{T}}"


@attr.s(frozen=True)
class RgRange(RangeSelector):
    _variable_name = "delta_R"
    _display_name = r"{\Delta R}"


@attr.s(frozen=True)
class ZgRange(RangeSelector):
    _variable_name = "zg"
    _dispaly_name = r"z_{\text{g}}"


def expand_wildcards_in_filenames(paths: Sequence[Path]) -> List[Path]:
    return_paths: List[Path] = []
    for path in paths:
        p = str(path)
        if "*" in p:
            # Glob all associated filenames.
            # NOTE: This assumes that the paths are relative to the execution directory. But that's
            #       almost always the case.
            return_paths.extend(list(Path(".").glob(p)))
        else:
            return_paths.append(path)

    # Sort in the expected order (just according to alphabetical, which should handle numbers
    # fine as long as they have leading 0s (ie. 03 instead of 3)).
    return_paths = sorted(return_paths, key=lambda p: str(p))
    return return_paths


def ensure_and_expand_paths(paths: Sequence[Union[str, Path]]) -> List[Path]:
    return expand_wildcards_in_filenames([Path(p) for p in paths])


def _AliEmcalList_to_TList(
    existing_list: Any,
) -> Any:
    """Convert an `AliEmcalList` to `TList` so we don't have to deal with `AliEmcalList` later...

    Ideally, we could just cast this, but I don't see how to do that with PyROOT...
    For example, `reinterpret_cast` maintains the type of the original object.

    Note:
        This isn't recursive, so it's possible to miss a conversion. However, it isn't
        super common to have nested `AliEmcalList` objects, so that's fine for now.

    Args:
        existing_list: Existing AliEmcalList to convert.
    Returns:
        TList containing the same contents.
    """
    # Delayed import because we want to depend on ROOT as little as possible.
    import ROOT

    temp_list = ROOT.TList()
    temp_list.SetName(existing_list.GetName())
    for el in existing_list:
        temp_list.Add(el)
    return temp_list


def split_tree(  # noqa: C901
    filenames: Sequence[Union[str, Path]],
    tree_name: str = "AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
    number_of_chunks: int = -1,
    chunk_size: float = 200e6,
    n_cores: int = 1,
) -> Dict[Path, List[Path]]:
    """Split tree into a given number of chunks.

    It will also skip storing bad entries in the new files.

    Note:
        To only repair the file, use only one chunk.

    Note:
        Even if we are chunking the file, this method will still try to avoid storing bad entries
        in the new files.

    Args:
        filenames: Name(s) of the file to split.
        tree_name: Name of the tree to split. Default: "AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl"
        number_of_chunks: Number of chunks to split the file into. Pass -1 to auto calculate the number of
            chunks (ie keeping each file before 1.25 GB). Default: -1.
        chunk_size: Size of the chunks in bytes. Only considered if auto calculating the number of
            chunks. Default: 200e6 (200 MB) (driven by memory requirements on the cluster).
        n_cores: Number of cores to utilize. Default: 1.

    Returns:
        Filenames of the chunked files.
    """
    # Validation
    validated_filenames = expand_wildcards_in_filenames([Path(f) for f in filenames])
    # Skip files which were already repaired.
    validated_filenames = [f for f in validated_filenames if "repaired" not in str(f) and "chunk" not in str(f)]
    # Automatically calculate the number of chunks, with the intention of keeping all
    # files below (approximately) chunk_size.
    auto_calculate_number_of_chunks = False
    if number_of_chunks < 0:
        logger.info(f"Automatically calculating number of chunks for chunk size < {int(chunk_size / 1e6)} MB")
        auto_calculate_number_of_chunks = True
    # Copy lists if we are just repairing. We can't do it if we're splitting into chunks
    # because it's unclear how to split the hists...
    attempt_to_copy_lists = False
    if number_of_chunks == 1:
        attempt_to_copy_lists = True

    # Setup
    # Delayed import since we may not want this as a hard dependency in such a base module.
    import enlighten

    # Delayed import because we want to depend on ROOT as little as possible.
    import ROOT

    # Just in case we enable multithreading later.
    ROOT.ROOT.EnableImplicitMT(n_cores)
    # Finish setup
    output_filenames: Dict[Path, List[Path]] = {}
    progress_manager = enlighten.get_manager()

    with progress_manager.counter(total=len(validated_filenames), desc="Processing", unit="file") as file_counter:
        for filename in file_counter(validated_filenames):
            # Determine number of chunks if requested.
            if auto_calculate_number_of_chunks:
                size = Path(filename).stat().st_size
                number_of_chunks = int(np.ceil(size / chunk_size))

            # Setup input tree
            input_file = ROOT.TFile(str(filename), "READ")
            logger.debug(f"Keys in input_file: {list(input_file.GetListOfKeys())}")
            # Lists
            embedding_helper_hists = None
            task_hists = None
            if attempt_to_copy_lists:
                embedding_helper_hists = input_file.Get("AliAnalysisTaskEmcalEmbeddingHelper_histos")
                if embedding_helper_hists:
                    embedding_helper_hists = _AliEmcalList_to_TList(embedding_helper_hists)
                # This will only work for my tasks. But this should be fine, and can be improved later if needed.
                task_hists_name = [
                    key.GetName()
                    for key in input_file.GetListOfKeys()
                    if "DynamicalGrooming" in key.GetName() and "Tree" not in key.GetName()
                ]
                if len(task_hists_name) != 1:
                    logger.warning(f"Cannot find unique task name. Names: {task_hists_name}. Skipping!")
                else:
                    task_hists = input_file.Get(task_hists_name[0])
                    if task_hists:
                        task_hists = _AliEmcalList_to_TList(task_hists)
            # Tree
            input_tree = input_file.Get(tree_name)

            number_of_entries = input_tree.GetEntries()
            logger.info(
                f"File: {filename}: Total of {number_of_entries} in the tree. Splitting into {number_of_chunks} chunk(s)."
            )

            output_filenames[filename] = []
            for n in range(number_of_chunks):
                start = int((number_of_entries / number_of_chunks) * n)
                end = int((number_of_entries / number_of_chunks) * (n + 1))

                # If we have only 1 chunk, then we're just trying to repair the file.
                if number_of_chunks == 1:
                    new_filename = filename.with_name(f"{filename.stem}.repaired.root")
                else:
                    new_filename = filename.with_name(f"{filename.stem}.chunk{n+1:02}.root")
                output_filenames[filename].append(new_filename)
                new_file = ROOT.TFile(str(new_filename), "RECREATE")

                # Attempt to copy lists when appropriate.
                if attempt_to_copy_lists:
                    if embedding_helper_hists:
                        embedding_helper_hists.Write(embedding_helper_hists.GetName(), ROOT.TObject.kSingleKey)
                    if task_hists:
                        task_hists.Write(task_hists.GetName(), ROOT.TObject.kSingleKey)

                # Now handle the tree
                new_tree = input_tree.CloneTree(0)

                logger.info(f"Fill tree {new_filename} with entries {start}-{end}")
                with progress_manager.counter(
                    total=end - start, desc="Converting", unit="event", leave=False
                ) as event_counter:
                    for i in event_counter(range(start, end)):
                        ret_val = input_tree.GetEntry(i)
                        if ret_val < 0:
                            # Skip this entry - something is wrong with it! (Probably a compression error).
                            # This shouldn't happen _too_ often, so we may as well log when it does.
                            logger.debug(
                                f"Skipping entry {i}, as it appears to be bad. GetEntry return value < 0: {ret_val}"
                            )
                            continue
                        new_tree.Fill()

                # Save the new tree.
                # It appears that we don't even need to write the tree because it's attached to the new_file...
                new_tree.Write()
                # NOTE: Don't use AutoSave - it could lead to writing with memberwise splittings, which won't
                #       be read by uproot...
                # Cleanup
                new_file.Close()

            # Cleanup
            input_file.Close()

    progress_manager.stop()
    return output_filenames


def split_tree_entry_point() -> None:
    """Entry point for splitting a tree into chunks.

    Args:
        None. It can be configured through command line arguments.

    Returns:
        None.
    """
    setup_logging()
    parser = argparse.ArgumentParser(description="Split tree into chunks.")

    parser.add_argument("-f", "--filenames", required=True, nargs="+", default=[])
    parser.add_argument(
        "-t",
        "--treeName",
        default="AliAnalysisTaskJetDynamicalGrooming_hybridLevelJets_AKTChargedR040_tracks_pT0150_E_schemeConstSub_RawTree_EventSub_Incl",
        type=str,
        help="Name of the tree to split",
    )
    parser.add_argument("-c", "--cores", default=1, type=int, help="Number of cores to use.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-n", "--nChunks", default=-1, type=int, help="Number of chunks. Default: -1, which then uses the chunk size."
    )
    # Default to approximately 200 MB chunks (in bytes).
    group.add_argument(
        "-s", "--chunkSize", default=200e6, type=float, help="Chunk size in bytes. Default: 200e6 (200 MB)"
    )
    args = parser.parse_args()

    output_filenames = split_tree(
        filenames=args.filenames,
        tree_name=args.treeName,
        number_of_chunks=args.nChunks,
        chunk_size=args.chunkSize,
        n_cores=args.cores,
    )

    import pprint

    logger.info(f"File output: {pprint.pformat(output_filenames)}")


def merge_ROOT_files(dir: Path, n_merged_files: int = 5) -> Dict[Path, List[Path]]:
    # Setup.
    merged_dir = dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    # Setup
    # Delayed import since we may not want this as a hard dependency in such a base module.
    import enlighten

    # Delayed import because we want to depend on ROOT as little as possible.
    import ROOT

    # Just in case we enable multithreading later.
    ROOT.ROOT.EnableImplicitMT()

    # To get started, we need to know which files are of interest.
    output_filenames: Dict[Path, List[Path]] = {}
    input_files = list(dir.glob("*.root"))
    progress_manager = enlighten.get_manager()

    with progress_manager.counter(total=n_merged_files, desc="Merging", unit="file") as file_counter:
        for n in file_counter(range(n_merged_files)):
            # Setup
            start = int((len(input_files) / n_merged_files) * n)
            end = int((len(input_files) / n_merged_files) * (n + 1))
            # n+1 so we start indexing at 0.
            output_filename = merged_dir / f"AnalysisResults.merged.{n+1:02}.root"
            output_filenames[output_filename] = []

            merger = ROOT.TFileMerger()
            merger.OutputFile(str(output_filename))
            for filename in input_files[start:end]:
                output_filenames[output_filename].append(filename)
                # False turns off the cp progress bar.
                merger.AddFile(str(filename), False)
            merger.Merge()

    return output_filenames


def merge_ROOT_files_entry_point() -> None:
    """Entry point for merging ROOT files into groups.

    Args:
        None. It can be configured through command line arguments.

    Returns:
        None.
    """
    setup_logging()
    parser = argparse.ArgumentParser(description="Merge files into groups.")

    parser.add_argument(
        "-d", "--directory", required=True, type=Path, help="Directory containing the ROOT files to merge."
    )
    parser.add_argument("-n", "--nGroups", default=-5, type=int, help="Number of groups (ie. output files). Default: 5")
    args = parser.parse_args()

    output_filenames = merge_ROOT_files(dir=args.directory, n_merged_files=args.nGroups)

    import pprint

    logger.info("Done!")
    logger.info(f"File output: {pprint.pformat(output_filenames)}")
