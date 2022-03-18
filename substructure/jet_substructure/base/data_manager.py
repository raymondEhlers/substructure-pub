""" Manager access to datasets and trees.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import functools
import logging
import typing
from collections import ChainMap
from functools import partial, reduce
from pathlib import Path
from typing import (
    Any,
    Callable,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

import attr
import awkward0 as ak
import h5py
import uproot3
from typing_extensions import Literal

from jet_substructure.base.helpers import (
    UprootArray,
    UprootArrays,
    ensure_and_expand_paths,
    expand_wildcards_in_filenames,
)


logger = logging.getLogger(__name__)


def _ensure_and_expand_hdf5_paths(paths: Sequence[Union[str, Path]]) -> List[Path]:
    return expand_wildcards_in_filenames([Path(p).with_suffix(".h5") for p in paths])


def _ensure_hdf5_path(path: Union[str, Path]) -> Path:
    return Path(path).with_suffix(".h5")


class TreeMixin:
    """Wrapper around an open tree.

    It keeps track of the tree itself, as well as tree metadata such as the available branches
    or the filename where it is stored.
    """

    _tree: Mapping[str, UprootArray[Any]]
    _filename: Path
    _tree_name: str
    _branches: FrozenSet[str]

    @property
    def filename(self) -> Path:
        """ Filename of the current tree. """
        return self._filename

    @property
    def tree_name(self) -> str:
        """ Name of the current tree. """
        return self._tree_name

    def _retrieve_branch_names(self) -> FrozenSet[str]:
        return frozenset(self._tree.keys())

    @property
    def branches(self) -> FrozenSet[str]:
        """Branches stored in the tree.

        Accessing them in this manner allows them to be set externally, but then we calculate
        them if they're not already set.
        """
        if len(self._branches) != 0:
            return self._branches
        else:
            self._branches = self._retrieve_branch_names()
        return self._branches

    def __len__(self) -> int:
        return len(self._tree)

    def __iter__(self) -> Iterator[str]:
        return iter(self.branches)

    def _retrieve_branches(self, key: Iterable[str]) -> UprootArrays:
        return {k: self._tree[k] for k in key}

    def _retrieve_branch(self, key: str) -> UprootArray[Any]:
        return self._tree[key]

    @typing.overload
    def __getitem__(self, key: str) -> UprootArray[Any]:  # type: ignore
        ...

    @typing.overload
    def __getitem__(self, key: Sequence[str]) -> UprootArrays:
        ...

    def __getitem__(self, key: Union[str, Sequence[str]]) -> Union[UprootArray[Any], UprootArrays]:
        if not isinstance(key, str):
            # Take an intersection with the requested keys, since we can only take onces
            # which actually exist in the tree.
            branches_to_return = self.branches.intersection(list(key))
            logger.debug(
                f"Branches: "
                f"\n\tRequested: {key}"
                f"\n\tReturning: {branches_to_return}"
                f"\n\tdifference: {set(key) - branches_to_return}"
            )
            # Retrieve all of the branches.
            return self._retrieve_branches(branches_to_return)

        return self._retrieve_branch(key)


@attr.s
class UprootTreeWrapper(TreeMixin, Mapping[str, UprootArray[Any]]):
    """

    Note:
        If iterating over many files, it's best to create the caches externally. Otherwise, we keep creating
        new caches.
    """

    _filename: Path = attr.ib(converter=Path)
    _file: uproot3.rootio.ROOTDirectory = attr.ib()
    _tree_name: str = attr.ib()
    # _tree: Mapping[str, UprootArray[Any]] = attr.ib()
    _tree: uproot3.tree.TTreeMethods = attr.ib()
    _branches: FrozenSet[str] = attr.ib(converter=frozenset, default=frozenset())
    _cache: MutableMapping[str, UprootArray[Any]] = attr.ib(factory=partial(uproot3.ThreadSafeArrayCache, "1 GB"))
    _key_cache: MutableMapping[str, UprootArray[Any]] = attr.ib(factory=partial(uproot3.ThreadSafeArrayCache, "100 MB"))

    def _retrieve_branch_names(self) -> FrozenSet[str]:
        return frozenset([name.decode("utf-8") for name in self._tree.allkeys()])

    def _retrieve_branches(self, key: Iterable[str]) -> UprootArrays:
        return cast(
            UprootArrays, self._tree.arrays(key, cache=self._cache, keycache=self._key_cache, namedecode="utf-8")
        )

    def _retrieve_branch(self, key: str) -> UprootArray[Any]:
        return cast(UprootArray[Any], self._tree.array(key, cache=self._cache, keycache=self._key_cache))

    @classmethod
    def from_filename(
        cls: Type["UprootTreeWrapper"], filename: Path, tree_name: str, **kwargs: MutableMapping[str, UprootArray[Any]]
    ) -> "UprootTreeWrapper":
        """Open a tree stored in a given file and wrap around the tree for a uniform API.

        Note:
            If iterating over many files, it's best to create the caches externally. Otherwise, we keep creating
            new caches.

        Args:
            filename: Filename which contains the tree.
            tree_name: Name of the tree inside of the file.
            cache: Uproot cache for tree array data.
            key_cache: Uproot cache for tree keys.

        Returns:
            Wrapped around the tree stored in HDF5, ready to provide data.
        """
        # Will be closed when this tree wrapper goes out of scope.
        f = uproot3.open(filename)

        return cls(
            file=f,
            tree=f[tree_name],
            filename=filename,
            tree_name=tree_name,
            branches=frozenset(),
            **kwargs,
        )


@attr.s
class HDF5TreeWrapper(TreeMixin, MutableMapping[str, UprootArray[Any]]):
    _filename: Path = attr.ib(converter=_ensure_hdf5_path)
    _file: h5py.File = attr.ib()
    _tree_name: str = attr.ib()
    # Technically, it would be more correct to be typed as ak.hdf5. However, that would be less informative
    # because it will be treated as Any. So instead we tell a white lie and type it with it's behavior.
    _tree: MutableMapping[str, UprootArray[Any]] = attr.ib()
    _branches: FrozenSet[str] = attr.ib(converter=frozenset, default=frozenset())

    def __setitem__(self, key: str, item: UprootArray[Any]) -> None:
        self._tree.__setitem__(key, item)

    def __delitem__(self, key: str) -> None:
        self._tree.__delitem__(key)

    @classmethod
    def from_filename(
        cls: Type["HDF5TreeWrapper"], filename: Union[Path, str], tree_name: str, file_mode: str = "a"
    ) -> "HDF5TreeWrapper":
        """Open a tree stored in a given file and wrap around the tree for a uniform API.

        Args:
            filename: Filename which contains the tree.
            tree_name: Name of the tree inside of the file.
            file_mode: Mode under which the file should be opened. Default to "a"
                so that we can read and write (and the file will be created if it
                doesn't exist).

        Returns:
            Wrapped around the tree stored in HDF5, ready to provide data.
        """
        # Validation
        filename = _ensure_hdf5_path(filename)
        file_mode = "a" if not filename.exists() else file_mode

        # Will be closed when this tree wrapper goes out of scope.
        f = h5py.File(filename, file_mode)

        whitelist = ak.persist.whitelist + [
            ["jet_substructure.base.substructure_methods", "JetConstituentArray", "from_jagged"],
            ["jet_substructure.base.substructure_methods", "SubjetArray", "_from_jagged_impl"],
            ["jet_substructure.base.substructure_methods", "JetSplittingArray", "from_jagged"],
            ["jet_substructure.base.substructure_methods", "SubstructureJetArray"],
        ]
        return cls(
            filename=filename,
            file=f,
            tree_name=tree_name,
            tree=ak.hdf5(f.require_group(tree_name), whitelist=whitelist),
        )

    def flush(self) -> None:
        """Flush the HDF5 to ensure that everything is written out.

        Otherwise, the file may not close safely. See: https://github.com/h5py/h5py/issues/714.
        """
        self._file.flush()


@attr.s
class UprootTreeIterator:
    _filenames: Sequence[Path] = attr.ib(converter=ensure_and_expand_paths)
    _tree_name: str = attr.ib()
    _cache: MutableMapping[str, UprootArray[Any]] = attr.ib(factory=partial(uproot3.ThreadSafeArrayCache, "1 GB"))
    _key_cache: MutableMapping[str, UprootArray[Any]] = attr.ib(factory=partial(uproot3.ThreadSafeArrayCache, "100 MB"))

    def __iter__(self) -> Iterator[UprootTreeWrapper]:
        # NOTE: If the file sizes get too big, can set entrysteps to something like `entrysteps=100000`, which
        #       is large, but less than the size of the file. We want to keep it as large as possible.
        for f, filename, tree in uproot3.iterate(
            path=self._filenames,
            treepath=self._tree_name,
            namedecode="utf-8",
            cache=self._cache,
            keycache=self._key_cache,
            reportpath=True,
            reportfile=True,
        ):
            yield UprootTreeWrapper(
                filename=filename,
                file=f,
                tree_name=self._tree_name,
                tree=tree,
                cache=self._cache,
                key_cache=self._key_cache,
            )
            # Once we're returned from the yield, we're done with the tree in file.
            # Consequently, we can clear the cache because we don't need those values anymore.
            self._cache.clear()
            self._key_cache.clear()


@attr.s
class HDF5TreeIterator:
    _filenames: Sequence[Path] = attr.ib(converter=_ensure_and_expand_hdf5_paths)
    _tree_name: str = attr.ib()
    _file_mode: str = attr.ib(default="r")

    def __iter__(self) -> Iterator[HDF5TreeWrapper]:
        for filename in self._filenames:
            # Need to ensure that the file is created if it doesn't already exist. Best way to do so is via "a"
            file_mode = "a" if not filename.exists() else self._file_mode
            with h5py.File(filename, file_mode) as f:
                storage = ak.hdf5(f.require_group(self._tree_name))
                yield HDF5TreeWrapper(filename=filename, file=f, tree_name=self._tree_name, tree=storage)


@attr.s
class Tree(MutableMapping[str, UprootArray[Any]]):
    _uproot_tree: UprootTreeWrapper = attr.ib()
    _hdf5_tree: HDF5TreeWrapper = attr.ib()

    @property
    def filename(self) -> Path:
        """The filename of the (uproot) tree.

        The HDF5 filename is the same, just with the extension replaced with `.h5`.
        """
        return self._uproot_tree.filename

    @property
    def tree_name(self) -> str:
        """The filename of the (uproot) tree.

        The HDF5 tree name is the same.
        """
        return self._uproot_tree.tree_name

    @property
    def _trees(self) -> List[TreeMixin]:
        return [self._uproot_tree, self._hdf5_tree]

    @property
    def branches(self) -> FrozenSet[str]:
        try:
            return self._branches
        except AttributeError:
            self._branches: FrozenSet[str] = reduce(frozenset.union, [tree.branches for tree in self._trees])
        return self._branches

    def __len__(self) -> int:
        return len(self._uproot_tree)

    def __iter__(self) -> Iterator[str]:
        return iter(self.branches)

    @typing.overload
    def __getitem__(self, key: str) -> UprootArray[Any]:  # type:ignore
        ...

    @typing.overload
    def __getitem__(self, key: Sequence[str]) -> UprootArrays:
        ...

    def __getitem__(self, key: Union[str, Sequence[str]]) -> Union[UprootArray[Any], UprootArrays]:
        if isinstance(key, str):
            for tree in self._trees:
                if key in tree.branches:
                    return tree[key]
            raise KeyError(f"Could not retrieve branch {key}")
        else:
            missing_branches = set(key) - self.branches
            if missing_branches:
                raise ValueError(
                    f"Not all requested branches are available. Missing: {missing_branches}. Requested branches: {key}"
                )

            # We rely on each tree ignoring branches that aren't relevant to it.
            return dict(ChainMap(*[tree[key] for tree in self._trees]))

    def __setitem__(self, key: str, item: UprootArray[Any]) -> None:
        # Can only store data in the HDF5 file.
        self._hdf5_tree[key] = item

    def __delitem__(self, key: str) -> None:
        del self._hdf5_tree[key]


@attr.s
class IterateTrees:
    _filenames: Sequence[Path] = attr.ib(converter=ensure_and_expand_paths)
    tree_name: str = attr.ib()
    branches: FrozenSet[str] = attr.ib(converter=frozenset)
    _current_tree: Optional[Tree] = attr.ib(default=None)

    def __len__(self) -> int:
        return len(self._filenames)

    def __contains__(self, key: str) -> bool:
        return Path(key) in self._filenames

    def __iter__(self) -> Iterator[Tree]:
        """Iterate over lazy trees."""
        # Allocate cache here so we only create it once.
        uproot_cache = uproot3.ThreadSafeArrayCache("1 GB")
        uproot_key_cache = uproot3.ThreadSafeArrayCache("100 MB")

        for filename in self._filenames:
            self._current_tree = Tree(
                UprootTreeWrapper.from_filename(
                    filename=filename, tree_name=self.tree_name, cache=uproot_cache, key_cache=uproot_key_cache
                ),
                HDF5TreeWrapper.from_filename(filename=filename, tree_name=self.tree_name, file_mode="a"),
            )

            yield self._current_tree

            # Flush the HDF5 to ensure that the data is written.
            self._current_tree._hdf5_tree.flush()

            # Once we're returned from the yield, we're done with the tree in file.
            # Consequently, we can clear the cache because we don't need those values anymore.
            uproot_cache.clear()
            uproot_key_cache.clear()

    def _fully_lazy_iteration(self) -> Iterator[Callable[[], Tree]]:
        """Fully lazy iterator over trees.

        Requires the calling function to call the return value to actually generate the tree. This way,
        we can pass the wrapper function via multiprocessing, and then instantiate it there.

        Note:
            Since we generate the objects it is the callers responsibility to ensure that the hdf5 is flushed!
            Otherwise, the hdf5 may not be written correctly, leading to possible corruption. This isn't ideal,
            but the usual approach (using a generator) doesn't work with multiprocessing.
        """

        def _wrap(filename: Path) -> Tree:
            """Wrap creation of the Tree.

            Note:
                We don't yet add this as a Tree classmethod because we only want to support a subset
                of arguments for now. If we need more generality, we can refactor later.

            Note:
                We don't make this a full closure (including specifying the filename) because the closure
                would always have the same name. So we would keep redefining a function with the same name,
                but a different filename in the closure. Multiprocessing selects the function by name, so
                this will cause multiprocessing to use the most recent function everywhere, thus calling the
                same file over and over again. We can avoid this by defining this wrapping function once,
                and then defining it for each filename with partial.

            Args:
                filename: Filename to be used with this tree.
            """
            return Tree(
                UprootTreeWrapper.from_filename(filename=filename, tree_name=self.tree_name),
                HDF5TreeWrapper.from_filename(filename=filename, tree_name=self.tree_name, file_mode="a"),
            )

        for filename in self._filenames:
            # We can't really maintain the current tree. It's just not meaningful here, so we leave it as None.
            self._current_tree = None
            yield functools.partial(_wrap, filename=filename)

    @typing.overload
    def lazy_iteration(self, fully_lazy: Literal[False]) -> Iterator[Tree]:
        ...

    @typing.overload
    def lazy_iteration(self, fully_lazy: Literal[True]) -> Iterator[Callable[[], Tree]]:
        ...

    @typing.overload
    def lazy_iteration(self, fully_lazy: bool = False) -> Union[Iterator[Tree], Iterator[Callable[[], Tree]]]:
        """ In case the user provides a bool. """
        ...

    def lazy_iteration(self, fully_lazy: bool = False) -> Union[Iterator[Tree], Iterator[Callable[[], Tree]]]:
        if fully_lazy:
            yield from self._fully_lazy_iteration()
        else:
            yield from self

    def active_iteration(self) -> Iterator[Tree]:
        """Iterate over actively loaded trees.

        It is recommended to use lazy iteration via `__iter__`!
        """
        for uproot_tree, hdf5_tree in zip(
            UprootTreeIterator(filenames=self._filenames, tree_name=self.tree_name),
            HDF5TreeIterator(filenames=self._filenames, tree_name=self.tree_name),
        ):
            self._current_tree = Tree(uproot_tree, hdf5_tree)

            yield self._current_tree
