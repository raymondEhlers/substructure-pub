""" Helpers and utilities.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import collections.abc
from typing import Optional, Sequence, Union

import attr

import awkward as ak
import numba as nb
import numpy as np


@attr.frozen
class Range:
    min: Optional[float]
    max: Optional[float]


def _lexsort_for_groupby(array: ak.Array, columns: Sequence[Union[str, int]]) -> ak.Array:
    """Sort for groupby."""
    sort = np.lexsort(tuple(np.asarray(array[:, col]) for col in reversed(columns)))  # type: ignore
    return array[sort]


# @nb.njit
def _run_lengths_for_multiple_columns(array: ak.Array, columns: Sequence[Union[str, int]]) -> nb.typed.List:
    _previous_main_value = -1
    _previous_secondary_value = -1
    # previous_values = nb.typed.List()
    run_lengths = nb.typed.List()
    current_run_length = 0
    for val in array:
        if val[columns[0]] != _previous_main_value or val[columns[1]] != _previous_secondary_value:
            _previous_main_value = val[columns[0]]
            _previous_secondary_value = val[columns[1]]
            run_lengths.append(current_run_length)
            current_run_length = 0
        current_run_length += 1

    return run_lengths


def group_by(array: ak.Array, by: Sequence[Union[str, int]]) -> ak.Array:
    """Group by for awkward arrays.

    Args:
        array: Array to be grouped. Must be convertable to numpy arrays.
        by: Names or indices of columns to group by. The first column is the primary index for sorting,
            second is secondary, etc.
    Returns:
        Array grouped by the columns.
    """
    # Validation
    if not (isinstance(by, collections.abc.Sequence) and not isinstance(by, str)):
        by = [by]

    # First, sort
    # See: https://stackoverflow.com/a/64053838/12907985
    sorted_array = _lexsort_for_groupby(array=array, columns=by)

    # Now, we need to combine the run lengths from the different columns. We need to split
    # every time any of them change.
    run_lengths = [ak.run_lengths(sorted_array[:, k]) for k in by]
    # We can match them up more easily by using the starting index of each run.
    run_starts = [np.cumsum(np.asarray(l)) for l in run_lengths]  # noqa: E741
    # Combine all of the indices together into one array. Note that this isn't unique.
    combined = np.concatenate(run_starts)  # type: ignore
    # TODO: Unique properly...
    # See: https://stackoverflow.com/a/12427633/12907985
    combined = np.unique(combined)  # type: ignore

    run_length = np.zeros(len(combined), dtype=np.int64)
    run_length[0] = combined[0]
    # run_length[1:] = combined[1:] - combined[:-1]
    run_length[1:] = np.diff(combined)  # type: ignore

    # And then construct the array
    return ak.unflatten(
        # Somehow, these seem to be equivalent, even though they shouldn't be...
        # sorted_array, ak.run_lengths(sorted_array[:, by[-1]])
        sorted_array,
        run_length,
    )
