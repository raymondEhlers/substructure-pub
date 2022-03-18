""" Unfolding results base functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from functools import reduce
from typing import Any, Dict, Optional, Type

import attr
import numpy as np
from pachyderm import binned_data

from jet_substructure.base import helpers


logger = logging.getLogger(__name__)


@attr.s(eq=False)
class AsymmetricErrors:
    low: np.ndarray = attr.ib()
    high: np.ndarray = attr.ib()

    def __eq__(self, other: Any) -> bool:
        if other.__class__ is not self.__class__:
            return NotImplemented
        return np.allclose(self.low, other.low) and np.allclose(self.high, other.high)

    @classmethod
    def calculate_errors(
        cls: Type[AsymmetricErrors], errors_one: np.ndarray, errors_two: Optional[np.ndarray] = None
    ) -> AsymmetricErrors:
        """Calculate asymmetric errors from given errors.

        Note:
            This returns positive, absolute errors in each direction.

        Args:
            errors_one: First error array. Doesn't matter if it's the upper or lower value.
            errors_two: Second error array. Doesn't matter if it's the upper or lower value.
                Default: None, in which case the first passed errors are duplicated. This duplication
                is used for single valued asymmetric errors.
        Returns:
            Asymmetric errors calculated based on the given errors. See the function for the
                precise algorithm.
        """
        # Validation
        # This allows us to calculate single value asymmetric errors.
        # This is equivalent to just pass the same error values twice, but this
        # is a cleaner interface.
        one_sided = False
        if errors_two is None:
            errors_two = np.array(errors_one, copy=True)
            one_sided = True

        # Determine when the errors are positive.
        # True if positive, false if negative
        positive_one = np.sign(errors_one) == 1
        positive_two = np.sign(errors_two) == 1
        # Calculate once for convenience.
        errors_one_abs = np.abs(errors_one)
        errors_two_abs = np.abs(errors_two)

        # Output arrays
        low = np.zeros_like(errors_one)
        high = np.zeros_like(errors_one)

        # First, handle when they have the same sign.
        # For this case, we take the maximum of either error, and assign that asymmetrically to the side of the sign
        # positive -> high
        # negative -> low
        same_sign = positive_one == positive_two
        # Both positive.
        mask = same_sign & positive_one
        # Don't need to set low because it's already zero for these points.
        high[mask] = np.maximum(errors_one_abs, errors_two_abs)[mask]
        # Both negative.
        mask = same_sign & ~positive_one
        # Don't need to set high because it's already zero for these points.
        low[mask] = np.maximum(errors_one_abs, errors_two_abs)[mask]

        # Next, handle opposite signs.
        # For this case, we assign the errors based on the sign.
        # positive -> high
        # negative -> low
        opposite_sign = positive_one != positive_two
        # one positive, two negative
        mask = opposite_sign & positive_one
        low[mask] = errors_two_abs[mask]
        high[mask] = errors_one_abs[mask]
        # one negative, two positive
        mask = opposite_sign & positive_two
        low[mask] = errors_one_abs[mask]
        high[mask] = errors_two_abs[mask]

        # Cross checks. We almost certainly will never have 0 in both bins.
        low_is_zero = low == 0
        high_is_zero = high == 0
        if np.any(low_is_zero & high_is_zero):
            logger.warning("Errors are identically zero for this calculation! Check this carefully!")
        # If it's one sided, then we always should have only one non-zero error.
        if one_sided:
            # not required because assert needs to be False for the assertion to fail.
            assert not np.any(
                ~low_is_zero & ~high_is_zero
            ), f"One sided errors should only have one non-zero value. low: {low}, high: {high}, test: {~low_is_zero & ~high_is_zero}"

        return cls(low=low, high=high)


@attr.s
class ErrorInput:
    value: np.ndarray = attr.ib()
    error: np.ndarray = attr.ib()


def relative_error(*inputs: ErrorInput) -> np.ndarray:
    if len(inputs) == 0:
        raise ValueError("Must pass at least one ErrorInput")
    if len(inputs) > 1:
        relative_error_squared = reduce(lambda x, y: ((x.error / x.value) ** 2) + ((y.error / y.value) ** 2), inputs)  # type: ignore
    else:
        relative_error_squared = (inputs[0].error / inputs[0].value) ** 2
    return np.sqrt(relative_error_squared)  # type: ignore


def select_hist_range(hist: binned_data.BinnedData, x_range: helpers.RangeSelector) -> binned_data.BinnedData:
    # Sanity check
    if len(hist.axes) > 1:
        raise ValueError("Can only handle 1D histogram")

    bin_center_mask = (hist.axes[0].bin_centers >= x_range.min) & (hist.axes[0].bin_centers <= x_range.max)
    first_bin_edge = np.where(bin_center_mask)[0][0]
    last_bin_edge = -1 * np.where(bin_center_mask[::-1])[0][0]
    # If everything is in range, then we'll get 0 for the last bin edge. However, this would translate to no
    # range included. In that case, we want to include everything on the upper edge, and so we need to set
    # it to None.
    if last_bin_edge == 0:
        last_bin_edge = None

    # Handle metadata
    metadata: Dict[str, Any] = {}
    for k, v in hist.metadata.items():
        if k == "y_systematic":
            y_systematic = {}
            for k_sys, v_sys in v.items():
                if isinstance(v_sys, AsymmetricErrors):
                    y_systematic[k_sys] = AsymmetricErrors(
                        low=v_sys.low[bin_center_mask],
                        high=v_sys.high[bin_center_mask],
                    )
            metadata["y_systematic"] = y_systematic
        else:
            if isinstance(v, AsymmetricErrors):
                metadata[k] = AsymmetricErrors(
                    low=v.low[bin_center_mask],
                    high=v.high[bin_center_mask],
                )

    return binned_data.BinnedData(
        axes=[hist.axes[0].bin_edges[first_bin_edge:last_bin_edge]],
        values=hist.values[bin_center_mask],
        variances=hist.variances[bin_center_mask],
        metadata=metadata,
    )
