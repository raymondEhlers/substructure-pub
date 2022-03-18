""" Base unfolding classes and functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from pathlib import Path
from typing import Any, Mapping, Type

import attr
import numpy as np
import numpy.typing as npt

from jet_substructure.base import helpers


def _np_array_converter(value: Any, dtype: npt.DTypeLike = np.float64) -> np.ndarray:
    """Convert the given value to a numpy array.

    Normally, we would just use np.array directly as the converter function. However, mypy will complain if
    the converter is untyped. So we add (trivial) typing here.  See: https://github.com/python/mypy/issues/6172.

    Note:
        To change the dtype for the converter, one would need to use `partial`.

    Args:
        value: Value to be converted to a numpy array.
        dtype: Dtype to utilize. Default: np.float64.
    Returns:
        The converted numpy array.
    """
    return np.array(value, dtype=dtype)


@attr.s
class ParameterSettings2D:
    """Parameter settings

    Args:
        true_bins: True bins.
        smeared_bins: Smeared bins.
    """

    true_bins: np.ndarray = attr.ib(converter=_np_array_converter)
    smeared_bins: np.ndarray = attr.ib(converter=_np_array_converter)


class JetPtSettings2D(ParameterSettings2D):
    """Jet pt parameter settings in 2D.

    Nothing specific here, but creating the type for consistency with the substructure variable.
    """

    def encode_for_filename(self) -> str:
        """ Encode the variable settings for a filename. """
        smeared_jet_pt = helpers.JetPtRange(min=self.smeared_bins[0], max=self.smeared_bins[-1])
        # description += f"_smeared_{smeared_jet_pt.zero_padded_str(self.filename_padding_factor)}"
        description = f"_smeared_{smeared_jet_pt}"
        return description


@attr.s
class SubstructureVariableSettings2D(ParameterSettings2D):
    """Settings specific to the substructure variable.

    Supports z, Rg, and kt.

    Args:
        name: Name of the substructure variable.
        variable_name: Name of the substructure variable in the tree.
        smeared_range: Smeared binning min and max. Values vary due to the location of the untagged bin.
        untagged_bin: Untagged bin min and max.
    """

    name: str = attr.ib()
    variable_name: str = attr.ib()
    smeared_range: helpers.RangeSelector = attr.ib()
    untagged_bin: helpers.RangeSelector = attr.ib()

    @property
    def untagged_value(self) -> float:
        return (self.untagged_bin.max - self.untagged_bin.min) / 2 + self.untagged_bin.min

    @property
    def disable_untagged_bin(self) -> bool:
        """If the untagged bin min and max are the same, we want to disable it."""
        return self.untagged_bin.min == self.untagged_bin.max

    def encocde_for_filename(self) -> str:
        """ Encode the variable settings for a filename. """
        # First, the substructure edges.
        description = f"_smeared_{self.smeared_range}"
        # Then the untagged
        description += f"_untagged_{self.untagged_bin}"
        return description

    @classmethod
    def from_binning(
        cls: Type["SubstructureVariableSettings2D"],
        true_bins: np.ndarray,
        smeared_bins: np.ndarray,
        name: str,
        variable_name: str,
        untagged_bin_below_range: bool = True,
    ) -> "SubstructureVariableSettings2D":
        # Determine the appropriate range class.
        # Either "Kt", "Rg", or "Zg"
        range_class_name = variable_name
        if variable_name != "kt" and "g" not in variable_name:
            range_class_name += "g"
        range_class_name = range_class_name.capitalize()
        range_class_name += "Range"
        range_class: Type[helpers.RangeSelector] = getattr(helpers, range_class_name)

        # Determine the binning
        if untagged_bin_below_range:
            smeared_range = range_class(min=smeared_bins[1], max=smeared_bins[-1])
            untagged_bin = range_class(min=smeared_bins[0], max=smeared_bins[1])
        else:
            smeared_range = range_class(min=smeared_bins[0], max=smeared_bins[-2])
            untagged_bin = range_class(min=smeared_bins[-2], max=smeared_bins[-1])

        # Account for disabled untagged bin.
        # We indicate it by making the untagged bin edges identical, but then
        # we need to drop that from the smeared_bins so we have valid binning.
        smeared_bins_selection = slice(None, None)
        if untagged_bin.min == untagged_bin.max:
            if untagged_bin_below_range:
                smeared_bins_selection = slice(1, None)
            else:
                smeared_bins_selection = slice(None, -1)

        return cls(
            true_bins=true_bins,
            smeared_bins=smeared_bins[smeared_bins_selection],
            name=name,
            variable_name=variable_name,
            smeared_range=smeared_range,
            untagged_bin=untagged_bin,
        )


@attr.s
class Settings2D:
    grooming_method: str = attr.ib()
    jet_pt: JetPtSettings2D = attr.ib()
    substructure_variable: SubstructureVariableSettings2D = attr.ib()
    suffix: str = attr.ib()
    output_dir: Path = attr.ib()
    label: str = attr.ib(default="")
    use_pure_matches: bool = attr.ib(default=False)
    filename_padding_factor: int = attr.ib(default=0)

    @property
    def output_tag(self) -> str:
        # Start with the basic information
        base_filename = f"unfolding_{self.substructure_variable.name}_grooming_method_{self.grooming_method}"
        # Then add the binning information.
        # Substructure
        base_filename += self.substructure_variable.encocde_for_filename()
        # Jet pt
        base_filename += self.jet_pt.encode_for_filename()
        # And then the required suffix
        base_filename += f"_{self.suffix}"
        # Additional options
        # Optional tag
        if self.label:
            base_filename += f"_{self.label}"
        # Put other possible options after the tag so we can sort by tag if it exists.
        if self.use_pure_matches:
            base_filename += "_pure_matches"
        return base_filename

    @property
    def output_filename(self) -> Path:
        # NOTE: We can't use with_suffix here because the filename may contain ".", which will mess up
        #       the detection of the suffix to replace.
        return Path(f"{self.output_dir / self.output_tag}.root")


def _encode_binning_in_str(array: np.ndarray) -> str:
    """Encode numpy array in safe string for a histogram name.

    Here, we put "_" between entries, leave "_" signs as is, and encode
    decimal points as "p". This is ugly, but for our purposes, unambiguous.

    Args:
        array: Array to be encoded.

    Returns:
        Array encoded into a string.
    """
    # Handle the no untagged case. We don't want to repeat, as the histogram we'll compare to won't repeat.
    array = np.unique(array)
    return "_".join([f"{v:g}".replace(".", "p") for v in array])


def hist_name_for_ratio_2D(
    grooming_method: str,
    prefix_for_ratio: str,
    smeared_substructure_variable_bins: np.ndarray,
    smeared_jet_pt_bins: np.ndarray,
) -> str:
    return f"{grooming_method}_{prefix_for_ratio}_kt_jet_pt_binning_smeared_kt_{_encode_binning_in_str(smeared_substructure_variable_bins)}_smeared_jet_pt_{_encode_binning_in_str(smeared_jet_pt_bins)}"


def get_binning(
    unfolding_settings: Mapping[str, Any],
    base_unfolding_config: Mapping[str, Any],
    name: str,
    grooming_method: str,
) -> np.ndarray:
    """Get unfolding binning for a particular axis name.

    If a axis isn't specified, we fall back to the nominal binning.

    Args:
        unfolding_settings: Unfolding settings for a particular case.
        base_unfolding_config: Base unfolding config for the system.
        name: Name of the axis to retrieve such as "smeared_jet_pt", etc.
        grooming_method: Name of the grooming method to allow for specialization
            of the binning based on the grooming method.
    Returns:
        Binning for that axis.
    """

    binning = None
    specialized_binning = unfolding_settings.get("binning", {})
    if specialized_binning:
        binning = specialized_binning.get(grooming_method, {}).get(name, [])
        if not binning:
            binning = specialized_binning.get("default", {}).get(name, [])
    # If not available in the specialized unfolding config, then grab it from the base config.
    if not binning:
        nominal_binning = base_unfolding_config["nominal_binning"]
        binning = nominal_binning.get(grooming_method, {}).get(name, [])
        if not binning:
            binning = nominal_binning["default"][name]

    return np.array(binning, dtype=np.float64)
