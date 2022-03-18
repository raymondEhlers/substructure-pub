#!/usr/bin/env python3

""" Jet substructure related functionality.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import functools
from typing import Callable, Dict, Tuple, Type, TypeVar, Union

import attr
import awkward0 as ak
import numpy as np


# T_Input = TypeVar("T_Input")
T_Array = Union[ak.JaggedArray, np.ndarray]
T_Input = T_Array


def _dynamical_hardness_measure(delta_R: T_Input, z: T_Input, parent_pt: T_Input, R: float, a: float) -> T_Input:
    return z * (1 - z) * parent_pt * (delta_R / R) ** a


_z_drop = functools.partial(_dynamical_hardness_measure, a=0.1)
_kt_drop = functools.partial(_dynamical_hardness_measure, a=1.0)
_time_drop = functools.partial(_dynamical_hardness_measure, a=2.0)


def _calculate_dynamical_grooming(
    delta_R: T_Input,
    z: T_Input,
    parent_pt: T_Input,
    R: float,
    grooming_func: Callable[[T_Input, T_Input, T_Input, float], T_Input],
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate dynamical grooming using the giving grooming function.

    Returns:
        Dynamical grooming value, index of value.
    """
    values = grooming_func(delta_R, z, parent_pt, R)
    arg_max = values.argmax()
    return values[arg_max], arg_max


calculate_z_drop = functools.partial(_calculate_dynamical_grooming, grooming_func=_z_drop)
calculate_kt_drop = functools.partial(_calculate_dynamical_grooming, grooming_func=_kt_drop)
calculate_time_drop = functools.partial(_calculate_dynamical_grooming, grooming_func=_time_drop)


def calculate_kt_leading(kt: T_Input) -> Tuple[np.ndarray, np.ndarray]:
    arg_max = kt.argmax()
    return kt[arg_max], arg_max


def calculate_soft_drop(z: T_Input, z_hard_cutoff: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Soft Drop.

    Args:
        z: Shared momentum fraction of the splitting.
        z_hard_cutoff: Hard cutoff in z for SoftDrop.
    Returns:
        Soft drop z (1 per event), number of recusrive splittings passing soft drop (1 per event), index of
            each selected splitting (1 per event, in a JaggedArray format so it can be applied elsewhere).
    """
    z_cutoff_mask = z > z_hard_cutoff
    # We use :1 because this maintains the jagged structure. That way, we can apply it to initial arrays.
    z_indices = z.localindex[z_cutoff_mask][:, :1]
    z_g = z[z_indices].flatten()
    n_sd = z[z_cutoff_mask].count_nonzero()

    return z_g, n_sd, z_indices


T_SubstructureResult = TypeVar("T_SubstructureResult", bound="SubstructureResult")


@attr.s
class SubstructureResult:
    name: str = attr.ib()
    title: str = attr.ib()
    values: T_Array = attr.ib()
    indices: T_Array = attr.ib()
    delta_R: T_Array = attr.ib()
    z: T_Array = attr.ib()
    kt: T_Array = attr.ib()

    def __getitem__(self, val: str) -> T_Array:
        return getattr(self, val)

    @property
    def splitting_number(self) -> T_Array:
        try:
            return self._splitting_number
        except AttributeError:
            # +1 because splittings counts from 1, but indexing starts from 0.
            splitting_number = self.indices + 1
            # If there were no splittings, we want to set that to 0.
            splitting_number = splitting_number.pad(1).fillna(0)
            # Must flatten because the indices are still jagged.
            self._splitting_number: T_Array = splitting_number.flatten()

        return self._splitting_number

    @splitting_number.setter
    def splitting_number(self, value: T_Array) -> None:
        self._splitting_number = value

    @classmethod
    def from_full_dataset(
        cls: Type[T_SubstructureResult],
        name: str,
        title: str,
        values: T_Array,
        indices: T_Array,
        delta_R: T_Array,
        z: T_Array,
        kt: T_Array,
    ) -> T_SubstructureResult:
        return cls(
            name=name,
            title=title,
            values=values,
            indices=indices,
            delta_R=delta_R[indices].flatten(),
            z=z[indices].flatten(),
            kt=kt[indices].flatten(),
        )


T_SoftDropGroomingResult = TypeVar("T_SoftDropGroomingResult", bound="SoftDropGroomingResult")


@attr.s
class SoftDropGroomingResult(SubstructureResult):
    hard_cutoff: float = attr.ib()
    n_sd: T_Array = attr.ib()

    @classmethod
    def from_full_dataset(  # type: ignore
        cls: Type[T_SoftDropGroomingResult],
        name: str,
        title: str,
        values: T_Array,
        indices: T_Array,
        delta_R: T_Array,
        z: T_Array,
        kt: T_Array,
        hard_cutoff: float,
        n_sd: T_Array,
    ) -> T_SoftDropGroomingResult:
        return cls(
            name=name,
            title=title,
            values=values,
            indices=indices,
            delta_R=delta_R[indices].flatten(),
            z=z[indices].flatten(),
            kt=kt[indices].flatten(),
            hard_cutoff=hard_cutoff,
            n_sd=n_sd,
        )


def calculate_substructure_variables(
    arrays: Dict[str, T_Array], R: float, prefix: str = ""
) -> Tuple[
    SubstructureResult,
    SubstructureResult,
    SubstructureResult,
    SoftDropGroomingResult,
    SubstructureResult,
    SubstructureResult,
]:
    """Calculate jet substructure variables.

    Note:
        The array keys of the stored data need to be renamed using `normalize_array_names(...)`.

    Args:
        arrays: delta R, z, and kT arrays in one dict.
        R: Jet radius.
        prefix: Prefix for the keys in the arrays dict.
    Returns:
        z drop, kt drop, time drop, SD, kt leading, kt leading passing hard cutoff.
    """
    # Validation
    if prefix:
        prefix = f"{prefix}_"
    # Setup
    delta_R_name = f"{prefix}deltaR"
    z_name = f"{prefix}z"
    kt_name = f"{prefix}kt"
    # Ensure that the necessary variables are in stored in the arrays.
    for name in [delta_R_name, z_name, kt_name]:
        if name not in arrays:
            raise ValueError(f"Array {name} is not present in the passed arrays. Keys: {arrays}.")

    # delta_R = Delta R between the two subjets.
    # z = subleading / (leading + subleading)
    # kt = subleading * sin(delta_R)
    # parent_pt = subleading / z = kt / sin(delta_R) / z
    parent_pt_name = f"{prefix}parent_pt"
    arrays[parent_pt_name] = arrays[kt_name] / np.sin(arrays[delta_R_name]) / arrays[z_name]

    # Calculate dynamical grooming variables.
    # zDrop
    dynamical_z_values, dynamical_z_indices = calculate_z_drop(
        delta_R=arrays[delta_R_name], z=arrays[z_name], parent_pt=arrays[parent_pt_name], R=R
    )
    dynamical_z = SubstructureResult.from_full_dataset(
        name="dynamical_z",
        title="zDrop",
        values=dynamical_z_values,
        indices=dynamical_z_indices,
        delta_R=arrays[delta_R_name],
        z=arrays[z_name],
        kt=arrays[kt_name],
    )
    # kt drop
    dynamical_kt_values, dynamical_kt_indices = calculate_kt_drop(
        delta_R=arrays[delta_R_name], z=arrays[z_name], parent_pt=arrays[parent_pt_name], R=R
    )
    # NOTE: Dynamical kt gives us the hardest kt, but to put into, for example, the Lund Plane, we need
    #       to use the standard kt value.
    dynamical_kt = SubstructureResult.from_full_dataset(
        name="dynamical_kt",
        title="ktDrop",
        values=dynamical_kt_values,
        indices=dynamical_kt_indices,
        delta_R=arrays[delta_R_name],
        z=arrays[z_name],
        kt=arrays[kt_name],
    )
    # Time Drop
    dynamical_time_values, dynamical_time_indices = calculate_time_drop(
        delta_R=arrays[delta_R_name], z=arrays[z_name], parent_pt=arrays[parent_pt_name], R=R
    )
    dynamical_time = SubstructureResult.from_full_dataset(
        name="dynamical_time",
        title="TimeDrop",
        values=dynamical_time_values,
        indices=dynamical_time_indices,
        delta_R=arrays[delta_R_name],
        z=arrays[z_name],
        kt=arrays[kt_name],
    )

    # Soft Drop
    z_hard_cutoff = 0.2
    z_g, n_sd, z_indices = calculate_soft_drop(z=arrays[z_name], z_hard_cutoff=z_hard_cutoff)
    soft_drop = SoftDropGroomingResult.from_full_dataset(
        name="SD",
        title=f"SoftDrop $z > {z_hard_cutoff}$",
        values=z_g,
        indices=z_indices,
        delta_R=arrays[delta_R_name],
        z=arrays[z_name],
        kt=arrays[kt_name],
        hard_cutoff=z_hard_cutoff,
        n_sd=n_sd,
    )

    ## Leading kt
    leading_kt_values, leading_kt_indices = calculate_kt_leading(arrays[kt_name])
    leading_kt = SubstructureResult.from_full_dataset(
        name="leading_kt",
        title=r"Leading $k_{\text{T}}$",
        values=leading_kt_values,
        indices=leading_kt_indices,
        delta_R=arrays[delta_R_name],
        z=arrays[z_name],
        kt=arrays[kt_name],
    )
    leading_kt_hard_cutoff_values, leading_kt_hard_cutoff_indices = calculate_kt_leading(
        arrays[kt_name][arrays[z_name] > z_hard_cutoff]
    )
    leading_kt_hard_cutoff = SubstructureResult.from_full_dataset(
        name="leading_kt_hard_cutoff",
        title=fr"SD $z > {z_hard_cutoff}$ Leading $k_{{\text{{T}}}}$",
        values=leading_kt_hard_cutoff_values,
        indices=leading_kt_hard_cutoff_indices,
        delta_R=arrays[delta_R_name],
        z=arrays[z_name],
        kt=arrays[kt_name],
    )

    # NOTE: The number of jets normalization is just len(jet_pt)

    return dynamical_z, dynamical_kt, dynamical_time, soft_drop, leading_kt, leading_kt_hard_cutoff


def run() -> None:
    ...


if __name__ == "__main__":
    run()
