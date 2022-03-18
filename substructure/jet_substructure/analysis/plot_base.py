""" Plot base module.

Defines utilizes and settings.

.. codeuathor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

from __future__ import annotations

import logging
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import attr
import matplotlib
import numpy as np
import pachyderm.plot
import seaborn as sns
from pachyderm.plot import AxisConfig, Figure, LegendConfig, Panel, PlotConfig, TextConfig


logger = logging.getLogger(__name__)

pachyderm.plot.configure()


label_to_display_string: Dict[str, Dict[str, str]] = {
    "ALICE": dict(
        work_in_progress="ALICE Work in Progress",
        preliminary="ALICE Preliminary",
        final="ALICE",
        simulation="ALICE Simulation",
    ),
    "collision_system": dict(
        PbPb=r"$\text{Pb--Pb}\;\sqrt{s_{\text{NN}}} = 5.02$ TeV",
        embedPythia=r"$\text{{PYTHIA8}} \bigotimes \text{{{main_system}}}\;\text{{Pb--Pb}}\;\sqrt{{s_{{\text{{NN}}}}}} = 5.02$ TeV",
        pp_PbPb_5TeV=r"$\text{pp},\:\text{Pb--Pb}\;\sqrt{s_{\text{NN}}} = 5.02$ TeV",
        pp_5TeV=r"$\text{pp}\;\sqrt{s} = 5.02$ TeV",
        pp_5TeV_NN=r"$\text{pp}\;\sqrt{s_{\text{NN}}} = 5.02$ TeV",
        pythia_5TeV=r"$\text{PYTHIA8}\;\sqrt{s} = 5.02$ TeV",
        pythia_5TeV_NN=r"$\text{PYTHIA8}\;\sqrt{s_{\text{NN}}} = 5.02$ TeV",
    ),
    "jets": {f"R0{i}": (f"$R=0.{i}," + fr"\:|\eta_{{\text{{jet}}}}| < 0.{9-i}$") for i in range(1, 7)},
}
label_to_display_string["jets"]["general"] = r"$\text{Anti-}k_{\text{T}}\:\text{charged jets}$"


@attr.s
class GroomingMethodStyle:
    color: str = attr.ib()
    marker: str = attr.ib()
    fillstyle: str = attr.ib()
    label: str = attr.ib()
    zorder: int = attr.ib()


def define_grooming_styles() -> Dict[str, GroomingMethodStyle]:
    # Setup
    styles = {}

    greens = sns.color_palette("Greens_d", 4)
    purples = sns.color_palette("Purples_d", 3)
    reds = sns.color_palette("Reds_d", 3)
    # greys = sns.color_palette("Greys_r", 5)
    blues = sns.color_palette("Blues_r", 3)
    oranges = sns.color_palette("Oranges_r", 3)
    for label in ["", "_compare"]:
        if label == "":
            # These are our main colors.
            # The methods are similar, but different, so we want to spread out the colors.
            # dynamical_grooming_colors = sns.color_palette(f"GnBu_d", 3)
            dynamical_grooming_colors = sns.color_palette("Greens_d", 4)
            leading_kt_colors = sns.color_palette("Purples_d", 3)
            soft_drop_colors = sns.color_palette("Reds_d", 3)
        else:
            # These are our comparison colors. Similar in order and often shade, but distinct.
            dynamical_grooming_colors = sns.color_palette("Greys_r", 5)
            leading_kt_colors = sns.color_palette("Blues_r", 3)
            soft_drop_colors = sns.color_palette("Oranges_r", 3)
        markers = ["o", "d", "s"]
        grooming_styling = {
            f"dynamical_z{label}": GroomingMethodStyle(
                color=dynamical_grooming_colors[0], marker=markers[0], fillstyle="full", label="DyG. $z$", zorder=10
            ),
            f"dynamical_kt{label}": GroomingMethodStyle(
                color=greens[1],
                marker=markers[0],
                fillstyle="full",
                label=r"DyG $k_{\text{T}}$",
                zorder=10,
            ),
            f"dynamical_time{label}": GroomingMethodStyle(
                color=reds[1], marker=markers[2], fillstyle="full", label=r"DyG time", zorder=10
            ),
            f"dynamical_core{label}": GroomingMethodStyle(
                color=oranges[1], marker=markers[2], fillstyle="full", label=r"DyG core", zorder=10
            ),
            f"leading_kt{label}": GroomingMethodStyle(
                color=purples[1],
                marker=markers[1],
                fillstyle="full",
                label=r"Leading $k_{\text{T}}$",
                zorder=10,
            ),
            f"leading_kt_z_cut_02{label}": GroomingMethodStyle(
                color=blues[1] if not label else purples[1],
                marker=markers[1],
                fillstyle="none",
                label=r"Leading $k_{\text{T}}$ $z > 0.2$",
                zorder=4,
            ),
            f"leading_kt_z_cut_04{label}": GroomingMethodStyle(
                color=leading_kt_colors[1],
                marker=markers[2],
                fillstyle="full",
                label=r"Leading $k_{\text{T}}$ $z > 0.4$",
                zorder=10,
            ),
            # Leading kt with z cuts, but n <= 1
            f"leading_kt_z_cut_02_first_split{label}": GroomingMethodStyle(
                color=leading_kt_colors[0],
                marker="P",
                fillstyle="none",
                label=r"Leading $k_{\text{T}}$ $z > 0.2$, $n \leq 1$",
                zorder=4,
            ),
            f"leading_kt_z_cut_04_first_split{label}": GroomingMethodStyle(
                color=leading_kt_colors[0],
                marker="P",
                fillstyle="full",
                label=r"Leading $k_{\text{T}}$ $z > 0.4$, $n \leq 1$",
                zorder=10,
            ),
            f"soft_drop_z_cut_02{label}": GroomingMethodStyle(
                color=soft_drop_colors[1], marker=markers[1], fillstyle="none", label=r"SoftDrop $z > 0.2$", zorder=4
            ),
            f"soft_drop_z_cut_04{label}": GroomingMethodStyle(
                color=soft_drop_colors[1], marker=markers[2], fillstyle="full", label=r"SoftDrop $z > 0.4$", zorder=5
            ),
        }
        styles.update(grooming_styling)

    return styles
