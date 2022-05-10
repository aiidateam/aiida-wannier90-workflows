#!/usr/bin/env python
"""Processthe Wannier function spreads."""
import typing as ty

import numpy as np

from aiida import orm

from aiida_wannier90.calculations import Wannier90Calculation


def get_wf_spreads(calculation: Wannier90Calculation, initial: bool = False) -> tuple:
    """Get Wannier function spreads.

    :param calculation: A finished ``Wannier90Calculation``.
    :type calculation: Wannier90Calculation
    :param initial: Get initial or final WF center.
    :type initial: bool
    :return:
    :rtype: tuple
    """
    if initial:
        wf_outputs_key = "wannier_functions_initial"
    else:
        wf_outputs_key = "wannier_functions_output"

    wf_outputs = calculation.outputs.output_parameters[wf_outputs_key]
    wf_spreads = np.zeros(len(wf_outputs))
    for wf in wf_outputs:  # pylint: disable=invalid-name
        wf_id = wf["wf_ids"] - 1
        wf_spreads[wf_id] = wf["wf_spreads"]

    return wf_spreads


def wf_spreads_for_group(group: ty.Union[orm.Group, str, int]) -> np.array:
    """Calculate distance of Wannier function center to nearest atom for a group of WorkChain.

    :param group: [description]
    :type group: orm.Group, str, int
    :return: [description]
    :rtype: np.array
    """
    from aiida_wannier90_workflows.utils.parser.center import get_last_wan_calc

    spreads = []

    if not isinstance(group, orm.Group):
        group = orm.load_group(group)

    for node in group.nodes:
        if not node.is_finished_ok:
            print(f"Skip unfinished node: {node}")
            continue

        calc = get_last_wan_calc(node)
        sprd = get_wf_spreads(calc)
        spreads.extend(sprd)

    spreads = np.array(spreads)

    return spreads


def plot_histogram(spreads: np.array, title: str = None):
    """Plot a histogram of Wannier function centers to nearest atom distances.

    :param distances: [description]
    :type distances: np.array
    """
    import matplotlib.pyplot as plt

    plt.hist(spreads, 100)

    plt.xlabel("WF spreads / Angstrom^2")
    plt.ylabel("Count")
    pre_title = f"Histogram of {len(spreads)} WFs"
    if title is not None:
        full_title = f"{pre_title}, {title}"
        plt.title(full_title)
    plt.grid(True)
    plt.annotate(
        f"average = {np.average(spreads):.4f}", xy=(0.7, 0.9), xycoords="axes fraction"
    )

    # plt.savefig('spreads.png')
    plt.show()
