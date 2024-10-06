#!/usr/bin/env python
"""Functions to calculate bands distance for Wannier90OptimizeWorkChain."""
import typing as ty

import matplotlib.pyplot as plt
import numpy as np

from aiida_wannier90_workflows.workflows.optimize import Wannier90OptimizeWorkChain


def compute_checkerboard(
    optimize_workchain: Wannier90OptimizeWorkChain,
) -> ty.Tuple[np.array, np.array, np.array]:
    """Compute bands distance checkerboard data for a Wannier90OptimizeWorkChain.

    :param optimize_workchain: [description]
    :type optimize_workchain: Wannier90OptimizeWorkChain
    :raises ValueError: [description]
    :raises ValueError: [description]
    :return: [description]
    :rtype: np.array
    """
    from aiida.common import LinkType

    from aiida_wannier90_workflows.utils.bands.distance import bands_distance
    from aiida_wannier90_workflows.utils.workflows import get_last_calcjob

    if "optimize_reference_bands" not in optimize_workchain.inputs:
        raise ValueError("No `optimize_reference_bands` in workchain inputs")

    if not optimize_workchain.inputs.optimize_disproj:
        raise ValueError("Optimization not enabled in the workchain?")

    pw_bands = optimize_workchain.inputs.optimize_reference_bands

    min_range = optimize_workchain.inputs.optimize_disprojmin_range.get_list()
    max_range = optimize_workchain.inputs.optimize_disprojmax_range.get_list()

    wannier_base = (
        optimize_workchain.base.links.get_outgoing(link_label_filter="wannier90")
        .one()
        .node
    )
    wannier_calc = get_last_calcjob(wannier_base)
    wan_parameters = wannier_calc.inputs["parameters"].get_dict()
    fermi_energy = wan_parameters.get("fermi_energy")
    exclude_list_dft = wan_parameters.get("exclude_bands", None)

    # Last dimension = 6 is EF to EF+5eV
    checkerboard = np.full((len(max_range), len(min_range), 6), np.nan)

    all_optimize_workchains = optimize_workchain.base.links.get_outgoing(
        link_type=LinkType.CALL_WORK, link_label_filter="wannier90_optimize_iteration%"
    ).all()
    all_optimize_workchains = [_.node for _ in all_optimize_workchains]
    all_minmax = []
    for workchain in all_optimize_workchains:
        params = workchain.inputs.wannier90.parameters.get_dict()
        minmax = (params["dis_proj_min"], params["dis_proj_max"])
        all_minmax.append(minmax)

    for i, dis_proj_max in enumerate(max_range):
        for j, dis_proj_min in enumerate(min_range):
            minmax = (dis_proj_min, dis_proj_max)
            if minmax not in all_minmax:
                continue
            idx = all_minmax.index(minmax)
            workchain = all_optimize_workchains[idx]
            wan_bands = workchain.outputs.interpolated_bands
            bands_dist = bands_distance(
                pw_bands, wan_bands, fermi_energy, exclude_list_dft
            )
            # 0th column is the energy, discard it
            checkerboard[i, j, :] = bands_dist[:, 1]

    return checkerboard, max_range, min_range


def plot_checkerboard_raw(  # pylint: disable=inconsistent-return-statements
    checkerboard: np.array,
    max_range: np.array,
    min_range: np.array,
    eta_index: int = 2,
    title: str = None,
    ax: plt.Axes = None,
    show: bool = False,
) -> None:
    """Plot bands distance checkerboard from raw data.

    :param checkerboard: dimension len(max_range) * len(min_range) * 6
    :type checkerboard: np.array
    :param max_range: [description]
    :type max_range: np.array
    :param min_range: [description]
    :type min_range: np.array
    :param eta_index: select which distance to plot, i.e. z index of ``checkerboard``,
    ``range(6)``, defaults to None
    :type eta_index: int, optional
    :param title: [description], defaults to None
    :type title: str, optional
    :raises ValueError: [description]
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    num_x, num_y, num_z = checkerboard.shape
    if num_x != len(max_range) or num_y != len(min_range) or num_z != 6:
        raise ValueError("Incompatible input array dimensions")

    if eta_index < 0 or eta_index >= num_z:
        raise ValueError(f"eta_index {eta_index} is out of range")

    if ax is None:
        fig, axs = plt.subplots(3, 2, figsize=(24, 8))
        axs_flat = axs.flat
    else:
        fig = plt.gcf()
        axs_flat = None

    # To meV
    checkerboard = checkerboard * 1e3

    # print(max_range, min_range)
    # print(checkerboard[:, :, -1])

    # To percentage
    # label_x = np.array([f'{_*100}' for _ in max_range])
    # label_y = np.array([f'{_*100}' for _ in min_range])
    label_x = np.array([_ * 100 for _ in max_range])
    label_y = np.array([_ * 100 for _ in min_range])

    # Rearrange in ascending order
    ind_sort_x = np.argsort(max_range)
    ind_sort_y = np.argsort(min_range)

    if eta_index is not None:
        plot_indexes = [eta_index]
    else:
        plot_indexes = range(num_z)

    for idx_z in plot_indexes:
        sorted_checkerboard = checkerboard[:, :, idx_z]
        sorted_checkerboard = sorted_checkerboard[ind_sort_x, :]
        sorted_checkerboard = sorted_checkerboard[:, ind_sort_y]
        # Note the transpose
        sorted_checkerboard = sorted_checkerboard.T

        if axs_flat is not None:
            ax = axs_flat[idx_z]

        im = ax.imshow(
            sorted_checkerboard, origin="lower", cmap="RdYlBu_r"
        )  # pylint: disable=invalid-name
        eta_min = np.nanmin(sorted_checkerboard)
        eta_max = np.nanmax(sorted_checkerboard)
        ax.set_title(f"E <= EF+{idx_z}eV (meV), min={eta_min:.3f}, max={eta_max:.3f}")
        ax.set_xticks(range(len(label_x)))
        ax.set_xticklabels(label_x[ind_sort_x])
        ax.set_xlabel("dis_proj_max (%)")
        ax.set_yticks(range(len(label_y)))
        ax.set_yticklabels(label_y[ind_sort_y])
        ax.set_ylabel("dis_proj_min (%)")

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        fig.colorbar(im, cax=cax, orientation="vertical")

        # cbar.ax.tick_params(labelsize=8)
        # cbar.ax.set_ylabel('Band distance (meV)', rotation=270, labelpad=20)#,fontsize=8)

    if title:
        fig.suptitle(title, y=0.92)
    else:
        fig.suptitle("Bands distances checkerboard", y=0.92)

    if show:
        plt.show()
    else:
        return fig


def plot_checkerboard(
    optimize_workchain: Wannier90OptimizeWorkChain, filename: str = None
) -> None:
    """Plot bands distance checkerboard for a Wannier90OptimizeWorkChain.

    :param optimize_workchain: [description]
    :type optimize_workchain: Wannier90OptimizeWorkChain
    """

    checkerboard, max_range, min_range = compute_checkerboard(optimize_workchain)
    fig, axs = plt.subplots(3, 2, figsize=(24, 8))

    for eta_idx in range(6):
        plot_checkerboard_raw(
            checkerboard,
            max_range,
            min_range,
            eta_index=eta_idx,
            title=(
                f"Bands distance checkerboard for {optimize_workchain.process_label}"
                f"<{optimize_workchain.pk}> {optimize_workchain.inputs.structure.get_formula()}"
            ),
            ax=axs.flat[eta_idx],
            show=False,
        )

    if filename:
        fig.savefig(filename, bbox_inches="tight", dpi=300)
    else:
        fig.show()
