"""Functions for analyzing the band distances of WorkChain results."""

import typing as ty

import numpy as np
import pandas as pd

from aiida import orm


def bands_distance_for_group(  # pylint: disable=too-many-statements,too-many-locals,too-many-branches
    wan_group: ty.Union[orm.Group, str],
    dft_group: ty.Union[orm.Group, str],
    match_by_formula: bool = False,
) -> pd.DataFrame:
    """Calculate bands distance for a group of DFT Calculation and Wannier WorkChain.

    :param wan_group: [description]
    :type wan_group: ty.Union[orm.Group, str]
    :param dft_group: [description]
    :type dft_group: ty.Union[orm.Group, str]
    :return: [description]
    :rtype: pd.DataFrame
    """
    from aiida.plugins import WorkflowFactory

    from aiida_quantumespresso.calculations.pw import PwCalculation
    from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

    from aiida_wannier90.calculations import Wannier90Calculation

    from aiida_wannier90_workflows.utils.bands.distance import bands_distance
    from aiida_wannier90_workflows.utils.workflows.group import get_mapping_for_group
    from aiida_wannier90_workflows.utils.workflows.plot.bands import (
        get_workchain_fermi_energy,
    )

    Wannier90BandsWorkChain = WorkflowFactory("wannier90_workflows.bands")
    Wannier90OptimizeWorkChain = WorkflowFactory("wannier90_workflows.optimize")

    if isinstance(wan_group, str):
        wan_group = orm.load_group(wan_group)
    if isinstance(dft_group, str):
        dft_group = orm.load_group(dft_group)

    if wan_group.nodes[0].process_class != Wannier90OptimizeWorkChain:
        mapping = get_mapping_for_group(wan_group, dft_group, match_by_formula)

    columns = [
        "formula",
        "wan_workchain",
        "dft_workchain",
        "fermi_energy",
    ]
    mu_range = range(0, 6)
    columns.extend([f"bands_dist_ef+{mu}" for mu in mu_range])
    columns.extend([f"bands_maxdist_ef+{mu}" for mu in mu_range])
    columns.extend([f"bands_maxdist2_ef+{mu}" for mu in mu_range])
    print(columns)

    result = []
    for wan_wc in wan_group.nodes:
        structure = wan_wc.inputs.structure
        formula = structure.get_formula()

        if not wan_wc.is_finished_ok:
            print(f"! Skip unfinished {wan_wc.process_label}<{wan_wc.pk}> of {formula}")
            continue

        if (
            wan_wc.process_class == Wannier90OptimizeWorkChain
            and "optimize_reference_bands" in wan_wc.inputs
        ):
            bands_wc = (
                wan_wc.inputs.optimize_reference_bands.base.links.get_incoming(
                    link_label_filter="band_structure"
                )
                .one()
                .node
            )
        else:
            bands_wc = mapping[wan_wc]

        if bands_wc is None:
            msg = f"! Cannot find DFT bands for {wan_wc.process_label}<{wan_wc.pk}> of {formula}"
            print(msg)
            continue

        if not bands_wc.is_finished_ok:
            print(
                f"! Skip unfinished DFT {wan_wc.process_label}<{bands_wc.pk}> of {formula}"
            )
            continue
        if bands_wc.process_class in (PwBaseWorkChain, PwCalculation):
            bands_dft_node = bands_wc.outputs.output_band
        elif bands_wc.process_class == PwBandsWorkChain:
            bands_dft_node = bands_wc.outputs.band_structure
        else:
            raise ValueError(
                f"Unsupported node type {bands_wc.process_class}<{bands_wc.pk}>"
            )

        if wan_wc.process_class == Wannier90Calculation:
            fermi_energy = wan_wc.inputs.parameters["fermi_energy"]
            bands_wannier_node = wan_wc.outputs.interpolated_bands
            try:
                exclude_list_dft = wan_wc.inputs.parameters["exclude_bands"]
            except KeyError:
                exclude_list_dft = []
        elif wan_wc.process_class in (
            Wannier90BandsWorkChain,
            Wannier90OptimizeWorkChain,
        ):
            fermi_energy = get_workchain_fermi_energy(wan_wc)
            # In very rare cases, the workchain did not output a correct bands,
            # e.g. the bands file was not correctly written due to disk issue,
            # so the outputs.band_structure might be empty.
            # Here if the band is empty, I try to use another output BandsData,
            # in principle they should be the same.
            if wan_wc.process_class == Wannier90OptimizeWorkChain:
                bands_wannier_node = wan_wc.outputs.wannier90_optimal.interpolated_bands
            else:
                bands_wannier_node = wan_wc.outputs.wannier90.interpolated_bands
            if np.prod(bands_wannier_node.base.attributes.all["array|bands"]) == 0:
                bands_wannier_node = wan_wc.outputs.band_structure
            try:
                last_wan = (
                    wan_wc.base.links.get_outgoing(link_label_filter="wannier90")
                    .one()
                    .node
                )
                if "parameters" in last_wan.inputs:
                    exclude_list_dft = last_wan.inputs["parameters"]["exclude_bands"]
                else:
                    exclude_list_dft = last_wan.inputs["wannier90"]["parameters"][
                        "exclude_bands"
                    ]
            except KeyError:
                exclude_list_dft = []

        print(bands_wc.pk, wan_wc.pk)
        dist = bands_distance(
            bands_dft_node, bands_wannier_node, fermi_energy, exclude_list_dft
        )

        res = [formula, wan_wc.pk, bands_wc.pk, fermi_energy]
        # bands_dist_ef+{mu}
        res.extend([dist[_, 1] for _ in range(len(mu_range))])
        # bands_maxdist_ef+{mu}
        res.extend([dist[_, 2] for _ in range(len(mu_range))])
        # bands_maxdist2_ef+{mu}
        res.extend([dist[_, 3] for _ in range(len(mu_range))])

        result.append(res)
        print(res)

    dataframe = pd.DataFrame(result, columns=columns)

    return dataframe


def save_distance(distance: pd.DataFrame, hdf_file: str):
    """Save bands distance to a HDF5 file.

    :param distance: [description]
    :type distance: pd.DataFrame
    :param hdf_file: [description]
    :type hdf_file: str
    """
    store = pd.HDFStore(hdf_file)

    # save it
    store["df"] = distance
    store.close()

    print(f"Saved to {hdf_file}")


def read_distance(hdf_file: str) -> pd.DataFrame:
    """Load bands distance stored in a HDF5 file."""
    import os.path

    if not os.path.exists(hdf_file):
        raise ValueError(f"File not existed: {hdf_file}")

    store = pd.HDFStore(hdf_file)

    # load it
    df = store["df"]
    store.close()

    return df


def plot_distance(  # pylint: disable=too-many-locals
    df: pd.DataFrame, max_dist: bool = False, show: bool = True
) -> None:  # pylint: disable=too-many-locals
    """Plot a histogram of bands distance."""
    import matplotlib.pyplot as plt

    # labels are the label for each column of distance
    mu_range = range(0, 6)
    labels = [f"Ef+{i}eV" for i in mu_range]
    # bands distance \eta
    distance = np.zeros((len(df), len(labels)))
    # pks are the wannier workchain PK of each row of eta
    pks = np.zeros(len(df), dtype=int)

    if max_dist:
        # I use the abs distance
        eta_index = "bands_maxdist2_ef+"
    else:
        eta_index = "bands_dist_ef+"

    # Some times the dataframe index is not continous, (after filtering the dataframe),
    # so we need a counter i to index distance[...].
    i = 0
    for _, row in df.iterrows():
        for j, mu in enumerate(mu_range):
            key = f"{eta_index}{mu}"
            distance[i, j] = row[key] * 1e3  # to meV
        pks[i] = row["wan_workchain"]
        i += 1

    fig, axs = plt.subplots(
        len(labels),
        1,
        sharex=True,
        sharey=True,
        # figsize=(10,8.21/6.47*10)
    )

    data_range = (distance.min(), distance.max())
    # print(f'data_range {data_range}')

    for i, lab in enumerate(labels):
        data = distance[:, i]
        label = f"{lab}, aver = {np.average(data):.3f}meV"
        # print(f'data ({data.min()}, {data.max()})')

        axs[i].hist(x=data, bins=100, range=data_range, label=label)
        axs[i].legend()
        # axs[i].grid(True)

    # Add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor="none", top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    if max_dist:
        plt.xlabel("eta_max (meV)")
    else:
        plt.xlabel("eta (meV)")
    plt.ylabel("Count")
    plt.title(f"Histogram of {len(pks)} structures")
    # plt.tight_layout()
    # plt.savefig('distances.pdf')
    # plt.savefig('wf_center_xyz/' + 'distances.png')

    if show:
        plt.show()

    return fig
