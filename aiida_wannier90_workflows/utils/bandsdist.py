#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions to calculate bands distance."""
import typing as ty
import numpy as np
import pandas as pd
from aiida import orm


def fermi_dirac(energy: np.array, mu: float, sigma: float) -> np.array:
    """Fermi-Dirac distribution function."""
    return 1.0 / (np.exp((energy - mu) / sigma) + 1.0)


def compute_lower_cutoff(energy: np.array, lower_cutoff: float) -> np.array:
    """Return a mask to remove eigenvalues smaller equal than ``lower_cutoff``."""
    if lower_cutoff is None:
        lower_cutoff = energy.min() - 1.0
    return np.array(energy > lower_cutoff, dtype=int)


def bands_distance_raw(
    dft_bands: np.array,
    wannier_bands: np.array,
    mu: float,
    sigma: float,
    exclude_list_dft: list = None,
    lower_cutoff: float = None
) -> tuple:
    """Calculate bands distance with specified ``mu`` and ``sigma``.

    :param dft_bands: a numpy array of size (num_k x num_dft) where num_dft is
       number of bands computed by the DFT code. In eV.
    :param wannier_bands: a numpy array of size (num_k x num_wan) where num_wan is
       number of Wannier functions.  In eV.
    :para mu, sigma: in eV.
    :param exclude_list_dft: if passed should be a list of the excluded bands,
       zero-indexed (subtract 1 from the input of Wannier)
    """
    if exclude_list_dft is None:
        exclude_list_dft = []
        dft_bands_filtered = dft_bands
    else:
        # Code taken and *adapted* from the workflow (function get_exclude_bands)
        xb_startzero_set = set(idx - 1 for idx in exclude_list_dft)
        # in Fortran/W90: 1-based; in py: 0-based
        keep_bands = np.array([idx for idx in range(dft_bands.shape[1]) if idx not in xb_startzero_set])

        dft_bands_filtered = dft_bands[:, keep_bands]

    # Check that the number of kpoints is the same
    assert dft_bands_filtered.shape[0] == wannier_bands.shape[
        0], f'Different number of kpoints {dft_bands_filtered.shape[0]} {wannier_bands.shape[0]}'
    assert dft_bands_filtered.shape[1] >= wannier_bands.shape[
        1], f'Too few DFT bands w.r.t. Wannier {dft_bands_filtered.shape[1]} {wannier_bands.shape[1]}'

    dft_bands_to_compare = dft_bands_filtered[:, :wannier_bands.shape[1]]

    bands_energy_difference = (dft_bands_to_compare - wannier_bands)
    bands_weight_dft = fermi_dirac(dft_bands_to_compare, mu,
                                   sigma) * compute_lower_cutoff(dft_bands_to_compare, lower_cutoff)
    bands_weight_wannier = fermi_dirac(wannier_bands, mu,
                                       sigma) * compute_lower_cutoff(dft_bands_to_compare, lower_cutoff)
    bands_weight = np.sqrt(bands_weight_dft * bands_weight_wannier)

    arr = bands_energy_difference**2 * bands_weight
    bands_dist = np.sqrt(np.sum(arr) / np.sum(bands_weight))

    # max distance
    max_dist = np.sqrt(np.max(arr))
    max_dist_loc = np.unravel_index(np.argmax(arr, axis=None), arr.shape)
    # print(np.shape(arr), max_distance_loc)

    arr_2 = np.abs(bands_energy_difference) * bands_weight
    # max abs difference
    max_dist_2 = np.max(arr_2)
    max_dist_2_loc = np.unravel_index(np.argmax(arr_2, axis=None), arr_2.shape)

    return (bands_dist, max_dist, max_dist_2, max_dist_loc, max_dist_2_loc)


def bands_distance(
    bands_dft: orm.BandsData,
    bands_wannier: orm.BandsData,
    fermi_energy: float,
    exclude_list_dft: list = None
) -> np.array:
    """Calculate bands distance with ``mu`` set as Ef to Ef+5.

    :param bands_dft: [description]
    :type bands_dft: orm.BandsData
    :param bands_wannier: [description]
    :type bands_wannier: orm.BandsData
    :param fermi_energy: [description]
    :type fermi_energy: float
    :param exclude_list_dft: [description], defaults to None
    :type exclude_list_dft: list, optional
    :return: [description], unit is eV.
    :rtype: np.array
    """
    dft_bands = bands_dft.get_bands()
    wannier_bands = bands_wannier.get_bands()

    # mu_range = np.arange(-60, 40, 0.5)
    start = fermi_energy
    stop = fermi_energy + 5
    # add a small eps to arange stop, so fermi+5 is always included
    mu_range = np.arange(start, stop + 0.0001, 1)

    dist = np.full((len(mu_range), 4), np.nan)

    for i, mu in enumerate(mu_range):
        res = bands_distance_raw(
            dft_bands=dft_bands,
            wannier_bands=wannier_bands,
            exclude_list_dft=exclude_list_dft,
            mu=mu,
            sigma=0.1,
            lower_cutoff=-30,
        )
        # mu, bands_distance, max_distance, max_distance_2
        dist[i, :] = [mu, res[0], res[1], res[2]]

    return dist


def remove_exclude_bands(bands: np.array, exclude_bands: list) -> np.array:
    """Remove bands according the index specified by `exclude_bands`.

    :param bands: num_kpoints x num_bands
    :type bands: np.array
    :param exclude_bands: the index of the to-be-excluded bands, 0-based indexing
    :type exclude_bands: list
    :return: the bands with exclude_bands removed
    :rtype: np.array
    """
    num_kpoints, num_bands = bands.shape  # pylint: disable=unused-variable

    if not set(exclude_bands).issubset(set(range(num_bands))):
        raise ValueError(f'exclude_bands {exclude_bands} not in the range of available bands {range(num_bands)}')

    # Remove repetition and sort
    exclude_bands = sorted(set(exclude_bands))

    sub_bands = np.delete(bands, exclude_bands, axis=1)

    return sub_bands


def bands_distance_for_group(  # pylint: disable=too-many-statements
    wan_group: ty.Union[orm.Group, str],
    dft_group: ty.Union[orm.Group, str],
    match_by_formula: bool = False
) -> pd.DataFrame:
    """Calculate bands distance for a group of DFT Calculation and Wannier WorkChain.

    :param wan_group: [description]
    :type wan_group: ty.Union[orm.Group, str]
    :param dft_group: [description]
    :type dft_group: ty.Union[orm.Group, str]
    :return: [description]
    :rtype: pd.DataFrame
    """
    from aiida_quantumespresso.calculations.pw import PwCalculation
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
    from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
    from aiida_wannier90.calculations import Wannier90Calculation
    from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain
    from aiida_wannier90_workflows.utils.plot import get_mapping_for_group, get_wannier_workchain_fermi_energy

    if isinstance(wan_group, str):
        wan_group = orm.load_group(wan_group)
    if isinstance(dft_group, str):
        dft_group = orm.load_group(dft_group)

    mapping = get_mapping_for_group(wan_group, dft_group, match_by_formula)

    columns = [
        'formula',
        'wan_workchain',
        'dft_workchain',
        'fermi_energy',
    ]
    mu_range = range(0, 6)
    columns.extend([f'bands_dist_ef+{mu}' for mu in mu_range])
    columns.extend([f'bands_maxdist_ef+{mu}' for mu in mu_range])
    columns.extend([f'bands_maxdist2_ef+{mu}' for mu in mu_range])
    print(columns)

    result = []
    for wan_wc in wan_group.nodes:
        structure = wan_wc.inputs.structure
        formula = structure.get_formula()

        if not wan_wc.is_finished_ok:
            print(f'! Skip unfinished {wan_wc.process_label}<{wan_wc.pk}> of {formula}')
            continue

        bands_wc = mapping[wan_wc]
        if bands_wc is None:
            msg = f'! Cannot find DFT bands for {wan_wc.process_label}<{wan_wc.pk}> of {formula}'
            print(msg)
            continue

        if not bands_wc.is_finished_ok:
            print(f'! Skip unfinished DFT {wan_wc.process_label}<{bands_wc.pk}> of {formula}')
            continue
        if bands_wc.process_class in (PwBaseWorkChain, PwCalculation):
            bands_dft_node = bands_wc.outputs.output_band
        elif bands_wc.process_class == PwBandsWorkChain:
            bands_dft_node = bands_wc.outputs.band_structure
        else:
            raise ValueError(f'Unsupported node type {bands_wc.process_class}<{bands_wc.pk}>')

        if wan_wc.process_class == Wannier90Calculation:
            fermi_energy = wan_wc.inputs.parameters['fermi_energy']
            bands_wannier_node = wan_wc.outputs.interpolated_bands
            try:
                exclude_list_dft = wan_wc.inputs.parameters['exclude_bands']
            except KeyError:
                exclude_list_dft = []
        elif wan_wc.process_class == Wannier90BandsWorkChain:
            fermi_energy = get_wannier_workchain_fermi_energy(wan_wc)
            bands_wannier_node = wan_wc.outputs.band_structure
            try:
                last_wan = wan_wc.get_outgoing(link_label_filter='wannier90').one().node
                if 'parameters' in last_wan.inputs:
                    exclude_list_dft = last_wan.inputs['parameters']['exclude_bands']
                else:
                    exclude_list_dft = last_wan.inputs['wannier90']['parameters']['exclude_bands']
            except KeyError:
                exclude_list_dft = []

        print(bands_wc.pk, wan_wc.pk)
        dist = bands_distance(bands_dft_node, bands_wannier_node, fermi_energy, exclude_list_dft)

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


def standardize_groupname(label: str) -> str:
    """Replace ``/`` in group label to ``_``, to be used as filename.

    :param label: [description]
    :type label: str
    :return: [description]
    :rtype: str
    """
    new_label = label.replace('/', '-')
    return new_label


def save_distance(distance: pd.DataFrame, hdf_file: str):
    """Save bands distance to a HDF5 file.

    :param distance: [description]
    :type distance: pd.DataFrame
    :param hdf_file: [description]
    :type hdf_file: str
    """
    store = pd.HDFStore(hdf_file)

    # save it
    store['df'] = distance
    store.close()

    print(f'Saved to {hdf_file}')


def read_distance(hdf_file: str) -> pd.DataFrame:
    """Load bands distance stored in a HDF5 file."""
    import os.path

    if not os.path.exists(hdf_file):
        raise ValueError(f'File not existed: {hdf_file}')

    store = pd.HDFStore(hdf_file)

    # load it
    df = store['df']
    store.close()

    return df


def plot_distance(df: pd.DataFrame, max_dist: bool = False, show: bool = True) -> None:
    """Plot a histogram of bands distance."""
    import matplotlib.pyplot as plt

    # labels are the label for each column of distance
    mu_range = range(0, 6)
    labels = [f'Ef+{i}eV' for i in mu_range]
    # bands distance \eta
    distance = np.zeros((len(df), len(labels)))
    # pks are the wannier workchain PK of each row of eta
    pks = np.zeros(len(df), dtype=int)

    if max_dist:
        # I use the abs distance
        eta_index = 'bands_maxdist2_ef+'
    else:
        eta_index = 'bands_dist_ef+'

    # Some times the dataframe index is not continous, (after filtering the dataframe),
    # so we need a counter i to index distance[...].
    i = 0
    for _, row in df.iterrows():
        for j, mu in enumerate(mu_range):
            key = f'{eta_index}{mu}'
            distance[i, j] = row[key] * 1e3  # to meV
        pks[i] = row['wan_workchain']
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
        label = f'{lab}, aver = {np.average(data):.3f}meV'
        # print(f'data ({data.min()}, {data.max()})')

        axs[i].hist(x=data, bins=100, range=data_range, label=label)
        axs[i].legend()
        # axs[i].grid(True)

    # Add a big axis, hide frame
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    if max_dist:
        plt.xlabel('eta_max (meV)')
    else:
        plt.xlabel('eta (meV)')
    plt.ylabel('Count')
    plt.title(f'Histogram of {len(pks)} structures')
    # plt.tight_layout()
    # plt.savefig('distances.pdf')
    # plt.savefig('wf_center_xyz/' + 'distances.png')

    if show:
        plt.show()

    return fig
