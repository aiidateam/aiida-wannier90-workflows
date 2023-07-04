#!/usr/bin/env python
"""Functions to calculate bands distance."""
import typing as ty

import numpy as np

from aiida import orm


def fermi_dirac(energy: np.array, mu: float, sigma: float) -> np.array:
    """Fermi-Dirac distribution function."""
    return 1.0 / (np.exp((energy - mu) / sigma) + 1.0)


def compute_lower_cutoff(energy: np.array, lower_cutoff: float) -> np.array:
    """Return a mask to remove eigenvalues smaller equal than ``lower_cutoff``."""
    if lower_cutoff is None:
        lower_cutoff = energy.min() - 1.0
    return np.array(energy > lower_cutoff, dtype=int)


def bands_distance_raw(  # pylint: disable=too-many-arguments,too-many-locals
    dft_bands: np.array,
    wannier_bands: np.array,
    mu: float,
    sigma: float,
    exclude_list_dft: list = None,
    lower_cutoff: float = None,
) -> tuple:
    """Calculate bands distance with specified ``mu`` and ``sigma``.

    :param dft_bands: a numpy array of size (num_k x num_dft) where num_dft is
       number of bands computed by the DFT code. In eV.
    :param wannier_bands: a numpy array of size (num_k x num_wan) where num_wan is
       number of Wannier functions.  In eV.
    :para mu, sigma: in eV.
    :param exclude_list_dft: if passed should be a list of the excluded bands,
       1-indexed
    """
    if exclude_list_dft is None:
        exclude_list_dft = []
        dft_bands_filtered = dft_bands
    else:
        # Code taken and *adapted* from the workflow (function get_exclude_bands)
        xb_startzero_set = {idx - 1 for idx in exclude_list_dft}
        # in Fortran/W90: 1-based; in py: 0-based
        keep_bands = np.array(
            [idx for idx in range(dft_bands.shape[1]) if idx not in xb_startzero_set]
        )

        dft_bands_filtered = dft_bands[:, keep_bands]

    # Check that the number of kpoints is the same
    assert (
        dft_bands_filtered.shape[0] == wannier_bands.shape[0]
    ), f"Different number of kpoints {dft_bands_filtered.shape[0]} {wannier_bands.shape[0]}"
    # assert dft_bands_filtered.shape[1] >= wannier_bands.shape[
    #     1], f'Too few DFT bands w.r.t. Wannier {dft_bands_filtered.shape[1]} {wannier_bands.shape[1]}'
    if dft_bands_filtered.shape[1] <= wannier_bands.shape[1]:
        wannier_bands_filtered = wannier_bands[:, : dft_bands_filtered.shape[1]]
    else:
        wannier_bands_filtered = wannier_bands

    dft_bands_to_compare = dft_bands_filtered[:, : wannier_bands_filtered.shape[1]]

    bands_energy_difference = dft_bands_to_compare - wannier_bands_filtered
    bands_weight_dft = fermi_dirac(
        dft_bands_to_compare, mu, sigma
    ) * compute_lower_cutoff(dft_bands_to_compare, lower_cutoff)
    bands_weight_wannier = fermi_dirac(
        wannier_bands_filtered, mu, sigma
    ) * compute_lower_cutoff(dft_bands_to_compare, lower_cutoff)
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
    bands_dft: ty.Union[orm.BandsData, np.array],
    bands_wannier: ty.Union[orm.BandsData, np.array],
    fermi_energy: float,
    exclude_list_dft: list = None,
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
    if isinstance(bands_dft, orm.BandsData):
        dft_bands = bands_dft.get_bands()
    else:
        dft_bands = bands_dft
    if isinstance(bands_wannier, orm.BandsData):
        wannier_bands = bands_wannier.get_bands()
    else:
        wannier_bands = bands_wannier

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


def bands_distance_isolated(  # pylint: disable=too-many-locals
    dft_bands: ty.Union[orm.BandsData, np.array],
    wannier_bands: ty.Union[orm.BandsData, np.array],
    exclude_list_dft: list = None,
    lower_cutoff: float = None,
) -> tuple:
    """Calculate bands distance with specified group of bands.

    :param dft_bands: a numpy array of size (num_k x num_dft) where num_dft is
       number of bands computed by the DFT code. In eV.
    :param wannier_bands: a numpy array of size (num_k x num_wan) where num_wan is
       number of Wannier functions.  In eV.
    :para mu, sigma: in eV.
    :param exclude_list_dft: if passed should be a list of the excluded bands,
       1-indexed
    """
    if isinstance(dft_bands, orm.BandsData):
        dft_bands = dft_bands.get_bands()
    if isinstance(wannier_bands, orm.BandsData):
        wannier_bands = wannier_bands.get_bands()

    if exclude_list_dft is None:
        exclude_list_dft = []
        dft_bands_filtered = dft_bands
    else:
        # Code taken and *adapted* from the workflow (function get_exclude_bands)
        xb_startzero_set = {idx - 1 for idx in exclude_list_dft}
        # in Fortran/W90: 1-based; in py: 0-based
        keep_bands = np.array(
            [idx for idx in range(dft_bands.shape[1]) if idx not in xb_startzero_set]
        )

        dft_bands_filtered = dft_bands[:, keep_bands]

    # Check that the number of kpoints is the same
    assert (
        dft_bands_filtered.shape[0] == wannier_bands.shape[0]
    ), f"Different number of kpoints {dft_bands_filtered.shape[0]} {wannier_bands.shape[0]}"
    # assert dft_bands_filtered.shape[1] >= wannier_bands.shape[
    #     1], f'Too few DFT bands w.r.t. Wannier {dft_bands_filtered.shape[1]} {wannier_bands.shape[1]}'
    if dft_bands_filtered.shape[1] <= wannier_bands.shape[1]:
        wannier_bands_filtered = wannier_bands[:, : dft_bands_filtered.shape[1]]
    else:
        wannier_bands_filtered = wannier_bands

    dft_bands_to_compare = dft_bands_filtered[:, : wannier_bands_filtered.shape[1]]

    bands_energy_difference = dft_bands_to_compare - wannier_bands_filtered
    bands_weight_dft = compute_lower_cutoff(dft_bands_to_compare, lower_cutoff)
    bands_weight_wannier = compute_lower_cutoff(dft_bands_to_compare, lower_cutoff)
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
