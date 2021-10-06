#!/usr/bin/env python
import numpy as np
from aiida import orm


def fermi_dirac(e: np.array, mu: float, sigma: float) -> np.array:
    return 1.0 / (np.exp((e - mu) / sigma) + 1.0)


def compute_lower_cutoff(e: np.array, lower_cutoff: float) -> np.array:
    if lower_cutoff is None:
        lower_cutoff = e.min() - 1.0
    return np.array(e > lower_cutoff, dtype=int)


def bands_distance_raw(
    dft_bands: np.array,
    wannier_bands: np.array,
    mu: float,
    sigma: float,
    exclude_list_dft: list = None,
    lower_cutoff: float = None
) -> tuple:
    """
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
        xb_startzero_set = set([idx - 1 for idx in exclude_list_dft])
        # in Fortran/W90: 1-based; in py: 0-based
        keep_bands = np.array([
            idx for idx in range(dft_bands.shape[1])
            if idx not in xb_startzero_set
        ])

        dft_bands_filtered = dft_bands[:, keep_bands]

    # Check that the number of kpoints is the same
    assert dft_bands_filtered.shape[0] == wannier_bands.shape[
        0
    ], f"Different number of kpoints {dft_bands_filtered.shape[0]} {wannier_bands.shape[0]}"
    assert dft_bands_filtered.shape[1] >= wannier_bands.shape[
        1
    ], f"Too few DFT bands w.r.t. Wannier {dft_bands_filtered.shape[1]} {wannier_bands.shape[1]}"

    dft_bands_to_compare = dft_bands_filtered[:, :wannier_bands.shape[1]]

    bands_energy_difference = (dft_bands_to_compare - wannier_bands)
    bands_weight_dft = fermi_dirac(
        dft_bands_to_compare, mu, sigma
    ) * compute_lower_cutoff(dft_bands_to_compare, lower_cutoff)
    bands_weight_wannier = fermi_dirac(
        wannier_bands, mu, sigma
    ) * compute_lower_cutoff(dft_bands_to_compare, lower_cutoff)
    bands_weight = np.sqrt(bands_weight_dft * bands_weight_wannier)

    bands_distance = np.sqrt(
        np.sum(bands_energy_difference**2 * bands_weight) /
        np.sum(bands_weight)
    )

    arr = bands_energy_difference**2 * bands_weight
    max_distance = np.sqrt(np.max(arr))
    max_distance_loc = np.unravel_index(np.argmax(arr, axis=None), arr.shape)
    # print(np.shape(arr), max_distance_loc)
    arr_2 = np.abs(bands_energy_difference) * bands_weight
    max_distance_2 = np.max(arr_2)
    max_distance_2_loc = np.unravel_index(
        np.argmax(arr_2, axis=None), arr_2.shape
    )

    return (
        bands_distance, max_distance, max_distance_2, max_distance_loc,
        max_distance_2_loc
    )


def bands_distance(
    bands_dft: orm.BandsData,
    bands_wannier: orm.BandsData,
    fermi_energy: float,
    exclude_list_dft: list = None
) -> np.array:
    dft_bands = bands_dft.get_bands()
    wannier_bands = bands_wannier.get_bands()

    # mu_range = np.arange(-60, 40, 0.5)
    start = fermi_energy - 1
    stop = fermi_energy + 5
    # add a small eps to arange stop, so fermi+5 is always included
    mu_range = np.arange(start, stop + 0.0001, 0.5)

    dist = -1 * np.ones((len(mu_range), 4))

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
    num_kpoints, num_bands = bands.shape

    if not set(exclude_bands).issubset(set(range(num_bands))):
        raise ValueError(
            f"exclude_bands {exclude_bands} not in the range of available bands {range(num_bands)}"
        )

    # Remove repetition and sort
    exclude_bands = sorted(set(exclude_bands))

    sub_bands = np.delete(bands, exclude_bands, axis=1)

    return sub_bands
