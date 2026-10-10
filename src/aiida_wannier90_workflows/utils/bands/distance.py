#!/usr/bin/env python
"""Functions to calculate bands distance."""
import typing as ty

import numpy as np

from aiida import orm


def fermi_dirac(energy: np.array, mu: float, sigma: float) -> np.array:
    """Fermi-Dirac distribution function."""
    return 1.0 / (np.exp((energy - mu) / sigma) + 1.0)


def gaussian(energy: np.array, mu: float, sigma: float) -> np.array:
    """Gaussian distribution function."""
    return np.exp(-((energy - mu) ** 2) / (2 * sigma**2))


def compute_lower_cutoff(energy: np.array, lower_cutoff: float) -> np.array:
    """Return a mask to remove eigenvalues smaller equal than ``lower_cutoff``."""
    if lower_cutoff is None:
        lower_cutoff = energy.min() - 1.0
    return np.array(energy > lower_cutoff, dtype=int)


def as_bands_array(bands: ty.Union[orm.BandsData, np.array]) -> np.array:
    """Return the eigenvalues of ``bands`` as a numpy array."""
    if isinstance(bands, orm.BandsData):
        return bands.get_bands()
    return bands


def _align_bands(
    dft_bands: np.array,
    wannier_bands: np.array,
    exclude_list_dft: list = None,
) -> ty.Tuple[np.array, np.array]:
    """Drop the excluded DFT bands and truncate both arrays to a common shape.

    Spin-collinear bands, of shape (num_spins, num_kpts, num_bands), are moved
    to (num_kpts, num_bands, num_spins) first, so that the band index is always
    the second one.

    :param exclude_list_dft: if passed should be a list of the excluded bands,
       1-indexed
    :return: ``(dft_bands, wannier_bands)``, of equal shape.
    """
    if len(dft_bands.shape) == 3 and len(wannier_bands.shape) == 3:
        dft_bands = np.moveaxis(dft_bands, [-2, -1], [0, 1])
        wannier_bands = np.moveaxis(wannier_bands, [-2, -1], [0, 1])

    if exclude_list_dft is None or len(exclude_list_dft) == 0:
        dft_bands_filtered = dft_bands
    else:
        # Code taken and *adapted* from the workflow (function get_exclude_bands)
        # in Fortran/W90: 1-based; in py: 0-based
        xb_startzero_set = {idx - 1 for idx in exclude_list_dft}
        keep_bands = np.array(
            [idx for idx in range(dft_bands.shape[1]) if idx not in xb_startzero_set]
        )

        dft_bands_filtered = dft_bands[:, keep_bands]

    # Check that the number of kpoints is the same
    assert (
        dft_bands_filtered.shape[0] == wannier_bands.shape[0]
    ), f"Different number of kpoints {dft_bands_filtered.shape[0]} {wannier_bands.shape[0]}"
    if dft_bands_filtered.shape[1] <= wannier_bands.shape[1]:
        wannier_bands_filtered = wannier_bands[:, : dft_bands_filtered.shape[1]]
    else:
        wannier_bands_filtered = wannier_bands

    dft_bands_to_compare = dft_bands_filtered[:, : wannier_bands_filtered.shape[1]]

    return dft_bands_to_compare, wannier_bands_filtered


def _compute_distance(
    bands_energy_difference: np.array,
    bands_weight: np.array,
) -> tuple:
    """Return the weighted RMS and maximum band differences.

    :return: (bands_dist, max_dist, max_dist_2, max_dist_loc, max_dist_2_loc)
    """
    arr = bands_energy_difference**2 * bands_weight
    bands_dist = np.sqrt(np.sum(arr) / np.sum(bands_weight))

    # max distance
    max_dist = np.sqrt(np.max(arr))
    max_dist_loc = np.unravel_index(np.argmax(arr, axis=None), arr.shape)

    arr_2 = np.abs(bands_energy_difference) * bands_weight
    # max abs difference
    max_dist_2 = np.max(arr_2)
    max_dist_2_loc = np.unravel_index(np.argmax(arr_2, axis=None), arr_2.shape)

    return (bands_dist, max_dist, max_dist_2, max_dist_loc, max_dist_2_loc)


def bands_distance_raw(  # pylint: disable=too-many-arguments
    dft_bands: np.array,
    wannier_bands: np.array,
    mu: float,
    sigma: float,
    exclude_list_dft: list = None,
    lower_cutoff: float = None,
    gaussian_weight: bool = False,
) -> tuple:
    """Calculate bands distance with specified ``mu`` and ``sigma``.

    :param dft_bands: a numpy array of size (num_k x num_dft) where num_dft is
       number of bands computed by the DFT code. In eV.
    :param wannier_bands: a numpy array of size (num_k x num_wan) where num_wan is
       number of Wannier functions.  In eV.
    :param mu: in eV.
    :param sigma: in eV.
    :param exclude_list_dft: if passed should be a list of the excluded bands,
       1-indexed
    :param gaussian_weight: if True, gaussian weight will be used instead of
        Fermi-Dirac
    """
    dft_bands_to_compare, wannier_bands_filtered = _align_bands(
        dft_bands, wannier_bands, exclude_list_dft
    )

    weight_function = gaussian if gaussian_weight else fermi_dirac
    cutoff_mask = compute_lower_cutoff(dft_bands_to_compare, lower_cutoff)
    bands_weight_dft = weight_function(dft_bands_to_compare, mu, sigma) * cutoff_mask
    bands_weight_wannier = (
        weight_function(wannier_bands_filtered, mu, sigma) * cutoff_mask
    )
    bands_weight = np.sqrt(bands_weight_dft * bands_weight_wannier)

    bands_energy_difference = dft_bands_to_compare - wannier_bands_filtered

    return _compute_distance(bands_energy_difference, bands_weight)


def bands_distance(
    bands_dft: ty.Union[orm.BandsData, np.array],
    bands_wannier: ty.Union[orm.BandsData, np.array],
    fermi_energy: float,
    exclude_list_dft: list = None,
    gaussian_weight: bool = False,
) -> np.array:
    """Calculate bands distance with ``mu`` set as Ef to Ef+5.

    :param bands_dft: [description]
    :param bands_wannier: [description]
    :param fermi_energy: [description]
    :param exclude_list_dft: [description], defaults to None
    :return: [description], unit is eV.
    """
    dft_bands = as_bands_array(bands_dft)
    wannier_bands = as_bands_array(bands_wannier)

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
            gaussian_weight=gaussian_weight,
        )
        # mu, bands_distance, max_distance, max_distance_2
        dist[i, :] = [mu, res[0], res[1], res[2]]
        # for gaussian weight only dist[0] contains the result for mu = fermi_energy, other rows are nan
        # this prevents numpy RuntimeWarning warnings due to division by zero
        # when there are no bands close to the shifted fermi level
        if gaussian_weight:
            break

    return dist


def bands_distance_isolated(
    dft_bands: ty.Union[orm.BandsData, np.array],
    wannier_bands: ty.Union[orm.BandsData, np.array],
    exclude_list_dft: list = None,
    lower_cutoff: float = None,
) -> tuple:
    """Calculate bands distance with every band above ``lower_cutoff`` weighted equally.

    :param dft_bands: a numpy array of size (num_k x num_dft) where num_dft is
       number of bands computed by the DFT code. In eV.
    :param wannier_bands: a numpy array of size (num_k x num_wan) where num_wan is
       number of Wannier functions.  In eV.
    :param exclude_list_dft: if passed should be a list of the excluded bands,
       1-indexed
    :param lower_cutoff: bands below this energy, in eV, are dropped from the
       comparison. Defaults to None, meaning keep every band.
    """
    dft_bands_to_compare, wannier_bands_filtered = _align_bands(
        as_bands_array(dft_bands), as_bands_array(wannier_bands), exclude_list_dft
    )

    bands_energy_difference = dft_bands_to_compare - wannier_bands_filtered
    bands_weight = compute_lower_cutoff(dft_bands_to_compare, lower_cutoff)

    return _compute_distance(bands_energy_difference, bands_weight)


def bands_distance_fermi_dirac(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    bands_dft: ty.Union[orm.BandsData, np.array],
    bands_wannier: ty.Union[orm.BandsData, np.array],
    mu: float,
    sigma: float,
    exclude_list_dft: list = None,
    lower_cutoff: float = None,
) -> float:
    """Calculate the Fermi-Dirac-weighted RMS bands distance, in eV.

    Unlike :func:`bands_distance`, which scans ``mu`` from the Fermi energy to
    the Fermi energy plus 5 eV at a fixed ``sigma`` of 0.1 eV and returns one
    row per ``mu``, this returns a single number for the ``mu`` and ``sigma``
    asked for.

    :param bands_dft: bands computed by the DFT code, in eV.
    :param bands_wannier: Wannier-interpolated bands, in eV.
    :param mu: center of the Fermi-Dirac weight, in eV.
    :param sigma: broadening of the Fermi-Dirac weight, in eV. The weight falls
       from 1 to 0 over a few ``sigma`` around ``mu``, so a small value counts
       the occupied bands only and a large one approaches
       :func:`bands_distance_unweighted`.
    :param exclude_list_dft: if passed should be a list of the excluded bands,
       1-indexed
    :param lower_cutoff: bands below this energy, in eV, are dropped from the
       comparison. Defaults to None, meaning keep every band.
    """
    res = bands_distance_raw(
        dft_bands=as_bands_array(bands_dft),
        wannier_bands=as_bands_array(bands_wannier),
        mu=mu,
        sigma=sigma,
        exclude_list_dft=exclude_list_dft,
        lower_cutoff=lower_cutoff,
    )
    return float(res[0])


def bands_distance_unweighted(
    bands_dft: ty.Union[orm.BandsData, np.array],
    bands_wannier: ty.Union[orm.BandsData, np.array],
    exclude_list_dft: list = None,
    lower_cutoff: float = None,
) -> float:
    """Calculate the RMS bands distance with every band weighted equally, in eV.

    This is the ``sigma`` to infinity limit of :func:`bands_distance_fermi_dirac`:
    the empty bands count as much as the occupied ones.

    :param bands_dft: bands computed by the DFT code, in eV.
    :param bands_wannier: Wannier-interpolated bands, in eV.
    :param exclude_list_dft: if passed should be a list of the excluded bands,
       1-indexed
    :param lower_cutoff: bands below this energy, in eV, are dropped from the
       comparison. Defaults to None, meaning keep every band.
    """
    res = bands_distance_isolated(
        dft_bands=bands_dft,
        wannier_bands=bands_wannier,
        exclude_list_dft=exclude_list_dft,
        lower_cutoff=lower_cutoff,
    )
    return float(res[0])
