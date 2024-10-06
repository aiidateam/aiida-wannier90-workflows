"""Functions for SCDM fitting."""

import typing as ty

import numpy as np

from aiida import orm

__all__ = (
    "erfc_scdm",
    "fit_scdm_mu_sigma_raw",
    "fit_scdm_mu_sigma",
    "get_energy_of_projectability",
)


def erfc_scdm(x, mu, sigma):
    """Error function."""
    from scipy.special import erfc  # pylint: disable=no-name-in-module

    return 0.5 * erfc((x - mu) / sigma)


def fit_erfc(f, xdata, ydata):  # pylint: disable=invalid-name
    """Fit error function."""
    from scipy.optimize import curve_fit

    return curve_fit(f, xdata, ydata, bounds=([-50, 0], [50, 50]))


def fit_scdm_mu_sigma_raw(
    bands: np.array,
    projections: np.array,
    sigma_factor: float = 3.0,
    return_data: bool = False,
) -> ty.Union[ty.Tuple[float, float], ty.Tuple[float, float, np.array]]:
    """Fit mu parameter for the SCDM-k method.

    The projectability of all orbitals is fitted using an erfc(x) function.
    Mu and sigma are extracted from the fitted distribution,
    with mu = mu_fit - k * sigma, sigma = sigma_fit and
    k a parameter with default k = 3.

    This function accepts numpy array inputs, the function `fit_scdm_mu_sigma`
    is the AiiDA wrapper which accepts AiiDA type as input parameters.

    :param bands: output of projwfc, it was computed in the nscf calc
    :param projections: output of projwfc
    :param sigma_factor: scdm_mu will be set to::
        scdm_mu = E(projectability==max_projectability) - sigma_factor * scdm_sigma
        Pass sigma_factor = 0 if you do not want to shift
    :return: scdm_mu, scdm_sigma,
        optional data (shape 2 * N, 0th row energy, 1st row projectability)
    """
    sorted_bands, sorted_projwfc = sort_projectability_arrays(bands, projections)

    popt, pcov = fit_erfc(  # pylint: disable=unbalanced-tuple-unpacking,unused-variable
        erfc_scdm, sorted_bands, sorted_projwfc
    )
    mu = popt[0]
    sigma = popt[1]

    scdm_sigma = sigma
    scdm_mu = mu - sigma * sigma_factor

    if return_data:
        data = np.zeros((2, len(sorted_bands)))
        data[0, :] = sorted_bands
        data[1, :] = sorted_projwfc
        return scdm_mu, scdm_sigma, data

    return scdm_mu, scdm_sigma


def fit_scdm_mu_sigma(
    bands: orm.BandsData,
    projections: orm.ProjectionData,
    sigma_factor: orm.Float,
    return_data: bool = False,
) -> ty.Union[ty.Tuple[float, float], ty.Tuple[float, float, np.array]]:
    """Fit scdm_mu & scdm_sigma based on projectability.

    This is the AiiDA wrapper of `fit_scdm_mu_sigma_raw`.

    :param pw2wan_parameters: pw2wannier90 input parameters (the one to update with this calcfunction)
    :param bands: band structure of the projwfc output
    :param projections: projectability of the projwfc output
    :param sigma_factor: sigma_factor of SCDM
    """
    bands_array, projections_array = get_projectability_arrays(bands, projections)
    return fit_scdm_mu_sigma_raw(
        bands_array, projections_array, sigma_factor.value, return_data
    )


def get_projectability_arrays(bands: orm.BandsData, projections: orm.ProjectionData):
    """Calculate projectability array.

    Accept aiida orm class, return numpy arrays:
        (bands_array, projections_array), where each array has shape (num_kpt, num_bands)

    :param bands: [description]
    :type bands: orm.BandsData
    :param projections: [description]
    :type projections: orm.ProjectionData
    """
    # List of specifications of atomic orbitals in dictionary form
    orbitals_list = [i.get_orbital_dict() for i in projections.get_orbitals()]
    # Remove the '_orbital_type' key from the dictionaries, otherwise the get_projections fail
    remove_key = "_orbital_type"
    for o in orbitals_list:
        if remove_key in o:
            o.pop(remove_key)
    # Sum of the projections on all atomic orbitals, shape num_kpoints * num_bands
    projections_array = sum(
        sum(x[1] for x in projections.get_projections(**orb_dict))
        for orb_dict in orbitals_list
    )
    # shape num_kpoints * num_bands, TODO support spin
    bands_array = bands.get_bands()
    return bands_array, projections_array


def sort_projectability_arrays(bands: np.array, projections: np.array):
    """Sort projectability arrays by energy in ascending order.

    :param bands: output of projwfc, it was computed in the nscf calc
    :param projections: output of projwfc
    """
    # Flattening (projection modulus squared according to QE, energies)
    projwfc_flat = projections.flatten()
    bands_flat = bands.flatten()

    # sort by energy
    # sorted_bands, sorted_projwfc = zip(*sorted(zip(bands_flat, projwfc_flat)))
    # use numpy, faster
    data = np.vstack((bands_flat, projwfc_flat))  # shape 2 * N
    ind = np.argsort(data, axis=1)[0, :]  # sort by energy
    data = data[:, ind]
    sorted_bands = data[0, :]
    sorted_projwfc = data[1, :]
    return sorted_bands, sorted_projwfc


def get_energy_of_projectability(
    bands: orm.BandsData, projections: orm.ProjectionData, thresholds: float = 0.9
):
    """Return energy corresponds to projectability = thresholds.

    :param bands: [description]
    :param projections: [description]
    :param thresholds: [description]
    """
    bands_array, projections_array = get_projectability_arrays(bands, projections)
    sorted_bands, sorted_projwfc = sort_projectability_arrays(
        bands_array, projections_array
    )
    max_ind = np.max(np.argwhere(sorted_projwfc >= thresholds).flatten())
    return sorted_bands[max_ind]
