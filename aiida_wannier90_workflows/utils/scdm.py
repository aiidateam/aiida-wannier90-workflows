import typing
import numpy as np
from scipy.special import erfc
from scipy.optimize import curve_fit
from aiida import orm

__all__ = ('erfc_scdm', 'fit_scdm_mu_sigma', 'fit_scdm_mu_sigma_aiida')

def erfc_scdm(x, mu, sigma):
    return 0.5 * erfc((x - mu) / sigma)

def fit_erfc(f, xdata, ydata):
    return curve_fit(f, xdata, ydata, bounds=([-50, 0], [50, 50]))

def fit_scdm_mu_sigma(bands: np.array, projections: np.array, thresholds: dict = {'sigma_factor', 3}, return_data: bool = False) -> typing.Union[typing.Tuple[float, float], typing.Tuple[float, float, np.array]]:
    '''Fit mu parameter for the SCDM-k method:
    The projectability of all orbitals is fitted using an erfc(x) function. 
    Mu and sigma are extracted from the fitted distribution,
    with mu = mu_fit - k * sigma, sigma = sigma_fit and
    k a parameter with default k = 3.

    This function accepts numpy array inputs, the function `fit_scdm_mu_sigma_aiida` 
    is the AiiDA wrapper which accepts AiiDA type as input parameters.

    :param bands: output of projwfc, it was computed in the nscf calc
    :param projections: output of projwfc
    :param thresholds: must contain 'sigma_factor'; scdm_mu will be set to::
        scdm_mu = E(projectability==max_projectability) - sigma_factor * scdm_sigma
        Pass sigma_factor = 0 if you do not want to shift
    :return: scdm_mu, scdm_sigma,
        optional data (shape 2 * N, 0th row energy, 1st row projectability)'''
    # Flattening (projection modulus squared according to QE, energies)
    projwfc_flat = projections.flatten()
    bands_flat = bands.flatten()

    # sort by energy
    #sorted_bands, sorted_projwfc = zip(*sorted(zip(bands_flat, projwfc_flat)))
    # use numpy, faster
    data = np.vstack((bands_flat, projwfc_flat)) # shape 2 * N
    ind = np.argsort(data, axis=1)[0, :] # sort by energy
    data = data[:, ind]
    sorted_bands = data[0, :]
    sorted_projwfc = data[1, :]

    popt, pcov = fit_erfc(erfc_scdm, sorted_bands, sorted_projwfc)
    mu = popt[0]
    sigma = popt[1]
    # TODO maybe check the quality of the fitting

    scdm_sigma = sigma
    sigma_factor = thresholds.get('sigma_factor', None)
    if sigma_factor is None:
        raise ValueError(f'no sigma_factor in input thresholds {thresholds}')
    scdm_mu = mu - sigma * sigma_factor

    if return_data:
        return scdm_mu, scdm_sigma, data
    else:
        return scdm_mu, scdm_sigma

def fit_scdm_mu_sigma_aiida(bands: orm.BandsData, projections: orm.ProjectionData, thresholds: dict, return_data: bool = False) -> typing.Union[typing.Tuple[float, float], typing.Tuple[float, float, np.array]]:
    """Fit scdm_mu & scdm_sigma based on projectability.
    This is the AiiDA wrapper of `fit_scdm_mu_sigma`.

    :param pw2wan_parameters: pw2wannier90 input parameters (the one to update with this calcfunction)
    :type pw2wan_parameters: orm.Dict
    :param bands: band structure of the projwfc output
    :type bands: orm.BandsData
    :param projections: projectability of the projwfc output
    :type projections: orm.ProjectionData
    :param thresholds: thresholds of SCDM
    :type thresholds: dict"""
    # List of specifications of atomic orbitals in dictionary form
    orbitals_list = [i.get_orbital_dict() for i in projections.get_orbitals()]
    # Sum of the projections on all atomic orbitals, shape num_kpoints * num_bands
    projections_array = sum([sum([x[1] for x in projections.get_projections(**orb_dict)]) 
                    for orb_dict in orbitals_list])
    # shape num_kpoints * num_bands, TODO support spin
    bands_array = bands.get_bands()

    return fit_scdm_mu_sigma(bands_array, projections_array, thresholds, return_data)