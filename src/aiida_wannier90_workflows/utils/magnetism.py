"""Functions to define initial magnetic moments."""
import typing as ty

import numpy as np

from aiida import orm

__all__ = (
    "erfc_scdm",
    "fit_scdm_mu_sigma_raw",
    "fit_scdm_mu_sigma",
    "get_energy_of_projectability",
)

def get_moments(
    moments: ty.Union[
        ty.Dict[str, ty.Tuple[float]],
        ty.Dict[str, float]
    ],
    is_collinear: bool,
):
    """Get moments in a unified way.

    :param moments: moments to be parsed
    :type moments: ty.Union[ty.Dict[float], ty.Dict[str, ty.Tuple[float]]]
    :param is_collinear: whether the calculation is collinear
    :type is_collinear: bool
    :return: if collinear, return m_z components of moments.
             if non-collinear, return (moments, thetas, phis)
    """
    if is_collinear:
        starting_magnetization = {}
        for kind, mom in moments.items():
            if isinstance(mom, float):
                starting_magnetization[kind] = mom
            elif isinstance(mom, (tuple, list)) and len(mom) == 3:
                starting_magnetization[kind] = mom[2]
            else:
                raise TypeError(
                    f"moments should be float or tuple of length 3, got {mom}"
                )
        return starting_magnetization
    else:
        starting_magnetization = {}
        angle1 = {}
        angle2 = {}
        for kind, mom in moments.items():
            if isinstance(mom, float):
                starting_magnetization[kind] = mom
                angle1[kind] = 0.0
                angle2[kind] = 0.0
            elif isinstance(mom, (tuple, list)) and len(mom) == 3:
                mx, my, mz = mom
                m = np.linalg.norm(mom,ord=2)
                theta = np.arccos(mz/m)
                phi = np.arctan2(my,mx)
                starting_magnetization[kind] = m
                angle1[kind] = np.degrees(theta)
                angle2[kind] = np.degrees(phi)
            else:
                raise TypeError(
                    f"moments should be float or tuple of length 3, got {mom}"
                )
        return starting_magnetization, angle1, angle2

