"""Hydrogenic radial functions related modules."""

import numpy as np
from scipy.optimize import curve_fit
from upf_tools.projectors import Projector


def hydrogenic(r: np.ndarray, l: int, n: int, alpha: float) -> np.ndarray:
    """Get Hydrogenic radial functions.

    :param r: Array of rmesh
    :param n: Num of semicore shells
    :param l: Angular momentum of orbials
    :param alpha: Shape parameter of orbitals
    :return: hydrogenic radial function
    """
    if l == 0:
        result = s(r, n, alpha)
    elif l == 1:
        result = p(r, n, alpha)
    elif l == 2:
        result = d(r, n, alpha)
    else:
        raise ValueError(
            "Only support s, p and d orbitals. The l should be either 0 or 1 or 2"
        )
    result[np.abs(result) < 1e-35] = 0.0
    return result


def r_hydrogenic(r: np.ndarray, l: int, n: int, alpha: float) -> np.ndarray:
    """Return r * hydrogenic radial function.

    integral(r_hyd * r_hyd dr) = 1
    :param r: Array of rmesh
    :param n: Num of semicore shells
    :param l: Angular momentum of orbials
    :param alpha: Shape parameter of orbitals
    :return: hydrogenic radial function
    """
    return r * hydrogenic(r, l, n, alpha)


def rsquare_hydrogenic(r: np.ndarray, l: int, n: int, alpha: float) -> np.ndarray:
    """Return r^2 * hydrogenic radial function."""

    # The pw2wan in fact Fourier transform the r^2 * hydrogenic
    return r**2 * hydrogenic(r, l, n, alpha)


def s(r: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Hydrogenic radial function for s orbitals."""

    if n == 0:
        result = 2 * alpha ** (3 / 2) * np.exp(-alpha * r)
    elif n == 1:
        result = (
            1 / np.sqrt(8) * alpha ** (3 / 2) * (2 - alpha * r) * np.exp(-alpha * r / 2)
        )
    else:
        raise ValueError("only support n = 0 or 1 for s orbitals")

    return result


def p(r: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Hydrogenic radial function for p orbitals."""

    if n == 0:
        result = 1 / np.sqrt(24) * alpha ** (3 / 2) * alpha * r * np.exp(-alpha * r / 2)
    elif n == 1:
        result = (
            4
            / 81
            / np.sqrt(6)
            * alpha ** (3 / 2)
            * (6 * alpha * r - alpha**2 * r**2)
            * np.exp(-alpha * r / 3)
        )
    else:
        raise ValueError("Only support n = 0 or 1 for p orbitals")

    return result


def d(r: np.ndarray, n: int, alpha: float) -> np.ndarray:
    """Hydrogenic radial function for d orbitals."""

    if n == 0:
        result = (
            4
            / 81
            / np.sqrt(30)
            * alpha ** (3 / 2)
            * (alpha * r) ** 2
            * np.exp(-alpha * r / 3)
        )
    else:
        raise ValueError("Only support n = 1 for d orbitals")

    return result


def fit_rsq_projector(proj: Projector, n: int) -> float:
    """Fit the hydrogenic radial function with given projectors.

    :param proj: The given Projector contains r, l and radial function.
    :param n: Num of semicore shells.
    :return: alpha that fits the given projectors best.
    """

    r = proj.r
    y = proj.y
    l = proj.l
    # y = norm_upf(r, y)
    rsquare_radial_func = lambda r, alpha: rsquare_hydrogenic(
        r, l, n, alpha
    )  # pylint:disable=unnecessary-lambda-assignment
    try:
        pos_popt, pos_pcov = curve_fit(rsquare_radial_func, r, r**2 * y, p0=[3.0])[0:2]
    except RuntimeError:
        pos_popt = [0.0]
        pos_pcov = [[100.0]]

    try:
        neg_popt, neg_pcov = curve_fit(rsquare_radial_func, r, r**2 * -y, p0=[3.0])[0:2]
    except RuntimeError:
        neg_popt = [0.0]
        neg_pcov = [[100.0]]

    if pos_pcov[0][0] < neg_pcov[0][0]:
        popt = pos_popt[0]
    else:
        popt = neg_popt[0]

    return popt


def fit_upf_projector(proj: Projector, n: int) -> float:
    """Fit the hydrogenic radial function with upf projectors (r * radial func).

    :param proj: The given Projector contains r, l and radial function.
    :param n: Num of semicore shells.
    :return: alpha that fits the given projectors best.
    """

    r = proj.r
    y = proj.y
    l = proj.l

    y = norm_upf(r, y)
    # cut tail of radial functions
    for idx in range(len(r)):
        if np.all(np.abs(y[idx:]) < 1e-2):
            y = y[:idx]
            r = r[:idx]
            break

    r_radial_func = lambda r, alpha: r_hydrogenic(
        r, l, n, alpha
    )  # pylint:disable=unnecessary-lambda-assignment
    try:
        pos_popt, pos_pcov = curve_fit(r_radial_func, r, y, p0=[3.0])[0:2]
    except RuntimeError:
        pos_popt = [0.0]
        pos_pcov = [[100.0]]

    try:
        neg_popt, neg_pcov = curve_fit(r_radial_func, r, -y, p0=[3.0])[0:2]
    except RuntimeError:
        neg_popt = [0.0]
        neg_pcov = [[100.0]]

    if pos_pcov[0][0] < neg_pcov[0][0]:
        popt = pos_popt[0]
    else:
        popt = neg_popt[0]

    return popt


def fit_projector(proj: Projector, n: int) -> float:
    """Fit the hydrogenic radial function with given projectors.

    :param proj: The given Projector contains r, l and radial function
    :param n: Num of semicore shells.
    :return: alpha that fits the given projectors best.
    """

    r = proj.r
    y = proj.y
    l = proj.l
    alpha_bounds = [0.0, np.inf]
    radial_func = lambda r, alpha: hydrogenic(
        r, l, n, alpha
    )  # pylint:disable=unnecessary-lambda-assignment
    pos_popt, pos_pcov = curve_fit(radial_func, r, y, p0=[5.0], bounds=alpha_bounds)[
        0:2
    ]
    neg_popt, neg_pcov = curve_fit(radial_func, r, -y, p0=[5.0], bounds=alpha_bounds)[
        0:2
    ]

    if pos_pcov[0][0] < neg_pcov[0][0]:
        popt = pos_popt[0]
    else:
        popt = neg_popt[0]

    return popt


def fit_ortho_projectors(proj: Projector, n: int) -> float:
    """Fit the hydrogenic radial function orthonormalization to given projectors.

    :param proj: The given Projector contains r, l and radial function
    :param n: Num of semicore shells.
    :return: alpha that fits the given projectors best.
    """
    # import matplotlib.pyplot as plt

    r = proj.r
    y = proj.y  # y = r * radial function
    l = proj.l
    dr = np.zeros_like(r)
    dr[1:] = r[1:] - r[:-1]
    dr[0] = r[0] - 0.0
    alpha_list = []
    overlap = []
    radial_func = lambda r, alpha: hydrogenic(
        r, l, n, alpha
    )  # pylint:disable=unnecessary-lambda-assignment
    for alpha in np.linspace(1.0, 10.0, 90, endpoint=False):
        alpha_list.append(alpha)
        ovlp = np.sum(  # 4*pi* radial1 * radial2 * r^2 *dr
            4 * np.pi * y * radial_func(r, alpha) * r * dr
        )
        overlap.append(ovlp)

    alpha_list = np.array(alpha_list)
    ortho_loc = np.where(np.abs(overlap) == np.min(np.abs(overlap)))[0][0]
    alpha = alpha_list[ortho_loc]
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(alpha, overlap)
    # print(label, label[1], alpha)
    # ax.set_title(f"{element}: {label[1]} overlap, alpha={alpha}")
    # fig.savefig(f"{element}_{label[1]}.png")
    return alpha


def norm_upf(r, y):
    """Normalize hydrogenic radial functions."""

    dr = np.zeros_like(r)
    dr[1:] = r[1:] - r[:-1]
    dr[0] = r[0] - 0.0

    inner_prod = np.sum(y**2 * dr)
    if (inner_prod - 1) > 0.05:
        y = np.sqrt(1.0 / inner_prod) * y

    return y
