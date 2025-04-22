from upf_tools.projectors import Projector

import numpy as np
from scipy.optimize import curve_fit


def hydrogenic(r: np.ndarray, l: int, n: int, alfa: float) -> np.ndarray:
    """Get Hydrogenic radial functions

    :param r: Array of rmesh
    :param n: Num of semicore shells
    :param l: Angular momentum of orbials
    :param alfa: Shape parameter of orbitals
    :return: hydrogenic radial function
    """
    if l == 0:
        result = s(r, n, alfa)
    elif l == 1:
        result = p(r, n, alfa)
    elif l == 2:
        result = d(r, n, alfa)
    else:
        raise ValueError(
            "Only support s, p and d orbitals."
            "The l should be either 0 or 1 or 2"
        )
    result[np.abs(result)<1e-35] = 0.0
    return result

def r_hydrogenic(r: np.ndarray, l: int, n: int, alfa: float) -> np.ndarray:
    """Return r * hydrogenic radial function

    integral(r_hyd * r_hyd dr) = 1
    :param r: Array of rmesh
    :param n: Num of semicore shells
    :param l: Angular momentum of orbials
    :param alfa: Shape parameter of orbitals
    :return: hydrogenic radial function
    """
    return r * hydrogenic(r, l, n, alfa)

def rsquare_hydrogenic(r: np.ndarray, l: int, n: int, alfa: float) -> np.ndarray:
    """Return r^2 * hydrogenic radial function"""

    # The pw2wan in fact Fourier transform the r^2 * hydrogenic
    return r**2 * hydrogenic(r, l, n, alfa)

def s(r: np.ndarray, n: int, alfa: float) -> np.ndarray:
    """Hydrogenic radial function for s orbitals"""

    if n == 0:
        return 2 * alfa**(3/2) * np.exp(-alfa * r)
    elif n == 1:
        return 1/np.sqrt(8) * alfa**(3/2) * (2 - alfa*r) * np.exp(-alfa*r/2)
    else:
        raise ValueError("only support n = 0 or 1 for s orbitals")

def p(r: np.ndarray, n: int, alfa: float) -> np.ndarray:
    """Hydrogenic radial function for p orbitals"""

    if n == 0:
        return 1/np.sqrt(24) * alfa**(3/2) * alfa*r * np.exp(-alfa*r/2)
    elif n == 1:
        return 4/81/np.sqrt(6) * alfa**(3/2) *(6*alfa*r - alfa**2 * r**2) * np.exp(-alfa*r/3)
    else:
        raise ValueError("Only support n = 0 or 1 for p orbitals")
    
def d(r: np.ndarray, n: int, alfa: float) -> np.ndarray:
    """Hydrogenic radial function for d orbitals"""

    if n == 0:
        return 4/81/np.sqrt(30) * alfa**(3/2) *(alfa*r)**2 * np.exp(-alfa*r/3)
    else:
        raise ValueError("Only support n = 1 for d orbitals")
    
def fit_rsq_projector(proj: Projector, n: int) -> float:
    """Fit the hydrogenic radial function with given projectors
    
    :param proj: The given Projector contains r, l and radial function.
    :param n: Num of semicore shells.
    :return: alfa that fits the given projectors best.
    """

    r = proj.r
    y = proj.y
    l = proj.l
    # y = norm_upf(r, y)
    rsquare_radial_func = lambda r, alfa: rsquare_hydrogenic(r, l, n, alfa)
    try:
        pos_popt, pos_pcov = curve_fit(rsquare_radial_func, r, r**2 * y, p0=[3.0])
    except RuntimeError:
        pos_popt = [0.0]
        pos_pcov = [[100.0]]

    try:
        neg_popt, neg_pcov = curve_fit(rsquare_radial_func, r, r**2 * -y, p0=[3.0])
    except RuntimeError:
        neg_popt = [0.0]
        neg_pcov = [[100.0]]
    
    if pos_pcov[0][0] < neg_pcov[0][0]:
        popt = pos_popt[0]
    else:
        popt = neg_popt[0]

    return popt

def fit_upf_projector(proj: Projector, n: int) -> float:
    """Fit the hydrogenic radial function with upf projectors (r * radial func)
    
    :param proj: The given Projector contains r, l and radial function.
    :param n: Num of semicore shells.
    :return: alfa that fits the given projectors best.
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

    r_radial_func = lambda r, alfa: r_hydrogenic(r, l, n, alfa)
    try:
        pos_popt, pos_pcov = curve_fit(r_radial_func, r, y, p0=[3.0])
    except RuntimeError:
        pos_popt = [0.0]
        pos_pcov = [[100.0]]

    try:
        neg_popt, neg_pcov = curve_fit(r_radial_func, r, -y, p0=[3.0])
    except RuntimeError:
        neg_popt = [0.0]
        neg_pcov = [[100.0]]
    
    if pos_pcov[0][0] < neg_pcov[0][0]:
        popt = pos_popt[0]
    else:
        popt = neg_popt[0]

    return popt

def fit_projector(proj: Projector, n: int) -> float:
    """Fit the hydrogenic radial function with given projectors
    
    :param proj: The given Projector contains r, l and radial function.
    :param n: Num of semicore shells.
    :return: alfa that fits the given projectors best.
    """

    r = proj.r
    y = proj.y
    l = proj.l
    alfa_bounds = ([0.0, np.inf])
    radial_func = lambda r, alfa: hydrogenic(r, l, n, alfa)
    pos_popt, pos_pcov = curve_fit(radial_func, r, y, p0=[5.0], bounds=alfa_bounds)
    neg_popt, neg_pcov = curve_fit(radial_func, r, -y, p0=[5.0], bounds=alfa_bounds)
    
    if pos_pcov[0][0] < neg_pcov[0][0]:
        popt = pos_popt[0]
    else:
        popt = neg_popt[0]

    return popt

def fit_ortho_projectors(proj: Projector, n:int, element:str) -> float:
    import matplotlib.pyplot as plt
    r = proj.r
    y = proj.y # y = r * radial function
    l = proj.l
    dr = np.zeros_like(r)
    dr[1:] = r[1:] - r[:-1]
    dr[0] = r[0] - 0.0
    label = proj.label
    alpha = []
    overlap = []
    radial_func = lambda r, alfa: hydrogenic(r, l, n, alfa)
    for alfa in np.linspace(1.0, 10.0, 90, endpoint=False):
        alpha.append(alfa)
        ovlp = np.sum( # 4*pi* radial1 * radial2 * r^2 *dr
            4 * np.pi * y * radial_func(r, alfa) * r * dr
        )
        overlap.append(ovlp)

    alpha = np.array(alpha)
    ortho_loc = np.where(np.abs(overlap)==np.min(np.abs(overlap)))[0][0]
    alfa = alpha[ortho_loc]
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # ax.plot(alpha, overlap)
    # print(label, label[1], alfa)
    # ax.set_title(f"{element}: {label[1]} overlap, alfa={alfa}")
    # fig.savefig(f"{element}_{label[1]}.png")
    return alfa

    
def norm_upf(r, y):
    dr = np.zeros_like(r)
    dr[1:] = r[1:] - r[:-1]
    dr[0] = r[0] - 0.0

    inner_prod = np.sum(y**2 * dr)
    if (inner_prod - 1) > 0.05:
        y = np.sqrt(1.0/inner_prod) * y
    
    return y