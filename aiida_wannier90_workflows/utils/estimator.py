#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Storage estimators for files related to wannier90."""
import typing as ty
from dataclasses import dataclass
from collections import namedtuple
import numpy as np
# import numpy.typing as npt
from aiida import orm
from aiida_wannier90_workflows.common.types import WannierFileFormat

# pylint: disable=too-many-lines

# Storage size of various Fortran types, all unit in bytes
# Fortran default integer
_SIZE_INTEGER = 4
# Fortran double precision real = 16 bytes
_SIZE_REAL_DP = 8
# Fortran double precision complex = 16 bytes
_SIZE_COMPLEX_DP = _SIZE_REAL_DP * 2
# Fortran logical
_SIZE_LOGICAL = 4
# Fortran adds 4 bytes to the leading and 4 bytes to the trailing of each record
_SIZE_RECORD_OVERHEAD = 8
# Fortran default character, ASCII: one char = 8 bits = 1 byte
_SIZE_CHARACTER = 1
# Some times Fortran prepend a space in the 0-th column
_SIZE_CARRIAGE_CONTROL = 1
# At the end there is a '\n', line feed = 1 character = 1 byte
_SIZE_LINE_FEED = 1


def estimate_amn(
    num_bands: int,
    num_wann: int,
    num_kpts: int,
    file_format: WannierFileFormat = WannierFileFormat.FORTRAN_FORMATTED
) -> int:
    """Estimate file size of seedname.amn.

    Tested with gfortran, Intel oneAPI 2021.4.0
    :param num_bands: number of bands
    :type num_bands: int
    :param num_wann: number of projections
    :type num_wann: int
    :param num_kpts: number of kpoints
    :type num_kpts: int
    :param file_format: type of the file format, defaults to WannierFileFormat.FORTRAN_FORMATTED
    :type file_format: WannierFileFormat, optional
    :return: file size in bytes
    :rtype: int
    """
    if file_format == WannierFileFormat.FORTRAN_FORMATTED:
        # header
        total_size = _SIZE_CARRIAGE_CONTROL + _SIZE_CHARACTER * 60 + _SIZE_LINE_FEED
        # 2nd line
        total_size += 12 * 3 + _SIZE_LINE_FEED
        # data
        total_size += (5 * 3 + 18 * 2 + _SIZE_LINE_FEED) * num_bands * num_wann * num_kpts
        return total_size

    if file_format == WannierFileFormat.FORTRAN_UNFORMATTED:
        # header
        total_size = _SIZE_CHARACTER * 60 + _SIZE_RECORD_OVERHEAD
        # 2nd line
        total_size += _SIZE_INTEGER * 3 + _SIZE_RECORD_OVERHEAD
        # data
        total_size += (_SIZE_INTEGER * 3 + _SIZE_COMPLEX_DP + _SIZE_RECORD_OVERHEAD) * num_bands * num_wann * num_kpts
        return total_size

    raise ValueError(f'Not supported type {file_format}')


def estimate_mmn(
    num_bands: int,
    num_kpts: int,
    nntot: int,
    file_format: WannierFileFormat = WannierFileFormat.FORTRAN_FORMATTED
) -> int:
    """Estimate file size of seedname.mmn.

    Tested with gfortran, Intel oneAPI 2021.4.0
    :param num_bands: number of bands
    :type num_bands: int
    :param num_kpts: number of kpoints
    :type num_kpts: int
    :param nntot: number of bvectors
    :type nntot: int
    :param file_format: type of the file format, defaults to WannierFileFormat.FORTRAN_FORMATTED
    :type file_format: WannierFileFormat, optional
    :return: file size in bytes
    :rtype: int
    """
    if file_format == WannierFileFormat.FORTRAN_FORMATTED:
        # header
        total_size = _SIZE_CARRIAGE_CONTROL + _SIZE_CHARACTER * 60 + _SIZE_LINE_FEED
        # 2nd line
        total_size += 12 * 3 + _SIZE_LINE_FEED
        # data
        # line for ik, ikp, (g_kpb(ipol,ik,ib), ipol=1,3)
        total_size += (5 * 5 + _SIZE_LINE_FEED) * num_kpts * nntot
        # line for MMN
        total_size += (18 * 2 + _SIZE_LINE_FEED) * num_bands**2 * num_kpts * nntot
        return total_size

    if file_format == WannierFileFormat.FORTRAN_UNFORMATTED:
        # header
        total_size = _SIZE_CHARACTER * 60 + _SIZE_RECORD_OVERHEAD
        # 2nd line
        total_size += _SIZE_INTEGER * 3 + _SIZE_RECORD_OVERHEAD
        # record for ik, ikp, (g_kpb(ipol,ik,ib), ipol=1,3)
        total_size += (_SIZE_INTEGER * 5 + _SIZE_RECORD_OVERHEAD) * num_kpts * nntot
        # record for MMN
        total_size += (_SIZE_COMPLEX_DP + _SIZE_RECORD_OVERHEAD) * num_bands**2 * num_kpts * nntot
        return total_size

    raise ValueError(f'Not supported type {file_format}')


def estimate_eig(
    num_bands: int, num_kpts: int, file_format: WannierFileFormat = WannierFileFormat.FORTRAN_FORMATTED
) -> int:
    """Estimate file size of seedname.mmn.

    Tested with gfortran, Intel oneAPI 2021.4.0
    :param num_bands: number of bands
    :type num_bands: int
    :param num_kpts: number of kpoints
    :type num_kpts: int
    :param file_format: type of the file format, defaults to WannierFileFormat.FORTRAN_FORMATTED
    :type file_format: WannierFileFormat, optional
    :return: file size in bytes
    :rtype: int
    """
    if file_format == WannierFileFormat.FORTRAN_FORMATTED:
        # data
        total_size = (5 * 2 + 18 + _SIZE_LINE_FEED) * num_bands * num_kpts
        return total_size

    if file_format == WannierFileFormat.FORTRAN_UNFORMATTED:
        total_size = (_SIZE_INTEGER * 2 + _SIZE_REAL_DP + _SIZE_RECORD_OVERHEAD) * num_bands * num_kpts
        return total_size

    raise ValueError(f'Not supported type {file_format}')


def estimate_unk(
    nr1: int,
    nr2: int,
    nr3: int,
    num_bands: int,
    num_kpts: int,
    reduce_unk: bool,
    file_format: WannierFileFormat = WannierFileFormat.FORTRAN_FORMATTED
) -> int:
    """Estimate total file size of all the UNK* files.

    Tested with gfortran, Intel oneAPI 2021.4.0
    :param num_bands: number of bands
    :type num_bands: int
    :param num_kpts: number of kpoints
    :type num_kpts: int
    :param file_format: type of the file format, defaults to WannierFileFormat.FORTRAN_FORMATTED
    :type file_format: WannierFileFormat, optional
    :return: file size in bytes
    :rtype: int
    """
    if reduce_unk:
        nr1 = int((nr1 + 1) / 2)
        nr2 = int((nr2 + 1) / 2)

    if file_format == WannierFileFormat.FORTRAN_FORMATTED:
        # header
        total_size = (12 * 5 + _SIZE_LINE_FEED) * num_kpts
        # data
        total_size += (20 * 2 + _SIZE_LINE_FEED) * nr1 * nr2 * nr3 * num_bands * num_kpts
        return total_size

    if file_format == WannierFileFormat.FORTRAN_UNFORMATTED:
        # header
        total_size = (_SIZE_INTEGER * 5 + _SIZE_RECORD_OVERHEAD) * num_kpts
        # data
        total_size += (_SIZE_COMPLEX_DP * nr1 * nr2 * nr3 + _SIZE_RECORD_OVERHEAD) * num_bands * num_kpts
        return total_size

    raise ValueError(f'Not supported type {file_format}')


@dataclass(order=True)
class WannierChk:  # pylint: disable=too-many-instance-attributes
    """Class storing the content of a seedname.chk file."""

    header: str
    num_bands: int
    num_exclude_bands: int
    exclude_bands: ty.List[int]
    real_lattice: np.array  # npt.ArrayLike
    recip_lattice: np.array
    num_kpts: int
    mp_grid: ty.List[int]
    kpt_latt: np.array
    nntot: int
    num_wann: int
    checkpoint: str
    have_disentangled: bool
    omega_invariant: float
    lwindow: np.array
    ndimwin: np.array
    u_matrix_opt: np.array
    u_matrix: np.array
    m_matrix: np.array
    wannier_centres: np.array
    wannier_spreads: np.array


def get_number_of_digits(val: int, /) -> int:
    """Get number of digits of an integer.

    E.g., 1 -> 1, 10 -> 2, 100 -> 3

    :param val: [description]
    :type val: int
    :return: [description]
    :rtype: [type]
    """
    return len(str(val))


def estimate_chk(
    num_exclude_bands: int,
    num_bands: int,
    num_wann: int,
    num_kpts: int,
    nntot: int,
    have_disentangled: bool,
    file_format: WannierFileFormat = WannierFileFormat.FORTRAN_FORMATTED
) -> int:
    """Estimate total file size of all the UNK* files.

    Tested with gfortran, Intel oneAPI 2021.4.0
    :param num_bands: number of bands
    :type num_bands: int
    :param num_kpts: number of kpoints
    :type num_kpts: int
    :param file_format: type of the file format, defaults to WannierFileFormat.FORTRAN_FORMATTED
    :type file_format: WannierFileFormat, optional
    :return: file size in bytes
    :rtype: int
    """
    if file_format == WannierFileFormat.FORTRAN_FORMATTED:
        # header
        total_size = _SIZE_CHARACTER * 33 + _SIZE_LINE_FEED
        # num_bands
        total_size += _SIZE_CHARACTER * get_number_of_digits(num_bands) + _SIZE_LINE_FEED
        # num_exclude_bands
        total_size += _SIZE_CHARACTER * get_number_of_digits(num_bands) + _SIZE_LINE_FEED
        # exclude_bands(num_exclude_bands), this is not exact, to be accurate I nned the exclude_bands
        total_size += (_SIZE_CHARACTER + _SIZE_LINE_FEED) * num_exclude_bands
        # real_lattice
        total_size += 25 * 3 * 3 + _SIZE_LINE_FEED
        # recip_lattice
        total_size += 25 * 3 * 3 + _SIZE_LINE_FEED
        # num_kpts
        total_size += _SIZE_CHARACTER * get_number_of_digits(num_kpts) + _SIZE_LINE_FEED
        # mp_grid, this is not exact
        total_size += _SIZE_CHARACTER * 3 + _SIZE_LINE_FEED
        # kpt_latt
        total_size += (25 * 3 + _SIZE_LINE_FEED) * num_kpts
        # nntot
        total_size += _SIZE_CHARACTER * get_number_of_digits(nntot) + _SIZE_LINE_FEED
        # num_wann
        total_size += _SIZE_CHARACTER * get_number_of_digits(num_wann) + _SIZE_LINE_FEED
        # checkpoint
        total_size += _SIZE_CHARACTER * 20 + _SIZE_LINE_FEED
        # have_disentangled
        total_size += 1 + _SIZE_LINE_FEED
        if have_disentangled:
            # omega_invariant
            total_size += 25 + _SIZE_LINE_FEED
            # lwindow
            total_size += (1 + _SIZE_LINE_FEED) * num_bands * num_kpts
            # ndimwin, this is not exact
            total_size += (1 + _SIZE_LINE_FEED) * num_kpts
            # u_matrix_opt
            total_size += (25 * 2 + _SIZE_LINE_FEED) * num_bands * num_wann * num_kpts
        # u_matrix
        total_size += (25 * 2 + _SIZE_LINE_FEED) * num_wann * num_wann * num_kpts
        # m_matrix
        total_size += (25 * 2 + _SIZE_LINE_FEED) * num_wann * num_wann * nntot * num_kpts
        # wannier_centres
        total_size += (25 * 3 + _SIZE_LINE_FEED) * num_wann
        # wannier_spreads
        total_size += (_SIZE_REAL_DP + _SIZE_LINE_FEED) * num_wann
        return total_size

    if file_format == WannierFileFormat.FORTRAN_UNFORMATTED:
        # header
        total_size = _SIZE_CHARACTER * 33 + _SIZE_RECORD_OVERHEAD
        # num_bands
        total_size += _SIZE_INTEGER + _SIZE_RECORD_OVERHEAD
        # num_exclude_bands
        total_size += _SIZE_INTEGER + _SIZE_RECORD_OVERHEAD
        # exclude_bands(num_exclude_bands)
        total_size += _SIZE_INTEGER * num_exclude_bands + _SIZE_RECORD_OVERHEAD
        # real_lattice
        total_size += _SIZE_REAL_DP * 3 * 3 + _SIZE_RECORD_OVERHEAD
        # recip_lattice
        total_size += _SIZE_REAL_DP * 3 * 3 + _SIZE_RECORD_OVERHEAD
        # num_kpts
        total_size += _SIZE_INTEGER + _SIZE_RECORD_OVERHEAD
        # mp_grid
        total_size += _SIZE_INTEGER * 3 + _SIZE_RECORD_OVERHEAD
        # kpt_latt
        total_size += _SIZE_REAL_DP * 3 * num_kpts + _SIZE_RECORD_OVERHEAD
        # nntot
        total_size += _SIZE_INTEGER + _SIZE_RECORD_OVERHEAD
        # num_wann
        total_size += _SIZE_INTEGER + _SIZE_RECORD_OVERHEAD
        # checkpoint
        total_size += _SIZE_CHARACTER * 20 + _SIZE_RECORD_OVERHEAD
        # have_disentangled
        total_size += _SIZE_LOGICAL + _SIZE_RECORD_OVERHEAD
        if have_disentangled:
            # omega_invariant
            total_size += _SIZE_REAL_DP + _SIZE_RECORD_OVERHEAD
            # lwindow
            total_size += _SIZE_LOGICAL * num_bands * num_kpts + _SIZE_RECORD_OVERHEAD
            # ndimwin
            total_size += _SIZE_INTEGER * num_kpts + _SIZE_RECORD_OVERHEAD
            # u_matrix_opt
            total_size += _SIZE_COMPLEX_DP * num_bands * num_wann * num_kpts + _SIZE_RECORD_OVERHEAD
        # u_matrix
        total_size += _SIZE_COMPLEX_DP * num_wann * num_wann * num_kpts + _SIZE_RECORD_OVERHEAD
        # m_matrix
        total_size += _SIZE_COMPLEX_DP * num_wann * num_wann * nntot * num_kpts + _SIZE_RECORD_OVERHEAD
        # wannier_centres
        total_size += _SIZE_REAL_DP * 3 * num_wann + _SIZE_RECORD_OVERHEAD
        # wannier_spreads
        total_size += _SIZE_REAL_DP * num_wann + _SIZE_RECORD_OVERHEAD
        return total_size

    raise ValueError(f'Not supported type {file_format}')


# This is not working
# def get_nrpts(cell: np.array, mp_grid: ty.Sequence[int]) -> int:
#     from aiida_wannier90_workflows.utils.center import generate_supercell, get_wigner_seitz
#     from shapely.geometry import Point, Polygon

#     # Wannier90 default
#     ws_search_size = 2
#     ws_distance_tol = 1.e-5

#     # Get the WS cell of the supercell
#     supercell = cell * np.array(mp_grid)

#     ws_supercell = get_wigner_seitz(supercell, ws_search_size)

#     # All the R points
#     all_points, _ = generate_supercell(cell, ws_search_size*np.array(mp_grid))

#     polygon = Polygon(ws_supercell)
#     print(polygon.contains(Point(0,0,0)))
#     print(all_points, all_points.shape)
#     contained_points = list(filter(lambda _: polygon.distance(Point(_)) <= ws_distance_tol, all_points))
#     nrpts = len(contained_points)

#     return nrpts


def get_nrpts_ndegen(cell: np.array, mp_grid: ty.Sequence[int]) -> int:
    """Calculate number of R points in Fourier transform.

    :param cell: [description]
    :type cell: np.array
    :param mp_grid: [description]
    :type mp_grid: ty.Sequence[int]
    :return: [description]
    :rtype: int
    """
    from aiida_wannier90_workflows.utils.center import generate_supercell

    # Wannier90 default
    ws_search_size = 2
    ws_distance_tol = 1.e-5

    # Get the WS cell of the supercell
    supercell = cell * np.array(mp_grid)

    supercell_points, _ = generate_supercell(supercell, ws_search_size + 1)

    all_points, _ = generate_supercell(cell, ws_search_size * np.array(mp_grid))

    dist = all_points[:, :, np.newaxis] - supercell_points.T[np.newaxis, :, :]
    dist = np.linalg.norm(dist, axis=1)

    # R = (0, 0, 0) <-> index = supercell_points.shape[0] // 2, (shape[0] is always odd)
    idx_r0 = supercell_points.shape[0] // 2
    # print(supercell_points[idx_R0, :])

    dist_min = np.min(dist, axis=1)
    is_in_wscell = np.abs(dist[:, idx_r0] - dist_min) < ws_distance_tol
    # Count number of True
    nrpts = np.sum(is_in_wscell)

    is_degen = np.abs(dist[is_in_wscell, :] - dist_min[is_in_wscell, np.newaxis]) < ws_distance_tol
    ndegen = np.sum(is_degen, axis=1)

    return nrpts, ndegen


def test_get_nrpts_ndegen():
    """Test nrpts."""

    # cell_angle = 20
    # cell = np.array([[1, 0, 0],
    # [np.cos(cell_angle / 180 * np.pi), np.sin(cell_angle / 180 * np.pi), 0],
    # [0, 0, 1]])
    # mp_grid = [2, 3, 4]

    # cell = np.eye(3)
    cell = np.array([[-2.69880000000000, 0.000000000000000E+000, 2.69880000000000],
                     [0.000000000000000E+000, 2.69880000000000, 2.69880000000000],
                     [-2.69880000000000, 2.69880000000000, 0.000000000000000E+00]])
    mp_grid = [4, 4, 4]

    nrpts, ndegen = get_nrpts_ndegen(cell, mp_grid)

    nrpts_ref = 93
    ndegen_ref = [
        4, 6, 2, 2, 2, 1, 2, 2, 1, 1, 2, 6, 2, 2, 2, 6, 2, 2, 4, 1, 1, 1, 4, 1, 1, 1, 1, 2, 1, 1, 1, 2, 2, 1, 1, 2, 4,
        2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 2, 4, 2, 1, 1, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1, 4, 1, 1, 1,
        4, 2, 2, 6, 2, 2, 2, 6, 2, 1, 1, 2, 2, 1, 2, 2, 2, 6, 4
    ]

    assert nrpts == nrpts_ref
    assert np.allclose(ndegen, ndegen_ref)


def estimate_hr_dat(nrpts: int, num_wann: int) -> int:
    """Estimate seedname_hr.dat file size.

    :param nrpts: [description]
    :type nrpts: int
    :param num_wann: [description]
    :type num_wann: int
    :return: [description]
    :rtype: int
    """
    # header
    total_size = _SIZE_CARRIAGE_CONTROL + _SIZE_CHARACTER * 33 + _SIZE_LINE_FEED
    # num_wann
    total_size += 12 + _SIZE_LINE_FEED
    # nrpts
    total_size += 12 + _SIZE_LINE_FEED
    # ndegen
    nline, rem = divmod(nrpts, 15)
    total_size += (5 * 15 + _SIZE_LINE_FEED) * nline + 5 * rem + _SIZE_LINE_FEED
    # H(R)
    total_size += (5 * 5 + 12 * 2 + _SIZE_LINE_FEED) * nrpts * num_wann**2

    return total_size


def estimate_r_dat(nrpts: int, num_wann: int) -> int:
    """Estimate seedname_r.dat file size.

    :param nrpts: [description]
    :type nrpts: int
    :param num_wann: [description]
    :type num_wann: int
    :return: [description]
    :rtype: int
    """
    # header
    total_size = _SIZE_CARRIAGE_CONTROL + _SIZE_CHARACTER * 33 + _SIZE_LINE_FEED
    # num_wann
    total_size += 12 + _SIZE_LINE_FEED
    # nrpts
    total_size += 12 + _SIZE_LINE_FEED
    # R(R)
    total_size += (5 * 5 + 12 * 6 + _SIZE_LINE_FEED) * nrpts * num_wann**2

    return total_size


def estimate_wsvec_dat(nrpts: int, num_wann: int, wdist_ndeg: list = None) -> int:
    """Estimate seedname_wsvec.dat file size.

    :param nrpts: [description]
    :type nrpts: int
    :param num_wann: [description]
    :type num_wann: int
    :param wdist_ndeg: [description], defaults to None
    :type wdist_ndeg: list, optional
    :return: [description]
    :rtype: int
    """
    # header
    total_size = _SIZE_CHARACTER * 100 + _SIZE_LINE_FEED
    # irvec
    total_size += (5 * 5 + _SIZE_LINE_FEED) * nrpts * num_wann**2
    # wdist_ndeg, not exact: I need wdist_ndeg
    total_size += (5 + _SIZE_LINE_FEED) * nrpts * num_wann**2
    # irdist_ws
    if wdist_ndeg:
        total_size += (5 * 3 + _SIZE_LINE_FEED) * np.sum(wdist_ndeg) * num_wann**2

    return total_size


def estimate_centres_xyz(num_wann: int, num_atoms: int) -> int:
    """Estimate seedname_centres.xyz file size.

    :param num_wann: [description]
    :type num_wann: int
    :param num_atoms: [description]
    :type num_atoms: int
    :return: [description]
    :rtype: int
    """
    # num_wann + num_atoms
    total_size = 6 + _SIZE_LINE_FEED
    # comment
    total_size += _SIZE_CARRIAGE_CONTROL + 62 + _SIZE_LINE_FEED
    # WFs
    total_size += (1 + 6 + (14 + 3) * 3 - 3 + _SIZE_LINE_FEED) * num_wann
    # atoms
    total_size += (2 + 5 + (14 + 3) * 3 - 3 + _SIZE_LINE_FEED) * num_atoms

    return total_size


def estimate_tb_dat(nrpts: int, num_wann: int) -> int:
    """Estimate seedname_tb.dat file size.

    :param nrpts: [description]
    :type nrpts: int
    :param num_wann: [description]
    :type num_wann: int
    :return: [description]
    :rtype: int
    """
    # header
    total_size = _SIZE_CARRIAGE_CONTROL + _SIZE_CHARACTER * 33 + _SIZE_LINE_FEED
    # lattice
    total_size += (24 * 3 + _SIZE_LINE_FEED) * 3
    # num_wann
    total_size += 12 + _SIZE_LINE_FEED
    # nrpts
    total_size += 12 + _SIZE_LINE_FEED
    # ndegen
    nline, rem = divmod(nrpts, 15)
    total_size += (5 * 15 + _SIZE_LINE_FEED) * nline + 5 * rem + _SIZE_LINE_FEED
    # H(R)
    total_size += (5 * 3 + _SIZE_LINE_FEED * 2) * nrpts
    total_size += (5 * 2 + 2 + 16 * 2 + _SIZE_LINE_FEED) * num_wann**2 * nrpts
    # R(R)
    total_size += (5 * 3 + _SIZE_LINE_FEED * 2) * nrpts
    total_size += (5 * 2 + 2 + 16 * 6 + _SIZE_LINE_FEED) * num_wann**2 * nrpts

    return total_size


def estimate_xsf(
    num_wann: int,
    num_atoms: int,
    nr1: int,
    nr2: int,
    nr3: int,
    reduce_unk: bool = False,
    wannier_plot_supercell: int = 2
) -> int:
    """Estimate seedname_0000*.xsf file size.

    :param num_wann: [description]
    :type num_wann: int
    :param num_atoms: [description]
    :type num_atoms: int
    :param nr1: [description]
    :type nr1: int
    :param nr2: [description]
    :type nr2: int
    :param nr3: [description]
    :type nr3: int
    :param reduce_unk: [description], defaults to False
    :type reduce_unk: bool, optional
    :param wannier_plot_supercell: [description], defaults to 2
    :type wannier_plot_supercell: int, optional
    :return: [description]
    :rtype: int
    """
    if reduce_unk:
        nr1 = (nr1 + 1) // 2
        nr2 = (nr2 + 1) // 2
        nr3 = (nr3 + 1) // 2

    # header
    total_size = _SIZE_CARRIAGE_CONTROL + 7 + _SIZE_LINE_FEED
    total_size += _SIZE_CARRIAGE_CONTROL + 62 + _SIZE_LINE_FEED
    total_size += _SIZE_CARRIAGE_CONTROL + 33 + _SIZE_LINE_FEED
    total_size += _SIZE_CARRIAGE_CONTROL + 7 + _SIZE_LINE_FEED
    # CRYSTAL
    total_size += 7 + _SIZE_LINE_FEED
    # PRIMVEC
    total_size += 7 + _SIZE_LINE_FEED
    total_size += (12 * 3 + _SIZE_LINE_FEED) * 3
    # CONVVEC
    total_size += 7 + _SIZE_LINE_FEED
    total_size += (12 * 3 + _SIZE_LINE_FEED) * 3
    # PRIMCOORD
    total_size += 9 + _SIZE_LINE_FEED
    # num_atoms
    total_size += 6 + 3 + _SIZE_LINE_FEED
    total_size += (2 + 3 + 12 * 3 + _SIZE_LINE_FEED) * num_atoms
    # newlines
    total_size += _SIZE_LINE_FEED * 2
    # BEGIN_BLOCK_DATAGRID_3D, 3D_field, BEGIN_DATAGRID_3D_UNKNOWN
    total_size += 23 + _SIZE_LINE_FEED
    total_size += 8 + _SIZE_LINE_FEED
    total_size += 25 + _SIZE_LINE_FEED
    # grid size
    total_size += 6 * 3 + _SIZE_LINE_FEED
    # x_0
    total_size += 12 * 3 + _SIZE_LINE_FEED
    # grid length
    total_size += (12 * 3 + _SIZE_LINE_FEED) * 3
    # data
    ngrid = wannier_plot_supercell**3 * nr1 * nr2 * nr3
    nline, rem = divmod(ngrid, 6)
    total_size += (13 * 6 + _SIZE_LINE_FEED) * nline
    if rem > 0:
        total_size += 13 * rem + _SIZE_LINE_FEED
    # END_DATAGRID_3D, END_BLOCK_DATAGRID_3D
    total_size += 15 + _SIZE_LINE_FEED
    total_size += 21 + _SIZE_LINE_FEED

    total_size *= num_wann

    return total_size


def get_number_of_nearest_neighbors(recip_lattice: np.array, kmesh: ty.List[int]) -> int:
    """Find the number of nearest neighors.

    :param recip_lattice: reciprocal lattice, 3x3, each row is a lattice vector
    :type recip_lattice: np.array
    :param kmesh: number of kpoints along kx,ky,kz direction
    :type kmesh: ty.List[int]
    :return: [description]
    :rtype: int
    """
    from sklearn.neighbors import KDTree

    # The search supercell is several folds larger
    supercell_size = 5
    # Tolerance for equal-distance kpoints
    kmesh_tol = 1e-6  # pylint: disable=unused-variable
    # max number of NN
    num_nnmax = 12

    nkx, nky, nkz = kmesh
    x = np.linspace(0, supercell_size, nkx * supercell_size)
    y = np.linspace(0, supercell_size, nky * supercell_size)
    z = np.linspace(0, supercell_size, nkz * supercell_size)
    samples = np.meshgrid(x, y, z)
    x = samples[0].reshape((-1, 1))
    y = samples[1].reshape((-1, 1))
    z = samples[2].reshape((-1, 1))
    samples = np.hstack([x, y, z])

    # scaled coordinates -> cartesian coordinates
    samples = samples @ recip_lattice
    print(samples)

    # the centering point
    center = np.array([2, 2, 2]).reshape((1, 3)) @ recip_lattice
    print(center)

    tree = KDTree(samples)
    dist, ind = tree.query(center, k=num_nnmax + 12)  # 0th element is always itself

    print(dist, ind)


WannierFileSize = namedtuple(
    'WannierFileSize',
    [
        # files
        'amn',
        'mmn',
        'eig',
        'unk',
        'unk_reduce',
        'chk',
        # files for Hamiltonian
        'centres_xyz',
        'hr_dat',
        'r_dat',
        'wsvec_dat',
        'tb_dat',
        'xsf',
        'xsf_reduce',
        'xsf_supercell3',
        'xsf_reduce_supercell3',
        # 'cube',
        # parameters
        'structure',  # formula
        'structure_pk',
        'num_bands',
        'num_exclude_bands',
        'num_wann',
        'num_kpts',
        'nntot',
        'nr1',
        'nr2',
        'nr3',
        'nrpts',
    ]
)


def estimate_workflow(  # pylint: disable=too-many-statements
    structure: orm.StructureData,
    file_format: WannierFileFormat = WannierFileFormat.FORTRAN_FORMATTED
) -> WannierFileSize:
    """Estimate AMN/MMN/EIG/UNK/CHK file sizes of a structure.

    :param structure: [description]
    :type structure: orm.StructureData
    :param file_format: [description], defaults to WannierFileFormat.FORTRAN_FORMATTED
    :type file_format: WannierFileFormat, optional
    :raises ValueError: [description]
    :raises ValueError: [description]
    :return: [description]
    :rtype: WannierFileSize
    """
    from aiida.plugins import GroupFactory
    from aiida.common import exceptions
    from aiida_wannier90_workflows.utils.upf import get_wannier_number_of_bands, get_number_of_projections
    from aiida_wannier90_workflows.utils.kmesh import create_kpoints_from_distance
    from aiida_wannier90_workflows.workflows.wannier import get_pseudo_orbitals, get_semicore_list
    from aiida_wannier90_workflows.utils.predict_smooth_grid import predict_smooth_grid

    SsspFamily = GroupFactory('pseudo.family.sssp')
    PseudoDojoFamily = GroupFactory('pseudo.family.pseudo_dojo')
    CutoffsPseudoPotentialFamily = GroupFactory('pseudo.family.cutoffs')

    pseudo_family = 'SSSP/1.1/PBE/efficiency'
    try:
        pseudo_set = (PseudoDojoFamily, SsspFamily, CutoffsPseudoPotentialFamily)
        pseudo_family = orm.QueryBuilder().append(pseudo_set, filters={'label': pseudo_family}).one()[0]
    except exceptions.NotExistent as exception:
        raise ValueError(
            f'required pseudo family `{pseudo_family}` is not installed. Please use `aiida-pseudo install` to'
            'install it.'
        ) from exception
    pseudos = pseudo_family.get_pseudos(structure=structure)

    try:
        cutoff_wfc, cutoff_rho = pseudo_family.get_recommended_cutoffs(structure=structure, unit='Ry')  # pylint: disable=unused-variable
    except ValueError as exception:
        raise ValueError(
            f'failed to obtain recommended cutoffs for pseudo family `{pseudo_family}`: {exception}'
        ) from exception

    nbands_factor = 2.0
    nbnd = get_wannier_number_of_bands(
        structure=structure,
        pseudos=pseudos,
        factor=nbands_factor,
        only_valence=False,
        spin_polarized=False,
        spin_orbit_coupling=False
    )

    pseudo_orbitals = get_pseudo_orbitals(pseudos)
    semicore_list = get_semicore_list(structure, pseudo_orbitals)
    num_exclude_bands = len(semicore_list)

    num_bands = nbnd - num_exclude_bands

    num_projs = get_number_of_projections(structure, pseudos, spin_orbit_coupling=False)
    num_wann = num_projs - num_exclude_bands

    kpoints_distance = 0.2
    kpoints = create_kpoints_from_distance(structure, kpoints_distance)
    kmesh = kpoints.get_kpoints_mesh()[0]
    num_kpts = np.prod(kmesh)

    # nntot depends on the BZ, usually in range 8 - 12, for now I set it as 10
    # FIXME this is the only one parameter which makes the prediction inaccurate  # pylint: disable=fixme
    nntot = 10
    # This is wrong, nntot is the bvectors satisfying B1 condition
    # nntot = get_number_of_nearest_neighbors(recip_lattice=kpoints.reciprocal_cell, kmesh=kmesh)

    # smooth FFT grid
    nr1, nr2, nr3 = predict_smooth_grid(structure=structure, ecutwfc=cutoff_wfc)

    # print(f'{structure.get_formula()=}')
    # print(f"{num_bands=}")
    # print(f"{num_wann=}")
    # print(f"{num_kpts=}")
    # print(f"{nntot=}")
    # print(f"{nr1=}")
    # print(f"{nr2=}")
    # print(f"{nr3=}")
    # print(f"{num_exclude_bands=}")

    amn_size = estimate_amn(num_bands=num_bands, num_wann=num_wann, num_kpts=num_kpts, file_format=file_format)
    mmn_size = estimate_mmn(num_bands=num_bands, num_kpts=num_kpts, nntot=nntot, file_format=file_format)
    eig_size = estimate_eig(num_bands=num_bands, num_kpts=num_kpts, file_format=file_format)
    unk_size = estimate_unk(
        nr1=nr1, nr2=nr2, nr3=nr3, num_bands=num_bands, num_kpts=num_kpts, reduce_unk=False, file_format=file_format
    )
    unk_reduce = estimate_unk(
        nr1=nr1, nr2=nr2, nr3=nr3, num_bands=num_bands, num_kpts=num_kpts, reduce_unk=True, file_format=file_format
    )
    # default chk is unformatted
    chk_size = estimate_chk(
        num_exclude_bands=num_exclude_bands,
        num_bands=num_bands,
        num_wann=num_wann,
        num_kpts=num_kpts,
        nntot=nntot,
        have_disentangled=True,
        # file_format=WannierFileFormat.FORTRAN_UNFORMATTED
        file_format=file_format
    )

    nrpts, _ = get_nrpts_ndegen(structure.cell, kmesh)
    num_atoms = len(structure.sites)

    centres_xyz = estimate_centres_xyz(num_wann, num_atoms)
    hr_dat = estimate_hr_dat(nrpts, num_wann)
    r_dat = estimate_r_dat(nrpts, num_wann)
    wsvec_dat = estimate_wsvec_dat(nrpts, num_wann)
    tb_dat = estimate_tb_dat(nrpts, num_wann)
    xsf = estimate_xsf(num_wann, num_atoms, nr1, nr2, nr3)
    xsf_reduce = estimate_xsf(num_wann, num_atoms, nr1, nr2, nr3, True)
    xsf_supercell3 = estimate_xsf(num_wann, num_atoms, nr1, nr2, nr3, False, 3)
    xsf_reduce_supercell3 = estimate_xsf(num_wann, num_atoms, nr1, nr2, nr3, True, 3)

    sizes = WannierFileSize(
        amn=amn_size,
        mmn=mmn_size,
        eig=eig_size,
        unk=unk_size,
        chk=chk_size,
        #
        unk_reduce=unk_reduce,
        centres_xyz=centres_xyz,
        hr_dat=hr_dat,
        r_dat=r_dat,
        wsvec_dat=wsvec_dat,
        tb_dat=tb_dat,
        xsf=xsf,
        xsf_reduce=xsf_reduce,
        xsf_supercell3=xsf_supercell3,
        xsf_reduce_supercell3=xsf_reduce_supercell3,
        #
        structure=structure.get_formula(),
        structure_pk=structure.pk,
        num_bands=num_bands,
        num_exclude_bands=num_exclude_bands,
        num_wann=num_wann,
        num_kpts=num_kpts,
        nntot=nntot,
        nr1=nr1,
        nr2=nr2,
        nr3=nr3,
        #
        nrpts=nrpts,
    )

    return sizes


def estimate_structure_group(group: ty.Union[orm.Group, str], hdf_file: str, file_format: WannierFileFormat):
    """Estimate AMN/MMN/EIG/UNK/CHK file sizes of all the structures in a group.

    :param group: the group containing all the structures to be estimated.
    :type group: orm.Group
    :param hdf_file: The hdf5 file name to store the results of estimation.
    :type hdf_file: str
    :param file_format: [description]
    :type file_format: WannierFileFormat
    """
    import pandas as pd

    if isinstance(group, str):
        group = orm.load_group(group)

    num_total = len(group.nodes)

    results = []
    for i, structure in enumerate(group.nodes):
        size = estimate_workflow(structure, file_format)
        print(f'{i+1}/{num_total}', size)
        results.append(size)

    store = pd.HDFStore(hdf_file)
    df = pd.DataFrame(results, columns=WannierFileSize._fields)
    # print(df)

    # save it
    store['df'] = df
    store.close()

    print(f'Estimation for structure group "{group.label}" stored in {hdf_file}')


def human_readable_size(num: int, suffix: str = 'B') -> str:
    """Return a human-readable file size for a given file size in bytes.

    :param num: file size, in bytes.
    :type num: int
    :param suffix: the ending char, defaults to 'B'
    :type suffix: str, optional
    :return: human-readable file size, e.g. 2.2MiB.
    :rtype: str
    """
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'


def print_estimation(hdf_file: str):
    """Print the estimation from results stored in a HDF5 file.

    :param hdf_file: [description]
    :type hdf_file: str
    :raises ValueError: [description]
    """
    import os.path
    import pandas as pd
    from tabulate import tabulate

    if not os.path.exists(hdf_file):
        raise ValueError(f'File not existed: {hdf_file}')

    store = pd.HDFStore(hdf_file)

    # load it
    df = store['df']

    num_structures = len(df)
    print(f'{num_structures=}')

    total = np.sum(df.amn) + np.sum(df.mmn) + np.sum(df.eig) + np.sum(df.chk)
    print(f'total_w/o_unk={human_readable_size(total)}')
    total += np.sum(df.unk)
    print(f'total_w/_unk={human_readable_size(total)}')
    print()

    headers = ['file', 'min', 'max', 'average', 'total']
    table = []
    for key in [
        'amn', 'mmn', 'eig', 'unk', 'unk_reduce', 'chk', 'centres_xyz', 'hr_dat', 'r_dat', 'wsvec_dat', 'tb_dat', 'xsf',
        'xsf_reduce', 'xsf_supercell3', 'xsf_reduce_supercell3'
    ]:
        minval = human_readable_size(min(df[key]))
        maxval = human_readable_size(max(df[key]))
        average = human_readable_size(np.average(df[key]))
        total = human_readable_size(np.sum(df[key]))
        table.append([key, minval, maxval, average, total])
    print(tabulate(table, headers))
    print()

    headers = ['param', 'min', 'max', 'average']
    table = []
    for key in ['num_bands', 'num_exclude_bands', 'num_wann', 'num_kpts', 'nr1', 'nr2', 'nr3', 'nrpts']:
        minval = min(df[key])
        maxval = max(df[key])
        average = np.average(df[key])
        table.append([key, minval, maxval, average])
    print(tabulate(table, headers))

    store.close()


def plot_histogram(hdf_file: str):
    """Plot a histogram for AMN/MMN/EIG/UNK/CHK file sizes.

    :param hdf_file: [description]
    :type hdf_file: str
    :raises ValueError: [description]
    """
    import os.path
    import pandas as pd
    import matplotlib.pyplot as plt
    from ase.formula import Formula

    def get_num_bins(x, step):
        """Calculate number of bins in histogram."""
        num_bins, mod = divmod(max(x), step)
        if mod != 0:
            num_bins += 1
        return num_bins

    def get_size_histogram(x, y, step):
        num_bins = get_num_bins(x, step)
        hist_y = np.zeros(num_bins, dtype=int)

        for i, y_i in enumerate(y):
            # minus 1 so the end point is included, e.g. when step=5, 5 in (0, 5], 5 not in (5, 10]
            # x[i] should > 0
            ind = (x[i] - 1) // step
            hist_y[ind] += y_i

        hist_x = np.arange(step, step * (num_bins + 1), step)
        return hist_x, hist_y

    get_num_atoms = lambda _: len(Formula(_))

    if not os.path.exists(hdf_file):
        raise ValueError(f'File not existed: {hdf_file}')

    store = pd.HDFStore(hdf_file)

    # load it
    df = store['df']

    num_atoms = list(map(get_num_atoms, df['structure']))
    # print(num_atoms)

    fig, axs = plt.subplots(2, 3)

    step = 1
    num_bins = get_num_bins(num_atoms, step)

    print('Processing #atoms histogram')
    axs[0, 0].hist(num_atoms, num_bins)
    axs[0, 0].set_title('Histogram for number of atoms')
    axs[0, 0].set_ylabel('Count')
    axs[0, 0].set_xlabel('number of atoms')
    axs[0, 0].set_xscale('log')

    ax_iter = iter(fig.axes)
    next(ax_iter)

    for ftype in ['amn', 'mmn', 'eig', 'unk', 'chk']:
        print(f'Processing {ftype} histogram')
        amn_x, amn_y = get_size_histogram(num_atoms, df[ftype].to_numpy(), step)
        # Convert into GiB
        amn_y = np.cumsum(amn_y / 1024**3)
        ax = next(ax_iter)
        ax.bar(amn_x, amn_y, width=1.5)
        ax.set_title(f'Cumulative sum for {ftype}')
        ax.set_ylabel('File size / GiB')
        ax.set_xlabel('number of atoms')
        ax.set_xscale('log')

    store.close()
    plt.show()


def test_estimators():
    """Test estimators."""
    # TODO some tests to be moved to other dir  # pylint: disable=fixme
    # at least this is true for oneAPI
    assert estimate_amn(12, 7, 64, WannierFileFormat.FORTRAN_FORMATTED) == 279651
    assert estimate_amn(12, 7, 64, WannierFileFormat.FORTRAN_UNFORMATTED) == 193624

    assert estimate_mmn(12, 64, 8, WannierFileFormat.FORTRAN_FORMATTED) == 2741347
    assert estimate_mmn(12, 64, 8, WannierFileFormat.FORTRAN_UNFORMATTED) == 1783896

    assert estimate_eig(12, 64, WannierFileFormat.FORTRAN_FORMATTED) == 22272
    assert estimate_eig(12, 64, WannierFileFormat.FORTRAN_UNFORMATTED) == 18432

    assert estimate_unk(18, 18, 18, 12, 64, False, WannierFileFormat.FORTRAN_FORMATTED) == 2869405 * 64
    assert estimate_unk(18, 18, 18, 12, 64, False, WannierFileFormat.FORTRAN_UNFORMATTED) == 1119868 * 64

    assert 0.95 < estimate_chk(
        num_exclude_bands=0,
        num_bands=12,
        num_wann=7,
        num_kpts=64,
        nntot=8,
        have_disentangled=True,
        file_format=WannierFileFormat.FORTRAN_FORMATTED
    ) / 1721453 < 1.05
    assert estimate_chk(
        num_exclude_bands=0,
        num_bands=12,
        num_wann=7,
        num_kpts=64,
        nntot=8,
        have_disentangled=True,
        file_format=WannierFileFormat.FORTRAN_UNFORMATTED
    ) == 543097

    test_get_nrpts_ndegen()

    assert estimate_hr_dat(93, 8) == 298133
    assert estimate_r_dat(93, 8) == 583357
    assert estimate_wsvec_dat(93, 8) == 190565  # in fact real file size is 299457
    assert estimate_centres_xyz(8, 2) == 631
    assert estimate_tb_dat(93, 8) == 920522
    assert estimate_xsf(7, 1, 18, 18, 18) == 614995 * 7
    assert estimate_xsf(7, 1, 18, 18, 18, True) == 614995 * 7
    assert estimate_xsf(7, 1, 18, 18, 18, True) == 77479 * 7

    # minimum: hr + wsvec + centres
    # medium:  hr + wsvec + r
    # medium2: tb + wsvec
    # maximum: tb + wsvec + xsf_reduce
    # maximum: tb + wsvec + xsf_full
