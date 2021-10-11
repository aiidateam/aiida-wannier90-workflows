#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Storage estimators for files related to wannier90."""
import typing as ty
from dataclasses import dataclass
import numpy as np
# import numpy.typing as npt
from aiida_wannier90_workflows.common.types import WannierFileFormat

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


if __name__ == '__main__':
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
