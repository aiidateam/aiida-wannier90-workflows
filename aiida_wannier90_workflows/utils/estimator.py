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
    kmesh_tol = 1e-6
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
    'WannierFileSize', [
        'amn',
        'mmn',
        'eig',
        'unk',
        'chk',
        'structure',
        'structure_pk',
        'num_bands',
        'num_exclude_bands',
        'num_wann',
        'num_kpts',
        'nntot',
        'nr1',
        'nr2',
        'nr3',
    ]
)


def estimate_workflow(structure: orm.StructureData) -> WannierFileSize:
    import numpy as np
    from aiida.plugins import CalculationFactory, GroupFactory
    from aiida.common import exceptions
    from aiida_wannier90_workflows.utils.upf import get_wannier_number_of_bands, get_number_of_projections
    from aiida_wannier90_workflows.utils.kmesh import create_kpoints_from_distance
    from aiida_wannier90_workflows.workflows.wannier import get_pseudo_orbitals, get_semicore_list
    from aiida_wannier90_workflows.utils.predict_smooth_grid import predict_smooth_grid

    PwCalculation = CalculationFactory('quantumespresso.pw')
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
        cutoff_wfc, cutoff_rho = pseudo_family.get_recommended_cutoffs(structure=structure, unit='Ry')
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
    # FIXME this is the only one parameter which makes the prediction inaccurate
    nntot = 10
    # This is wrong, nntot is the bvectors satisfying B1 condition
    # nntot = get_number_of_nearest_neighbors(recip_lattice=kpoints.reciprocal_cell, kmesh=kmesh)

    # smooth FFT grid
    nr1, nr2, nr3 = predict_smooth_grid(structure=structure, ecutwfc=cutoff_wfc)

    # file_format = WannierFileFormat.FORTRAN_FORMATTED
    file_format = WannierFileFormat.FORTRAN_UNFORMATTED

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

    sizes = WannierFileSize(
        amn=amn_size,
        mmn=mmn_size,
        eig=eig_size,
        unk=unk_size,
        chk=chk_size,
        structure=structure.get_formula(),
        structure_pk=structure.pk,
        num_bands=num_bands,
        num_exclude_bands=num_exclude_bands,
        num_wann=num_wann,
        num_kpts=num_kpts,
        nntot=nntot,
        nr1=nr1,
        nr2=nr2,
        nr3=nr3
    )

    return sizes


def estimate_structure_group(group_label: str):
    import pandas as pd

    group = orm.load_group(group_label)

    num_total = len(group.nodes)

    results = []
    for i, structure in enumerate(group.nodes):
        size = estimate_workflow(structure)
        print(f'{i+1}/{num_total}', size)
        results.append(size)

    store = pd.HDFStore('wannier_storage_estimator.h5')
    df = pd.DataFrame(results, columns=WannierFileSize._fields)
    # print(df)
    # save it
    store['df'] = df


def human_readable_size(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return f'{num:3.1f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.1f}Yi{suffix}'


def print_sizes():
    import pandas as pd
    from tabulate import tabulate

    byte2mb = lambda _: _ / 1024**2

    # store = pd.HDFStore('wannier_storage_estimator_all_formatted_chk_unformatted.h5')
    store = pd.HDFStore('wannier_storage_estimator_all_unformatted.h5')
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
    for key in ['amn', 'mmn', 'eig', 'unk', 'chk']:
        minval = human_readable_size(min(df[key]))
        maxval = human_readable_size(max(df[key]))
        average = human_readable_size(np.average(df[key]))
        total = human_readable_size(np.sum(df[key]))
        table.append([key, minval, maxval, average, total])
    print(tabulate(table, headers))
    print()

    headers = ['param', 'min', 'max', 'average']
    table = []
    for key in ['num_bands', 'num_exclude_bands', 'num_wann', 'num_kpts', 'nr1', 'nr2', 'nr3']:
        minval = min(df[key])
        maxval = max(df[key])
        average = np.average(df[key])
        table.append([key, minval, maxval, average])
    print(tabulate(table, headers))


def test_estimators():
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


if __name__ == '__main__':
    import aiida
    aiida.load_profile()
    # import ase

    # ase_dict = {'numbers': np.array([29]),
    #             'positions': np.array([[0., 0., 0.]]),
    #             'initial_magmoms': np.array([0.]),
    #             'cell': np.array([[-1.80502346,  0.        ,  1.80502346],
    #                     [ 0.        ,  1.80502346,  1.80502346],
    #                     [-1.80502346,  1.80502346,  0.        ]]),
    #             'pbc': np.array([ True,  True,  True])}
    # ase_atoms = ase.Atoms.fromdict(ase_dict)
    # structure = orm.StructureData(ase=ase_atoms)

    # structure = orm.load_node(63497).inputs.structure
    # print(estimate_workflow(structure))
    # Actual size @ eiger with intel oneAPI
    # amn = 3640099, mmn = 58224099, eig = 406000, chk = 4804585 (unformatted)

    # estimate_structure_group('3DD_relax_structures')
    estimate_structure_group('structure/3dcd/experimental')

    # print_sizes()
