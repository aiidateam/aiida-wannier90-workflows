from collections import defaultdict
import math

import numpy as np


def get_prime_factors(number):
    """Get a dictionary of all prime factors."""
    start_number = number
    assert number > 1

    factors = defaultdict(int)

    for factor in range(2, start_number + 1):
        # while i divides n , print i ad divide n
        while number % factor == 0:
            factors[factor] += 1
            number = number // factor

    # No factor found: it is a prime number
    if not factors:
        factors[start_number] = 1

    check = 1
    for factor, power in factors.items():
        check *= factor**power
    assert check == start_number, str(factors)

    return dict(factors)


def good_fft_order(start_val):
    """Find the next integer that only contains, as prime factors, only 2, 3 or 5.
    Thus is what QE also does (everywhere except on IBM where a few more values are ok).
    """
    retval = start_val
    # Continue if there are other factors other than 2, 3 or 5
    while set(get_prime_factors(retval).keys()).difference([2, 3, 5]):
        retval += 1
    return retval


def grid_set(bg, gcut, nr1, nr2, nr3):
    """
    q-e/FFTXlib/fft_types.f90
    this routine returns in nr1, nr2, nr3 the minimal 3D real-space FFT
    grid required to fit the G-vector sphere with G^2 <= gcut
    On input, nr1,nr2,nr3 must be set to values that match or exceed
    the largest i,j,k (Miller) indices in G(i,j,k) = i*b1 + j*b2 + k*b3

    Units of bg, gcut are 2pi/ang
    """
    nb1 = 0
    nb2 = 0
    nb3 = 0

    # calculate moduli of G vectors and the range of indices where |G|^2 < gcut

    for k in range(-nr3, nr3 + 1):
        for j in range(-nr2, nr2 + 1):
            for i in range(-nr1, nr1 + 1):
                ijk = np.array([[i, j, k]]).T  # column vector
                g = np.matmul(bg, ijk)
                # calculate modulus
                gsq = np.linalg.norm(g)
                if gsq < gcut:
                    # calculate maximum index
                    nb1 = max(nb1, abs(i))
                    nb2 = max(nb2, abs(j))
                    nb3 = max(nb3, abs(k))

    # the size of the 3d FFT matrix depends upon the maximum indices. With
    # the following choice, the sphere in G-space "touches" its periodic image

    nr1 = 2 * nb1 + 1
    nr2 = 2 * nb2 + 1
    nr3 = 2 * nb3 + 1
    return nr1, nr2, nr3


def grid_set_vectorized(bg, gcut, nr1, nr2, nr3):
    """
    Vectorized version of grid_set, remove for loop to run faster
    Approximately 16x faster.
    """
    # calculate moduli of G vectors and the range of indices where |G|^2 < gcut
    indices = np.meshgrid(
        range(-nr1, nr1 + 1), range(-nr2, nr2 + 1), range(-nr3, nr3 + 1)
    )
    indices = np.array(indices)
    vectors = np.tensordot(indices, bg, axes=(0, 0))
    norms = np.linalg.norm(vectors, axis=-1)
    # set vectors with norm >= gcut to 0
    mask = norms >= gcut
    mask = np.repeat(mask[np.newaxis, :, :, :], 3, axis=0)
    indices[mask] = 0
    nb = np.max(np.abs(indices), axis=(1, 2, 3))
    # the size of the 3d FFT matrix depends upon the maximum indices. With
    # the following choice, the sphere in G-space "touches" its periodic image
    return 2 * nb + 1


def get_smooth_grid_internal(wfc_cutoff_ry, max_kpt, cell_ang):
    """Return the number of plane waves, knowing the cutoff (in Ry)
    and the 3x3 cell matrix in Ang."""

    # set_cutoff called via the call stack
    # init_run -> init_dimensions -> fft_type_init -> fft_type_allocate -> realspace_grid_init

    wfc_cutoff_eV = 13.6056980659 * wfc_cutoff_ry

    # gcutw = ecutwfc / tpiba2; here I will remove all 'alat' factors that
    # cancel out at the end (you can imagine alat=1 Ang if you want, it's just
    # a scaling factor)

    # this is sqrt(gcutw), in a sense
    # I use: hbar^2 /2 / electron mass  in angstrom^2 * eV = 3.80998208
    k_radius_inv_angstrom = np.sqrt(wfc_cutoff_eV / 3.80998208) / 2.0 / math.pi
    # Factor 1/2pi to get in units of 2pi/ang

    # kcut in the code is the max norm of kpoints, that we get here from the
    # outside
    # sqrt(kcut) is (in QE) in units of 2pi/a (in a.u.)

    # gkcut = ( sqrt( kcut ) + sqrt( gcutw ) ) ** 2

    # This is sqrt(gkcut), in a sense
    max_k_radius = k_radius_inv_angstrom + max_kpt

    # in init_dimensions (data_structure.f90):
    # CALL fft_type_init( dffts, smap, "wave", gamma_only, lpara, intra_bgrp_comm,&
    #   at, bg, gkcut, gcutms/gkcut, fft_fact=fft_fact, nyfft=nyfft )

    # so gkcut is passed as gcut_in to fft_type_init, and dual is
    # gcutms/gkcut
    #
    # with gcutms = 4.D0 * ecutwfc / tpiba2
    #   (so 4 * gcutw)
    #
    # so:
    # gcutms/gkcut = (4.D0 * ecutwfc / tpiba2) / gkcut** 2
    # gcutms/gkcut = (4.D0 * gcutw) / gkcut**2
    # (Again, here we remove the 'alat' factor)
    dual_in = 4 * k_radius_inv_angstrom**2 / max_k_radius**2

    # in fft_type_init:
    # ELSE IF ( pers == 'wave' ) THEN
    #    gkcut = gcut_in
    #    gcut = gkcut * dual

    # This is the square root of gcut in the code
    ## NOTE [GP]: I am not sure of this step, shouldn't it be sqrt(dual_in)?
    ## Or I'm getting something wrong?
    gcut = max_k_radius * dual_in

    # and 'gcut' is passed as gcutm to fft_type_allocate -> realspace_grid_init

    # from fft_typs.f90, realspace_grid_init:
    # dfft%nr1 = int ( sqrt (gcutm) * sqrt (at(1, 1)**2 + at(2, 1)**2 + at(3, 1)**2) ) + 1
    grid1 = int(gcut * np.linalg.norm(cell_ang[0])) + 1
    grid2 = int(gcut * np.linalg.norm(cell_ang[1])) + 1
    grid3 = int(gcut * np.linalg.norm(cell_ang[2])) + 1

    # reciprocal cell
    # bg = np.linalg.inv(cell_ang).T # w/o (* 2. * math.pi), in unit 2pi/ang
    # grid1, grid2, grid3 = grid_set(bg, gcut, grid1, grid2, grid3)
    # grid1, grid2, grid3 = grid_set_vectorized(bg, gcut, grid1, grid2, grid3)

    return good_fft_order(grid1), good_fft_order(grid2), good_fft_order(grid3)


def get_dense_grid_internal(rho_cutoff_ry, cell_ang):
    """Return the dimension of dense FFT grid, knowing the cutoff (in Ry)
    and the 3x3 cell matrix in Ang."""

    # set_cutoff called via the call stack
    # init_run -> init_dimensions -> fft_type_init -> fft_type_allocate -> realspace_grid_init

    rho_cutoff_eV = 13.6056980659 * rho_cutoff_ry

    # gcutm = ecutrho / tpiba2; here I will remove all 'alat' factors that
    # cancel out at the end (you can imagine alat=1 Ang if you want, it's just
    # a scaling factor)

    # this is sqrt(gcutm), in a sense
    # I use: hbar^2 /2 / electron mass  in angstrom^2 * eV = 3.80998208
    k_radius_inv_angstrom = np.sqrt(rho_cutoff_eV / 3.80998208) / 2.0 / math.pi
    # Factor 1/2pi to get in units of 2pi/ang

    # in init_dimensions (data_structure.f90):
    # CALL fft_type_init( dfftp, smap, "rho" , gamma_only, lpara, intra_bgrp_comm,&
    #   at, bg, gcutm , 4.d0, fft_fact=fft_fact, nyfft=nyfft )
    dual_in = 4

    # in fft_type_init:
    # IF( pers == 'rho' ) THEN
    #     gcut = gcut_in
    #     gkcut = gcut / dual
    gcut = k_radius_inv_angstrom

    # and 'gcut' is passed as gcutm to fft_type_allocate -> realspace_grid_init

    # from fft_typs.f90, realspace_grid_init:
    # dfft%nr1 = int ( sqrt (gcutm) * sqrt (at(1, 1)**2 + at(2, 1)**2 + at(3, 1)**2) ) + 1
    grid1 = int(gcut * np.linalg.norm(cell_ang[0])) + 1
    grid2 = int(gcut * np.linalg.norm(cell_ang[1])) + 1
    grid3 = int(gcut * np.linalg.norm(cell_ang[2])) + 1

    # reciprocal cell
    bg = np.linalg.inv(cell_ang).T  # w/o (* 2. * math.pi), in unit 2pi/ang
    # grid1, grid2, grid3 = grid_set(bg, gcut, grid1, grid2, grid3)
    grid1, grid2, grid3 = grid_set_vectorized(bg, gcut, grid1, grid2, grid3)

    return good_fft_order(grid1), good_fft_order(grid2), good_fft_order(grid3)


def get_actual_max_kpt(calc):
    # Need to check the grid of the SCF, not of the bands
    kpoints_rel = calc.inputs.parent_folder.creator.outputs.output_band.get_array(
        "kpoints"
    )
    rec_cell = np.linalg.inv(calc.inputs.structure.cell).T  # In units of 2pi/ang
    kpoints_abs = np.matmul(kpoints_rel, rec_cell)
    return np.sqrt((kpoints_abs**2).sum(axis=1)).max()


def get_uniform_grid(n1, n2, n3, shift1=False, shift2=False, shift3=False):
    """Return a uniform grid in relative coordinates, possibly shifted, with values between 0 and 1."""
    dir1 = np.linspace(0, 1, n1, endpoint=False)
    dir2 = np.linspace(0, 1, n2, endpoint=False)
    dir3 = np.linspace(0, 1, n3, endpoint=False)
    if shift1:
        dir1 += 1.0 / n1 / 2.0
    if shift2:
        dir2 += 1.0 / n2 / 2.0
    if shift3:
        dir3 += 1.0 / n3 / 2.0

    dir1v, dir2v, dir3v = np.meshgrid(dir1, dir2, dir3)

    return np.array(list(zip(dir1v.flatten(), dir2v.flatten(), dir3v.flatten())))


def get_predicted_max_kpt(structure):
    """Predict the maximum value of a kpoint length."""
    # I just do a relatively dense grid
    # to estimate a reasonable value for max norm of a kpoint
    kpoints_rel = get_uniform_grid(10, 10, 10)

    # Move values from the range [0,1[ to the range [-0.5, 0.5[
    kpoints_rel = ((kpoints_rel + 0.5) % 1.0) - 0.5
    rec_cell = np.linalg.inv(structure.cell).T  # In units of 2pi/ang
    kpoints_abs = np.matmul(kpoints_rel, rec_cell)
    return np.sqrt((kpoints_abs**2).sum(axis=1)).max()


def predict_smooth_grid(structure, ecutwfc):
    """Predict the smooth grid."""
    max_kpt = get_predicted_max_kpt(structure)

    grid = get_smooth_grid_internal(ecutwfc, max_kpt, structure.cell)
    return grid


def predict_dense_grid(structure, ecutrho):
    """Predict the dense grid."""
    predicted_grid = get_dense_grid_internal(ecutrho, structure.cell)
    return predicted_grid


def compare_grid(calc):
    structure = calc.inputs.structure
    print("**", calc.pk)

    # actual_ecutwfc = calc.inputs.parameters.dict.SYSTEM['ecutwfc']
    # predicted_ecutwfc = get_cutoffs(structure, 'SSSP/v1.1/efficiency/PBE')[0]
    # print("  ACTUAL vs PREDICTED ECUTWFC:", actual_ecutwfc, predicted_ecutwfc)

    actual_max_kpt = get_actual_max_kpt(calc)
    predicted_max_kpt = get_predicted_max_kpt(structure)
    print("  ACTUAL vs PREDICTED MAX_KPT:", actual_max_kpt, predicted_max_kpt)

    actual_grid = calc.res.smooth_fft_grid
    predicted_grid = predict_smooth_grid(structure)
    print("  ACTUAL vs PREDICTED GRID:", actual_grid, predicted_grid)
    return np.prod(predicted_grid) / np.prod(actual_grid)


def test_get_dense_grid_internal():
    cell_ang = np.array(
        [
            [0.0000000000, 2.8400940897, 2.8400940897],
            [2.8400940897, 0.0000000000, 2.8400940897],
            [2.8400940897, 2.8400940897, 0.0000000000],
        ]
    )
    rho_cutoff_ry = 280
    for i in range(20):
        get_dense_grid_internal(rho_cutoff_ry, cell_ang)


if __name__ == "__main__":

    # test_get_dense_grid_internal()

    from aiida.orm import load_node

    pks = [
        142016,
        183805,
        183884,
        184002,
        183926,
        184078,
        183811,
        184023,
        183978,
        183891,
        184296,
        184144,
        183872,
        184493,
        184100,
        183941,
        184121,
        183935,
        184032,
        183914,
        184471,
        184050,
        184396,
        187927,
        184208,
        184373,
        184173,
        198863,
        184535,
        188705,
        184555,
        184365,
        184600,
        184202,
        201441,
        184509,
        184756,
        184447,
        185122,
        184840,
        185046,
        185134,
        184826,
        185036,
        185501,
        184750,
        184895,
        185273,
        184732,
        185017,
        189199,
        184976,
        184561,
        185031,
        185608,
        184772,
        185596,
        184762,
        184948,
        185561,
        185113,
        185642,
        187992,
    ]

    ratios = {}
    for pk in pks:
        ratios[pk] = compare_grid(load_node(pk))
