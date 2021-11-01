#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-
"""Estimate AMN/MMN/EIG/UNK/CHK file sizes of all the structures in a group."""
from aiida_wannier90_workflows.utils.estimator import (WannierFileFormat, estimate_structure_group, print_estimation, plot_histogram_hamiltonian)
from aiida_wannier90_workflows.utils.bandsdist import standardize_groupname

if __name__ == '__main__':
    # pylint: disable=invalid-name

    structure_group = 'structure/3dcd/experimental'
    hdf_file = f'estimator-{standardize_groupname(structure_group)}.h5'

    estimate_structure_group(structure_group, hdf_file, WannierFileFormat.FORTRAN_UNFORMATTED)

    print_estimation(hdf_file)

    plot_histogram_hamiltonian(hdf_file)
