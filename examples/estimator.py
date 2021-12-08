#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-
"""Estimate AMN/MMN/EIG/UNK/CHK file sizes of all the structures in a group."""
from aiida_wannier90_workflows.utils.workflows.estimator import (
    WannierFileFormat, estimate_structure_group, print_estimation, plot_histogram_hamiltonian
)
from aiida_wannier90_workflows.utils.bands.distance import standardize_groupname

# pylint: disable=invalid-name

if __name__ == '__main__':

    structure_group = 'structure/3dcd/experimental'
    hdf_file = f'estimator-{standardize_groupname(structure_group)}.h5'

    estimate_structure_group(structure_group, hdf_file, WannierFileFormat.FORTRAN_UNFORMATTED)

    print_estimation(hdf_file)

    plot_histogram_hamiltonian(hdf_file)
