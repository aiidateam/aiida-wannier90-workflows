#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-
"""Estimate AMN/MMN/EIG/UNK/CHK file sizes of all the structures in a group."""
from aiida_wannier90_workflows.utils.estimator import (WannierFileFormat, estimate_structure_group, print_estimation)

if __name__ == '__main__':
    # pylint: disable=invalid-name

    structure_group = 'structure/3dcd/experimental'
    hdf_file = 'wannier_storage_estimator.h5'

    estimate_structure_group(structure_group, hdf_file, WannierFileFormat.FORTRAN_FORMATTED)

    print_estimation(hdf_file)
