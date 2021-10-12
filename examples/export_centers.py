#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-
"""Export a XSF file to visualize Wannier function centers."""
from aiida import orm
from aiida_wannier90_workflows.utils.center import export_centers_xsf

if __name__ == '__main__':
    # pylint: disable=invalid-name

    # The PK of a `Wannier90Calculation`
    calculation_pk = 123
    # The name of the XSF file to be saved
    xsf_file = f'wannier_centers_{calculation_pk}.xsf'

    calculation = orm.load_node(calculation_pk)

    export_centers_xsf(calculation, xsf_file)
