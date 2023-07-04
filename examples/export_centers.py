#!/usr/bin/env runaiida
"""Export a XSF file to visualize Wannier function centers."""
from aiida import orm

from aiida_wannier90_workflows.utils.parser.center import export_wf_centers_to_xyz

# pylint: disable=invalid-name

if __name__ == "__main__":
    # The PK of a `Wannier90Calculation`
    calculation_pk = 123
    # The name of the XSF file to be saved
    xsf_file = f"wannier_centers_{calculation_pk}.xsf"

    calculation = orm.load_node(calculation_pk)

    export_wf_centers_to_xyz(calculation, xsf_file)
