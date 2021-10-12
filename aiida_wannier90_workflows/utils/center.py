#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Processing the Wannier centers."""
from aiida_wannier90.calculations import Wannier90Calculation


def export_centers_xsf(calculation: Wannier90Calculation, filename: str = 'wannier_centers.xsf'):
    """Export a XSF file to visualize Wannier function centers.

    :param calculation: [description]
    :type calculation: Wannier90Calculation
    """
    import ase

    structure = calculation.inputs.structure.get_ase()
    new_structure = structure.copy()

    wannier_functions_output = calculation.outputs.output_parameters.get_dict()['wannier_functions_output']
    coordinates = [_['coordinates'] for _ in wannier_functions_output]

    for coord in coordinates:
        new_structure.append(ase.Atom('X', coord))

    new_structure.write(filename)
