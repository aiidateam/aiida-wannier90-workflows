#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Find PwBandsWorkChain of a StructureData or a corresponding Wannier90BandsWorkChain."""
from aiida import orm
from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain


def find_pwbands(wannier_workchain: Wannier90BandsWorkChain) -> list[PwBandsWorkChain]:
    """Find a corresponding PwBaseWorkChain or PwBandsWorkChain for Wannier90BandsWorkChain.

    :param wannier_workchain: [description]
    :type wannier_workchain: [type]
    """
    # First find pwbands with same input structure
    structure = wannier_workchain.inputs['structure']
    pwbands = find_pwbands_for_structure(structure)

    if len(pwbands) > 0:
        return pwbands

    # Then try to find pwbands from the wannier output primitive_structure
    if 'primitive_structure' in wannier_workchain.outputs:
        structure = wannier_workchain.outputs['primitive_structure']
        pwbands = find_pwbands_for_structure(structure)
        return pwbands

    return []


def find_pwbands_for_structure(structure: orm.StructureData) -> list[PwBandsWorkChain]:
    """Find a `PwBandsWorkChain` with the specified input `structure`."""
    qb = orm.QueryBuilder()

    qb.append(orm.StructureData, tag='structure', filters={'id': structure.pk})
    qb.append(PwBandsWorkChain, with_incoming='structure', tag='pw_wc', filters={'attributes.exit_status': 0})

    pw_workchains = [i[0] for i in qb.all()]
    pw_workchains.sort(key=lambda i: i.pk)

    return pw_workchains
