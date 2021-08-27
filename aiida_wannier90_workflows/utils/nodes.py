#!/usr/bin/env python
from aiida import orm
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain

def find_pwbands(wannier_workchain):
    """Find a corresponding PwBaseWorkChain or PwBandsWorkChain for Wannier90BandsWorkChain.

    :param wannier_workchain: [description]
    :type wannier_workchain: [type]
    """
    qb = orm.QueryBuilder()

    if 'primitive_structure' in wannier_workchain.outputs:
        structure = wannier_workchain.outputs['primitive_structure']
    else:
        structure = wannier_workchain.inputs['structure']

    qb.append(
        orm.StructureData, tag='structure', filters={'id': structure.pk}
    )
    qb.append((PwBaseWorkChain, PwBandsWorkChain),
                with_incoming='structure',
                tag='pw_wc',
                filters={'attributes.exit_status': 0})

    pw_workchains = [i[0] for i in qb.all()]
    pw_workchains.sort(key=lambda i: i.pk)

    return pw_workchains
