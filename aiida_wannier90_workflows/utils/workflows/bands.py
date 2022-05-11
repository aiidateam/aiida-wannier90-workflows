#!/usr/bin/env python
"""Find PwBandsWorkChain of a StructureData or a corresponding Wannier90BandsWorkChain."""
import typing as ty

from aiida import orm

from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain


def find_pwbands(
    wannier_workchain: Wannier90BandsWorkChain,
) -> ty.List[PwBandsWorkChain]:
    """Find a corresponding PwBaseWorkChain or PwBandsWorkChain for Wannier90BandsWorkChain.

    :param wannier_workchain: [description]
    :type wannier_workchain: [type]
    """
    # First find pwbands with same input structure
    structure = wannier_workchain.inputs["structure"]
    pwbands = find_pwbands_for_structure(structure)

    if len(pwbands) > 0:
        return pwbands

    # Then try to find pwbands from the wannier output primitive_structure
    if "primitive_structure" in wannier_workchain.outputs:
        structure = wannier_workchain.outputs["primitive_structure"]
        pwbands = find_pwbands_for_structure(structure)
        return pwbands

    return []


def find_pwbands_for_structure(
    structure: orm.StructureData,
) -> ty.List[ty.Union[PwBandsWorkChain, PwBaseWorkChain]]:
    """Find a `PwBandsWorkChain` or `PwBaseWorkChain` with a kpath for the specified input `structure`."""
    qb = orm.QueryBuilder()

    qb.append(orm.StructureData, tag="structure", filters={"id": structure.pk})
    qb.append(
        (PwBandsWorkChain, PwBaseWorkChain),
        with_incoming="structure",
        tag="pw_wc",
        filters={"attributes.exit_status": 0},
    )

    pw_workchains = []
    for i in qb.all(flat=True):
        if i.process_class == PwBandsWorkChain:
            pw_workchains.append(i)
        elif i.process_class == PwBaseWorkChain:
            # I only append PwBaseWorkChain which has a high-symmetry kpath
            if "kpoints" in i.inputs and i.inputs["kpoints"] is not None:
                pw_workchains.append(i)

    pw_workchains.sort(key=lambda i: i.pk)

    return pw_workchains


def get_structure_and_bands_kpoints(
    workchain: PwBandsWorkChain,
) -> ty.Tuple[orm.StructureData, orm.KpointsData]:
    """Return the primitive structure and the explicit kpoint path of a PwBandsWorkChain.

    Assuming the PwBandsWorkChain runs a seekpath internally.

    :param workchain: [description]
    :type workchain: PwBandsWorkChain
    :return: [description]
    :rtype: ty.Tuple[orm.StructureData, orm.KpointsData]
    """
    from aiida.common.links import LinkType

    # internal PwBaseWorkChain for pw.x bands calculation
    bands_workchain = (
        workchain.get_outgoing(
            node_class=PwBaseWorkChain,
            link_type=LinkType.CALL_WORK,
            link_label_filter="bands",
        )
        .one()
        .node
    )

    structure = bands_workchain.inputs.pw.structure
    bands_kpoints = bands_workchain.inputs.kpoints

    return structure, bands_kpoints
