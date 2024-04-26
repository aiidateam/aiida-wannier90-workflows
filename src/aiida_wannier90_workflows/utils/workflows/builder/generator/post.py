#!/usr/bin/env python
"""Functions to generator PwBands or WannierBands builder from a finished PwBands/WanneirBandsWorkChain.

For "post-processing" of finished WorkChains.
"""
from aiida import orm
from aiida.engine import ProcessBuilder

from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain

from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain


def get_pwbands_builder_from_wannier(
    wannier_workchain: Wannier90BandsWorkChain,
) -> ProcessBuilder:
    """Get a `PwBaseWorkChain` builder for calculating bands strcutre from a finished `Wannier90BandsWorkChain`.

    Useful for comparing QE and Wannier90 interpolated bands structures.

    :param wannier_workchain: [description]
    :type wannier_workchain: Wannier90BandsWorkChain
    """
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

    if not wannier_workchain.is_finished_ok:
        raise ValueError(
            f"The {wannier_workchain.process_label}<{wannier_workchain.pk}> has not finished, "
            f"current status: {wannier_workchain.process_state}, "
            "please retry after workchain has successfully finished."
        )

    scf_inputs = wannier_workchain.inputs["scf"]
    scf_outputs = wannier_workchain.outputs["scf"]

    builder = PwBaseWorkChain.get_builder()

    # wannier_workchain.outputs['scf']['pw'] has no `structure`, I will fill it in later
    excluded_inputs = ["pw"]
    for key in scf_inputs:
        if key in excluded_inputs:
            continue
        builder[key] = scf_inputs[key]

    structure = wannier_workchain.inputs["structure"]
    if "primitive_structure" in wannier_workchain.outputs:
        structure = wannier_workchain.outputs["primitive_structure"]

    pw_inputs = scf_inputs["pw"]
    pw_inputs["structure"] = structure
    builder["pw"] = pw_inputs

    # Should use wannier90 kpath, otherwise number of kpoints
    # of DFT and w90 are not consistent
    wannier_outputs = wannier_workchain.outputs["wannier90"]
    wannier_bands = wannier_outputs["interpolated_bands"]

    wannier_kpoints = orm.KpointsData()
    wannier_kpoints.set_kpoints(wannier_bands.get_kpoints())
    wannier_kpoints.base.attributes.set_many(
        {
            "cell": wannier_bands.base.attributes.all["cell"],
            "pbc1": wannier_bands.base.attributes.all["pbc1"],
            "pbc2": wannier_bands.base.attributes.all["pbc2"],
            "pbc3": wannier_bands.base.attributes.all["pbc3"],
            "labels": wannier_bands.base.attributes.all["labels"],
            # 'array|kpoints': ,
            "label_numbers": wannier_bands.base.attributes.all["label_numbers"],
        }
    )
    builder.kpoints = wannier_kpoints

    builder["pw"]["parent_folder"] = scf_outputs["remote_folder"]

    parameters = builder["pw"]["parameters"].get_dict()
    parameters.setdefault("CONTROL", {})
    parameters.setdefault("SYSTEM", {})
    parameters.setdefault("ELECTRONS", {})
    parameters["CONTROL"]["calculation"] = "bands"
    # parameters['ELECTRONS'].setdefault('diagonalization', 'cg')
    parameters["ELECTRONS"].setdefault("diago_full_acc", True)

    if "nscf" in wannier_workchain.inputs:
        nscf_inputs = wannier_workchain.inputs["nscf"]
        nbnd = nscf_inputs["pw"]["parameters"]["SYSTEM"]["nbnd"]
        parameters["SYSTEM"]["nbnd"] = nbnd

    builder["pw"]["parameters"] = orm.Dict(parameters)

    return builder


def get_wannier_builder_from_pwbands(
    workchain: PwBandsWorkChain, codes: dict
) -> ProcessBuilder:
    """Get a ``Wannier90BandsWorkChain`` builder from a finished ``PwBandsWorkChain``.

    Useful for comparing QE and Wannier90 interpolated bands structures.

    :param workchain: _description_
    :type workchain: PwBandsWorkChain
    :return: _description_
    :rtype: ProcessBuilder
    """
    from aiida.common import LinkType

    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

    from aiida_wannier90_workflows.utils.workflows.bands import (
        get_structure_and_bands_kpoints,
    )

    if workchain.process_class != PwBandsWorkChain:
        raise ValueError(f"Input workchain is not a `PwBandsWorkChain`: {workchain}")

    scf_workchain = (
        workchain.base.links.get_outgoing(
            node_class=PwBaseWorkChain,
            link_type=LinkType.CALL_WORK,
            link_label_filter="scf",
        )
        .one()
        .node
    )

    parent_folder = scf_workchain.outputs.remote_folder
    structure, bands_kpoints = get_structure_and_bands_kpoints(workchain)

    builder = Wannier90BandsWorkChain.get_builder_from_protocol(
        codes,
        structure,
        # This use let wannier90 auto generate kpoints by `bands_num_points`
        # kpoint_path=bands_kpoints,
        # This use ask wannier90 to use kpoints in `explicit_kpath` and `explicit_kpath_labels`
        bands_kpoints=bands_kpoints,
        # pseudo_family=pseudo_family,
    )

    builder.pop("scf", None)
    builder.nscf["pw"]["parent_folder"] = parent_folder

    return builder
