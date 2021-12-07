#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions to generator relax/scf/nscf builder."""
from aiida import orm
from aiida.engine import ProcessBuilder

from aiida_quantumespresso.common.types import SpinType

from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain


def get_relax_builder(
    code: orm.Code,
    kpoints_distance: float = None,
    pseudo_family: str = None,
    spin_type: SpinType = SpinType.NONE,
    clean_workdir: bool = True,
    **kwargs
) -> ProcessBuilder:
    """Generate a `PwRelaxWorkChain` builder for SOC or non-SOC."""
    from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

    overrides = kwargs.get('overrides', {})
    if clean_workdir:
        overrides['clean_workdir'] = clean_workdir
    if kpoints_distance:
        overrides.setdefault('base', {})
        overrides['base']['kpoints_distance'] = kpoints_distance
    if pseudo_family:
        overrides.setdefault('base', {})
        overrides['base']['pseudo_family'] = pseudo_family

    # PwBaseWorkChain.get_builder_from_protocol() does not support SOC, I have to
    # pretend that I am doing an non-SOC calculation and add SOC parameters later.
    if spin_type == SpinType.SPIN_ORBIT:
        kwargs['spin_type'] = SpinType.NONE
        if pseudo_family is None:
            raise ValueError('`pseudo_family` must be explicitly set for SOC')

    builder = PwRelaxWorkChain.get_builder_from_protocol(code=code, overrides=overrides, **kwargs)

    parameters = builder.base['pw']['parameters'].get_dict()
    if spin_type == SpinType.NON_COLLINEAR:
        parameters['SYSTEM']['noncolin'] = True
    if spin_type == SpinType.SPIN_ORBIT:
        parameters['SYSTEM']['noncolin'] = True
        parameters['SYSTEM']['lspinorb'] = True
    builder.base['pw']['parameters'] = orm.Dict(dict=parameters)

    return builder


def get_scf_builder(
    code: orm.Code,
    kpoints_distance: float = None,
    pseudo_family: str = None,
    spin_type: SpinType = SpinType.NONE,
    clean_workdir: bool = True,
    **kwargs
) -> ProcessBuilder:
    """Generate a `PwBaseWorkChain` builder for scf, with or without SOC."""
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

    overrides = kwargs.pop('overrides', {})
    if clean_workdir:
        overrides['clean_workdir'] = clean_workdir
    if kpoints_distance:
        overrides['kpoints_distance'] = kpoints_distance
    if pseudo_family:
        overrides['pseudo_family'] = pseudo_family

    # PwBaseWorkChain.get_builder_from_protocol() does not support SOC, I have to
    # pretend that I am doing an non-SOC calculation and add SOC parameters later.
    if spin_type == SpinType.SPIN_ORBIT:
        kwargs['spin_type'] = SpinType.NONE
        if pseudo_family is None:
            raise ValueError('`pseudo_family` must be explicitly set for SOC')

    builder = PwBaseWorkChain.get_builder_from_protocol(code=code, overrides=overrides, **kwargs)

    parameters = builder['pw']['parameters'].get_dict()
    if spin_type == SpinType.NON_COLLINEAR:
        parameters['SYSTEM']['noncolin'] = True
    if spin_type == SpinType.SPIN_ORBIT:
        parameters['SYSTEM']['noncolin'] = True
        parameters['SYSTEM']['lspinorb'] = True
    builder['pw']['parameters'] = orm.Dict(dict=parameters)

    # Currently only support magnetic with SOC
    # for magnetic w/o SOC, needs 2 separate wannier90 calculations for spin up and down.
    # if self.inputs.spin_polarized and self.inputs.spin_orbit_coupling:
    #     # Magnetization from Kittel, unit: Bohr magneton
    #     magnetizations = {'Fe': 2.22, 'Co': 1.72, 'Ni': 0.606}
    #     from aiida_wannier90_workflows.utils.upf import get_number_of_electrons_from_upf
    #     for i, kind in enumerate(self.inputs.structure.kinds):
    #         if kind.name in magnetizations:
    #             zvalence = get_number_of_electrons_from_upf(
    #                 self.ctx.pseudos[kind.name]
    #             )
    #             spin_polarization = magnetizations[kind.name] / zvalence
    #             pw_parameters['SYSTEM'][f"starting_magnetization({i+1})"
    #                                     ] = spin_polarization

    return builder


def get_nscf_builder(
    code: orm.Code,
    kpoints_distance: float = None,
    kpoints: orm.KpointsData = None,
    nbnd: int = None,
    pseudo_family: str = None,
    spin_type: SpinType = SpinType.NONE,
    clean_workdir: bool = True,
    **kwargs
) -> ProcessBuilder:
    """Generate a `PwBaseWorkChain` builder for nscf, with or without SOC."""

    if kpoints and kpoints_distance:
        raise ValueError('Cannot accept both `kpoints` and `kpoints_distance`')

    builder = get_scf_builder(
        code=code,
        kpoints_distance=kpoints_distance,
        pseudo_family=pseudo_family,
        spin_type=spin_type,
        clean_workdir=clean_workdir,
        **kwargs
    )

    parameters = builder['pw']['parameters'].get_dict()

    parameters['SYSTEM']['nbnd'] = nbnd

    parameters['SYSTEM']['nosym'] = True
    parameters['SYSTEM']['noinv'] = True

    parameters['CONTROL']['calculation'] = 'nscf'
    parameters['CONTROL']['restart_mode'] = 'from_scratch'
    parameters['ELECTRONS']['startingpot'] = 'file'
    # I switched to the QE default `david` diagonalization, since now
    # aiida-qe has an error handler to switch to `cg` if `david` fails.
    # See https://github.com/aiidateam/aiida-quantumespresso/pull/744
    # parameters['ELECTRONS']['diagonalization'] = 'david'
    parameters['ELECTRONS']['diago_full_acc'] = True

    builder['pw']['parameters'] = orm.Dict(dict=parameters)

    if kpoints:
        builder.pop('kpoints_distance', None)
        builder['kpoints'] = kpoints

    return builder


def get_pwbands_builder(wannier_workchain: Wannier90BandsWorkChain) -> ProcessBuilder:
    """Get a `PwBaseWorkChain` builder for calculating bands strcutre from a finished `Wannier90BandsWorkChain`.

    Useful for comparing QE and Wannier90 interpolated bands structures.

    :param wannier_workchain: [description]
    :type wannier_workchain: Wannier90BandsWorkChain
    """
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

    if not wannier_workchain.is_finished_ok:
        print(
            f'The {wannier_workchain.process_label}<{wannier_workchain.pk}> has not finished, '
            f'current status: {wannier_workchain.process_state}, '
            'please retry after workchain has successfully finished.'
        )
        return

    scf_inputs = wannier_workchain.inputs['scf']
    scf_outputs = wannier_workchain.outputs['scf']

    builder = PwBaseWorkChain.get_builder()

    # wannier_workchain.outputs['scf']['pw'] has no `structure`, I will fill it in later
    excluded_inputs = ['pw']
    for key in scf_inputs:
        if key in excluded_inputs:
            continue
        builder[key] = scf_inputs[key]

    structure = wannier_workchain.inputs['structure']
    if 'primitive_structure' in wannier_workchain.outputs:
        structure = wannier_workchain.outputs['primitive_structure']

    pw_inputs = scf_inputs['pw']
    pw_inputs['structure'] = structure
    builder['pw'] = pw_inputs

    # Should use wannier90 kpath, otherwise number of kpoints
    # of DFT and w90 are not consistent
    wannier_outputs = wannier_workchain.outputs['wannier90']
    wannier_bands = wannier_outputs['interpolated_bands']

    wannier_kpoints = orm.KpointsData()
    wannier_kpoints.set_kpoints(wannier_bands.get_kpoints())
    wannier_kpoints.set_attribute_many({
        'cell': wannier_bands.attributes['cell'],
        'pbc1': wannier_bands.attributes['pbc1'],
        'pbc2': wannier_bands.attributes['pbc2'],
        'pbc3': wannier_bands.attributes['pbc3'],
        'labels': wannier_bands.attributes['labels'],
        # 'array|kpoints': ,
        'label_numbers': wannier_bands.attributes['label_numbers']
    })
    builder.kpoints = wannier_kpoints

    builder['pw']['parent_folder'] = scf_outputs['remote_folder']

    parameters = builder['pw']['parameters'].get_dict()
    parameters.setdefault('CONTROL', {})
    parameters.setdefault('SYSTEM', {})
    parameters.setdefault('ELECTRONS', {})
    parameters['CONTROL']['calculation'] = 'bands'
    # parameters['ELECTRONS'].setdefault('diagonalization', 'cg')
    parameters['ELECTRONS'].setdefault('diago_full_acc', True)

    if 'nscf' in wannier_workchain.inputs:
        nscf_inputs = wannier_workchain.inputs['nscf']
        nbnd = nscf_inputs['pw']['parameters']['SYSTEM']['nbnd']
        parameters['SYSTEM']['nbnd'] = nbnd

    builder['pw']['parameters'] = orm.Dict(dict=parameters)

    return builder
