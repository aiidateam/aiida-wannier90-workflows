#!/usr/bin/env python
"""Functions to generator relax/scf/nscf builder."""
from aiida import orm
from aiida.engine import ProcessBuilder

from aiida_quantumespresso.common.types import SpinType


def get_relax_builder(
    code: orm.Code,
    kpoints_distance: float = None,
    pseudo_family: str = None,
    spin_type: SpinType = SpinType.NONE,
    clean_workdir: bool = True,
    **kwargs,
) -> ProcessBuilder:
    """Generate a `PwRelaxWorkChain` builder for SOC or non-SOC."""
    from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

    overrides = kwargs.get("overrides", {})
    if clean_workdir:
        overrides["clean_workdir"] = clean_workdir
    if kpoints_distance:
        overrides.setdefault("base", {})
        overrides["base"]["kpoints_distance"] = kpoints_distance
    if pseudo_family:
        overrides.setdefault("base", {})
        overrides["base"]["pseudo_family"] = pseudo_family

    # PwBaseWorkChain.get_builder_from_protocol() does not support SOC, I have to
    # pretend that I am doing an non-SOC calculation and add SOC parameters later.
    if spin_type == SpinType.SPIN_ORBIT:
        kwargs["spin_type"] = SpinType.NONE
        if pseudo_family is None:
            raise ValueError("`pseudo_family` must be explicitly set for SOC")

    builder = PwRelaxWorkChain.get_builder_from_protocol(
        code=code, overrides=overrides, **kwargs
    )

    parameters = builder.base["pw"]["parameters"].get_dict()

    if spin_type == SpinType.SPIN_ORBIT:
        parameters["SYSTEM"]["noncolin"] = True
        parameters["SYSTEM"]["lspinorb"] = True
    builder.base["pw"]["parameters"] = orm.Dict(parameters)

    return builder


def get_scf_builder(
    code: orm.Code,
    kpoints_distance: float = None,
    pseudo_family: str = None,
    spin_type: SpinType = SpinType.NONE,
    clean_workdir: bool = True,
    **kwargs,
) -> ProcessBuilder:
    """Generate a `PwBaseWorkChain` builder for scf, with or without SOC."""
    from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

    overrides = kwargs.pop("overrides", {})
    if clean_workdir:
        overrides["clean_workdir"] = clean_workdir
    if kpoints_distance:
        overrides["kpoints_distance"] = kpoints_distance
    if pseudo_family:
        overrides["pseudo_family"] = pseudo_family

    # PwBaseWorkChain.get_builder_from_protocol() does not support SOC, I have to
    # pretend that I am doing an non-SOC calculation and add SOC parameters later.
    if spin_type == SpinType.SPIN_ORBIT:
        kwargs["spin_type"] = SpinType.NONE
        if pseudo_family is None:
            raise ValueError("`pseudo_family` must be explicitly set for SOC")

    builder = PwBaseWorkChain.get_builder_from_protocol(
        code=code, overrides=overrides, **kwargs
    )

    parameters = builder["pw"]["parameters"].get_dict()
    if spin_type == SpinType.SPIN_ORBIT:
        parameters["SYSTEM"]["noncolin"] = True
        parameters["SYSTEM"]["lspinorb"] = True
    builder["pw"]["parameters"] = orm.Dict(parameters)

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


def get_nscf_builder(  # pylint: disable=too-many-arguments
    code: orm.Code,
    kpoints_distance: float = None,
    kpoints: orm.KpointsData = None,
    nbnd: int = None,
    pseudo_family: str = None,
    spin_type: SpinType = SpinType.NONE,
    clean_workdir: bool = True,
    **kwargs,
) -> ProcessBuilder:
    """Generate a `PwBaseWorkChain` builder for nscf, with or without SOC."""

    if kpoints and kpoints_distance:
        raise ValueError("Cannot accept both `kpoints` and `kpoints_distance`")

    builder = get_scf_builder(
        code=code,
        kpoints_distance=kpoints_distance,
        pseudo_family=pseudo_family,
        spin_type=spin_type,
        clean_workdir=clean_workdir,
        **kwargs,
    )

    parameters = builder["pw"]["parameters"].get_dict()

    parameters["SYSTEM"]["nbnd"] = nbnd

    parameters["SYSTEM"]["nosym"] = True
    parameters["SYSTEM"]["noinv"] = True

    parameters["CONTROL"]["calculation"] = "nscf"
    parameters["CONTROL"]["restart_mode"] = "from_scratch"
    parameters["ELECTRONS"]["startingpot"] = "file"
    # I switched to the QE default `david` diagonalization, since now
    # aiida-qe has an error handler to switch to `cg` if `david` fails.
    # See https://github.com/aiidateam/aiida-quantumespresso/pull/744
    # parameters['ELECTRONS']['diagonalization'] = 'david'
    parameters["ELECTRONS"]["diago_full_acc"] = True

    builder["pw"]["parameters"] = orm.Dict(parameters)

    if kpoints:
        builder.pop("kpoints_distance", None)
        builder["kpoints"] = kpoints

    return builder
