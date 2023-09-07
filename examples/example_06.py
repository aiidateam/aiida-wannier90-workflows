#!/usr/bin/env python
"""Run a ``Wannier90OptimizeWorkChain`` on top of a PwBandsWorkChain.

Usage: ./example_06.py
"""
import click

from aiida import cmdline, orm

from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain

from aiida_wannier90_workflows.cli.params import RUN
from aiida_wannier90_workflows.utils.workflows.builder.serializer import print_builder
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_parallelization
from aiida_wannier90_workflows.utils.workflows.builder.submit import (
    submit_and_add_group,
)
from aiida_wannier90_workflows.workflows.optimize import Wannier90OptimizeWorkChain


def submit(
    codes: dict[orm.Code],
    pwbands_workchain: PwBandsWorkChain,
    group: orm.Group = None,
    run: bool = False,
):
    """Submit a ``Wannier90OptimizeWorkChain``."""
    scf_calc = pwbands_workchain.outputs.scf_parameters.creator
    parent_folder = scf_calc.outputs.remote_folder
    structure = scf_calc.inputs.structure
    reference_bands = pwbands_workchain.outputs.band_structure
    bands_kpoints = reference_bands.creator.inputs.kpoints
    builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
        codes,
        structure,
        reference_bands=reference_bands,
        bands_kpoints=bands_kpoints,
    )
    builder.pop("scf")
    builder.nscf.pw.parent_folder = parent_folder

    # You can change parallelization here
    parallelization = {
        "num_mpiprocs_per_machine": 8,
        "npool": 4,
    }
    set_parallelization(
        builder, parallelization, process_class=Wannier90OptimizeWorkChain
    )

    print_builder(builder)

    if run:
        submit_and_add_group(builder, group)


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.GROUP(help="The group to add the submitted workchain.")
@cmdline.params.arguments.NODE("pwbands_workchain")
@RUN()
def cli(group, pwbands_workchain, run):
    """Run a ``OpenGridCalculation`` to unfold irreducible BZ to full BZ.

    PARENT_FOLDER: the RemoteData folder of a scf calculation.
    """
    codes = {
        "pw": "qe-git-pw@localhost",
        "pw2wannier90": "qe-git-pw2wannier90@localhost",
        "wannier90": "wannier90-git@localhost",
    }
    # pwbands_workchain = orm.load_node(126896)
    submit(codes, pwbands_workchain, group, run)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
