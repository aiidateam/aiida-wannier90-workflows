#!/usr/bin/env python
"""Run a ``PwBandsWorkChain`` for PW band structure.

Usage: ./example_01.py
"""
import click

from aiida import cmdline, orm

from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain

from aiida_wannier90_workflows.cli.params import RUN
from aiida_wannier90_workflows.utils.structure import read_structure
from aiida_wannier90_workflows.utils.workflows.builder.serializer import print_builder
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_parallelization
from aiida_wannier90_workflows.utils.workflows.builder.submit import (
    submit_and_add_group,
)


def submit(
    code: orm.Code,
    structure: orm.StructureData,
    group: orm.Group = None,
    run: bool = False,
):
    """Submit a ``PwBandsWorkChain`` to calculate PW bands."""
    builder = PwBandsWorkChain.get_builder_from_protocol(code, structure=structure)

    # You can change parallelization here
    parallelization = {
        "num_mpiprocs_per_machine": 1,
        "npool": 1,
    }
    set_parallelization(builder, parallelization, process_class=PwBandsWorkChain)

    print_builder(builder)

    if run:
        submit_and_add_group(builder, group)


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.CODE(help="The pw.x code identified by its ID, UUID or label.")
@cmdline.params.options.GROUP(help="The group to add the submitted workchain.")
@click.argument("filename", type=click.Path(exists=True))
@RUN()
def cli(filename, code, group, run):
    """Run a ``PwBandsWorkChain`` to calculate QE band structure.

    FILENAME: a crystal structure file, e.g., ``input_files/GaAs.xsf``.
    """
    struct = read_structure(filename, store=True)
    submit(code, struct, group, run)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
