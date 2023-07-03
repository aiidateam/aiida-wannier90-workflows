#!/usr/bin/env python
"""Run a ``OpenGridCalculation`` on top of a scf calculation.

Usage: ./example_05.py
"""
import click

from aiida import cmdline, orm

from aiida_quantumespresso.calculations.open_grid import OpenGridCalculation

from aiida_wannier90_workflows.cli.params import RUN
from aiida_wannier90_workflows.utils.workflows.builder.serializer import print_builder

# from aiida_wannier90_workflows.utils.workflows.builder.setter import set_parallelization
from aiida_wannier90_workflows.utils.workflows.builder.submit import (
    submit_and_add_group,
)


def submit(
    code: orm.Code,
    parent_folder: orm.RemoteData,
    group: orm.Group = None,
    run: bool = False,
):
    """Submit a ``OpenGridCalculation``."""

    builder = OpenGridCalculation.get_builder()
    builder.code = code
    builder.parent_folder = parent_folder

    # You can change parallelization here
    # parallelization = {
    #     "num_mpiprocs_per_machine": 8,
    #     "npool": 4,
    # }
    # set_parallelization(builder, parallelization, process_class=OpenGridCalculation)

    print_builder(builder)

    if run:
        submit_and_add_group(builder, group)


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.CODE()
@cmdline.params.options.GROUP(help="The group to add the submitted workchain.")
@cmdline.params.arguments.NODE("parent_folder")
@RUN()
def cli(code, group, parent_folder, run):
    """Run a ``OpenGridCalculation`` to unfold irreducible BZ to full BZ.

    PARENT_FOLDER: the RemoteData folder of a scf calculation.
    """
    submit(code, parent_folder, group, run)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
