#!/usr/bin/env python
"""Run a ``Wannier90BandsWorkChain`` for Wannier90 band structure.

Usage: ./example_02.py
"""
import click

from aiida import cmdline, orm

from aiida_wannier90_workflows.cli.params import RUN
from aiida_wannier90_workflows.utils.code import check_codes, identify_codes
from aiida_wannier90_workflows.utils.structure import read_structure
from aiida_wannier90_workflows.utils.workflows.builder import (
    print_builder,
    set_parallelization,
    submit_and_add_group,
)
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain


def submit(
    codes: dict,
    structure: orm.StructureData,
    group: orm.Group = None,
    run: bool = False,
):
    """Submit a ``Wannier90BandsWorkChain`` to calculate Wannier bands."""
    codes = identify_codes(codes)
    check_codes(codes)

    builder = Wannier90BandsWorkChain.get_builder_from_protocol(
        codes,
        structure,
        protocol="fast",
    )

    # You can change parallelization here
    parallelization = {
        "num_mpiprocs_per_machine": 1,
        "npool": 1,
    }
    set_parallelization(builder, parallelization, process_class=Wannier90BandsWorkChain)

    print_builder(builder)

    if run:
        submit_and_add_group(builder, group)


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.CODES()
@cmdline.params.options.GROUP(help="The group to add the submitted workchain.")
@click.argument("filename", type=click.Path(exists=True))
@RUN()
def cli(filename, codes, group, run):
    """Run a ``Wannier90BandsWorkChain`` to calculate Wannier90 band structure.

    FILENAME: a crystal structure file, e.g., ``input_files/GaAs.xsf``.
    """
    struct = read_structure(filename, store=True)
    submit(codes, struct, group, run)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
