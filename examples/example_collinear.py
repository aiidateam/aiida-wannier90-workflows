#!/usr/bin/env runaiida
"""Run a ``Wannier90BandsWorkChain`` for Wannier90 band structure.

Usage: ./example_02.py
"""
import click

from aiida import cmdline, orm

from aiida_wannier90_workflows.cli.params import RUN
from aiida_wannier90_workflows.utils.code import check_codes, identify_codes
from aiida_wannier90_workflows.utils.structure import read_structure
from aiida_wannier90_workflows.utils.workflows.builder.serializer import print_builder
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_parallelization
from aiida_wannier90_workflows.utils.workflows.builder.submit import (
    submit_and_add_group,
)
from aiida_wannier90_workflows.utils.workflows.builder.generator.post import (
    get_wannier_builder_from_pwbands,
)
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain, Wannier90WorkChain
from aiida_quantumespresso.common.types import SpinType


def submit(
    codes: dict,
    structure: orm.StructureData,
    group: orm.Group = None,
    run: bool = False,
):
    """Submit a ``Wannier90BandsWorkChain`` to calculate Wannier bands."""
    codes = identify_codes(codes)
    check_codes(codes)

    builder = Wannier90WorkChain.get_builder_from_protocol(
        codes,
        structure,
        protocol="fast",
        initial_magnetic_moments={"Fe":[0,0,3]},
        spin_type=SpinType.COLLINEAR,
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
    # cli()  # pylint: disable=no-value-for-parameter

    from aiida.orm import load_code
    structure=read_structure("./input_files/bcc_Fe.cif")
    codes=["qe-pw","qe-pw2wannier90","wannier90","qe-projwfc"]
    submit(codes, structure,run=True)
    # Run like this:
    # ./example_02.py input_files/GaAs.xsf -X qe-pw@localhost qe-pw2wannier90@localhost wannier90@localhost qe-projwfc@localhost -r
