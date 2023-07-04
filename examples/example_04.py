#!/usr/bin/env python
"""Run a ``Wannier90BandsWorkChain`` restarting from a ``PwBandsWorkChain``.

Usage: ./example_04.py
"""
import click

from aiida import cmdline, orm

from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain

from aiida_wannier90_workflows.cli.params import RUN, FilteredWorkflowParamType
from aiida_wannier90_workflows.utils.code import check_codes, identify_codes
from aiida_wannier90_workflows.utils.workflows.builder.generator.post import (
    get_wannier_builder_from_pwbands,
)
from aiida_wannier90_workflows.utils.workflows.builder.serializer import print_builder
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_parallelization
from aiida_wannier90_workflows.utils.workflows.builder.submit import (
    submit_and_add_group,
)
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain


def submit(
    pwbands_wkchain: PwBandsWorkChain,
    codes: dict,
    group: orm.Group = None,
    run: bool = False,
):
    """Submit a ``Wannier90BandsWorkChain`` to calculate Wannier bands.

    Load a finished ``PwBandsWorkChain``, and reuse the scf calculation.
    """
    codes = identify_codes(codes)
    check_codes(codes)

    # pwbands_wkchain = orm.load_node(139623)
    builder = get_wannier_builder_from_pwbands(pwbands_wkchain, codes)

    # You can change parallelization here
    parallelization = {
        "num_mpiprocs_per_machine": 16,
        "npool": 8,
    }
    set_parallelization(builder, parallelization, process_class=Wannier90BandsWorkChain)

    print_builder(builder)

    if run:
        submit_and_add_group(builder, group)


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.arguments.WORKFLOW(
    type=FilteredWorkflowParamType(
        process_classes=("aiida.workflows:quantumespresso.pw.bands",)
    ),
)
@cmdline.params.options.CODES()
@cmdline.params.options.GROUP(help="The group to add the submitted workchain.")
@RUN()
def cli(workflow, codes, group, run):
    """Run a ``Wannier90BandsWorkChain`` restarting from a ``PwBandsWorkChain``.

    Reuse the scf calculation from a finished ``PwBandsWorkChain``.
    """
    submit(codes, workflow, group, run)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
