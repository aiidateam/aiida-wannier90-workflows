#!/usr/bin/env python
"""Run a ``PwBaseWorkChain`` for PW band structure, restarting from a ``Wannier90BandsWorkChain``.

Usage: ./example_03.py
"""
import click

from aiida import cmdline, orm

from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain

from aiida_wannier90_workflows.cli.params import RUN, FilteredWorkflowParamType
from aiida_wannier90_workflows.utils.workflows.builder.generator.post import (
    get_pwbands_builder_from_wannier,
)
from aiida_wannier90_workflows.utils.workflows.builder.serializer import print_builder
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_parallelization
from aiida_wannier90_workflows.utils.workflows.builder.submit import (
    submit_and_add_group,
)
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain


def submit(
    w90_wkchain: Wannier90BandsWorkChain,
    group: orm.Group = None,
    run: bool = False,
):
    """Submit a ``PwBaseWorkChain`` to calculate PW bands.

    Load a finished ``Wannier90BandsWorkChain``, and reuse the scf calculation.
    """
    # w90_wkchain = orm.load_node(139623)
    builder = get_pwbands_builder_from_wannier(w90_wkchain)

    # You can change parallelization here
    parallelization = {
        "num_mpiprocs_per_machine": 16,
        "npool": 8,
    }
    set_parallelization(builder, parallelization, process_class=PwBaseWorkChain)

    print_builder(builder)

    if run:
        submit_and_add_group(builder, group)


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.arguments.WORKFLOW(
    type=FilteredWorkflowParamType(
        process_classes=(
            "aiida.workflows:wannier90_workflows.bands",
            "aiida.workflows:wannier90_workflows.optimize",
        )
    ),
)
@cmdline.params.options.GROUP(help="The group to add the submitted workchain.")
@RUN()
def cli(workflow, group, run):
    """Run a ``PwBaseWorkChain`` for PW band structure.

    Reuse the scf calculation from a finished Wannier90BandsWorkChain.
    """
    submit(workflow, group, run)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
