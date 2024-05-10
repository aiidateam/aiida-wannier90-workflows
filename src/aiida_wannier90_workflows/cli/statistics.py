"""Commands to manipulate groups of `Wannier90BandsWorkChain`."""

import click

from aiida.cmdline.params import types
from aiida.cmdline.utils import decorators, echo

from .root import cmd_root


@cmd_root.group("statistics")
def cmd_statistics():
    """Collect statistics of WorkChain."""


@cmd_statistics.command("optimize")
@click.argument("group", type=types.GroupParamType(), nargs=1)
@decorators.with_dbenv()
def cmd_optimize(group):
    """Show number of optimization and bands distance of `Wannier90OptimizeWorkChain` in a group."""
    from tabulate import tabulate

    from aiida.common import LinkType

    from aiida_wannier90_workflows.utils.workflows import get_last_calcjob
    from aiida_wannier90_workflows.workflows.optimize import (
        Wannier90OptimizeWorkChain,
        get_bands_distance_ef2,
    )

    def get_minmax(workchain):
        base_calc = get_last_calcjob(workchain)
        base_params = base_calc.inputs.parameters.get_dict()
        dis_proj_min = base_params["dis_proj_min"]
        dis_proj_max = base_params["dis_proj_max"]
        return dis_proj_min, dis_proj_max

    data = []
    for workchain in group.nodes:
        if workchain.process_class != Wannier90OptimizeWorkChain:
            echo.echo(
                f"{workchain.process_label}<{workchain.pk}> is not {Wannier90OptimizeWorkChain.__class__.__name__}"
            )
            continue
        if workchain.exit_status != 0:
            echo.echo(
                f"{workchain.process_label}<{workchain.pk}> failed with exit_status {workchain.exit_status}"
            )
            continue

        entry = [workchain.pk, workchain.inputs.structure.get_formula()]

        optimize_workchains = [
            _.node
            for _ in workchain.base.links.get_outgoing(
                link_type=LinkType.CALL_WORK,
                link_label_filter="wannier90_optimize_iteration%",
            ).all()
        ]
        num_optimization = len(optimize_workchains)
        entry.append(num_optimization)

        base_workchain = (
            workchain.base.links.get_outgoing(
                link_type=LinkType.CALL_WORK, link_label_filter="wannier90"
            )
            .one()
            .node
        )
        dis_proj_min, dis_proj_max = get_minmax(base_workchain)
        entry.append(dis_proj_min)
        entry.append(dis_proj_max)

        ref_bands = workchain.inputs["optimize_reference_bands"]
        bandsdist = get_bands_distance_ef2(ref_bands, base_workchain)
        entry.append(bandsdist)

        data.append(entry)

        for base_workchain in optimize_workchains:
            entry = [None, None, None]
            dis_proj_min, dis_proj_max = get_minmax(base_workchain)
            entry.append(dis_proj_min)
            entry.append(dis_proj_max)

            bandsdist = get_bands_distance_ef2(ref_bands, base_workchain)
            entry.append(bandsdist)

            data.append(entry)

    headers = [
        "PK",
        "structure",
        "num_optimization",
        "dis_proj_min",
        "dis_proj_max",
        "bands distance",
    ]
    tabulated = tabulate(data, headers=headers)
    echo.echo(tabulated)
