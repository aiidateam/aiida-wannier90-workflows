"""Commands to manipulate groups of `Wannier90BandsWorkChain`."""

import click

from aiida import orm
from aiida.cmdline.params import types
from aiida.cmdline.utils import decorators, echo
from aiida.tools.query.calculation import CalculationQueryBuilder

from .root import cmd_root


@cmd_root.group("group")
def cmd_group():
    """Manipulate groups of WorkChain."""


@cmd_group.command("movefailed")
@click.argument("src-group", type=types.GroupParamType(), nargs=1)
@click.argument("dest-group", type=types.GroupParamType(), nargs=1)
@decorators.with_dbenv()
@click.pass_context
def cmd_movefailed(ctx, src_group, dest_group):
    """Move failed workchains in source group to destination group.

    src-group: The source group identified by its ID, UUID or label.
    dest-group: The destination group identified by its ID, UUID or label.
    """
    from plumpy import ProcessState

    from aiida.cmdline.commands.cmd_group import group_add_nodes, group_remove_nodes

    from .list import print_process_table

    echo.echo("Workchains failed:", bold=True)
    projected_failed, headers = print_process_table(  # pylint: disable=unused-variable
        process_label=None,
        all_entries=True,
        group=src_group,
        process_state=None,
        paused=False,
        exit_status=None,
        failed=True,
        past_days=None,
        limit=None,
        project=CalculationQueryBuilder.default_projections,
        raw=False,
        order_by="ctime",
        order_dir="asc",
    )

    echo.echo("\nWorkchains excepted or killed:", bold=True)
    projected_excepted, headers = print_process_table(
        process_label=None,
        all_entries=False,
        group=src_group,
        process_state=(ProcessState.EXCEPTED.value, ProcessState.KILLED.value),
        paused=False,
        exit_status=None,
        failed=False,
        past_days=None,
        limit=None,
        project=CalculationQueryBuilder.default_projections,
        raw=False,
        order_by="ctime",
        order_dir="asc",
    )

    # 0th is PK
    nodes = [orm.load_node(_[0]) for _ in projected_failed + projected_excepted]

    echo.echo(
        f"\nWorkchains to be moved: {' '.join([str(_.pk) for _ in nodes])}", bold=True
    )

    message = (
        f"Are you sure you want to move {len(nodes)} nodes from "
        f"{src_group.__class__.__name__}<{src_group.label}> to "
        f"{dest_group.__class__.__name__}<{dest_group.label}>?"
    )
    click.confirm(message, abort=True)

    ctx.invoke(group_remove_nodes, group=src_group, nodes=nodes, force=True)
    ctx.invoke(group_add_nodes, group=dest_group, nodes=nodes, force=True)
