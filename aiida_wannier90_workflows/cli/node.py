# -*- coding: utf-8 -*-
"""Command line interface `aiida-wannier90-workflows`."""
import click
from aiida import orm
from aiida.cmdline.params import arguments, options, types
from aiida.cmdline.utils import decorators, echo

from .root import cmd_root


@cmd_root.group('node')
def cmd_node():  # pylint: disable=unused-argument
    """Inspect a node"""


@cmd_node.command('show')
@arguments.NODES()
@click.pass_context
def cmd_node_show(ctx, nodes):  # pylint: disable=unused-argument
    """Show info of a node"""
    from aiida.cmdline.commands.cmd_node import node_show

    for node in nodes:
        ctx.invoke(node_show, nodes=nodes, print_groups=False)

        if isinstance(node, orm.RemoteData):
            path = f'{node.get_computer_name()}:{node.get_remote_path()}'
            echo.echo(f'\n{path}')


@cmd_node.command('gotocomputer')
@arguments.NODE()
@click.pass_context
def cmd_node_gotocomputer(ctx, node):
    """Open a shell in the remote folder of the calcjob, or the last calcjob of the workflow"""
    from aiida.cmdline.commands.cmd_calcjob import calcjob_gotocomputer

    if isinstance(node, orm.CalcJobNode):
        last_calcjob = node
    elif isinstance(node, orm.WorkChainNode):
        calcs = []
        for called_descendant in node.called_descendants:
            if not isinstance(called_descendant, orm.CalcJobNode):
                continue
            calcs.append(called_descendant)
        # Sort by PK to get latest calcjob
        calcs.sort(key=lambda x: x.pk)
        last_calcjob = calcs[-1]

        # Get call link label
        link_triples = node.get_outgoing().link_triples
        link = list(filter(lambda x: x.node == last_calcjob, link_triples))[0]
        link_label = link.link_label
        msg = f"Parent WorkChain: {node.process_label}<{node.pk}>\n"
        msg += f" Lastest CalcJob: {last_calcjob.process_label}<{last_calcjob.pk}>\n"
        msg += f" Call link label: {link_label}\n"
        echo.echo(msg)
    else:
        echo.echo(f'Unsupported type of node: {node}')

    ctx.invoke(calcjob_gotocomputer, calcjob=last_calcjob)


@cmd_node.command('cleanworkdir')
@arguments.NODES()
def cmd_node_clean(nodes):
    """Clean the workdir of CalcJobNode/WorkChainNode"""

    for node in nodes:
        calcs = []
        if isinstance(node, orm.CalcJobNode):
            calcs.append(node)
        elif isinstance(node, orm.WorkChainNode):
            for called_descendant in node.called_descendants:
                if not isinstance(called_descendant, orm.CalcJobNode):
                    continue
                calcs.append(called_descendant)
        else:
            echo.echo(f'Unsupported type of node: {node}')

        cleaned_calcs = []
        for calc in calcs:
            try:
                calc.outputs.remote_folder._clean()  # pylint: disable=protected-access
                cleaned_calcs.append(calc.pk)
            except (IOError, OSError, KeyError):
                pass
        if cleaned_calcs:
            echo.echo(
                f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
            )
