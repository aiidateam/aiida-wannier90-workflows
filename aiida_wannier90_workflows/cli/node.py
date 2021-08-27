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
