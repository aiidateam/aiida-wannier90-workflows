# -*- coding: utf-8 -*-
"""Command line interface `aiida-wannier90-workflows`."""
import click
from aiida import orm
from aiida.cmdline.params import arguments
from aiida.cmdline.utils import echo

from .root import cmd_root


@cmd_root.group('node')
def cmd_node():  # pylint: disable=unused-argument
    """Inspect a node."""


@cmd_node.command('show')
@arguments.NODES()
@click.pass_context
def cmd_node_show(ctx, nodes):  # pylint: disable=unused-argument
    """Show info of a node."""
    from aiida.cmdline.commands.cmd_node import node_show

    for node in nodes:
        ctx.invoke(node_show, nodes=nodes, print_groups=False)

        if isinstance(node, orm.RemoteData):
            path = f'{node.get_computer_name()}:{node.get_remote_path()}'
            echo.echo(f'\n{path}')


def get_last_calcjob(workchain: orm.WorkChainNode) -> orm.CalcJobNode:
    """Return the last CalcJob of a WorkChain."""
    calcs = []
    for called_descendant in workchain.called_descendants:
        if not isinstance(called_descendant, orm.CalcJobNode):
            continue
        calcs.append(called_descendant)

    if len(calcs) == 0:
        return None

    # Sort by PK to get latest calcjob
    calcs.sort(key=lambda x: x.pk)
    last_calcjob = calcs[-1]

    return last_calcjob


@cmd_node.command('gotocomputer')
@arguments.NODE()
@click.option(
    '-l',
    '--link-label',
    'link_label',
    type=click.STRING,
    required=False,
    help='Goto the calcjob with this call link label.'
)
@click.pass_context  # pylint: disable=too-many-statements
def cmd_node_gotocomputer(ctx, node, link_label):
    """Open a shell in the remote folder of the calcjob, or the last calcjob of the workflow."""
    import os
    from aiida.common.exceptions import NotExistent
    from aiida.common.links import LinkType
    from aiida.cmdline.commands.cmd_calcjob import calcjob_gotocomputer
    from aiida.plugins import DataFactory

    RemoteData = DataFactory('remote')  # pylint: disable=invalid-name
    RemoteStashFolderData = DataFactory('remote.stash.folder')  # pylint: disable=invalid-name
    FolderData = DataFactory('folder')  # pylint: disable=invalid-name

    echo.echo(f'Node<{node.pk}> type {type(node)}')

    if isinstance(node, orm.CalcJobNode):
        last_calcjob = node
        ctx.invoke(calcjob_gotocomputer, calcjob=last_calcjob)
    elif isinstance(node, orm.WorkChainNode):
        if link_label is None:
            last_calcjob = get_last_calcjob(node)
            if last_calcjob is None:
                echo.echo(f'No CalcJob for {node}?')
                return

            # Get call link label
            link_triples = node.get_outgoing(link_type=(LinkType.CALL_CALC, LinkType.CALL_WORK)).link_triples
            link = list(filter(lambda _: last_calcjob in _.node.called_descendants, link_triples))[0]
            link_label = link.link_label
        else:
            try:
                called = node.get_outgoing(link_label_filter=link_label).one().node
            except ValueError:
                link_triples = node.get_outgoing(link_type=(LinkType.CALL_CALC, LinkType.CALL_WORK)).link_triples
                valid_lables = [x.link_label for x in link_triples]
                valid_lables = '\n'.join(valid_lables)
                echo.echo(f"No nodes found with call link label '{link_label}', valid labels are:")
                echo.echo(f'{valid_lables}')
                return

            if isinstance(called, orm.CalcJobNode):
                last_calcjob = called
            elif isinstance(called, orm.WorkChainNode):
                last_calcjob = get_last_calcjob(called)
            else:
                echo.echo(f'Unsupported type of node: {called}')
                return

        msg = f'Parent WorkChain: {node.process_label}<{node.pk}>\n'
        msg += f' Lastest CalcJob: {last_calcjob.process_label}<{last_calcjob.pk}>\n'
        msg += f' Call link label: {link_label}\n'
        echo.echo(msg)

        ctx.invoke(calcjob_gotocomputer, calcjob=last_calcjob)
    elif isinstance(node, (RemoteData, RemoteStashFolderData)):
        computer = node.computer
        try:
            transport = computer.get_transport()
        except NotExistent as exception:
            echo.echo_critical(repr(exception))

        if isinstance(node, RemoteData):
            remote_workdir = node.get_remote_path()
        elif isinstance(node, RemoteStashFolderData):
            remote_workdir = node.target_basepath

        if not remote_workdir:
            echo.echo_critical('no remote work directory for this node')

        command = transport.gotocomputer_command(remote_workdir)
        echo.echo_info('going to the remote work directory...')
        os.system(command)
        return
    elif isinstance(node, FolderData):
        # Seems FolderData.computer is None
        # I assume the repository is on localhost
        workdir = node._repository._get_base_folder().abspath  # pylint: disable=protected-access
        command = f'cd {workdir}; bash -i'
        echo.echo_info('going to the work directory...')
        os.system(command)
        return
    else:
        echo.echo_critical(f'Unsupported type of node: {type(node)} {node}')


@cmd_node.command('cleanworkdir')
@arguments.NODES()
def cmd_node_clean(nodes):
    """Clean the workdir of CalcJobNode/WorkChainNode."""

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
            echo.echo(f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}")
