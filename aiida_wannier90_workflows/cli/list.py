# -*- coding: utf-8 -*-
"""Commands to list instances of `PseudoPotentialFamily`."""
import click
from aiida.cmdline.params import options as options_core
from aiida.cmdline.utils import decorators, echo

from .root import cmd_root

PROJECTIONS_VALID = ('pk', 'ctime', 'process_state', 'process_status', 'exit_status', 'process_label', 'process_type')
PROJECTIONS_DEFAULT = ('pk', 'ctime', 'process_state', 'process_status', 'exit_status')

def get_workchains_builder():
    """Return a query builder that will query for instances of `Wannier90BandsWorkChain` or its subclasses.

    :return: `QueryBuilder` instance
    """
    from aiida.orm import QueryBuilder
    from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain

    builder = QueryBuilder().append(Wannier90BandsWorkChain)

    return builder


@cmd_root.command('list')
@options_core.PROJECT(type=click.Choice(PROJECTIONS_VALID), default=PROJECTIONS_DEFAULT)
@options_core.RAW()
@decorators.with_dbenv()
def cmd_list(project, raw):
    """List all instances of `Wannier90BandsWorkChain`."""
    from tabulate import tabulate

    if get_workchains_builder().count() == 0:
        echo.echo_info('no `Wannier90BandsWorkChain` has been submitted yet.')
        return

    rows = []

    for wc, in get_workchains_builder().iterall():

        row = []

        for projection in project:
            projected = getattr(wc, projection)
            row.append(projected)

        rows.append(row)

    if not rows:
        echo.echo_info('no `Wannier90BandsWorkChain` found that match the filtering criteria.')
        return

    if raw:
        echo.echo(tabulate(rows, disable_numparse=True, tablefmt='plain'))
    else:
        headers = [projection.replace('_', ' ').capitalize() for projection in project]
        echo.echo(tabulate(rows, headers=headers, disable_numparse=True))
