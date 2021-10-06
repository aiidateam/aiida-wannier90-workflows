# -*- coding: utf-8 -*-
"""Commands to list instances of `PseudoPotentialFamily`."""
from pprint import pprint
import click
from aiida import orm
from aiida.cmdline.params import options as options_core
from aiida.cmdline.utils import decorators, echo
from aiida.cmdline.utils.query.calculation import CalculationQueryBuilder
from aiida.cmdline.commands.cmd_process import process_list

from .root import cmd_root


@cmd_root.command('list')
@options_core.GROUP(
    help='Only include entries that are a member of this group.'
)
@options_core.PROJECT(
    type=click.Choice(CalculationQueryBuilder.valid_projections),
    default=CalculationQueryBuilder.default_projections
)
@options_core.RAW()
@decorators.with_dbenv()
@click.pass_context
def cmd_list(ctx, group, project, raw):
    """List all instances of `Wannier90BandsWorkChain`."""
    from tabulate import tabulate

    process_label = 'Wannier90BandsWorkChain'

    result = ctx.invoke(
        process_list,
        all_entries=True,
        group=group,
        process_label=process_label,
        project=project,
        raw=raw,
    )

    # Collect statistics of failed workflows
    # similar to aiida.cmdline.commands.cmd_process.process_list
    relationships = {}

    if group:
        relationships['with_node'] = group

    builder = CalculationQueryBuilder()
    filters = builder.get_filters(
        all_entries=True,
        process_state=None,
        process_label=process_label,
        paused=False,
        exit_status=None,
        failed=False
    )
    query_set = builder.get_query_set(
        relationships=relationships,
        filters=filters,
        order_by={'ctime': 'asc'},
        past_days=None,
        limit=None
    )
    state_projections = (
        'pk', 'state', 'process_state', 'process_status', 'exit_status',
        'job_state', 'scheduler_state', 'exception'
    )
    projected = builder.get_projected(query_set, projections=state_projections)
    headers = projected.pop(0)
    # pprint(projected)

    state_projections_idx = {
        state_projections[i]: i
        for i in range(len(state_projections))
    }
    excepted_workflows = {}
    num_success = 0
    num_excepted = 0
    for wc in projected:
        pk = str(wc[state_projections_idx['pk']])
        process_state = wc[state_projections_idx['process_state']]
        exit_status = str(wc[state_projections_idx['exit_status']])

        if process_state == 'Finished':
            if exit_status == '0':
                num_success += 1
                continue
            elif exit_status is None:
                raise Exception(
                    f'{process_label}<{pk}> process_state = Finished but exit_status is None?'
                )

            if exit_status not in excepted_workflows:
                excepted_workflows[exit_status] = []
            excepted_workflows[exit_status].append(pk)
        else:
            if process_state is None:
                raise Exception(
                    f'{process_label}<{pk}> process_state is None?'
                )

            if process_state not in excepted_workflows:
                excepted_workflows[process_state] = []
            excepted_workflows[process_state].append(pk)

        num_excepted += 1

    # print(excepted_workflows)
    data = [[k, str(len(v)), ' '.join(v)]
            for k, v in excepted_workflows.items()]
    data.append(['-' * 8, None, None])
    data.append(['Total excepted', num_excepted, None])
    data.append(['Total success', num_success, None])
    data.append(['Total', num_success + num_excepted, None])
    headers = ['exit_status', 'count', 'pk']
    tabulated = tabulate(data, headers=headers)

    tabulated = f'\n{tabulated}'
    echo.echo(tabulated)

    return result
