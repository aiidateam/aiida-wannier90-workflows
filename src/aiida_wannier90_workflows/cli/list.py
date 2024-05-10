"""Commands to list instances of `Wannier90BandsWorkChain`."""

import click

from aiida import orm
from aiida.cmdline.params import options as options_core
from aiida.cmdline.utils import decorators, echo
from aiida.tools.query.calculation import CalculationQueryBuilder

from .root import cmd_root

# pylint: disable=too-many-arguments,too-many-locals,too-many-branches,too-many-statements


def print_process_table(
    process_label,
    all_entries,
    group,
    process_state,
    paused,
    exit_status,
    failed,
    past_days,
    limit,
    project,
    raw,
    order_by,
    order_dir,
):
    """Print process table.

    Mostly the same as `verdi process list`, but I also print structure formula.
    """
    from tabulate import tabulate

    from aiida.cmdline.utils.common import print_last_process_state_change

    relationships = {}

    if group:
        relationships["with_node"] = group

    builder = CalculationQueryBuilder()
    filters = builder.get_filters(
        all_entries, process_state, process_label, paused, exit_status, failed
    )
    query_set = builder.get_query_set(
        relationships=relationships,
        filters=filters,
        order_by={order_by: order_dir},
        past_days=past_days,
        limit=limit,
    )
    projected = builder.get_projected(query_set, projections=project)

    headers = projected.pop(0)

    projected_with_structure = []
    # Add structure
    for entry in projected:
        # I assume 0th column is PK
        pk = entry[0]
        node = orm.load_node(pk)
        if "structure" in node.inputs:
            formula = node.inputs.structure.get_formula()
        elif "valcond" in node.inputs:
            # Wannier90SplitWorkChain
            formula = node.inputs["valcond"].structure.get_formula()
        elif "formula_hill" in node.base.extras.all:
            formula = node.base.extras.all["formula_hill"]
        else:
            formula = "?"
        entry_with_structure = [pk, formula, *entry[1:]]
        projected_with_structure.append(entry_with_structure)
    projected = projected_with_structure
    headers = [headers[0], "structure", *headers[1:]]

    if raw:
        tabulated = tabulate(projected, tablefmt="plain")
        echo.echo(tabulated)
    else:
        tabulated = tabulate(projected, headers=headers)
        echo.echo(tabulated)
        echo.echo(f"\nTotal results: {len(projected)}\n")
        print_last_process_state_change()

    return projected, headers


@cmd_root.command("list")
@click.option(
    "-L",
    "--process-label",
    type=str,
    default=None,
    show_default=True,
    help="Process label to filter. If group is provided, the process label is ignored.",
)
@click.option(
    "-s",
    "--show-statistics",
    is_flag=True,
    default=False,
    show_default=True,
    help="Show statistics about exit status of all workchains.",
)
@options_core.PROJECT(
    type=click.Choice(CalculationQueryBuilder.valid_projections),
    default=CalculationQueryBuilder.default_projections,
)
@options_core.ORDER_BY()
@options_core.ORDER_DIRECTION()
@options_core.GROUP(help="Only include entries that are a member of this group.")
@options_core.ALL(
    help="Show all entries, regardless of their process state.", default=True
)
@options_core.PROCESS_STATE()
@options_core.PAUSED()
@options_core.EXIT_STATUS()
@options_core.FAILED()
@options_core.PAST_DAYS()
@options_core.LIMIT()
@options_core.RAW()  # pylint: disable=too-many-statements
@decorators.with_dbenv()  # pylint: disable=too-many-statements
@click.pass_context
def cmd_list(
    ctx,  # pylint: disable=unused-argument
    process_label,
    show_statistics,
    all_entries,
    group,
    process_state,
    paused,
    exit_status,
    failed,
    past_days,
    limit,
    project,
    raw,
    order_by,
    order_dir,
):
    """List all instances of `Wannier90BandsWorkChain`."""
    from tabulate import tabulate

    # from aiida.cmdline.commands.cmd_process import process_list
    # result = ctx.invoke(
    #     process_list,
    #     all_entries=True,
    #     group=group,
    #     process_label=process_label,
    #     project=project,
    #     raw=raw,
    # )
    # Copied from process_list
    # from aiida.cmdline.utils.common import check_worker_load

    if process_label is None:
        # If group is present, I will auto set process_label
        if group:
            if len(group.nodes) > 0 and "process_label" in dir(group.nodes[0]):
                process_label = group.nodes[0].process_label
        else:
            process_label = "Wannier90BandsWorkChain"

    print_process_table(
        process_label,
        all_entries,
        group,
        process_state,
        paused,
        exit_status,
        failed,
        past_days,
        limit,
        project,
        raw,
        order_by,
        order_dir,
    )

    # This is very slow, I skip it
    if not show_statistics:
        return

    # if not raw:
    #     # Second query to get active process count
    #     # Currently this is slow but will be fixed wiith issue #2770
    #     # We place it at the end so that the user can Ctrl+C after getting the process table.
    #     builder = CalculationQueryBuilder()
    #     filters = builder.get_filters(process_state=('created', 'waiting', 'running'))
    #     query_set = builder.get_query_set(filters=filters)
    #     projected = builder.get_projected(query_set, projections=['pk'])
    #     worker_slot_use = len(projected) - 1
    #     check_worker_load(worker_slot_use)

    # Collect statistics of failed workflows
    # similar to aiida.cmdline.commands.cmd_process.process_list
    relationships = {}

    if group:
        relationships["with_node"] = group

        # If group is present, I will auto set process_label
        if len(group.nodes) > 0 and "process_label" in dir(group.nodes[0]):
            process_label = group.nodes[0].process_label

    builder = CalculationQueryBuilder()
    filters = builder.get_filters(
        all_entries=True,
        process_state=None,
        process_label=process_label,
        paused=False,
        exit_status=None,
        failed=False,
    )
    query_set = builder.get_query_set(
        relationships=relationships,
        filters=filters,
        order_by={"ctime": "asc"},
        past_days=None,
        limit=None,
    )
    state_projections = (
        "pk",
        "state",
        "process_state",
        "process_status",
        "exit_status",
        "job_state",
        "scheduler_state",
        "exception",
    )
    projected = builder.get_projected(query_set, projections=state_projections)
    headers = projected.pop(0)
    # pprint(projected)

    state_projections_idx = {
        state_projections[i]: i for i in range(len(state_projections))
    }
    excepted_workflows = {}
    num_success = 0
    num_excepted = 0
    for entry in projected:
        pk = str(entry[state_projections_idx["pk"]])
        process_state = entry[state_projections_idx["process_state"]]
        exit_status = str(entry[state_projections_idx["exit_status"]])

        if process_state == "Finished":
            if exit_status == "0":
                num_success += 1
                continue
            if exit_status is None:
                raise click.ClickException(
                    f"{process_label}<{pk}> process_state = Finished but exit_status is None?"
                )

            if exit_status not in excepted_workflows:
                excepted_workflows[exit_status] = []
            excepted_workflows[exit_status].append(pk)
        else:
            if process_state is None:
                raise click.ClickException(
                    f"{process_label}<{pk}> process_state is None?"
                )

            if process_state not in excepted_workflows:
                excepted_workflows[process_state] = []
            excepted_workflows[process_state].append(pk)

        num_excepted += 1

    # print(excepted_workflows)
    data = [[k, str(len(v)), " ".join(v)] for k, v in excepted_workflows.items()]
    data.append(["-" * 8, None, None])
    data.append(["Total excepted", num_excepted, None])
    data.append(["Total success", num_success, None])
    data.append(["Total", num_success + num_excepted, None])
    headers = ["exit_status", "count", "pk"]
    tabulated = tabulate(data, headers=headers)

    tabulated = f"\n{tabulated}"
    echo.echo(tabulated)
