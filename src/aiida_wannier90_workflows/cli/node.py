"""Command line interface `aiida-wannier90-workflows`."""

import click

from aiida import orm
from aiida.cmdline.params import arguments
from aiida.cmdline.utils import echo
from aiida.common.links import LinkType

from .root import cmd_root


@cmd_root.group("node")
def cmd_node():  # pylint: disable=unused-argument
    """Inspect a node."""


@cmd_node.command("show")
@arguments.NODES()
@click.pass_context
def cmd_node_show(ctx, nodes):  # pylint: disable=unused-argument
    """Show info of a node."""
    from pprint import pprint

    from aiida.cmdline.commands.cmd_node import node_show

    from aiida_wannier90_workflows.utils.workflows.builder.serializer import serialize

    header = "\n--- Additional info: {:s} ---"

    for node in nodes:
        ctx.invoke(node_show, nodes=nodes, print_groups=False)

        if isinstance(node, orm.RemoteData):
            path = f"{node.computer.label}:{node.get_remote_path()}"
            echo.echo(header.format("path"))
            echo.echo(f"{path}")
        elif isinstance(node, orm.FolderData):
            path = f"{node.base.repository.list_object_names()}"
            echo.echo(header.format("objects in repository"))
            echo.echo(f"{path}")
        elif isinstance(node, orm.RemoteStashFolderData):
            path = f'{node.base.attributes.all["target_basepath"]}'
            echo.echo(header.format("path"))
            echo.echo(f"{path}")
        elif isinstance(node, orm.SinglefileData):
            path = f'{node._repository._get_base_folder().abspath}/{node.base.attributes.all["filename"]}'  # pylint: disable=protected-access
            echo.echo(header.format("path"))
            echo.echo(f"{path}")
        elif isinstance(node, (orm.CalculationNode, orm.WorkflowNode)):
            inputs = {}
            for key in node.inputs:
                inputs[key] = serialize(node.inputs[key])
            echo.echo(header.format(f"{node.process_label}.inputs"))
            pprint(inputs)
        else:
            echo.echo(header.format(f"serialize({node.__class__.__name__})"))
            pprint(serialize(node))


def find_calcjob(node: orm.Node, link_label: str) -> orm.CalcJobNode:
    """Find CalcJob of a workchain with the specified link label."""
    from aiida_wannier90_workflows.utils.workflows import get_last_calcjob

    last_calcjob = None
    if isinstance(node, orm.CalcJobNode):
        last_calcjob = node
    elif isinstance(node, orm.WorkChainNode):
        if link_label is None:
            last_calcjob = get_last_calcjob(node)
            if last_calcjob is None:
                echo.echo_critical(f"No CalcJob for {node}?")

            # Get call link label
            link_triples = node.base.links.get_outgoing(
                link_type=(LinkType.CALL_CALC, LinkType.CALL_WORK)
            ).link_triples
            link = list(
                filter(
                    lambda _: _.node == last_calcjob
                    or last_calcjob in _.node.called_descendants,
                    link_triples,
                )
            )[0]
            link_label = link.link_label
        else:
            try:
                called = (
                    node.base.links.get_outgoing(link_label_filter=link_label)
                    .one()
                    .node
                )
            except ValueError:
                link_triples = node.base.links.get_outgoing(
                    link_type=(LinkType.CALL_CALC, LinkType.CALL_WORK)
                ).link_triples
                valid_lables = [x.link_label for x in link_triples]
                valid_lables = "\n".join(valid_lables)
                echo.echo_critical(
                    f"No nodes found with call link label `{link_label}`, valid labels are:\n"
                    f"{valid_lables}"
                )

            if isinstance(called, orm.CalcJobNode):
                last_calcjob = called
            elif isinstance(called, orm.WorkChainNode):
                last_calcjob = get_last_calcjob(called)
            else:
                echo.echo_critical(f"Unsupported type of node: {called}")

        msg = f"Parent WorkChain: {node.process_label}<{node.pk}>\n"
        msg += f" Lastest CalcJob: {last_calcjob.process_label}<{last_calcjob.pk}>\n"
        msg += f" Call link label: {link_label}\n"
        echo.echo(msg)
    else:
        echo.echo_critical(f"Unsupported type of node: {type(node)} {node}")

    return last_calcjob


@cmd_node.command("inputcat")
@arguments.NODE()
@click.option(
    "-l",
    "--link-label",
    "link_label",
    type=click.STRING,
    required=False,
    help="Goto the calcjob with this call link label.",
)
@click.option(
    "-s",
    "--show-scheduler",
    is_flag=True,
    default=False,
    show_default=True,
    help="Show scheduler submission script instead of calculation stdout.",
)
@click.option(
    "-r",
    "--show-remote",
    is_flag=True,
    default=False,
    show_default=True,
    help="Show input file in the remote_folder RemoteData. Otherwise show info from database.",
)
def cmd_node_inputcat(node, link_label, show_scheduler, show_remote):
    """Show input or scheduler submission file of a CalcJob."""
    from pprint import pprint
    from tempfile import NamedTemporaryFile

    from aiida_wannier90_workflows.utils.workflows.builder.serializer import serialize

    def show_input(
        calcjob: orm.CalcJobNode, show_scheduler: bool, show_remote: bool
    ) -> None:
        if show_remote:
            if show_scheduler:
                echo.echo(
                    f"===== {calcjob.process_label}<{calcjob.pk}> remote scheduler script =====",
                    bold=True,
                )
                file_path = calcjob.base.attributes.all["submit_script_filename"]
            else:
                echo.echo(
                    f"===== {calcjob.process_label}<{calcjob.pk}> remote input file =====",
                    bold=True,
                )
                file_path = calcjob.base.attributes.all["input_filename"]

            with NamedTemporaryFile("w+") as out_file:
                calcjob.outputs.remote_folder.getfile(file_path, out_file.name)
                out_file.seek(0)
                input_lines = out_file.read()
                echo.echo(input_lines)
        else:
            if show_scheduler:
                echo.echo(
                    f"===== {calcjob.process_label}<{calcjob.pk}> scheduler info =====",
                    bold=True,
                )
                data = {}
                for key in (
                    "withmpi",
                    "resources",
                    "queue_name",
                    "num_machines",
                    "num_mpiprocs",
                    "mpirun_extra_params",
                    "max_wallclock_seconds",
                    "custom_scheduler_commands",
                    "append_text",
                    "prepend_text",
                ):
                    if key not in calcjob.base.attributes.all:
                        continue
                    data[key] = calcjob.base.attributes.all[key]
                pprint(data)
            else:
                echo.echo(
                    f"===== {calcjob.process_label}<{calcjob.pk}> inputs =====",
                    bold=True,
                )
                if isinstance(calcjob, (orm.CalculationNode, orm.WorkflowNode)):
                    inputs = {}
                    for key in calcjob.inputs:
                        inputs[key] = serialize(calcjob.inputs[key])
                    pprint(inputs)

    calcjob = find_calcjob(node, link_label)
    show_input(calcjob, show_scheduler, show_remote)


@cmd_node.command("outputcat")
@arguments.NODE()
@click.option(
    "-l",
    "--link-label",
    "link_label",
    type=click.STRING,
    required=False,
    help="Goto the calcjob with this call link label.",
)
@click.option(
    "-s",
    "--show-scheduler",
    is_flag=True,
    default=False,
    show_default=True,
    help="Show scheduler stdout/stderr instead of calculation stdout.",
)
def cmd_node_outputcat(node, link_label, show_scheduler):
    """Show stdout or scheduler output of a CalcJob from retrieved FolderData."""

    def show_output(calcjob: orm.CalcJobNode, show_scheduler: bool) -> None:
        if show_scheduler:
            echo.echo(
                f"===== {calcjob.process_label}<{calcjob.pk}> scheduler stdout =====",
                bold=True,
            )
            echo.echo(calcjob.get_scheduler_stdout())
            echo.echo("\n")
            echo.echo(
                f"===== {calcjob.process_label}<{calcjob.pk}> scheduler stderr =====",
                bold=True,
            )
            echo.echo(calcjob.get_scheduler_stderr())
        else:
            output_filename = calcjob.base.attributes.all["output_filename"]
            output_lines = calcjob.outputs.retrieved.get_object_content(output_filename)
            echo.echo(
                f"===== {calcjob.process_label}<{calcjob.pk}> stdout =====", bold=True
            )
            echo.echo(output_lines)

    calcjob = find_calcjob(node, link_label)
    show_output(calcjob, show_scheduler)


@cmd_node.command("gotocomputer")
@arguments.NODE()
@click.option(
    "-l",
    "--link-label",
    "link_label",
    type=click.STRING,
    required=False,
    help="Goto the calcjob with this call link label.",
)
@click.pass_context  # pylint: disable=too-many-statements
def cmd_node_gotocomputer(ctx, node, link_label):
    """Open a shell in the remote folder of the calcjob, or the last calcjob of the workflow."""
    import os

    from aiida.cmdline.commands.cmd_calcjob import calcjob_gotocomputer
    from aiida.common.exceptions import NotExistent
    from aiida.plugins import DataFactory

    RemoteData = DataFactory("core.remote")  # pylint: disable=invalid-name
    RemoteStashFolderData = DataFactory(
        "core.remote.stash.folder"
    )  # pylint: disable=invalid-name
    FolderData = DataFactory("core.folder")  # pylint: disable=invalid-name

    echo.echo(f"Node<{node.pk}> type {type(node)}")

    if isinstance(node, (orm.CalcJobNode, orm.WorkChainNode)):
        calcjob = find_calcjob(node, link_label)
        ctx.invoke(calcjob_gotocomputer, calcjob=calcjob)
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
            echo.echo_critical("no remote work directory for this node")

        command = transport.gotocomputer_command(remote_workdir)
        echo.echo_info("going to the remote work directory...")
        os.system(command)
        return
    elif isinstance(node, FolderData):
        # Since AiiDA 2.0, the files in the repository are not stored as raw data,
        # so I only list the filenames in the FolderData
        echo.echo_info("listing files in the repository folder...")
        echo.echo("")
        for filename in node.base.repository.list_object_names():
            echo.echo(filename)
        return
    else:
        echo.echo_critical(f"Unsupported type of node: {type(node)} {node}")


@cmd_node.command("cleanworkdir")
# @arguments.WORKFLOWS("workflows")
@arguments.PROCESSES("workflows")  # support both Calculation and Workflow
@click.option(
    "-r",
    "--raw",
    is_flag=True,
    default=False,
    show_default=True,
    help="Only print the remote dir of CalcJobs.",
)
@click.option(
    "-unk",
    "--only-unk",
    is_flag=True,
    default=False,
    show_default=True,
    help="Only clean UNK* symlinks of finished Wannier90Calculation.",
)
@click.option(
    "-f",
    "--fast",
    is_flag=True,
    default=False,
    show_default=True,
    help="Reuse transport to speed up the cleaning.",
)  # pylint: disable=too-many-statements
def cmd_node_clean(workflows, only_unk, fast, raw):
    """Clean the workdir of CalcJobNode/WorkChainNode."""
    from aiida.common.exceptions import NotExistentAttributeError
    from aiida.orm.utils.remote import clean_remote

    from aiida_wannier90.calculations import Wannier90Calculation

    for node in workflows:  # pylint: disable=too-many-nested-blocks
        calcs = []
        if isinstance(node, orm.CalcJobNode):
            calcs.append(node)
        elif isinstance(node, orm.WorkChainNode):
            for called_descendant in node.called_descendants:
                if not isinstance(called_descendant, orm.CalcJobNode):
                    continue
                calcs.append(called_descendant)
        else:
            echo.echo_critical(f"Unsupported type of node: {node}")

        if only_unk:
            # In Wannier90OptimizeWorkChain, if there are many iterations, there might be
            # a large amount of UNK* symlinks wasting inodes. Here I only remove the UNK*
            # of finished Wannier90Calculation.
            calcs = filter(lambda _: _.process_class == Wannier90Calculation, calcs)
            calcs = filter(lambda _: _.is_finished, calcs)
            calcs = list(calcs)

        if raw:
            for calc in calcs:
                remote_dir = calc.get_remote_workdir()
                if remote_dir is None:
                    continue
                print(remote_dir)
            continue

        cleaned_calcs = []
        if fast:
            if len(calcs) > 0:
                calc_computers = [_.computer.uuid for _ in calcs]
                if len(set(calc_computers)) > 1:
                    echo.echo_error(
                        "Cannot reuse transport: the CalcJobs of the workchain are not on the same computer."
                    )
                    return
                authinfo = calcs[0].get_authinfo()
                transport = authinfo.get_transport()
                transport.open()
            for calc in calcs:
                try:
                    remote_dir = calc.get_remote_workdir()
                    if remote_dir is None:
                        continue
                    if only_unk:
                        transport.chdir(remote_dir)
                        transport.exec_command_wait("rm UNK*")
                        cleaned_calcs.append(calc.pk)
                    else:
                        clean_remote(transport, remote_dir)
                        cleaned_calcs.append(calc.pk)
                except (OSError, KeyError):
                    pass
            if len(calcs) > 0:
                transport.close()
        else:
            for calc in calcs:
                if only_unk:
                    try:
                        remote_dir = calc.get_remote_workdir()
                        if remote_dir is None:
                            continue
                        authinfo = calc.get_authinfo()
                        transport = authinfo.get_transport()
                        with transport:
                            transport.chdir(remote_dir)
                            # for file_name in transport.listdir():
                            #     if file_name.startswith('UNK'):
                            #         transport.rmtree(file_name)
                            # This is much faster
                            transport.exec_command_wait("rm UNK*")
                        cleaned_calcs.append(calc.pk)
                    except (OSError, KeyError):
                        pass
                else:
                    try:
                        calc.outputs.remote_folder._clean()
                        cleaned_calcs.append(calc.pk)
                    except NotExistentAttributeError:
                        # NotExistentAttributeError: when calc was excepted and has no remote_folder
                        # Some times if the CalcJob is killed and has no outputs.remote_folder,
                        # I need to remove it manually.
                        remote_dir = calc.get_remote_workdir()
                        if remote_dir is None:
                            continue
                        authinfo = calc.get_authinfo()
                        transport = authinfo.get_transport()
                        with transport:
                            clean_remote(transport, remote_dir)
                        cleaned_calcs.append(calc.pk)
                    except (OSError, KeyError):
                        pass

        if len(cleaned_calcs) > 0:
            if only_unk:
                echo.echo(
                    f"cleaned UNK* symlinks of calculations: {' '.join(map(str, cleaned_calcs))}"
                )
            else:
                echo.echo(
                    f"cleaned remote folders of calculations: {' '.join(map(str, cleaned_calcs))}"
                )


@cmd_node.command("saveinput")
@arguments.NODE("workflow")
@click.option(
    "--path",
    "-p",
    type=click.Path(),
    default=".",
    show_default=True,
    help="The directory to save all the input files.",
)
@click.pass_context
def cmd_node_saveinput(ctx, workflow, path):
    """Download scf/nscf/open_grid/pw2wan/wannier90 input files."""
    from contextlib import redirect_stdout
    from pathlib import Path

    from aiida.cmdline.commands.cmd_calcjob import calcjob_inputcat

    from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain

    from aiida_wannier90_workflows.utils.workflows import get_last_calcjob
    from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain
    from aiida_wannier90_workflows.workflows.open_grid import Wannier90OpenGridWorkChain
    from aiida_wannier90_workflows.workflows.optimize import Wannier90OptimizeWorkChain
    from aiida_wannier90_workflows.workflows.projwfcbands import ProjwfcBandsWorkChain
    from aiida_wannier90_workflows.workflows.wannier90 import Wannier90WorkChain

    supported_class = (
        PwBandsWorkChain,
        ProjwfcBandsWorkChain,
        Wannier90WorkChain,
        Wannier90OpenGridWorkChain,
        Wannier90BandsWorkChain,
        Wannier90OptimizeWorkChain,
    )
    if workflow.process_class not in supported_class:
        echo.echo_error(f"Only support {supported_class}, input is {workflow}")
        return

    dir_path = Path(path)
    if not dir_path.exists():
        dir_path.mkdir()

    links = workflow.base.links.get_outgoing(
        link_type=(LinkType.CALL_CALC, LinkType.CALL_WORK)
    )

    for link in links:
        link_label = link.link_label

        if link_label in (
            "scf",
            "nscf",
            "open_grid",
            "pw2wannier90",
            "projwfc",
            "bands",
        ):
            calcjob = link.node
            if isinstance(calcjob, orm.WorkChainNode):
                calcjob = get_last_calcjob(calcjob)
            save_path = dir_path / f"{link_label}.in"
        elif link_label in ["wannier90", "wannier90_plot"]:
            calcjob = get_last_calcjob(link.node)
            save_path = dir_path / f"{link_label}.win"
        else:
            continue

        with open(save_path, "w", encoding="utf-8") as handle:
            with redirect_stdout(handle):
                ctx.invoke(calcjob_inputcat, calcjob=calcjob)
            echo.echo(f"Saved to {save_path}")
