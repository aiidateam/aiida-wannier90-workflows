#!/usr/bin/env python
"""Command to plot figures for Wannier90WorkChain."""
import sys

import click

from aiida import orm
from aiida.cmdline.params.types import GroupParamType, NodeParamType, WorkflowParamType
from aiida.cmdline.utils import decorators, echo

from .root import cmd_root


@cmd_root.group("plot")
def cmd_plot():
    """Plot band structures of WorkChain."""


@cmd_plot.command("scdm")
@click.argument("workchain", type=NodeParamType(), nargs=1)
@click.option(
    "-s",
    "--save",
    default=False,
    help="save as a PNG instead of showing matplotlib window",
)
@decorators.with_dbenv()
def cmd_plot_scdm(workchain, save):
    """Plot SCDM projectability fitting.

    WORKCHAIN is the identifier of a Wannier90WorkChain.
    """
    from aiida_wannier90_workflows.utils.workflows.plot.bands import plot_scdm_fit

    plot_scdm_fit(workchain, save)


@cmd_plot.command("band")
@click.argument("node", type=NodeParamType(), nargs=1)
@click.option(
    "-s",
    "--save",
    is_flag=True,
    default=False,
    help="save as a python plotting script instead of showing matplotlib window",
)
@click.option(
    "-f",
    "--save_format",
    type=str,
    default=None,
    help="Save as a python plotting script (default) or png/pdf",
)
@decorators.with_dbenv()
@click.pass_context  # pylint: disable=invalid-name
def cmd_plot_band(ctx, node, save, save_format):  # pylint: disable=unused-argument
    """Plot band structures for BandsData or WorkChain.

    node is the PK of a BandsData, or a PwBaseWorkChain,
    or Wannier90BandsWorkChain.
    """
    from aiida_wannier90_workflows.utils.workflows.plot.bands import (
        get_output_bands,
        get_workchain_fermi_energy,
    )

    bands = get_output_bands(node)

    # pylint: disable=protected-access
    mpl_code = bands._exportcontent(fileformat="mpl_singlefile", main_file_name="")[0]

    if isinstance(node, orm.WorkChainNode):
        title = f"{node.process_label}<{node.pk}>"
    else:
        title = f"{node.__class__.__name__}<{node.pk}>"
    replacement = f'p.set_title("{title}")\npl.show()'
    mpl_code = mpl_code.replace(b"pl.show()", replacement.encode())

    # fermi energy
    if isinstance(node, orm.WorkChainNode):
        fermi_energy = get_workchain_fermi_energy(node)
        replacement = f"fermi_energy = {fermi_energy}\n\n"
        replacement += "p.axhline(y=fermi_energy, color='blue', linestyle='--', label='Fermi', zorder=-1)\n"
        replacement += "pl.legend()\n\n"
        replacement += "for path in paths:"
        mpl_code = mpl_code.replace(b"for path in paths:", replacement.encode())

    mpl_code = mpl_code.replace(
        b"plt.rcParams.update({'text.latex.preview': True})", b""
    )

    if save:
        if save_format:
            filename = f"band_{node.pk}.{save_format}"
            replacement = f'pl.savefig("{filename}")'
            mpl_code = mpl_code.replace(b"pl.show()", replacement.encode())
            # print(mpl_code.decode())
            exec(mpl_code, {})  # pylint: disable=exec-used
            return

        filename = f"band_{node.pk}.py"
        with open(filename, "w", encoding="utf-8") as handle:
            handle.write(mpl_code.decode())

    # print(mpl_code.decode())
    exec(mpl_code, {})  # pylint: disable=exec-used


@cmd_plot.command("bands")
@click.argument("pw", type=NodeParamType(), nargs=-1)
@click.argument("wannier", type=NodeParamType(), nargs=1)
@click.option(
    "-s",
    "--save",
    is_flag=True,
    default=False,
    help="save as a python plotting script instead of showing matplotlib window",
)
@click.option(
    "-f",
    "--save_format",
    type=str,
    default=None,
    help="Save as a python plotting script (default) or png/pdf",
)
@decorators.with_dbenv()
@click.pass_context
def cmd_plot_bands(  # pylint: disable=invalid-name,inconsistent-return-statements
    ctx, pw, wannier, save, save_format
):
    """Compare DFT and Wannier band structures.

    PW is the PK of a PwBaseWorkChain, or a BandsData,
    WANNIER is the PK of a Wannier90BandsWorkChain, or a BandsData.

    If only WANNIER is passed, I will invoke QueryBuilder to search for corresponding PW bands.
    """
    from pprint import pprint

    from aiida.cmdline.commands.cmd_data.cmd_bands import bands_show

    from aiida_wannier90_workflows.utils.workflows.bands import find_pwbands
    from aiida_wannier90_workflows.utils.workflows.plot.bands import (
        get_mpl_code_for_bands,
        get_mpl_code_for_workchains,
    )

    if len(pw) > 1:
        print("Only accept at most 1 PW bands")
        sys.exit()
    elif len(pw) == 0:
        if isinstance(wannier, orm.BandsData):
            print(
                f"Input is a single BandsData, I will invoke `verdi data bands show {wannier.pk}`"
            )
            return ctx.invoke(bands_show, data=[wannier])

        pw_workchains = find_pwbands(wannier)
        if len(pw_workchains) == 0:
            print(
                "Did not find a PW band structrue for comparison, I will only show Wannier bands"
            )
            if "band_structure" not in wannier.outputs:
                print(
                    f"No output band_structure in {wannier.process_label}<{wannier.pk}>!"
                )
                return
            return ctx.invoke(bands_show, data=[wannier.outputs["band_structure"]])

        if len(pw_workchains) == 1:
            pw = pw_workchains[0]
            print(
                f"Found a PW workchain {pw.process_label}<{pw.pk}> with an output band structrue for comparison"
            )
        else:
            pw = pw_workchains[-1]
            print(
                f"Found multiple PW band structure calculations, using the last one<{pw.pk}>:"
            )
            pprint(pw_workchains)
    else:
        pw = pw[0]

    is_workchain = isinstance(pw, orm.WorkChainNode)
    is_workchain = is_workchain and isinstance(wannier, orm.WorkChainNode)
    is_bands = isinstance(pw, orm.BandsData)
    is_bands = is_bands and isinstance(wannier, orm.BandsData)
    if is_workchain:
        mpl_code = get_mpl_code_for_workchains(pw, wannier, save=save)
    elif is_bands:
        mpl_code = get_mpl_code_for_bands(pw, wannier, save=save)
    else:
        print("Unsupported type for")
        print(f"  PW     : {type(pw)}")
        print(f"  WANNIER: {type(wannier)}")
        sys.exit()

    mpl_code = mpl_code.replace(
        b"plt.rcParams.update({'text.latex.preview': True})", b""
    )

    if save_format:
        formula = wannier.inputs.structure.get_formula()
        filename = f"bandsdiff_{formula}_{pw.pk}_{wannier.pk}.{save_format}"
        replacement = f'pl.savefig("{filename}")'
        mpl_code = mpl_code.replace(b"pl.show()", replacement.encode())
        # print(mpl_code.decode())
        exec(mpl_code, {})  # pylint: disable=exec-used
        return

    if not save:
        # print(mpl_code.decode())
        exec(mpl_code, {})  # pylint: disable=exec-used


@cmd_plot.command("bandsdist")
@click.argument("pw", type=GroupParamType(), nargs=1)
@click.argument("wannier", type=GroupParamType(), nargs=1)
@click.option(
    "-s", "--save", is_flag=True, default=False, help="Save bands distance as HDF5"
)
@click.option(
    "-m",
    "--match-by-formula",
    is_flag=True,
    default=False,
    help="Find PwBandsWorkChain by formula instead of structure structure",
)
@decorators.with_dbenv()
def cmd_plot_bandsdist(pw, wannier, save, match_by_formula):
    """Plot bands distance for a group of PwBandsWorkChain and a group of Wannier90BandsWorkChain.

    PW is the PK of a group which contains PwBandsWorkChain,
    WANNIER is the PK of a group which contains Wannier90BandsWorkChain.
    """
    from aiida_wannier90_workflows.utils.workflows.group import standardize_groupname
    from aiida_wannier90_workflows.utils.workflows.plot.distance import (
        bands_distance_for_group,
        plot_distance,
        save_distance,
    )

    df = bands_distance_for_group(wannier, pw, match_by_formula=match_by_formula)
    plot_distance(df)

    if save:
        filename = f"bandsdist_{standardize_groupname(pw.label)}_{standardize_groupname(wannier.label)}.h5"
        save_distance(df, filename)


@cmd_plot.command("checkerboard")
@click.argument("workchain", type=WorkflowParamType(), nargs=1)
@click.option(
    "-s",
    "--save",
    is_flag=True,
    default=False,
    help="save as a PNG instead of showing matplotlib window",
)
@decorators.with_dbenv()
def cmd_plot_checkerboard(workchain, save):
    """Plot bands distance checkerboard a Wannier90OptimizeWorkChain."""
    from aiida_wannier90_workflows.utils.workflows.plot.checkerboard import (
        plot_checkerboard,
    )
    from aiida_wannier90_workflows.workflows.optimize import Wannier90OptimizeWorkChain

    if workchain.process_class != Wannier90OptimizeWorkChain:
        echo.echo_error(f"Input workchain should be {Wannier90OptimizeWorkChain}")

    formula = workchain.inputs.structure.get_formula()

    if save:
        filename = f"checkerboard_{formula}_{workchain.pk}.png"
    else:
        filename = None

    plot_checkerboard(workchain, filename)

    if save:
        echo.echo(f"Saved to {filename}")


@cmd_plot.command("exportbands")
@click.argument("pw", type=GroupParamType(), nargs=1)
@click.argument("wannier", type=GroupParamType(), nargs=1)
@click.option(
    "-s",
    "--savedir",
    type=str,
    default="exportbands",
    show_default=True,
    help="Directory to save exported bands.",
)
@decorators.with_dbenv()
def cmd_plot_exportbands(pw, wannier, savedir):
    """Export python scripts for comparing a group of ``PwBandsWorkChain`` and a group of ``Wannier90BandsWorkChain``.

    PW is the PK of a group which contains ``PwBandsWorkChain``,
    WANNIER is the PK of a group which contains ``Wannier90BandsWorkChain``, or a ``BandsData``.
    """
    import os

    from aiida_wannier90_workflows.utils.workflows.group import get_mapping_for_group
    from aiida_wannier90_workflows.utils.workflows.plot.bands import (
        get_mpl_code_for_workchains,
    )

    dft_group = pw
    wan_group = wannier

    match_by_formula = True
    mapping = get_mapping_for_group(wan_group, dft_group, match_by_formula)

    if savedir != "" and not os.path.exists(savedir):
        os.mkdir(savedir)

    for wan_wc in wan_group.nodes:
        formula = wan_wc.inputs.structure.get_formula()

        if not wan_wc.is_finished_ok:
            print(f"! Skip unfinished {wan_wc.process_label}<{wan_wc.pk}> of {formula}")
            continue

        bands_wc = mapping[wan_wc]
        if bands_wc is None:
            msg = f"! Cannot find DFT bands for {wan_wc.process_label}<{wan_wc.pk}> of {formula}"
            print(msg)
            continue

        if not bands_wc.is_finished_ok:
            print(
                f"! Skip unfinished DFT {wan_wc.process_label}<{bands_wc.pk}> of {formula}"
            )
            continue

        filename = f"bandsdiff_{formula}_{bands_wc.pk}_{wan_wc.pk}.py"
        if savedir != "":
            filename = savedir + "/" + filename

        get_mpl_code_for_workchains(bands_wc, wan_wc, save=True, filename=filename)

        print(f"Saved to {filename}")
