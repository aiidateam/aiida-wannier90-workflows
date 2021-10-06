# -*- coding: utf-8 -*-
"""Command to plot figures for Wannier90WorkChain."""
import sys
import click

from aiida import orm
from aiida.cmdline.utils import decorators

from .root import cmd_root


@cmd_root.group('plot')
def cmd_plot():
    """A set of plotting utilities"""


@cmd_plot.command('scdm')
@click.argument('workchain', type=int, nargs=1)
@click.option(
    '-s',
    '--save',
    default=False,
    help="save as a PNG instead of showing matplotlib window"
)
@decorators.with_dbenv()
def cmd_plot_scdm(workchain, save):
    """Plot SCDM projectability fitting.
    
    WORKCHAIN is the PK of a Wannier90WorkChain."""
    from ..utils.plot import plot_scdm_fit

    wc = orm.load_node(workchain)
    plot_scdm_fit(wc, save)


@cmd_plot.command('bands')
@click.argument('pw', type=int, nargs=-1)
@click.argument('wannier', type=int, nargs=1)
@click.option(
    '-s',
    '--save',
    is_flag=True,
    default=False,
    help="save as a python plotting script instead of showing matplotlib window"
)
@decorators.with_dbenv()
@click.pass_context
def cmd_plot_bands(ctx, pw, wannier, save):
    """Compare DFT and Wannier band structures.

    PW is the PK of a PwBaseWorkChain, or a BandsData,
    WANNIER is the PK of a Wannier90BandsWorkChain, or a BandsData.

    If only WANNIER is passed, I will invoke QueryBuilder to search for corresponding PW bands.
    """
    from pprint import pprint
    from aiida_wannier90_workflows.utils.nodes import find_pwbands
    from aiida_wannier90_workflows.utils.plot import get_mpl_code_for_bands, get_mpl_code_for_workchains

    pk1 = orm.load_node(wannier)

    if len(pw) > 1:
        print('Only accept at most 1 PW bands')
        sys.exit()
    elif len(pw) == 0:
        if isinstance(pk1, orm.BandsData):
            print(
                f'Input is a single BandsData, I will invoke `verdi data bands show {pk1.pk}`'
            )
            from aiida.cmdline.commands.cmd_data.cmd_bands import bands_show
            return ctx.invoke(bands_show, data=[pk1])

        pw_workchains = find_pwbands(pk1)
        if len(pw_workchains) == 0:
            print('Did not find a PW band structrue for comparison')
            sys.exit()
        elif len(pw_workchains) == 1:
            pk0 = pw_workchains[0]
        else:
            pk0 = pw_workchains[-1]
            print(
                f'Found multiple PW band structure calculations, using the last one<{pk0.pk}>:'
            )
            pprint(pw_workchains)
    else:
        pk0 = orm.load_node(pw[0])

    is_workchain = isinstance(pk0, orm.WorkChainNode)
    is_workchain = is_workchain and isinstance(pk1, orm.WorkChainNode)
    is_bands = isinstance(pk0, orm.BandsData)
    is_bands = is_bands and isinstance(pk1, orm.BandsData)
    if is_workchain:
        mpl_code = get_mpl_code_for_workchains(pk0, pk1, save=save)
    elif is_bands:
        mpl_code = get_mpl_code_for_bands(pk0, pk1, save=save)
    else:
        print("Unsupported type for")
        print(f"  PW     : {type(pk0)}")
        print(f"  WANNIER: {type(pk1)}")
        sys.exit()

    # print(mpl_code.decode())

    if not save:
        exec(mpl_code)
