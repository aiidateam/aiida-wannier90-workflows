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
    from ..utils.plots import plot_scdm_fit

    wc = orm.load_node(workchain)
    plot_scdm_fit(wc, save)


@cmd_plot.command('bands')
@click.argument('pw', type=int, nargs=-1)
@click.argument('wannier', type=int, nargs=1)
@click.option(
    '-s',
    '--save',
    default=False,
    help="save as a python plotting script instead of showing matplotlib window"
)
@decorators.with_dbenv()
def cmd_plot_bands(pw, wannier, save):
    """Compare DFT and Wannier band structures.

    PW is the PK of a PwBaseWorkChain, or a BandsData,
    WANNIER is the PK of a Wannier90BandsWorkChain, or a BandsData.

    If only WANNIER is passed, I will invoke QueryBuilder to search for corresponding PW bands.
    """
    from pprint import pprint
    from ..utils.plots import get_mpl_code_for_bands, get_mpl_code_for_workchains

    pk1 = orm.load_node(wannier)

    if len(pw) > 1:
        print('Only accept at most 1 PW bands')
        sys.exit()
    elif len(pw) == 0:
        from aiida_quantumespresso.workflows.pw.bands import PwBaseWorkChain, PwBandsWorkChain
        qb = orm.QueryBuilder()
        if 'primitive_structure' in pk1.outputs:
            structure = pk1.outputs['primitive_structure']
        else:
            structure = pk1.inputs['structure']
        qb.append(
            orm.StructureData, tag='structure', filters={'id': structure.pk}
        )
        qb.append((PwBaseWorkChain, PwBandsWorkChain),
                  with_incoming='structure',
                  tag='pw_wc',
                  filters={'attributes.exit_status': 0})
        pw_workchains = [i[0] for i in qb.all()]
        pw_workchains.sort(key=lambda i: i.pk)
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
