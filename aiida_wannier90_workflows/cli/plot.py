# -*- coding: utf-8 -*-
"""Command to plot figures for Wannier90WorkChain."""
import click

from aiida import orm
from aiida.cmdline.params import options as options_core
from aiida.cmdline.utils import decorators, echo

from .root import cmd_root

@cmd_root.group('plot')
def cmd_plot():
    """A set of plotting utilities"""

@cmd_plot.command('scdm')
@click.argument('workchain', type=int, nargs=1)
@click.option('-s', '--save', default=False, help="save as a PNG instead of showing matplotlib window")
@decorators.with_dbenv()
def cmd_plot_scdm(workchain, save):
    """Plot SCDM projectability fitting.
    
    WORKCHAIN is the PK of a Wannier90WorkChain."""
    from ..utils.plots import plot_scdm_fit

    wc = orm.load_node(workchain)
    plot_scdm_fit(wc, save)

@cmd_plot.command('bands')
@click.argument('pw', type=int, nargs=1)
@click.argument('wannier', type=int, nargs=1)
@click.option('-s', '--save', default=False, help="save as a python plotting script instead of showing matplotlib window")
@decorators.with_dbenv()
def cmd_plot_bands(pw, wannier, save):
    """Compare DFT and Wannier band structures.

    PW is the PK of a PwBaseWorkChain, or a BandsData,
    WANNIER is the PK of a Wannier90BandsWorkChain, or a BandsData."""
    from ..utils.plots import get_mpl_code_for_bands, get_mpl_code_for_workchains

    pk0 = orm.load_node(pw)
    pk1 = orm.load_node(wannier)

    is_workchain = isinstance(pk0, orm.WorkChainNode) 
    is_workchain = is_workchain and isinstance(pk1, orm.WorkChainNode)
    is_bands = isinstance(pk0, orm.BandsData) 
    is_bands = is_bands and isinstance(pk1, orm.BandsData)
    if is_workchain:
        mpl_code = get_mpl_code_for_workchains(pk0, pk1, save=save)
    elif is_bands:
        mpl_code = get_mpl_code_for_bands(pk0, pk1, save=save)
    else:
        print(f"Unsupported type for")
        print(f"  PW     : {type(pk0)}")
        print(f"  WANNIER: {type(pk1)}")
        exit()

    # print(mpl_code.decode())

    if not save:
        exec(mpl_code)
