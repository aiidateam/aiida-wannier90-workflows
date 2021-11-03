#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Command to plot figures for Wannier90WorkChain."""
import sys
import click

from aiida import orm
from aiida.cmdline.utils import decorators
from aiida.cmdline.params.types import NodeParamType, GroupParamType
from .root import cmd_root


@cmd_root.group('plot')
def cmd_plot():
    """Plot band structures of WorkChain."""


@cmd_plot.command('scdm')
@click.argument('workchain', type=NodeParamType(), nargs=1)
@click.option('-s', '--save', default=False, help='save as a PNG instead of showing matplotlib window')
@decorators.with_dbenv()
def cmd_plot_scdm(workchain, save):
    """Plot SCDM projectability fitting.

    WORKCHAIN is the identifier of a Wannier90WorkChain.
    """
    from ..utils.plot import plot_scdm_fit

    plot_scdm_fit(workchain, save)


@cmd_plot.command('bands')
@click.argument('pw', type=NodeParamType(), nargs=-1)
@click.argument('wannier', type=NodeParamType(), nargs=1)
@click.option(
    '-s',
    '--save',
    is_flag=True,
    default=False,
    help='save as a python plotting script instead of showing matplotlib window'
)
@decorators.with_dbenv()
@click.pass_context  # pylint: disable=invalid-name
def cmd_plot_bands(ctx, pw, wannier, save):
    """Compare DFT and Wannier band structures.

    PW is the PK of a PwBaseWorkChain, or a BandsData,
    WANNIER is the PK of a Wannier90BandsWorkChain, or a BandsData.

    If only WANNIER is passed, I will invoke QueryBuilder to search for corresponding PW bands.
    """
    from pprint import pprint
    from aiida.cmdline.commands.cmd_data.cmd_bands import bands_show
    from aiida_wannier90_workflows.utils.node.bands import find_pwbands
    from aiida_wannier90_workflows.utils.plot import get_mpl_code_for_bands, get_mpl_code_for_workchains

    if len(pw) > 1:
        print('Only accept at most 1 PW bands')
        sys.exit()
    elif len(pw) == 0:
        if isinstance(wannier, orm.BandsData):
            print(f'Input is a single BandsData, I will invoke `verdi data bands show {wannier.pk}`')
            return ctx.invoke(bands_show, data=[wannier])

        pw_workchains = find_pwbands(wannier)
        if len(pw_workchains) == 0:
            print('Did not find a PW band structrue for comparison, I will only show Wannier bands')
            if 'band_structure' not in wannier.outputs:
                print(f'No output band_structure in {wannier.process_label}<{wannier.pk}>!')
                return
            return ctx.invoke(bands_show, data=[wannier.outputs['band_structure']])

        if len(pw_workchains) == 1:
            pw = pw_workchains[0]
            print(f'Found a PW workchain {pw.process_label}<{pw.pk}> with an output band structrue for comparison')
        else:
            pw = pw_workchains[-1]
            print(f'Found multiple PW band structure calculations, using the last one<{pw.pk}>:')
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
        print('Unsupported type for')
        print(f'  PW     : {type(pw)}')
        print(f'  WANNIER: {type(wannier)}')
        sys.exit()

    if not save:
        # print(mpl_code.decode())
        exec(mpl_code, {})  # pylint: disable=exec-used


@cmd_plot.command('bandsdist')
@click.argument('pw', type=GroupParamType(), nargs=1)
@click.argument('wannier', type=GroupParamType(), nargs=1)
@click.option('-s', '--save', type=str, help='Save bands distance as HDF5')
@decorators.with_dbenv()
def cmd_plot_bandsdist(pw, wannier, save):
    """Plot bands distance for a group of PwBandsWorkChain and a group of Wannier90BandsWorkChain.

    PW is the PK of a PwBaseWorkChain, or a BandsData,
    WANNIER is the PK of a Wannier90BandsWorkChain, or a BandsData.

    If only WANNIER is passed, I will invoke QueryBuilder to search for corresponding PW bands.
    """
    from aiida_wannier90_workflows.utils.bandsdist import bands_distance_for_group, plot_distance, save_distance

    df = bands_distance_for_group(wannier, pw, match_by_formula=True)
    plot_distance(df)

    if save is not None:
        save_distance(df, save)
