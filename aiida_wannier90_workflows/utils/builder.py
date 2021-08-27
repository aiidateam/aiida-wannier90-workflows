#!/usr/bin/env python
import typing
from aiida import orm
from aiida.engine.processes.builder import ProcessBuilder

from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain


def get_pwbuilder_for_stash(
    pw_calc: typing.Union[PwCalculation, PwBaseWorkChain],
    stash_folder: typing.Optional[str] = None
) -> ProcessBuilder:
    import os, os.path
    from aiida import get_profile
    from aiida.common.datastructures import StashMode

    builder = pw_calc.get_builder_restart()

    if stash_folder is None:
        profile = get_profile()
        stash_folder = os.path.join(profile.repository_path, 'stash')

    if not os.path.exists(stash_folder):
        os.mkdir(stash_folder)

    builder.metadata.options['stash'] = {
        'source_list': [
            'out/aiida.save/charge-density.dat',
            'out/aiida.save/charge-density.hdf5',
            'out/aiida.save/data-file-schema.xml',
            'out/aiida.save/paw.txt',
        ],
        'target_base':
        stash_folder,
        'stash_mode':
        StashMode.COPY.value,
    }

    return builder


def unstash(remote_stash_folder: orm.RemoteStashFolderData) -> orm.RemoteData:
    from hith.calculations import stash_to_remote

    return stash_to_remote(remote_stash_folder)


def get_builder_for_stash_restart(
    codes: dict,
    structure: orm.StructureData,
    parent_folder: typing.Union[orm.RemoteData, orm.RemoteStashFolderData],
    kwargs: typing.Optional[dict] = None
) -> ProcessBuilder:
    """Generate a `ProcessBuilder` for `Wannier90BandsWorkChain` to restart from a `parent_folder`
    which contians converged charge density. The builder will run a nscf calculation with 
    increased number of bands suitable for Wannierisation, and using a symmetry-reduced kpoint mesh.
    A subsequent `OpengridCalculation` will unfold the kmesh for following Wannierisation.

    :param codes: [description]
    :type codes: dict
    :param structure: [description]
    :type structure: orm.StructureData
    :param parent_folder: [description]
    :type parent_folder: orm.RemoteData
    :param kwargs: [description]
    :type kwargs: dict
    :return: [description]
    :rtype: ProcessBuilder
    """

    if kwargs is not None:
        if 'run_opengrid' in kwargs:
            print(f'`run_opengrid` is ignored')
            kwargs.pop('run_opengrid', None)
        if 'opengrid_only_scf' in kwargs:
            print(f'`opengrid_only_scf` is ignored')
            kwargs.pop('opengrid_only_scf', None)
    else:
        kwargs = {}

    builder = Wannier90BandsWorkChain.get_builder_from_protocol(
        codes, structure, run_opengrid=True, opengrid_only_scf=False, **kwargs
    )

    if isinstance(parent_folder, orm.RemoteStashFolderData):
        parent_folder = unstash(parent_folder)

    builder.pop('scf', None)
    builder.nscf['pw']['parent_folder'] = parent_folder

    return builder
