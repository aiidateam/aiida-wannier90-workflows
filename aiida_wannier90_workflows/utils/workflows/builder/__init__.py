#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for manipulating builder."""
import typing as ty
import numpy as np

from aiida import orm
from aiida.common.lang import type_check
from aiida.engine import ProcessBuilder, ProcessBuilderNamespace

from aiida_quantumespresso.common.types import ElectronicType

from aiida_wannier90_workflows.common.types import WannierProjectionType, WannierDisentanglementType, WannierFrozenType


def serializer(node: orm.Node, show_pk: bool = True) -> ty.Any:  # pylint: disable=too-many-statements
    """Serialize arbitrary aiida object to ordinary python type, for pretty print.

    Usage:
    ```
    pprint.pprint(serializer(builder))
    ```

    :param node: arbitrary aiida node
    :type node: orm.Node
    """
    from aiida_pseudo.data.pseudo import UpfData

    # print(type(node), node)

    if isinstance(node, orm.Dict):
        res = {}
        for key, val in node.get_dict().items():
            res[key] = serializer(val, show_pk)

    elif isinstance(node, dict):
        res = {}
        for key, val in node.items():
            res[key] = serializer(val, show_pk)

    elif isinstance(node, ProcessBuilderNamespace):
        res = serializer(node._inputs(prune=True), show_pk)  # pylint: disable=protected-access

    elif isinstance(node, (orm.Float, orm.Bool, orm.Int, orm.Str, orm.BaseType)):
        res = node.value

    elif isinstance(node, orm.List):
        res = serializer(node.get_list())

    # BandsData is a subclass of KpointsData, need to before KpointsData
    elif isinstance(node, orm.BandsData):
        num_kpoints, num_bands = node.attributes['array|bands']
        res = f'nkpt={num_kpoints},nbnd={num_bands}'
        if 'labels' in node.attributes:
            res += f',{serialize_kpoints(node, show_pk)}'
        elif show_pk:
            res = f'{res}<{node.pk}>'

    elif isinstance(node, orm.KpointsData):
        res = serialize_kpoints(node, show_pk)

    elif isinstance(node, orm.Code):
        res = node.full_label
        if show_pk:
            res = f'{res}<{node.pk}>'

    elif isinstance(node, orm.StructureData):
        res = node.get_formula()
        if show_pk:
            res = f'{res}<{node.pk}>'

    elif isinstance(node, UpfData):
        res = node.filename
        if show_pk:
            res = f'{res}<{node.pk}>'

    elif isinstance(node, (orm.WorkflowNode, orm.CalculationNode)):
        res = node.process_label
        if show_pk:
            res = f'{res}<{node.pk}>'

    elif isinstance(node, orm.RemoteData):
        res = node.__class__.__name__
        if show_pk:
            res = f'{res}@{node.computer.label}<{node.pk}>'

    elif isinstance(node, orm.FolderData):
        res = node.__class__.__name__
        if show_pk:
            res = f'{res}<{node.pk}>'

    elif isinstance(node, orm.SinglefileData):
        res = node.__class__.__name__
        if show_pk:
            res = f'{res}<{node.pk}>'

    elif isinstance(node, range):
        res = list(node)

    # pytest_regressions.data_regression cannot dump np.array
    # https://github.com/ESSS/pytest-regressions/issues/26
    elif isinstance(node, (list, np.ndarray)):
        res = [serializer(_) for _ in node]

    # Is numpy type?
    # 'numpy' == 'numpy'
    elif type(node).__module__ == np.__name__:
        res = node.item()

    else:
        res = node

    return res


def serialize_numpy(array: ty.Union[list, np.ndarray]) -> list:
    """Serialize numpy array, list of numpy type to python list.

    Consider the following cases, e.g.
    1. np.array([...], dtype=np.int64)
    2. [np.int64, np.int64, ...]
    3. [int, int, ...]

    This is useful in the following case:
    pytest_regressions.data_regression cannot dump np.array
    https://github.com/ESSS/pytest-regressions/issues/26

    :param array: list, numpy array, or list of numpy type
    :type array: ty.Union[list, np.ndarray]
    :return: list of ordinary python type
    :rtype: list
    """

    if isinstance(array, np.ndarray):
        res = array.tolist()
    elif isinstance(array, list):
        res = []
        for i in array:
            # Is numpy type?
            # 'numpy' == 'numpy'
            if type(i).__module__ == np.__name__:
                res.append(i.item())
            else:
                res.append(i)
    else:
        raise ValueError(f'Unsupported type {type(array)} for {array}')

    return res


def serialize_kpoints(kpoints: orm.KpointsData, show_pk: bool = True) -> str:
    """Return str representation of KpointsData.

    :param kpoints: [description]
    :type kpoints: orm.KpointsData
    :return: [description]
    :rtype: str
    """
    if 'labels' in kpoints.attributes:
        res = '-'.join(kpoints.attributes['labels'])
    elif 'mesh' in kpoints.attributes:
        res = f"{kpoints.attributes['mesh']} mesh + {kpoints.attributes['offset']} offset"
    else:
        res = f"{kpoints.attributes['array|kpoints'][0]} kpts"

    if show_pk:
        res = f'{res}<{kpoints.pk}>'
    return res


def print_builder(builder: ty.Union[ProcessBuilderNamespace, dict]) -> None:
    """Pretty print builder.

    :param builder: [description]
    :type builder: ProcessBuilderNamespace
    """
    from pprint import pprint

    # To avoid empty dict, e.g.
    #  'relax': {'base': {'metadata': {},
    #                     'pw': {'metadata': {'options': {'stash': {}}},
    #                            'pseudos': {}}},
    #            'base_final_scf': {'metadata': {},
    #                               'pw': {'metadata': {'options': {'stash': {}}},
    #                                      'pseudos': {}}},
    #            'metadata': {}},
    if isinstance(builder, ProcessBuilderNamespace):
        pruned_builder = builder._inputs(prune=True)  # pylint: disable=protected-access
    else:
        pruned_builder = builder

    pprint(serializer(pruned_builder))


def submit_builder(builder: ProcessBuilder,
                   group_label: ty.Optional[str] = None,
                   dry_run: bool = True) -> ty.Optional[orm.ProcessNode]:
    """Submit builder and add to group.

    :param builder: [description]
    :type builder: ProcessBuilder
    :param group_label: [description], defaults to None
    :type group_label: ty.Optional[str], optional
    :param dry_run: [description], defaults to True
    :type dry_run: bool, optional
    :return: [description]
    :rtype: ty.Optional[orm.ProcessNode]
    """
    from aiida.engine import submit
    from colorama import Fore

    if dry_run:
        print(Fore.GREEN + '>>> Inputs of the builder' + Fore.RESET)
        print_builder(builder)
        print(Fore.GREEN + '>>> Use `dry_run=False` to submit' + Fore.RESET)
        if group_label:
            print('Destination group name: ' + Fore.GREEN + group_label + Fore.RESET)
    else:
        res = submit(builder)
        print(Fore.GREEN + f'>>> Submitted {res.process_label}<{res.pk}>' + Fore.RESET)

        if 'structure' in builder:
            structure = builder.structure
            if isinstance(structure, orm.StructureData):
                print(f'    for {structure.get_formula()}<{structure.pk}>')

        if group_label:
            group, _ = orm.Group.objects.get_or_create(group_label)
            group.add_nodes([res])

        return res


def check_codes(codes: ty.Mapping[str, ty.Union[str, int, orm.Code]]) -> ty.Mapping[str, orm.Code]:
    """Check and load pw.x, pw2wannier90.x, wannier90.x, projwfc.x, open_grid.x codes for Wannier workchain.

    :param codes: [description]
    :type codes: dict
    :raises ValueError: [description]
    :raises ValueError: [description]
    """
    required_codes = ('pw', 'pw2wannier90', 'wannier90')
    optional_codes = ('projwfc', 'opengrid')

    if not isinstance(codes, ty.Mapping):
        raise ValueError(
            f"`codes` must be a dict with the following required keys: `{'`, `'.join(required_codes)}` "
            f"and optional keys: `{'`, `'.join(optional_codes)}`"
        )

    for key in required_codes:
        if key not in codes.keys():
            raise ValueError(f'`codes` does not contain the required key: {key}')

    for key, code in codes.items():
        if isinstance(code, (int, str)):
            code = orm.load_code(code)
        type_check(code, orm.Code)
        codes[key] = code

    return codes


def guess_wannier_projection_types(
    electronic_type: ElectronicType,
    projection_type: WannierProjectionType = None,
    disentanglement_type: WannierDisentanglementType = None,
    frozen_type: WannierFrozenType = None
) -> ty.Tuple[WannierProjectionType, WannierDisentanglementType, WannierFrozenType]:
    """Automatically guess Wannier projection, disentanglement, and frozen types."""

    if electronic_type == ElectronicType.INSULATOR:
        if disentanglement_type is None:
            disentanglement_type = WannierDisentanglementType.NONE
        elif disentanglement_type == WannierDisentanglementType.NONE:
            pass
        else:
            raise ValueError(
                'For insulators there should be no disentanglement, '
                f'current disentanglement type: {disentanglement_type}'
            )
        if frozen_type is None:
            frozen_type = WannierFrozenType.NONE
        elif frozen_type == WannierFrozenType.NONE:
            pass
        else:
            raise ValueError(f'For insulators there should be no frozen states, current frozen type: {frozen_type}')
    elif electronic_type == ElectronicType.METAL:
        if projection_type == WannierProjectionType.SCDM:
            if disentanglement_type is None:
                # No disentanglement when using SCDM, otherwise the wannier interpolated bands are wrong
                disentanglement_type = WannierDisentanglementType.NONE
            elif disentanglement_type == WannierDisentanglementType.NONE:
                pass
            else:
                raise ValueError(
                    'For SCDM there should be no disentanglement, '
                    f'current disentanglement type: {disentanglement_type}'
                )
            if frozen_type is None:
                frozen_type = WannierFrozenType.NONE
            elif frozen_type == WannierFrozenType.NONE:
                pass
            else:
                raise ValueError(f'For SCDM there should be no frozen states, current frozen type: {frozen_type}')
        elif projection_type in [WannierProjectionType.ANALYTIC, WannierProjectionType.RANDOM]:
            if disentanglement_type is None:
                disentanglement_type = WannierDisentanglementType.SMV
            if frozen_type is None:
                frozen_type = WannierFrozenType.ENERGY_FIXED
            if disentanglement_type == WannierDisentanglementType.NONE and frozen_type != WannierFrozenType.NONE:
                raise ValueError(f'Disentanglement is explicitly disabled but frozen type {frozen_type} is required')
        elif projection_type in [
            WannierProjectionType.ATOMIC_PROJECTORS_QE, WannierProjectionType.ATOMIC_PROJECTORS_OPENMX
        ]:
            if disentanglement_type is None:
                disentanglement_type = WannierDisentanglementType.SMV
            if frozen_type is None:
                frozen_type = WannierFrozenType.FIXED_PLUS_PROJECTABILITY
            if disentanglement_type == WannierDisentanglementType.NONE and frozen_type != WannierFrozenType.NONE:
                raise ValueError(f'Disentanglement is explicitly disabled but frozen type {frozen_type} is required')
        else:
            if disentanglement_type is None or frozen_type is None:
                raise ValueError(
                    'Cannot automatically guess disentanglement and frozen types '
                    f'from projection type: {projection_type}'
                )
    else:
        raise ValueError(f'Not supported electronic type {electronic_type}')

    return projection_type, disentanglement_type, frozen_type


def recursive_merge_container(left: ty.Union[ty.Mapping, ty.Iterable],
                              right: ty.Union[ty.Mapping, ty.Iterable]) -> ty.Union[ty.Mapping, ty.Iterable]:
    """Merge right dict into left dict.

    If a key is present in both left and right, they are merge into one. Otherwise the right key will
    replace the left key.

    :param left: [description]
    :type left: ty.Union[ty.Mapping, ty.Iterable]
    :param right: [description]
    :type right: ty.Union[ty.Mapping, ty.Iterable]
    :return: [description]
    :rtype: ty.Union[ty.Mapping, ty.Iterable]
    """
    from collections import abc
    import copy

    if all(isinstance(_, orm.List) for _ in (left, right)):
        # I need to put orm.List before abc.Sequence since isinstance(a_orm.List_object, abc.Sequence) is True
        merged = orm.List(list=[*left.get_list(), *right.get_list()])
    elif all(isinstance(_, abc.Sequence) for _ in (left, right)):
        merged = [*left, *right]
    elif all(isinstance(_, abc.Mapping) for _ in (left, right)):
        merged = copy.copy(left)
        for key in right:
            if key in merged:
                merged[key] = recursive_merge_container(merged[key], right[key])
            else:
                merged[key] = right[key]
    elif all(isinstance(_, orm.Dict) for _ in (left, right)):
        right_dict = right.get_dict()
        merged = copy.copy(left.get_dict())
        for key in right_dict:
            if key in merged:
                merged[key] = recursive_merge_container(merged[key], right_dict[key])
            else:
                merged[key] = right_dict[key]
        merged = orm.Dict(dict=merged)
    elif isinstance(left, orm.Dict) and isinstance(right, abc.Mapping):
        merged = copy.copy(left.get_dict())
        for key in right:
            if key in merged:
                merged[key] = recursive_merge_container(merged[key], right[key])
            else:
                merged[key] = right[key]
        merged = orm.Dict(dict=merged)
    else:
        merged = right

    return merged


def recursive_merge_builder(builder: ProcessBuilderNamespace, right: ty.Mapping) -> ProcessBuilder:
    """Recursively merge a dictionaries into a ``ProcessBuilderNamespace``.

    If any key is present in both ``left`` and ``right`` dictionaries, the value of the ``right`` dictionary is
    merged into (instead of replace) the builder.

    :param builder: the builder
    :param right: second dictionary
    :return: the recursively merged builder
    """
    inputs = builder._inputs(prune=True)  # pylint: disable=protected-access
    inputs = recursive_merge_container(inputs, right)
    try:
        builder._update(inputs)  # pylint: disable=protected-access
    except ValueError as exc:
        raise ValueError(f'{exc}\n{builder=}\n{inputs=}') from exc

    return builder


def set_parallelization(  # pylint: disable=too-many-locals
    builder: ProcessBuilder, parallelization: dict = None
) -> ProcessBuilder:
    """Set parallelization for Wannier90BandsWorkChain.

    :param builder: [description]
    :type builder: ProcessBuilder
    :return: [description]
    :rtype: ProcessBuilder
    """
    from copy import deepcopy

    default_max_wallclock_seconds = 5 * 3600
    default_num_mpiprocs_per_machine = 1
    default_npool = 1
    default_num_machines = 1

    if parallelization is None:
        parallelization = {}

    max_wallclock_seconds = parallelization.get('max_wallclock_seconds', default_max_wallclock_seconds)
    num_mpiprocs_per_machine = parallelization.get('num_mpiprocs_per_machine', default_num_mpiprocs_per_machine)
    npool = parallelization.get('npool', default_npool)
    num_machines = parallelization.get('num_machines', default_num_machines)

    # I need to prune the builder, otherwise e.g. initially builder.relax is
    # an empty dict but the following code will change it to non-empty,
    # leading to invalid process spec such as code not found for PwRelaxWorkChain.
    pruned_builder = builder._inputs(prune=True)  # pylint: disable=protected-access
    if 'scf' in pruned_builder and 'parallelization' in builder.scf['pw']:
        pw_parallelization = builder.scf['pw']['parallelization'].get_dict()
    else:
        pw_parallelization = {}

    pw_parallelization['npool'] = npool
    metadata = get_metadata(
        num_mpiprocs_per_machine=num_mpiprocs_per_machine,
        max_wallclock_seconds=max_wallclock_seconds,
        num_machines=num_machines,
    )
    settings = get_settings_for_kpool(npool=npool)

    if 'relax' in pruned_builder:
        builder.relax['pw']['parallelization'] = orm.Dict(dict=pw_parallelization)
        builder.relax['pw']['metadata'] = metadata

    if 'scf' in pruned_builder:
        builder.scf['pw']['parallelization'] = orm.Dict(dict=pw_parallelization)
        builder.scf['pw']['metadata'] = metadata

    if 'nscf' in pruned_builder and 'pw' in builder.nscf:
        builder.nscf['pw']['parallelization'] = orm.Dict(dict=pw_parallelization)
        builder.nscf['pw']['metadata'] = metadata

    if 'opengrid' in pruned_builder:
        # builder.opengrid['metadata'] = metadata
        # builder.opengrid['settings'] = settings
        # For now opengrid has memory issue, I run it with less cores
        opengrid_metadata = deepcopy(metadata)
        opengrid_metadata['options']['resources']['num_mpiprocs_per_machine'] = (num_mpiprocs_per_machine // npool)
        builder.opengrid['opengrid']['metadata'] = opengrid_metadata

    if 'projwfc' in pruned_builder:
        builder.projwfc['projwfc']['metadata'] = metadata
        builder.projwfc['projwfc']['settings'] = settings

    builder.pw2wannier90['pw2wannier90']['metadata'] = metadata
    builder.pw2wannier90['pw2wannier90']['settings'] = settings

    builder.wannier90['wannier90']['metadata'] = metadata

    return builder


def get_metadata(num_mpiprocs_per_machine: int, max_wallclock_seconds: int, num_machines: int) -> dict:
    """Return metadata with the given number of mpiproces.

    :param num_mpiprocs_per_machine: [description]
    :type num_mpiprocs_per_machine: int
    :param max_wallclock_seconds: [description]
    :type max_wallclock_seconds: int
    :return: [description]
    :rtype: dict
    """
    metadata = {
        'options': {
            'resources': {
                'num_machines': num_machines,
                'num_mpiprocs_per_machine': num_mpiprocs_per_machine,
                # 'num_mpiprocs_per_machine': num_mpiprocs_per_machine // npool,
                # memory is not enough if I use 128 cores
                # 'num_mpiprocs_per_machine': 16,
                #
                # 'tot_num_mpiprocs': ,
                # 'num_cores_per_machine': ,
                # 'num_cores_per_mpiproc': ,
            },
            'max_wallclock_seconds': max_wallclock_seconds,
            'withmpi': True,
        }
    }

    return metadata


def get_settings_for_kpool(npool: int):
    """Return settings for kpool parallelization.

    :param npool: [description]
    :type npool: int
    :return: [description]
    :rtype: [type]
    """
    settings = orm.Dict(dict={'cmdline': ['-nk', f'{npool}']})

    return settings


def submit_and_add_group(builder: ProcessBuilder, group: orm.Group = None) -> None:
    """Submit a builder and add to a group.

    :param builder: the builder to be submitted.
    :type builder: ProcessBuilder
    :param group: the group to add the submitted workflow.
    :type group: orm.Group
    """
    from aiida.engine import submit

    result = submit(builder)
    print(f'Submitted {result.process_label}<{result.pk}>')

    if group:
        group.add_nodes([result])
        print(f'Added to group {group.label}<{group.pk}>')
