#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for manipulating builder."""
import typing as ty

from aiida import orm
from aiida.common.lang import type_check
from aiida.engine import ProcessBuilder, ProcessBuilderNamespace

from aiida_quantumespresso.common.types import ElectronicType

from aiida_wannier90_workflows.common.types import WannierProjectionType, WannierDisentanglementType, WannierFrozenType


def serializer(node: orm.Node) -> ty.Any:
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
            res[key] = serializer(val)

    elif isinstance(node, dict):
        res = {}
        for key, val in node.items():
            res[key] = serializer(val)

    elif isinstance(node, ProcessBuilderNamespace):
        res = serializer(node._inputs(prune=True))  # pylint: disable=protected-access

    elif isinstance(node, (orm.Float, orm.Bool, orm.Int, orm.Str, orm.BaseType)):
        res = node.value

    elif isinstance(node, orm.List):
        res = node.get_list()

    # BandsData is a subclass of KpointsData, need to before KpointsData
    elif isinstance(node, orm.BandsData):
        num_kpoints, num_bands = node.attributes['array|bands']
        res = f'nkpt={num_kpoints},nbnd={num_bands}'
        if 'labels' in node.attributes:
            res += f',{serialize_kpoints(node)}'

    elif isinstance(node, orm.KpointsData):
        res = serialize_kpoints(node)

    elif isinstance(node, orm.Code):
        res = f'{node.full_label}<{node.pk}>'

    elif isinstance(node, orm.StructureData):
        res = f'{node.get_formula()}<{node.pk}>'

    elif isinstance(node, UpfData):
        res = f'{node.filename}<{node.pk}>'

    elif isinstance(node, (orm.WorkflowNode, orm.CalculationNode)):
        res = f'{node.process_label}<{node.pk}>'

    elif isinstance(node, orm.RemoteData):
        res = f'{node.__class__.__name__}<{node.pk}>'

    elif isinstance(node, range):
        res = list(node)

    else:
        res = node

    return res


def serialize_kpoints(kpoints: orm.KpointsData) -> str:
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
        res = f"{kpoints.attributes['array|kpoints'][0]} points"

    res += f'<{kpoints.pk}>'
    return res


def print_builder(builder: ProcessBuilderNamespace) -> None:
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
    pruned_builder = builder._inputs(prune=True)  # pylint: disable=protected-access

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
    else:
        merged = right

    return merged


def recursive_merge_builder(builder: ProcessBuilderNamespace, right: ty.Mapping) -> ProcessBuilder:
    """Recursively merge a dictionaries into a ProcessBuilderNamespace.

    If any key is present in both ``left`` and ``right`` dictionaries, the value of the ``right`` dictionary is
    merged into (instead of replace) the builder.

    :param builder: the builder
    :param right: second dictionary
    :return: the recursively merged builder
    """
    inputs = builder._inputs(prune=True)  # pylint: disable=protected-access
    inputs = recursive_merge_container(inputs, right)
    builder.update(inputs)

    return builder
