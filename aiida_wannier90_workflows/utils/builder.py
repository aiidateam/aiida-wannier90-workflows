#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Functions for manipulating builder."""
import typing as ty
from aiida import orm
from aiida.engine import ProcessBuilder, ProcessBuilderNamespace


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


def print_builder(builder: ProcessBuilderNamespace):
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
