#!/usr/bin/env python
import typing
from aiida import orm
from aiida.engine import ProcessBuilder, ProcessBuilderNamespace


def serializer(node: orm.Node) -> typing.Any:
    """transform arbitrary aiida object to ordinary python type, for pretty print.

    Usage:
    ```
    pprint.pprint(node_serializer(builder))
    ```

    :param node: arbitrary aiida node
    :type node: orm.Node
    """
    from aiida_pseudo.data.pseudo import UpfData

    # print(type(node), node)

    if isinstance(node, orm.Dict):
        res = {}
        for k, v in node.get_dict().items():
            res[k] = serializer(v)
    elif isinstance(node, dict):
        res = {}
        for k, v in node.items():
            res[k] = serializer(v)
    elif isinstance(node, ProcessBuilderNamespace):
        res = {}
        for k in node:
            res[k] = serializer(node[k])
    elif isinstance(node, (orm.Float, orm.Bool, orm.Int, orm.Str)):
        res = node.value
    elif isinstance(node, orm.Code):
        res = f'{node.label}@{node.computer.label}<{node.pk}>'
    elif isinstance(node, orm.StructureData):
        res = f'{node.get_formula()}<{node.pk}>'
    elif isinstance(node, UpfData):
        res = f'{node.filename}<{node.pk}>'
    else:
        res = node

    return res


def print_builder(builder: ProcessBuilderNamespace):
    from pprint import pprint

    pprint(serializer(builder))


def submit_builder(builder: ProcessBuilder,
                   group_label: typing.Optional[str] = None,
                   dry_run: bool = True) -> typing.Optional[orm.ProcessNode]:
    from aiida.engine import submit
    from colorama import Fore

    if dry_run:
        print(Fore.GREEN + ">>> Inputs of the builder" + Fore.RESET)
        print_builder(builder)
        print(Fore.GREEN + ">>> Use `dry_run=False` to submit" + Fore.RESET)
        if group_label:
            print("Destination group name: " + Fore.GREEN + group_label +
                  Fore.RESET)
    else:
        res = submit(builder)
        print(Fore.GREEN + f">>> Submitted {res.process_label}<{res.pk}>" +
              Fore.RESET)

        if 'structure' in builder:
            structure = builder.structure
            if isinstance(structure, orm.StructureData):
                print(f'    for {structure.get_formula()}<{structure.pk}>')

        if group_label:
            group, _ = orm.Group.objects.get_or_create(group_label)
            group.add_nodes([res])

        return res
