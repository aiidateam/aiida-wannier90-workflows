#!/usr/bin/env python
"""Functions for manipulating builder."""
import typing as ty

from aiida import orm
from aiida.common.lang import type_check
from aiida.engine import ProcessBuilder, ProcessBuilderNamespace


def submit_builder(  # pylint: disable=inconsistent-return-statements
    builder: ProcessBuilder, group_label: ty.Optional[str] = None, dry_run: bool = True
) -> ty.Optional[orm.ProcessNode]:
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
    from colorama import Fore

    from aiida.engine import submit

    from aiida_wannier90_workflows.utils.workflows.builder.serializer import (
        print_builder,
    )

    if dry_run:
        print(Fore.GREEN + ">>> Inputs of the builder" + Fore.RESET)
        print_builder(builder)
        print(Fore.GREEN + ">>> Use `dry_run=False` to submit" + Fore.RESET)
        if group_label:
            print("Destination group name: " + Fore.GREEN + group_label + Fore.RESET)
    else:
        res = submit(builder)
        print(Fore.GREEN + f">>> Submitted {res.process_label}<{res.pk}>" + Fore.RESET)

        if "structure" in builder:
            structure = builder.structure
            if isinstance(structure, orm.StructureData):
                print(f"    for {structure.get_formula()}<{structure.pk}>")

        if group_label:
            group, _ = orm.Group.objects.get_or_create(group_label)
            group.add_nodes([res])

        return res


def check_codes(
    codes: ty.Mapping[str, ty.Union[str, int, orm.Code]]
) -> ty.Mapping[str, orm.Code]:
    """Check and load pw.x, pw2wannier90.x, wannier90.x, projwfc.x, open_grid.x codes for Wannier workchain.

    :param codes: [description]
    :type codes: dict
    :raises ValueError: [description]
    :raises ValueError: [description]
    """
    required_codes = ("pw", "pw2wannier90", "wannier90")
    optional_codes = ("projwfc", "open_grid")

    if not isinstance(codes, ty.Mapping):
        raise ValueError(
            f"`codes` must be a dict with the following required keys: `{'`, `'.join(required_codes)}` "
            f"and optional keys: `{'`, `'.join(optional_codes)}`"
        )

    for key in required_codes:
        if key not in codes.keys():
            raise ValueError(f"`codes` does not contain the required key: {key}")

    for key, code in codes.items():
        if isinstance(code, (int, str)):
            code = orm.load_code(code)
        type_check(code, orm.Code)
        codes[key] = code

    return codes


def recursive_merge_container(
    left: ty.Union[ty.Mapping, ty.Iterable], right: ty.Union[ty.Mapping, ty.Iterable]
) -> ty.Union[ty.Mapping, ty.Iterable]:
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
    # pylint: disable=too-many-branches
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
        merged = orm.Dict(merged)
    elif isinstance(left, orm.Dict) and isinstance(right, abc.Mapping):
        merged = copy.copy(left.get_dict())
        for key in right:
            if key in merged:
                merged[key] = recursive_merge_container(merged[key], right[key])
            else:
                merged[key] = right[key]
        merged = orm.Dict(merged)
    else:
        merged = right

    return merged


def recursive_merge_builder(
    builder: ProcessBuilderNamespace, right: ty.Mapping
) -> ProcessBuilder:
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
        raise ValueError(f"{exc}\n{builder=}\n{inputs=}") from exc

    return builder


def submit_and_add_group(
    builder: ProcessBuilder, group: orm.Group = None
) -> orm.ProcessNode:
    """Submit a builder and add to a group.

    :param builder: the builder to be submitted.
    :type builder: ProcessBuilder
    :param group: the group to add the submitted workflow.
    :type group: orm.Group
    """
    from aiida.engine import submit

    result = submit(builder)
    print(f"Submitted {result.process_label}<{result.pk}>")

    if group:
        group.add_nodes([result])
        print(f"Added to group {group.label}<{group.pk}>")

    return result
