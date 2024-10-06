"""Functions for serialize builder."""

import typing as ty

import numpy as np

from aiida import orm
from aiida.engine import ProcessBuilderNamespace


def serialize(node: orm.Node, show_pk: bool = True) -> ty.Any:
    """Serialize arbitrary aiida object to ordinary python type, for pretty print.

    Usage:
    ```
    pprint.pprint(serialize(builder))
    ```

    :param node: arbitrary aiida node
    :type node: orm.Node
    """
    # pylint: disable=too-many-statements,too-many-branches
    from aiida_pseudo.data.pseudo import UpfData

    # print(type(node), node)

    if isinstance(node, orm.Dict):
        res = {}
        for key, val in node.get_dict().items():
            res[key] = serialize(val, show_pk)

    elif isinstance(node, dict):
        res = {}
        for key, val in node.items():
            res[key] = serialize(val, show_pk)

    elif isinstance(node, ProcessBuilderNamespace):
        res = serialize(
            node._inputs(prune=True), show_pk  # pylint: disable=protected-access
        )

    elif isinstance(node, (orm.Float, orm.Bool, orm.Int, orm.Str, orm.BaseType)):
        res = node.value

    elif isinstance(node, orm.List):
        res = serialize(node.get_list())

    # BandsData is a subclass of KpointsData, need to before KpointsData
    elif isinstance(node, orm.BandsData):
        num_kpoints, num_bands = node.base.attributes.all["array|bands"]
        res = f"nkpt={num_kpoints},nbnd={num_bands}"
        if "labels" in node.base.attributes.all:
            res += f",{serialize_kpoints(node, show_pk)}"
        elif show_pk:
            res = f"{res}<{node.pk}>"

    elif isinstance(node, orm.KpointsData):
        res = serialize_kpoints(node, show_pk)

    elif isinstance(node, orm.Code):
        res = node.full_label
        if show_pk:
            res = f"{res}<{node.pk}>"

    elif isinstance(node, orm.StructureData):
        res = node.get_formula()
        if show_pk:
            res = f"{res}<{node.pk}>"

    elif isinstance(node, UpfData):
        res = node.filename
        if show_pk:
            res = f"{res}<{node.pk}>"

    elif isinstance(node, (orm.WorkflowNode, orm.CalculationNode)):
        res = node.process_label
        if show_pk:
            res = f"{res}<{node.pk}>"

    elif isinstance(node, orm.RemoteData):
        res = node.__class__.__name__
        if show_pk:
            res = f"{res}@{node.computer.label}<{node.pk}>"

    elif isinstance(node, orm.FolderData):
        res = node.__class__.__name__
        if show_pk:
            res = f"{res}<{node.pk}>"

    elif isinstance(node, orm.SinglefileData):
        res = node.__class__.__name__
        if show_pk:
            res = f"{res}<{node.pk}>"

    elif isinstance(node, range):
        res = list(node)

    # pytest_regressions.data_regression cannot dump np.array
    # https://github.com/ESSS/pytest-regressions/issues/26
    elif isinstance(node, (list, np.ndarray)):
        res = [serialize(_) for _ in node]

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
        raise ValueError(f"Unsupported type {type(array)} for {array}")

    return res


def serialize_kpoints(kpoints: orm.KpointsData, show_pk: bool = True) -> str:
    """Return str representation of KpointsData.

    :param kpoints: [description]
    :type kpoints: orm.KpointsData
    :return: [description]
    :rtype: str
    """
    if "labels" in kpoints.base.attributes.all:
        res = "-".join(kpoints.base.attributes.all["labels"])
    elif "mesh" in kpoints.base.attributes.all:
        res = f"{kpoints.base.attributes.all['mesh']} mesh + {kpoints.base.attributes.all['offset']} offset"
    else:
        res = f"{kpoints.base.attributes.all['array|kpoints'][0]} kpts"

    if show_pk:
        res = f"{res}<{kpoints.pk}>"
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

    pprint(serialize(pruned_builder))
