#!/usr/bin/env python
"""Functions for manipulating builder."""
import typing as ty

import numpy as np

from aiida import orm
from aiida.common import AttributeDict
from aiida.common.lang import type_check
from aiida.engine import ProcessBuilder, ProcessBuilderNamespace

from aiida_quantumespresso.calculations import BasePwCpInputGenerator
from aiida_quantumespresso.calculations.namelists import NamelistsCalculation
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.common.types import ElectronicType
from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain

# from aiida_quantumespresso.calculations.opengrid import OpengridCalculation
# from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation
# from aiida_quantumespresso.calculations.pw2wannier90 import Pw2wannier90Calculation
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

from aiida_wannier90.calculations import Wannier90Calculation

from aiida_wannier90_workflows.common.types import (
    WannierDisentanglementType,
    WannierFrozenType,
    WannierProjectionType,
)
from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain
from aiida_wannier90_workflows.workflows.base.opengrid import OpengridBaseWorkChain
from aiida_wannier90_workflows.workflows.base.projwfc import ProjwfcBaseWorkChain
from aiida_wannier90_workflows.workflows.base.pw2wannier90 import (
    Pw2wannier90BaseWorkChain,
)
from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain
from aiida_wannier90_workflows.workflows.wannier90 import Wannier90WorkChain


def serializer(node: orm.Node, show_pk: bool = True) -> ty.Any:
    """Serialize arbitrary aiida object to ordinary python type, for pretty print.

    Usage:
    ```
    pprint.pprint(serializer(builder))
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
            res[key] = serializer(val, show_pk)

    elif isinstance(node, dict):
        res = {}
        for key, val in node.items():
            res[key] = serializer(val, show_pk)

    elif isinstance(node, ProcessBuilderNamespace):
        res = serializer(
            node._inputs(prune=True), show_pk  # pylint: disable=protected-access
        )

    elif isinstance(node, (orm.Float, orm.Bool, orm.Int, orm.Str, orm.BaseType)):
        res = node.value

    elif isinstance(node, orm.List):
        res = serializer(node.get_list())

    # BandsData is a subclass of KpointsData, need to before KpointsData
    elif isinstance(node, orm.BandsData):
        num_kpoints, num_bands = node.attributes["array|bands"]
        res = f"nkpt={num_kpoints},nbnd={num_bands}"
        if "labels" in node.attributes:
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
        raise ValueError(f"Unsupported type {type(array)} for {array}")

    return res


def serialize_kpoints(kpoints: orm.KpointsData, show_pk: bool = True) -> str:
    """Return str representation of KpointsData.

    :param kpoints: [description]
    :type kpoints: orm.KpointsData
    :return: [description]
    :rtype: str
    """
    if "labels" in kpoints.attributes:
        res = "-".join(kpoints.attributes["labels"])
    elif "mesh" in kpoints.attributes:
        res = (
            f"{kpoints.attributes['mesh']} mesh + {kpoints.attributes['offset']} offset"
        )
    else:
        res = f"{kpoints.attributes['array|kpoints'][0]} kpts"

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

    pprint(serializer(pruned_builder))


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
    optional_codes = ("projwfc", "opengrid")

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


def guess_wannier_projection_types(
    electronic_type: ElectronicType,
    projection_type: WannierProjectionType = None,
    disentanglement_type: WannierDisentanglementType = None,
    frozen_type: WannierFrozenType = None,
) -> ty.Tuple[WannierProjectionType, WannierDisentanglementType, WannierFrozenType]:
    """Automatically guess Wannier projection, disentanglement, and frozen types."""
    # pylint: disable=too-many-branches

    if electronic_type == ElectronicType.INSULATOR:
        if disentanglement_type is None:
            disentanglement_type = WannierDisentanglementType.NONE
        elif disentanglement_type == WannierDisentanglementType.NONE:
            pass
        else:
            raise ValueError(
                "For insulators there should be no disentanglement, "
                f"current disentanglement type: {disentanglement_type}"
            )
        if frozen_type is None:
            frozen_type = WannierFrozenType.NONE
        elif frozen_type == WannierFrozenType.NONE:
            pass
        else:
            raise ValueError(
                f"For insulators there should be no frozen states, current frozen type: {frozen_type}"
            )
    elif electronic_type == ElectronicType.METAL:
        if projection_type == WannierProjectionType.SCDM:
            if disentanglement_type is None:
                # No disentanglement when using SCDM, otherwise the wannier interpolated bands are wrong
                disentanglement_type = WannierDisentanglementType.NONE
            elif disentanglement_type == WannierDisentanglementType.NONE:
                pass
            else:
                raise ValueError(
                    "For SCDM there should be no disentanglement, "
                    f"current disentanglement type: {disentanglement_type}"
                )
            if frozen_type is None:
                frozen_type = WannierFrozenType.NONE
            elif frozen_type == WannierFrozenType.NONE:
                pass
            else:
                raise ValueError(
                    f"For SCDM there should be no frozen states, current frozen type: {frozen_type}"
                )
        elif projection_type in [
            WannierProjectionType.ANALYTIC,
            WannierProjectionType.RANDOM,
        ]:
            if disentanglement_type is None:
                disentanglement_type = WannierDisentanglementType.SMV
            if frozen_type is None:
                frozen_type = WannierFrozenType.ENERGY_FIXED
            if (
                disentanglement_type == WannierDisentanglementType.NONE
                and frozen_type != WannierFrozenType.NONE
            ):
                raise ValueError(
                    f"Disentanglement is explicitly disabled but frozen type {frozen_type} is required"
                )
        elif projection_type in [
            WannierProjectionType.ATOMIC_PROJECTORS_QE,
            WannierProjectionType.ATOMIC_PROJECTORS_OPENMX,
        ]:
            if disentanglement_type is None:
                disentanglement_type = WannierDisentanglementType.SMV
            if frozen_type is None:
                frozen_type = WannierFrozenType.FIXED_PLUS_PROJECTABILITY
            if (
                disentanglement_type == WannierDisentanglementType.NONE
                and frozen_type != WannierFrozenType.NONE
            ):
                raise ValueError(
                    f"Disentanglement is explicitly disabled but frozen type {frozen_type} is required"
                )
        else:
            if disentanglement_type is None or frozen_type is None:
                raise ValueError(
                    "Cannot automatically guess disentanglement and frozen types "
                    f"from projection type: {projection_type}"
                )
    else:
        raise ValueError(f"Not supported electronic type {electronic_type}")

    return projection_type, disentanglement_type, frozen_type


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


def set_parallelization(
    builder: ty.Union[ProcessBuilder, ProcessBuilderNamespace, AttributeDict],
    parallelization: dict = None,
    process_class: ty.Union[
        PwCalculation,
        PwBaseWorkChain,
        PwRelaxWorkChain,
        PwBandsWorkChain,
        Wannier90Calculation,
        Wannier90BaseWorkChain,
        Wannier90BandsWorkChain,
    ] = Wannier90BandsWorkChain,
) -> None:
    """Set parallelization for Wannier90BandsWorkChain.

    :param builder: a builder or its subport, or a ``AttributeDict`` which is the inputs for the builder.
    :type builder: ProcessBuilderNamespace
    """
    # pylint: disable=too-many-branches,too-many-statements,too-many-locals
    default_max_wallclock_seconds = 12 * 3600
    default_num_mpiprocs_per_machine = None
    default_npool = 1
    default_num_machines = 1
    default_queue_name = None
    default_account = None

    if parallelization is None:
        parallelization = {}

    max_wallclock_seconds = parallelization.get(
        "max_wallclock_seconds",
        default_max_wallclock_seconds,
    )
    num_mpiprocs_per_machine = parallelization.get(
        "num_mpiprocs_per_machine",
        default_num_mpiprocs_per_machine,
    )
    npool = parallelization.get(
        "npool",
        default_npool,
    )
    num_machines = parallelization.get(
        "num_machines",
        default_num_machines,
    )
    queue_name = parallelization.get(
        "queue_name",
        default_queue_name,
    )
    account = parallelization.get(
        "account",
        default_account,
    )

    # I need to prune the builder, otherwise e.g. initially builder.relax is
    # an empty dict but the following code will change it to non-empty,
    # leading to invalid process spec such as code not found for PwRelaxWorkChain.
    pruned_builder = builder._inputs(prune=True)  # pylint: disable=protected-access

    metadata = get_metadata(
        num_mpiprocs_per_machine=num_mpiprocs_per_machine,
        max_wallclock_seconds=max_wallclock_seconds,
        num_machines=num_machines,
        queue_name=queue_name,
        account=account,
    )
    settings = get_settings_for_kpool(npool=npool)

    # PwCalculation is a subclass of BasePwCpInputGenerator,
    # `parallelization` is defined in BasePwCpInputGenerator
    if issubclass(process_class, BasePwCpInputGenerator):
        if "parallelization" in builder:
            base_parallelization = builder["parallelization"].get_dict()
        else:
            base_parallelization = {}

        base_parallelization["npool"] = npool

        builder["parallelization"] = orm.Dict(dict=base_parallelization)
        builder["metadata"] = metadata

    elif process_class == PwBaseWorkChain:
        set_parallelization(
            builder["pw"],
            parallelization=parallelization,
            process_class=BasePwCpInputGenerator,
        )

    # Includes PwCalculation, OpengridCalculation, ProjwfcCalculation
    elif issubclass(process_class, NamelistsCalculation):
        builder["metadata"] = metadata
        builder["settings"] = settings

    elif process_class == OpengridBaseWorkChain:
        set_parallelization(
            builder["opengrid"],
            parallelization=parallelization,
            process_class=NamelistsCalculation,
        )
        # For now opengrid has memory issue, I run it with less cores
        # opengrid_metadata = copy.deepcopy(metadata)
        # opengrid_metadata['options']['resources']['num_mpiprocs_per_machine'] = (num_mpiprocs_per_machine // npool)
        # builder.opengrid['opengrid']['metadata'] = opengrid_metadata

    elif process_class == ProjwfcBaseWorkChain:
        set_parallelization(
            builder["projwfc"],
            parallelization=parallelization,
            process_class=NamelistsCalculation,
        )

    elif process_class == Pw2wannier90BaseWorkChain:
        set_parallelization(
            builder["pw2wannier90"],
            parallelization=parallelization,
            process_class=NamelistsCalculation,
        )

    elif process_class == Wannier90Calculation:
        builder["metadata"] = metadata

    elif process_class == Wannier90BaseWorkChain:
        set_parallelization(
            builder["wannier90"],
            parallelization=parallelization,
            process_class=Wannier90Calculation,
        )

    elif process_class == PwRelaxWorkChain:
        if "base" in pruned_builder:
            set_parallelization(
                builder["base"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )
        if "base_final_scf" in pruned_builder:
            set_parallelization(
                builder["base_final_scf"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )

    elif process_class == PwBandsWorkChain:

        if "relax" in pruned_builder:
            set_parallelization(
                builder["relax"],
                parallelization=parallelization,
                process_class=PwRelaxWorkChain,
            )

        if "scf" in pruned_builder:
            set_parallelization(
                builder["scf"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )

        if "bands" in pruned_builder:
            set_parallelization(
                builder["bands"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )

    elif process_class == Wannier90BandsWorkChain:

        if "scf" in pruned_builder:
            set_parallelization(
                builder["scf"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )

        if "nscf" in pruned_builder:
            set_parallelization(
                builder["nscf"],
                parallelization=parallelization,
                process_class=PwBaseWorkChain,
            )

        if "opengrid" in pruned_builder:
            set_parallelization(
                builder["opengrid"],
                parallelization=parallelization,
                process_class=OpengridBaseWorkChain,
            )

        if "projwfc" in pruned_builder:
            set_parallelization(
                builder["projwfc"],
                parallelization=parallelization,
                process_class=ProjwfcBaseWorkChain,
            )

        if "pw2wannier90" in pruned_builder:
            set_parallelization(
                builder["pw2wannier90"],
                parallelization=parallelization,
                process_class=Pw2wannier90BaseWorkChain,
            )

        if "wannier90" in pruned_builder:
            set_parallelization(
                builder["wannier90"],
                parallelization=parallelization,
                process_class=Wannier90BaseWorkChain,
            )


def get_metadata(
    *,
    num_mpiprocs_per_machine: int = None,
    max_wallclock_seconds: int = 24 * 3600,
    num_machines: int = 1,
    queue_name: str = None,
    account: str = None,
    **_,
) -> dict:
    """Return metadata with the given MPI specification.

    Usage
    ```
    # Create a dict
    parallelization = {
        'max_wallclock_seconds': 1800,
        'num_machines': 10,
        'npool': 3*10,
        'num_mpiprocs_per_machine': 12,
        'queue_name': 'debug',
        'account': 'mr0',
    }
    # The dict can be used by
    # set_parallelization(builder, parallelization)
    # Or
    metadata = get_metadata(**parallelization)
    ```

    :param num_mpiprocs_per_machine: defaults to None, meaning it is not set
    and will the default number of CPUs in the `computer` configuration.
    :type num_mpiprocs_per_machine: int, optional
    :param max_wallclock_seconds: defaults to 24*3600
    :type max_wallclock_seconds: int, optional
    :param num_machines: defaults to 1
    :type num_machines: int, optional
    :param queue_name: slurm queue name
    :type queue_name: str, optional
    :param account: slurm account
    :type account: str, optional
    :return: metadata dict
    :rtype: dict
    """
    metadata = {
        "options": {
            "resources": {
                "num_machines": num_machines,
                # 'num_mpiprocs_per_machine': num_mpiprocs_per_machine,
                # 'num_mpiprocs_per_machine': num_mpiprocs_per_machine // npool,
                # memory is not enough if I use 128 cores
                # 'num_mpiprocs_per_machine': 16,
                #
                # 'tot_num_mpiprocs': ,
                # 'num_cores_per_machine': ,
                # 'num_cores_per_mpiproc': ,
            },
            "max_wallclock_seconds": max_wallclock_seconds,
            "withmpi": True,
        }
    }
    if num_mpiprocs_per_machine:
        metadata["options"]["resources"][
            "num_mpiprocs_per_machine"
        ] = num_mpiprocs_per_machine
    if queue_name:
        metadata["options"]["queue_name"] = queue_name
    if queue_name:
        metadata["options"]["account"] = account

    return metadata


def get_settings_for_kpool(npool: int):
    """Return settings for kpool parallelization.

    :param npool: [description]
    :type npool: int
    :return: [description]
    :rtype: [type]
    """
    settings = orm.Dict(dict={"cmdline": ["-nk", f"{npool}"]})

    return settings


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


def guess_process_class_from_builder(
    builder: ty.Union[ProcessBuilder, ProcessBuilderNamespace]
) -> orm.ProcessNode:
    """Try to guess the process class of the ``builder``.

    May fail.

    :param builder: a builder or its subport
    :type builder: ty.Union[ProcessBuilder, ProcessBuilderNamespace]
    :return: its process class, e.g. ``Wannier90BandsWorkChain``
    :rtype: orm.ProcessNode
    """
    supported_classes = [
        Wannier90Calculation,
        Wannier90BaseWorkChain,
        Wannier90BandsWorkChain,
        PwCalculation,
        PwBaseWorkChain,
        PwBandsWorkChain,
    ]

    valid_fields = set(builder._valid_fields)  # pylint: disable=protected-access

    for proc in supported_classes:
        fields = set(proc.spec().inputs.keys())
        # print(f"{proc} {field=}")
        if valid_fields in fields:
            process_class = proc
            break
    else:
        raise ValueError(
            f"{builder=}\nis not one of supported classes: {supported_classes}"
        )

    return process_class


def set_kpoints(
    builder: ty.Union[ProcessBuilder, ProcessBuilderNamespace, AttributeDict],
    kpoints: orm.KpointsData,
    process_class: ty.Union[
        PwCalculation,
        PwBaseWorkChain,
        Wannier90Calculation,
        Wannier90BaseWorkChain,
        Wannier90BandsWorkChain,
    ] = None,
) -> None:
    """Set ``kpoints`` and ``mp_grid`` of e.g. ``Wannier90BaseWorkChain``.

    :param builder: a builder or its subport, or a ``AttributeDict`` which is the inputs for the builder.
    :type builder: ProcessBuilderNamespace
    :param kpoints: a kpoints mesh or a list of kpoints
    :type kpoints: orm.KpointsData
    :param process_class: WorkChain class of the builder
    :type process_class: Wannier90Calculation, Wannier90BaseWorkChain, Wannier90BandsWorkChain
    """
    from aiida_wannier90_workflows.utils.kpoints import (
        get_explicit_kpoints,
        get_mesh_from_kpoints,
    )

    if process_class is None:
        process_class = guess_process_class_from_builder(builder)

    if process_class not in (
        PwCalculation,
        PwBaseWorkChain,
        Wannier90Calculation,
        Wannier90BaseWorkChain,
        Wannier90BandsWorkChain,
    ) and not issubclass(process_class, Wannier90WorkChain):
        raise ValueError(f"Not supported process_class {process_class}")

    if process_class == Wannier90Calculation:
        # Test if it is a mesh
        try:
            kpoints.get_kpoints_mesh()
        except AttributeError:
            kpoints_explicit = kpoints
        else:
            kpoints_explicit = get_explicit_kpoints(kpoints)
        mp_grid = get_mesh_from_kpoints(kpoints_explicit)

        builder["kpoints"] = kpoints_explicit
        params = builder["parameters"].get_dict()
        params["mp_grid"] = mp_grid
        builder["parameters"] = orm.Dict(dict=params)

    elif process_class == Wannier90BaseWorkChain:
        set_kpoints(builder["wannier90"], kpoints, process_class=Wannier90Calculation)

    elif issubclass(process_class, Wannier90WorkChain):
        set_kpoints(builder["wannier90"], kpoints, process_class=Wannier90BaseWorkChain)

        kpoints_explicit = builder["wannier90"]["wannier90"]["kpoints"]
        set_kpoints(builder["scf"], kpoints, process_class=PwBaseWorkChain)
        set_kpoints(builder["nscf"], kpoints_explicit, process_class=PwBaseWorkChain)

    elif process_class == PwCalculation:
        builder["kpoints"] = kpoints

    elif process_class == PwBaseWorkChain:
        builder["kpoints"] = kpoints

    elif process_class == PwBandsWorkChain:
        set_kpoints(builder["bands"], kpoints, process_class=PwBaseWorkChain)


def set_num_bands(
    builder: ty.Union[ProcessBuilder, ProcessBuilderNamespace, AttributeDict],
    num_bands: int,
    exclude_bands: ty.Sequence = None,
    process_class: ty.Union[
        PwCalculation,
        PwBaseWorkChain,
        PwBandsWorkChain,
        Wannier90Calculation,
        Wannier90BaseWorkChain,
        Wannier90BandsWorkChain,
    ] = None,
) -> None:
    """Set number of bands of for various ``WorkChain``.

    :param builder: a builder or its subport, or a ``AttributeDict`` which is the inputs for the builder.
    :type builder: ProcessBuilderNamespace
    :param num_bands: number of bands, including excluded bands.
    Specifically, this is the ``nbnd`` for ``pw.x``, NOT the `num_bands` for ``wannier90.x``.
    :type num_bands: int
    :param exclude_bands: a list of band index to be excluded, starting from 1.
    Only useful for wannier90 related ``WorkChain``s.
    :type exclude_bands: ty.Sequence
    :param process_class: WorkChain class of the builder
    :type process_class: Wannier90Calculation, Wannier90BaseWorkChain, Wannier90BandsWorkChain
    """
    if process_class is None:
        process_class = guess_process_class_from_builder(builder)

    if process_class not in (
        PwCalculation,
        PwBaseWorkChain,
        PwBandsWorkChain,
        Wannier90Calculation,
        Wannier90BaseWorkChain,
        Wannier90BandsWorkChain,
    ) and not issubclass(process_class, Wannier90WorkChain):
        raise ValueError(f"Not supported process_class {process_class}")

    if process_class == Wannier90Calculation:
        # W90 needs to subtract excluded bands
        params = builder["parameters"].get_dict()
        if exclude_bands:
            num_bands -= len(exclude_bands)
            params["exclude_bands"] = exclude_bands
        params["num_bands"] = num_bands
        builder["parameters"] = orm.Dict(dict=params)

    elif process_class == Wannier90BaseWorkChain:
        set_num_bands(
            builder["wannier90"],
            num_bands,
            exclude_bands,
            process_class=Wannier90Calculation,
        )

    elif issubclass(process_class, Wannier90WorkChain):
        set_num_bands(
            builder["wannier90"],
            num_bands,
            exclude_bands,
            process_class=Wannier90BaseWorkChain,
        )
        set_num_bands(builder["nscf"], num_bands, process_class=PwBaseWorkChain)

    elif process_class == PwCalculation:
        params = builder["parameters"].get_dict()
        params["SYSTEM"]["nbnd"] = num_bands
        builder["parameters"] = orm.Dict(dict=params)

    elif process_class == PwBaseWorkChain:
        set_num_bands(builder["pw"], num_bands, process_class=PwCalculation)

    elif process_class == PwBandsWorkChain:
        set_num_bands(builder["bands"], num_bands, process_class=PwBaseWorkChain)
