"""Functions for changing builder."""

import typing as ty

from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import CalcJob, ProcessBuilder, ProcessBuilderNamespace

from aiida_quantumespresso.calculations import BasePwCpInputGenerator
from aiida_quantumespresso.calculations.namelists import NamelistsCalculation
from aiida_quantumespresso.calculations.pw import PwCalculation
from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain

# from aiida_quantumespresso.calculations.open_grid import OpenGridCalculation
# from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation
# from aiida_quantumespresso.calculations.pw2wannier90 import Pw2wannier90Calculation
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

from aiida_wannier90.calculations import Wannier90Calculation

from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain
from aiida_wannier90_workflows.workflows.base.open_grid import OpenGridBaseWorkChain
from aiida_wannier90_workflows.workflows.base.projwfc import ProjwfcBaseWorkChain
from aiida_wannier90_workflows.workflows.base.pw2wannier90 import (
    Pw2wannier90BaseWorkChain,
)
from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain
from aiida_wannier90_workflows.workflows.optimize import Wannier90OptimizeWorkChain
from aiida_wannier90_workflows.workflows.projwfcbands import ProjwfcBandsWorkChain
from aiida_wannier90_workflows.workflows.wannier90 import Wannier90WorkChain

try:
    from aiida_hyperqueue.scheduler import HyperQueueScheduler

    aiida_hq_installed = True
except ModuleNotFoundError:
    aiida_hq_installed = False


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
        Wannier90OptimizeWorkChain,
        CalcJob,
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
    default_code = None
    if aiida_hq_installed:
        # Default value setting only for AiiDA-HyperQueue
        default_num_cpus = 1
        # If do not set num_cpus, It will get num_mpiprocs_per_machine as num_cpus.
        # As the num_mpiprocs_per_machine can be None and num_cpus must be int,
        # we have to set an int to num_cpus.
        default_memory_mb = None  # Use all availble the memory on the worke

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
    code = builder.get(
        "code",
        default_code,
    )
    if aiida_hq_installed:
        num_cpus = parallelization.get(
            "num_cpus",
            default_num_cpus,
        )
        memory_mb = parallelization.get(
            "memory_mb",
            default_memory_mb,
        )

    # I need to prune the builder, otherwise e.g. initially builder.relax is
    # an empty dict but the following code will change it to non-empty,
    # leading to invalid process spec such as code not found for PwRelaxWorkChain.
    if isinstance(builder, dict):
        pruned_builder = builder
    else:
        pruned_builder = builder._inputs(prune=True)  # pylint: disable=protected-access

    run_hyperqueue = False
    if code is not None:
        run_hyperqueue = isinstance(code.computer.get_scheduler(), HyperQueueScheduler)

    if run_hyperqueue:
        if aiida_hq_installed:
            metadata = get_metadata_hq(
                num_cpus=num_cpus,
                max_wallclock_seconds=max_wallclock_seconds,
                memory_mb=memory_mb,
                queue_name=queue_name,
                account=account,
            )
        else:  # aiida-hyperqueue not installed
            raise ModuleNotFoundError(
                "Must install aiida-hyperqueue using AiiDA with hyperqueue"
            )
    else:
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

        builder["parallelization"] = orm.Dict(base_parallelization)
        builder["metadata"] = metadata

    elif process_class == PwBaseWorkChain:
        set_parallelization(
            builder["pw"],
            parallelization=parallelization,
            process_class=BasePwCpInputGenerator,
        )

    # Includes PwCalculation, OpenGridCalculation, ProjwfcCalculation
    elif issubclass(process_class, NamelistsCalculation):
        builder["metadata"] = metadata
        builder["settings"] = settings

    elif issubclass(process_class, CalcJob):
        builder["metadata"] = metadata

    elif process_class == OpenGridBaseWorkChain:
        set_parallelization(
            builder["open_grid"],
            parallelization=parallelization,
            process_class=NamelistsCalculation,
        )
        # For now open_grid.x has memory issue, I run it with less cores
        # open_grid_metadata = copy.deepcopy(metadata)
        # open_grid_metadata['options']['resources']['num_mpiprocs_per_machine'] = (num_mpiprocs_per_machine // npool)
        # builder.open_grid['open_grid']['metadata'] = open_grid_metadata

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

    elif process_class == ProjwfcBandsWorkChain:
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

        if "projwfc" in pruned_builder:
            set_parallelization(
                builder["projwfc"],
                parallelization=parallelization,
                process_class=ProjwfcBaseWorkChain,
            )

    elif process_class in [Wannier90BandsWorkChain, Wannier90OptimizeWorkChain]:
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

        if "open_grid" in pruned_builder:
            set_parallelization(
                builder["open_grid"],
                parallelization=parallelization,
                process_class=OpenGridBaseWorkChain,
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


def get_metadata_hq(
    *,
    num_cpus: int = None,
    max_wallclock_seconds: int = 24 * 3600,
    memory_mb: int = 2048,
    queue_name: str = None,
    account: str = None,
    **_,
) -> dict:
    """Return metadata with the given MPI specification (HyperQueue).

    Usage
    ```
    # Create a dict
    parallelization = {
        'num_cpus': 16,
        'npool': 4,
        'memory_mb': 2048,
        'queue_name': 'debug',
        'account': 'mr0',
    }
    # The dict can be used by
    # set_parallelization(builder, parallelization)
    # Or
    metadata = get_metadata(**parallelization)
    ```

    :param num_cpus: defaults to None, meaning it is not set
    and will the default number of CPUs in the `computer` configuration.
    :type num_cpus: int, optional
    :param max_wallclock_seconds: defaults to 24*3600
    :type max_wallclock_seconds: int, optional
    :param memory_mb: defaults to 2048 MB
    :type memory: int, optional
    :param queue_name: slurm queue name
    :type queue_name: str, optional
    :param account: slurm account
    :type account: str, optional
    :return: metadata dict
    :rtype: dict
    """
    metadata = {
        "options": {
            "resources": {"num_cpus": num_cpus, "memory_mb": memory_mb},
            "max_wallclock_seconds": max_wallclock_seconds,
            "withmpi": True,
        }
    }
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
    settings = orm.Dict({"cmdline": ["-nk", f"{npool}"]})

    return settings


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
        builder["parameters"] = orm.Dict(params)

    elif process_class == Wannier90BaseWorkChain:
        set_kpoints(builder["wannier90"], kpoints, process_class=Wannier90Calculation)

    elif issubclass(process_class, Wannier90WorkChain):
        set_kpoints(builder["wannier90"], kpoints, process_class=Wannier90BaseWorkChain)

        kpoints_explicit = builder["wannier90"]["wannier90"]["kpoints"]
        if "scf" in builder:
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
        builder["parameters"] = orm.Dict(params)

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
        builder["parameters"] = orm.Dict(params)

    elif process_class == PwBaseWorkChain:
        set_num_bands(builder["pw"], num_bands, process_class=PwCalculation)

    elif process_class == PwBandsWorkChain:
        set_num_bands(builder["bands"], num_bands, process_class=PwBaseWorkChain)
