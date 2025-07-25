#!/usr/bin/env python
"""Run a ``Wannier90BandsWorkChain`` for Wannier90 band structure with external projector.

Usage: ./example_07.py
"""
import json
import os

import click

from aiida import cmdline, orm

from aiida_wannier90_workflows.cli.params import RUN
from aiida_wannier90_workflows.common.types import WannierProjectionType
from aiida_wannier90_workflows.utils.code import check_codes, identify_codes
from aiida_wannier90_workflows.utils.structure import read_structure
from aiida_wannier90_workflows.utils.workflows.builder.serializer import print_builder
from aiida_wannier90_workflows.utils.workflows.builder.setter import set_parallelization
from aiida_wannier90_workflows.utils.workflows.builder.submit import (
    submit_and_add_group,
)
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain


def submit(  # pylint: disable=too-many-positional-arguments
    codes: dict,
    structure: orm.StructureData,
    pseudo_family: str,
    external_projectors: dict,
    external_projectors_path: str,
    group: orm.Group = None,
    run: bool = False,
):
    """Submit a ``Wannier90BandsWorkChain`` to calculate Wannier bands."""

    builder = Wannier90BandsWorkChain.get_builder_from_protocol(
        codes,
        structure,
        pseudo_family=pseudo_family,
        projection_type=WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL,
        external_projectors=external_projectors,
        external_projectors_path=external_projectors_path,
        protocol="fast",
    )

    # You can change parallelization here
    parallelization = {
        "num_mpiprocs_per_machine": 1,
        "npool": 1,
    }
    set_parallelization(builder, parallelization, process_class=Wannier90BandsWorkChain)

    print_builder(builder)

    if run:
        submit_and_add_group(builder, group)


def projectors_exists_check(computer, external_projectors_path):
    """Check whether external_projectors_path is valid on <computer>.

    An additional check that external_projectors_path exists on compute node <computer>.
    If exists, get projectors information and return.
    When running HT calculations, we recommend locate the projectors on the <computer>,
    but keep an addition ``projectors.json`` locally to facilitate workflow extracting information.
    """
    local_compute = computer.transport_type == "core.local"
    if local_compute:
        external_projectors_path = os.path.abspath(external_projectors_path)
    remote_path = orm.RemoteData(
        computer=computer, remote_path=external_projectors_path
    )
    # Check if external_projectors_path exist on computer.
    try:
        list_projectors = remote_path.listdir()
    except OSError as exc:
        raise OSError(
            f"{remote_path.get_remote_path()} is not a valid directory "
            f"on computer<{computer.label}>"
        ) from exc

    if not "projectors.json" in list_projectors:
        if not local_compute:
            transport_errormessage = (
                f" and transport the projectors to computer<{computer.label}>"
            )
        else:
            transport_errormessage = ""
        raise FileNotFoundError(
            f"Can not find projectors.json in ``{external_projectors_path}``. "
            "Try to regenerate the external projector files referring to the script "
            "``aiida-wannier90-workflows/dev/projectors/example_extend_aiida_pseudo.py``"
            + transport_errormessage
        )
    # Parse ``projectors.json``, if the file exists on remote computer, transport it to local as tmp file.
    if not local_compute:
        tmp_json_path = os.path.abspath("./tmp_projectors.json")
        remote_path.getfile("./projectors.json", tmp_json_path)
        with open(tmp_json_path, encoding="utf-8") as fp:
            external_projectors = json.load(fp)
        os.remove(tmp_json_path)
    else:
        with open(
            external_projectors_path + "/projectors.json", encoding="utf-8"
        ) as fp:
            external_projectors = json.load(fp)
    return external_projectors, external_projectors_path


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.CODES()
@cmdline.params.options.GROUP(help="The group to add the submitted workchain.")
@click.argument("filename", type=click.Path(exists=True))
@click.argument("pseudo_family", type=str)
@click.argument("external_projectors_path")
@RUN()
def cli(
    filename, codes, pseudo_family, external_projectors_path, group, run
):  # pylint: disable=too-many-positional-arguments
    """Run a ``Wannier90BandsWorkChain`` with external projectors.

    FILENAME: a crystal structure file, e.g., ``input_files/GaAs.xsf``.
    PSEUDO_FAMILY: label of pseudo family, e.g., ``SSSP/1.3/PBEsol/efficiency``.
    EXTERNAL_PROJECTORS_PATH: the path to the directory on computing node which includes the external projectors.
    e.g., ``input_files/external_projectors/``
    """
    struct = read_structure(filename, store=True)

    codes = identify_codes(codes)
    check_codes(codes)
    computer = codes["pw2wannier90"].computer

    external_projectors, external_projectors_path = projectors_exists_check(
        computer, external_projectors_path
    )

    submit(
        codes,
        struct,
        pseudo_family,
        external_projectors,
        external_projectors_path,
        group,
        run,
    )


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
