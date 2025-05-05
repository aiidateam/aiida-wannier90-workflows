#!/usr/bin/env python
"""Run a ``Wannier90BandsWorkChain`` for Wannier90 band structure with external projector.

Usage: ./example_07.py
"""
import json

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
    codes = identify_codes(codes)
    check_codes(codes)

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


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.CODES()
@cmdline.params.options.GROUP(help="The group to add the submitted workchain.")
@click.argument("filename", type=click.Path(exists=True))
@click.argument("pseudo_family", type=str)
@click.argument("external_projectors_path", type=click.Path(exists=True))
@RUN()
def cli(
    filename, codes, pseudo_family, external_projectors_path, group, run
):  # pylint: disable=too-many-positional-arguments
    """Run a ``Wannier90BandsWorkChain`` with external projectors.

    FILENAME: a crystal structure file, e.g., ``input_files/GaAs.xsf``.
    PSEUDO_FAMILY: label of pseudo family, e.g., ``SSSP/1.1/PBE/efficiency``.
    EXTERNAL_PROJECTORS_PATH: the path to the directory which includes the external projectors.
    e.g., ``input_files/external_projectors/``
    """
    struct = read_structure(filename, store=True)
    try:
        with open(
            external_projectors_path + "/projectors.json", encoding="utf-8"
        ) as fp:
            external_projectors = json.load(fp)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Can not find projectors.json in ``{external_projectors_path}``. "
            "Try to regenerate the external projector files referring to the script "
            "``aiida-wannier90-workflows/dev/projectors/example_extend_aiida_pseudo.py``"
        ) from exc
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
