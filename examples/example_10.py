#!/usr/bin/env python
"""Run a ``Wannier90SplitWorkChain`` for Wannier90 band structure.

Usage: ./example_10.py
"""
# pylint: disable=unused-import
from pathlib import Path

import click

from aiida import cmdline, orm

from aiida_quantumespresso.workflows.pw.bands import PwBandsWorkChain

from aiida_wannier90_workflows.cli.params import RUN
from aiida_wannier90_workflows.common.types import WannierProjectionType
from aiida_wannier90_workflows.utils.code import check_codes, identify_codes
from aiida_wannier90_workflows.utils.structure import read_structure
from aiida_wannier90_workflows.utils.workflows.bands import (
    get_structure_and_bands_kpoints,
)
from aiida_wannier90_workflows.utils.workflows.builder import (
    print_builder,
    set_parallelization,
    submit_and_add_group,
)
from aiida_wannier90_workflows.workflows import (
    Wannier90BandsWorkChain,
    Wannier90BaseWorkChain,
)
from aiida_wannier90_workflows.workflows.split import Wannier90SplitWorkChain


def submit(
    codes: dict,
    structure: orm.StructureData,
    group: orm.Group = None,
    run: bool = False,
):
    """Submit a ``Wannier90BandsWorkChain`` to calculate Wannier bands."""
    codes = identify_codes(codes)
    check_codes(codes, required_codes=["pw", "pw2wannier90", "wannier90", "split"])

    qb = orm.QueryBuilder()
    qb.append(orm.Group, tag="group", filters={"label": "tests/pwbands/valcond"})
    qb.append(PwBandsWorkChain, tag="wkchain", with_group="group", project="*")
    qb.append(
        orm.StructureData,
        tag="structure",
        with_outgoing="wkchain",
        filters={"id": structure.pk},
    )
    pwbands_wkchain = qb.one()[0]
    print(f"Found PwBandsWorkChain: {pwbands_wkchain}")

    primitive_structure, bands_kpoints = get_structure_and_bands_kpoints(
        pwbands_wkchain
    )

    builder = Wannier90SplitWorkChain.get_builder_from_protocol(
        codes,
        primitive_structure,
        bands_kpoints=bands_kpoints,
        projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
        plot_wannier_functions=True,
        reference_bands=pwbands_wkchain.outputs.band_structure,
        # protocol="fast",
    )

    builder["valcond"]["optimize_disproj"] = False
    builder["valcond"]["separate_plotting"] = False

    # You can change parallelization here
    parallelization = {
        "num_mpiprocs_per_machine": 8,
        "npool": 4,
    }
    set_parallelization(
        builder["valcond"], parallelization, process_class=Wannier90BandsWorkChain
    )
    set_parallelization(
        builder["val"], parallelization, process_class=Wannier90BaseWorkChain
    )
    set_parallelization(
        builder["cond"], parallelization, process_class=Wannier90BaseWorkChain
    )

    print_builder(builder)

    if run:
        submit_and_add_group(builder, group)


@click.command()
@cmdline.utils.decorators.with_dbenv()
@cmdline.params.options.CODES()
@cmdline.params.options.GROUP(help="The group to add the submitted workchain.")
# @click.argument("filename", type=click.Path(exists=True))
@cmdline.params.arguments.NODE("filename")
@RUN()
def cli(filename, codes, group, run):  # pylint: disable=unused-argument
    """Run a ``Wannier90BandsWorkChain`` to calculate Wannier90 band structure.

    FILENAME: a crystal structure file, e.g., ``input_files/GaAs.xsf``.
    """
    struct_group = orm.load_group("tests/structures/valcond")

    # struct_path = Path(__file__).parent / 'materials_from_Nicola_Colonna'
    # for fname in struct_path.iterdir():
    #     print(fname)

    #     filename = fname / 'PWSCF' / 'pw_scf.in'

    #     struct = read_structure(filename, store=True)

    #     struct_group.add_nodes([struct])

    # struct = filename
    # submit(codes, struct, group, run)

    for struct in struct_group.nodes[2:]:
        submit(codes, struct, group, run)


if __name__ == "__main__":
    cli()  # pylint: disable=no-value-for-parameter
