#!/usr/bin/env runaiida
"""Launch a wannier90 workflow."""
from aiida import orm
from aiida.engine import submit

from aiida_wannier90_workflows.utils.workflows.builder.generator import (
    get_pwbands_builder,
)
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain


def read_structure(filename):
    """Read a xsf/xyz file and return aiida StructureData."""
    from ase.io import read as aseread

    struct = orm.StructureData(ase=aseread(filename))
    struct.store()
    print(f"Read and stored structure {struct.get_formula()}<{struct.pk}>")
    return struct


if __name__ == "__main__":
    codes = {
        "pw": "qe-git-pw@localhost",
        "projwfc": "qe-git-projwfc@localhost",
        "pw2wannier90": "qe-git-pw2wannier90@localhost",
        "wannier90": "wannier90-git-wannier90@localhost",
        #'opengrid': 'qe-git-opengrid@localhost'
    }

    # load a structure or read from file
    # structure = orm.load_node(PK_OF_A_STRUCTURE)
    structure = read_structure("GaAs.xsf")

    builder = Wannier90BandsWorkChain.get_builder_from_protocol(
        codes, structure, protocol="fast"
    )

    wc = submit(builder)

    print(f"Submitted workflow<{wc.pk}> for {structure.get_formula()}")

    # Once the workflow has finished, launch a QE bands workflow for comparison
    pw_builder = get_pwbands_builder(wc)
    pw_wc = submit(pw_builder)
    print(f"Submitted pw bands workflow<{pw_wc.pk}>")
