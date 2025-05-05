"""Functions for structures."""

import pathlib
import typing as ty

from aiida import orm


def read_structure(
    filename: ty.Union[str, pathlib.Path], store: bool = False
) -> orm.StructureData:
    """Read a xsf/xyz/cif/.. file and return aiida ``StructureData``."""
    from ase.io import read as aseread

    struct = orm.StructureData(ase=aseread(filename))

    if store:
        struct.store()
        print(f"Read and stored structure {struct.get_formula()}<{struct.pk}>")

    return struct


def read_magentic_structure(
    filename: ty.Union[str, pathlib.Path], store: bool = False
) -> orm.StructureData:
    """Read a mcif file and return ``MagneticStructureData``."""
    from pymatgen.core import Structure

    from aiida_wannier90_workflows.data.structure import MagneticStructureData

    struct = MagneticStructureData(pymatgen=Structure.from_file(filename))

    if store:
        struct.store()
        print(f"Read and stored structure {struct.get_formula()}<{struct.pk}>")

    return struct
