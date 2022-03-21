# -*- coding: utf-8 -*-
"""Functions for structures."""
import typing as ty
import pathlib

from aiida import orm


def read_structure(filename: ty.Union[str, pathlib.Path], store: bool = False) -> orm.StructureData:
    """Read a xsf/xyz/cif/.. file and return aiida ``StructureData``."""
    from ase.io import read as aseread

    struct = orm.StructureData(ase=aseread(filename))

    if store:
        struct.store()
        print(f'Read and stored structure {struct.get_formula()}<{struct.pk}>')

    return struct
