#!/usr/bin/env runaiida
from ase.io import read as aseread

from aiida import orm
from aiida.engine import submit

from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain

# Code labels for `pw.x`, `pw2wannier90.x`, `projwfc.x`, and `wannier90.x`.
# Change these according to your aiida setup.
codes = {
    "pw": "qe-7.0-pw@localhost",
    "projwfc": "qe-7.0-projwfc@localhost",
    "pw2wannier90": "qe-7.0-pw2wannier90@localhost",
    "wannier90": "wannier90-3.1-wannier90@localhost",
}

# Filename of a structure.
filename = "GaAs.xsf"

# Read a structure file and store as an `orm.StructureData`.
structure = orm.StructureData(ase=aseread(filename))
structure.store()
print(f"Read and stored structure {structure.get_formula()}<{structure.pk}>")

# Prepare the builder to launch the workchain.
# We use fast protocol to converge faster.
builder = Wannier90BandsWorkChain.get_builder_from_protocol(
    codes,
    structure,
    protocol="fast",
)

# Submit the workchain.
workchain = submit(builder)
print(f"Submitted {workchain.process_label}<{workchain.pk}>")

print(
    "Run any of these commands to check the progress:"
    "verdi process report {workchain.pk}"
    "verdi process show {workchain.pk}"
    "verdi process list"
)
