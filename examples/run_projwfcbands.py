#!/usr/bin/env runaiida
"""Launch a projected bands workflow."""
from aiida import orm
from aiida.engine import submit

from aiida_wannier90_workflows.utils.workflows.builder import print_builder
from aiida_wannier90_workflows.workflows import ProjwfcBandsWorkChain

# pylint: disable=invalid-name,undefined-variable

pw_code = "qe-git-pw@localhost"
projwfc_code = "qe-git-projwfc@localhost"

# pw_code = 'qe-git-pw@prnmarvelcompute5'
# projwfc_code = 'qe-git-projwfc@prnmarvelcompute5'

# load a structure or read from file
structure = orm.load_node(PK_OF_A_STRUCTUREDATA)

builder = ProjwfcBandsWorkChain.get_builder_from_protocol(
    pw_code=pw_code, projwfc_code=projwfc_code, structure=structure
)

print_builder(builder)

wkchain = submit(builder)

print(f"Submitted workflow<{wkchain.pk}> for {structure.get_formula()}")
