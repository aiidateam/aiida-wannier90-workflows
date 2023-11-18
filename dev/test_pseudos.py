"""Test that the data provided for the pseudopotentials is complete and correct."""

from aiida import load_profile, orm

from aiida_wannier90_workflows.utils.pseudo import get_pseudo_orbitals

load_profile()


for family in (
    "PseudoDojo/0.4/LDA/SR/standard/upf",
    "PseudoDojo/0.4/LDA/SR/stringent/upf",
    "PseudoDojo/0.4/PBE/SR/standard/upf",
    "PseudoDojo/0.4/PBE/SR/stringent/upf",
    "PseudoDojo/0.5/PBE/SR/standard/upf",
    "PseudoDojo/0.5/PBE/SR/stringent/upf",
):
    print(f"Testing family {family}")
    for el, pseudo in orm.load_group(family).pseudos.items():
        try:
            get_pseudo_orbitals({el: pseudo})
        except ValueError as exc:
            print(exc)
