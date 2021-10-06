#!/usr/bin/env runaiida
from aiida import orm
from aiida.engine import submit
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain
from aiida_wannier90_workflows.workflows.bands import get_builder_for_pwbands


def read_structure(filename):
    from ase.io import read as aseread
    structure = orm.StructureData(ase=aseread(filename))
    structure.store()
    print(
        f'Read and stored structure {structure.get_formula()}<{structure.pk}>'
    )
    return structure


if __name__ == '__main__':
    codes = {
        'pw': 'qe-git-pw@localhost',
        'projwfc': 'qe-git-projwfc@localhost',
        'pw2wannier90': 'qe-git-pw2wannier90@localhost',
        'wannier90': 'wannier90-git-wannier90@localhost',
        #'opengrid': 'qe-git-opengrid@localhost'
    }

    # load a structure or read from file
    # structure = orm.load_node(PK_OF_A_STRUCTURE)
    structure = read_structure('GaAs.xsf')

    builder = Wannier90BandsWorkChain.get_builder_from_protocol(
        codes, structure, protocol='fast'
    )

    wc = submit(builder)

    print(f'Submitted workflow<{wc.pk}> for {structure.get_formula()}')

    # Once the workflow has finished, launch a QE bands workflow for comparison
    # pw_builder = get_builder_for_pwbands(wc)
    # pw_wc = submit(pw_builder)
    # print(f'Submitted pw bands workflow<{pw_wc.pk}>')
