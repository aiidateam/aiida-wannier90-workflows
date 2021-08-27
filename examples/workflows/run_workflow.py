#!/usr/bin/env runaiida

from aiida import orm
from aiida.engine import submit
from aiida_wannier90_workflows.workflows import Wannier90BandsWorkChain
from aiida_wannier90_workflows.workflows.bands import get_builder_for_pwbands

if __name__ == '__main__':
    codes = {
        'pw': 'qe-git-pw@localhost',
        'projwfc': 'qe-git-projwfc@localhost',
        'pw2wannier90': 'qe-git-pw2wannier90@localhost',
        'wannier90': 'wannier90-git@localhost',
        'opengrid': 'qe-git-opengrid@localhost'
    }
    structure = orm.load_node(91)

    builder = Wannier90BandsWorkChain.get_builder_from_protocol(
        codes, structure, protocol='fast',
        run_opengrid=True,
    )

    wc = submit(builder)

    print(f'submitted workflow<{wc.pk}> for {structure.get_formula()}')

    # pw_builder = get_builder_for_pwbands(wc)
    # submit(pw_builder)
