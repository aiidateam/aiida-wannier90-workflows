#!/usr/bin/env runaiida

from aiida import orm
from aiida.engine import submit
from aiida_wannier90_workflows.utils.builder import get_builder_for_stash_restart

if __name__ == '__main__':
    codes = {
        'pw': 'qe-git-pw@localhost',
        'projwfc': 'qe-git-projwfc@localhost',
        'pw2wannier90': 'qe-git-pw2wannier90@localhost',
        'wannier90': 'wannier90-git@localhost',
        'opengrid': 'qe-git-opengrid@localhost'
    }
    pw_calc = orm.load_node(8610)
    structure = pw_calc.inputs.structure
    # parent_folder = pw_calc.outputs.remote_folder
    parent_folder = pw_calc.outputs.remote_stash

    builder = get_builder_for_stash_restart(
        codes, structure, parent_folder
    )
    from pprint import pprint
    pprint(builder._data)

    wc = submit(builder)

    print(f'submitted workflow<{wc.pk}> for {structure.get_formula()}')
