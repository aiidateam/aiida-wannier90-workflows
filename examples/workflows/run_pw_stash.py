#!/usr/bin/env runaiida
# coding: utf-8
from aiida import orm
from aiida.engine import submit
from aiida_wannier90_workflows.utils.builder import get_pwbuilder_for_stash

if __name__ == '__main__':
    pw_calc = orm.load_node(3794)
    builder = get_pwbuilder_for_stash(pw_calc)

    wc = submit(builder)

    print(f'submitted {wc.process_label}<{wc.pk}>')
    # PwCalculation 8610
    # RemoteStashFolderData 8612
