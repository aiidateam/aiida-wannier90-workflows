#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-

import os
import sys

from aiida.common.exceptions import NotExistent
from aiida.orm.data.base import Str
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.upf import UpfData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.work.run import run,submit
from aiida_wannier90_theosworkflows.workflows.W90 import SimpleWannier90WorkChain
from aiida.orm.data.base import List
import ase, ase.io
from aiida.common.example_helpers import test_and_get_code

#queue = None
queue_name = 'debug'
w90_queue_name = None
num_cpus = 28

try:
    dontsend = sys.argv[1]
    if dontsend == "--dont-send":
        submit_test = True
    elif dontsend == "--send":
        submit_test = False
    else:
        raise IndexError
except IndexError:
    print >> sys.stderr, ("The first parameter can only be either "
                          "--send or --dont-send")
    sys.exit(1)

required_elements = ['Ga', 'As']
try:
    pw_codename = sys.argv[2]
    pw_code = test_and_get_code(pw_codename, expected_code_type='quantumespresso.pw')

    pw2wannier90_codename = sys.argv[3]
    pw2wannier90_code = test_and_get_code(pw2wannier90_codename, expected_code_type='quantumespresso.pw2wannier90')

    wannier90_codename = sys.argv[4]
    wannier90_code = test_and_get_code(wannier90_codename, expected_code_type='wannier90.wannier90')

    pseudo_family_name = sys.argv[5]
    valid_pseudo_group_names = [_.name for _ in UpfData.get_upf_groups(filter_elements=required_elements)]
    print valid_pseudo_group_names


    try:
        UpfData.get_upf_group(pseudo_family_name)
    except NotExistent:
        print >> sys.stderr, "pseudo_family_name='{}'".format(pseudo_family_name)
        print >> sys.stderr, "   but no group with such a name found in the DB."
        print >> sys.stderr, "   Valid UPF groups are:"
        print >> sys.stderr, "   " + ",".join(i.name for i in valid_pseudo_groups)
        sys.exit(1)
    if pseudo_family_name not in valid_pseudo_group_names:
        print >> sys.stderr, "Error: pseudo_family_name='{}'".format(pseudo_family_name)
        print >> sys.stderr, "   does not contain the pseudos for at least one of the required elements:"
        print >> sys.stderr, "   " + ", ".join(required_elements)
        sys.exit(1)
        
except IndexError:
    print >> sys.stderr, ("Must provide as further parameters:\n"
        "- a pw codename\n"
        "- a pw2wannier90 codename\n"
        "- a wannier90 codename\n"
        "- a pseudo family name")
    sys.exit(1)

scf_parameters = ParameterData(dict={
        'CONTROL': {
            'restart_mode': 'from_scratch',
        },
        'SYSTEM': {
            'ecutwfc': 30.,
            'ecutrho': 240.,
        },
    })
nscf_parameters = ParameterData(dict={
        'SYSTEM': {
            'nbnd': 10,
        },
    })
scf_settings =  ParameterData(dict={})
max_wallclock_seconds = 60*10
scf_options  = ParameterData(dict={
                        'resources': {
                                    'num_machines': 1,
                                    'tot_num_mpiprocs': num_cpus,
                                     },
                        'max_wallclock_seconds': max_wallclock_seconds, 'queue_name':queue_name,
                             })
pw2wannier90_options  = ParameterData(dict={
                        'resources': {
                                    'num_machines': 1,
                                    'tot_num_mpiprocs': num_cpus,
                                     },
                        'max_wallclock_seconds': 60*10, 'queue_name':queue_name,
                             })
wannier90_parameters = ParameterData(dict={'bands_plot':True,
                                'write_hr':True,
              #                  'use_ws_distance':True,
                                'num_iter': 12,
                                'guiding_centres': True,
                                'num_wann': 4,
                                'num_bands':10
                                #'exclude_bands': exclude_bands,
                                # 'wannier_plot':True,
                                # 'wannier_plot_list':[1]
                                })
wannier_pp_options = ParameterData(dict={
    'resources': {
        'num_machines': 1,
        'tot_num_mpiprocs': 1,
    },
    'max_wallclock_seconds': 60 * 10, 'queue_name':w90_queue_name,
})
wannier_options = ParameterData(dict={
    'resources': {
        'num_machines': 1,
        'tot_num_mpiprocs': 1,
    },
    'max_wallclock_seconds': 60 * 10, 'queue_name':w90_queue_name,
})
structure = StructureData(ase=ase.io.read(
    os.path.join(
        os.path.split(os.path.abspath(__file__))[0],
        'GaAs.xsf'
        ),
    )
)
scf_kpoints = KpointsData()
scf_kpoints.set_kpoints_mesh([4,4,4])
nscf_kpoints = KpointsData()
nscf_kpoints.set_kpoints_mesh([4,4,4])
projections = List()
projections.extend(['As:s','As:p'])

if submit_test:
    print "Asked not to send: stopping"
    sys.exit(0)

wc = submit(SimpleWannier90WorkChain,
        pw_code=pw_code,
        pw2wannier90_code=pw2wannier90_code,
        wannier90_code=wannier90_code,
        structure=structure,
        pseudo_family=Str(pseudo_family_name),
        scf={'parameters':scf_parameters,'kpoints':scf_kpoints,'settings':scf_settings,'options':scf_options},
        nscf={'parameters':nscf_parameters,'kpoints':nscf_kpoints},
        mlwf={'projections':projections, 'parameters':wannier90_parameters,'pp_options':wannier_pp_options,
              'options':wannier_options}, # 'kpoints_path':None},
        matrices={'_options':pw2wannier90_options},
#        restart_options = {'scf_workchain':load_node(712),'nscf_workchain':load_node(3038),'mlwf_pp':load_node(3048),
#                           'pw2wannier90':load_node(3085)},
    )

print 'launched WorkChain pk {}'.format(wc.pid)

