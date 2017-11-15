#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-

import argparse
from aiida.common.exceptions import NotExistent
from aiida.orm.data.base import Str
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.work.run import run,submit
from aiida_wannier90_theosworkflows.workflows.W90 import SimpleWannier90WorkChain
from aiida.orm.data.base import List
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
            'nbnd': 12,
            'ecutwfc': 30.,
            'ecutrho': 240.,
        },
    })
scf_settings =  ParameterData(dict={})
max_wallclock_seconds = 60*10
scf_options  = ParameterData(dict={
                        'resources': {
                                    'num_machines': 1,
                                    'tot_num_mpiprocs': 28,
                                     },
                        'max_wallclock_seconds': max_wallclock_seconds,'queue_name':'debug',
                             })
nscf_options  = ParameterData(dict={
                        'resources': {
                                    'num_machines': 1,
                                    'tot_num_mpiprocs': 28,
                                     },
                        'max_wallclock_seconds': 60*5,'queue_name':'debug',
                             })
pw2wannier90_options  = ParameterData(dict={
                        'resources': {
                                    'num_machines': 1,
                                    'tot_num_mpiprocs': 28,
                                     },
                        'max_wallclock_seconds': 60*10,'queue_name':'debug',
                             })
wannier90_parameters = ParameterData(dict={'bands_plot':True,
                                'write_hr':True,
              #                  'use_ws_distance':True,
                                'num_iter': 0,
                                'dis_num_iter':0,
                             #   'guiding_centres': True,
                                'num_wann': 8,
                                'num_bands':12,
                                'dis_win_max':10.0,
                                'dis_froz_max':0.5,
                                'scdm_proj': True,
                                'scdm_entanglement':'erfc',
                                'scdm_mu':4,
                                'scdm_sigma':4,
                                #'dis_num_iter':120,
                                #'dis_mix_ratio':1.d0,
                                #'exclude_bands': exclude_bands,
                                # 'wannier_plot':True,
                                # 'wannier_plot_list':[1]
                                })
wannier_pp_options = ParameterData(dict={
    'resources': {
        'num_machines': 1,
        'tot_num_mpiprocs': 1,
    },
    'max_wallclock_seconds': 60 * 10,'queue_name':'debug',
})
wannier_options = ParameterData(dict={
    'resources': {
        'num_machines': 1,
        'tot_num_mpiprocs': 1,
    },
    'max_wallclock_seconds': 60 * 10,'queue_name':'debug',
})
wc_control = {'group_name':'w90wc_trial_1','retrieve_hamiltonian':True,'zero_is_fermi':True}
structure = load_node(8490)
scf_kpoints = KpointsData()
scf_kpoints.set_kpoints_mesh([8,8,8])
nscf_kpoints = KpointsData()
nscf_kpoints.set_kpoints_mesh([6,6,6])
#projections = List()
#projections.extend(['Si:sp3'])
wc = submit(SimpleWannier90WorkChain,
        pw_code=Code.get_from_string('pw_6.1@fidis'),
        pw2wannier90_code=Code.get_from_string('pw2wannier90_6.1_scdm@fidis'),
        wannier90_code=Code.get_from_string('wannier90_scdm@fidis'),
        #orbital_projections=projections,
        structure=structure,
        pseudo_family=Str('SSSP_efficiency_v0.95'),
        #wannier90_parameters=wannier90_parameters,
        scf={'parameters':scf_parameters,'kpoints':scf_kpoints,'settings':scf_settings,'options':scf_options},
        nscf={'parameters':nscf_parameters,'kpoints':nscf_kpoints, 'options':nscf_options},
        mlwf={ 'parameters':wannier90_parameters,'pp_options':wannier_pp_options,
              'options':wannier_options}, #'projections':projections, 'kpoint_path':None},
        matrices={'_options':pw2wannier90_options},
        restart_options = {'scf_workchain':load_node(9502),'nscf_workchain':load_node(9515),'mlwf_pp':load_node(9526),
                           'pw2wannier90':load_node(9532)},
        workchain_control = ParameterData(dict=wc_control),
    )

print 'launched WorkChain pk {}'.format(wc.pid)

