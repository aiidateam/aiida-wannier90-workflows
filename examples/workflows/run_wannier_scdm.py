#!/usr/bin/env runaiida
# -*- coding: utf-8 -*-

import argparse
from aiida.common.exceptions import NotExistent
from aiida.orm import Group
from aiida.orm.data.base import Str
from aiida.orm.data.upf import UpfData
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.utils import WorkflowFactory
from aiida.work.run import run,submit
from aiida_quantumespresso.workflows.pw.custom_band_structure_workchain import CustomPwBandStructureWorkChain
from aiida.orm.calculation.work import WorkCalculation
from aiida_wannier90_theosworkflows.workflows.W90 import SimpleWannier90WorkChain
#from custom_band_structure_workchain import CustomPwBandStructureWorkChain

#only_valence = True
only_valence = False
do_disen = False
do_mlwf = True
exclude_bands = None
plot_wfs = True
#exclude_bands = range(1,12+1) # skip bands 1,2,...,12

# Add a suffix to the output group name, to distinguish from previous runs, if needed
#g_name_suffix = ''
g_name_suffix = '_musigmaauto1-sigmashift0'

use_antimo_codes = False

## The following is irrelevant if use_antimo_codes is True
#gp_computer = 'localhost-openmpi'
#gp_computer = 'deneb'
gp_computer = 'fidis'

system_set = 'default'
#system_set = 'gp-onesystem'

only_these_structures = None # Then do all of them
#only_these_structures = ['S2Ta'] # Only this
#only_these_structures = ['He', 'Ne', 'Kr2', 'O2Pb2', 'Be2I4', 'BaTe', 'Se4Tl4', 'Ar2', 'F2Xe', 'CsH', 'CaO', 'Hg3O3', 'O2Rb2', 'C2Cd2O6'] # All insulators but InP, 14 systems

max_sec_scf = 4*3600
max_sec_nscf = 12*3600
max_sec_pw2wannier90 = 12*3600
max_sec_projwfc = 60 * 30
max_sec_wannier_pp = 60 * 10
max_sec_wannier = 3600 * 4
num_pools = 4

def update_g_name(in_g_name):
    # This is the output group name
    global only_valence, do_disen, do_mlwf, exclude_bands

    g_name = in_g_name
    if only_valence:
       g_name += "_onlyvalence"
    else:
       g_name += "_withconduction"
    if do_disen:
      g_name += '_dis'
    if do_mlwf:
        g_name += '_mlwf'
    if exclude_bands is not None:
        g_name += '_excluded{}'.format(len(exclude_bands))
    return g_name

if system_set == 'default':
    # output group (base)name
    g_name = 'w90_scdm_v4_dense'

    # Used for filtering - requires to have already run the first part with DFT, seekpath etc.
    wc_group_name = "workchain_out_1"
    group_name = "structures_final_test_bench_small_v2" # It's only the basename, see below

    # Add suffix to output group name depending on flags
    g_name = update_g_name(g_name)

    ## Update the input set depending on whether to run for the metals input 
    ## set or the metallic input set
    #
    ## Note: also for _insulating one can do only_valence = False    
    #suffix = '_metallic'
    suffix = '_insulating'
    group_name = group_name + suffix
    g_name += suffix

elif system_set == 'gp-onesystem':
    # output group (base)name
    g_name = 'gptest_out_w90_1'

    # Used for filtering - requires to have already run the first part with DFT, seekpath etc.
    wc_group_name = "gptest_out_1"
    group_name = "structures_gptest_1"

    # Add suffix to output group name depending on flags
    g_name = update_g_name(g_name)
else:
    raise NotImplementedError("unknown system_set")

g_name += g_name_suffix

## Constants
tot_num_mpiprocs_fidis = 28
tot_num_mpiprocs_deneb = 24

if use_antimo_codes:
    pw_code=Code.get_from_string('pw_6.1@fidis')
    pw2wannier90_code=Code.get_from_string('pw2wannier90_6.1_scdm@fidis')
    wannier90_code=Code.get_from_string('wannier90_scdm@fidis')
    projwfc_code=Code.get_from_string('projwfc_6.1_scdm@fidis')
    tot_num_mpiprocs = tot_num_mpiprocs_fidis
else:
    if gp_computer == 'fidis':
        pw_code=Code.get_from_string('pw_6.1_scdm_pizzi@fidis')
        ## New version of the code (2018.01)
        pw2wannier90_code=Code.get_from_string('pw2wannier90_6.1_scdm-2018.01_pizzi@fidis')
        wannier90_code=Code.get_from_string('wannier90_scdm_pizzi@fidis')
        projwfc_code=Code.get_from_string('projwfc_6.1_scdm_pizzi@fidis')
        tot_num_mpiprocs = tot_num_mpiprocs_fidis
    elif gp_computer == 'deneb':
        raise NotImplementedError("Bugged version on deneb, not recompiled yet")
        pw_code=Code.get_from_string('pw_6.1_scdm-2018.01_pizzi@deneb')
        pw2wannier90_code=Code.get_from_string('pw2wannier90_6.1_scdm-2018.01_pizzi@deneb')
        wannier90_code=Code.get_from_string('wannier90_scdm_pizzi@deneb')
        projwfc_code=Code.get_from_string('projwfc_6.1_scdm-2018.01_pizzi@deneb')
        tot_num_mpiprocs = tot_num_mpiprocs_deneb
    elif gp_computer == 'localhost-openmpi':
        raise NotImplementedError("Bugged version on localhost-openmpi, not recompiled yet")
        pw_code=Code.get_from_string('pw_6.1_scdm-2018.01_pizzi@localhost-openmpi')
        pw2wannier90_code=Code.get_from_string('pw2wannier90_6.1_scdm-2018.01_pizzi@localhost-openmpi')
        wannier90_code=Code.get_from_string('wannier90_scdm_pizzi@localhost-openmpi')
        projwfc_code=Code.get_from_string('projwfc_6.1_scdm-2018.01_pizzi@localhost-openmpi')
        tot_num_mpiprocs = 8
    else:
        raise NotImplementedError("computer {} not implemented in the launcher".format(gp_computer))
        
###################################################################################

qb = QueryBuilder()
qb.append(WorkCalculation,tag='workchain',project='*', filters={'attributes._finished': True})
qb.append(StructureData,input_of='workchain', tag='input_structure',project='*')
qb.append(Group,group_of='input_structure',filters={'name':group_name})
qb.append(Group,group_of='workchain',filters={'name':wc_group_name})

all_results = qb.all()
print "Found {} systems to potentially process".format(len(all_results))

#structure_list = list(g.nodes)
import copy
import sys
for (wc,old_structure) in  all_results:
    if wc.get_attr('_failed',False):
        print '*'*10,wc.inp.structure.get_formula(),' failed!'
        continue
    if only_these_structures is not None:
        if old_structure.get_formula() not in only_these_structures:
            print "> Skipping {} as requested.".format(old_structure.get_formula())
            continue
    #else:
    #    print old_structure.get_formula(),' launching'
    #    print wc
        #sys.exit()
    seekpath_params = 	wc.get_outputs_dict()['seekpath_parameters']
    structure = wc.out.primitive_structure
    tmp_params = copy.deepcopy(wc.out.scf_parameters.inp.output_parameters.inp.parameters.get_dict())
    tmp_params['CONTROL']['max_seconds'] = max_sec_scf-60
    tmp_params['CONTROL']['wf_collect'] = True
    scf_parameters = ParameterData(dict=tmp_params)
    tmp_params = copy.deepcopy(tmp_params)
    if only_valence:
        nbnd = int(wc.out.scf_parameters.dict.number_of_electrons)/2
    else:
        nbnd = int(wc.out.scf_parameters.dict.number_of_electrons*1.5)  #Three times the number of occupied bands
    tmp_params['SYSTEM']['nbnd'] = nbnd
    tmp_params['CONTROL']['calculation'] = 'nscf'
    tmp_params['CONTROL']['max_seconds'] = max_sec_nscf-60
    nscf_parameters = ParameterData(dict=tmp_params)

#    scf_parameters = ParameterData(dict={
#        'CONTROL': {
#            'restart_mode': 'from_scratch',
#        },
#        'SYSTEM': {
#            'ecutwfc': 30.,
#            'ecutrho': 240.,
#        },
#    })
#    nscf_parameters = ParameterData(dict={
#        'SYSTEM': {
#            'nbnd': 12,
#            'ecutwfc': 30.,
#            'ecutrho': 240.,
#        },
#    })
    scf_settings =  ParameterData(dict={'cmdline':['-nk',str(num_pools)]})
    scf_options  = ParameterData(dict={
                        'resources': {
                                    'num_machines': 1,
                                    'tot_num_mpiprocs': tot_num_mpiprocs,
                                     },
                        'max_wallclock_seconds': max_sec_scf,#'queue_name':'debug',
                             })
    nscf_options  = ParameterData(dict={
                        'resources': {
                                    'num_machines': 1,
                                    'tot_num_mpiprocs': tot_num_mpiprocs,
                                     },
                        'max_wallclock_seconds': max_sec_nscf,#'queue_name':'debug',
                             })
    pw2wannier90_options  = ParameterData(dict={
                        'resources': {
                                    'num_machines': 1,
                                    'tot_num_mpiprocs': tot_num_mpiprocs,
                                     },
                        'max_wallclock_seconds': max_sec_pw2wannier90,#'queue_name':'debug',
                             })
    wannier90_params_dict = {'bands_plot':True,
                                'write_hr':True,
                                'write_xyz':True,
                                'use_ws_distance':True,
                                'num_iter': 0,
                                #'guiding_centres': True,
                                #'dis_win_max':10.0,
                                'scdm_proj': True,
                                #'dis_num_iter':120,
                                #'dis_mix_ratio':1.d0,
                                #'exclude_bands': range(5,13),
                                'wannier_plot': plot_wfs,
                                # 'wannier_plot_list':[1]
                                }
    if exclude_bands is not None:
        wannier90_params_dict['exclude_bands'] = exclude_bands
    if only_valence:
        wannier90_params_dict['scdm_entanglement'] = 'isolated'
        wannier90_params_dict['num_wann'] = nbnd
    else:
        wannier90_params_dict['num_bands'] = nbnd
        wannier90_params_dict['scdm_entanglement'] = 'erfc'
        ## Removed because I set set_mu_from_projections later
        #wannier90_params_dict['scdm_mu'] = 10
        # This has to be set in the input (required internally) ONLY FOR THE METHOD THAT SETS ONLY MU
        #wannier90_params_dict['scdm_sigma'] = 4.
    
        wannier90_params_dict['dis_num_iter'] = 0
        if do_disen:
            raise NotImplementedError("We currently disable this branch of the code (do_disen) because this would need zero_is_fermi that does not work with set_mu_prom_projections")
            wannier90_params_dict['dis_froz_max'] = 1.
            wannier90_params_dict['dis_num_iter'] = 1000
    if do_mlwf:
        wannier90_params_dict.update({
                                'num_iter': 10000,
                                'conv_tol':1e-7,
                                'conv_window':3,
                                })
    #
    wannier90_parameters = ParameterData(dict=wannier90_params_dict)

    projwfc_parameters = ParameterData(dict={'PROJWFC': {'DeltaE' : 0.2}})
    projwfc_options = ParameterData(dict={
    'resources': {
        'num_machines': 1,
        'tot_num_mpiprocs': tot_num_mpiprocs,
    },
    'max_wallclock_seconds': max_sec_projwfc,#'queue_name':'debug', ## always 30 mins
    })                               

    wannier_pp_options = ParameterData(dict={
    'resources': {
        'num_machines': 1,
        'tot_num_mpiprocs': 1,
    },
    'max_wallclock_seconds': max_sec_wannier_pp,#'queue_name':'debug',
    })
    wannier_options = ParameterData(dict={
    'resources': {
        'num_machines': 1,
        'tot_num_mpiprocs': 1,
    },
    'max_wallclock_seconds': max_sec_wannier,#'queue_name':'debug',
    })
    wc_control = {'group_name':g_name,'retrieve_hamiltonian':True} 
    # I do not use 'zero_is_fermi':True because now I use set_mu_from_projections

    wc_control['write_unk'] =  plot_wfs

    if only_valence:
        wc_control['set_auto_wann'] = False
    else:
        wc_control['set_auto_wann'] = True
        wc_control['set_mu_and_sigma_from_projections'] = True
      #  wc_control['max_projectability'] = 0.95
        wc_control['sigma_factor_shift'] = 0.


#   scf_kpoints = KpointsData()
#   scf_kpoints.set_kpoints_mesh([8,8,8])
    scf_kpoints = wc.out.scf_parameters.inp.output_parameters.inp.kpoints

    nscf_kpoints = KpointsData()
    nscf_kpoints.set_cell_from_structure(structure)
    nscf_kpoints.set_kpoints_mesh_from_density(0.2)   

#    nscf_kpoints.set_kpoints_mesh([6,6,6])
    #projections = List()
    #projections.extend(['Si:sp3'])
    #print structure.get_formula(), nscf_parameters.get_dict(),nscf_kpoints.get_kpoints_mesh()
    #continue
   
    try:
        output_group = Group.get(name=g_name)
        g_statistics = "that already contains {} nodes".format(len(output_group.nodes))
    except NotExistent:
        g_statistics = "that does not exist yet"
    if only_valence:
        print "Running 'only_valence/insulating', WANN={} for {}".format(
            do_mlwf, structure.get_formula())
    else:
        print "Running 'with conduction bands', DISENT={}, WANN={} for {}".format(
            do_disen, do_mlwf, structure.get_formula())
    if exclude_bands is None:
        print "NO excluded bands"
    else:
        print "Excluded bands: {}".format(", ".join(str(num) for num in exclude_bands))

    print "Using codes:"
    print "- {}@{}".format(pw_code.label, pw_code.get_computer().name)
    print "- {}@{}".format(pw2wannier90_code.label, pw2wannier90_code.get_computer().name)
    print "- {}@{}".format(projwfc_code.label, projwfc_code.get_computer().name)
    print "- {}@{}".format(wannier90_code.label, wannier90_code.get_computer().name)

    print "Then adding to '{}' group {}".format(
        g_name, g_statistics)
    print "Continue? [CTRL+C to stop]"
    raw_input()
    wc = submit(SimpleWannier90WorkChain,
        pw_code=pw_code,
        pw2wannier90_code=pw2wannier90_code,
        wannier90_code=wannier90_code,
        projwfc_code=projwfc_code,
        #orbital_projections=projections,
        structure=structure,
        pseudo_family=Str('SSSP_efficiency_v0.95_with_wfc'),
        #wannier90_parameters=wannier90_parameters,
        scf={'parameters':scf_parameters,'kpoints':scf_kpoints,'settings':scf_settings,'options':scf_options},
        nscf={'parameters':nscf_parameters,'kpoints':nscf_kpoints, 'options':nscf_options},
        projwfc={'parameters':projwfc_parameters,'_options':projwfc_options},
        mlwf={ 'parameters':wannier90_parameters,'pp_options':wannier_pp_options,
              'options':wannier_options ,'kpoint_path': seekpath_params},
        matrices={'_options':pw2wannier90_options},
#        restart_options = {'scf_workchain':load_node(18142),'nscf_workchain':load_node(18155)},
#'mlwf_pp':load_node(9526),
#                         'pw2wannier90':load_node(9532)
#                          },
        workchain_control = ParameterData(dict=wc_control),
    )

    print 'launched WorkChain pk {} for structure {}'.format(wc.pid,structure.get_formula())
print 'Output will be added to group: {}'.format(g_name)

