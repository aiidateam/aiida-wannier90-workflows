# -*- coding: utf-8 -*-
from aiida.orm import Code
from aiida.orm.data.base import Str, Float
from aiida.orm.data.parameter import ParameterData
from aiida.orm.data.structure import StructureData
from aiida.orm.data.array.bands import BandsData
from aiida.orm.data.array.kpoints import KpointsData
from aiida.orm.data.singlefile import SinglefileData
from aiida.orm.data.folder import FolderData
from aiida.orm.data.remote import RemoteData
from aiida.orm.data.orbital import OrbitalData
from aiida.common.links import LinkType
from aiida.common.exceptions import AiidaException, NotExistent
from aiida.common.datastructures import calc_states
from aiida.common.example_helpers import test_and_get_code
from aiida.work.run import submit
from aiida.work.workchain import WorkChain, ToContext, while_, append_,if_
from aiida.work.workfunction import workfunction
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.calculations.pw2wannier90 import Pw2wannier90Calculation
from aiida_wannier90.calculations import Wannier90Calculation
#from seekpath.aiidawrappers import get_path, get_explicit_k_path
from aiida.orm.data.base import List

class SimpleWannier90WorkChain(WorkChain):
    """
    Workchain to obtain maximally localised Wannier functions
    """

    @classmethod
    def define(cls, spec):
        super(SimpleWannier90WorkChain, cls).define(spec)
        spec.input('pw_code', valid_type=Code)
        spec.input('pw2wannier90_code', valid_type=Code)
        spec.input('wannier90_code', valid_type=Code)
        spec.input('structure', valid_type=StructureData)
        spec.input('pseudo_family', valid_type=Str)
        spec.input('vdw_table', valid_type=SinglefileData, required=False)
        spec.input('workchain_control',valid_type=ParameterData, required=False)
        spec.input_group('scf')
        spec.input_group('nscf')
        spec.input_group('mlwf')
        spec.input_group('matrices')
        spec.input_group('restart_options')
        spec.outline(
            cls.setup,
            if_(cls.should_run_scf)(
                cls.run_scf),
            if_(cls.should_run_nscf)(
                cls.run_nscf),
           #cls.run_bands,
            if_(cls.should_run_mlwf_pp)(
                cls.run_wannier90_pp),
            if_(cls.should_run_pw2wannier90)(
            cls.run_pw2wannier90),
            cls.run_wannier90,
            cls.results,
                    )

        #Workchain parameters/options
        #-Retrieve amn mmn
        #-Get TB model...

        #SCF and NSCF OUTPUT PARAMS (scf_out_parameters)
        spec.output('scf_output_parameters', valid_type=ParameterData)
        spec.output('nscf_output_parameters', valid_type=ParameterData)
        spec.output('overlap_matrices_remote_folder',valid_type=RemoteData)
        spec.output('overlap_matrices_local_folder',valid_type=FolderData,required=False)
        #spec.output('pw2wannier90_output_parameters')
        spec.output('mlwf_output_parameters')
        spec.output('mlwf_interpolated_bands', required=False)

        #spec.output('band_parameters', valid_type=ParameterData)
        #spec.output('dft_band_structure', valid_type=BandsData)
        #spec.output('interpolated_band_structure', valid_type=BandsData)

    def setup(self):
        """
        Input validation and context setup
        """
        self.ctx.inputs = {
            'code': self.inputs.pw_code,
       #     'parameters': self.inputs.scf_parameters.get_dict(),
            #'settings': self.inputs.settings,
            #'options': self.inputs.options,
        }

        # Evaluating restart options
        try:
            restart_options = self.inputs.restart_options
            self.report('Restart options found in input.')
        except AttributeError:
            restart_options = None
        if restart_options is not None:
            try:
                scf_WC = restart_options['scf_workchain']
                self.report('Previous SCF WorkChain (pk: {} ) found in input'.format(scf_WC.pk))
                # TO ADD: check that scf parameters ARE NOT SPECIFIED...
                self.ctx.do_scf = False
                self.ctx.workchain_scf = scf_WC
            except KeyError:
                self.ctx.do_scf = True
            try:
                nscf_WC = restart_options['nscf_workchain']
                self.report('Previous NSCF WorkChain (pk: {} ) found in input'.format(nscf_WC.pk))
                self.ctx.do_nscf = False
                self.ctx.workchain_nscf = nscf_WC
            except KeyError:
                self.ctx.do_nscf = True
            try:
                calc_mlwf_pp = restart_options['mlwf_pp']
                self.report('Previous mlwf post_proc calc (pk: {} ) found in input'.format(calc_mlwf_pp.pk))
                self.ctx.do_mlwf_pp = False
                self.ctx.calc_mlwf_pp = calc_mlwf_pp
            except KeyError:
                self.ctx.do_mlwf_pp = True
            try:
                calc_pw2wannier90 = restart_options['pw2wannier90']
                self.report('Previous pw2wannier90 calc (pk: {} ) found in input'.format(calc_pw2wannier90.pk))
                self.ctx.do_pw2wannier90 = False
                self.ctx.calc_pw2wannier90 = calc_pw2wannier90
            except KeyError:
                self.ctx.do_pw2wannier90 = True
        else:
            self.ctx.do_scf = True
            self.ctx.do_nscf = True
            self.ctx.do_mlwf_pp = True
            self.ctx.do_pw2wannier90 = True

        try:
            control_options = self.inputs.workchain_control
            self.report('Workchain control options found in input.')
        except AttributeError:
            control_options = None

            # We expect either a KpointsData with given mesh or a desired distance between k-points
#        if all([key not in self.inputs for key in ['kpoints_mesh', 'kpoints_distance']]):
#            self.abort_nowait('neither the kpoints_mesh nor a kpoints_distance was specified in the inputs')
#           return

        # Add the van der Waals kernel table file if specified
#        if 'vdw_table' in self.inputs:
#            self.ctx.inputs['vdw_table'] = self.inputs.vdw_table
#            self.inputs.relax['vdw_table'] = self.inputs.vdw_table

#        # Set the correct relaxation scheme in the input parameters
#        if 'CONTROL' not in self.ctx.inputs['parameters']:
#            self.ctx.inputs['parameters']['CONTROL'] = {}
        self.report('Setup step completed.')

        return

    def run_relax(self):
        """
        Run the PwRelaxWorkChain to run a relax PwCalculation
        """
        inputs = self.inputs.relax
        inputs.update({
            'code': self.inputs.pw_code,
            'structure': self.inputs.structure,
            'pseudo_family': self.inputs.pseudo_family,
        })

        running = submit(PwRelaxWorkChain, **inputs)

        self.report('launching PwRelaxWorkChain<{}>'.format(running.pid))

        return ToContext(workchain_relax=running)


    def run_scf(self):
        """
        Run the PwBaseWorkChain in scf mode on the primitive cell of the relaxed input structure
        """
        #inputs = dict(self.ctx.inputs)
        inputs = self.inputs.scf
        inputs.update({
            'code': self.inputs.pw_code,
            'structure': self.inputs.structure,
            'pseudo_family': self.inputs.pseudo_family,
        })

        calculation_mode = 'scf'

        # Set the correct pw.x input parameters
        parameters = inputs['parameters'].get_dict()
        parameters['CONTROL']['calculation'] = calculation_mode
        inputs['parameters'] = parameters
        # Construct a new kpoint mesh on the current structure or pass the static mesh
        if 'scf_kpoints_distance' in self.inputs:
            kpoints_mesh = KpointsData()
            kpoints_mesh.set_cell_from_structure(self.inputs.structure)
            kpoints_mesh.set_kpoints_mesh_from_density(self.inputs.scf_kpoints_distance.value)
        else:
            kpoints_mesh = inputs['kpoints']

        # Final input preparation, wrapping dictionaries in ParameterData nodes
        inputs['kpoints'] = kpoints_mesh
        inputs['parameters'] = ParameterData(dict=inputs['parameters'])

        running = submit(PwBaseWorkChain, **inputs)

        self.report('SCF step - launching PwBaseWorkChain<{}> in {} mode'.format(running.pid, calculation_mode))

        return ToContext(workchain_scf=running)

    def run_nscf(self):
        """
        Run the PwBaseWorkChain in nscf mode
        """
        try:
            remote_folder = self.ctx.workchain_scf.out.remote_folder
        except AttributeError as exception:
            self.abort_nowait('the scf workchain did not output a remote_folder node')
            return
        inputs = self.inputs.scf
        inputs.update({
            'code': self.inputs.pw_code,
            'structure': self.inputs.structure,
            'pseudo_family': self.inputs.pseudo_family,
        })

        restart_mode = 'from_scratch'
        calculation_mode = 'nscf'
        # Set the correct pw.x input parameters
        parameters = inputs['parameters'].get_dict()
        nscf_params = self.inputs.nscf
        nscf_params = nscf_params['parameters'].get_dict()
        parameters.update(nscf_params)
        parameters['CONTROL']['restart_mode'] = restart_mode
        parameters['CONTROL']['calculation'] = calculation_mode
        parameters['SYSTEM']['nosym'] = True

        settings = inputs.pop('settings',None)
        if settings is not None:
            settings = settings.get_dict()
            settings['FORCE_KPOINTS_LIST'] = True
        else:
            settings = {'FORCE_KPOINTS_LIST':True}

        inputs['settings'] = ParameterData(dict=settings)


        try:
            parameters['ELECTRONS']['diagonalization'] = 'cg'
        except KeyError:
            parameters['ELECTRONS'] = {'diagonalization': 'cg '}

        parameters['ELECTRONS']['diago_full_acc'] = True

       # parameters['SYSTEM']['nbnd'] =

        inputs['parameters'] = parameters
        inputs['parameters'] = ParameterData(dict=inputs['parameters'])

        # Tell the plugin to retrieve the bands
        #settings = inputs['settings'].get_dict()
        #settings['also_bands'] = True
        # Construct a new kpoint mesh on the current structure or pass the static mesh
        if 'nscf_kpoints_distance' in self.inputs:
            kpoints_mesh = KpointsData()
            kpoints_mesh.set_cell_from_structure(self.inputs.structure)
            kpoints_mesh.set_kpoints_mesh_from_density(self.inputs.scf_kpoints_distance.value)
        else:
            kpoints_mesh = self.inputs.nscf['kpoints']
        # Final input preparation, wrapping dictionaries in ParameterData nodes
        inputs['kpoints'] = kpoints_mesh
        inputs['parent_folder'] = remote_folder
        inputs['pseudo_family'] = self.inputs.pseudo_family

        # Final input preparation, wrapping dictionaries in ParameterData nodes

        running = submit(PwBaseWorkChain, **inputs)

        self.report('NSCF step - launching PwBaseWorkChain<{}> in {} mode'.format(running.pid, calculation_mode))

        return ToContext(workchain_nscf=running)

    def run_wannier90_pp(self):
        try:
            remote_folder = self.ctx.workchain_nscf.out.remote_folder
        except AttributeError as exception:
            self.abort_nowait('the nscf workchain did not output a remote_folder node')
            return

        #try:
        #    orbital_projections = self.inputs.orbital_projections
        #except KeyError:
        #    pass
        inputs = self.inputs.mlwf
        #if local_input:
        #    calc.use_local_input_folder(input_folder)
        #else:
        #    calc.use_remote_input_folder(remote_folder)
        #calc = self.inputs.wannier90_code.new_calc()
        inputs['code'] = self.inputs.wannier90_code
        inputs['kpoints'] = self.ctx.workchain_nscf.inp.kpoints
        structure = self.inputs.structure
        inputs['structure'] = structure
        inputs['settings'] = ParameterData(dict={'postproc_setup':True})
        wannier_pp_options = inputs.pop('pp_options',None)
        wannier_options = inputs.pop('options',None)

        inputs['_options'] = wannier_pp_options.get_dict()

        w90_params = inputs['parameters'].get_dict()
        try:
            bands_plot = w90_params['bands_plot']
        except KeyError:
            bands_plot = False
        self.ctx.bands_plot = bands_plot
        if bands_plot:
            kpoints_path = KpointsData()
            kpoints_path.set_cell_from_structure(structure)
            try:
                kpath = inputs['kpoint_path']
            except KeyError:
                kpath = None
            kpoints_path.set_kpoints_path(kpath)
            inputs['kpoint_path'] = kpoints_path


        # settings that can only be enabled if parent is nscf
#        settings_dict = {'seedname':'gaas','random_projections':True}
        process = Wannier90Calculation.process()
        running = submit(process, **inputs)

        #calc.store_all()
        #running = calc.submit()
        self.report('MLWF PP step - launching Wannier90 Calculation <{}> in pp mode'.format(running.pid))

        return ToContext(calc_mlwf_pp=running)


    def run_pw2wannier90(self):
        try:
            remote_folder = self.ctx.workchain_nscf.out.remote_folder
        except AttributeError as exception:
            self.abort_nowait('the nscf workchain did not output a remote_folder node')
            return
        inputs = self.inputs.matrices
        inputs['parent_folder'] = remote_folder
        inputs['code'] = self.inputs.pw2wannier90_code
        inputs['nnkp_file'] = self.ctx.calc_mlwf_pp.out.output_nnkp
        inputs['_options']= inputs['_options'].get_dict()
#        inputs['_options'] = {
#            'resources': {
#                'num_machines': 1,
#                'tot_num_mpiprocs': 1,
#            },
#            'max_wallclock_seconds': 60*10,
#        }
#        inputs['settings'] =  ParameterData(dict={})
        inputs['parameters'] = ParameterData(dict={'inputpp':{'write_mmn':True,'write_amn':True}})
        process = Pw2wannier90Calculation.process()
        running = submit(process, **inputs)
        self.report('Pw2wannier90 step - launching Pw2Wannier90 Calculation <{}>'.format(running.pid))
        return ToContext(pw2wannier90=running)

    def run_wannier90(self):
        try:
            remote_folder = self.ctx.calc_pw2wannier90.out.remote_folder
        except AttributeError as exception:
            self.abort_nowait('the pw2wannier90 calculation di not output a remote_folder node')
            return

        inputs = self.inputs.mlwf
        inputs['remote_input_folder'] = remote_folder
        inputs['code'] = self.inputs.wannier90_code
        inputs['kpoints'] = self.ctx.workchain_nscf.inp.kpoints
        structure = self.inputs.structure
        inputs['structure'] = structure

        w90_params = inputs['parameters'].get_dict()
        try:
            bands_plot = w90_params['bands_plot']
        except KeyError:
            bands_plot = False
        self.ctx.bands_plot = bands_plot
        if bands_plot:
            kpoints_path = KpointsData()
            kpoints_path.set_cell_from_structure(structure)
            try:
                kpath = inputs['kpoint_path']
            except KeyError:
                kpath = None
            kpoints_path.set_kpoints_path(kpath)
            inputs['kpoint_path'] = kpoints_path

        wannier_pp_options = inputs.pop('pp_options',None)
        wannier_options = inputs.pop('options',None)
        inputs['_options'] = wannier_pp_options.get_dict()

        process = Wannier90Calculation.process()
        running = submit(process, **inputs)
        self.report('MLWF step - launching Wannier90 Calculation <{}>'.format(running.pid))

        return ToContext(calc_mlwf=running)

    def run_bands(self):
        """
        Run the PwBaseWorkChain to run a bands PwCalculation
        """
        try:
            remote_folder = self.ctx.workchain_scf.out.remote_folder
        except AttributeError as exception:
            self.abort_nowait('the scf workchain did not output a remote_folder node')
            return

        inputs = dict(self.ctx.inputs)
        structure = self.ctx.structure_relaxed_primitive
        restart_mode = 'restart'
        calculation_mode = 'bands'

        # Set the correct pw.x input parameters
        inputs['parameters']['CONTROL']['restart_mode'] = restart_mode
        inputs['parameters']['CONTROL']['calculation'] = calculation_mode

        # Tell the plugin to retrieve the bands
        settings = inputs['settings'].get_dict()
        settings['also_bands'] = True

        # Final input preparation, wrapping dictionaries in ParameterData nodes
        inputs['kpoints'] = self.ctx.kpoints_path
        inputs['structure'] = structure
        inputs['parent_folder'] = remote_folder
        inputs['parameters'] = ParameterData(dict=inputs['parameters'])
        inputs['settings'] = ParameterData(dict=settings)
        inputs['pseudo_family'] = self.inputs.pseudo_family

        running = submit(PwBaseWorkChain, **inputs)

        self.report('launching PwBaseWorkChain<{}> in {} mode'.format(running.pid, calculation_mode))

        return ToContext(workchain_bands=running)

    def results(self):
        """
        Attach the desired output nodes directly as outputs of the workchain
        """
        #calculation_band = self.ctx.workchain_bands.get_outputs(link_type=LinkType.CALL)[0]
        self.report('Final step, preparing outputs')
        self.out('scf_output_parameters', self.ctx.workchain_scf.out.output_parameters)
        self.out('nscf_output_parameters', self.ctx.workchain_nscf.out.output_parameters)
        self.out('mlwf_output_parameters', self.ctx.calc_mlwf.out.output_parameters)
        self.out('overlap_matrices_remote_folder',self.ctx.calc_pw2wannier90.out.remote_folder)
        try:
            self.out('overlap_matrices_local_folder',self.ctx.calc_pw2wannier90.out.retrieved)
        except AttributeError:
            pass

        if self.ctx.bands_plot:
            try:
                self.out('mlwf_interpolated_bands', self.ctx.calc_mlwf.out.interpolated_bands)
            except:
                self.report('WARNING: interpolated bands missing, while they should be there.')

        self.report('Wannier90WorkChain succesfully completed.')


    def should_run_scf(self):
        """
        Return whether a scf WorkChain should be run or it is provided by input
        """
        return self.ctx.do_scf
    def should_run_nscf(self):
        """
        Return whether a nscf WorkChain should be run or it is provided by input
        """
        return self.ctx.do_nscf
    def should_run_mlwf_pp(self):
        """
        Return whether a nscf WorkChain should be run or it is provided by input
        """
        return self.ctx.do_mlwf_pp
    def should_run_pw2wannier90(self):
        """
        Return whether a nscf WorkChain should be run or it is provided by input
        """
        return self.ctx.do_pw2wannier90
