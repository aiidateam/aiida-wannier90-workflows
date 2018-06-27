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
from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation
from aiida_wannier90.calculations import Wannier90Calculation
#from seekpath.aiidawrappers import get_path, get_explicit_k_path
from aiida.orm.data.base import List
from aiida.orm import Group
import copy

class Wannier90WorkChain(WorkChain):
    """
    Workchain to obtain maximally localised Wannier functions (MLWF)
    Authors: Antimo Marrazzo (antimo.marrazzo@epfl.ch), Giovanni Pizzi (giovanni.pizzi@epfl.ch)

    Scheme: SETUP-->SCF-->NSCF-->PROJWFC -> W90_PP-->PW2WANNIER90-->WANNIER90-->RESULTS

    """

    @classmethod
    def define(cls, spec):
        super(SimpleWannier90WorkChain, cls).define(spec)
        spec.input('pw_code', valid_type=Code)
        spec.input('projwfc_code', valid_type=Code, required=False)
        spec.input('pw2wannier90_code', valid_type=Code)
        spec.input('wannier90_code', valid_type=Code)
        spec.input('structure', valid_type=StructureData)
        spec.input('pseudo_family', valid_type=Str)
        spec.input('vdw_table', valid_type=SinglefileData, required=False)
        spec.input('workchain_control',valid_type=ParameterData, required=False)
        spec.input_group('scf')
        spec.input_group('nscf')
        spec.input_group('projwfc',required=False)
        spec.input_group('mlwf')
        spec.input_group('matrices')
        spec.input_group('restart_options')
        spec.outline(
            cls.setup,
            if_(cls.should_run_scf)(
                cls.run_scf),
            if_(cls.should_run_nscf)(
                cls.run_nscf),
            if_(cls.should_run_projwfc)(
                cls.run_projwfc),
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
        spec.output('mlwf_output_parameters')
        spec.output('mlwf_interpolated_bands', required=False)


    def setup(self):
        """
        Input validation and context setup
        """
        self.ctx.inputs = {
            'code': self.inputs.pw_code,
            #'parameters': self.inputs.scf_parameters.get_dict(),
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
        #Checking workchain options specified in input
        try:
            control_options = self.inputs.workchain_control
            self.report('Workchain control options found in input.')
            control_dict = control_options.get_dict()
        except AttributeError:
            control_dict = {}

        #Write UNK files (to plot WFs)
        self.ctx.write_unk = control_dict.get('write_unk', False)
        self.report("UNK files will {}be written.".format("" if self.ctx.write_unk else "NOT "))

        #Retrive Hamiltonian
        try:
            self.ctx.retrieve_ham = control_dict['retrieve_hamiltonian']
        except KeyError:
            self.ctx.retrieve_ham = False
        if self.ctx.retrieve_ham:
            self.report("The Wannier Hamiltonian will be retrieved.")
        #Group name
        try:
            self.ctx.group_name = control_dict['group_name']
        except KeyError:
            self.ctx.retrieve_ham = None
        if self.ctx.group_name is not None:
            self.report("Wannier90 calc will be added to the group {}.".format(self.ctx.group_name))
        #Zero at Fermi
        try:
            self.ctx.zero_is_fermi = control_dict['zero_is_fermi']
        except KeyError:
            self.ctx.zero_is_fermi = False
        if self.ctx.zero_is_fermi:
            self.report("W90 windows defined with the Fermi level as zero.")
        # Do projwfc
        try:
            self.ctx.do_projwfc = control_dict['use_projwfc']
            use_projwfc_specified = True
        except KeyError:
            self.ctx.do_projwfc = False
            use_projwfc_specified = False

        if self.ctx.do_projwfc:
            self.report("A Projwfc calculation is performed.")

        try:
            self.ctx.set_auto_wann = control_dict['set_auto_wann']
        except KeyError:
            self.ctx.set_auto_wann = False
        if self.ctx.set_auto_wann:
            if not self.ctx.do_projwfc and use_projwfc_specified:
                self.abort_nowait('set_auto_wann = True and use_projwfc = False are incompatible')
            else:
                self.ctx.do_projwfc = True
            self.report("The number of WFs auto-set.")

        try:
            self.ctx.max_projectability = control_dict['max_projectability']
            self.report("Max projectability set to {}.".format(self.ctx.max_projectability))
        except KeyError:
            self.ctx.max_projectability = 0.95
            self.report("Max projectability set to {} (DEFAULT).".format(self.ctx.max_projectability))

        try:
            self.ctx.sigma_factor_shift = control_dict['sigma_factor_shift']
            self.report("Sigma factor shift set to {}.".format(self.ctx.sigma_factor_shift))
        except KeyError:
            self.ctx.sigma_factor_shift = 3.
            self.report("Sigma factor shift set to {} (DEFAULT).".format(self.ctx.sigma_factor_shift))


        try:
            self.ctx.set_mu_and_sigma_from_projections = control_dict['set_mu_and_sigma_from_projections']
        except KeyError:
            self.ctx.set_mu_and_sigma_from_projections = False
        if self.ctx.set_mu_and_sigma_from_projections:
            if self.ctx.zero_is_fermi:
                self.abort_nowait('zero_is_fermi = True and set_mu_and_sigma_from_projections = True are incompatible. '
                                  'You do not need to use zero_is_fermi in this case!')
            if not self.ctx.do_projwfc:
                self.abort_nowait('use_projwfc = False and set_mu_and_sigma_from_projections = True are incompatible. '
                                  'You need to do a projwfc calculations to do that! Set use_projwfc = True.')

            self.report("SCDM mu is auto-set using projectability.")
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
        try:
            inputs['options'] = self.inputs.nscf['options']
        except KeyError:
            self.report("No options (walltime, resources, etc.) specified for NSCF, I will use those for the SCF step.")
        # Final input preparation, wrapping dictionaries in ParameterData nodes

        running = submit(PwBaseWorkChain, **inputs)

        self.report('NSCF step - launching PwBaseWorkChain<{}> in {} mode'.format(running.pid, calculation_mode))

        return ToContext(workchain_nscf=running)

    def run_projwfc(self):
        """
        Projwfc step
        :return:
        """
        inputs = self.inputs.projwfc
        try:
            remote_folder = self.ctx.workchain_nscf.out.remote_folder
        except AttributeError as exception:
            self.abort_nowait('the nscf workchain did not output a remote_folder node')
            return

        inputs['code'] = self.inputs.projwfc_code
        inputs['parent_folder'] = remote_folder
        projwfc_options = inputs.pop('_options',None)
        inputs['_options'] = projwfc_options.get_dict()

        process = ProjwfcCalculation.process()
        running = submit(process, **inputs)
        self.report('PROJWFC step - launching Projwfc Calculation <{}>.'.format(running.pid))
        return ToContext(calc_projwfc=running)


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
        inputs = copy.deepcopy(self.inputs.mlwf)
        #if local_input:
        #    calc.use_local_input_folder(input_folder)
        #else:
        #    calc.use_remote_input_folder(remote_folder)
        #calc = self.inputs.wannier90_code.new_calc()


        if self.ctx.set_auto_wann:
            parameters = inputs['parameters']
            inputs['parameters'] = \
                set_auto_numwann(parameters, self.ctx.calc_projwfc.out.projections)['output_parameters']

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
        try:
            efermi = self.ctx.workchain_scf.out.output_parameters.get_dict()['fermi_energy']
            efermi_units = self.ctx.workchain_scf.out.output_parameters.get_dict()['fermi_energy_units']
            if efermi_units != 'eV':
                raise TypeError("Error: Fermi energy is not in eV!"
                                "it is {}".format(efermi_units))
        except AttributeError:
            raise TypeError("Error in retriving the SCF Fermi energy "
                            "from pk: {}".format(self.ctx.workchain_scf.res.fermi_energy.pk))

        if self.ctx.zero_is_fermi:
            inputs['parameters'] = update_w90_params_zero_with_fermi(
                parameters=inputs['parameters'],
                fermi=Float(efermi)
                )['output_parameters']
        if self.ctx.set_mu_and_sigma_from_projections:
            results = set_mu_and_sigma_from_projections(parameters=inputs['parameters'],
                bands = self.ctx.calc_projwfc.out.bands,
                projections=self.ctx.calc_projwfc.out.projections,
                thresholds=ParameterData(dict={
                    #'max_projectability': self.ctx.max_projectability,
                    'sigma_factor_shift': self.ctx.sigma_factor_shift,
                }),
            )
            if not results['success'].value:
                self.abort_nowait('WARNING: set_mu_and_sigma_from_projection failed!')
            inputs['parameters'] = results['output_parameters']



        self.ctx.bands_plot = bands_plot
        if bands_plot:
            try:
                kpath = inputs['kpoint_path']
            except KeyError:
                kpoint_path_tmp = KpointsData()
                kpoint_path_tmp.set_cell_from_structure(structure)
                kpoint_path_tmp.set_kpoints_path()
                point_coords, path = kpoint_path_tmp.get_special_points()
                kpoint_path = ParameterData(dict={
                    'path': path,
                    'point_coords': point_coords,
                })
            else:
                if isinstance(kpath,ParameterData):
                    kpoint_path = from_seekpath_to_wannier(kpath)['kpoint_info']
                else:
                    self.abort_nowait('WARNING: aborting, kpoint_path for W90 not recognised!')

            inputs['kpoint_path'] = kpoint_path


        # settings that can only be enabled if parent is nscf
#       settings_dict = {'seedname':'gaas','random_projections':True}
        process = Wannier90Calculation.process()
        running = submit(process, **inputs)

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
        inputs['parameters'] = ParameterData(dict={
            'inputpp':{
                'write_mmn':True,
                'write_amn':True,
                'write_unk': self.ctx.write_unk,
                }
            })
        process = Pw2wannier90Calculation.process()
        running = submit(process, **inputs)
        self.report('Pw2wannier90 step - launching Pw2Wannier90 Calculation <{}>'.format(running.pid))
        return ToContext(calc_pw2wannier90=running)

    def run_wannier90(self):
        try:
            remote_folder = self.ctx.calc_pw2wannier90.out.remote_folder
        except AttributeError as exception:
            self.abort_nowait('the pw2wannier90 calculation di not output a remote_folder node')
            return

        inputs = copy.deepcopy(self.inputs.mlwf)
        inputs['remote_input_folder'] = remote_folder
        inputs['code'] = self.inputs.wannier90_code
        inputs['kpoints'] = self.ctx.workchain_nscf.inp.kpoints
        structure = self.inputs.structure
        inputs['structure'] = structure
        #Using the parameters of the W90 pp
        inputs['parameters'] =  self.ctx.calc_mlwf_pp.inp.parameters
        w90_params = self.ctx.calc_mlwf_pp.inp.parameters.get_dict()
        try:
            bands_plot = w90_params['bands_plot']
        except KeyError:
            bands_plot = False
        self.ctx.bands_plot = bands_plot
        if bands_plot:
            try:
                kpath = inputs['kpoint_path']
            except KeyError:
                kpoint_path_tmp = KpointsData()
                kpoint_path_tmp.set_cell_from_structure(structure)
                kpoint_path_tmp.set_kpoints_path()
                point_coords, path = kpoint_path_tmp.get_special_points()
                kpoint_path = ParameterData(dict={
                    'path': path,
                    'point_coords': point_coords,
                })
            else:
                if isinstance(kpath,ParameterData):
                    kpoint_path = from_seekpath_to_wannier(kpath)['kpoint_info']
                else:
                    self.abort_nowait('WARNING: aborting, kpoint_path for W90 not recognised!')

            inputs['kpoint_path'] = kpoint_path



        wannier_pp_options = inputs.pop('pp_options',None)
        wannier_options = inputs.pop('options',None)
        inputs['_options'] = wannier_options.get_dict()

        #Check if settings is given in input
        try:
            settings = inputs['settings']
        except KeyError:
            settings = None
        #Check if the hamiltonian needs to be retrieved or not
        if self.ctx.retrieve_ham:
            if settings is None:
                settings = {}
            settings['retrieve_hoppings'] = True
        if settings is not None:
            inputs['settings'] = ParameterData(dict=settings)

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
        if self.ctx.group_name is not None:
            g,_ = Group.get_or_create(name=self.ctx.group_name)
            g.add_nodes(self.ctx.calc_mlwf)

        self.out('overlap_matrices_remote_folder',self.ctx.calc_pw2wannier90.out.remote_folder)
        try:
            self.out('overlap_matrices_local_folder',self.ctx.calc_pw2wannier90.out.retrieved)
        except AttributeError:
            pass

        if self.ctx.bands_plot:
            try:
                self.out('MLWF_interpolated_bands', self.ctx.calc_mlwf.out.interpolated_bands)
                self.report('Interpolated bands pk: {}'.format(self.ctx.calc_mlwf.out.interpolated_bands.pk))
            except:
                self.report('WARNING: interpolated bands missing, while they should be there.')
        w90_state = self.ctx.calc_mlwf.get_state()
        if w90_state=='FAILED':
            self.report("Wannier90 calc pk: {} FAILED!".format(self.ctx.calc_mlwf.pk))
        else:
            self.report('Wannier90WorkChain successfully completed.')


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
        Return whether a pp W90 calc should be run or it is provided by input
        """
        return self.ctx.do_mlwf_pp
    def should_run_pw2wannier90(self):
        """
        Return whether a Pw2wannier90 calc should be run or it is provided by input
        """
        return self.ctx.do_pw2wannier90
    def should_run_projwfc(self):
        """
        Return whether a Projwfc calculation should be run or it is provided by input
        """
        return self.ctx.do_projwfc

@workfunction
def update_w90_params_zero_with_fermi(parameters,fermi):
    """
    Updated W90 windows with Fermi energy as zero.
    :param fermi:
    :return:
    """
    params = parameters.get_dict()
    var_list = []
    try:
        dwmax = params['dis_win_max']
        dwmax += fermi
        params['dis_win_max'] = dwmax
        var_list.append('dis_win_max')
    except KeyError:
        pass
    try:
        dfmax = params['dis_froz_max']
        dfmax += fermi
        params['dis_froz_max'] = dfmax
        var_list.append('dis_froz_max')
    except KeyError:
        pass
    try:
        dwmin = params['dis_win_min']
        dwmin += fermi
        params['dis_win_min'] = dwmin
        var_list.append('dis_win_min')
    except KeyError:
        pass
    try:
        dfmin = params['dis_froz_min']
        dfmin += fermi
        params['dis_froz_min'] = dfmin
        var_list.append('dis_froz_min')
    except KeyError:
        pass
    try:
        scdm_mu = params['scdm_mu']
        scdm_mu += fermi
        params['scdm_mu'] = scdm_mu
        var_list.append('scdm_mu')
    except KeyError:
        pass
    results = {'output_parameters':ParameterData(dict=params)}
    return results

@workfunction
def set_auto_numwann(parameters,projections):
    """
    Updated W90 params.
    :param
    :return:
    """
    params = parameters.get_dict()
    params['num_wann'] = len(projections.get_orbitals())
    return {'output_parameters': ParameterData(dict = params)}

@workfunction
def from_seekpath_to_wannier(seekpath_parameters):
    kinfo = {
        'path': seekpath_parameters.dict.path,
        'point_coords': seekpath_parameters.dict.point_coords,
    }
    results = {'kpoint_info':ParameterData(dict=kinfo)}
    return results

@workfunction
def set_mu_from_projections(bands,parameters,projections,thresholds):
    '''
    Setting mu parameter for the SCDM-k method:
    mu is such that the projectability on all the atomic orbitals
    contained in the pseudos is exactly equal to a a max_projectability
    values passed through the thresholds ParameterData.

    :param bands: output of projwfc, it was computed in the nscf calc
    :param parameters: wannier90 input params (the one to update with this wf)
    :param projections: output of projwfc
    :param thresholds: must contain a 'max_projectability' value (e.g. 0.95) to 
        evaluate an energy, and a 'sigma_factor_shift'; scdm_mu will be set to::
          
          scdm_mu = E(projectability==max_projectability) - sigma_factor_shift * scdm_sigma
        
        Note that you have to set the scdm_mu in the input parameters first! 
        Pass sigma_factor_shift = 0 if you do not want to shift
    :return: a modified ParameterData in output_parameters, with the proper value for scdm_mu set,
        and a Bool called 'success' that tells if the algorithm could find the energy at which
        the required projectability is achieved.
    '''
    from aiida.orm.data.base import Bool
    import numpy as np
    params = parameters.get_dict()
    #params['scdm_mu'] = len(projections.get_orbitals())
    # List of specifications of atomic orbitals in dictionary form
    dict_list = [i.get_orbital_dict() for i in projections.get_orbitals()]
    # Sum of the projections on all atomic orbitals (shape kpoints x nbands)
    out_array = sum([sum([x[1] for x in projections.get_projections(
        **get_dict)]) for get_dict in dict_list])
    #Flattening (projection modulus squared according to QE, energies)
    projwfc_flat, bands_flat = out_array.flatten(), bands.get_bands().flatten()
    #Sorted by energy
    sorted_bands, sorted_projwfc = zip(*sorted(zip(bands_flat, projwfc_flat)))
    Nk = len(bands.get_kpoints())
    #Cumulative sum and normalisation
    int_pdos = np.cumsum(sorted_projwfc)
    int_pdos = int_pdos/float(Nk)  # Normalizes over kpoints (not really needed)
    total_charge = np.max(int_pdos)
    int_pdos = int_pdos/float(total_charge)
    thr = thresholds.get_dict()['max_projectability']
    #Indices where projctability is larger than a threshold value
    indices_true = np.where(int_pdos>thr)[0]
    #Take the first energy eigenvalue (n,k) such that proj>thr
    if len(indices_true)>0:
        success = True
        params['scdm_mu'] = sorted_bands[indices_true[0]] - params['scdm_sigma'] * thresholds.get_dict()['sigma_factor_shift']
    else:
        success = False
    return {'output_parameters': ParameterData(dict = params),'success': Bool(success)}

@workfunction
def set_mu_and_sigma_from_projections(bands, parameters, projections, thresholds):
    '''
    Setting mu parameter for the SCDM-k method:
    mu is such that the projectability on all the atomic orbitals
    contained in the pseudos is exactly equal to a a max_projectability
    values passed through the thresholds ParameterData.

    :param bands: output of projwfc, it was computed in the nscf calc
    :param parameters: wannier90 input params (the one to update with this wf)
    :param projections: output of projwfc
    :param thresholds: must contain a 'max_projectability' value (e.g. 0.95) to
           evaluate an energy, and a 'sigma_factor_shift'; scdm_mu will be set to::
           scdm_mu = E(projectability==max_projectability) - sigma_factor_shift * scdm_sigma
           Note that you have to set the scdm_mu in the input parameters first!
           Pass sigma_factor_shift = 0 if you do not want to shift
    :return: a modified ParameterData in output_parameters, with the proper value for scdm_mu set,
             and a Bool called 'success' that tells if the algorithm could find the energy at which
             the required projectability is achieved.
    '''
    from aiida.orm.data.base import Bool
    import numpy as np

    def erfc_scdm(x,mu,sigma):
        from scipy.special import erfc
        return 0.5*erfc((x-mu)/sigma)

    def find_max(proj_list,max_value):
        f = lambda x : True if x<max_value else False
        bool_list = map(f,proj_list)
        for i,item in enumerate(bool_list):
            if item:
                break
        print i,proj_list[i]
    def fit_erfc(f,xdata,ydata):
        from scipy.optimize import curve_fit
        return curve_fit(f, xdata, ydata,bounds=([-50,0],[50,50]))


    params = parameters.get_dict()
    # List of specifications of atomic orbitals in dictionary form
    dict_list = [i.get_orbital_dict() for i in projections.get_orbitals()]
    # Sum of the projections on all atomic orbitals (shape kpoints x nbands)
    out_array = sum([sum([x[1] for x in projections.get_projections(
        **get_dict)]) for get_dict in dict_list])
    # Flattening (projection modulus squared according to QE, energies)
    projwfc_flat, bands_flat = out_array.flatten(), bands.get_bands().flatten()
    # Sorted by energy
    sorted_bands, sorted_projwfc = zip(*sorted(zip(bands_flat, projwfc_flat)))
    popt,pcov = fit_erfc(erfc_scdm,sorted_bands,sorted_projwfc)
    mu = popt[0]
    sigma = popt[1]
    # Temporary, TODO add check on interpolation
    success = True
    params['scdm_sigma'] = sigma
    params['scdm_mu'] = mu - sigma * thresholds.get_dict()['sigma_factor_shift']
    
    return {'output_parameters': ParameterData(dict=params), 'success': Bool(success)}
