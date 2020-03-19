from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext
from aiida.plugins import WorkflowFactory

from aiida_quantumespresso.workflows.functions.seekpath_structure_analysis import seekpath_structure_analysis
from aiida_quantumespresso.utils.protocols.pw import ProtocolManager
from aiida_quantumespresso.utils.pseudopotential import get_pseudos_from_dict
from .wannier import Wannier90WorkChain


class Wannier90BandsWorkChain(WorkChain):
    """
    A high level workchain which can automatically compute a Wannier band structure for a given structure. Can also output Wannier Hamiltonian.
    """
    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)
        # spec.input('vdw_table', valid_type=orm.SinglefileData, required=False)
        spec.input(
            'code.pw',
            valid_type=orm.Code,
            help='The `pw.x` code to use for the `PwCalculations`.'
        )
        spec.input(
            'code.pw2wannier90',
            valid_type=orm.Code,
            help=
            'The `pw2wannier90.x` code to use for the `Pw2WannierCalculations`.'
        )
        spec.input(
            'code.wannier90',
            valid_type=orm.Code,
            help='The `wannier90.x` code to use for the `PwCalculations`.'
        )
        spec.input(
            'code.projwfc',
            valid_type=orm.Code,
            required=False,
            help='The `projwfc.x` code to use for the `PwCalculations`.'
        )
        spec.input(
            'structure',
            valid_type=orm.StructureData,
            help='The input structure.'
        )
        spec.input(
            'protocol',
            valid_type=orm.Dict,
            default=lambda: orm.Dict(dict={'name': 'theos-ht-1.0'}),
            help='The protocol to use for the workchain.',
            validator=validate_protocol
        )
        spec.input(
            'controls.auto_projections',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help=
            'Whether using SCDM to automatically construct Wannier functions or not.'
        )
        spec.input(
            'controls.only_valence',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='Group name that the calculations will be added to.'
        )
        spec.input(
            'controls.retrieve_hamiltonian',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='Group name that the calculations will be added to.'
        )
        spec.input(
            'controls.plot_wannier_functions',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='Group name that the calculations will be added to.'
        )
        spec.input(
            'controls.do_disentanglement',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help=
            'Used only if only_valence == False. Usually disentanglement worsens SCDM bands, keep it default to False.'
        )
        spec.input(
            'controls.do_mlwf',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help='Group name that the calculations will be added to.'
        )
        spec.input(
            'controls.kpoints_distance_for_bands',
            valid_type=orm.Float,
            default=lambda: orm.Float(0.01),
            help='Kpoint mesh density of the resulting band structure.'
        )
        # spec.input('controls.nbands_factor', valid_type=orm.Float, default=orm.Float(1.5),
        # help='The number of bands for the NSCF calculation is that used for the SCF multiplied by this factor.')

        spec.output(
            'primitive_structure',
            valid_type=orm.StructureData,
            help=
            'The normalized and primitivized structure for which the calculations are computed.'
        )
        spec.output(
            'seekpath_parameters',
            valid_type=orm.Dict,
            help=
            'The parameters used in the SeeKpath call to normalize the input or relaxed structure.'
        )
        spec.output(
            'scf_parameters',
            valid_type=orm.Dict,
            help='The output parameters of the SCF `PwBaseWorkChain`.'
        )
        spec.output(
            'nscf_parameters',
            valid_type=orm.Dict,
            help='The output parameters of the NSCF `PwBaseWorkChain`.'
        )
        spec.output(
            'projwfc_bands',
            valid_type=orm.BandsData,
            required=False,
            help='The output bands of projwfc run.'
        )
        spec.output(
            'projwfc_projections',
            valid_type=orm.ProjectionData,
            required=False,
            help='The output projections of projwfc run.'
        )
        spec.output(
            'pw2wannier90_remote_folder',
            valid_type=orm.RemoteData,
            required=False
        )
        spec.output('wannier90_parameters', valid_type=orm.Dict)
        spec.output('wannier90_retrieved', valid_type=orm.FolderData)
        spec.output(
            'wannier90_remote_folder',
            valid_type=orm.RemoteData,
            required=False
        )
        spec.output(
            'wannier90_interpolated_bands',
            valid_type=orm.BandsData,
            required=False,
            help='The computed band structure.'
        )

        spec.outline(
            cls.setup, cls.run_seekpath, cls.setup_parameters,
            cls.run_wannier_workchain, cls.results
        )

        spec.exit_code(
            201,
            'ERROR_INVALID_INPUT_UNRECOGNIZED_KIND',
            message='Input `StructureData` contains an unsupported kind.'
        )
        spec.exit_code(
            401,
            'ERROR_SUB_PROCESS_FAILED_BANDS',
            message='The `PwBandsWorkChain` sub process failed.'
        )

    def _get_protocol(self):
        """Return a `ProtocolManager` instance and a dictionary of modifiers."""
        protocol_data = self.inputs.protocol.get_dict()
        protocol_name = protocol_data['name']
        protocol = ProtocolManager(protocol_name)

        protocol_modifiers = protocol_data.get('modifiers', {})

        return protocol, protocol_modifiers

    def setup_protocol(self):
        """Set up context variables and inputs for the `PwBandsWorkChain`.

        Based on the specified protocol, we define values for variables that affect the execution of the calculations.
        """
        protocol, protocol_modifiers = self._get_protocol()
        self.report(
            'running the workchain with the "{}" protocol'.format(
                protocol.name
            )
        )
        self.ctx.protocol = protocol.get_protocol_data(
            modifiers=protocol_modifiers
        )
        # self.report('protocol: ' + str(self.ctx.protocol))

    def setup(self):

        self.setup_protocol()
        try:
            controls = self.inputs.controls
            if controls.only_valence:
                valence_info = "valence bands only"
            else:
                valence_info = "valence + conduction bands"
            self.report('workchain controls found in inputs: ' + valence_info)
        except AttributeError:
            controls = {}
        controls = AttributeDict(controls)
        # try:
        #     controls.group_name
        # except AttributeError:
        #     self.ctx.group_name = self._DEFAULT_CONTROLS_GROUP_NAME
        # else:
        #     self.ctx.group_name = controls.group_name

    def run_seekpath(self):
        """Run the structure through SeeKpath to get the primitive and normalized structure.
        """
        kpoints_distance_for_bands = self.inputs.controls.get(
            'kpoints_distance_for_bands',
            self.ctx.protocol['kpoints_distance_for_bands']
        )
        seekpath_parameters = orm.Dict(
            dict={'reference_distance': kpoints_distance_for_bands}
        )
        structure_formula = self.inputs.structure.get_formula()
        self.report(
            'running seekpath to get primitive structure for: {}'.
            format(structure_formula)
        )
        result = seekpath_structure_analysis(
            self.inputs.structure, seekpath_parameters
        )
        self.ctx.current_structure = result['primitive_structure']
        self.ctx.explicit_kpoints_path = result['explicit_kpoints']
        # save kpoint_path for bands_plot
        self.ctx.kpoints_path = {
            'path': result['parameters']['path'],
            'point_coords': result['parameters']['point_coords']
        }

        self.out('primitive_structure', result['primitive_structure'])
        self.out('seekpath_parameters', result['parameters'])

    def setup_parameters(self):
        """setup input parameters of each calculations, 
        since there are some dependencies between input parameters, 
        we store them in context variables.
        """
        self.setup_scf_parameters()
        # self.setup_nscf_parameters()
        self.setup_projwfc_parameters()
        self.setup_pw2wannier90_parameters()
        self.setup_wannier90_parameters()
        # self.report("scf:" + str(self.ctx.scf_parameters.get_dict()))
        # self.report("nscf:" + str(self.ctx.nscf_parameters.get_dict()))
        # self.report("projwfc:" + str(self.ctx.projwfc_parameters.get_dict()))
        # self.report("pw2wannier90:" + str(self.ctx.pw2wannier90_parameters.get_dict()))
        # self.report("wannier90:" + str(self.ctx.wannier90_parameters.get_dict()))
        self.report(
            'number of machines {} auto-set according to number of atoms'.
            format(estimate_num_machines(self.ctx.current_structure))
        )

    def setup_scf_parameters(self):
        """Set up the default input parameters required for the `PwBandsWorkChain`, and store it in self.ctx"""
        ecutwfc = []
        ecutrho = []

        for kind in self.ctx.current_structure.get_kind_names():
            try:
                dual = self.ctx.protocol['pseudo_data'][kind]['dual']
                cutoff = self.ctx.protocol['pseudo_data'][kind]['cutoff']
                cutrho = dual * cutoff
                ecutwfc.append(cutoff)
                ecutrho.append(cutrho)
            except KeyError:
                self.report(
                    'failed to retrieve the cutoff or dual factor for {}'.
                    format(kind)
                )
                return self.exit_codes.ERROR_INVALID_INPUT_UNRECOGNIZED_KIND

        number_of_atoms = len(self.ctx.current_structure.sites)
        pw_parameters = {
            'CONTROL': {
                'restart_mode': 'from_scratch',
                'tstress': self.ctx.protocol['tstress'],
                'tprnfor': self.ctx.protocol['tprnfor'],
            },
            'SYSTEM': {
                'ecutwfc': max(ecutwfc),
                'ecutrho': max(ecutrho),
                'smearing': self.ctx.protocol['smearing'],
                'degauss': self.ctx.protocol['degauss'],
                'occupations': self.ctx.protocol['occupations'],
            },
            'ELECTRONS': {
                'conv_thr':
                self.ctx.protocol['convergence_threshold_per_atom'] *
                number_of_atoms,
            }
        }

        self.ctx.scf_parameters = orm.Dict(dict=pw_parameters)

    # we do not need this anymore, since now Wannier90WorkChain accepts input only_valence,
    # it will set nbnd & occupations in itself.
    # def setup_nscf_parameters(self):
    #     """almost identical to scf_parameters, but need set nbnd

    #     :return: [description]
    #     :rtype: [type]
    #     """
    #     def get_z_valence_from_upf(upf_content):
    #         """a fragile parser for upf file, to get z_valence

    #         :param upf_content: the content of a upf file stored in a string
    #         :type upf_content: str
    #         :return: z_valence of this upf file
    #         :rtype: float
    #         """
    #         for l in upf_content.split('\n'):
    #             if 'Z valence' in l:
    #                 #e.g. 11.00000000000      Z valence
    #                 z = float(l.strip().split()[0])
    #                 break
    #             elif 'z_valence' in l:
    #                 #e.g. z_valence="3.000000000000000E+000"
    #                 z = float(l.strip().split("=")[1][1:-1])
    #                 break
    #         try:
    #             z
    #         except NameError:
    #             raise KeyError('z_valence not found!')
    #         return z

    #     protocol, protocol_modifiers = self._get_protocol()
    #     checked_pseudos = protocol.check_pseudos(
    #         modifier_name=protocol_modifiers.get('pseudo', None),
    #         pseudo_data=protocol_modifiers.get('pseudo_data', None))
    #     known_pseudos = checked_pseudos['found']
    #     pseudos = get_pseudos_from_dict(self.ctx.current_structure, known_pseudos)

    #     # we try to parse the upf file to generate z_valence
    #     # maybe use nbands_factor (aiida_qe/workflows/pw/bands.py) in the future
    #     number_of_electrons = 0
    #     composition = self.ctx.current_structure.get_composition()
    #     for kind in self.ctx.current_structure.get_kind_names():
    #         try:
    #             upf_name = pseudos[kind].list_object_names()[0]
    #             upf_content = pseudos[kind].get_object_content(upf_name)
    #             z_valence = get_z_valence_from_upf(upf_content)
    #             # self.report("found z_valence of " + kind +": " + str(z))
    #             number_of_electrons += z_valence*composition[kind]
    #         except KeyError:
    #             self.report('failed to retrieve the z_valence for {}'.format(kind))
    #             return self.exit_codes.ERROR_INVALID_INPUT_UNRECOGNIZED_KIND
    #     if self.inputs.controls.only_valence:
    #         nbnd = int(number_of_electrons / 2)
    #     else:
    #         #Three times the number of occupied bands
    #         nbnd = int(number_of_electrons * 1.5)
    #     self.report('nscf nbnd calculated from pseudos: ' + str(nbnd))

    #     # we need deepcopy for 'nbnd', otherwise scf_parameters will change as well
    #     from copy import deepcopy
    #     nscf_parameters = deepcopy(self.ctx.scf_parameters.get_dict())
    #     nscf_parameters['SYSTEM']['nbnd'] = nbnd

    #     if self.inputs.controls.only_valence:
    #         nscf_parameters['SYSTEM']['occupations'] = 'fixed'
    #         nscf_parameters['SYSTEM'].pop('smearing')
    #         nscf_parameters['SYSTEM'].pop('degauss')

    #     nscf_parameters = orm.Dict(dict=nscf_parameters)
    #     self.ctx.nscf_parameters = nscf_parameters

    def setup_projwfc_parameters(self):

        projwfc_parameters = orm.Dict(dict={'PROJWFC': {'DeltaE': 0.2}})
        self.ctx.projwfc_parameters = projwfc_parameters

    def setup_pw2wannier90_parameters(self):

        parameters = {}
        #Write UNK files (to plot WFs)
        if self.inputs.controls.plot_wannier_functions:
            parameters['write_unk'] = True
            self.report("UNK files will be written.")

        pw2wannier90_parameters = orm.Dict(dict={'inputpp': parameters})
        self.ctx.pw2wannier90_parameters = pw2wannier90_parameters

    def setup_wannier90_parameters(self):
        parameters = {
            'use_ws_distance': True,
            # no need anymore since kmesh_tol is handled by Wannier90BaseWorkChain
            # 'kmesh_tol': 1e-8
        }

        if self.inputs.controls.auto_projections:
            parameters['auto_projections'] = True

        parameters['bands_plot'] = True
        # parameters['num_bands'] = self.ctx.nscf_parameters['SYSTEM']['nbnd']

        if self.inputs.controls.plot_wannier_functions:
            parameters['wannier_plot'] = True
            # 'wannier_plot_list':[1]

        number_of_atoms = len(self.ctx.current_structure.sites)
        if self.inputs.controls.do_mlwf:
            parameters.update({
                'num_iter': 400,
                'conv_tol': 1e-7 * number_of_atoms,
                'conv_window': 3,
            })
        else:
            parameters.update({'num_iter': 0})

        # TODO exclude_bands
        # if exclude_bands is not None:
        #     wannier90_params_dict['exclude_bands'] = exclude_bands
        #'exclude_bands': range(5,13),

        if self.inputs.controls.only_valence:
            parameters['dis_num_iter'] = 0
        else:
            if self.inputs.controls.do_disentanglement:
                parameters.update({
                    'dis_num_iter': 200,
                    'dis_conv_tol': parameters['conv_tol'],
                    'dis_froz_max': 1.0,  #TODO a better value??
                    #'dis_mix_ratio':1.d0,
                    #'dis_win_max':10.0,
                })
            else:
                parameters.update({'dis_num_iter': 0})

        if self.inputs.controls.retrieve_hamiltonian:
            parameters['write_tb'] = True
            parameters['write_hr'] = True
            parameters['write_xyz'] = True

        # try:
        #     self.ctx.random_projections = control_dict['random_projections']
        #     if self.ctx.random_projections:
        #         self.report("projections override: set to random from input.")
        # except KeyError:
        #     self.ctx.random_projections = False


#        We expect either a KpointsData with given mesh or a desired distance between k-points
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
        wannier90_parameters = orm.Dict(dict=parameters)
        self.ctx.wannier90_parameters = wannier90_parameters

    def get_pw_common_inputs(self):
        """Return the dictionary of inputs to be used as the basis for each `PwBaseWorkChain`."""
        protocol, protocol_modifiers = self._get_protocol()
        checked_pseudos = protocol.check_pseudos(
            modifier_name=protocol_modifiers.get('pseudo', None),
            pseudo_data=protocol_modifiers.get('pseudo_data', None)
        )
        known_pseudos = checked_pseudos['found']

        inputs = AttributeDict({
            'pw': {
                'code':
                self.inputs.code.pw,
                'pseudos':
                get_pseudos_from_dict(
                    self.ctx.current_structure, known_pseudos
                ),
                'parameters':
                self.ctx.scf_parameters,
                'metadata': {},
            }
        })

        if 'options' in self.inputs:
            inputs.pw.metadata.options = self.inputs.options.get_dict()
        else:
            inputs.pw.metadata.options = self.get_default_options(
                with_mpi=True
            )

        return inputs

    def get_scf_inputs(self):
        """
        For the scf calculation, we use m-v cold smearing, thus the default nbnd 
        of QE is a bit higher than num_electrons/2, so we can get Fermi energy 
        from scf calculation.
        """
        return self.get_pw_common_inputs()

    def get_nscf_inputs(self):
        """
        The nscf nbnd will be set in Wannier90WorkChain, if only_valence then 
        nbnd = num_electrons/2 and occupations = fixed (no smearing), 
        otherwise nbnd = num_electrons * 1.5 and m-v cold smearing.
        Here the num_electrons is obtained from scf output_parameters, 
        so we do not need to calculate num_electrons from pseudos.
        """
        inputs = self.get_pw_common_inputs()
        # inputs['pw']['parameters'] = self.ctx.nscf_parameters
        return inputs

    def get_projwfc_inputs(self):
        inputs = AttributeDict({
            'code': self.inputs.code.projwfc,
            'parameters': self.ctx.projwfc_parameters,
            'metadata': {},
        })

        if 'options' in self.inputs:
            inputs.metadata.options = self.inputs.options.get_dict()
        else:
            inputs.metadata.options = self.get_default_options(with_mpi=True)
            # inputs.metadata.options = get_manual_options()

        return inputs

    def get_pw2wannier90_inputs(self):
        inputs = AttributeDict({
            'code': self.inputs.code.pw2wannier90,
            'parameters': self.ctx.pw2wannier90_parameters,
            'metadata': {},
        })

        # TODO max_projectability, sigma_factor_shift
        # try:
        #     self.ctx.max_projectability = control_dict['max_projectability']
        #     self.report("Max projectability set to {}.".format(
        #         self.ctx.max_projectability))
        # except KeyError:
        #     self.ctx.max_projectability = 0.95
        #     self.report("Max projectability set to {} (DEFAULT).".format(
        #         self.ctx.max_projectability))

        # try:
        #     self.ctx.sigma_factor_shift = control_dict['sigma_factor_shift']
        #     self.report("Sigma factor shift set to {}.".format(
        #         self.ctx.sigma_factor_shift))
        # except KeyError:
        #     self.ctx.sigma_factor_shift = 3.
        #     self.report("Sigma factor shift set to {} (DEFAULT).".format(
        #         self.ctx.sigma_factor_shift))

        # dict={
        #     #'max_projectability': self.ctx.max_projectability,
        #     'sigma_factor_shift': self.ctx.sigma_factor_shift,
        # }

        if 'options' in self.inputs:
            inputs.metadata.options = self.inputs.options.get_dict()
        else:
            inputs.metadata.options = self.get_default_options(with_mpi=True)

        return inputs

    def get_wannier90_inputs(self):
        inputs = AttributeDict({
            'code':
            self.inputs.code.wannier90,
            'parameters':
            self.ctx.wannier90_parameters,
            'metadata': {},
            'kpoint_path':
            orm.Dict(dict=self.ctx.kpoints_path)
        })

        #TODO ramdom_projections
        # if self.ctx.random_projections:
        #     if settings is not None:
        #         settings['random_projections'] = True
        #     else:
        #         settings = {'random_projections': True}

        if self.inputs.controls.retrieve_hamiltonian:
            settings = {}
            # settings['retrieve_hoppings'] = True
            # tbmodels needs aiida.win file
            from aiida.plugins import CalculationFactory
            Wannier90Calculation = CalculationFactory('wannier90.wannier90')
            settings['additional_retrieve_list'] = [
                Wannier90Calculation._DEFAULT_INPUT_FILE
            ]
            # also retrieve .chk file in case we need it later
            # seedname = Wannier90Calculation._DEFAULT_INPUT_FILE.split('.')[0]
            # settings['additional_retrieve_list'] += [
            #     '{}.{}'.format(seedname, ext)
            #     for ext in ['chk', 'eig', 'amn', 'mmn', 'spn']
            # ]
            inputs['settings'] = orm.Dict(dict=settings)

        # try:
        #     self.ctx.options.energies_relative_to_fermi
        # except KeyError:
        #     self.report("W90 windows defined with the Fermi level as zero.")

        if 'options' in self.inputs:
            inputs.metadata.options = self.inputs.options.get_dict()
        else:
            inputs.metadata.options = self.get_default_options(with_mpi=True)

        return inputs

    def run_wannier_workchain(self):
        """Run the `PwBandsWorkChain` to compute the band structure."""

        inputs = AttributeDict({
            'structure': self.ctx.current_structure,
            'only_valence': self.inputs.controls.only_valence,
            # 'relax': {
            #     'base': get_pw_common_inputs(),
            #     'relaxation_scheme': orm.Str('vc-relax'),
            #     'meta_convergence': orm.Bool(self.ctx.protocol['meta_convergence']),
            #     'volume_convergence': orm.Float(self.ctx.protocol['volume_convergence']),
            # },
            'scf': self.get_scf_inputs(),
            # 'bands': get_pw_common_inputs(),
            'nscf': self.get_nscf_inputs(),
            'projwfc': self.get_projwfc_inputs(),
            'pw2wannier90': self.get_pw2wannier90_inputs(),
            'wannier90': self.get_wannier90_inputs(),
        })

        # inputs.relax.base.kpoints_distance = orm.Float(self.ctx.protocol['kpoints_mesh_density'])
        inputs.scf.kpoints_distance = orm.Float(
            self.ctx.protocol['kpoints_mesh_density']
        )
        # inputs.bands.kpoints_distance = orm.Float(self.ctx.protocol['kpoints_distance_for_bands'])
        inputs.nscf.kpoints_distance = orm.Float(
            self.ctx.protocol['kpoints_mesh_density']
        )

        # num_bands_factor = self.ctx.protocol.get('num_bands_factor', None)
        # if num_bands_factor is not None:
        #     inputs.nbands_factor = orm.Float(num_bands_factor)

        running = self.submit(Wannier90WorkChain, **inputs)

        self.report('launching Wannier90WorkChain<{}>'.format(running.pk))

        return ToContext(workchain_bands=running)

    def results(self):
        """Attach the relevant output nodes from the band calculation to the workchain outputs for convenience."""

        workchain = self.ctx.workchain_bands

        if not workchain.is_finished_ok:
            self.report(
                'sub process Wannier90WorkChain<{}> failed'.format(
                    workchain.pk
                )
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

        self.out('scf_parameters', workchain.outputs.scf__output_parameters)
        self.out('nscf_parameters', workchain.outputs.nscf__output_parameters)
        if 'projwfc__bands' in workchain.outputs:
            self.out('projwfc_bands', workchain.outputs.projwfc__bands)
            self.out(
                'projwfc_projections', workchain.outputs.projwfc__projections
            )
        self.out(
            'pw2wannier90_remote_folder',
            workchain.outputs.pw2wannier90__remote_folder
        )
        self.out(
            'wannier90_parameters',
            workchain.outputs.wannier90__output_parameters
        )
        self.out('wannier90_retrieved', workchain.outputs.wannier90__retrieved)
        self.out(
            'wannier90_remote_folder',
            workchain.outputs.wannier90__remote_folder
        )
        if 'wannier90__interpolated_bands' in workchain.outputs:
            self.out(
                'wannier90_interpolated_bands',
                workchain.outputs.wannier90__interpolated_bands
            )
            self.report(
                'wannier90 interpolated bands pk: {}'.format(
                    workchain.outputs.wannier90__interpolated_bands.pk
                )
            )

        self.report('Wannier90BandsWorkChain successfully completed')

    def get_default_options(self, with_mpi=False):
        """Increase wallclock to 5 hour, use mpi, set number of machines according to 
        number of atoms.
        
        :param with_mpi: [description], defaults to False
        :type with_mpi: bool, optional
        :return: [description]
        :rtype: [type]
        """
        from aiida_quantumespresso.utils.resources import get_default_options as get_opt
        num_machines = estimate_num_machines(self.ctx.current_structure)
        opt = get_opt(
            max_num_machines=num_machines,
            max_wallclock_seconds=3600 * 5,
            with_mpi=with_mpi
        )
        return opt


def estimate_num_machines(structure):
    """
     1 <= num_atoms <= 10 -> 1 machine
    11 <= num_atoms <= 20 -> 2 machine
    
    :param structure: [description]
    :type structure: [type]
    :return: [description]
    :rtype: [type]
    """
    from math import ceil
    number_of_atoms = len(structure.sites)
    return ceil(number_of_atoms / 10)


def validate_protocol(protocol_dict, ctx):
    """Check that the protocol is one for which we have a definition."""
    try:
        protocol_name = protocol_dict['name']
    except KeyError as exception:
        return 'Missing key `name` in protocol dictionary'
    try:
        ProtocolManager(protocol_name)
    except ValueError as exception:
        return str(exception)


def get_manual_options():
    # QE projwfc.x complains
    #         Calling projwave ....
    #     linear algebra parallelized on  16 procs

    # Problem Sizes
    # natomwfc =           14
    # nx       =            4
    # nbnd     =           24
    # nkstot   =          216
    # npwx     =           83
    # nkb      =           20

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #     Error in routine  blk2cyc_zredist (1):

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #     Error in routine  blk2cyc_zredist (1):
    #     nb less than the number of proc

    # https://aiida-core.readthedocs.io/en/v1.1.1/scheduler/index.html#nodenumberjobresource
    return {
        'resources': {
            'num_machines': 1,
            'num_mpiprocs_per_machine': 8,
            # 'tot_num_mpiprocs': ,
            # 'num_cores_per_machine': ,
            # 'num_cores_per_mpiproc': ,
        },
        'max_wallclock_seconds': 3600 * 5,
        'withmpi': True,
    }
