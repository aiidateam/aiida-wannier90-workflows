from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import WorkChain, ToContext, if_
from aiida.plugins import WorkflowFactory
from aiida_quantumespresso.utils.protocols.pw import ProtocolManager
# from ..tools.pseudopotential import get_pseudos_from_dict
from .wannier import Wannier90WorkChain

import collections.abc
def deepupdate(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deepupdate(d.get(k, {}), v)
        else:
            d[k] = v
    return d

class Wannier90HamiltonianWorkChain(WorkChain):
    """
    A high level workchain which can automatically compute a Wannier tight_binding Hamiltonian for a given structure. Can also output Wannier Hamiltonian.
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
            'moments',
            valid_type=orm.List,
            required=False,
            help='The input magnetic moment.'
        )
        spec.input(
            'protocol',
            valid_type=orm.Dict,
            default=lambda: orm.Dict(dict={'name': 'theos-ht-1.0'}),
            help='The protocol to use for the workchain.',
            validator=validate_protocol
        )
        # spec.input(
        #     'controls.auto_projections',
        #     valid_type=orm.Bool,
        #     default=lambda: orm.Bool(True),
        #     help=
        #     'Whether using guessing or SCDM to automatically construct Wannier functions or not.'
        # )
        spec.input(
            'controls.projection_policy',
            valid_type=orm.Str,
            default=lambda: orm.Str('scdm'),
            help=
            'Specify the projection policy [scdm | manual | light | ...].'
        )
        spec.input(
            'controls.only_valence',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='For SCDM calculation. If True, calculate only valence band else valence and conduction band.'
        )
        spec.input(
            'controls.retrieve_hamiltonian',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help='If True, retrieve seedname_hr.dat.'
        )
        spec.input(
            'controls.retrieve_position',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help='If True, retrieve seedname_rmn.dat.'
        )
        spec.input(
            'controls.plot_wannier_functions',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help='If True, plot Wannier functions.'
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
            help='If True, calculate maximary localized Wannier function.'
        )
        spec.input(
            'controls.use_primitive_cell',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help='Find primitive cell before calculation (Default : True).'
        )
        spec.input(
            'controls.with_soi',
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help=
            'Specify the projection policy [scdm | manual | light | ...].'
        )
        # spec.input('controls.nbands_factor', valid_type=orm.Float, default=orm.Float(1.5),
        # help='The number of bands for the NSCF calculation is that used for the SCF multiplied by this factor.')
        spec.input(
            'parameters',
            valid_type=orm.Dict,
            required=False,
            default=lambda: orm.Dict(dict={}),
            help='Parameters for [scf, nscf, pw2wannier, wannier90, projwfc].'
        )
        # spec.expose_inputs(
        #     Wannier90WorkChain,
        #     include=['pseudo_family']
        # )
        #TODO Enable to input a dict of UpfData
        spec.input(
            'pseudo_family',
            valid_type=orm.Str,
            required=True,
            default=lambda: orm.Str('SSSP/1.1/PBE/efficiency'),
            help='Pseudo-family.'
        )
        spec.input(
            'options',
            valid_type=orm.Dict,
            required=False,
            default=lambda: orm.Dict(dict={}),
            help='metadata.options for [scf, nscf, pw2wannier, wannier90, projwfc].'
        )
        # spec.expose_inputs(
        #     Wannier90WorkChain,
        #     exclude=(
        #         "wannier90.wannier90.kpoint_path",
        #         "wannier90.wannier90.bands_kpoints",
        #     ),
        #     namespace_options={"required": False},
        # )


        spec.output(
            'computed_structure',
            valid_type=orm.StructureData,
            help=
            'The structure for which the calculations are computed.'
        )
        spec.output(
            'computed_moments',
            valid_type=orm.List,
            required=False,
            help=
            'The magnetic moments for which the calculations are computed.'
        )
        spec.output(
            'quantization_axis',
            valid_type=orm.List,
            required=False,
            help=
            'The quantization axis for which the calculations are computed.'
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

        spec.outline(
            cls.setup, 
            if_(cls.is_magnetic_structure)(
                cls.check_moments,
            ),
            cls.optimize_structure, 
            cls.setup_parameters,
            cls.run_wannier_workchain, 
            cls.results
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

        # Set magnetism mode
        self.ctx.current_moments = getattr(self.inputs, 'moments', None)
        if self.ctx.current_moments is not None:
            self.ctx.magnetism = True
        else:
            self.ctx.magnetism = False

        # try:
        #     controls.group_name
        # except AttributeError:
        #     self.ctx.group_name = self._DEFAULT_CONTROLS_GROUP_NAME
        # else:
        #     self.ctx.group_name = controls.group_name
    
    def is_magnetic_structure(self):
        """
        Whether the structure is magnetic or not.
        """
        return self.ctx.magnetism
    
    def check_moments(self):
        """
        Check if the moments are consistent with the structure.
        """
        self.report("Magnetism is considered. Checking the moments...")

        if len(self.ctx.current_moments) != len(self.inputs.structure.sites):
            raise ValueError("The number of moments is not equal to the number of atom sites.")

        # Whether the all moments are 3D vectors
        momlen=map(len,self.ctx.current_moments.get_list())
        for i, l in enumerate(momlen):
            if l != 3:
                raise ValueError(f"The {i}'s moment is not a 3D vector.")
        self.report("The moments are OK.")


    def optimize_structure(self):
        """Optimize input structure to primitive and normalized structure.
        """
        # if self.ctx.magnetism:
        #     raise ValueError("Magnetism is not supported yet.")
        #     # self.ctx.current_structure = self.inputs.structure
        #     # original_mat=self.ctx.current_structure.get_pymatgen().lattice.reciprocal_lattice.matrix
        #     # stand_prim_str_mat=result['primitive_structure'].get_pymatgen().lattice.reciprocal_lattice.matrix
        #     # self.ctx.transform_matrix = original_mat.T @ stand_prim_str_mat
        # else:
        #     self.ctx.current_structure = result['primitive_structure']
        #     self.ctx.transform_matrix = None
        # self.ctx.explicit_kpoints_path = result['explicit_kpoints']

        if self.inputs.controls.use_primitive_cell:
            # self.report("Using primitive structure.")
            self.report("Structure optimization not implemented yet --> naiively use input.")
            self.ctx.current_structure = self.inputs.structure
        else:
            self.ctx.current_structure = self.inputs.structure
        self.out('computed_structure', self.ctx.current_structure)

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
        # ecutwfc = []
        # ecutrho = []
        # for kind in self.ctx.current_structure.get_kind_names():
        #     try:
        #         dual = self.ctx.protocol['pseudo_data'][kind]['dual']
        #         cutoff = self.ctx.protocol['pseudo_data'][kind]['cutoff']
        #         cutrho = dual * cutoff
        #         ecutwfc.append(cutoff)
        #         ecutrho.append(cutrho)
        #     except KeyError:
        #         self.report(
        #             'failed to retrieve the cutoff or dual factor for {}'.
        #             format(kind)
        #         )
        #         return self.exit_codes.ERROR_INVALID_INPUT_UNRECOGNIZED_KIND

        family=orm.load_group(self.inputs.pseudo_family.value)
        pseudos=family.get_pseudos(structure=self.ctx.current_structure)
        self.ctx.pseudos=pseudos
        [ecutwfc,ecutrho]=family.get_recommended_cutoffs(structure=self.ctx.current_structure)


        number_of_atoms = len(self.ctx.current_structure.sites)
        pw_parameters = {
            'CONTROL': {
                'restart_mode': 'from_scratch',
                'tstress': self.ctx.protocol['tstress'],
                'tprnfor': self.ctx.protocol['tprnfor'],
            },
            'SYSTEM': {
                # 'ecutwfc': max(ecutwfc),
                # 'ecutrho': max(ecutrho),
                'ecutwfc': ecutwfc,
                'ecutrho': ecutrho,
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
        overwrite = self.inputs.parameters.get_dict()
        if 'scf' in overwrite.keys():
            deepupdate(pw_parameters,overwrite['scf'])
            
        self.ctx.scf_parameters = orm.Dict(dict=pw_parameters)

    def setup_projwfc_parameters(self):
        parameters={'PROJWFC': {'DeltaE': 0.2}}
        overwrite = self.inputs.parameters.get_dict()
        if 'projwfc' in overwrite.keys():
            deepupdate(parameters,overwrite['projwfc'])
        projwfc_parameters = orm.Dict(dict=parameters)
        self.ctx.projwfc_parameters = projwfc_parameters

    def setup_pw2wannier90_parameters(self):

        parameters = {}
        #Write UNK files (to plot WFs)
        overwrite = self.inputs.parameters.get_dict()
        if 'pw2wannier90' in overwrite.keys():
            deepupdate(parameters,overwrite['pw2wannier90'])

        if self.inputs.controls.plot_wannier_functions:
            parameters['write_unk'] = True
            self.report("UNK files will be written.")

        pw2wannier90_parameters = orm.Dict(dict={'inputpp': parameters})
        self.ctx.pw2wannier90_parameters = pw2wannier90_parameters
        return

    def setup_wannier90_parameters(self):
        parameters = {
            'use_ws_distance': True,
            # no need anymore since kmesh_tol is handled by Wannier90BaseWorkChain
            # 'kmesh_tol': 1e-8
        }
        
        overwrite = self.inputs.parameters.get_dict()
        if 'wannier90' in overwrite.keys():
            deepupdate(parameters,overwrite['wannier90'])

        policy=self.inputs.controls.projection_policy.value.lower()
        if policy == 'scdm':
            # Use SCDM. "auto_projection" means SCDM method
            parameters['auto_projections'] = True

        parameters['bands_plot'] = False
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
            # parameters['write_tb'] = True
            parameters['write_hr'] = True
            parameters['write_xyz'] = True

        wannier90_parameters = orm.Dict(dict=parameters)
        self.ctx.wannier90_parameters = wannier90_parameters
        return

    def get_pw_common_inputs(self):
        """Return the dictionary of inputs to be used as the basis for each `PwBaseWorkChain`."""
        # protocol, protocol_modifiers = self._get_protocol()
        # checked_pseudos = protocol.check_pseudos(
        #     modifier_name=protocol_modifiers.get('pseudo', None),
        #     pseudo_data=protocol_modifiers.get('pseudo_data', None)
        # )
        # known_pseudos = checked_pseudos['found']

        inputs = AttributeDict({
            'pw': {
                'code':
                self.inputs.code.pw,
                'pseudos': 
                self.ctx.pseudos,
                # get_pseudos_from_dict(
                #     self.ctx.current_structure, known_pseudos
                # ),
                'parameters':
                self.ctx.scf_parameters,
                'metadata': {},
            }
        })

        # if 'options' in self.inputs:
        #     inputs.pw.metadata.options = self.inputs.options.get_dict()
        # else:
        #     inputs.pw.metadata.options = self.get_default_options(
        #         with_mpi=True
        #     )

        return inputs

    def get_scf_inputs(self):
        """
        For the scf calculation, we use m-v cold smearing, thus the default nbnd 
        of QE is a bit higher than num_electrons/2, so we can get Fermi energy 
        from scf calculation.
        """
        scf_inputs = self.get_pw_common_inputs()

        opt = self.get_default_options(with_mpi=True)
        overwrite = self.inputs.options.get_dict()
        if 'scf' in overwrite.keys():
            deepupdate(opt,overwrite['scf'])
        scf_inputs.pw.metadata.options = opt

        return scf_inputs

    def get_nscf_inputs(self):
        """
        The nscf nbnd will be set in Wannier90WorkChain, if only_valence then 
        nbnd = num_electrons/2 and occupations = fixed (no smearing), 
        otherwise nbnd = num_electrons * 1.5 and m-v cold smearing.
        Here the num_electrons is obtained from scf output_parameters, 
        so we do not need to calculate num_electrons from pseudos.
        """
        # inputs = self.get_pw_common_inputs()
        # # inputs['pw']['parameters'] = self.ctx.nscf_parameters
        # return inputs
        nscf_inputs = self.get_pw_common_inputs()

        opt = self.get_default_options(with_mpi=True)
        overwrite = self.inputs.options.get_dict()
        if 'nscf' in overwrite.keys():
            deepupdate(opt,overwrite['nscf'])
        nscf_inputs.pw.metadata.options = opt

        return nscf_inputs

    def get_projwfc_inputs(self):
        inputs = AttributeDict({
            'code': self.inputs.code.projwfc,
            'parameters': self.ctx.projwfc_parameters,
            'metadata': {},
        })

        # if 'options' in self.inputs:
        #     inputs.metadata.options = self.inputs.options.get_dict()
        # else:
        #     inputs.metadata.options = self.get_default_options(with_mpi=True)
        #     # inputs.metadata.options = get_manual_options()

        opt = self.get_default_options(with_mpi=True)
        overwrite = self.inputs.options.get_dict()
        if 'projwfc' in overwrite.keys():
            deepupdate(opt,overwrite['projwfc'])
        inputs.metadata.options = opt

        return inputs
        # return inputs

    def get_pw2wannier90_inputs(self):
        inputs = AttributeDict({
            'code': self.inputs.code.pw2wannier90,
            'parameters': self.ctx.pw2wannier90_parameters,
            'metadata': {},
        })

        opt = self.get_default_options(with_mpi=True)
        overwrite = self.inputs.options.get_dict()
        if 'pw2wannier90' in overwrite.keys():
            deepupdate(opt,overwrite['pw2wannier90'])
        inputs.metadata.options = opt

        return inputs

    def get_wannier90_inputs(self):
        inputs = AttributeDict({
            'code':
            self.inputs.code.wannier90,
            'parameters':
            self.ctx.wannier90_parameters,
            'metadata': {}
            # 'kpoint_path':
            # orm.Dict(dict=self.ctx.kpoints_path)
        })

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

        # if 'options' in self.inputs:
        #     inputs.metadata.options = self.inputs.options.get_dict()
        # else:
        #     inputs.metadata.options = self.get_default_options(with_mpi=True)
        opt = self.get_default_options(with_mpi=True)
        overwrite = self.inputs.options.get_dict()
        if 'wannier90' in overwrite.keys():
            deepupdate(opt,overwrite['wannier90'])
        inputs.metadata.options = opt

        return inputs

    def run_wannier_workchain(self):
        """Run the `PwBandsWorkChain` to compute the band structure."""

        inputs = AttributeDict({
            'structure': self.ctx.current_structure,
            'only_valence': self.inputs.controls.only_valence,
            'projection_policy' : self.inputs.controls.projection_policy,
            'moments' : self.ctx.current_moments,
            'with_soi' : self.inputs.controls.with_soi,
            # 'relax': {
            #     'base': get_pw_common_inputs(),
            #     'relaxation_scheme': orm.Str('vc-relax'),
            #     'meta_convergence': orm.Bool(self.ctx.protocol['meta_convergence']),
            #     'volume_convergence': orm.Float(self.ctx.protocol['volume_convergence']),
            # },
            # 'pseudo_family' : self.inputs.pseudo_family,
            'scf': self.get_scf_inputs(),
            'nscf': self.get_nscf_inputs(),
            'projwfc': self.get_projwfc_inputs(),
            'pw2wannier90': self.get_pw2wannier90_inputs(),
            'wannier90': self.get_wannier90_inputs(),
        })
        # inputs.update(self.exposed_inputs(Wannier90WorkChain))

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

        return ToContext(workchain_ham=running)

    def results(self):
        """Attach the relevant output nodes from the band calculation to the workchain outputs for convenience."""

        workchain = self.ctx.workchain_ham

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
