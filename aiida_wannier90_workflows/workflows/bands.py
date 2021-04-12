
from copy import deepcopy
from aiida import orm
from aiida.common import AttributeDict, LinkType
from aiida.engine import WorkChain, ToContext, if_

from aiida_quantumespresso.calculations.functions.seekpath_structure_analysis import seekpath_structure_analysis
from aiida_quantumespresso.workflows.pw.band_structure import validate_protocol
from aiida_quantumespresso.utils.protocols.pw import ProtocolManager
from aiida_quantumespresso.utils.pseudopotential import get_pseudos_from_dict
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.calculations.functions.create_kpoints_from_distance import create_kpoints_from_distance

from aiida_wannier90_workflows.workflows.wannier import Wannier90WorkChain
from aiida_wannier90_workflows.utils.upf import get_number_of_electrons, get_number_of_projections, get_wannier_number_of_bands, _load_pseudo_metadata, get_projections
from aiida_wannier90_workflows.calculations.functions.kmesh import convert_kpoints_mesh_to_list

__all__ = ['Wannier90BandsWorkChain']

class Wannier90BandsWorkChain(WorkChain):
    """
    A high level workchain which can automatically compute a Wannier band structure for a given structure. Can also output Wannier Hamiltonian.
    """
    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        spec.input_namespace('codes', required=True, valid_type=orm.Code)
        spec.input('codes.pw', valid_type=orm.Code, help='The `pw.x` code for the `PwCalculations`.')
        spec.input('codes.pw2wannier90', valid_type=orm.Code, help='The `pw2wannier90.x` code for the `Pw2WannierCalculations`.')
        spec.input('codes.wannier90', valid_type=orm.Code, help='The `wannier90.x` code for the `PwCalculations`.')
        spec.input('codes.projwfc', valid_type=orm.Code, required=False, help='Optional `projwfc.x` code for the `PwCalculations`.')
        spec.input('codes.opengrid', valid_type=orm.Code, required=False, help='Optional `open_grid.x` code for the `OpengridCalculations`.')

        spec.input('structure', valid_type=orm.StructureData, help='The input structure.')
        spec.input('protocol', valid_type=orm.Dict, default=lambda: orm.Dict(dict={'name': 'theos-ht-1.0'}), help='The protocol to use for the workchain.', validator=validate_protocol)
        spec.input('options', valid_type=orm.Dict, required=False, help='Optional `options` to use for the workchain.')

        # control variables for the workchain
        spec.input('scdm_projections', valid_type=orm.Bool, default=lambda: orm.Bool(True),
            help='If True use SCDM projections.')
        spec.input('spdf_projections', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If True use atom-centred s,p,d orbitals as projections.')
        spec.input('pswfc_projections', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If True use PP_PSWFC projections.')
        spec.input('use_opengrid', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If True use open_grid.x to accelerate calculations.')
        spec.input('opengrid_only_scf', valid_type=orm.Bool, default=lambda: orm.Bool(True),
            help='If True only one scf calculation will be performed in the OpengridWorkChain.')
        spec.input('only_valence', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If True only Wannierise valence bands.')
        spec.input('exclude_semicore', valid_type=orm.Bool, default=lambda: orm.Bool(True),
            help='If True do not Wannierise semicore states.')
        spec.input('compare_dft_bands', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If True perform another DFT band structure calculation for comparing Wannier interpolated bands with DFT bands.')
        # TODO if no user input for disentanglement, we should set its default value according to 
        # scdm_projections; if user has specified disentanglement, we should not modify its value.
        # I need a populate_defaults=False mechanism for this, but there is no one exist.
        # Maybe later when I switched to InputGenerator mechansim, no need to play with this one.
        spec.input('disentanglement', valid_type=orm.Bool, required=False,
            help='Used only if only_valence == False. The default disentanglement depends on scdm_projections: when scdm_projections = True, disentanglement = False; when scdm_projections = False, disentanglement = True. These improve the quality of Wannier interpolated bands for the two cases.')
        spec.input('auto_froz_max', valid_type=orm.Bool, required=False,
            help='Used only if pswfc_projections == True. If True use the energy corresponding to projectability = 0.9 as dis_froz_max for wannier90 disentanglement.')
        spec.input('maximal_localisation', valid_type=orm.Bool, default=lambda: orm.Bool(True),
            help='If true do maximal localisation of Wannier functions.')
        spec.input('spin_polarized', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If True perform magnetic calculations.')
        spec.input('spin_orbit_coupling', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If True perfrom spin-orbit-coupling calculations.')
        spec.input('retrieve_hamiltonian', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If True retrieve Wannier Hamiltonian.')
        spec.input('plot_wannier_functions', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If True plot Wannier functions as xsf files.')
        spec.input('bands_kpoints_distance', valid_type=orm.Float, required=False,
            help='Kpoint mesh density of the resulting band structure.')

        spec.output('primitive_structure', valid_type=orm.StructureData,
            help='The normalized and primitivized structure for which the calculations are computed.')
        spec.output('seekpath_parameters', valid_type=orm.Dict,
            help='The parameters used in the SeeKpath call to normalize the input or relaxed structure.')
        spec.output('scf_parameters', valid_type=orm.Dict, help='The output parameters of the scf `PwBaseWorkChain`.')
        spec.output('nscf_parameters', valid_type=orm.Dict, required=False,
            help='The output parameters of the nscf `PwBaseWorkChain`.')
        spec.output('projwfc_bands', valid_type=orm.BandsData, required=False,
            help='The output bands of projwfc run.')
        spec.output('projwfc_projections', valid_type=orm.ProjectionData, required=False,
            help='The output projections of projwfc run.')
        spec.output('scdm_projectability', valid_type=orm.Dict, required=False,
            help='SCDM mu sigma projectability plot')
        spec.output('pw2wannier90_remote_folder', valid_type=orm.RemoteData, required=False)
        spec.output('wannier90_parameters', valid_type=orm.Dict)
        spec.output('wannier90_retrieved', valid_type=orm.FolderData)
        spec.output('wannier90_remote_folder', valid_type=orm.RemoteData, required=False)
        spec.output('wannier90_interpolated_bands', valid_type=orm.BandsData, required=False,
            help='The Wannier interpolated band structure.')
        spec.output('bands_parameters', valid_type=orm.Dict, required=False,
            help='The output parameters of DFT bands calculation.')
        spec.output('dft_bands', valid_type=orm.BandsData, required=False,
            help='The computed DFT band structure.')

        spec.outline(
            cls.setup,
            cls.run_seekpath,
            cls.setup_parameters,
            cls.run_wannier_workchain,
            cls.inspect_wannier_workchain,
            if_(cls.should_run_bands)(
                cls.run_bands,
                cls.inspect_bands),
            cls.results
        )

        spec.exit_code(401, 'ERROR_INVALID_INPUT_UNRECOGNIZED_KIND',
            message='Input `StructureData` contains an unsupported kind.')
        spec.exit_code(402, 'ERROR_INVALID_INPUT_OPENGRID',
            message='No open_grid.x Code provided.')
        spec.exit_code(403, 'ERROR_INVALID_INPUT_PSEUDOPOTENTIAL',
            message='Invalid pseudopotentials.')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_WANNIER',
            message='The `Wannier90WorkChain` sub process failed.')
        spec.exit_code(405, 'ERROR_SUB_PROCESS_FAILED_BANDS',
            message='The bands PwBasexWorkChain sub process failed')

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
        self.report('running the workchain with the "{}" protocol'.format(protocol.name))

        # For SOC case, if no user input, replace SSSP by pslibrary/pseudo-dojo, which has SOC pseudos.
        if protocol_modifiers == {} and self.inputs.spin_orbit_coupling:
            # pseudo_data = _load_pseudo_metadata('pslibrary_paw_relpbe_1.0.0.json')
            # switch to pseudo-dojo
            pseudo_data = _load_pseudo_metadata('dojo_nc_fr.json')
            protocol_modifiers = {'pseudo': 'custom', 'pseudo_data': pseudo_data}

        self.ctx.protocol = protocol.get_protocol_data(modifiers=protocol_modifiers)
        checked_pseudos = protocol.check_pseudos(
            modifier_name=protocol_modifiers.get('pseudo', None),
            pseudo_data=protocol_modifiers.get('pseudo_data', None))
        known_pseudos = checked_pseudos['found']
        #self.ctx.pseudos = get_pseudos_from_dict(self.inputs.structure, known_pseudos)

        # with aiida-pseudo plugin
        family = orm.load_group('SSSP/1.1/PBE/efficiency')
        self.ctx.pseudos = family.get_pseudos(elements=self.inputs.structure.get_kind_names())

        if self.inputs.exclude_semicore:
            # TODO now only consider SSSP
            pseudo_data = _load_pseudo_metadata('semicore_sssp_efficiency_1.1.json')
            pseudo_semicores = {}
            for element in self.ctx.pseudos:
                if pseudo_data[element]['md5'] != self.ctx.pseudos[element].md5:
                    return self.exit_codes.ERROR_INVALID_INPUT_PSEUDOPOTENTIAL
                pseudo_semicores[element] = pseudo_data[element]['semicores']
            self.ctx.pseudo_semicores = pseudo_semicores

    def setup(self):
        """Check inputs"""
        if self.inputs.only_valence:
            valence_info = "valence bands"
        else:
            valence_info = "valence + conduction bands"
        self.report(f'generate Wannier functions for {valence_info}')

        self.setup_protocol()

        projections = [
            self.inputs.scdm_projections.value, 
            self.inputs.spdf_projections.value, 
            self.inputs.pswfc_projections.value
            ]
        if projections.count(True) != 1:
            self.report('Can only use 1 type of projection')
            return self.exit_codes.ERROR_INVALID_INPUT_OPENGRID

        if self.inputs.use_opengrid:
            if self.inputs.spin_orbit_coupling:
                self.report('open_grid.x does not support spin orbit coupling')
                return self.exit_codes.ERROR_INVALID_INPUT_OPENGRID

            try:
                self.inputs.codes.opengrid
            except AttributeError:
                return self.exit_codes.ERROR_INVALID_INPUT_OPENGRID
            self.report('open_grid.x will be used to unfold kmesh')

    def run_seekpath(self):
        """Run the structure through SeeKpath to get the primitive and normalized structure."""
        structure_formula = self.inputs.structure.get_formula()
        self.report(f'running seekpath to get primitive structure for: {structure_formula}')
        kpoints_distance_for_bands = self.inputs.get('bands_kpoints_distance',
            orm.Float(self.ctx.protocol['kpoints_distance_for_bands']))
        args = {'structure': self.inputs.structure,
                'reference_distance': kpoints_distance_for_bands,
                'metadata': {'call_link_label': 'seekpath_structure_analysis'}}
        result = seekpath_structure_analysis(**args)

        self.ctx.current_structure = result['primitive_structure']
        # save explicit_kpoints_path for DFT bands
        self.ctx.explicit_kpoints_path = result['explicit_kpoints']
        # save kpoint_path for Wannier bands
        self.ctx.kpoints_path = {
            'path': result['parameters']['path'],
            'point_coords': result['parameters']['point_coords']
        }

        self.out('primitive_structure', result['primitive_structure'])
        self.out('seekpath_parameters', result['parameters'])

    def setup_parameters(self):
        """setup input parameters of each calculations, 
        since there are some dependencies between input parameters, 
        we store them in context variables."""
        if 'options' in self.inputs:
            self.ctx.options = self.inputs.options.get_dict()
        else:
            self.ctx.options = get_default_options(self.ctx.current_structure, with_mpi=True)
            # or manually assign parallelizations here
            # self.ctx.options = get_manual_options()
        self.report('number of machines {} auto-set according to number of atoms'.
            format(self.ctx.options['resources']['num_machines']))

        # save variables to ctx because they maybe used in several difference methods
        args = {'structure': self.ctx.current_structure, 'pseudos': self.ctx.pseudos}
        # number_of_electrons & number_of_projections will be used in setting wannier parameters,
        # and since they are calculated manually, they will be checked in the
        # `inspect_wannier_workchain` against QE outputs, to ensure they are correct.
        self.ctx.number_of_electrons = get_number_of_electrons(**args)
        self.ctx.number_of_projections = get_number_of_projections(**args)
        # if not using SCDM, then use atom-centred s,p,d orbitals
        if self.inputs.spdf_projections:
            self.ctx.wannier_projections = orm.List(list=get_projections(**args))
        # nscf_nbnd will be used in
        # 1. setting nscf number of bands, or
        # 2. setting nscf number of bands when opengrid is used & opengrid has nscf step
        # 3. setting scf number of bands when opengrid is used & opengrid only has scf step
        args.update({
            'only_valence': self.inputs.only_valence.value,
            'spin_polarized': self.inputs.spin_polarized.value
            })
        self.ctx.nscf_nbnd = get_wannier_number_of_bands(**args)

        self.setup_scf_parameters()
        self.setup_nscf_parameters()
        self.setup_projwfc_parameters()
        self.setup_pw2wannier90_parameters()
        self.setup_wannier90_parameters()

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
                self.report(f'failed to retrieve the cutoff or dual factor for {kind}')
                return self.exit_codes.ERROR_INVALID_INPUT_UNRECOGNIZED_KIND

        number_of_atoms = len(self.ctx.current_structure.sites)
        pw_parameters = {
            'CONTROL': {
                'restart_mode': 'from_scratch',
                'tstress': self.ctx.protocol['tstress'],
                'tprnfor': self.ctx.protocol['tprnfor'],
                'calculation': 'scf',
            },
            'SYSTEM': {
                'ecutwfc': max(ecutwfc),
                'ecutrho': max(ecutrho),
                'smearing': self.ctx.protocol['smearing'],
                'degauss': self.ctx.protocol['degauss'],
                'occupations': self.ctx.protocol['occupations'],
            },
            'ELECTRONS': {
                'conv_thr': self.ctx.protocol['convergence_threshold_per_atom'] * number_of_atoms,
            }
        }
        if self.inputs.use_opengrid and self.inputs.opengrid_only_scf:
            pw_parameters['SYSTEM']['nbnd'] = self.ctx.nscf_nbnd

        if self.inputs.spin_orbit_coupling:
            pw_parameters['SYSTEM']['noncolin'] = True
            pw_parameters['SYSTEM']['lspinorb'] = True

        # Currently only support magnetic with SOC
        # for magnetic w/o SOC, needs 2 separate wannier90 calculations for spin up and down.
        if self.inputs.spin_polarized and self.inputs.spin_orbit_coupling:
            # Magnetization from Kittel, unit: Bohr magneton
            magnetizations = {'Fe': 2.22, 'Co': 1.72, 'Ni': 0.606}
            from aiida_wannier90_workflows.utils.upf import get_number_of_electrons_from_upf
            for i, kind in enumerate(self.inputs.structure.kinds):
                if kind.name in magnetizations:
                    zvalence = get_number_of_electrons_from_upf(self.ctx.pseudos[kind.name])
                    spin_polarization = magnetizations[kind.name] / zvalence
                    pw_parameters['SYSTEM'][f"starting_magnetization({i+1})"] = spin_polarization

        self.ctx.scf_parameters = orm.Dict(dict=pw_parameters)

    def setup_nscf_parameters(self):
        """almost identical to scf_parameters, but need to set nbnd"""
        # we need deepcopy for 'nbnd', otherwise scf_parameters will change as well
        nscf_parameters = deepcopy(self.ctx.scf_parameters.get_dict())
        nscf_parameters['SYSTEM']['nbnd'] = self.ctx.nscf_nbnd
        self.report(f'nscf number of bands set as {self.ctx.nscf_nbnd}')

        if self.inputs.only_valence:
            nscf_parameters['SYSTEM']['occupations'] = 'fixed'
            # pop None to avoid KeyError
            nscf_parameters['SYSTEM'].pop('smearing', None)
            nscf_parameters['SYSTEM'].pop('degauss', None)

        if not self.inputs.use_opengrid or self.inputs.spin_orbit_coupling:
            nscf_parameters['SYSTEM']['nosym'] = True
            nscf_parameters['SYSTEM']['noinv'] = True

        nscf_parameters['CONTROL']['restart_mode'] = 'restart'
        nscf_parameters['CONTROL']['calculation'] = 'nscf'
        nscf_parameters['ELECTRONS']['diagonalization'] = 'cg'
        nscf_parameters['ELECTRONS']['diago_full_acc'] = True

        nscf_parameters = orm.Dict(dict=nscf_parameters)
        self.ctx.nscf_parameters = nscf_parameters

    def setup_projwfc_parameters(self):
        projwfc_parameters = orm.Dict(dict={'PROJWFC': {'DeltaE': 0.2}})

        if self.inputs.pswfc_projections:
            from aiida_wannier90.calculations import Wannier90Calculation
            projwfc_parameters['PROJWFC']['write_amn'] = True
            seedname = Wannier90Calculation._DEFAULT_INPUT_FILE[:-len(Wannier90Calculation._REQUIRED_INPUT_SUFFIX)]
            projwfc_parameters['PROJWFC']['seedname'] = seedname

        self.ctx.projwfc_parameters = projwfc_parameters

    def setup_pw2wannier90_parameters(self):
        """Here no need to set scdm_mu, scdm_sigma"""
        parameters = {
            'write_mmn': True,
            'write_amn': True,
        }
        # write UNK files (to plot WFs)
        if self.inputs.plot_wannier_functions:
            parameters['write_unk'] = True
            self.report("UNK files will be written.")

        if self.inputs.scdm_projections:
            parameters['scdm_proj'] = True

            if self.inputs.only_valence:
                parameters['scdm_entanglement'] = 'isolated'
            else:
                parameters['scdm_entanglement'] = 'erfc'
                # scdm_mu, scdm_sigma will be set after projwfc run

        pw2wannier90_parameters = orm.Dict(dict={'inputpp': parameters})
        self.ctx.pw2wannier90_parameters = pw2wannier90_parameters

    def setup_wannier90_parameters(self):
        parameters = {
            'use_ws_distance': True,
            # no need anymore since kmesh_tol is handled by Wannier90BaseWorkChain
            # 'kmesh_tol': 1e-8
        }

        parameters['num_bands'] = self.ctx.nscf_parameters['SYSTEM']['nbnd']
        # TODO check nospin, spin, soc
        if self.inputs.only_valence:
            num_wann = parameters['num_bands']
        else:
            num_wann = self.ctx.number_of_projections
        if self.inputs.exclude_semicore:
            num_exclude_bands = 0
            for site in self.ctx.current_structure.sites:
                for orb in self.ctx.pseudo_semicores[site.kind_name]:
                    if 'S' in orb:
                        num_exclude_bands += 1
                    elif 'P' in orb:
                        num_exclude_bands += 3
                    elif 'D' in orb:
                        num_exclude_bands += 5
                    elif 'F' in orb:
                        num_exclude_bands += 7
                    else:
                        return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER
            parameters['exclude_bands'] = range(1, num_exclude_bands+1)
            num_wann -= num_exclude_bands
            parameters['num_bands'] -= num_exclude_bands
        if num_wann <= 0:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER
        parameters['num_wann'] = num_wann
        self.report(f'number of Wannier functions set as {num_wann}')

        if self.inputs.scdm_projections:
            parameters['auto_projections'] = True
            projections_info = 'SCDM'
        elif self.inputs.pswfc_projections:
            parameters['auto_projections'] = True
            projections_info = 'PSWFC'
        else:
            # random_projections will be set in the settings input of Wannier90Calculation
            # projections_info = 'random'
            # random projections is rarely used, so if not SCDM then use atomic-orbital projections
            projections_info = 'SPDF'
        self.report(f"using {projections_info} projections")

        if self.inputs.spin_orbit_coupling:
            parameters['spinors'] = True

        parameters['bands_plot'] = True

        if self.inputs.plot_wannier_functions:
            parameters['wannier_plot'] = True
            # 'wannier_plot_list':[1]

        number_of_atoms = len(self.ctx.current_structure.sites)
        if self.inputs.maximal_localisation:
            parameters.update({
                'num_iter': 400,
                'conv_tol': 1e-7 * number_of_atoms,
                'conv_window': 3,
            })
        else:
            parameters.update({'num_iter': 0})

        if self.inputs.only_valence:
            parameters['dis_num_iter'] = 0
        else:
            # if self.inputs.disentanglement:
            if self.inputs.scdm_projections:
                # No disentanglement when using SCDM, otherwise the wannier interpolated bands are wrong
                parameters.update({'dis_num_iter': 0})
            else:
                # Require disentanglement when using s,p,d projections, otherwise interpolated bands are wrong
                parameters.update({
                    'dis_num_iter': 400,
                    'dis_conv_tol': parameters['conv_tol'],
                    # Here +3 means Fermi+3 eV, however Fermi energy is calculated dynamically,
                    # so later in Wannier90WorkChain, it will add Fermi energy on top of this dis_froz_max,
                    # so this dis_froz_max is a relative value.
                    #TODO +3 eV is a bit arbitrary, consider a better value?
                    'dis_froz_max': +3.0,
                    #'dis_mix_ratio':1.d0,
                    #'dis_win_max':10.0,
                })
                if self.inputs.pswfc_projections and 'auto_froz_max' in self.inputs and self.inputs.auto_froz_max:
                    # Use None to represent automatically choose froz_max based on projectability
                    parameters['dis_froz_max'] = None

        if self.inputs.retrieve_hamiltonian:
            parameters['write_tb'] = True
            parameters['write_hr'] = True
            parameters['write_xyz'] = True

        wannier90_parameters = orm.Dict(dict=parameters)
        self.ctx.wannier90_parameters = wannier90_parameters

    def prepare_scf_inputs(self):
        """Return the dictionary of inputs to be used as the basis for each `PwBaseWorkChain`."""
        inputs = AttributeDict({
            'pw': {
                'code': self.inputs.codes.pw,
                'pseudos': self.ctx.pseudos,
                'parameters': self.ctx.scf_parameters,
                'metadata': {},
            }
        })
        inputs.kpoints_distance = orm.Float(self.ctx.protocol['kpoints_mesh_density'])
        inputs.pw.metadata.options = self.ctx.options
        return inputs

    def prepare_nscf_inputs(self):
        """Return the dictionary of inputs to be used as the basis for each `PwBaseWorkChain`."""
        inputs = AttributeDict({
            'pw': {
                'code': self.inputs.codes.pw,
                'pseudos': self.ctx.pseudos,
                'parameters': self.ctx.nscf_parameters,
                'metadata': {},
            }
        })

        kpoints_distance = orm.Float(self.ctx.protocol['kpoints_mesh_density'])
        force_parity = self.inputs.get('kpoints_force_parity', orm.Bool(False))
        args = {
            'structure': self.ctx.current_structure, 
            'distance': kpoints_distance, 
            'force_parity': force_parity, 
            'metadata': {'call_link_label': 'create_kpoints_from_distance'}
        }
        # store it for wannier90 kpoints
        self.ctx.nscf_kpoints = create_kpoints_from_distance(**args)
        if self.inputs.use_opengrid:
            # set a kmesh, nscf will use symmetry and reduce it to IBZ
            inputs.kpoints = self.ctx.nscf_kpoints
        else:
            # convert kmesh to explicit list, since auto generated kpoints
            # maybe different between QE & Wannier90. Here we explicitly
            # generate a list of kpoint to avoid discrepencies between
            # QE's & Wannier90's automatically generated kpoints.
            args = {
                'kmesh': self.ctx.nscf_kpoints,
                'metadata': {'call_link_label': 'convert_kpoints_mesh_to_list'}
            }
            inputs.kpoints = convert_kpoints_mesh_to_list(**args)
            # store it for setting wannier90 mp_grid
            self.ctx.nscf_explicit_kpoints = inputs.kpoints

        inputs.pw.metadata.options = self.ctx.options
        return inputs

    def prepare_projwfc_inputs(self):
        inputs = AttributeDict({
            'code': self.inputs.codes.projwfc,
            'parameters': self.ctx.projwfc_parameters,
            'metadata': {},
        })
        inputs.metadata.options = self.ctx.options
        return inputs

    def prepare_pw2wannier90_inputs(self):
        inputs = AttributeDict({
            'code': self.inputs.codes.pw2wannier90,
            'parameters': self.ctx.pw2wannier90_parameters,
            'metadata': {},
        })
        inputs.metadata.options = self.ctx.options
        return inputs

    def prepare_wannier90_inputs(self):
        inputs = AttributeDict({
            'code': self.inputs.codes.wannier90,
            'parameters': self.ctx.wannier90_parameters,
            'metadata': {},
            'kpoint_path': orm.Dict(dict=self.ctx.kpoints_path)
        })

        # if inputs.kpoints is a kmesh, mp_grid will be auto-set, 
        # otherwise we need to set it manually
        if self.inputs.use_opengrid:
            # kpoints will be set dynamically after opengrid calculation,
            # the self.ctx.nscf_kpoints won't be used.
            inputs.kpoints = self.ctx.nscf_kpoints
        else:
            inputs.kpoints = self.ctx.nscf_explicit_kpoints
            parameters = self.ctx.wannier90_parameters.get_dict()
            parameters['mp_grid'] = self.ctx.nscf_kpoints.get_kpoints_mesh()[0]
            inputs.parameters = orm.Dict(dict=parameters)

        if self.inputs.spdf_projections:
            inputs.projections = self.ctx.wannier_projections

        settings = {}
        # ramdom projections are rarely used, switch to atom-centred s,p,d orbitals
        # if not self.inputs.auto_projections:
        #     settings['random_projections'] = True

        if self.inputs.retrieve_hamiltonian:
            # settings['retrieve_hoppings'] = True
            # tbmodels needs aiida.win file
            settings['additional_retrieve_list'] = ['*.win']
            # also retrieve .chk file in case we need it later
            # seedname = Wannier90Calculation._DEFAULT_INPUT_FILE.split('.')[0]
            # settings['additional_retrieve_list'] += [
            #     '{}.{}'.format(seedname, ext)
            #     for ext in ['chk', 'eig', 'amn', 'mmn', 'spn']
            # ]

        inputs['settings'] = orm.Dict(dict=settings)
        inputs.metadata.options = self.ctx.options
        return inputs

    def prepare_opengrid_inputs(self):
        inputs = AttributeDict({
            'code': self.inputs.codes.opengrid,
            'metadata': {},
        })
        inputs.metadata.options = self.ctx.options
        return inputs

    def run_wannier_workchain(self):
        """Run the `PwBandsWorkChain` to compute the band structure."""
        inputs = AttributeDict({
            'structure': self.ctx.current_structure,
            'scf': self.prepare_scf_inputs(),
            'nscf': self.prepare_nscf_inputs(),
            'projwfc': self.prepare_projwfc_inputs(),
            'pw2wannier90': self.prepare_pw2wannier90_inputs(),
            'wannier90': self.prepare_wannier90_inputs(),
        })
        inputs.metadata = {'call_link_label': 'wannier'}

        if self.inputs.use_opengrid:
            from aiida_wannier90_workflows.workflows.opengrid import Wannier90OpengridWorkChain
            inputs['opengrid'] = {'code': self.inputs.codes.opengrid}
            inputs['opengrid_only_scf'] = self.inputs.opengrid_only_scf
            running = self.submit(Wannier90OpengridWorkChain, **inputs)
        else:
            running = self.submit(Wannier90WorkChain, **inputs)
        self.report(f'launching {running.process_label}<{running.pk}>')

        return ToContext(workchain_wannier=running)

    def inspect_wannier_workchain(self):
        workchain = self.ctx.workchain_wannier

        if not workchain.is_finished_ok:
            self.report(f'sub process {workchain.process_label}<{workchain.pk}> failed')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER

        # check the calculated number of projections is consistent with QE projwfc.x
        workchain_outputs = workchain.get_outgoing(link_type=LinkType.RETURN).nested()
        if not self.inputs.only_valence and 'projwfc' in workchain_outputs:
            num_proj = len(workchain_outputs['projwfc']['projections'].get_orbitals())
            if self.ctx.number_of_projections != num_proj:
                self.report(f'number of projections {self.ctx.number_of_projections} != projwfc.x output {num_proj}')
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER
        # check that the number of electrons is consistent with QE output
        num_elec = workchain_outputs['scf']['output_parameters']['number_of_electrons']
        if self.ctx.number_of_electrons != num_elec:
            self.report(f'number of electrons {self.ctx.number_of_electrons} != QE output {num_elec}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER

    def should_run_bands(self):
        result = 'compare_dft_bands' in self.inputs and self.inputs.compare_dft_bands
        if result:
            self.report('running a DFT bands calculation for comparing with Wannier interpolated bands')
        return result

    def prepare_bands_inputs(self):
        inputs = AttributeDict({
            'pw': {
                'structure': self.ctx.current_structure,
                'code': self.inputs.codes.pw,
                'pseudos': self.ctx.pseudos,
                'parameters': {},
                'metadata': {'options': self.ctx.options}
            },
            'metadata': {'call_link_label': 'bands'}
        })
        wannier_outputs = self.ctx.workchain_wannier.get_outgoing(link_type=LinkType.RETURN).nested()
        # should use wannier90 kpath, otherwise number of kpoints
        # of DFT and w90 is not consistent
        # inputs.kpoints = self.ctx.explicit_kpoints_path
        wannier_bands = wannier_outputs['wannier90']['interpolated_bands']
        wannier_kpoints = orm.KpointsData()
        wannier_kpoints.set_kpoints(wannier_bands.get_kpoints())
        wannier_kpoints.set_attribute_many({
            'cell': wannier_bands.attributes['cell'],
            'pbc1': wannier_bands.attributes['pbc1'],
            'pbc2': wannier_bands.attributes['pbc2'],
            'pbc3': wannier_bands.attributes['pbc3'],
            'labels': wannier_bands.attributes['labels'],
            # 'array|kpoints': ,
            'label_numbers': wannier_bands.attributes['label_numbers']
        })
        inputs.kpoints = wannier_kpoints
        inputs.pw.parent_folder = wannier_outputs['scf']['remote_folder']

        inputs.pw.parameters = self.ctx.nscf_parameters.get_dict()
        inputs.pw.parameters.setdefault('CONTROL', {})
        inputs.pw.parameters.setdefault('SYSTEM', {})
        inputs.pw.parameters.setdefault('ELECTRONS', {})
        inputs.pw.parameters['CONTROL']['calculation'] = 'bands'
        inputs.pw.parameters['ELECTRONS'].setdefault('diagonalization', 'cg')
        inputs.pw.parameters['ELECTRONS'].setdefault('diago_full_acc', True)
        return inputs

    def run_bands(self):
        """run a DFT bands calculation for comparison."""
        inputs = self.prepare_bands_inputs()
        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'launching {running.process_label}<{running.pk}> in {"bands"} mode')
        return ToContext(workchain_bands=running)

    def inspect_bands(self):
        """Verify that the PwBaseWorkChain for the bands run finished successfully."""
        workchain = self.ctx.workchain_bands

        if not workchain.is_finished_ok:
            self.report(f'bands {workchain.process_label} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_BANDS

    def results(self):
        """Attach the relevant output nodes from the band calculation to the workchain outputs for convenience."""
        workchain_outputs = self.ctx.workchain_wannier.get_outgoing(link_type=LinkType.RETURN).nested()

        self.out('scf_parameters', workchain_outputs['scf']['output_parameters'])
        if 'nscf' in workchain_outputs:
            self.out('nscf_parameters', workchain_outputs['nscf']['output_parameters'])
        if 'projwfc' in workchain_outputs:
            self.out('projwfc_bands', workchain_outputs['projwfc']['bands'])
            self.out('projwfc_projections', workchain_outputs['projwfc']['projections'])
        self.out('pw2wannier90_remote_folder', workchain_outputs['pw2wannier90']['remote_folder'])
        self.out('wannier90_parameters', workchain_outputs['wannier90']['output_parameters'])
        self.out('wannier90_retrieved', workchain_outputs['wannier90']['retrieved'])
        self.out('wannier90_remote_folder', workchain_outputs['wannier90']['remote_folder'])
        if 'interpolated_bands' in workchain_outputs['wannier90']:
            w90_bands = workchain_outputs['wannier90']['interpolated_bands']
            self.out('wannier90_interpolated_bands', w90_bands)
            self.report(f'wannier90 interpolated bands pk: {w90_bands.pk}')

        if 'workchain_bands' in self.ctx:
            self.out('bands_parameters', self.ctx.workchain_bands.outputs.output_parameters)
            dft_bands = self.ctx.workchain_bands.outputs.output_band
            self.out('dft_bands', dft_bands)
            self.report(f'DFT bands pk: {dft_bands.pk}')

        self.report(f'{self.get_name()} successfully completed')

def get_default_options(structure, with_mpi=False):
    """Increase wallclock to 5 hour, use mpi, set number of machines according to 
    number of atoms.
    
    :param with_mpi: [description], defaults to False
    :type with_mpi: bool, optional
    :return: [description]
    :rtype: [type]
    """
    def estimate_num_machines(structure):
        """
        1 <= num_atoms <= 10 -> 1 machine
        11 <= num_atoms <= 20 -> 2 machine
        ...
        
        :param structure: crystal structure
        :type structure: aiida.orm.StructureData
        :return: estimated number of machines based on number of atoms
        :rtype: int
        """
        from math import ceil
        num_atoms = len(structure.sites)
        return ceil(num_atoms / 10)

    from aiida_quantumespresso.utils.resources import get_default_options as get_opt
    num_machines = estimate_num_machines(structure)
    opt = get_opt(
        max_num_machines=num_machines,
        max_wallclock_seconds=3600 * 5, # 5 hours
        with_mpi=with_mpi
    )
    return opt

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
