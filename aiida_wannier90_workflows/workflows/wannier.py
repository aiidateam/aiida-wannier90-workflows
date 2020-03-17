from aiida import orm
from aiida.common import AttributeDict
from aiida.engine.processes import WorkChain, ToContext, if_
from aiida.engine.processes import calcfunction
from aiida.plugins import WorkflowFactory, CalculationFactory
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation
from aiida_quantumespresso.calculations.pw2wannier90 import Pw2wannier90Calculation
from aiida_wannier90.calculations import Wannier90Calculation


class Wannier90WorkChain(WorkChain):
    """
    Workchain to obtain maximally localised Wannier functions (MLWF)
    Authors: Antimo Marrazzo (antimo.marrazzo@epfl.ch), Giovanni Pizzi (giovanni.pizzi@epfl.ch), Junfeng Qiao(junfeng.qiao@epfl.ch)
    
    MIT License - Copyright (c), 2018, ECOLE POLYTECHNIQUE FEDERALE DE LAUSANNE
    (Theory and Simulation of Materials (THEOS) and National Centre for 
    Computational Design and Discovery of Novel Materials (NCCR MARVEL)).
    All rights reserved.

    Scheme: setup --> relax(optional) --> scf --> nscf --> projwfc 
            -> wannier90_postproc --> pw2wannier90 --> wannier90 --> results
    
    This is a very basic workchain, in that user needs to specify 
    inputs of every step. Please consider using Wannier90BandsWorkChain, 
    which automatically generates inputs.
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)
        spec.input(
            'structure',
            valid_type=orm.StructureData,
            help='The inputs structure.'
        )
        spec.input(
            'clean_workdir',
            valid_type=orm.Bool,
            default=orm.Bool(False),
            help=
            'If `True`, work directories of all called calculation will be cleaned at the end of execution.'
        )
        spec.input(
            'only_valence',
            valid_type=orm.Bool,
            default=orm.Bool(False),
            help='Whether only wannierise valence band or not'
        )
        spec.input(
            'wannier_energies_relative_to_fermi',
            valid_type=orm.Bool,
            default=orm.Bool(False),
            help=
            'determines if the energies(dis_froz_min/max, dis_win_min/max) defined in the input parameters '
            + 'are relative to scf Fermi energy or not.'
        )
        spec.input(
            'scdm_thresholds',
            valid_type=orm.Dict,
            default=orm.Dict(
                dict={
                    'max_projectability': 0.95,
                    'sigma_factor': 3
                }
            ),
            help='can contain two keyword: max_projectability, sigma_factor'
        )
        spec.expose_inputs(
            PwRelaxWorkChain,
            namespace='relax',
            exclude=('clean_workdir', 'structure'),
            namespace_options={
                'required':
                False,
                'populate_defaults':
                False,
                'help':
                'Inputs for the `PwRelaxWorkChain`, if not specified at all, the relaxation step is skipped.'
            }
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace='scf',
            exclude=('clean_workdir', 'pw.structure'),
            namespace_options={
                'help':
                'Inputs for the `PwBaseWorkChain` for the SCF calculation.'
            }
        )
        spec.expose_inputs(
            PwBaseWorkChain,
            namespace='nscf',
            exclude=('clean_workdir', 'pw.structure'),
            namespace_options={
                'help':
                'Inputs for the `PwBaseWorkChain` for the NSCF calculation.'
            }
        )
        spec.expose_inputs(
            ProjwfcCalculation,
            namespace='projwfc',
            exclude=('parent_folder', ),
            namespace_options={
                'required':
                False,
                'help':
                'Inputs for the `ProjwfcCalculation` for the Projwfc calculation.'
            }
        )
        spec.expose_inputs(
            Pw2wannier90Calculation,
            namespace='pw2wannier90',
            exclude=('parent_folder', 'nnkp_file'),
            namespace_options={
                'help':
                'Inputs for the `Pw2wannier90Calculation` for the pw2wannier90 calculation.'
            }
        )
        spec.expose_inputs(
            Wannier90Calculation,
            namespace='wannier90',
            exclude=('structure', 'kpoints'),
            namespace_options={
                'help':
                'Inputs for the `Wannier90Calculation` for the Wannier90 calculation.'
            }
        )
        spec.outline(
            cls.setup,
            if_(cls.should_do_relax)(
                cls.run_relax,
                cls.inspect_relax,
            ),
            cls.run_scf,
            cls.inspect_scf,
            cls.run_nscf,
            cls.inspect_nscf,
            if_(cls.should_do_projwfc)(
                cls.run_projwfc,
                cls.inspect_projwfc,
            ),
            cls.run_wannier90_pp,
            cls.inspect_wannier90_pp,
            cls.run_pw2wannier90,
            cls.inspect_pw2wannier90,
            cls.run_wannier90,
            cls.inspect_wannier90,
            cls.results,
        )

        spec.expose_outputs(
            PwRelaxWorkChain,
            namespace='relax',
            namespace_options={'required': False}
        )
        spec.expose_outputs(PwBaseWorkChain, namespace='scf')
        spec.expose_outputs(PwBaseWorkChain, namespace='nscf')
        spec.expose_outputs(
            ProjwfcCalculation,
            namespace='projwfc',
            namespace_options={'required': False}
        )
        spec.expose_outputs(Pw2wannier90Calculation, namespace='pw2wannier90')
        spec.expose_outputs(Wannier90Calculation, namespace='wannier90_pp')
        spec.expose_outputs(Wannier90Calculation, namespace='wannier90')

        spec.exit_code(
            401,
            'ERROR_SUB_PROCESSS_FAILED_SETUP',
            message='setup failed, check your input'
        )
        spec.exit_code(
            402,
            'ERROR_SUB_PROCESS_FAILED_RELAX',
            message='the PwRelaxWorkChain sub process failed'
        )
        spec.exit_code(
            403,
            'ERROR_SUB_PROCESS_FAILED_SCF',
            message='the scf PwBasexWorkChain sub process failed'
        )
        spec.exit_code(
            404,
            'ERROR_SUB_PROCESS_FAILED_NSCF',
            message='the nscf PwBasexWorkChain sub process failed'
        )
        spec.exit_code(
            405,
            'ERROR_SUB_PROCESS_FAILED_PROJWFC',
            message='the ProjwfcCalculation sub process failed'
        )
        spec.exit_code(
            406,
            'ERROR_SUB_PROCESS_FAILED_WANNIER90PP',
            message='the postproc Wannier90Calculation sub process failed'
        )
        spec.exit_code(
            407,
            'ERROR_SUB_PROCESS_FAILED_PW2WANNIER90',
            message='the Pw2wannier90Calculation sub process failed'
        )
        spec.exit_code(
            408,
            'ERROR_SUB_PROCESS_FAILED_WANNIER90',
            message='the Wannier90Calculation sub process failed'
        )

    def setup(self):
        """
        Define the current structure in the context to be the input structure.
        """
        self.ctx.current_structure = self.inputs.structure

        inputs = AttributeDict(
            self.exposed_inputs(Wannier90Calculation, namespace='wannier90')
        )
        parameters = inputs.parameters.get_dict()

        self.ctx.bands_plot = parameters.get('bands_plot', False)
        self.ctx.auto_projections = parameters.get('auto_projections', False)

        # check bands_plot kpoint_path
        if self.ctx.bands_plot:
            kpoint_path = inputs.get('kpoint_path', None)
            if kpoint_path is None:
                self.report(
                    'bands_plot is required but no kpoint_path provided'
                )
                return self.exit_codes.ERROR_SUB_PROCESSS_FAILED_SETUP

        # debug
        # self.ctx.workchain_scf = orm.load_node(13623) # CsH
        # self.ctx.workchain_nscf = orm.load_node(13643)
        # self.ctx.calc_projwfc = orm.load_node(13652)
        # self.ctx.nscf_kmesh = orm.KpointsData()
        # self.ctx.nscf_kmesh.set_kpoints_mesh([9, 9, 9])
        # self.ctx.workchain_wannier90_pp = orm.load_node(13712)
        # self.ctx.calc_pw2wannier90 = orm.load_node(13726)
        # self.ctx.calc_wannier90 = orm.load_node(13798)

    def should_do_relax(self):
        """
        If the 'relax' input namespace was specified, we relax the input structure.
        """
        return 'relax' in self.inputs

    def run_relax(self):
        """
        Run the PwRelaxWorkChain to run a relax calculation
        """
        inputs = AttributeDict(
            self.exposed_inputs(PwRelaxWorkChain, namespace='relax')
        )
        inputs.structure = self.ctx.current_structure

        running = self.submit(PwRelaxWorkChain, **inputs)

        self.report('launching PwRelaxWorkChain<{}>'.format(running.pk))

        return ToContext(workchain_relax=running)

    def inspect_relax(self):
        """
        verify that the PwRelaxWorkChain successfully finished.
        """
        workchain = self.ctx.workchain_relax

        if not workchain.is_finished_ok:
            self.report(
                'PwRelaxWorkChain failed with exit status {}'.format(
                    workchain.exit_status
                )
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        self.ctx.current_structure = workchain.outputs.output_structure

    def run_scf(self):
        """
        Run the PwBaseWorkChain in scf mode on the primitive cell of (optionally relaxed) input structure.
        """
        inputs = AttributeDict(
            self.exposed_inputs(PwBaseWorkChain, namespace='scf')
        )
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.parameters = inputs.pw.parameters.get_dict()
        inputs.pw.parameters.setdefault('CONTROL', {})
        inputs.pw.parameters['CONTROL']['calculation'] = 'scf'

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report(
            'scf step - launching PwBaseWorkChain<{}> in {} mode'.format(
                running.pk, 'scf'
            )
        )

        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """Verify that the PwBaseWorkChain for the scf run successfully finished."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(
                'scf PwBaseWorkChain failed with exit status {}'.format(
                    workchain.exit_status
                )
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.current_folder = workchain.outputs.remote_folder
        self.report("scf PwBaseWorkChain successfully finished")

    def run_nscf(self):
        """
        Run the PwBaseWorkChain in nscf mode
        """
        inputs = AttributeDict(
            self.exposed_inputs(PwBaseWorkChain, namespace='nscf')
        )
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.parent_folder = self.ctx.current_folder
        inputs.pw.parameters = inputs.pw.parameters.get_dict()
        inputs.pw.parameters.setdefault('CONTROL', {})
        inputs.pw.parameters.setdefault('SYSTEM', {})
        inputs.pw.parameters.setdefault('ELECTRONS', {})
        inputs.pw.parameters['CONTROL']['restart_mode'] = 'from_scratch'
        inputs.pw.parameters['CONTROL']['calculation'] = 'nscf'
        inputs.pw.parameters['SYSTEM']['nosym'] = True
        inputs.pw.parameters['SYSTEM']['noinv'] = True
        inputs.pw.parameters['ELECTRONS']['diagonalization'] = 'cg'
        inputs.pw.parameters['ELECTRONS']['diago_full_acc'] = True

        if self.inputs.only_valence:
            inputs.pw.parameters['SYSTEM']['occupations'] = 'fixed'
            inputs.pw.parameters['SYSTEM'].pop(
                'smearing', None
            )  # pop None to avoid KeyError
            inputs.pw.parameters['SYSTEM'].pop(
                'degauss', None
            )  # pop None to avoid KeyError

        # inputs.pw.pseudos is an AttributeDict, but calcfunction only accepts
        # orm.Data, so we unpack it to pass in orm.UpfData
        inputs.pw.parameters = update_nscf_num_bands(
            orm.Dict(dict=inputs.pw.parameters),
            self.ctx.workchain_scf.outputs.output_parameters,
            self.ctx.current_structure, self.inputs.only_valence,
            **inputs.pw.pseudos
        )
        self.report(
            'nscf number of bands set as ' +
            str(inputs.pw.parameters['SYSTEM']['nbnd'])
        )

        # check kmesh
        try:
            inputs.kpoints
        except AttributeError:
            # then kpoints_distance must exists, since this is ensured by inputs check of this workchain
            from aiida_quantumespresso.workflows.functions.create_kpoints_from_distance import create_kpoints_from_distance
            force_parity = inputs.get('kpoints_force_parity', orm.Bool(False))
            kmesh = create_kpoints_from_distance(
                self.ctx.current_structure, inputs.kpoints_distance,
                force_parity
            )
            #kpoints_data = orm.KpointsData()
            # kpoints_data.set_cell_from_structure(self.ctx.current_structure)
            # kmesh = kpoints_data.set_kpoints_mesh_from_density(inputs.kpoints_distance.value)
        else:
            try:
                inputs.kpoints.get_kpoints_mesh()
            except AttributeError:
                self.report("nscf only support `mesh' type KpointsData")
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_NSCF
            else:
                kmesh = inputs.kpoints
        # convert kmesh to explicit list, since auto generated kpoints
        # maybe different between QE & Wannier90. Here we explicitly
        # generate a list of kpoint to avoid discrepencies between
        # QE's & Wannier90's automatically generated kpoints.
        self.ctx.nscf_kmesh = kmesh  # store it since it will be used by w90
        inputs.kpoints = convert_kpoints_mesh_to_list(kmesh)

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)

        self.report(
            'nscf step - launching PwBaseWorkChain<{}> in {} mode'.format(
                running.pk, 'nscf'
            )
        )

        return ToContext(workchain_nscf=running)

    def inspect_nscf(self):
        """Verify that the PwBaseWorkChain for the nscf run successfully finished."""
        workchain = self.ctx.workchain_nscf

        if not workchain.is_finished_ok:
            self.report(
                'nscf PwBaseWorkChain failed with exit status {}'.format(
                    workchain.exit_status
                )
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_NSCF

        self.ctx.current_folder = workchain.outputs.remote_folder
        self.report("nscf PwBaseWorkChain successfully finished")

    def should_do_projwfc(self):
        """
        If the 'auto_projections = true' && only_valence, we 
        run projwfc calculation to extract SCDM mu & sigma.
        """
        self.report("SCDM mu & sigma are auto-set using projectability")
        return self.ctx.auto_projections and not self.inputs.only_valence.value

    def run_projwfc(self):
        """
        Projwfc step
        :return:
        """
        inputs = AttributeDict(
            self.exposed_inputs(ProjwfcCalculation, namespace='projwfc')
        )
        inputs.parent_folder = self.ctx.current_folder

        inputs = prepare_process_inputs(ProjwfcCalculation, inputs)
        running = self.submit(ProjwfcCalculation, **inputs)

        self.report(
            'projwfc step - launching ProjwfcCalculation<{}>'.format(
                running.pk
            )
        )

        return ToContext(calc_projwfc=running)

    def inspect_projwfc(self):
        """Verify that the ProjwfcCalculation for the projwfc run successfully finished."""
        calculation = self.ctx.calc_projwfc

        if not calculation.is_finished_ok:
            self.report(
                'ProjwfcCalculation failed with exit status {}'.format(
                    calculation.exit_status
                )
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PROJWFC

        self.ctx.current_folder = calculation.outputs.remote_folder
        self.report("projwfc ProjwfcCalculation successfully finished")

    def run_wannier90_pp(self):
        """The input of wannier90 calculation is build here.
        
        :return: [description]
        :rtype: [type]
        """
        inputs = AttributeDict(
            self.exposed_inputs(Wannier90Calculation, namespace='wannier90')
        )
        inputs.structure = self.ctx.current_structure
        parameters = inputs.parameters.get_dict()

        # get nscf kmesh
        inputs.kpoints = self.ctx.workchain_nscf.inputs.kpoints
        # the input kpoints of nscf is an explicitly generated list of kpoints,
        # we mush retrieve the original kmesh, and explicitly set w90 mp_grid keyword
        parameters['mp_grid'] = self.ctx.nscf_kmesh.get_kpoints_mesh()[0]

        # check num_bands, exclude_bands, nscf nbnd
        nbnd = self.ctx.workchain_nscf.outputs.output_parameters.get_dict(
        )['number_of_bands']
        num_ex_bands = len(get_exclude_bands(inputs.parameters.get_dict()))
        parameters['num_bands'] = nbnd - num_ex_bands

        # set num_wann for auto_projections
        if self.ctx.auto_projections:
            if self.inputs.only_valence:
                parameters['num_wann'] = parameters['num_bands']
                inputs.parameters = orm.Dict(dict=parameters)
            else:
                inputs.parameters = orm.Dict(dict=parameters)
                inputs.parameters = update_w90_params_numwann(
                    inputs.parameters,
                    self.ctx.calc_projwfc.outputs.projections
                )
                self.report(
                    'number of Wannier functions extracted from projections: '
                    + str(inputs.parameters['num_wann'])
                )

        # get scf Fermi energy
        try:
            energies_relative_to_fermi = self.inputs.get(
                'wannier_energies_relative_to_fermi'
            )
            inputs.parameters = update_w90_params_fermi(
                inputs.parameters,
                self.ctx.workchain_scf.outputs.output_parameters,
                energies_relative_to_fermi
            )
        except TypeError:
            self.report(
                "Error in retriving the SCF Fermi energy "
                "from pk: {}".format(self.ctx.workchain_scf)
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90PP

        #Check if settings is given in input
        try:
            settings = inputs['settings'].get_dict()
        except KeyError:
            settings = {}
        settings['postproc_setup'] = True
        inputs['settings'] = settings

        inputs = prepare_process_inputs(Wannier90Calculation, inputs)
        running = self.submit(Wannier90Calculation, **inputs)

        self.report(
            'wannier90 postproc step - launching Wannier90Calculation<{}> in postproc mode'
            .format(running.pk)
        )

        return ToContext(calc_wannier90_pp=running)

    def inspect_wannier90_pp(self):
        """Verify that the Wannier90Calculation for the wannier90 run successfully finished."""
        workchain = self.ctx.calc_wannier90_pp

        if not workchain.is_finished_ok:
            self.report(
                'wannier90 postproc Wannier90Calculation failed with exit status {}'
                .format(workchain.exit_status)
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90PP

        self.ctx.current_folder = workchain.outputs.remote_folder
        self.report(
            "wannier90 postproc Wannier90Calculation successfully finished"
        )

    def run_pw2wannier90(self):
        inputs = AttributeDict(
            self.exposed_inputs(
                Pw2wannier90Calculation, namespace='pw2wannier90'
            )
        )

        try:
            remote_folder = self.ctx.workchain_nscf.outputs.remote_folder
        except AttributeError:
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PW2WANNIER90(
                'the nscf WorkChain did not output a remote_folder node'
            )

        inputs['parent_folder'] = remote_folder
        inputs['nnkp_file'] = self.ctx.calc_wannier90_pp.outputs.nnkp_file
        inputs.parameters = inputs.parameters.get_dict()
        inputs.parameters['inputpp'].update({
            'write_mmn': True,
            'write_amn': True,
        })

        if self.ctx.auto_projections:
            inputs.parameters['inputpp']['scdm_proj'] = True

            # TODO: if exclude_band: gaussian
            # TODO: auto check if is insulator?
            if self.inputs.only_valence:
                inputs.parameters['inputpp']['scdm_entanglement'] = 'isolated'
            else:
                inputs.parameters['inputpp']['scdm_entanglement'] = 'erfc'
                try:
                    inputs.parameters = update_pw2wan_params_mu_sigma(
                        parameters=orm.Dict(dict=inputs.parameters),
                        wannier_parameters=self.ctx.calc_wannier90_pp.inputs.parameters,
                        bands=self.ctx.calc_projwfc.outputs.bands,
                        projections=self.ctx.calc_projwfc.outputs.projections,
                        thresholds=self.inputs.get('scdm_thresholds')
                    )
                except ValueError:
                    self.report(
                        'WARNING: update_pw2wan_params_mu_sigma failed!'
                    )
                    return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PW2WANNIER90

        inputs = prepare_process_inputs(Pw2wannier90Calculation, inputs)
        running = self.submit(Pw2wannier90Calculation, **inputs)

        self.report(
            'pw2wannier90 step - launching Pw2Wannier90Calculation<{}>'.format(
                running.pk
            )
        )
        return ToContext(calc_pw2wannier90=running)

    def inspect_pw2wannier90(self):
        """Verify that the PwBaseWorkChain for the wannier90 run successfully finished."""
        workchain = self.ctx.calc_pw2wannier90

        if not workchain.is_finished_ok:
            self.report(
                'Pw2wannier90Calculation failed with exit status {}'.format(
                    workchain.exit_status
                )
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PW2WANNIER90

        self.ctx.current_folder = workchain.outputs.remote_folder
        self.report("Pw2wannier90Calculation successfully finished")

    def run_wannier90(self):
        try:
            remote_folder = self.ctx.calc_pw2wannier90.outputs.remote_folder
        except AttributeError:
            self.report(
                'the Pw2wannier90Calculation did not output a remote_folder node'
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90

        # we need metadata in exposed_inputs
        inputs = AttributeDict(
            self.exposed_inputs(Wannier90Calculation, namespace='wannier90')
        )
        pp_inputs = self.ctx.calc_wannier90_pp.inputs
        pp_keys = [
            'code', 'parameters', 'kpoint_path', 'structure', 'kpoints',
            'settings'
        ]
        for key in pp_keys:
            inputs[key] = pp_inputs[key]

        inputs['remote_input_folder'] = remote_folder

        settings = inputs.settings.get_dict()
        settings['postproc_setup'] = False
        inputs.settings = settings

        inputs = prepare_process_inputs(Wannier90Calculation, inputs)
        running = self.submit(Wannier90Calculation, **inputs)

        self.report(
            'wannier90 step - launching Wannier90Calculation<{}>'.format(
                running.pk
            )
        )

        return ToContext(calc_wannier90=running)

    def inspect_wannier90(self):
        """Verify that the PwBaseWorkChain for the wannier90 run successfully finished."""
        workchain = self.ctx.calc_wannier90

        if not workchain.is_finished_ok:
            self.report(
                'Wannier90Calculation failed with exit status {}'.format(
                    workchain.exit_status
                )
            )
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90

        self.report("Wannier90Calculation successfully finished")
        self.ctx.current_folder = workchain.outputs.remote_folder

    # def run_bands(self):  # pylint: disable=inconsistent-return-statements
    #     """
    #     Run the PwBaseWorkChain to run a bands PwCalculation
    #     """
    #     try:
    #         remote_folder = self.ctx.workchain_scf.out.remote_folder
    #     except AttributeError:
    #         self.abort_nowait(
    #             'the scf workchain did not output a remote_folder node')
    #         return

    #     inputs = dict(self.ctx.inputs)
    #     structure = self.ctx.structure_relaxed_primitive
    #     restart_mode = 'restart'
    #     calculation_mode = 'bands'

    #     # Set the correct pw.x input parameters
    #     inputs['parameters']['CONTROL']['restart_mode'] = restart_mode
    #     inputs['parameters']['CONTROL']['calculation'] = calculation_mode

    #     # Tell the plugin to retrieve the bands
    #     settings = inputs['settings'].get_dict()
    #     settings['also_bands'] = True

    #     # Final input preparation, wrapping dictionaries in Dict nodes
    #     inputs['kpoints'] = self.ctx.kpoints_path
    #     inputs['structure'] = structure
    #     inputs['parent_folder'] = remote_folder
    #     inputs['parameters'] = Dict(dict=inputs['parameters'])
    #     inputs['settings'] = Dict(dict=settings)
    #     inputs['pseudo_family'] = self.inputs.pseudo_family

    #     running = submit(PwBaseWorkChain, **inputs)

    #     self.report('launching PwBaseWorkChain<{}> in {} mode'.format(
    #         running.pk, calculation_mode))

    #     return ToContext(workchain_bands=running)

    def results(self):
        """
        Attach the desired output nodes directly as outputs of the workchain
        """
        self.report('final step - preparing outputs')
        try:
            self.ctx.workchain_relax
        except AttributeError:
            pass
        else:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.workchain_relax,
                    PwRelaxWorkChain,
                    namespace='relax'
                )
            )
        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_scf, PwBaseWorkChain, namespace='scf'
            )
        )
        self.out_many(
            self.exposed_outputs(
                self.ctx.workchain_nscf, PwBaseWorkChain, namespace='nscf'
            )
        )
        try:
            self.ctx.calc_projwfc
        except AttributeError:
            pass
        else:
            self.out_many(
                self.exposed_outputs(
                    self.ctx.calc_projwfc,
                    ProjwfcCalculation,
                    namespace='projwfc'
                )
            )
        self.out_many(
            self.exposed_outputs(
                self.ctx.calc_pw2wannier90,
                Pw2wannier90Calculation,
                namespace='pw2wannier90'
            )
        )
        self.out_many(
            self.exposed_outputs(
                self.ctx.calc_wannier90_pp,
                Wannier90Calculation,
                namespace='wannier90_pp'
            )
        )
        self.out_many(
            self.exposed_outputs(
                self.ctx.calc_wannier90,
                Wannier90Calculation,
                namespace='wannier90'
            )
        )
        self.report('Wannier90WorkChain successfully completed')


@calcfunction
def convert_kpoints_mesh_to_list(kmesh):
    """works just like kmesh.pl in Wannier90
    
    :param kmesh: [description]
    :type kmesh: KpointsData
    :raises ValueError: [description]
    :return: [description]
    :rtype: [type]
    """
    import numpy as np
    try:  # test if it is a mesh
        results = kmesh.get_kpoints_mesh()
    except AttributeError:
        try:
            kmesh.get_kpoints()
        except AttributeError as e:  # an empty input
            e.args = ('input kmesh is empty!', )
            raise e
        else:
            return kmesh
    else:
        # currently we ignore offset
        mesh = results[0]

        # following is similar to wannier90/kmesh.pl
        totpts = np.prod(mesh)
        weights = np.ones([totpts]) / totpts

        kpoints = np.zeros([totpts, 3])
        ind = 0
        for x in range(mesh[0]):
            for y in range(mesh[1]):
                for z in range(mesh[2]):
                    kpoints[ind, :] = [x / mesh[0], y / mesh[1], z / mesh[2]]
                    ind += 1
        klist = orm.KpointsData()
        klist.set_kpoints(kpoints=kpoints, cartesian=False, weights=weights)
        return klist


@calcfunction
def update_nscf_num_bands(
    nscf_input_parameters, scf_output_parameters, structure, only_valence,
    **pseudos
):
    '''
    this calcfunction does 2 works:
        1. calculate nbnd based on scf output_parameters
        2. calculate number of projections based on pseudos
    The resulting nbnd is the max of the two.
    '''
    def get_num_projections_from_pseudos(structure, **pseudos):
        def get_nprojs_from_upf(upf):
            upf_name = upf.list_object_names()[0]
            upf_content = upf.get_object_content(upf_name)
            upf_content = upf_content.split('\n')
            # get PP_PSWFC block
            pswfc_block = ''
            found_begin = False
            found_end = False
            for line in upf_content:
                if 'PP_PSWFC' in line:
                    pswfc_block += line + '\n'
                    if not found_begin:
                        found_begin = True
                        continue
                    else:
                        if not found_end:
                            found_end = True
                            break
                if found_begin:
                    pswfc_block += line + '\n'

            num_projections = 0
            # parse XML
            import xml.etree.ElementTree as ET
            PP_PSWFC = ET.XML(pswfc_block)
            if len(PP_PSWFC.getchildren()) == 0:
                # old upf format
                import re
                r = re.compile(r'[\d]([SPDF])')
                spdf = r.findall(PP_PSWFC.text)
                for orbit in spdf:
                    orbit = orbit.lower()
                    if orbit == 's':
                        l = 0
                    elif orbit == 'p':
                        l = 1
                    elif orbit == 'd':
                        l = 2
                    elif orbit == 'f':
                        l = 3
                    num_projections += 2 * l + 1
            else:
                # upf format 2.0.1
                for child in PP_PSWFC:
                    l = int(child.get('l'))
                    num_projections += 2 * l + 1
            return num_projections

        tot_nprojs = 0
        composition = structure.get_composition()
        for kind in pseudos:
            upf = pseudos[kind]
            nprojs = get_nprojs_from_upf(upf)
            tot_nprojs += nprojs * composition[kind]
        return tot_nprojs

    # Get info from SCF on number of electrons and number of spin components
    scf_out_dict = scf_output_parameters.get_dict()
    nelectron = int(scf_out_dict['number_of_electrons'])
    nspin = int(scf_out_dict['number_of_spin_components'])
    if only_valence:
        nbands = int(nelectron / 2)
    else:
        nbands = int(0.5 * nelectron * nspin + 4 * nspin)
        # nbands must > num_projections = num_wann
        nprojs = get_num_projections_from_pseudos(structure, **pseudos)
        nbands = max(nbands, nprojs + 10)
    nscf_in_dict = nscf_input_parameters.get_dict()
    nscf_in_dict['SYSTEM']['nbnd'] = nbands
    return orm.Dict(dict=nscf_in_dict)


def get_exclude_bands(parameters):
    """get exclude_bands from Wannier90 parameters
    
    :param parameters: Wannier90Calculation input parameters
    :type parameters: dict, NOT Dict, this is not a calcfunction
    :return: the indices of the bands to be excluded
    :rtype: list
    """
    exclude_bands = parameters.get('exclude_bands', [])
    return exclude_bands


def get_keep_bands(parameters):
    """get keep_bands from Wannier90 parameters
    
    :param parameters: Wannier90Calculation input parameters
    :type parameters: dict, NOT Dict, this is not a calcfunction
    :return: the indices of the bands to be kept
    :rtype: list
    """
    import numpy as np
    exclude_bands = get_exclude_bands(parameters)
    xb_startzero_set = set([idx - 1 for idx in exclude_bands]
                           )  # in Fortran/W90: 1-based; in py: 0-based
    keep_bands = np.array([
        idx for idx in range(parameters['num_bands'] + len(exclude_bands))
        if idx not in xb_startzero_set
    ])
    return keep_bands


@calcfunction
def update_w90_params_fermi(
    parameters, scf_output_parameters, relative_to_fermi
):
    """
    Updated W90 windows with the specified Fermi energy.
    
    :param parameters: Wannier90Calculation input parameters
    :type parameters: Dict
    :param fermi_energy: [description]
    :type fermi_energy: Float
    :param relative_to_fermi: if energies in input parameters are defined relative 
    scf Fermi energy.
    :type relative_to_fermi: Bool
    :return: updated parameters
    :rtype: Dict
    """
    def get_fermi_energy():
        """get Fermi energy from scf output parameters, unit is eV
        """
        try:
            scf_out_dict = scf_output_parameters.get_dict()
            efermi = scf_out_dict['fermi_energy']
            efermi_units = scf_out_dict['fermi_energy_units']
            if efermi_units != 'eV':
                raise TypeError(
                    "Error: Fermi energy is not in eV!"
                    "it is {}".format(efermi_units)
                )
        except AttributeError:
            raise TypeError(
                "Error in retriving the SCF Fermi energy from pk: {}".format(
                    scf_output_parameters.pk
                )
            )
        return efermi

    params = parameters.get_dict()
    fermi = get_fermi_energy()
    params['fermi_energy'] = fermi

    if relative_to_fermi:
        try:
            dwmax = params['dis_win_max']
            dwmax += fermi
            params['dis_win_max'] = dwmax
        except KeyError:
            pass
        try:
            dfmax = params['dis_froz_max']
            dfmax += fermi
            params['dis_froz_max'] = dfmax
        except KeyError:
            pass
        try:
            dwmin = params['dis_win_min']
            dwmin += fermi
            params['dis_win_min'] = dwmin
        except KeyError:
            pass
        try:
            dfmin = params['dis_froz_min']
            dfmin += fermi
            params['dis_froz_min'] = dfmin
        except KeyError:
            pass
    return orm.Dict(dict=params)


@calcfunction
def update_w90_params_numwann(parameters, projections):
    def get_numwann_from_projections():
        """
        Calculate num_wann from projections, also consider exclude_bands
        :param
        :return:
        """
        num_wann = len(projections.get_orbitals())
        num_wann -= len(get_exclude_bands(parameters.get_dict()))
        return num_wann

    parameters_dict = parameters.get_dict()
    parameters_dict['num_wann'] = get_numwann_from_projections()
    return orm.Dict(dict=parameters_dict)


@calcfunction
def update_pw2wan_params_mu_sigma(
    parameters, wannier_parameters, bands, projections, thresholds
):
    """[summary]
    
    :param pw2wan_parameters: [description]
    :type pw2wan_parameters: Dict
    :param mu_sigma: [description]
    :type mu_sigma: Dict
    :return: [description]
    :rtype: Dict
    """
    def get_mu_and_sigma_from_projections(
        wannier_parameters,
        bands,
        projections,  # pylint: disable=too-many-locals
        thresholds
    ):
        '''
        Setting mu parameter for the SCDM-k method:
        The projectability of all orbitals is fitted using an erfc(x)
        function. Mu and sigma are extracted from the fitted distribution,
        with mu = mu_fit - k * sigma, sigma = sigma_fit and
        k a parameter with default k = 3.

        :param bands: output of projwfc, it was computed in the nscf calc
        :param parameters: wannier90 input params (the one to update with this calcfunction)
        :param projections: output of projwfc
        :param thresholds: must contain 'sigma_factor'; scdm_mu will be set to::
            scdm_mu = E(projectability==max_projectability) - sigma_factor * scdm_sigma
            Pass sigma_factor = 0 if you do not want to shift
        :return: a modified Dict in output_parameters, with the proper value for scdm_mu set,
                and a Bool called 'success' that tells if the algorithm could find the energy at which
                the required projectability is achieved.
        '''
        def erfc_scdm(x, mu, sigma):
            from scipy.special import erfc  # pylint: disable=E0611
            return 0.5 * erfc((x - mu) / sigma)

        def fit_erfc(f, xdata, ydata):
            from scipy.optimize import curve_fit
            return curve_fit(f, xdata, ydata, bounds=([-50, 0], [50, 50]))

        # List of specifications of atomic orbitals in dictionary form
        dict_list = [i.get_orbital_dict() for i in projections.get_orbitals()]

        keep_bands = get_keep_bands(wannier_parameters.get_dict())
        # Sum of the projections on all atomic orbitals (shape kpoints x nbands)
        # WITHOUT EXCLUDE BANDS out_array = sum([sum([x[1] for x in projections.get_projections(
        #    **get_dict)]) for get_dict in dict_list])
        out_array = sum([
            sum([
                x[1][:, keep_bands]
                for x in projections.get_projections(**get_dict)
            ]) for get_dict in dict_list
        ])

        # Flattening (projection modulus squared according to QE, energies)
        projwfc_flat, bands_flat = out_array.flatten(), bands.get_bands(
        )[:, keep_bands].flatten()
        # Sorted by energy
        sorted_bands, sorted_projwfc = zip(
            *sorted(zip(bands_flat, projwfc_flat))
        )
        popt, pcov = fit_erfc(erfc_scdm, sorted_bands, sorted_projwfc)
        mu = popt[0]
        sigma = popt[1]
        # Temporary, TODO add check on interpolation
        success = True
        params = {}
        params['scdm_sigma'] = sigma
        sigma_factor = thresholds.get_dict().get('sigma_factor', 3)
        params['scdm_mu'] = mu - sigma * sigma_factor
        result = {
            'success': orm.Bool(success),
            'scdm_parameters': orm.Dict(dict=params)
        }
        return result

    results = get_mu_and_sigma_from_projections(
        wannier_parameters, bands, projections, thresholds
    )
    if not results['success']:
        raise ValueError('mu and sigma failed')
    parameters_dict = parameters.get_dict()
    parameters_dict['inputpp'].update(results['scdm_parameters'].get_dict())
    return orm.Dict(dict=parameters_dict)
