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
from aiida_wannier90_workflows.utils.scdm import fit_scdm_mu_sigma_aiida
from aiida_wannier90_workflows.workflows.base import Wannier90BaseWorkChain

__all__ = ['Wannier90WorkChain']

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

        spec.input('structure', valid_type=orm.StructureData, help='The input structure.')
        spec.input('clean_workdir', valid_type=orm.Bool, default=lambda: orm.Bool(False),
            help='If `True`, work directories of all called calculation will be cleaned at the end of execution.')
        spec.input('scdm_thresholds', valid_type=orm.Dict, default=lambda: orm.Dict(dict={'sigma_factor': 3}), 
            help='Used only if `auto_projections` is in the wannier input parameters. Contains one keyword: sigma_factor')
        spec.expose_inputs(PwRelaxWorkChain, namespace='relax', exclude=('clean_workdir', 'structure'),
            namespace_options={'required': False, 'populate_defaults': False,
                'help': 'Inputs for the `PwRelaxWorkChain`, if not specified at all, the relaxation step is skipped.'})
        spec.expose_inputs(PwBaseWorkChain, namespace='scf', exclude=('clean_workdir', 'pw.structure'),
            namespace_options={'help': 'Inputs for the `PwBaseWorkChain` for the SCF calculation.'})
        spec.expose_inputs(PwBaseWorkChain, namespace='nscf', exclude=('clean_workdir', 'pw.structure'),
            namespace_options={'help': 'Inputs for the `PwBaseWorkChain` for the NSCF calculation.'})
        spec.expose_inputs(ProjwfcCalculation, namespace='projwfc', exclude=('parent_folder', ),
            namespace_options={'required': False,
                'help': 'Inputs for the `ProjwfcCalculation` for the Projwfc calculation.'})
        spec.expose_inputs(Pw2wannier90Calculation, namespace='pw2wannier90', exclude=('parent_folder', 'nnkp_file'),
            namespace_options={'help': 'Inputs for the `Pw2wannier90Calculation` for the pw2wannier90 calculation.'})
        spec.expose_inputs(Wannier90Calculation, namespace='wannier90', exclude=('structure', ),
            namespace_options={'help': 'Inputs for the `Wannier90Calculation` for the Wannier90 calculation.'})

        spec.outline(
            cls.setup,
            if_(cls.should_run_relax)(
                cls.run_relax,
                cls.inspect_relax),
            cls.run_scf,
            cls.inspect_scf,
            cls.run_nscf,
            cls.inspect_nscf,
            if_(cls.should_run_projwfc)(
                cls.run_projwfc,
                cls.inspect_projwfc),
            cls.run_wannier90_pp,
            cls.inspect_wannier90_pp,
            cls.run_pw2wannier90,
            cls.inspect_pw2wannier90,
            cls.run_wannier90,
            cls.inspect_wannier90,
            cls.results
        )

        spec.expose_outputs(PwRelaxWorkChain, namespace='relax', namespace_options={'required': False})
        spec.expose_outputs(PwBaseWorkChain, namespace='scf')
        # here nscf is optional, since the subclass Wannier90OpengridWorkChain might skip nscf step.
        spec.expose_outputs(PwBaseWorkChain, namespace='nscf', namespace_options={'required': False})
        spec.expose_outputs(ProjwfcCalculation, namespace='projwfc', namespace_options={'required': False})
        spec.expose_outputs(Pw2wannier90Calculation, namespace='pw2wannier90')
        spec.expose_outputs(Wannier90BaseWorkChain, namespace='wannier90_pp')
        spec.expose_outputs(Wannier90Calculation, namespace='wannier90')

        spec.exit_code(401, 'ERROR_SUB_PROCESSS_FAILED_SETUP', message='setup failed, check your input')
        spec.exit_code(402, 'ERROR_SUB_PROCESS_FAILED_RELAX', message='the PwRelaxWorkChain sub process failed')
        spec.exit_code(403, 'ERROR_SUB_PROCESS_FAILED_SCF', message='the scf PwBasexWorkChain sub process failed')
        spec.exit_code(404, 'ERROR_SUB_PROCESS_FAILED_NSCF', message='the nscf PwBasexWorkChain sub process failed')
        spec.exit_code(405, 'ERROR_SUB_PROCESS_FAILED_PROJWFC', message='the ProjwfcCalculation sub process failed')
        spec.exit_code(406, 'ERROR_SUB_PROCESS_FAILED_WANNIER90PP', message='the postproc Wannier90Calculation sub process failed')
        spec.exit_code(407, 'ERROR_SUB_PROCESS_FAILED_PW2WANNIER90', message='the Pw2wannier90Calculation sub process failed')
        spec.exit_code(408, 'ERROR_SUB_PROCESS_FAILED_WANNIER90', message='the Wannier90Calculation sub process failed')

    def setup(self):
        """Define the current structure in the context to be the input structure."""
        self.ctx.current_structure = self.inputs.structure

        inputs = AttributeDict(self.exposed_inputs(Wannier90Calculation, namespace='wannier90'))
        parameters = inputs.parameters.get_dict()

        self.ctx.auto_projections = parameters.get('auto_projections', False)

        # check bands_plot kpoint_path
        bands_plot = parameters.get('bands_plot', False)
        if bands_plot:
            kpoint_path = inputs.get('kpoint_path', None)
            if kpoint_path is None:
                self.report('bands_plot is required but no kpoint_path provided')
                return self.exit_codes.ERROR_SUB_PROCESSS_FAILED_SETUP

    def should_run_relax(self):
        """If the 'relax' input namespace was specified, we relax the input structure."""
        return 'relax' in self.inputs

    def run_relax(self):
        """Run the PwRelaxWorkChain to run a relax calculation"""
        inputs = AttributeDict(self.exposed_inputs(PwRelaxWorkChain, namespace='relax'))
        inputs.structure = self.ctx.current_structure
        inputs.metadata.call_link_label = 'relax'

        inputs = prepare_process_inputs(PwRelaxWorkChain, inputs)
        running = self.submit(PwRelaxWorkChain, **inputs)
        self.report(f'launching {running.process_label}<{running.pk}>')
        return ToContext(workchain_relax=running)

    def inspect_relax(self):
        """verify that the PwRelaxWorkChain successfully finished."""
        workchain = self.ctx.workchain_relax

        if not workchain.is_finished_ok:
            self.report(f'{workchain.process_label} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_RELAX

        self.ctx.current_structure = workchain.outputs.output_structure

    def run_scf(self):
        """Run the PwBaseWorkChain in scf mode on the primitive cell of (optionally relaxed) input structure."""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='scf'))
        inputs.pw.structure = self.ctx.current_structure
        inputs.metadata.call_link_label = 'scf'

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'scf step - launching {running.process_label}<{running.pk}>')
        return ToContext(workchain_scf=running)

    def inspect_scf(self):
        """Verify that the PwBaseWorkChain for the scf run successfully finished."""
        workchain = self.ctx.workchain_scf

        if not workchain.is_finished_ok:
            self.report(f'scf {workchain.process_label} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_SCF

        self.ctx.current_folder = workchain.outputs.remote_folder
        self.report(f"scf {workchain.process_label} successfully finished")

    def run_nscf(self):
        """Run the PwBaseWorkChain in nscf mode"""
        inputs = AttributeDict(self.exposed_inputs(PwBaseWorkChain, namespace='nscf'))
        inputs.pw.structure = self.ctx.current_structure
        inputs.pw.parent_folder = self.ctx.current_folder
        inputs.metadata.call_link_label = 'nscf'

        inputs = prepare_process_inputs(PwBaseWorkChain, inputs)
        running = self.submit(PwBaseWorkChain, **inputs)
        self.report(f'nscf step - launching {running.process_label}<{running.pk}>')
        return ToContext(workchain_nscf=running)

    def inspect_nscf(self):
        """Verify that the PwBaseWorkChain for the nscf run successfully finished."""
        workchain = self.ctx.workchain_nscf

        if not workchain.is_finished_ok:
            self.report(f'nscf {workchain.process_label} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_NSCF

        self.ctx.current_folder = workchain.outputs.remote_folder
        self.report(f"nscf {workchain.process_label} successfully finished")

    def should_run_projwfc(self):
        """If the 'auto_projections = true' && only_valence, we run projwfc calculation to extract SCDM mu & sigma."""
        inputs = AttributeDict(self.exposed_inputs(Pw2wannier90Calculation, namespace='pw2wannier90'))
        parameters = inputs.parameters.get_dict()
        scdm_entanglement = parameters.get('inputpp', {}).get('scdm_entanglement', None)
        scdm_mu = parameters.get('inputpp', {}).get('scdm_mu', None)
        scdm_sigma = parameters.get('inputpp', {}).get('scdm_sigma', None)

        if not self.ctx.auto_projections:
            return False

        if scdm_entanglement == 'erfc':
            if scdm_mu is not None and scdm_sigma is not None:
                self.report("found scdm_mu & scdm_sigma in input, skip projwfc calculation.")
                return False
            else:
                self.report("SCDM mu & sigma are auto-set using projectability.")
                return True
        elif scdm_entanglement == 'isolated':
            return False
        elif scdm_entanglement == 'gaussian':
            if scdm_mu is None or scdm_sigma is None:
                self.report("scdm_entanglement is gaussian but scdm_mu or scdm_sigma is empty.")
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PW2WANNIER90
            else:
                return False

    def run_projwfc(self):
        """Projwfc step"""
        inputs = AttributeDict(self.exposed_inputs(ProjwfcCalculation, namespace='projwfc'))
        inputs.parent_folder = self.ctx.current_folder
        inputs.metadata.call_link_label = 'projwfc'

        inputs = prepare_process_inputs(ProjwfcCalculation, inputs)
        running = self.submit(ProjwfcCalculation, **inputs)
        self.report(f'projwfc step - launching {running.process_label}<{running.pk}>')
        return ToContext(calc_projwfc=running)

    def inspect_projwfc(self):
        """Verify that the ProjwfcCalculation for the projwfc run successfully finished."""
        calculation = self.ctx.calc_projwfc

        if not calculation.is_finished_ok:
            self.report(f'{calculation.process_label} failed with exit status {calculation.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PROJWFC

        # no need to set current_folder, because the next place which needs a parent_folder
        # is the pw2wannier90 calculation, which needs the remote_folder of the nscf calculation.
        # So the current_folder is kept to the remote_folder of nscf calculation,
        # or in the Wannier90OpengridWorkChain, the current_folder is set as the remote_folder
        # of open_grid calculation.
        # self.ctx.current_folder = calculation.outputs.remote_folder
        self.report(f"projwfc {calculation.process_label} successfully finished")

    def prepare_wannier90_inputs(self):
        """The input of wannier90 calculation is build here.
        Here it is separated out from `run_wannier90_pp`, so it can be overridden by subclasses."""
        inputs = AttributeDict(self.exposed_inputs(Wannier90Calculation, namespace='wannier90'))
        inputs.structure = self.ctx.current_structure
        inputs.metadata.call_link_label = 'wannier_pp'

        # add scf Fermi energy
        scf_output_parameters = self.ctx.workchain_scf.outputs.output_parameters
        if get_fermi_energy(scf_output_parameters) != None:
            args = {
                'wannier_input_parameters': inputs.parameters, 
                'scf_output_parameters': scf_output_parameters,
                'metadata': {'call_link_label': 'update_fermi_energy'}
            }
            inputs.parameters = update_fermi_energy(**args)

        if 'settings' in inputs:
            settings = inputs['settings'].get_dict()
        else:
            settings = {}
        settings['postproc_setup'] = True
        inputs['settings'] = settings
        return inputs

    def run_wannier90_pp(self):
        inputs = self.prepare_wannier90_inputs()
        # use Wannier90BaseWorkChain to automatically handle kmesh_tol related errors
        inputs = prepare_process_inputs(Wannier90BaseWorkChain, {'wannier90': inputs})
        running = self.submit(Wannier90BaseWorkChain, **inputs)
        self.report(f'wannier90 postproc step - launching {running.process_label}<{running.pk}>')
        return ToContext(workchain_wannier90_pp=running)

    def inspect_wannier90_pp(self):
        """Verify that the Wannier90Calculation for the wannier90 run successfully finished."""
        workchain = self.ctx.workchain_wannier90_pp

        if not workchain.is_finished_ok:
            self.report(f'wannier90 postproc {workchain.process_label} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90PP

        # no need to set current_folder, because the following pw2wannier90 calculation
        # relies on the remote_folder of nscf calculation,
        # or that of opengrid calculation, which will be set by the `inspect_opengrid` method.
        # self.ctx.current_folder = workchain.outputs.remote_folder
        self.report(f"wannier90 postproc {workchain.process_label} successfully finished")

    def run_pw2wannier90(self):
        inputs = AttributeDict(self.exposed_inputs(Pw2wannier90Calculation, namespace='pw2wannier90'))
        inputs.metadata.call_link_label = 'pw2wannier90'

        # if 'remote_folder' in self.ctx.workchain_nscf.outputs:
        #     remote_folder = self.ctx.workchain_nscf.outputs.remote_folder
        # else:
        #     self.report('the nscf workchain did not output a remote_folder node!')
        #     return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PW2WANNIER90
        remote_folder = self.ctx.current_folder

        inputs['parent_folder'] = remote_folder
        inputs['nnkp_file'] = self.ctx.workchain_wannier90_pp.outputs.nnkp_file

        if 'calc_projwfc' in self.ctx:
            try:
                args = {
                    'parameters': inputs.parameters,
                    'bands': self.ctx.calc_projwfc.outputs.bands,
                    'projections': self.ctx.calc_projwfc.outputs.projections,
                    'thresholds': self.inputs.get('scdm_thresholds'),
                    'metadata': {'call_link_label': 'update_scdm_mu_sigma'}
                }
                inputs.parameters = update_scdm_mu_sigma(**args)
            except Exception as e:
                self.report(f'update_scdm_mu_sigma failed! {e.args}')
                return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PW2WANNIER90

        inputs = prepare_process_inputs(Pw2wannier90Calculation, inputs)
        running = self.submit(Pw2wannier90Calculation, **inputs)
        self.report(f'pw2wannier90 step - launching {running.process_label}<{running.pk}>')
        return ToContext(calc_pw2wannier90=running)

    def inspect_pw2wannier90(self):
        """Verify that the PwBaseWorkChain for the wannier90 run successfully finished."""
        workchain = self.ctx.calc_pw2wannier90

        if not workchain.is_finished_ok:
            self.report(f'{workchain.process_label} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PW2WANNIER90

        self.ctx.current_folder = workchain.outputs.remote_folder
        self.report(f"{workchain.process_label} successfully finished")

    def run_wannier90(self):
        inputs = AttributeDict(self.exposed_inputs(Wannier90Calculation, namespace='wannier90'))
        inputs.metadata.call_link_label = 'wannier90'

        # if 'remote_folder' in self.ctx.calc_pw2wannier90.outputs:
        #     remote_folder = self.ctx.calc_pw2wannier90.outputs.remote_folder
        # else:
        #     self.report('the Pw2wannier90Calculation did not output a remote_folder node!')
        #     return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90
        # inputs['remote_input_folder'] = remote_folder
        inputs['remote_input_folder'] = self.ctx.current_folder

        # copy postproc inputs
        pp_inputs = self.ctx.workchain_wannier90_pp.get_incoming().nested()['wannier90']
        pp_keys = ['code', 'parameters', 'kpoint_path', 'structure', 'kpoints', 'settings']
        for key in pp_keys:
            inputs[key] = pp_inputs[key]
        # use the Wannier90BaseWorkChain-corrected parameters
        # sort by pk, since the last Wannier90Calculation in Wannier90BaseWorkChain
        # should have the largest pk
        last_calc = max(self.ctx.workchain_wannier90_pp.called, key=lambda calc: calc.pk)
        inputs['parameters'] = last_calc.inputs.parameters

        if 'settings' in inputs:
            settings = inputs.settings.get_dict()
        else:
            settings = {}
        settings['postproc_setup'] = False
        inputs.settings = settings

        inputs = prepare_process_inputs(Wannier90Calculation, inputs)
        running = self.submit(Wannier90Calculation, **inputs)
        self.report(f'wannier90 step - launching {running.process_label}<{running.pk}>')
        return ToContext(calc_wannier90=running)

    def inspect_wannier90(self):
        """Verify that the PwBaseWorkChain for the wannier90 run successfully finished."""
        workchain = self.ctx.calc_wannier90

        if not workchain.is_finished_ok:
            self.report(f'{workchain.process_label} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_WANNIER90

        self.ctx.current_folder = workchain.outputs.remote_folder
        self.report(f"{workchain.process_label} successfully finished")

    def results(self):
        """Attach the desired output nodes directly as outputs of the workchain"""
        if 'workchain_relax' in self.ctx:
            self.out_many(self.exposed_outputs(self.ctx.workchain_relax, PwRelaxWorkChain, namespace='relax'))

        self.out_many(self.exposed_outputs(self.ctx.workchain_scf, PwBaseWorkChain, namespace='scf'))

        # here nscf is optional, since the subclass Wannier90OpengridWorkChain might skip nscf step.
        if 'workchain_nscf' in self.ctx:
            self.out_many(self.exposed_outputs(self.ctx.workchain_nscf, PwBaseWorkChain, namespace='nscf'))

        if 'calc_projwfc' in self.ctx:
            self.out_many(self.exposed_outputs(self.ctx.calc_projwfc, ProjwfcCalculation, namespace='projwfc'))

        self.out_many(self.exposed_outputs(self.ctx.calc_pw2wannier90, Pw2wannier90Calculation, namespace='pw2wannier90'))
        self.out_many(self.exposed_outputs(self.ctx.workchain_wannier90_pp, Wannier90BaseWorkChain, namespace='wannier90_pp'))
        self.out_many(self.exposed_outputs(self.ctx.calc_wannier90, Wannier90Calculation, namespace='wannier90'))
        self.report(f'{self.get_name()} successfully completed')

def get_fermi_energy(output_parameters):
    """get Fermi energy from scf output parameters, unit is eV

    :param output_parameters: scf output parameters
    :type output_parameters: orm.Dict
    :return: if found return Fermi energy, else None
    :rtype: float, None
    """
    out_dict = output_parameters.get_dict()
    fermi = out_dict.get('fermi_energy', None)
    fermi_units = out_dict.get('fermi_energy_units', None)
    if fermi_units != 'eV':
        return None
    else:
        return fermi

@calcfunction
def update_fermi_energy(wannier_input_parameters, scf_output_parameters):
    """Extract Fermi energy from scf calculation and add it to Wannier input parameters,
    also update dis_froz_max if it exists.

    :param wannier_input_parameters: Wannier input parameters
    :type wannier_input_parameters: orm.Dict
    :param scf_output_parameters: scf output parameters
    :type scf_output_parameters: orm.Dict
    """
    params = wannier_input_parameters.get_dict()
    fermi = get_fermi_energy(scf_output_parameters)
    params['fermi_energy'] = fermi
    if 'dis_froz_max' in params:
        params['dis_froz_max'] += fermi
    return orm.Dict(dict=params)

@calcfunction
def update_scdm_mu_sigma(parameters, bands, projections, thresholds):
    """Use erfc fitting to extract scdm_mu & scdm_sigma, and update the pw2wannier90 input parameters.

    :param parameters: pw2wannier90 input parameters
    :type parameters: aiida.orm.Dict
    :param bands: band structure
    :type bands: aiida.orm.BandsData
    :param projections: projectability from projwfc.x
    :type projections: aiida.orm.ProjectionData
    :param thresholds: sigma shift factor
    :type thresholds: aiida.orm.Dict
    """
    parameters_dict = parameters.get_dict()
    mu, sigma = fit_scdm_mu_sigma_aiida(bands, projections, thresholds.get_dict())
    scdm_parameters = dict(scdm_mu=mu, scdm_sigma=sigma)
    parameters_dict['inputpp'].update(scdm_parameters)
    return orm.Dict(dict=parameters_dict)