from aiida import orm
from aiida.common import AttributeDict
from aiida.engine.processes import calcfunction
from aiida.engine.processes import WorkChain, ToContext, if_
from aiida_quantumespresso.utils.mapping import prepare_process_inputs
from aiida_quantumespresso.workflows.pw.base import PwBaseWorkChain
from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain
from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation
from aiida_quantumespresso.calculations.pw2wannier90 import Pw2wannier90Calculation
from aiida_wannier90.calculations import Wannier90Calculation
from aiida_wannier90_workflows.workflows.wannier import Wannier90WorkChain
from aiida_quantumespresso.calculations.opengrid import OpengridCalculation

__all__ = ['Wannier90OpengridWorkChain']

class Wannier90OpengridWorkChain(Wannier90WorkChain):
    """This WorkChain uses open_grid.x to unfold the 
    symmetrized kmesh to a full kmesh in the Wannier90WorkChain.
    The full-kmesh nscf can be avoided.

    2 schemes:
    1. scf w/ symmetry, more nbnd -> open_grid 
       -> pw2wannier90 -> wannier90
    2. scf w/ symmetry, default nbnd -> nscf w/ symm, more nbnd 
       -> open_grid -> pw2wannier90 -> wannier90

    :param Wannier90WorkChain: [description]
    :type Wannier90WorkChain: [type]
    """
    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input('opengrid_only_scf', valid_type=orm.Bool, default=lambda: orm.Bool(True),
            help='If True first do a scf with symmetry and increased number of bands, then open_grid to unfold kmesh; If False first do a scf with symmetry and default number of bands, then a nscf with symmetry and increased number of bands, followed by open_grid.')
        spec.expose_inputs(OpengridCalculation, namespace='opengrid', exclude=('parent_folder', 'structure'), namespace_options={'help': 'Inputs for the `OpengridCalculation`.'})

        spec.outline(
            cls.setup,
            if_(cls.should_run_relax)(
                cls.run_relax,
                cls.inspect_relax),
            cls.run_scf,
            cls.inspect_scf,
            if_(cls.should_run_nscf)(
                cls.run_nscf,
                cls.inspect_nscf),
            cls.run_opengrid,
            cls.inspect_opengrid,
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

        spec.expose_outputs(OpengridCalculation, namespace='opengrid')

        spec.exit_code(451, 'ERROR_SUB_PROCESS_FAILED_PW', message='the scf/nscf WorkChain did not output a remote_folder node')
        spec.exit_code(452, 'ERROR_SUB_PROCESS_FAILED_OPENGRID', message='the OpengridCalculation sub process failed')

    def should_run_nscf(self):
        """
        """
        return not self.inputs.opengrid_only_scf

    def run_opengrid(self):
        """open_grid.x to unfold kmesh
        """
        inputs = AttributeDict(self.exposed_inputs(OpengridCalculation, namespace='opengrid'))
        inputs.metadata.call_link_label = 'opengrid'

        inputs.parent_folder = self.ctx.current_folder
        # ProjwfcCalculation requires the previous Calculation having
        # an input StructureData to parse atomic orbitals.
        inputs.structure = self.ctx.current_structure
        # or do not rely on self.ctx.current_structure
        # if self.should_run_nscf():
        #     previous_workchain = self.ctx.workchain_nscf
        # else:
        #     previous_workchain = self.ctx.workchain_scf
        # try:
        #     remote_folder = previous_workchain.outputs.remote_folder
        #     structure = previous_workchain.get_incoming(node_class=orm.StructureData).one().node
        # except Exception as e:
        #     self.report(f"Exception when running open_grid: {e.args}")
        #     return self.exit_codes.ERROR_SUB_PROCESS_FAILED_PW
        # inputs.parent_folder = remote_folder
        # inputs.structure = structure

        # open_grid.x should not append '_open' to the prefix of QE
        inputs.parameters = {'inputpp': {'overwrite_outdir': True}}

        inputs = prepare_process_inputs(OpengridCalculation, inputs)
        running = self.submit(OpengridCalculation, **inputs)
        self.report(f'open_grid step - launching {running.process_label}<{running.pk}>')
        return ToContext(calc_opengrid=running)

    def inspect_opengrid(self):
        """Verify that the OpengridCalculation run successfully finished."""
        workchain = self.ctx.calc_opengrid

        if not workchain.is_finished_ok:
            self.report(f'{workchain.process_label} failed with exit status {workchain.exit_status}')
            return self.exit_codes.ERROR_SUB_PROCESS_FAILED_OPENGRID

        self.ctx.current_folder = workchain.outputs.remote_folder
        self.report(f"{workchain.process_label} successfully finished")

    def prepare_wannier90_inputs(self):
        """Override the parent method in Wannier90WorkChain.
        Here the kpoints are set as the parsed results from opengrid calculation."""
        inputs = super().prepare_wannier90_inputs()

        inputs.kpoints = self.ctx.calc_opengrid.outputs.kpoints
        parameters = inputs.parameters.get_dict()
        parameters['mp_grid'] = self.ctx.calc_opengrid.outputs.kpoints_mesh.get_kpoints_mesh()[0]
        inputs.parameters = parameters

        self.report('The open_grid.x output kmesh is used as Wannier90 kpoints')
        return inputs

    def results(self):
        """Override"""
        self.out_many(self.exposed_outputs(self.ctx.calc_opengrid, OpengridCalculation, namespace='opengrid'))
        super().results()
