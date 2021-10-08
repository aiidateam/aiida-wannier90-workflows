# -*- coding: utf-8 -*-
"""Wrapper workchain for Wannier90Calculation to automatically handle several errors."""
from aiida import orm
from aiida.common import AttributeDict
from aiida.engine import while_
from aiida.engine import BaseRestartWorkChain
from aiida.engine import process_handler, ProcessHandlerReport
from aiida_wannier90.calculations import Wannier90Calculation


class Wannier90BaseWorkChain(BaseRestartWorkChain):
    """Workchain to run a wannier90 calculation with automated error handling and restarts."""

    _process_class = Wannier90Calculation

    _WANNIER90_DEFAULT_KMESH_TOL = 1e-6
    _WANNIER90_DEFAULT_DIS_PROJ_MIN = 0.1
    _WANNIER90_DEFAULT_DIS_PROJ_MAX = 0.9

    @classmethod
    def define(cls, spec):
        """Define the process spec."""
        super().define(spec)
        spec.expose_inputs(Wannier90Calculation, namespace='wannier90')

        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        spec.expose_outputs(Wannier90Calculation)

        spec.exit_code(400, 'ERROR_BVECTORS', message='Unrecoverable bvectors error.')
        spec.exit_code(401, 'ERROR_DISENTANGLEMENT_NOT_ENOUGH_STATES', message='Unrecoverable disentanglement error.')

    def setup(self):
        """Call the `setup` of the `BaseRestartWorkChain` and then create the inputs dictionary in `self.ctx.inputs`.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to submit
        the calculations in the internal loop.
        """
        super().setup()
        self.ctx.inputs = AttributeDict(self.exposed_inputs(Wannier90Calculation, 'wannier90'))
        self.ctx.kmeshtol_new = [self._WANNIER90_DEFAULT_KMESH_TOL, 1e-8, 1e-4]
        self.ctx.disprojmin_multipliers = [0.5, 0.25, 0.125]

    def report_error_handled(self, calculation, action):
        """Report an action taken for a calculation that has failed.

        This should be called in a registered error handler if its condition is met and an action was taken.

        :param calculation: the failed calculation node
        :param action: a string message with the action taken
        """
        message = f'{calculation.process_label}<{calculation.pk}> failed'
        message += f' with exit status {calculation.exit_status}: {calculation.exit_message}'
        self.report(message)
        self.report(f'Action taken: {action}')

    @process_handler(exit_codes=[Wannier90Calculation.exit_codes.ERROR_BVECTORS])
    def handle_bvectors(self, calculation):
        """Try to fix Wannier90 bvectors errors by tunning `kmesh_tol`.

        The handler will try to use kmesh_tol = 1e-6, 1e-8, 1e-4.
        """
        parameters = self.ctx.inputs.parameters.get_dict()

        # If the user has specified `kmesh_tol` in the input parameters and it is different
        # from the default, we will first try to use the default `kmesh_tol`.
        current_kmeshtol = parameters.get('kmesh_tol', self._WANNIER90_DEFAULT_KMESH_TOL)
        if current_kmeshtol in self.ctx.kmeshtol_new:
            self.ctx.kmeshtol_new.remove(current_kmeshtol)

        if len(self.ctx.kmeshtol_new) == 0:
            action = 'Unrecoverable bvectors error after several trials of kmesh_tol'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(True, self.exit_codes.ERROR_BVECTORS)

        new_kmeshtol = self.ctx.kmeshtol_new.pop(0)
        parameters['kmesh_tol'] = new_kmeshtol
        action = f'Bvectors error, current kmesh_tol = {current_kmeshtol}, new kmesh_tol = {new_kmeshtol}'
        self.report_error_handled(calculation, action)
        self.ctx.inputs.parameters = orm.Dict(dict=parameters)

        return ProcessHandlerReport(True)

    @process_handler(exit_codes=[Wannier90Calculation.exit_codes.ERROR_DISENTANGLEMENT_NOT_ENOUGH_STATES])
    def handle_disentanglement_not_enough_states(self, calculation):
        """Try to fix Wannier90 wout error message related to projectability disentanglement.

        The error message is: 'Energy window contains fewer states than number of target WFs,
        consider reducing dis_proj_min/increasing dis_win_max?'.

        The handler will try to use decrease 'dis_proj_min' to allow for more states for disentanglement.
        """
        parameters = self.ctx.inputs.parameters.get_dict()
        if 'dis_proj_min' not in parameters and 'dis_proj_max' not in parameters:
            # If neither is present, I should never encounter this exit_code
            action = 'Unrecoverable bvectors error: the error handler is only for projectability disentanglement'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(True, self.exit_codes.ERROR_DISENTANGLEMENT_NOT_ENOUGH_STATES)

        if len(self.ctx.disprojmin_multipliers) == 0:
            action = 'Unrecoverable error after several trials of dis_proj_min'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(True, self.exit_codes.ERROR_DISENTANGLEMENT_NOT_ENOUGH_STATES)

        current_disprojmin = parameters.get('dis_proj_min', self._WANNIER90_DEFAULT_KMESH_TOL)
        multiplier = self.ctx.disprojmin_multipliers.pop(0)
        new_disprojmin = current_disprojmin * multiplier
        parameters['dis_proj_min'] = new_disprojmin

        action = 'Not enough states for disentanglement, '
        action += f'current dis_proj_min = {current_disprojmin}, new dis_proj_min = {new_disprojmin}'
        self.report_error_handled(calculation, action)
        self.ctx.inputs.parameters = orm.Dict(dict=parameters)

        return ProcessHandlerReport(True)
