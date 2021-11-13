# -*- coding: utf-8 -*-
"""Wrapper workchain for Pw2wannier90Calculation to automatically handle several errors."""
import re
from aiida.common import AttributeDict
from aiida.engine import while_
from aiida.engine import BaseRestartWorkChain
from aiida.engine import process_handler, ProcessHandlerReport
from aiida_quantumespresso.calculations.pw2wannier90 import Pw2wannier90Calculation


class Pw2wannier90BaseWorkChain(BaseRestartWorkChain):
    """Workchain to run a pw2wannier90 calculation with automated error handling and restarts."""

    _process_class = Pw2wannier90Calculation

    @classmethod
    def define(cls, spec):
        """Define the process spec."""
        super().define(spec)
        spec.expose_inputs(cls._process_class, namespace='pw2wannier90')

        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        spec.expose_outputs(cls._process_class)

        spec.exit_code(
            311,
            'ERROR_OUTPUT_STDOUT_INCOMPLETE',
            message='The stdout output file was incomplete probably because the calculation got interrupted.'
        )

    def setup(self):
        """Call the `setup` of the `BaseRestartWorkChain` and then create the inputs dictionary in `self.ctx.inputs`.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to submit
        the calculations in the internal loop.
        """
        super().setup()
        self.ctx.inputs = AttributeDict(self.exposed_inputs(self._process_class, 'pw2wannier90'))

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

    @process_handler(exit_codes=[_process_class.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE])  # pylint: disable=no-member
    def handle_output_stdout_incomplete(self, calculation):
        """Try to fix incomplete stdout error by reducing the number of cores.

        Often the ERROR_OUTPUT_STDOUT_INCOMPLETE is due to out-of-memory.
        The handler will try to set `num_mpiprocs_per_machine` to 1.
        """
        regex = re.compile(r'Detected \d+ oom-kill event\(s\) in step')
        scheduler_stderr = calculation.get_scheduler_stderr()
        for line in scheduler_stderr.split('\n'):
            if regex.search(line) or 'Out Of Memory' in line:
                break
        else:
            action = 'Unrecoverable incomplete stdout error'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(True, self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE)

        metadata = self.ctx.inputs['metadata']
        current_num_mpiprocs_per_machine = metadata['options']['resources'].get('num_mpiprocs_per_machine', 1)
        # num_mpiprocs_per_machine = calculation.attributes['resources'].get('num_mpiprocs_per_machine', 1)

        if current_num_mpiprocs_per_machine == 1:
            action = 'Unrecoverable out-of-memory error after setting num_mpiprocs_per_machine to 1'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(True, self.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE)

        new_num_mpiprocs_per_machine = 1
        metadata['options']['resources']['num_mpiprocs_per_machine'] = new_num_mpiprocs_per_machine
        action = f'Out-of-memory error, current num_mpiprocs_per_machine = {current_num_mpiprocs_per_machine}'
        action += f', new num_mpiprocs_per_machine = {new_num_mpiprocs_per_machine}'
        self.report_error_handled(calculation, action)
        self.ctx.inputs['metadata'] = metadata

        return ProcessHandlerReport(True)
