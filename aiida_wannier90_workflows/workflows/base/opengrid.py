# -*- coding: utf-8 -*-
"""Wrapper workchain for OpengridCalculation to automatically handle several errors."""
from aiida_quantumespresso.calculations.opengrid import OpengridCalculation
from aiida.engine import process_handler
from .qebaserestart import QeBaseRestartWorkChain


class OpengridBaseWorkChain(QeBaseRestartWorkChain):
    """Workchain to run a opengrid calculation with automated error handling and restarts."""

    _process_class = OpengridCalculation
    _expose_inputs_namespace = 'opengrid'

    @process_handler(exit_codes=[_process_class.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE])  # pylint: disable=no-member
    def handle_output_stdout_incomplete(self, calculation):
        """Overide parent function."""
        return super().handle_output_stdout_incomplete(calculation)
