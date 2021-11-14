# -*- coding: utf-8 -*-
"""Wrapper workchain for Pw2wannier90Calculation to automatically handle several errors."""
from aiida_quantumespresso.calculations.pw2wannier90 import Pw2wannier90Calculation
from aiida.engine import process_handler
from .qebaserestart import QeBaseRestartWorkChain


class Pw2wannier90BaseWorkChain(QeBaseRestartWorkChain):
    """Workchain to run a pw2wannier90 calculation with automated error handling and restarts."""

    _process_class = Pw2wannier90Calculation
    _expose_inputs_namespace = 'pw2wannier90'

    @process_handler(exit_codes=[_process_class.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE])  # pylint: disable=no-member
    def handle_output_stdout_incomplete(self, calculation):
        """Overide parent function."""
        return super().handle_output_stdout_incomplete(calculation)
