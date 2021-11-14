# -*- coding: utf-8 -*-
"""Wrapper workchain for OpengridCalculation to automatically handle several errors."""
from aiida_quantumespresso.calculations.opengrid import OpengridCalculation
from .qebaserestart import QeBaseRestartWorkChain


class OpengridBaseWorkChain(QeBaseRestartWorkChain):
    """Workchain to run a opengrid calculation with automated error handling and restarts."""

    _process_class = OpengridCalculation
    _expose_inputs_namespace = 'opengrid'
