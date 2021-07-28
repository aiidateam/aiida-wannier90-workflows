# -*- coding: utf-8 -*-
################################################################################
# Copyright (c), AiiDA team and individual contributors.                       #
#  All rights reserved.                                                        #
# This file is part of the AiiDA-wannier90 code.                               #
#                                                                              #
# The code is hosted on GitHub at https://github.com/aiidateam/aiida-wannier90 #
# For further information on the license, see the LICENSE.txt file             #
################################################################################
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

    @classmethod
    def define(cls, spec):
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

        spec.exit_code(
            400, 'ERROR_BVECTORS', message='Unrecoverable bvectors error.'
        )

    def setup(self):
        """Call the `setup` of the `BaseRestartWorkChain` and then create the inputs dictionary in `self.ctx.inputs`.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to submit the calculations in the
        internal loop.
        """
        super().setup()
        self.ctx.inputs = AttributeDict(
            self.exposed_inputs(Wannier90Calculation, 'wannier90')
        )
        self.ctx.kmeshtol_new = [self._WANNIER90_DEFAULT_KMESH_TOL, 1e-8, 1e-4]

    @process_handler(exit_codes=Wannier90Calculation.exit_codes.ERROR_BVECTORS)
    def handle_bvectors(self, calculation):
        """Try to fix Wannier90 bvectors errors by tunning `kmesh_tol`.
        The handler will try to use kmesh_tol = 1e-6, 1e-8, 1e-4.
        """
        KMESHTOL_KEY = 'kmesh_tol'
        parameters = self.ctx.inputs.parameters.get_dict()

        # If the user has specified `kmesh_tol` in the input parameters and it is different
        # from the default, we will first try to use the default `kmesh_tol`.
        if KMESHTOL_KEY in parameters:
            kmeshtol_cur = parameters[KMESHTOL_KEY]
        else:
            kmeshtol_cur = self._WANNIER90_DEFAULT_KMESH_TOL
        if kmeshtol_cur in self.ctx.kmeshtol_new:
            self.ctx.kmeshtol_new.remove(kmeshtol_cur)

        if len(self.ctx.kmeshtol_new) == 0:
            action = 'Unrecoverable bvectors error after several trials of kmesh_tol'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(True, self.exit_codes.ERROR_BVECTORS)

        kmeshtol_new = self.ctx.kmeshtol_new.pop(0)
        parameters[KMESHTOL_KEY] = kmeshtol_new
        action = f'Bvectors error, current kmesh_tol = {kmeshtol_cur}, new kmesh_tol = {kmeshtol_new}'
        self.report_error_handled(calculation, action)
        self.ctx.inputs.parameters = orm.Dict(dict=parameters)

        return ProcessHandlerReport(True)
