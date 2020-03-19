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
        super(Wannier90BaseWorkChain, cls).define(spec)
        spec.expose_inputs(Wannier90Calculation, namespace='wannier90')
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                # cls.prepare_inputs,
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results
        )
        spec.expose_outputs(Wannier90Calculation)

        spec.exit_code(
            400,
            'ERROR_KMESH_TOL_FAILED',
            message='Not enough bvectors found even after reducing kmesh_tol.'
        )

    def setup(self):
        """Call the `setup` of the `BaseRestartWorkChain` and then create the inputs dictionary in `self.ctx.inputs`.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to submit the calculations in the
        internal loop.
        """
        super(Wannier90BaseWorkChain, self).setup()
        self.ctx.inputs = AttributeDict(
            self.exposed_inputs(Wannier90Calculation, 'wannier90')
        )
        self.ctx.kmesh_tol_trails = [
            self._WANNIER90_DEFAULT_KMESH_TOL, 1e-8, 1e-4
        ]

    # def prepare_inputs(self):
    # if self.ctx.iteration == 0:
    #     return
    # elif self.ctx.iteration == 1:
    #     # fix: Not enough bvectors found
    #     self.ctx.inputs.parameters['kmesh_tol'] = 1e-8
    #     return

    @process_handler(
        exit_codes=Wannier90Calculation.exit_codes.
        ERROR_EXITING_MESSAGE_IN_STDOUT
    )
    def handle_kemsh_tol(self, calc):
        """Try fixing wannier90 error message: Not enough bvectors found.
        Will try to reduce kmesh_tol to 1e-8.
        
        :param calc: This is the process node that finished and is to be investigated
        :type calc: 
        :return: [description]
        :rtype: [type]
        """
        kmesh_tol_key = 'kmesh_tol'
        error_msg = calc.outputs.output_parameters['error_msg']
        kmesh_tol_error_msg = 'kmesh_get_bvector: Not enough bvectors found'
        is_kmesh_tol = any(kmesh_tol_error_msg in line for line in error_msg)
        too_many_bvec_msg = 'kmesh_get: something wrong, found too many nearest neighbours'
        is_kmesh_tol |= any(too_many_bvec_msg in line for line in error_msg)
        b1_error_msg = 'Unable to satisfy B1'
        is_kmesh_tol |= any(b1_error_msg in line for line in error_msg)
        if is_kmesh_tol:
            parameters = self.ctx.inputs.parameters.get_dict()
            if kmesh_tol_key in parameters:
                current_kmesh_tol = parameters[kmesh_tol_key]
            else:
                current_kmesh_tol = self._WANNIER90_DEFAULT_KMESH_TOL
            if current_kmesh_tol in self.ctx.kmesh_tol_trails:
                self.ctx.kmesh_tol_trails.remove(current_kmesh_tol)
            if len(self.ctx.kmesh_tol_trails) == 0:
                self.report(
                    'Not enough bvectors found after several trials of kmesh_tol'
                )
                return ProcessHandlerReport(
                    exit_code=self.exit_codes.ERROR_KMESH_TOL_FAILED
                )
            new_kmesh_tol = self.ctx.kmesh_tol_trails.pop(0)
            parameters[kmesh_tol_key] = new_kmesh_tol
            self.report(
                'Not enough bvectors found, previous kmesh_tol: {}, trying a new kmesh_tol: {}'
                .format(current_kmesh_tol, new_kmesh_tol)
            )
            self.ctx.inputs.parameters = orm.Dict(dict=parameters)
            return ProcessHandlerReport()
        return None
