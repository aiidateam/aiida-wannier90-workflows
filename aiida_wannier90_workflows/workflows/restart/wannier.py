# -*- coding: utf-8 -*-
"""Wrapper workchain for Wannier90Calculation to automatically handle several errors."""
import os.path
from aiida import orm
from aiida.orm.nodes.data.base import to_aiida_type
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

    _WANNIER90_DEFAULT_WANNIER_PLOT_SUPERCELL = 2

    @classmethod
    def define(cls, spec):
        """Define the process spec."""
        super().define(spec)
        spec.expose_inputs(Wannier90Calculation, namespace='wannier90')

        spec.input(
            'settings',
            valid_type=orm.Dict,
            required=False,
            serializer=to_aiida_type,
            help="""Additional settings, valid keys: `remote_symlink_files`"""
        )

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
        spec.exit_code(402, 'ERROR_PLOT_WF_CUBE', message='Unrecoverable cube format error.')

    def setup(self):
        """Call the `setup` of the `BaseRestartWorkChain` and then create the inputs dictionary in `self.ctx.inputs`.

        This `self.ctx.inputs` dictionary will be used by the `BaseRestartWorkChain` to submit
        the calculations in the internal loop.
        """
        super().setup()

        inputs = AttributeDict(self.exposed_inputs(Wannier90Calculation, 'wannier90'))

        if 'remote_input_folder' in inputs:
            # Note there is an `additional_remote_symlink_list` in Wannier90Calculation.inputs.settings,
            # however it requires user providing a list of
            #   (computer_uuid, remote_input_folder_abs_path, dest_path)
            # This is impossible if we launch a Wannier90Calculation inside a workflow since we don't
            # know the remote_input_folder when setting the inputs of the workflow.
            # Thus I add an `inputs.settings['remote_symlink_files']` to Wannier90BaseWorkChain,
            # which only accepts a list of filenames and generate the full
            # `additional_remote_symlink_list` here.
            remote_input_folder = inputs['remote_input_folder']
            remote_input_folder_path = remote_input_folder.get_remote_path()
            workflow_settings = self.inputs.settings.get_dict()
            calc_settings = inputs.settings.get_dict()
            remote_symlink_list = calc_settings.get('additional_remote_symlink_list', [])
            existed_symlinks = [_[-1] for _ in remote_symlink_list]
            for filename in workflow_settings.get('remote_symlink_files', []):
                if filename in existed_symlinks:
                    continue
                remote_symlink_list.append(
                    (remote_input_folder.computer.uuid, os.path.join(remote_input_folder_path, filename), filename)
                )
            calc_settings['additional_remote_symlink_list'] = remote_symlink_list
            inputs.settings = orm.Dict(dict=calc_settings)

        self.ctx.inputs = inputs

        self.ctx.kmeshtol_new = [self._WANNIER90_DEFAULT_KMESH_TOL, 1e-8, 1e-4]
        self.ctx.disprojmin_multipliers = [0.5, 0.25, 0.125, 0]
        self.ctx.wannier_plot_supercell_new = [4, 6, 8, 10]

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

    @process_handler(exit_codes=[Wannier90Calculation.exit_codes.ERROR_PLOT_WF_CUBE])
    def handle_plot_wf_cube(self, calculation):
        """Try to fix Wannier90 wout error message related to cube format.

        The error message is: 'Error plotting WF cube. Try one of the following:'.

        The handler will try to increase 'wannier_plot_supercell'.
        """
        parameters = self.ctx.inputs.parameters.get_dict()

        current_supercell = parameters.get('wannier_plot_supercell', self._WANNIER90_DEFAULT_WANNIER_PLOT_SUPERCELL)
        # Remove sizes which are smaller equal than current supercell size
        self.ctx.wannier_plot_supercell_new = [_ for _ in self.ctx.wannier_plot_supercell_new if _ > current_supercell]

        if len(self.ctx.wannier_plot_supercell_new) == 0:
            action = 'Unrecoverable error after several trials of wannier_plot_supercell'
            self.report_error_handled(calculation, action)
            return ProcessHandlerReport(True, self.exit_codes.ERROR_PLOT_WF_CUBE)

        new_supercell = self.ctx.wannier_plot_supercell_new.pop(0)
        parameters['wannier_plot_supercell'] = new_supercell

        action = 'Error plotting WFs in cube format, '
        action += f'current wannier_plot_supercell = {current_supercell}, new wannier_plot_supercell = {new_supercell}'
        self.report_error_handled(calculation, action)
        self.ctx.inputs.parameters = orm.Dict(dict=parameters)

        return ProcessHandlerReport(True)

    @process_handler(exit_codes=[Wannier90Calculation.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE])
    def handle_output_stdout_incomplete(self, calculation):
        """Try to fix incomplete stdout error by reducing the number of cores.

        Often the ERROR_OUTPUT_STDOUT_INCOMPLETE is due to out-of-memory.
        The handler will try to set `num_mpiprocs_per_machine` to 1.
        """
        import re

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

        new_num_mpiprocs_per_machine = current_num_mpiprocs_per_machine // 2
        metadata['options']['resources']['num_mpiprocs_per_machine'] = new_num_mpiprocs_per_machine
        action = f'Out-of-memory error, current num_mpiprocs_per_machine = {current_num_mpiprocs_per_machine}'
        action += f', new num_mpiprocs_per_machine = {new_num_mpiprocs_per_machine}'
        self.report_error_handled(calculation, action)
        self.ctx.inputs['metadata'] = metadata

        return ProcessHandlerReport(True)
