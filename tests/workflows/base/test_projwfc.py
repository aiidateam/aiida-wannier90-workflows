# -*- coding: utf-8 -*-
"""Tests for the `ProjwfcBaseWorkChain` class."""
import pytest

from plumpy import ProcessState

from aiida.common import AttributeDict
from aiida.engine import ProcessHandlerReport

from aiida_quantumespresso.calculations.projwfc import ProjwfcCalculation

from aiida_wannier90_workflows.workflows.base.projwfc import ProjwfcBaseWorkChain

# pylint: disable=no-member,redefined-outer-name


@pytest.fixture
def generate_inputs_projwfc(fixture_sandbox, fixture_localhost, fixture_code, generate_remote_data):
    """Generate default inputs for a `ProjwfcCalculation."""

    def _generate_inputs_projwfc():
        """Generate default inputs for a `ProjwfcCalculation."""
        from aiida_quantumespresso.utils.resources import get_default_options

        inputs = {
            'code': fixture_code('quantumespresso.projwfc'),
            'parent_folder':
            generate_remote_data(fixture_localhost, fixture_sandbox.abspath, 'quantumespresso.projwfc'),
            'metadata': {
                'options': get_default_options()
            }
        }

        return inputs

    return _generate_inputs_projwfc


@pytest.fixture
def generate_workchain_projwfc(generate_workchain, generate_inputs_projwfc, fixture_localhost, generate_calc_job_node):
    """Generate an instance of a `ProjwfcBaseWorkChain`."""

    def _generate_workchain_projwfc(exit_code=None, inputs=None, test_name=None):
        entry_point = 'wannier90_workflows.base.projwfc'
        if not inputs:
            inputs = {'projwfc': generate_inputs_projwfc()}
        process = generate_workchain(entry_point, inputs)

        if exit_code is not None:
            entry_point_calc_job = 'quantumespresso.projwfc'
            node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, test_name, inputs['projwfc'])
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(exit_code.status)

            process.ctx.iteration = 1
            process.ctx.children = [node]

        return process

    return _generate_workchain_projwfc


def test_setup(generate_workchain_projwfc):
    """Test `ProjwfcBaseWorkChain.setup`."""
    process = generate_workchain_projwfc()
    process.setup()

    assert isinstance(process.ctx.inputs, AttributeDict)


@pytest.mark.parametrize('npool_value', (
    4,
    2,
))
@pytest.mark.parametrize('npool_key', (
    '-nk',
    '-npools',
))
def test_handle_output_stdout_incomplete(generate_workchain_projwfc, generate_inputs_projwfc, npool_key, npool_value):
    """Test `ProjwfcBaseWorkChain.handle_output_stdout_incomplete` for restarting from OOM."""
    from aiida import orm

    inputs = {'projwfc': generate_inputs_projwfc()}
    # E.g. when number of MPI procs = 4, the next trial is 2
    inputs['projwfc']['metadata']['options'] = {
        'resources': {
            'num_machines': 1,
            'num_mpiprocs_per_machine': npool_value
        },
        'max_wallclock_seconds': 3600,
        'withmpi': True,
        'scheduler_stderr': '_scheduler-stderr.txt'
    }
    inputs['projwfc']['settings'] = orm.Dict(dict={'cmdline': [npool_key, f'{npool_value}']})
    process = generate_workchain_projwfc(
        exit_code=ProjwfcCalculation.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE,
        inputs=inputs,
        test_name='out_of_memory'
    )
    process.setup()

    # Direct call to the handler
    result = process.handle_output_stdout_incomplete(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert result.do_break
    assert result.exit_code.status == 0

    new_npool_value = npool_value // 2
    assert process.ctx.inputs['metadata']['options']['resources']['num_mpiprocs_per_machine'] == new_npool_value
    assert process.ctx.inputs['settings']['cmdline'] == [npool_key, f'{new_npool_value}']

    # The `inspect_process` will call again the `handle_output_stdout_incomplete` because the
    # `ERROR_OUTPUT_STDOUT_INCOMPLETE` exit code is still there.
    result = process.inspect_process()
    new_npool_value = npool_value // 4
    if new_npool_value == 0:
        assert result == ProjwfcBaseWorkChain.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE
        new_npool_value = 1
    else:
        assert result.status == 0
    assert process.ctx.inputs['metadata']['options']['resources']['num_mpiprocs_per_machine'] == new_npool_value
    assert process.ctx.inputs['settings']['cmdline'] == [npool_key, f'{new_npool_value}']
