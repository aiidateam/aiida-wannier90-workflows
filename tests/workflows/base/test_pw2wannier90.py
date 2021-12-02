# -*- coding: utf-8 -*-
"""Tests for the `Pw2wannier90BaseWorkChain` class."""
from pathlib import Path
import pytest

from plumpy import ProcessState

from aiida.common import AttributeDict
from aiida.engine import ProcessHandlerReport

from aiida_quantumespresso.calculations.pw2wannier90 import Pw2wannier90Calculation

from aiida_wannier90_workflows.workflows.base.pw2wannier90 import Pw2wannier90BaseWorkChain

# pylint: disable=no-member,redefined-outer-name


@pytest.fixture
def generate_inputs_pw2wannier90(
    filepath_fixtures, fixture_sandbox, fixture_localhost, fixture_code, generate_remote_data
):
    """Generate default inputs for a `Pw2wannier90Calculation."""

    def _generate_inputs_pw2wannier90():
        """Generate default inputs for a `Pw2wannier90Calculation."""
        from aiida import orm
        from aiida_quantumespresso.utils.resources import get_default_options

        nnkp_filepath = Path(filepath_fixtures) / 'calcjob' / 'pw2wannier90' / 'out_of_memory' / 'aiida.nnkp'

        inputs = {
            'code':
            fixture_code('quantumespresso.pw2wannier90'),
            'parameters':
            orm.Dict(dict={'inputpp': {}}),
            'nnkp_file':
            orm.SinglefileData(file=nnkp_filepath).store(),
            'parent_folder':
            generate_remote_data(fixture_localhost, fixture_sandbox.abspath, 'quantumespresso.pw2wannier90'),
            'metadata': {
                'options': get_default_options()
            }
        }

        return inputs

    return _generate_inputs_pw2wannier90


@pytest.fixture
def generate_workchain_pw2wannier90(
    generate_workchain, generate_inputs_pw2wannier90, fixture_localhost, generate_calc_job_node
):
    """Generate an instance of a `Pw2wannier90BaseWorkChain`."""

    def _generate_workchain_pw2wannier90(exit_code=None, inputs=None, test_name=None):
        entry_point = 'wannier90_workflows.base.pw2wannier90'
        if not inputs:
            inputs = {'pw2wannier90': generate_inputs_pw2wannier90()}
        process = generate_workchain(entry_point, inputs)

        if exit_code is not None:
            entry_point_calc_job = 'quantumespresso.pw2wannier90'
            node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, test_name, inputs['pw2wannier90'])
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(exit_code.status)

            process.ctx.iteration = 1
            process.ctx.children = [node]

        return process

    return _generate_workchain_pw2wannier90


def test_setup(generate_workchain_pw2wannier90):
    """Test `Pw2wannier90BaseWorkChain.setup`."""
    process = generate_workchain_pw2wannier90()
    process.setup()

    assert isinstance(process.ctx.inputs, AttributeDict)


def test_prepare_inputs(
    generate_inputs_pw2wannier90, generate_workchain_pw2wannier90, generate_bands_data, generate_projection_data
):
    """Test `Pw2wannier90BaseWorkChain.prepare_inputs`."""
    from aiida.orm import Dict

    inputs = generate_inputs_pw2wannier90()
    parameters = inputs['parameters'].get_dict()['inputpp']

    # Test SCDM fitting is working
    parameters['scdm_proj'] = True
    parameters['scdm_entanglement'] = 'erfc'
    inputs['parameters'] = Dict(dict={'inputpp': parameters})

    inputs = {'pw2wannier90': inputs}
    inputs['bands'] = generate_bands_data()
    inputs['bands_projections'] = generate_projection_data()

    process = generate_workchain_pw2wannier90(inputs=inputs)
    inputs = process.prepare_inputs()

    assert isinstance(inputs, AttributeDict)

    parameters = inputs['parameters'].get_dict()['inputpp']
    assert 'scdm_mu' in parameters, parameters
    assert 'scdm_sigma' in parameters, parameters
    assert abs(parameters['scdm_mu'] - 6.023033662603666) < 1e-5, parameters
    assert abs(parameters['scdm_sigma'] - 0.21542103913166902) < 1e-5, parameters


@pytest.mark.parametrize('npool_value', (
    4,
    2,
))
@pytest.mark.parametrize('npool_key', (
    '-nk',
    '-npools',
))
def test_handle_output_stdout_incomplete(
    generate_workchain_pw2wannier90, generate_inputs_pw2wannier90, npool_key, npool_value
):
    """Test `Pw2wannier90BaseWorkChain.handle_output_stdout_incomplete` for restarting from OOM."""
    from aiida import orm

    inputs = {'pw2wannier90': generate_inputs_pw2wannier90()}
    # E.g. when number of MPI procs = 4, the next trial is 2
    inputs['pw2wannier90']['metadata']['options'] = {
        'resources': {
            'num_machines': 1,
            'num_mpiprocs_per_machine': npool_value
        },
        'max_wallclock_seconds': 3600,
        'withmpi': True,
        'scheduler_stderr': '_scheduler-stderr.txt'
    }
    inputs['pw2wannier90']['settings'] = orm.Dict(dict={'cmdline': [npool_key, f'{npool_value}']})
    process = generate_workchain_pw2wannier90(
        exit_code=Pw2wannier90Calculation.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE,
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
        assert result == Pw2wannier90BaseWorkChain.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE
        new_npool_value = 1
    else:
        assert result.status == 0
    assert process.ctx.inputs['metadata']['options']['resources']['num_mpiprocs_per_machine'] == new_npool_value
    assert process.ctx.inputs['settings']['cmdline'] == [npool_key, f'{new_npool_value}']
