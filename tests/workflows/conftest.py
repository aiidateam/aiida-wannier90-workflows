# -*- coding: utf-8 -*-
"""Fixtures for testing workflows."""
from pathlib import Path
import pytest

from plumpy import ProcessState

# pylint: disable=redefined-outer-name,too-many-statements


@pytest.fixture
def generate_inputs_projwfc_base(fixture_sandbox, fixture_localhost, fixture_code, generate_remote_data):
    """Generate default inputs for a `ProjwfcCalculation."""

    def _generate_inputs_projwfc_base():
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

    return _generate_inputs_projwfc_base


@pytest.fixture
def generate_workchain_projwfc_base(
    generate_workchain, generate_inputs_projwfc_base, fixture_localhost, generate_calc_job_node
):
    """Generate an instance of a `ProjwfcBaseWorkChain`."""

    def _generate_workchain_projwfc_base(exit_code=None, inputs=None, test_name=None):
        entry_point = 'wannier90_workflows.base.projwfc'
        if not inputs:
            inputs = {'projwfc': generate_inputs_projwfc_base()}
        process = generate_workchain(entry_point, inputs)

        if exit_code is not None:
            entry_point_calc_job = 'quantumespresso.projwfc'
            node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, test_name, inputs['projwfc'])
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(exit_code.status)

            process.ctx.iteration = 1
            process.ctx.children = [node]

        return process

    return _generate_workchain_projwfc_base


@pytest.fixture
def generate_inputs_opengrid_base(fixture_sandbox, fixture_localhost, fixture_code, generate_remote_data):
    """Generate default inputs for a `OpengridCalculation."""

    def _generate_inputs_opengrid_base():
        """Generate default inputs for a `OpengridCalculation."""
        from aiida_quantumespresso.utils.resources import get_default_options

        inputs = {
            'code': fixture_code('quantumespresso.opengrid'),
            'parent_folder':
            generate_remote_data(fixture_localhost, fixture_sandbox.abspath, 'quantumespresso.opengrid'),
            'metadata': {
                'options': get_default_options()
            }
        }

        return inputs

    return _generate_inputs_opengrid_base


@pytest.fixture
def generate_workchain_opengrid_base(
    generate_workchain, generate_inputs_opengrid_base, fixture_localhost, generate_calc_job_node
):
    """Generate an instance of a `OpengridBaseWorkChain`."""

    def _generate_workchain_opengrid_base(exit_code=None, inputs=None, test_name=None):
        entry_point = 'wannier90_workflows.base.opengrid'
        if not inputs:
            inputs = {'opengrid': generate_inputs_opengrid_base()}
        process = generate_workchain(entry_point, inputs)

        if exit_code is not None:
            entry_point_calc_job = 'quantumespresso.opengrid'
            node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, test_name, inputs['opengrid'])
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(exit_code.status)

            process.ctx.iteration = 1
            process.ctx.children = [node]

        return process

    return _generate_workchain_opengrid_base


@pytest.fixture
def generate_inputs_pw2wannier90_base(
    filepath_fixtures, fixture_sandbox, fixture_localhost, fixture_code, generate_remote_data
):
    """Generate default inputs for a `Pw2wannier90Calculation."""

    def _generate_inputs_pw2wannier90_base():
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

    return _generate_inputs_pw2wannier90_base


@pytest.fixture
def generate_workchain_pw2wannier90_base(
    generate_workchain, generate_inputs_pw2wannier90_base, fixture_localhost, generate_calc_job_node
):
    """Generate an instance of a `Pw2wannier90BaseWorkChain`."""

    def _generate_workchain_pw2wannier90_base(exit_code=None, inputs=None, test_name=None):
        entry_point = 'wannier90_workflows.base.pw2wannier90'
        if not inputs:
            inputs = {'pw2wannier90': generate_inputs_pw2wannier90_base()}
        process = generate_workchain(entry_point, inputs)

        if exit_code is not None:
            entry_point_calc_job = 'quantumespresso.pw2wannier90'
            node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, test_name, inputs['pw2wannier90'])
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(exit_code.status)

            process.ctx.iteration = 1
            process.ctx.children = [node]

        return process

    return _generate_workchain_pw2wannier90_base


@pytest.fixture
def generate_inputs_wannier90_base(
    fixture_sandbox, fixture_localhost, fixture_code, generate_remote_data, generate_structure
):
    """Generate default inputs for a `Wannier90Calculation."""

    def _generate_inputs_wannier90_base():
        """Generate default inputs for a `Wannier90Calculation."""
        from aiida import orm
        from aiida_quantumespresso.utils.resources import get_default_options

        inputs = {
            'code': fixture_code('wannier90.wannier90'),
            'structure': generate_structure(),
            'parameters': orm.Dict(),
            'kpoints': orm.KpointsData(),
            'remote_input_folder':
            generate_remote_data(fixture_localhost, fixture_sandbox.abspath, 'wannier90.wannier90'),
            'metadata': {
                'options': get_default_options()
            }
        }

        return inputs

    return _generate_inputs_wannier90_base


@pytest.fixture
def generate_workchain_wannier90_base(
    generate_workchain, generate_inputs_wannier90_base, fixture_localhost, generate_calc_job_node
):
    """Generate an instance of a `Wannier90BaseWorkChain`."""

    def _generate_workchain_wannier90_base(exit_code=None, inputs=None, test_name=None):
        entry_point = 'wannier90_workflows.base.wannier90'
        if not inputs:
            inputs = {'wannier90': generate_inputs_wannier90_base()}
        process = generate_workchain(entry_point, inputs)

        if exit_code is not None:
            entry_point_calc_job = 'wannier90.wannier90'
            node = generate_calc_job_node(entry_point_calc_job, fixture_localhost, test_name, inputs['wannier90'])
            node.set_process_state(ProcessState.FINISHED)
            node.set_exit_status(exit_code.status)

            process.ctx.iteration = 1
            process.ctx.children = [node]

        return process

    return _generate_workchain_wannier90_base


@pytest.fixture
def generate_builder_inputs(fixture_code, generate_structure):
    """Generate a set of default inputs for the ``Wannier90BandsWorkChain.get_builder_from_protocol()`` method."""

    def _generate_builder_inputs(structure_id='Si'):
        inputs = {
            'codes': {
                'pw': fixture_code('quantumespresso.pw'),
                'pw2wannier90': fixture_code('quantumespresso.pw2wannier90'),
                'wannier90': fixture_code('wannier90.wannier90'),
                'projwfc': fixture_code('quantumespresso.projwfc'),
                'opengrid': fixture_code('quantumespresso.opengrid'),
            },
            'structure': generate_structure(structure_id=structure_id)
        }
        return inputs

    return _generate_builder_inputs


@pytest.fixture
def generate_inputs_wannier90(generate_inputs_pw, fixture_code):
    """Generate default inputs for a `Wannier90WorkChain."""
    from aiida.orm import Dict
    from aiida_quantumespresso.utils.resources import get_default_options

    def _generate_inputs_wannier90():
        """Generate default inputs for a `Wannier90WorkChain."""
        scf_pw_inputs = generate_inputs_pw()
        kpoints = scf_pw_inputs.pop('kpoints')
        structure = scf_pw_inputs.pop('structure')
        scf = {'pw': scf_pw_inputs, 'kpoints': kpoints}

        # Copy it, note I cannot call `generate_inputs_pw` twice
        nscf_pw_inputs = dict(scf_pw_inputs.items())
        params = nscf_pw_inputs['parameters'].get_dict()
        params['CONTROL']['calculation'] = 'nscf'
        params['SYSTEM']['nosym'] = True
        nscf_pw_inputs['parameters'] = Dict(dict=params)
        nscf = {'pw': nscf_pw_inputs, 'kpoints': kpoints}

        projwfc_params = {'projwfc': {'deltae': 0.01}}
        projwfc = {
            'code': fixture_code('quantumespresso.projwfc'),
            'parameters': Dict(dict=projwfc_params),
            'metadata': {
                'options': get_default_options()
            }
        }
        pw2wan_params = {
            'inputpp': {
                'scdm_proj': True,
                'scdm_entanglement': 'erfc',
            }
        }
        pw2wan = {
            'code': fixture_code('quantumespresso.pw2wannier90'),
            'parameters': Dict(dict=pw2wan_params),
            'metadata': {
                'options': get_default_options()
            }
        }
        w90_params = {}
        w90 = {
            'code': fixture_code('wannier90.wannier90'),
            'parameters': Dict(dict=w90_params),
            'kpoints': kpoints,
            'metadata': {
                'options': get_default_options()
            }
        }
        inputs = {
            'structure': structure,
            'scf': scf,
            'nscf': nscf,
            'projwfc': {
                'projwfc': projwfc
            },
            'pw2wannier90': {
                'pw2wannier90': pw2wan
            },
            'wannier90': {
                'wannier90': w90
            },
        }
        return inputs

    return _generate_inputs_wannier90


@pytest.fixture
def generate_workchain_wannier90(generate_workchain, generate_inputs_wannier90):
    """Generate an instance of a `Wannier90WorkChain`."""

    def _generate_workchain_wannier90():
        entry_point = 'wannier90_workflows.wannier90'
        inputs = generate_inputs_wannier90()
        return generate_workchain(entry_point, inputs)

    return _generate_workchain_wannier90
