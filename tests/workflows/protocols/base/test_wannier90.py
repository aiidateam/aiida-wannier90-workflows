# -*- coding: utf-8 -*-
"""Tests for the ``Wannier90BaseWorkChain.get_builder_from_protocol`` method."""
import pytest

from aiida.engine import ProcessBuilder

from aiida_quantumespresso.common.types import ElectronicType, SpinType

from aiida_wannier90_workflows.common.types import (
    WannierDisentanglementType, WannierFrozenType, WannierProjectionType
)
from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain


def test_get_available_protocols():
    """Test ``Wannier90BaseWorkChain.get_available_protocols``."""
    protocols = Wannier90BaseWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['fast', 'moderate', 'precise']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``Wannier90BaseWorkChain.get_default_protocol``."""
    assert Wannier90BaseWorkChain.get_default_protocol() == 'moderate'


def test_default(fixture_code, generate_structure, data_regression, serialize_builder):
    """Test ``Wannier90BaseWorkChain.get_builder_from_protocol`` for the default protocol."""
    code = fixture_code('wannier90.wannier90')
    structure = generate_structure('Si')
    builder = Wannier90BaseWorkChain.get_builder_from_protocol(code, structure=structure)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize('electronic_type', (ElectronicType.METAL, ElectronicType.INSULATOR))
def test_electronic_type(fixture_code, generate_structure, data_regression, serialize_builder, electronic_type):
    """Test ``Wannier90BaseWorkChain.get_builder_from_protocol`` with ``electronic_type`` keyword."""
    code = fixture_code('wannier90.wannier90')
    structure = generate_structure('Si')

    with pytest.raises(NotImplementedError):
        Wannier90BaseWorkChain.get_builder_from_protocol(
            code, structure=structure, electronic_type=ElectronicType.AUTOMATIC
        )

    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code, structure=structure, electronic_type=electronic_type
    )

    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize('spin_type', (SpinType.NONE, SpinType.SPIN_ORBIT))
def test_spin_type(fixture_code, generate_structure, data_regression, serialize_builder, spin_type):
    """Test ``Wannier90BaseWorkChain.get_builder_from_protocol`` with ``spin_type`` keyword."""
    code = fixture_code('wannier90.wannier90')
    structure = generate_structure('Si')

    if spin_type == SpinType.SPIN_ORBIT:
        with pytest.raises(ValueError, match='Need to explicitly specify `pseudo_family`'):
            builder = Wannier90BaseWorkChain.get_builder_from_protocol(code, structure=structure, spin_type=spin_type)
        return

    builder = Wannier90BaseWorkChain.get_builder_from_protocol(code, structure=structure, spin_type=spin_type)

    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize('projection_type', (WannierProjectionType.ATOMIC_PROJECTORS_QE, WannierProjectionType.SCDM))
def test_projection_type(fixture_code, generate_structure, data_regression, serialize_builder, projection_type):
    """Test ``Wannier90BaseWorkChain.get_builder_from_protocol`` with invalid ``initial_magnetic_moments`` keyword."""
    code = fixture_code('wannier90.wannier90')
    structure = generate_structure('Si')

    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code, structure=structure, projection_type=projection_type
    )

    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize('disentanglement_type', (WannierDisentanglementType.NONE, WannierDisentanglementType.SMV))
def test_disentanglement_type(
    fixture_code, generate_structure, data_regression, serialize_builder, disentanglement_type
):
    """Test ``Wannier90BaseWorkChain.get_builder_from_protocol`` with ``initial_magnetic_moments`` keyword."""
    code = fixture_code('wannier90.wannier90')
    structure = generate_structure('Si')

    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code, structure=structure, disentanglement_type=disentanglement_type
    )

    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize(
    'frozen_type', (
        WannierFrozenType.NONE, WannierFrozenType.ENERGY_AUTO, WannierFrozenType.ENERGY_FIXED,
        WannierFrozenType.FIXED_PLUS_PROJECTABILITY, WannierFrozenType.PROJECTABILITY
    )
)
def test_frozen_type(fixture_code, generate_structure, data_regression, serialize_builder, frozen_type):
    """Test magnetization ``overrides`` for the ``Wannier90BaseWorkChain.get_builder_from_protocol`` method."""
    code = fixture_code('wannier90.wannier90')
    structure = generate_structure('Si')

    builder = Wannier90BaseWorkChain.get_builder_from_protocol(code, structure=structure, frozen_type=frozen_type)

    data_regression.check(serialize_builder(builder))


def test_parameter_overrides(fixture_code, generate_structure, data_regression, serialize_builder):
    """Test specifying parameter ``overrides`` for the ``get_builder_from_protocol()`` method."""
    code = fixture_code('wannier90.wannier90')
    structure = generate_structure('Si')

    overrides = {'wannier90': {'parameters': {'fake_input': 'fake'}}}
    builder = Wannier90BaseWorkChain.get_builder_from_protocol(code, structure=structure, overrides=overrides)

    data_regression.check(serialize_builder(builder))


def test_settings_overrides(fixture_code, generate_structure, data_regression, serialize_builder):
    """Test specifying settings ``overrides`` for the ``get_builder_from_protocol()`` method."""
    code = fixture_code('wannier90.wannier90')
    structure = generate_structure('Si')

    overrides = {'wannier90': {'settings': {'cmdline': ['-nk', 6]}}}
    builder = Wannier90BaseWorkChain.get_builder_from_protocol(code, structure=structure, overrides=overrides)

    data_regression.check(serialize_builder(builder))


def test_metadata_overrides(fixture_code, generate_structure, data_regression, serialize_builder):
    """Test specifying metadata ``overrides`` for the ``get_builder_from_protocol()`` method."""
    code = fixture_code('wannier90.wannier90')
    structure = generate_structure('Si')

    overrides = {
        'wannier90': {
            'metadata': {
                'options': {
                    'resources': {
                        'num_machines': 1e90
                    },
                    'max_wallclock_seconds': 1
                }
            }
        }
    }
    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code,
        structure=structure,
        overrides=overrides,
    )

    data_regression.check(serialize_builder(builder))
