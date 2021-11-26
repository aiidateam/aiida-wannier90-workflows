# -*- coding: utf-8 -*-
# pylint: disable=redefined-outer-name
"""Tests for the ``Wannier90OptimizeWorkChain.get_builder_from_protocol`` method."""
import pytest

from aiida.engine import ProcessBuilder
from aiida.plugins import WorkflowFactory

from aiida_quantumespresso.common.types import ElectronicType, SpinType
from aiida_wannier90_workflows.common.types import WannierProjectionType

Wannier90OptimizeWorkChain = WorkflowFactory('wannier90_workflows.optimize')


@pytest.fixture
def get_bands_generator_inputs(fixture_code, generate_structure):
    """Generate a set of default inputs for the ``Wannier90OptimizeWorkChain.get_builder_from_protocol()`` method."""

    def _get_inputs(structure_id='Si'):
        return {
            'codes': {
                'pw': fixture_code('quantumespresso.pw'),
                'pw2wannier90': fixture_code('quantumespresso.pw2wannier90'),
                'wannier90': fixture_code('wannier90.wannier90'),
                'projwfc': fixture_code('quantumespresso.projwfc'),
                'opengrid': fixture_code('quantumespresso.opengrid'),
            },
            'structure': generate_structure(structure_id=structure_id)
        }

    return _get_inputs


def test_get_available_protocols():
    """Test ``Wannier90OptimizeWorkChain.get_available_protocols``."""
    protocols = Wannier90OptimizeWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ['fast', 'moderate', 'precise']
    assert all('description' in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``Wannier90OptimizeWorkChain.get_default_protocol``."""
    assert Wannier90OptimizeWorkChain.get_default_protocol() == 'moderate'


@pytest.mark.parametrize('structure', ('Si', 'H2O', 'GaAs', 'BaTiO3'))
def test_scdm(get_bands_generator_inputs, data_regression, serialize_builder, structure):
    """Test ``Wannier90OptimizeWorkChain.get_builder_from_protocol`` for the default protocol."""

    inputs = get_bands_generator_inputs(structure)
    builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(**inputs)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize('structure', ('Si', 'H2O', 'GaAs', 'BaTiO3'))
def test_atomic_projectors_qe(get_bands_generator_inputs, data_regression, serialize_builder, structure):
    """Test ``Wannier90OptimizeWorkChain.get_builder_from_protocol`` for the default protocol."""

    inputs = get_bands_generator_inputs(structure)
    builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
        **inputs, projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE
    )

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_electronic_type(get_bands_generator_inputs):
    """Test ``Wannier90OptimizeWorkChain.get_builder_from_protocol`` with ``electronic_type`` keyword."""
    with pytest.raises(NotImplementedError):
        builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
            **get_bands_generator_inputs(), electronic_type=ElectronicType.AUTOMATIC
        )

    builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
        **get_bands_generator_inputs(), electronic_type=ElectronicType.INSULATOR
    )
    for namespace, occupations in zip((builder.scf, builder.nscf), ('fixed', 'fixed')):
        parameters = namespace['pw']['parameters'].get_dict()
        assert parameters['SYSTEM']['occupations'] == occupations
        assert 'degauss' not in parameters['SYSTEM']
        assert 'smearing' not in parameters['SYSTEM']

    builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
        **get_bands_generator_inputs(), electronic_type=ElectronicType.METAL
    )
    for namespace, occupations in zip((builder.scf, builder.nscf), ('smearing', 'smearing')):
        parameters = namespace['pw']['parameters'].get_dict()
        assert parameters['SYSTEM']['occupations'] == occupations
        assert 'degauss' in parameters['SYSTEM']
        assert 'smearing' in parameters['SYSTEM']


def test_spin_type(get_bands_generator_inputs):
    """Test ``Wannier90OptimizeWorkChain.get_builder_from_protocol`` with ``spin_type`` keyword."""
    with pytest.raises(NotImplementedError):
        for spin_type in [SpinType.COLLINEAR, SpinType.NON_COLLINEAR]:
            builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
                **get_bands_generator_inputs(), spin_type=spin_type
            )

    builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
        **get_bands_generator_inputs(), spin_type=SpinType.NONE
    )
    for namespace in [builder.scf, builder.nscf]:
        parameters = namespace['pw']['parameters'].get_dict()
        assert 'nspin' not in parameters['SYSTEM']
        assert 'starting_magnetization' not in parameters['SYSTEM']


def test_projection_type(get_bands_generator_inputs):
    """Test ``Wannier90OptimizeWorkChain.get_builder_from_protocol`` with ``projection_type`` keyword."""
    # with pytest.raises(NotImplementedError):
    #     for projection_type in [
    #         WannierProjectionType.ANALYTIC, WannierProjectionType.RANDOM,
    #         WannierProjectionType.ATOMIC_PROJECTORS_OPENMX
    #     ]:
    #         builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
    #             **get_bands_generator_inputs(), projection_type=projection_type
    #         )

    builder = Wannier90OptimizeWorkChain.get_builder_from_protocol(
        **get_bands_generator_inputs(), projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE
    )
    for namespace in [
        builder.wannier90,
    ]:
        parameters = namespace['wannier90']['parameters'].get_dict()
        assert 'auto_projections' in parameters
