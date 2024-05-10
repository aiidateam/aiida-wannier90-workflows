"""Tests for the ``Pw2wannier90BaseWorkChain.get_builder_from_protocol`` method."""

import pytest

from aiida.engine import ProcessBuilder

from aiida_quantumespresso.common.types import ElectronicType

from aiida_wannier90_workflows.common.types import WannierProjectionType
from aiida_wannier90_workflows.workflows.base.pw2wannier90 import (
    Pw2wannier90BaseWorkChain,
)


def test_get_available_protocols():
    """Test ``Pw2wannier90BaseWorkChain.get_available_protocols``."""
    protocols = Pw2wannier90BaseWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ["fast", "moderate", "precise"]
    assert all("description" in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``Pw2wannier90BaseWorkChain.get_default_protocol``."""
    assert Pw2wannier90BaseWorkChain.get_default_protocol() == "moderate"


def test_default(fixture_code, data_regression, serialize_builder):
    """Test ``Pw2wannier90BaseWorkChain.get_builder_from_protocol`` for the default protocol."""
    code = fixture_code("quantumespresso.pw2wannier90")

    builder = Pw2wannier90BaseWorkChain.get_builder_from_protocol(code=code)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_overrides(fixture_code, data_regression, serialize_builder):
    """Test ``Pw2wannier90BaseWorkChain.get_builder_from_protocol`` for the ``overrides`` input."""
    code = fixture_code("quantumespresso.pw2wannier90")

    overrides = {"pw2wannier90": {"parameters": {"inputpp": {"fake_input": "fake"}}}}
    builder = Pw2wannier90BaseWorkChain.get_builder_from_protocol(
        code=code, overrides=overrides
    )

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize(
    "electronic_type",
    ((ElectronicType.INSULATOR, "isolated"), (ElectronicType.METAL, "erfc")),
)
def test_electronic_type(fixture_code, electronic_type):
    """Test ``Pw2wannier90BaseWorkChain.get_builder_from_protocol`` with ``electronic_type`` keyword."""
    code = fixture_code("quantumespresso.pw2wannier90")

    builder = Pw2wannier90BaseWorkChain.get_builder_from_protocol(
        code=code,
        electronic_type=electronic_type[0],
        projection_type=WannierProjectionType.SCDM,
    )

    parameters = builder["pw2wannier90"]["parameters"].get_dict()["inputpp"]
    assert parameters["scdm_entanglement"] == electronic_type[1]


@pytest.mark.parametrize(
    "projection_type",
    (
        (WannierProjectionType.SCDM, "scdm_proj"),
        (WannierProjectionType.ATOMIC_PROJECTORS_QE, "atom_proj"),
    ),
)
def test_projection_type(fixture_code, projection_type):
    """Test ``Pw2wannier90BaseWorkChain.get_builder_from_protocol`` with ``projection_type`` keyword."""
    code = fixture_code("quantumespresso.pw2wannier90")

    builder = Pw2wannier90BaseWorkChain.get_builder_from_protocol(
        code=code, projection_type=projection_type[0]
    )

    parameters = builder["pw2wannier90"]["parameters"].get_dict()["inputpp"]
    assert projection_type[1] in parameters
    assert parameters[projection_type[1]]


def test_exclude_projectors(fixture_code):
    """Test ``Pw2wannier90BaseWorkChain.get_builder_from_protocol`` setting the ``exclude_projectors`` input."""
    code = fixture_code("quantumespresso.pw2wannier90")

    exclude_projectors = [2, 3]
    builder = Pw2wannier90BaseWorkChain.get_builder_from_protocol(
        code=code,
        projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
        exclude_projectors=exclude_projectors,
    )

    parameters = builder["pw2wannier90"]["parameters"].get_dict()["inputpp"]
    assert parameters["atom_proj_exclude"] == exclude_projectors
