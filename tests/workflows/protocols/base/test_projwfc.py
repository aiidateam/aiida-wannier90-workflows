"""Tests for the ``ProjwfcBaseWorkChain.get_builder_from_protocol`` method."""
from aiida.engine import ProcessBuilder

from aiida_wannier90_workflows.workflows.base.projwfc import ProjwfcBaseWorkChain


def test_get_available_protocols():
    """Test ``ProjwfcBaseWorkChain.get_available_protocols``."""
    protocols = ProjwfcBaseWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ["fast", "moderate", "precise"]
    assert all("description" in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``ProjwfcBaseWorkChain.get_default_protocol``."""
    assert ProjwfcBaseWorkChain.get_default_protocol() == "moderate"


def test_default(fixture_code, data_regression, serialize_builder):
    """Test ``ProjwfcBaseWorkChain.get_builder_from_protocol`` for the default protocol."""
    code = fixture_code("quantumespresso.projwfc")

    builder = ProjwfcBaseWorkChain.get_builder_from_protocol(code)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_overrides(fixture_code, data_regression, serialize_builder):
    """Test ``ProjwfcBaseWorkChain.get_builder_from_protocol`` for the ``overrides`` input."""
    code = fixture_code("quantumespresso.projwfc")

    overrides = {"projwfc": {"metadata": {"options": {"withmpi": False}}}}
    builder = ProjwfcBaseWorkChain.get_builder_from_protocol(code, overrides=overrides)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))
