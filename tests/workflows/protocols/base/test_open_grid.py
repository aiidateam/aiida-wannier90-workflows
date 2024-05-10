"""Tests for the ``OpenGridBaseWorkChain.get_builder_from_protocol`` method."""

from aiida.engine import ProcessBuilder

from aiida_wannier90_workflows.workflows.base.open_grid import OpenGridBaseWorkChain


def test_get_available_protocols():
    """Test ``OpenGridBaseWorkChain.get_available_protocols``."""
    protocols = OpenGridBaseWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ["fast", "moderate", "precise"]
    assert all("description" in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``OpenGridBaseWorkChain.get_default_protocol``."""
    assert OpenGridBaseWorkChain.get_default_protocol() == "moderate"


def test_default(fixture_code, data_regression, serialize_builder):
    """Test ``OpenGridBaseWorkChain.get_builder_from_protocol`` for the default protocol."""
    code = fixture_code("quantumespresso.open_grid")

    builder = OpenGridBaseWorkChain.get_builder_from_protocol(code)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_overrides(fixture_code, data_regression, serialize_builder):
    """Test ``OpenGridBaseWorkChain.get_builder_from_protocol`` for the ``overrides`` input."""
    code = fixture_code("quantumespresso.open_grid")

    overrides = {"open_grid": {"metadata": {"options": {"withmpi": False}}}}
    builder = OpenGridBaseWorkChain.get_builder_from_protocol(code, overrides=overrides)

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))
