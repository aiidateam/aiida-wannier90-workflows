"""Tests for the projected bands work chain."""

from aiida_wannier90_workflows.workflows.projwfcbands import ProjwfcBandsWorkChain


def test_spec():
    """Test that the process specification can be constructed."""
    spec = ProjwfcBandsWorkChain.spec()

    assert "projwfc" in spec.inputs
    assert ("relax" in spec.inputs) == hasattr(
        ProjwfcBandsWorkChain, "should_run_relax"
    )
