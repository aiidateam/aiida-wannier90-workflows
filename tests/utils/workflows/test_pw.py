"""Unit tests for :py:mod:`aiida_wannier90_workflows.utils.workflows.pw`."""

import pytest

from aiida_quantumespresso.calculations.pw import PwCalculation

from aiida_wannier90_workflows.utils.workflows.pw import get_fermi_energy_from_nscf

# stdout of an nscf run that still prints the scf-Fermi companion marker.
NSCF_STDOUT_WITH_MARKER = """
     End of band structure calculation

     the Fermi energy is     5.9816 ev
     (compare with:     5.9034 eV, computed in scf)

     Writing output data file aiida.save
"""

# stdout of an nscf run that only prints its own Fermi energy (no marker).
NSCF_STDOUT_WITHOUT_MARKER = """
     End of band structure calculation

     the Fermi energy is     5.9816 ev

     Writing output data file aiida.save
"""


class _FakeRetrieved:
    """Stand in for the ``retrieved`` FolderData node."""

    def __init__(self, content):
        self._content = content

    def get_object_content(self, name):  # pylint: disable=unused-argument
        return self._content


class _FakeDict:
    """Stand in for an ``orm.Dict`` output node."""

    def __init__(self, dictionary):
        self._dictionary = dictionary

    def get_dict(self):
        return dict(self._dictionary)


class _FakeOutputs:
    def __init__(self, stdout, output_parameters):
        self.retrieved = _FakeRetrieved(stdout)
        self.output_parameters = _FakeDict(output_parameters)


class _FakeNscfCalc:
    """Minimal stub of a finished nscf ``PwCalculation`` node.

    Only the attributes accessed by ``get_fermi_energy_from_nscf`` are
    implemented, so the test needs no AiiDA profile or database.
    """

    process_class = PwCalculation
    is_finished_ok = True

    def __init__(self, stdout, output_parameters):
        self.outputs = _FakeOutputs(stdout, output_parameters)


def test_get_fermi_energy_from_nscf_marker_present():
    """Marker present: the scf value from stdout is used, not the fallback."""
    calc = _FakeNscfCalc(
        NSCF_STDOUT_WITH_MARKER,
        # A different value here would be returned only if the fallback ran.
        {"fermi_energy": 7.0, "fermi_energy_units": "eV"},
    )
    assert get_fermi_energy_from_nscf(calc) == pytest.approx(5.9034)


def test_get_fermi_energy_from_nscf_fallback_to_parsed_value():
    """Marker absent: fall back to the parsed nscf Fermi energy."""
    calc = _FakeNscfCalc(
        NSCF_STDOUT_WITHOUT_MARKER,
        {"fermi_energy": 5.9816, "fermi_energy_units": "eV"},
    )
    assert get_fermi_energy_from_nscf(calc) == pytest.approx(5.9816)


def test_get_fermi_energy_from_nscf_fallback_spin_polarised():
    """Marker absent, two Fermi levels: return the higher of the two."""
    calc = _FakeNscfCalc(
        NSCF_STDOUT_WITHOUT_MARKER,
        {
            "fermi_energy_up": 5.5,
            "fermi_energy_down": 6.1,
            "fermi_energy_units": "eV",
        },
    )
    assert get_fermi_energy_from_nscf(calc) == pytest.approx(6.1)


def test_get_fermi_energy_from_nscf_returns_none_when_unavailable():
    """Marker absent and nothing parsed: return None (guarded by callers)."""
    calc = _FakeNscfCalc(
        NSCF_STDOUT_WITHOUT_MARKER,
        {"fermi_energy_units": "eV"},
    )
    assert get_fermi_energy_from_nscf(calc) is None


def test_get_fermi_energy_from_nscf_ignores_non_ev_units():
    """A parsed Fermi energy in non-eV units is not used by the fallback."""
    calc = _FakeNscfCalc(
        NSCF_STDOUT_WITHOUT_MARKER,
        {"fermi_energy": 0.44, "fermi_energy_units": "Ry"},
    )
    assert get_fermi_energy_from_nscf(calc) is None
