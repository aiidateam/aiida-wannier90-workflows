"""Tests for the projector counting in :py:mod:`~aiida_wannier90_workflows.utils.pseudo`."""

import pytest

from aiida_wannier90_workflows.utils.pseudo import (
    get_number_of_projections,
    get_pseudo_and_cutoff,
)

# The bundled fixtures carry silicon in both families, so the same structure can
# be counted with a scalar- and a fully relativistic pseudopotential.
SCALAR_RELATIVISTIC_FAMILY = "SSSP/1.3/PBEsol/efficiency"
FULLY_RELATIVISTIC_FAMILY = "PseudoDojo/0.4/PBE/FR/standard/upf"


@pytest.mark.parametrize(
    ("spin_non_collinear", "spin_orbit_coupling", "expected"),
    (
        # Averaged 3S + 3P per atom: (2*0+1) + (2*1+1) = 4.
        (False, False, 8),
        # Non-collinear without spin-orbit: two spinors per averaged channel.
        (True, False, 16),
        # lspinorb: the j channels stay split, 2 + 4 + 2 = 8 per atom.
        (True, True, 16),
        # No regime given: infer it from `spin_non_collinear`.
        (False, None, 8),
        (True, None, 16),
    ),
)
def test_number_of_projections_fully_relativistic(
    generate_structure, spin_non_collinear, spin_orbit_coupling, expected
):
    """Count the projectors of a fully relativistic pseudo in each spin regime."""
    structure = generate_structure("Si")
    pseudos, _, _ = get_pseudo_and_cutoff(FULLY_RELATIVISTIC_FAMILY, structure)

    assert (
        get_number_of_projections(
            structure, pseudos, spin_non_collinear, spin_orbit_coupling
        )
        == expected
    )


@pytest.mark.parametrize("spin_non_collinear", (False, True))
def test_number_of_projections_relativity_agnostic(
    generate_structure, spin_non_collinear
):
    """Outside a spin-orbit run an FR pseudo counts as its scalar counterpart.

    pw.x averages the j = l +/- 1/2 pairs whenever ``lspinorb`` is false, so
    the two families describe the same 3S and 3P channels of silicon.
    """
    structure = generate_structure("Si")
    fully_relativistic, _, _ = get_pseudo_and_cutoff(
        FULLY_RELATIVISTIC_FAMILY, structure
    )
    scalar_relativistic, _, _ = get_pseudo_and_cutoff(
        SCALAR_RELATIVISTIC_FAMILY, structure
    )

    assert get_number_of_projections(
        structure, fully_relativistic, spin_non_collinear, spin_orbit_coupling=False
    ) == get_number_of_projections(
        structure, scalar_relativistic, spin_non_collinear, spin_orbit_coupling=False
    )


@pytest.mark.parametrize("spin_non_collinear", (False, True))
def test_number_of_projections_scalar_relativistic(
    generate_structure, spin_non_collinear
):
    """A scalar-relativistic pseudo is unaffected by the spin-orbit flag."""
    structure = generate_structure("Si")
    pseudos, _, _ = get_pseudo_and_cutoff(SCALAR_RELATIVISTIC_FAMILY, structure)

    expected = 16 if spin_non_collinear else 8
    for spin_orbit_coupling in (False, True, None):
        assert (
            get_number_of_projections(
                structure, pseudos, spin_non_collinear, spin_orbit_coupling
            )
            == expected
        )


def test_number_of_pswfc_averaged():
    """``average_soc`` counts the channels QE's ``average_pp`` leaves behind."""
    from aiida_wannier90_workflows.utils.pseudo.upf import parse_number_of_pswfc

    # A 3S (j = 1/2) and a 3P split into j = 1/2 and j = 3/2, as PseudoDojo
    # writes silicon in its fully relativistic set.
    content = (
        '<UPF version="2.0.1">\n'
        '<PP_HEADER element="Si" z_valence="4.0" has_so="T" number_of_wfc="8"/>\n'
        "<PP_SPIN_ORB>\n"
        '<PP_RELWFC.1 index="1" lchi="0" jchi="0.5" nn="1"/>\n'
        '<PP_RELWFC.2 index="2" lchi="1" jchi="1.5" nn="2"/>\n'
        '<PP_RELWFC.3 index="3" lchi="1" jchi="0.5" nn="2"/>\n'
        "</PP_SPIN_ORB>\n"
        "</UPF>\n"
    )

    assert parse_number_of_pswfc(content) == 8
    assert parse_number_of_pswfc(content, average_soc=True) == 4
