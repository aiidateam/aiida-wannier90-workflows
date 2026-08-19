"""Tests for the :mod:`aiida_wannier90_workflows.utils.pseudo` module."""

import pytest

from aiida_wannier90_workflows.utils.pseudo import (
    get_frozen_list_ext,
    get_pseudo_and_cutoff,
    get_pseudos,
)


def test_get_pseudos_cutoffs_family_without_stringency(
    cutoffs_family_without_stringency, generate_structure
):
    """A family that recommends no cutoffs still provides its pseudos."""
    pseudos = get_pseudos(
        cutoffs_family_without_stringency.label, generate_structure("Si")
    )

    assert sorted(pseudos) == ["Si"]


def test_get_pseudos_plain_family(plain_pseudo_family, generate_structure):
    """A family that cannot carry cutoffs at all still provides its pseudos."""
    pseudos = get_pseudos(plain_pseudo_family.label, generate_structure("Si"))

    assert sorted(pseudos) == ["Si"]


def test_get_pseudos_family_with_cutoffs(pseudos, generate_structure):
    """A family that does recommend cutoffs is served by the same entry point."""
    sssp, _ = pseudos

    assert sorted(get_pseudos(sssp.label, generate_structure("Si"))) == ["Si"]


def test_get_pseudos_family_not_installed(generate_structure):
    """An unknown label names itself in the error."""
    with pytest.raises(ValueError, match="`NotAFamily/1.0` is not installed"):
        get_pseudos("NotAFamily/1.0", generate_structure("Si"))


def test_get_pseudo_and_cutoff_returns_cutoffs(pseudos, generate_structure):
    """The cutoffs of a family that has them are unchanged."""
    sssp, _ = pseudos

    found, cutoff_wfc, cutoff_rho = get_pseudo_and_cutoff(
        sssp.label, generate_structure("Si")
    )

    assert sorted(found) == ["Si"]
    assert (cutoff_wfc, cutoff_rho) == (30.0, 240.0)


def test_get_pseudo_and_cutoff_requires_cutoffs(
    cutoffs_family_without_stringency, generate_structure
):
    """Asking for cutoffs a family does not have still raises."""
    with pytest.raises(ValueError, match="failed to obtain recommended cutoffs"):
        get_pseudo_and_cutoff(
            cutoffs_family_without_stringency.label, generate_structure("Si")
        )


def test_get_pseudo_and_cutoff_rejects_plain_family(
    plain_pseudo_family, generate_structure
):
    """A family that cannot carry cutoffs is not a candidate for this function."""
    with pytest.raises(ValueError, match="`MyPseudos/local` is not installed"):
        get_pseudo_and_cutoff(plain_pseudo_family.label, generate_structure("Si"))


@pytest.mark.parametrize(
    ("orbital", "expect_frozen"),
    (
        # No key at all: a pseudo-atomic orbital, frozen.
        ({"label": "S", "l": 0}, True),
        # Yuhao-protocol alpha bookkeeping: "UPF" marks the original
        # pseudo-atomic orbital, a number marks a generated one.
        ({"label": "S", "l": 0, "alpha": "UPF"}, True),
        ({"label": "S", "l": 0, "alpha": 1.5}, False),
        # An explicit ``frozen`` takes precedence over ``alpha``.
        ({"label": "S", "l": 0, "frozen": False}, False),
        ({"label": "S", "l": 0, "frozen": False, "alpha": "UPF"}, False),
        ({"label": "S", "l": 0, "frozen": True, "alpha": 1.5}, True),
    ),
)
def test_get_frozen_list_ext(generate_structure, orbital, expect_frozen):
    """Test the per-orbital frozen selection of ``get_frozen_list_ext``."""
    structure = generate_structure("Si")

    frozen_list = get_frozen_list_ext(
        structure=structure,
        external_projectors={"Si": [orbital]},
        spin_non_collinear=False,
    )

    # Bulk silicon has two sites; an s orbital contributes one projector each.
    assert frozen_list == ([1, 2] if expect_frozen else [])


def test_get_frozen_list_ext_mixed(generate_structure):
    """Test frozen indexing across a mixed orbital table."""
    structure = generate_structure("Si")

    frozen_list = get_frozen_list_ext(
        structure=structure,
        external_projectors={
            "Si": [
                {"label": "S", "l": 0, "alpha": "UPF"},
                {"label": "P", "l": 1, "frozen": False},
            ]
        },
        spin_non_collinear=False,
    )

    # Per site: s (1 projector, frozen) then p (3 projectors, not frozen).
    assert frozen_list == [1, 5]
