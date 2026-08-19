"""Tests for the :mod:`aiida_wannier90_workflows.utils.pseudo` module."""

import pytest

from aiida_wannier90_workflows.utils.pseudo import get_frozen_list_ext


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
