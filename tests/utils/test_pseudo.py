"""Unit tests for the :py:mod:`~aiida_wannier90_workflows.utils.pseudo` module."""

import pytest

from aiida_wannier90_workflows.utils.pseudo import (
    get_frozen_list_ext,
    get_pseudo_and_cutoff,
    get_pseudos,
)


def _upf_content(
    element, z_valence, chi, has_so=False, labels=None, occupations=None, n_attrs=None
):
    """Build minimal UPF v2 content that ``upf_tools`` can parse.

    Only the pieces the valence-orbital derivation reads are populated with
    meaningful values: the ``PP_HEADER`` ``has_so``/``z_valence`` fields and one
    ``PP_CHI`` per atomic wave function carrying its ``n``, ``l`` and label.

    :param chi: for a scalar-relativistic pseudo, a list of ``(n, l)`` pairs;
        for a fully-relativistic pseudo (``has_so=True``), a list of
        ``(n, l, j)`` triples with one entry per j channel.
    :param labels: optional per-entry override of the ``label`` attribute, to
        exercise reconstruction from ``n``/``l`` when the label is unusable.
    :param occupations: optional per-entry occupation values (by default the
        ``z_valence`` electrons are spread evenly so the sum agrees), to
        exercise the occupation/z_valence sanity check.
    :param n_attrs: optional per-entry override of the emitted ``n`` attribute
        (defaults to the label's ``n``), to exercise UPFs whose ``n`` attribute
        is the pseudo (node-based) principal number rather than the atomic one.
    """
    has_so_str = "true" if has_so else "false"
    letters = {0: "S", 1: "P", 2: "D", 3: "F"}
    chi_blocks, relwfc_blocks = [], []
    for index, entry in enumerate(chi, start=1):
        n, l = entry[0], entry[1]
        label = f"{n}{letters[l]}" if labels is None else labels[index - 1]
        occupation = (
            float(z_valence) / len(chi)
            if occupations is None
            else occupations[index - 1]
        )
        n_attr = n if n_attrs is None else n_attrs[index - 1]
        label_attr = "" if label is None else f'label="{label}" '
        chi_blocks.append(
            f'<PP_CHI.{index} l="{l}" n="{n_attr}" occupation="{occupation}" '
            f"{label_attr}> 0.0 0.0 0.0 </PP_CHI.{index}>"
        )
        if has_so:
            j = entry[2]
            relwfc_blocks.append(f'<PP_RELWFC.{index} jchi="{j}" lchi="{l}" nn="{n}"/>')
    spin_orb = (
        f"<PP_SPIN_ORB>\n{chr(10).join(relwfc_blocks)}\n</PP_SPIN_ORB>\n"
        if has_so
        else ""
    )
    return (
        '<UPF version="2.0.1">\n'
        f'<PP_HEADER element="{element}" z_valence="{z_valence}" '
        f'number_of_proj="0" number_of_wfc="{len(chi)}" mesh_size="3" '
        f'core_correction="false" pseudo_type="NC" is_ultrasoft="false" '
        f'is_paw="false" has_so="{has_so_str}" l_max="3" l_max_rho="0"/>\n'
        '<PP_MESH><PP_R size="3"> 0 0.1 0.2 </PP_R></PP_MESH>\n'
        '<PP_LOCAL size="3"> 0 0 0 </PP_LOCAL>\n'
        "<PP_NONLOCAL><PP_DIJ> 0 </PP_DIJ></PP_NONLOCAL>\n"
        f"<PP_PSWFC>\n{chr(10).join(chi_blocks)}\n</PP_PSWFC>\n"
        f"{spin_orb}"
        '<PP_RHOATOM size="3"> 0 0 0 </PP_RHOATOM>\n'
        "</UPF>\n"
    )


class FakePseudo:
    """Duck-typed stand-in for a ``PseudoPotentialData``."""

    def __init__(self, element, z_valence, content="", md5="0" * 32):
        self.element = element
        self.z_valence = z_valence
        self.md5 = md5
        self.filename = f"{element}.upf"
        self._content = content

    def get_content(self):
        """Return the (possibly invalid) UPF content read by the derivation."""
        return self._content


@pytest.mark.parametrize(
    ("element", "z_valence", "chi", "expected"),
    (
        ("Si", 4, [(3, 0), (3, 1)], ["3S", "3P"]),
        ("O", 6, [(2, 0), (2, 1)], ["2S", "2P"]),
        ("H", 1, [(1, 0)], ["1S"]),
        # Semicore-in-valence pseudisation: the PP_CHI labels are read off in
        # the order they appear in the file (which is the order the curated
        # tables and QE use), not re-sorted into an aufbau order.
        ("Ti", 12, [(3, 0), (3, 1), (3, 2), (4, 0)], ["3S", "3P", "3D", "4S"]),
        ("Cu", 19, [(3, 0), (3, 1), (3, 2), (4, 0)], ["3S", "3P", "3D", "4S"]),
        # Heavy element the old aufbau walk got wrong (it inferred ['4F','5D']):
        # reading PP_CHI gives the physical 5s/5p/5d/6s valence.
        ("Au", 19, [(5, 0), (5, 1), (5, 2), (6, 0)], ["5S", "5P", "5D", "6S"]),
    ),
)
def test_derive_pseudo_orbitals(element, z_valence, chi, expected):
    """The valence orbitals are read directly from the UPF's PP_PSWFC labels."""
    from aiida_wannier90_workflows.utils.pseudo import _derive_pseudo_orbitals_from_upf

    pseudo = FakePseudo(element, z_valence, _upf_content(element, z_valence, chi))
    entry = _derive_pseudo_orbitals_from_upf(pseudo)
    assert entry is not None
    assert entry["pswfcs"] == expected
    assert entry["semicores"] == []


def test_derive_pseudo_orbitals_fully_relativistic():
    """A fully-relativistic pseudo lists one PP_CHI per j channel; the j-split
    is collapsed to one label per (n, l), preserving order."""
    from aiida_wannier90_workflows.utils.pseudo import _derive_pseudo_orbitals_from_upf

    # Pb-like: 5d (j=3/2, 5/2), 6s (j=1/2), 6p (j=1/2, 3/2). The raw label list
    # has 5D and 6P twice each; collapsing the j-channels yields the unique set.
    chi = [(5, 2, 1.5), (5, 2, 2.5), (6, 0, 0.5), (6, 1, 0.5), (6, 1, 1.5)]
    content = _upf_content("Pb", 14, chi, has_so=True)
    entry = _derive_pseudo_orbitals_from_upf(FakePseudo("Pb", 14, content))
    assert entry is not None
    assert entry["pswfcs"] == ["5D", "6S", "6P"]


def test_derive_pseudo_orbitals_reconstructs_unusable_label():
    """When a PP_CHI's label field is empty or non-standard, the label is
    reconstructed from its ``n`` and ``l`` rather than failing."""
    from aiida_wannier90_workflows.utils.pseudo import _derive_pseudo_orbitals_from_upf

    # First label empty, second label non-standard: both must be rebuilt from
    # the intact n/l attributes.
    content = _upf_content("Si", 4, [(3, 0), (3, 1)], labels=[None, "3p_orbital"])
    entry = _derive_pseudo_orbitals_from_upf(FakePseudo("Si", 4, content))
    assert entry is not None
    assert entry["pswfcs"] == ["3S", "3P"]


def test_derive_pseudo_orbitals_label_wins_over_node_based_n():
    """Some ultrasoft/PAW UPFs write the pseudo (node-based) principal number in
    the ``n`` attribute -- a node-free 6s/6p/5d projector gets n=1/2/3 -- while
    the label keeps the true atomic n. The label must win."""
    from aiida_wannier90_workflows.utils.pseudo import _derive_pseudo_orbitals_from_upf

    # Au.pz-rrkjus_aewfc-like: labels 6P/5D/6S with node-based n attributes
    # 2/3/1. Reading the n attribute would give the nonsense ['2P','3D','1S'].
    content = _upf_content(
        "Au",
        11,
        [(6, 1), (5, 2), (6, 0)],
        occupations=[0.0, 10.0, 1.0],
        n_attrs=[2, 3, 1],
    )
    entry = _derive_pseudo_orbitals_from_upf(FakePseudo("Au", 11, content))
    assert entry is not None
    assert entry["pswfcs"] == ["6P", "5D", "6S"]


def test_derive_pseudo_orbitals_no_pswfc_returns_none():
    """A UPF without a PP_PSWFC block cannot be read and fails closed."""
    from aiida_wannier90_workflows.utils.pseudo import _derive_pseudo_orbitals_from_upf

    content = (
        '<UPF version="2.0.1">\n'
        '<PP_HEADER element="Si" z_valence="4" number_of_proj="0" '
        'number_of_wfc="0" mesh_size="3" has_so="false"/>\n'
        '<PP_MESH><PP_R size="3"> 0 0.1 0.2 </PP_R></PP_MESH>\n'
        "</UPF>\n"
    )
    assert _derive_pseudo_orbitals_from_upf(FakePseudo("Si", 4, content)) is None


def test_derive_pseudo_orbitals_unparseable_returns_none():
    """Unparseable content cannot be read for its wave functions (fail closed,
    not fail open)."""
    from aiida_wannier90_workflows.utils.pseudo import _derive_pseudo_orbitals_from_upf

    assert (
        _derive_pseudo_orbitals_from_upf(FakePseudo("Au", 19, "not a upf file")) is None
    )


def test_derive_pseudo_orbitals_fails_closed_on_occupation_mismatch():
    """A disagreement between the summed PP_CHI occupations and z_valence can
    mean a valence orbital is missing from PP_PSWFC, so the derivation warns
    and fails closed rather than returning a possibly-incomplete set."""
    from aiida_wannier90_workflows.utils.pseudo import _derive_pseudo_orbitals_from_upf

    # Occupations sum to 10.0 but z_valence claims 12 -- e.g. a semicore 3S
    # missing from PP_PSWFC.
    content = _upf_content(
        "Ti", 12, [(3, 1), (3, 2), (4, 0)], occupations=[6.0, 2.0, 2.0]
    )
    with pytest.warns(UserWarning, match="disagrees with"):
        assert _derive_pseudo_orbitals_from_upf(FakePseudo("Ti", 12, content)) is None


def test_get_pseudo_orbitals_overrides_win():
    """An explicit override bypasses tables and UPF derivation."""
    from aiida_wannier90_workflows.utils.pseudo import (
        PseudoOrbitals,
        get_pseudo_orbitals,
    )

    override = PseudoOrbitals(pswfcs=["3S", "3P", "4S", "3D"], semicores=["3S", "3P"])
    result = get_pseudo_orbitals(
        {"Ti": FakePseudo("Ti", 12)}, overrides={"Ti": override}
    )
    assert result["Ti"]["semicores"] == ["3S", "3P"]


def test_get_pseudo_orbitals_derivation_warns():
    """A pseudo missing from the tables resolves by UPF derivation, with a
    warning that states the orbitals were read from the pseudopotential file."""
    from aiida_wannier90_workflows.utils.pseudo import get_pseudo_orbitals

    pseudo = FakePseudo("Si", 4, _upf_content("Si", 4, [(3, 0), (3, 1)]))
    with pytest.warns(UserWarning, match="read directly from the pseudopotential file"):
        result = get_pseudo_orbitals({"Si": pseudo})
    assert result["Si"]["pswfcs"] == ["3S", "3P"]
    assert result["Si"]["semicores"] == []


def test_get_pseudo_orbitals_unresolvable_raises():
    """When derivation is impossible the error explains the override escape hatch."""
    from aiida_wannier90_workflows.utils.pseudo import get_pseudo_orbitals

    with pytest.raises(ValueError, match="overrides"):
        get_pseudo_orbitals({"Xx": FakePseudo("Xx", 4, "not a upf file")})


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
