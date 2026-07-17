"""Unit tests for the :py:mod:`~aiida_wannier90_workflows.utils.pseudo` module."""

import pytest


def _upf_content(element, z_valence, chi, has_so=False):
    """Build minimal UPF v2 content that ``upf_tools`` can parse.

    Only the pieces the valence-orbital cross-check reads are populated with
    meaningful values: the ``PP_HEADER`` ``has_so`` flag and one ``PP_CHI``
    per atomic wave function carrying its angular momentum ``l``.

    :param chi: for a scalar-relativistic pseudo, a list of ``l`` values; for a
        fully-relativistic pseudo (``has_so=True``), a list of ``(l, j)`` pairs
        with one entry per j channel.
    """
    has_so_str = "true" if has_so else "false"
    chi_blocks, relwfc_blocks = [], []
    for index, entry in enumerate(chi, start=1):
        l = entry[0] if has_so else entry
        chi_blocks.append(
            f'<PP_CHI.{index} l="{l}" occupation="1.0" label="{l + 1}L" '
            f'n="{l + 1}"> 0.0 0.0 0.0 </PP_CHI.{index}>'
        )
        if has_so:
            j = entry[1]
            relwfc_blocks.append(
                f'<PP_RELWFC.{index} jchi="{j}" lchi="{l}" nn="{l + 1}"/>'
            )
    spin_orb = (
        f'<PP_SPIN_ORB>\n{chr(10).join(relwfc_blocks)}\n</PP_SPIN_ORB>\n'
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
        '<PP_NONLOCAL><PP_DIJ> 0 </PP_DIJ></PP_NONLOCAL>\n'
        f'<PP_PSWFC>\n{chr(10).join(chi_blocks)}\n</PP_PSWFC>\n'
        f'{spin_orb}'
        '<PP_RHOATOM size="3"> 0 0 0 </PP_RHOATOM>\n'
        '</UPF>\n'
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
        """Return the (possibly invalid) UPF content used by the cross-check."""
        return self._content


@pytest.mark.parametrize(
    ("element", "z_valence", "chi", "expected"),
    (
        ("Si", 4, [0, 1], ["3S", "3P"]),
        ("O", 6, [0, 1], ["2S", "2P"]),
        # Semicore-in-valence pseudisation: shells collected outermost-inward
        # until the valence electrons are accounted for.
        ("Ti", 12, [0, 0, 1, 2], ["3S", "3P", "4S", "3D"]),
        ("Cu", 19, [0, 0, 1, 2], ["3S", "3P", "4S", "3D"]),
    ),
)
def test_infer_pseudo_orbitals(element, z_valence, chi, expected):
    """Aufbau inference reproduces the curated-table labels and passes the
    cross-check against a UPF whose angular momenta agree."""
    from aiida_wannier90_workflows.utils.pseudo import _infer_pseudo_orbitals

    pseudo = FakePseudo(element, z_valence, _upf_content(element, z_valence, chi))
    entry = _infer_pseudo_orbitals(pseudo)
    assert entry is not None
    assert entry["pswfcs"] == expected
    assert entry["semicores"] == []


def test_infer_pseudo_orbitals_fully_relativistic():
    """A fully-relativistic pseudo lists one wavefunction per j channel; the
    j-split is collapsed before comparing against the per-shell inference."""
    from aiida_wannier90_workflows.utils.pseudo import _infer_pseudo_orbitals

    # 3S (l=0, j=1/2), 3P (l=1, j=1/2), 3P (l=1, j=3/2): raw l-multiset is
    # {0, 1, 1}; only after collapsing the p j-channels does it match the
    # inferred {0, 1}, so a passing result proves the collapse ran.
    content = _upf_content("Si", 4, [(0, 0.5), (1, 0.5), (1, 1.5)], has_so=True)
    entry = _infer_pseudo_orbitals(FakePseudo("Si", 4, content))
    assert entry is not None
    assert entry["pswfcs"] == ["3S", "3P"]


def test_infer_pseudo_orbitals_rejects_contradicting_upf():
    """A heavy element whose aufbau inference disagrees with the UPF's own
    wave functions is downgraded to ``None`` rather than trusted.

    Au (z_valence 11) infers ['4F', '5D'] (l-multiset {2, 3}) from the
    reverse-Madelung walk, but the real pseudo carries 5d/6s wave functions
    (l-multiset {0, 2}); the mismatch must reject the inference.
    """
    from aiida_wannier90_workflows.utils.pseudo import _infer_pseudo_orbitals

    content = _upf_content("Au", 11, [0, 2])
    assert _infer_pseudo_orbitals(FakePseudo("Au", 11, content)) is None


def test_infer_pseudo_orbitals_fails_closed_when_unparseable():
    """When the UPF cannot be parsed for its angular momenta the inference is
    unvalidated and must not be returned (fail closed, not fail open)."""
    from aiida_wannier90_workflows.utils.pseudo import _infer_pseudo_orbitals

    assert _infer_pseudo_orbitals(FakePseudo("Au", 11, "not a upf file")) is None


def test_infer_pseudo_orbitals_without_z_valence():
    """No ``z_valence`` means no inference."""
    from aiida_wannier90_workflows.utils.pseudo import _infer_pseudo_orbitals

    assert _infer_pseudo_orbitals(FakePseudo("Si", None)) is None


def test_get_pseudo_orbitals_overrides_win():
    """An explicit override bypasses tables and inference."""
    from aiida_wannier90_workflows.utils.pseudo import PseudoOrbitals, get_pseudo_orbitals

    override = PseudoOrbitals(pswfcs=["3S", "3P", "4S", "3D"], semicores=["3S", "3P"])
    result = get_pseudo_orbitals({"Ti": FakePseudo("Ti", 12)}, overrides={"Ti": override})
    assert result["Ti"]["semicores"] == ["3S", "3P"]


def test_get_pseudo_orbitals_inference_warns():
    """A pseudo missing from the tables resolves by inference, with a warning
    that states the orbitals were validated against the UPF."""
    from aiida_wannier90_workflows.utils.pseudo import get_pseudo_orbitals

    pseudo = FakePseudo("Si", 4, _upf_content("Si", 4, [0, 1]))
    with pytest.warns(UserWarning, match="validated against the UPF"):
        result = get_pseudo_orbitals({"Si": pseudo})
    assert result["Si"]["pswfcs"] == ["3S", "3P"]
    assert result["Si"]["semicores"] == []


def test_get_pseudo_orbitals_unresolvable_raises():
    """When inference is impossible the error explains the override escape hatch."""
    from aiida_wannier90_workflows.utils.pseudo import get_pseudo_orbitals

    with pytest.raises(ValueError, match="overrides"):
        get_pseudo_orbitals({"Xx": FakePseudo("Xx", None)})


def test_get_pseudo_orbitals_heavy_element_raises():
    """A heavy element whose inference contradicts its UPF is unresolvable and
    raises rather than silently yielding a wrong orbital set."""
    from aiida_wannier90_workflows.utils.pseudo import get_pseudo_orbitals

    pseudo = FakePseudo("Au", 11, _upf_content("Au", 11, [0, 2]))
    with pytest.raises(ValueError, match="overrides"):
        get_pseudo_orbitals({"Au": pseudo})
