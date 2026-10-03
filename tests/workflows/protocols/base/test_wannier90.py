"""Tests for the ``Wannier90BaseWorkChain.get_builder_from_protocol`` method."""

import pytest

from aiida.engine import ProcessBuilder

from aiida_quantumespresso.common.types import ElectronicType, SpinType

from aiida_wannier90_workflows.common.types import (
    WannierDisentanglementType,
    WannierFrozenType,
    WannierProjectionType,
)
from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain


def test_get_available_protocols():
    """Test ``Wannier90BaseWorkChain.get_available_protocols``."""
    protocols = Wannier90BaseWorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ["fast", "moderate", "precise"]
    assert all("description" in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``Wannier90BaseWorkChain.get_default_protocol``."""
    assert Wannier90BaseWorkChain.get_default_protocol() == "moderate"


def test_default(fixture_code, generate_structure, data_regression, serialize_builder):
    """Test ``Wannier90BaseWorkChain.get_builder_from_protocol`` for the default protocol."""
    code = fixture_code("wannier90.wannier90")
    structure = generate_structure("Si")
    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code, structure=structure
    )

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize(
    "electronic_type", (ElectronicType.METAL, ElectronicType.INSULATOR)
)
def test_electronic_type(
    fixture_code,
    generate_structure,
    data_regression,
    serialize_builder,
    electronic_type,
):
    """Test ``Wannier90BaseWorkChain.get_builder_from_protocol`` with ``electronic_type`` keyword."""
    code = fixture_code("wannier90.wannier90")
    structure = generate_structure("Si")

    with pytest.raises(NotImplementedError):
        Wannier90BaseWorkChain.get_builder_from_protocol(
            code, structure=structure, electronic_type=ElectronicType.AUTOMATIC
        )

    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code, structure=structure, electronic_type=electronic_type
    )

    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize("spin_type", (SpinType.NONE, SpinType.SPIN_ORBIT))
def test_spin_type(
    fixture_code, generate_structure, data_regression, serialize_builder, spin_type
):
    """Test ``Wannier90BaseWorkChain.get_builder_from_protocol`` with ``spin_type`` keyword."""
    code = fixture_code("wannier90.wannier90")
    structure = generate_structure("Si")

    if spin_type == SpinType.SPIN_ORBIT:
        with pytest.raises(
            ValueError, match="Need to explicitly specify `pseudo_family`"
        ):
            builder = Wannier90BaseWorkChain.get_builder_from_protocol(
                code, structure=structure, spin_type=spin_type
            )
        return

    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code, structure=structure, spin_type=spin_type
    )

    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize(
    "projection_type",
    (WannierProjectionType.ATOMIC_PROJECTORS_QE, WannierProjectionType.SCDM),
)
def test_projection_type(
    fixture_code,
    generate_structure,
    data_regression,
    serialize_builder,
    projection_type,
):
    """Test ``Wannier90BaseWorkChain.get_builder_from_protocol`` with invalid ``initial_magnetic_moments`` keyword."""
    code = fixture_code("wannier90.wannier90")
    structure = generate_structure("Si")

    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code, structure=structure, projection_type=projection_type
    )

    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize(
    "disentanglement_type",
    (WannierDisentanglementType.NONE, WannierDisentanglementType.SMV),
)
def test_disentanglement_type(
    fixture_code,
    generate_structure,
    data_regression,
    serialize_builder,
    disentanglement_type,
):
    """Test ``Wannier90BaseWorkChain.get_builder_from_protocol`` with ``initial_magnetic_moments`` keyword."""
    code = fixture_code("wannier90.wannier90")
    structure = generate_structure("Si")

    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code, structure=structure, disentanglement_type=disentanglement_type
    )

    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize(
    "frozen_type",
    (
        WannierFrozenType.NONE,
        WannierFrozenType.ENERGY_AUTO,
        WannierFrozenType.ENERGY_FIXED,
        WannierFrozenType.FIXED_PLUS_PROJECTABILITY,
        WannierFrozenType.PROJECTABILITY,
    ),
)
def test_frozen_type(
    fixture_code, generate_structure, data_regression, serialize_builder, frozen_type
):
    """Test magnetization ``overrides`` for the ``Wannier90BaseWorkChain.get_builder_from_protocol`` method."""
    code = fixture_code("wannier90.wannier90")
    structure = generate_structure("Si")

    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code, structure=structure, frozen_type=frozen_type
    )

    data_regression.check(serialize_builder(builder))


def test_parameter_overrides(
    fixture_code, generate_structure, data_regression, serialize_builder
):
    """Test specifying parameter ``overrides`` for the ``get_builder_from_protocol()`` method."""
    code = fixture_code("wannier90.wannier90")
    structure = generate_structure("Si")

    overrides = {"wannier90": {"parameters": {"fake_input": "fake"}}}
    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code, structure=structure, overrides=overrides
    )

    data_regression.check(serialize_builder(builder))


def test_settings_overrides(
    fixture_code, generate_structure, data_regression, serialize_builder
):
    """Test specifying settings ``overrides`` for the ``get_builder_from_protocol()`` method."""
    code = fixture_code("wannier90.wannier90")
    structure = generate_structure("Si")

    overrides = {"wannier90": {"settings": {"cmdline": ["-nk", 6]}}}
    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code, structure=structure, overrides=overrides
    )

    data_regression.check(serialize_builder(builder))


def test_metadata_overrides(
    fixture_code, generate_structure, data_regression, serialize_builder
):
    """Test specifying metadata ``overrides`` for the ``get_builder_from_protocol()`` method."""
    code = fixture_code("wannier90.wannier90")
    structure = generate_structure("Si")

    overrides = {
        "wannier90": {
            "metadata": {
                "options": {
                    "resources": {"num_machines": 1e90},
                    "max_wallclock_seconds": 1,
                }
            }
        }
    }
    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code,
        structure=structure,
        overrides=overrides,
    )

    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize(
    "family_fixture", ("cutoffs_family_without_stringency", "plain_pseudo_family")
)
def test_pseudo_family_without_cutoffs(
    fixture_code, generate_structure, request, family_fixture
):
    """A pseudo family that recommends no cutoffs builds the same inputs.

    This builder counts bands and projections from the pseudos and never uses
    the cutoffs, so a family that has none must serve it as well as one that
    does.
    """
    code = fixture_code("wannier90.wannier90")
    structure = generate_structure("Si")
    family = request.getfixturevalue(family_fixture)

    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code, structure=structure, pseudo_family=family.label
    )
    reference = Wannier90BaseWorkChain.get_builder_from_protocol(
        code, structure=structure
    )

    assert isinstance(builder, ProcessBuilder)
    assert (
        builder.wannier90.parameters.get_dict()
        == reference.wannier90.parameters.get_dict()
    )


def test_analytic_without_projections_requires_orbitals(
    fixture_code, generate_structure, pseudo_family_without_pswfc
):
    """Negative control: ``ANALYTIC`` without an explicit projection list
    still needs the pseudos' valence orbitals, and fails closed with the
    existing message when a pseudopotential carries none -- even with
    ``exclude_semicore`` off, isolating the failure to ``ANALYTIC``'s own
    derivation rather than the separate ``exclude_semicore`` lookup."""
    code = fixture_code("wannier90.wannier90")
    structure = generate_structure("Si")

    with pytest.raises(ValueError, match="valence orbitals could not be read"):
        Wannier90BaseWorkChain.get_builder_from_protocol(
            code,
            structure=structure,
            pseudo_family=pseudo_family_without_pswfc.label,
            electronic_type=ElectronicType.INSULATOR,
            projection_type=WannierProjectionType.ANALYTIC,
            overrides={"meta_parameters": {"exclude_semicore": False}},
        )


def test_analytic_explicit_projections_skip_lookup(
    fixture_code, generate_structure, pseudo_family_without_pswfc
):
    """An explicit projection list, given via ``overrides``, is honored as
    given and needs no pseudo-orbital lookup -- the only way to build a
    Wannier90 step for a pseudopotential with no ``PP_PSWFC`` content."""
    code = fixture_code("wannier90.wannier90")
    structure = generate_structure("Si")
    given = ["Si:s", "Si:p"]

    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code,
        structure=structure,
        pseudo_family=pseudo_family_without_pswfc.label,
        electronic_type=ElectronicType.INSULATOR,
        projection_type=WannierProjectionType.ANALYTIC,
        overrides={
            "meta_parameters": {"exclude_semicore": False},
            "wannier90": {"projections": given},
        },
    )

    assert builder.wannier90.projections.get_list() == given


def test_analytic_resolvable_pseudo_unaffected(
    fixture_code, generate_structure, plain_pseudo_family
):
    """A pseudopotential whose orbitals are resolvable still derives its
    projections the same way: the new path only takes over when ``overrides``
    already supply an explicit list."""
    code = fixture_code("wannier90.wannier90")
    structure = generate_structure("Si")

    builder = Wannier90BaseWorkChain.get_builder_from_protocol(
        code,
        structure=structure,
        pseudo_family=plain_pseudo_family.label,
        electronic_type=ElectronicType.INSULATOR,
        projection_type=WannierProjectionType.ANALYTIC,
    )

    assert builder.wannier90.projections.get_list() == ["Si:s", "Si:p"]
