# pylint: disable=redefined-outer-name
"""Tests for the ``Wannier90WorkChain.get_builder_from_protocol`` method."""
import pytest

from aiida.engine import ProcessBuilder
from aiida.plugins import WorkflowFactory

from aiida_quantumespresso.common.types import ElectronicType, SpinType

from aiida_wannier90_workflows.common.types import (
    WannierFrozenType,
    WannierProjectionType,
)

Wannier90WorkChain = WorkflowFactory("wannier90_workflows.wannier90")


def test_get_available_protocols():
    """Test ``Wannier90WorkChain.get_available_protocols``."""
    protocols = Wannier90WorkChain.get_available_protocols()
    assert sorted(protocols.keys()) == ["fast", "moderate", "precise"]
    assert all("description" in protocol for protocol in protocols.values())


def test_get_default_protocol():
    """Test ``Wannier90WorkChain.get_default_protocol``."""
    assert Wannier90WorkChain.get_default_protocol() == "moderate"


@pytest.mark.parametrize("structure", ("Si", "H2O", "GaAs", "BaTiO3"))
def test_scdm(generate_builder_inputs, data_regression, serialize_builder, structure):
    """Test ``Wannier90WorkChain.get_builder_from_protocol`` for the default protocol."""

    inputs = generate_builder_inputs(structure)
    builder = Wannier90WorkChain.get_builder_from_protocol(
        **inputs, print_summary=False
    )

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize("structure", ("Si", "H2O", "GaAs", "BaTiO3"))
def test_atomic_projectors_qe(
    generate_builder_inputs, data_regression, serialize_builder, structure
):
    """Test ``Wannier90WorkChain.get_builder_from_protocol`` for the default protocol."""

    inputs = generate_builder_inputs(structure)
    builder = Wannier90WorkChain.get_builder_from_protocol(
        **inputs,
        projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
        print_summary=False,
    )

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize("structure", ("Si", "H2O", "GaAs", "BaTiO3"))
def test_spin_orbit(
    generate_builder_inputs, data_regression, serialize_builder, structure
):
    """Test ``Wannier90WorkChain.get_builder_from_protocol`` for the default protocol."""

    inputs = generate_builder_inputs(structure)
    builder = Wannier90WorkChain.get_builder_from_protocol(
        **inputs,
        spin_type=SpinType.SPIN_ORBIT,
        print_summary=False,
    )

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_electronic_type(generate_builder_inputs):
    """Test ``Wannier90WorkChain.get_builder_from_protocol`` with ``electronic_type`` keyword."""
    with pytest.raises(NotImplementedError):
        builder = Wannier90WorkChain.get_builder_from_protocol(
            **generate_builder_inputs(),
            electronic_type=ElectronicType.AUTOMATIC,
            print_summary=False,
        )

    builder = Wannier90WorkChain.get_builder_from_protocol(
        **generate_builder_inputs(),
        electronic_type=ElectronicType.INSULATOR,
        print_summary=False,
    )
    for namespace, occupations in zip((builder.scf, builder.nscf), ("fixed", "fixed")):
        parameters = namespace["pw"]["parameters"].get_dict()
        assert parameters["SYSTEM"]["occupations"] == occupations
        assert "degauss" not in parameters["SYSTEM"]
        assert "smearing" not in parameters["SYSTEM"]

    builder = Wannier90WorkChain.get_builder_from_protocol(
        **generate_builder_inputs(),
        electronic_type=ElectronicType.METAL,
        print_summary=False,
    )
    for namespace, occupations in zip(
        (builder.scf, builder.nscf), ("smearing", "smearing")
    ):
        parameters = namespace["pw"]["parameters"].get_dict()
        assert parameters["SYSTEM"]["occupations"] == occupations
        assert "degauss" in parameters["SYSTEM"]
        assert "smearing" in parameters["SYSTEM"]


def test_spin_type(generate_builder_inputs):
    """Test ``Wannier90WorkChain.get_builder_from_protocol`` with ``spin_type`` keyword."""
    # with pytest.raises(NotImplementedError):
    #     for spin_type in [SpinType.COLLINEAR, SpinType.NON_COLLINEAR]:
    #         builder = Wannier90WorkChain.get_builder_from_protocol(
    #             **generate_builder_inputs(), spin_type=spin_type, print_summary=False
    #         )
    # TODO: add tests for collinear and noncollinears

    builder = Wannier90WorkChain.get_builder_from_protocol(
        **generate_builder_inputs(), spin_type=SpinType.NONE, print_summary=False
    )
    for namespace in [builder.scf, builder.nscf]:
        parameters = namespace["pw"]["parameters"].get_dict()
        assert "nspin" not in parameters["SYSTEM"]
        assert "starting_magnetization" not in parameters["SYSTEM"]

    builder = Wannier90WorkChain.get_builder_from_protocol(
        **generate_builder_inputs(), spin_type=SpinType.SPIN_ORBIT, print_summary=False
    )
    for namespace in [builder.scf, builder.nscf]:
        parameters = namespace["pw"]["parameters"].get_dict()
        assert parameters["SYSTEM"]["lspinorb"] is True
        assert parameters["SYSTEM"]["noncolin"] is True


def test_projection_type(generate_builder_inputs):
    """Test ``Wannier90WorkChain.get_builder_from_protocol`` with ``projection_type`` keyword."""
    # with pytest.raises(NotImplementedError):
    #     for projection_type in [
    #         WannierProjectionType.ANALYTIC, WannierProjectionType.RANDOM,
    #         WannierProjectionType.ATOMIC_PROJECTORS_EXTERNAL
    #     ]:
    #         builder = Wannier90WorkChain.get_builder_from_protocol(
    #             **generate_builder_inputs(), projection_type=projection_type, print_summary=False
    #         )

    builder = Wannier90WorkChain.get_builder_from_protocol(
        **generate_builder_inputs(),
        projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
        print_summary=False,
    )
    for namespace in [
        builder.wannier90,
    ]:
        parameters = namespace["wannier90"]["parameters"].get_dict()
        assert "auto_projections" in parameters

    builder = Wannier90WorkChain.get_builder_from_protocol(
        **generate_builder_inputs(),
        projection_type=WannierProjectionType.ANALYTIC,
        print_summary=False,
    )
    for namespace in [
        builder.wannier90,
    ]:
        assert "projections" in namespace["wannier90"]
        assert namespace["wannier90"]["projections"].get_list() == ["Si:s", "Si:p"]


def test_force_parity(generate_builder_inputs, data_regression, serialize_builder):
    """Test ``Wannier90WorkChain.get_builder_from_protocol`` for the force_parity."""

    inputs = generate_builder_inputs("Si")

    overrides = {"wannier90": {"meta_parameters": {"kpoints_force_parity": True}}}
    builder = Wannier90WorkChain.get_builder_from_protocol(
        **inputs, overrides=overrides, print_summary=False
    )

    assert isinstance(builder, ProcessBuilder)
    data_regression.check(serialize_builder(builder))


def test_parent_folders_mutually_exclusive(
    generate_builder_inputs, generate_remote_data, fixture_localhost
):
    """Test passing both parent folders is rejected."""
    remote = generate_remote_data(fixture_localhost, "/tmp", "quantumespresso.pw")

    with pytest.raises(ValueError, match=r"mutually exclusive"):
        Wannier90WorkChain.get_builder_from_protocol(
            **generate_builder_inputs(),
            scf_parent_folder=remote,
            nscf_parent_folder=remote,
            print_summary=False,
        )


def test_pw_code_required_by_default(generate_builder_inputs):
    """Test the `pw` code stays required when the scf/nscf namespaces are populated."""
    inputs = generate_builder_inputs()
    inputs["codes"].pop("pw")

    with pytest.raises(ValueError, match=r"does not contain the required key: pw"):
        Wannier90WorkChain.get_builder_from_protocol(**inputs, print_summary=False)


def test_scf_parent_folder(
    generate_builder_inputs, generate_remote_data, fixture_localhost
):
    """Test an scf parent folder drops the scf namespace but keeps the nscf one."""
    remote = generate_remote_data(fixture_localhost, "/tmp", "quantumespresso.pw")

    builder = Wannier90WorkChain.get_builder_from_protocol(
        **generate_builder_inputs(),
        scf_parent_folder=remote,
        print_summary=False,
    )

    inputs = builder._inputs(prune=True)  # pylint: disable=protected-access
    assert "scf" not in inputs
    assert "nscf" in inputs
    assert inputs["nscf"]["pw"]["parent_folder"] is remote

    # The nscf still runs pw.x, so the `pw` code remains required.
    codes = generate_builder_inputs()
    codes["codes"].pop("pw")
    with pytest.raises(ValueError, match=r"does not contain the required key: pw"):
        Wannier90WorkChain.get_builder_from_protocol(
            **codes, scf_parent_folder=remote, print_summary=False
        )


def test_scf_parent_folder_validates(
    generate_builder_inputs, generate_remote_data, fixture_localhost
):
    """Test the workchain accepts an scf-less builder without further wiring."""
    remote = generate_remote_data(fixture_localhost, "/tmp", "quantumespresso.pw")

    builder = Wannier90WorkChain.get_builder_from_protocol(
        **generate_builder_inputs(),
        scf_parent_folder=remote,
        print_summary=False,
    )

    assert (
        Wannier90WorkChain.spec().inputs.validate(
            builder._inputs(prune=True)  # pylint: disable=protected-access
        )
        is None
    )


def test_nscf_parent_folder(
    generate_builder_inputs, generate_remote_data, fixture_localhost
):
    """Test an nscf parent folder drops both pw namespaces and the `pw` code."""
    remote = generate_remote_data(fixture_localhost, "/tmp", "quantumespresso.pw")

    inputs = generate_builder_inputs()
    inputs["codes"].pop("pw")

    builder = Wannier90WorkChain.get_builder_from_protocol(
        **inputs,
        nscf_parent_folder=remote,
        exclude_semicore=False,
        print_summary=False,
    )

    assert isinstance(builder, ProcessBuilder)

    pruned = builder._inputs(prune=True)  # pylint: disable=protected-access
    assert "scf" not in pruned
    assert "nscf" not in pruned
    assert "wannier90" in pruned
    assert pruned["pw2wannier90"]["pw2wannier90"]["parent_folder"] is remote


def test_nscf_parent_folder_validates(
    generate_builder_inputs, generate_remote_data, fixture_localhost
):
    """Test the workchain accepts the builder as returned, with no post-assembly wiring."""
    remote = generate_remote_data(fixture_localhost, "/tmp", "quantumespresso.pw")

    inputs = generate_builder_inputs()
    inputs["codes"].pop("pw")

    builder = Wannier90WorkChain.get_builder_from_protocol(
        **inputs,
        nscf_parent_folder=remote,
        exclude_semicore=False,
        print_summary=False,
    )

    assert (
        Wannier90WorkChain.spec().inputs.validate(
            builder._inputs(prune=True)  # pylint: disable=protected-access
        )
        is None
    )


def _atom_proj_exclude(builder):
    """Return the pw2wannier90 `atom_proj_exclude`, whatever case the namelist key has."""
    parameters = builder.pw2wannier90.pw2wannier90.parameters.get_dict()
    namelist = {key.lower(): value for key, value in parameters.items()}["inputpp"]
    return namelist["atom_proj_exclude"]


@pytest.mark.parametrize("structure", ("BaTiO3", "GaAs"))
def test_nscf_parent_folder_semicore(
    generate_builder_inputs, generate_remote_data, fixture_localhost, structure
):
    """Test the semicore states are the same whether or not a pw namespace is assembled."""
    remote = generate_remote_data(fixture_localhost, "/tmp", "quantumespresso.pw")
    # Atomic projectors carry the semicore list into the pw2wannier90 parameters,
    # where SCDM would not expose it.
    kwargs = {
        "projection_type": WannierProjectionType.ATOMIC_PROJECTORS_QE,
        "print_summary": False,
    }

    default = Wannier90WorkChain.get_builder_from_protocol(
        **generate_builder_inputs(structure), **kwargs
    )

    inputs = generate_builder_inputs(structure)
    inputs["codes"].pop("pw")
    reused = Wannier90WorkChain.get_builder_from_protocol(
        **inputs, nscf_parent_folder=remote, **kwargs
    )

    excluded = _atom_proj_exclude(default)
    # The structures are chosen to have semicore states, so this is not a
    # comparison of two empty lists.
    assert excluded
    assert _atom_proj_exclude(reused) == excluded


def test_nscf_parent_folder_semicore_unknown_family(
    generate_builder_inputs, generate_remote_data, fixture_localhost
):
    """Test an unresolvable pseudopotential family is reported when no pw namespace holds one."""
    remote = generate_remote_data(fixture_localhost, "/tmp", "quantumespresso.pw")
    inputs = generate_builder_inputs()
    inputs["codes"].pop("pw")

    with pytest.raises(
        ValueError, match=r"pseudo family `NoSuchFamily/9.9` is not installed"
    ):
        Wannier90WorkChain.get_builder_from_protocol(
            **inputs,
            nscf_parent_folder=remote,
            pseudo_family="NoSuchFamily/9.9",
            exclude_semicore=True,
            print_summary=False,
        )


def test_nscf_parent_folder_scdm_setup(
    generate_builder_inputs, generate_remote_data, fixture_localhost
):
    """Test the default SCDM projector (which runs projwfc) starts `setup` cleanly.

    With both scf and nscf skipped, `setup` reaches its projwfc branch before its
    pw2wannier90 one, so the builder must wire the reused folder onto `projwfc` too.
    """
    remote = generate_remote_data(fixture_localhost, "/tmp", "quantumespresso.pw")

    inputs = generate_builder_inputs()
    inputs["codes"].pop("pw")

    builder = Wannier90WorkChain.get_builder_from_protocol(
        **inputs,
        nscf_parent_folder=remote,
        exclude_semicore=False,
        print_summary=False,
    )
    pruned = builder._inputs(prune=True)  # pylint: disable=protected-access
    assert pruned["projwfc"]["projwfc"]["parent_folder"] is remote

    process = Wannier90WorkChain(inputs=pruned)
    process.setup()
    assert process.ctx.current_folder is remote


def test_nscf_parent_folder_atomic_projectors_qe_setup(
    generate_builder_inputs, generate_remote_data, fixture_localhost
):
    """Test atomic projectors (no projwfc) still starts `setup` from pw2wannier90.

    `ATOMIC_PROJECTORS_QE` without `ENERGY_AUTO` never populates the `projwfc`
    namespace, so `setup` falls through to its pw2wannier90 branch, unaffected by
    the projwfc wiring added for the SCDM/`ENERGY_AUTO` cases.
    """
    remote = generate_remote_data(fixture_localhost, "/tmp", "quantumespresso.pw")

    inputs = generate_builder_inputs()
    inputs["codes"].pop("pw")

    builder = Wannier90WorkChain.get_builder_from_protocol(
        **inputs,
        nscf_parent_folder=remote,
        exclude_semicore=False,
        projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
        print_summary=False,
    )
    pruned = builder._inputs(prune=True)  # pylint: disable=protected-access
    assert "projwfc" not in pruned
    assert pruned["pw2wannier90"]["pw2wannier90"]["parent_folder"] is remote

    process = Wannier90WorkChain(inputs=pruned)
    process.setup()
    assert process.ctx.current_folder is remote


def test_nscf_parent_folder_energy_auto_setup(
    generate_builder_inputs, generate_remote_data, fixture_localhost
):
    """Test `frozen_type=ENERGY_AUTO` with a non-SCDM projector also runs projwfc.

    This reaches the same `setup` projwfc branch as the SCDM default, via a
    different `run_projwfc` condition in `get_builder_from_protocol`.
    """
    remote = generate_remote_data(fixture_localhost, "/tmp", "quantumespresso.pw")

    inputs = generate_builder_inputs()
    inputs["codes"].pop("pw")

    builder = Wannier90WorkChain.get_builder_from_protocol(
        **inputs,
        nscf_parent_folder=remote,
        exclude_semicore=False,
        projection_type=WannierProjectionType.ATOMIC_PROJECTORS_QE,
        frozen_type=WannierFrozenType.ENERGY_AUTO,
        print_summary=False,
    )
    pruned = builder._inputs(prune=True)  # pylint: disable=protected-access
    assert pruned["projwfc"]["projwfc"]["parent_folder"] is remote

    process = Wannier90WorkChain(inputs=pruned)
    process.setup()
    assert process.ctx.current_folder is remote
