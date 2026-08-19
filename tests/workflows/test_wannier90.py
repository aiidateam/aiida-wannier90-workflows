"""Tests for the `Wannier90WorkChain` class."""

import io

from plumpy.process_states import ProcessState
import pytest

from aiida import orm
from aiida.common import AttributeDict, LinkType

from aiida_quantumespresso.calculations.helpers import pw_input_helper

from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain


def test_scdm(
    generate_workchain_wannier90,
    fixture_localhost,
    generate_remote_data,
    generate_bands_data,
    generate_projection_data,
    generate_calc_job_node,
):  # pylint: disable=redefined-outer-name,too-many-statements
    """Test instantiating the WorkChain, then mock its process, by calling methods in the ``spec.outline``."""

    workchain = generate_workchain_wannier90()
    assert workchain.setup() is None

    # run scf
    scf_workchain = workchain.run_scf()["workchain_scf"]

    # mock scf outputs
    remote = generate_remote_data(
        computer=fixture_localhost, remote_path="/path/on/remote"
    )
    remote.store()
    remote.base.links.add_incoming(
        scf_workchain, link_type=LinkType.RETURN, link_label="remote_folder"
    )

    params = orm.Dict({"fermi_energy": 6.0, "number_of_electrons": 8})
    params.store()
    params.base.links.add_incoming(
        scf_workchain, link_type=LinkType.RETURN, link_label="output_parameters"
    )

    scf_workchain.set_process_state(ProcessState.FINISHED)
    scf_workchain.set_exit_status(0)
    workchain.ctx.workchain_scf = scf_workchain

    pw_input_helper(
        scf_workchain.inputs.pw.parameters.get_dict(), scf_workchain.inputs.pw.structure
    )
    assert workchain.inspect_scf() is None
    assert workchain.ctx.current_folder == remote

    # run nscf
    nscf_workchain = workchain.run_nscf()["workchain_nscf"]

    # mock nscf outputs
    remote = generate_remote_data(
        computer=fixture_localhost, remote_path="/path/on/remote"
    )
    remote.store()
    remote.base.links.add_incoming(
        nscf_workchain, link_type=LinkType.RETURN, link_label="remote_folder"
    )

    nscf_workchain.set_process_state(ProcessState.FINISHED)
    nscf_workchain.set_exit_status(0)
    workchain.ctx.workchain_nscf = nscf_workchain

    pw_input_helper(
        nscf_workchain.inputs.pw.parameters.get_dict(),
        nscf_workchain.inputs.pw.structure,
    )
    assert (
        workchain.ctx.workchain_nscf.inputs.pw.parent_folder
        == workchain.ctx.workchain_scf.outputs.remote_folder
    )
    assert workchain.inspect_nscf() is None
    assert workchain.ctx.current_folder == remote

    # mock run projwfc
    projwfc_workchain = workchain.run_projwfc()["workchain_projwfc"]

    # mock projwfc outputs
    bands_data = generate_bands_data()
    bands_data.store()
    bands_data.base.links.add_incoming(
        projwfc_workchain, link_type=LinkType.RETURN, link_label="bands"
    )

    # Set 8 orbitals for workchain.sanity_check()
    projection_data = generate_projection_data(8)
    projection_data.store()
    projection_data.base.links.add_incoming(
        projwfc_workchain, link_type=LinkType.RETURN, link_label="projections"
    )

    projwfc_workchain.set_process_state(ProcessState.FINISHED)
    projwfc_workchain.set_exit_status(0)
    workchain.ctx.workchain_projwfc = projwfc_workchain

    assert (
        workchain.ctx.workchain_projwfc.inputs.projwfc.parent_folder
        == workchain.ctx.workchain_nscf.outputs.remote_folder
    )
    assert workchain.inspect_projwfc() is None

    # mock run wannier90 pp
    w90pp_workchain = workchain.run_wannier90_pp()["workchain_wannier90_pp"]

    # The wannier90 step will use `get_last_calcjob` to retrieve input parameters of the calcjob
    entry_point_calc_job = "wannier90.wannier90"
    calcjob = generate_calc_job_node(
        entry_point_calc_job,
        fixture_localhost,
        inputs={"parameters": orm.Dict()},
        store=False,
    )
    calcjob.set_process_state(ProcessState.FINISHED)
    calcjob.set_exit_status(0)
    calcjob.base.links.add_incoming(
        workchain.inputs.structure,
        link_type=LinkType.INPUT_CALC,
        link_label="structure",
    )
    calcjob.base.links.add_incoming(
        w90pp_workchain, link_type=LinkType.CALL_CALC, link_label="iteration_01"
    )
    calcjob.store()

    assert w90pp_workchain.called_descendants == [calcjob]

    # mock wannier90 outputs
    nnkp_file = orm.SinglefileData(io.BytesIO(b"content"))
    nnkp_file.store()
    nnkp_file.base.links.add_incoming(
        w90pp_workchain, link_type=LinkType.RETURN, link_label="nnkp_file"
    )

    w90pp_workchain.set_process_state(ProcessState.FINISHED)
    w90pp_workchain.set_exit_status(0)
    workchain.ctx.workchain_wannier90_pp = w90pp_workchain

    assert workchain.inspect_wannier90_pp() is None

    # mock run pw2wannier90
    pw2wan_workchain = workchain.run_pw2wannier90()["workchain_pw2wannier90"]

    # mock pw2wannier90 outputs
    remote = generate_remote_data(
        computer=fixture_localhost, remote_path="/path/on/remote"
    )
    remote.store()
    remote.base.links.add_incoming(
        pw2wan_workchain, link_type=LinkType.RETURN, link_label="remote_folder"
    )

    pw2wan_workchain.set_process_state(ProcessState.FINISHED)
    pw2wan_workchain.set_exit_status(0)
    workchain.ctx.workchain_pw2wannier90 = pw2wan_workchain

    assert (
        workchain.ctx.workchain_pw2wannier90.inputs.pw2wannier90.parent_folder
        == workchain.ctx.workchain_nscf.outputs.remote_folder
    )
    assert workchain.inspect_pw2wannier90() is None
    assert workchain.ctx.current_folder == remote

    # mock run wannier90
    w90_workchain = workchain.run_wannier90()["workchain_wannier90"]

    # mock wannier90 outputs
    remote = generate_remote_data(
        computer=fixture_localhost, remote_path="/path/on/remote"
    )
    remote.store()
    remote.base.links.add_incoming(
        w90_workchain, link_type=LinkType.RETURN, link_label="remote_folder"
    )

    w90_workchain.set_process_state(ProcessState.FINISHED)
    w90_workchain.set_exit_status(0)
    workchain.ctx.workchain_wannier90 = w90_workchain

    assert workchain.inspect_wannier90() is None
    assert workchain.ctx.current_folder == remote

    assert workchain.results() is None

    assert all(
        _ in workchain.outputs
        for _ in ("scf", "nscf", "projwfc", "wannier90_pp", "pw2wannier90", "wannier90")
    )


def _install_submit_capture(workchain):
    """Monkeypatch ``workchain.submit`` to also record each child's submitted metadata.

    The synchronous ``instantiate_process`` test harness never registers
    ``workchain`` on plumpy's process stack, so ``self.submit()`` never
    creates the ``CALL_WORK`` link that would normally carry
    ``call_link_label`` in the database -- there's nothing to inspect after
    the fact. Capturing the raw ``metadata`` at the point of submission
    tests the exact same code, keyed by the label so a caller can look up
    what its call actually carried.

    :return: a dict, populated as submissions happen, mapping each child's
        ``call_link_label`` to its full submitted ``metadata`` dict.
    """
    captured = {}
    original_submit = workchain.submit

    def _submit(process_class, **kwargs):
        metadata = dict(kwargs.get("metadata") or {})
        captured[metadata.get("call_link_label")] = metadata
        return original_submit(process_class, **kwargs)

    workchain.submit = _submit
    return captured


def _run_through_wannier90_pp(
    workchain, generate_remote_data, fixture_localhost, generate_calc_job_node
):
    """Mock the scf step and drive ``workchain`` through ``run_wannier90_pp``.

    Sets ``ctx.workchain_wannier90_pp`` and ``ctx.current_folder`` so a
    subsequent ``run_wannier90()`` call has what ``prepare_wannier90_inputs``
    needs, without mocking the nscf/projwfc/pw2wannier90 steps this fix
    doesn't touch.

    :return: the submitted preprocessing `Wannier90BaseWorkChain` node, and the
        dict from :func:`_install_submit_capture`.
    """
    captured = _install_submit_capture(workchain)
    assert workchain.setup() is None

    scf_workchain = workchain.run_scf()["workchain_scf"]
    remote = generate_remote_data(
        computer=fixture_localhost, remote_path="/path/on/remote"
    )
    remote.store()
    remote.base.links.add_incoming(
        scf_workchain, link_type=LinkType.RETURN, link_label="remote_folder"
    )
    params = orm.Dict({"fermi_energy": 6.0, "number_of_electrons": 8})
    params.store()
    params.base.links.add_incoming(
        scf_workchain, link_type=LinkType.RETURN, link_label="output_parameters"
    )
    scf_workchain.set_process_state(ProcessState.FINISHED)
    scf_workchain.set_exit_status(0)
    workchain.ctx.workchain_scf = scf_workchain
    pw_input_helper(
        scf_workchain.inputs.pw.parameters.get_dict(), scf_workchain.inputs.pw.structure
    )
    assert workchain.inspect_scf() is None

    w90pp_workchain = workchain.run_wannier90_pp()["workchain_wannier90_pp"]

    entry_point_calc_job = "wannier90.wannier90"
    calcjob = generate_calc_job_node(
        entry_point_calc_job,
        fixture_localhost,
        inputs={"parameters": orm.Dict()},
        store=False,
    )
    calcjob.set_process_state(ProcessState.FINISHED)
    calcjob.set_exit_status(0)
    calcjob.base.links.add_incoming(
        workchain.inputs.structure,
        link_type=LinkType.INPUT_CALC,
        link_label="structure",
    )
    calcjob.base.links.add_incoming(
        w90pp_workchain, link_type=LinkType.CALL_CALC, link_label="iteration_01"
    )
    calcjob.store()

    nnkp_file = orm.SinglefileData(io.BytesIO(b"content"))
    nnkp_file.store()
    nnkp_file.base.links.add_incoming(
        w90pp_workchain, link_type=LinkType.RETURN, link_label="nnkp_file"
    )
    w90pp_workchain.set_process_state(ProcessState.FINISHED)
    w90pp_workchain.set_exit_status(0)
    workchain.ctx.workchain_wannier90_pp = w90pp_workchain
    assert workchain.inspect_wannier90_pp() is None

    workchain.ctx.current_folder = remote

    return w90pp_workchain, captured


def test_metadata_survives_shared_wannier90_namespace(
    generate_workchain,
    generate_inputs_wannier90,
    fixture_localhost,
    generate_remote_data,
    generate_calc_job_node,
):
    """A caller-set ``wannier90.metadata`` label reaches both wannier90.x steps.

    Both the preprocessing (``wannier90_pp``) and minimization (``wannier90``)
    runs build from the one exposed ``wannier90`` namespace when no
    ``wannier90_pp`` override is given, so they carry the identical
    caller-set label. This pins the metadata-preservation fix, ambiguity
    and all -- it does not claim the two steps become distinguishable.
    """
    inputs = generate_inputs_wannier90()
    inputs["wannier90"]["metadata"] = {
        "label": "MLWF run",
        "description": "shared label",
    }

    workchain = generate_workchain("wannier90_workflows.wannier90", inputs)
    w90pp_workchain, captured = _run_through_wannier90_pp(
        workchain, generate_remote_data, fixture_localhost, generate_calc_job_node
    )

    # The stored node proves the label survived to persistence; the
    # captured submission proves ``call_link_label`` still reads
    # ``wannier90_pp`` (see ``_install_submit_capture`` for why this can't
    # be read back off the node's incoming link in this test harness).
    assert w90pp_workchain.label == "MLWF run"
    assert w90pp_workchain.description == "shared label"
    assert captured["wannier90_pp"]["call_link_label"] == "wannier90_pp"

    w90_workchain = workchain.run_wannier90()["workchain_wannier90"]
    assert w90_workchain.label == "MLWF run"
    assert w90_workchain.description == "shared label"
    assert captured["wannier90"]["call_link_label"] == "wannier90"


def test_wannier90_pp_namespace_gives_distinct_label(
    generate_workchain,
    generate_inputs_wannier90,
    fixture_localhost,
    generate_remote_data,
    generate_calc_job_node,
):
    """A caller-set ``wannier90_pp.metadata`` names the preprocessing node independently.

    ``wannier90_pp`` is optional and metadata-only; when a caller sets its
    ``metadata``, ``prepare_wannier90_pp_inputs`` swaps that in for the
    preprocessing run while still building every physics input off
    ``wannier90`` -- see ``test_wannier90_pp_metadata_leaves_physics_unchanged``
    for that half. Here: the two wannier90.x steps can carry distinct
    labels, the point of adding the namespace.
    """
    inputs = generate_inputs_wannier90()
    inputs["wannier90"]["metadata"] = {"label": "Minimization"}
    inputs["wannier90_pp"] = {"metadata": {"label": "Preprocessing"}}

    workchain = generate_workchain("wannier90_workflows.wannier90", inputs)
    assert "wannier90_pp" in workchain.inputs

    w90pp_workchain, captured = _run_through_wannier90_pp(
        workchain, generate_remote_data, fixture_localhost, generate_calc_job_node
    )
    assert w90pp_workchain.label == "Preprocessing"
    assert captured["wannier90_pp"]["call_link_label"] == "wannier90_pp"

    w90_workchain = workchain.run_wannier90()["workchain_wannier90"]
    assert w90_workchain.label == "Minimization"
    assert captured["wannier90"]["call_link_label"] == "wannier90"


def test_wannier90_pp_metadata_leaves_physics_unchanged(
    generate_workchain, generate_inputs_wannier90
):
    """Setting ``wannier90_pp.metadata`` doesn't touch the preprocessing run's physics inputs.

    Physics inputs come from ``wannier90`` unconditionally now -- there's no
    branch left that could read them from ``wannier90_pp`` (see
    ``test_wannier90_pp_rejects_physics_inputs``, which pins that the spec
    itself refuses to accept them there). This is the construction
    guarantee that check exists for: even with ``wannier90_pp`` populated,
    ``prepare_wannier90_pp_inputs``'s code/parameters still match
    ``wannier90``'s directly.
    """
    inputs = generate_inputs_wannier90()
    inputs["wannier90"]["wannier90"]["parameters"] = orm.Dict({"fermi_energy": 6.0})
    inputs["wannier90_pp"] = {"metadata": {"label": "Preprocessing"}}

    workchain = generate_workchain("wannier90_workflows.wannier90", inputs)
    assert workchain.setup() is None
    pp_inputs = workchain.prepare_wannier90_pp_inputs()
    direct_inputs = AttributeDict(
        workchain.exposed_inputs(Wannier90BaseWorkChain, namespace="wannier90")
    )
    pp_calc, direct_calc = pp_inputs["wannier90"], direct_inputs["wannier90"]

    assert pp_calc["code"].uuid == direct_calc["code"].uuid
    # `prepare_wannier90_pp_inputs` always re-derives `parameters` (it
    # (re-)injects `fermi_energy`, here already present with the same
    # value), so content rather than node identity is the fair comparison.
    assert pp_calc["parameters"].get_dict() == direct_calc["parameters"].get_dict()
    assert pp_inputs["metadata"]["label"] == "Preprocessing"


def test_wannier90_pp_rejects_physics_inputs(
    generate_workchain, generate_inputs_wannier90, fixture_code
):
    """``wannier90_pp`` only accepts ``metadata`` -- physics inputs are rejected outright.

    The pp and minimization runs are two phases of one calculation (the
    ``.nnkp`` preprocessing writes is what pw2wannier90 and the
    minimization consume), so divergent physics between them would be a
    silent-wrong-results bug class. Making the spec itself refuse a
    ``wannier90_pp.wannier90`` override -- rather than merely not reading
    one -- is what makes that class of bug unrepresentable.

    The override below is a *complete*, otherwise-valid
    ``Wannier90BaseWorkChain`` input set (code, parameters, kpoints) --
    before this namespace was restricted to metadata-only, the spec
    accepted exactly this and built the preprocessing run from it,
    silently, instead of ``wannier90``. An incomplete override (missing
    `kpoints`, say) would raise a "required value was not provided" error
    on both the old and new spec, which wouldn't discriminate between them.
    """
    inputs = generate_inputs_wannier90()
    inputs["wannier90_pp"] = {
        "metadata": {"label": "Preprocessing"},
        "wannier90": {
            "code": fixture_code("wannier90.wannier90"),
            "parameters": orm.Dict({"fermi_energy": 99.0}),
            "kpoints": inputs["wannier90"]["wannier90"]["kpoints"],
        },
    }

    with pytest.raises(ValueError, match="Unexpected ports"):
        generate_workchain("wannier90_workflows.wannier90", inputs)
