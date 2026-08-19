"""Tests for the `Wannier90WorkChain` class."""

import io

from plumpy.process_states import ProcessState

from aiida import orm
from aiida.common import AttributeDict, LinkType

from aiida_quantumespresso.calculations.helpers import pw_input_helper
from aiida_quantumespresso.utils.resources import get_default_options

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

    :return: the submitted postproc `Wannier90BaseWorkChain` node, and the
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

    Both the postproc (``wannier90_pp``) and minimization (``wannier90``)
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
    fixture_code,
):
    """A caller-set ``wannier90_pp`` namespace names the postproc node independently.

    ``wannier90_pp`` is optional; when a caller supplies it,
    ``prepare_wannier90_pp_inputs`` builds entirely from it instead of
    falling back to ``wannier90``, so the two wannier90.x steps can carry
    distinct labels -- the point of adding the namespace.
    """
    inputs = generate_inputs_wannier90()
    inputs["wannier90"]["metadata"] = {"label": "Minimization"}
    inputs["wannier90_pp"] = {
        "wannier90": {
            "code": fixture_code("wannier90.wannier90"),
            "parameters": orm.Dict({"fermi_energy": 6.0}),
            "kpoints": inputs["wannier90"]["wannier90"]["kpoints"],
            "metadata": {"options": get_default_options()},
        },
        "metadata": {"label": "Preprocessing"},
    }

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


def test_wannier90_pp_namespace_absent_reuses_wannier90_inputs(
    generate_workchain, generate_inputs_wannier90
):
    """With no ``wannier90_pp`` input, the postproc step still builds off ``wannier90`` alone.

    Every existing caller never sets ``wannier90_pp``, so
    ``prepare_wannier90_pp_inputs`` must fall back to exactly the
    ``wannier90`` namespace's own code/kpoints -- unchanged from before this
    namespace existed.
    """
    inputs = generate_inputs_wannier90()
    inputs["wannier90"]["wannier90"]["parameters"] = orm.Dict({"fermi_energy": 6.0})

    workchain = generate_workchain("wannier90_workflows.wannier90", inputs)
    assert "wannier90_pp" not in workchain.inputs

    assert workchain.setup() is None
    pp_inputs = workchain.prepare_wannier90_pp_inputs()
    direct_inputs = AttributeDict(
        workchain.exposed_inputs(Wannier90BaseWorkChain, namespace="wannier90")
    )

    assert (
        pp_inputs["wannier90"]["code"].uuid == direct_inputs["wannier90"]["code"].uuid
    )
    assert (
        pp_inputs["wannier90"]["kpoints"].uuid
        == direct_inputs["wannier90"]["kpoints"].uuid
    )
