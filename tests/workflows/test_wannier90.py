"""Tests for the `Wannier90WorkChain` class."""
import io

from plumpy.process_states import ProcessState

from aiida import orm
from aiida.common import LinkType

from aiida_quantumespresso.calculations.helpers import pw_input_helper


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
