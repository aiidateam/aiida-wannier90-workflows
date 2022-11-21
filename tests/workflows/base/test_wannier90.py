"""Tests for the `Wannier90BaseWorkChain` class."""
import pytest

from aiida.common import AttributeDict
from aiida.engine import ProcessHandlerReport

from aiida_wannier90.calculations import Wannier90Calculation

from aiida_wannier90_workflows.workflows.base.wannier90 import Wannier90BaseWorkChain

# pylint: disable=no-member,redefined-outer-name


def test_setup(generate_workchain_wannier90_base):
    """Test `Wannier90BaseWorkChain.setup`."""
    process = generate_workchain_wannier90_base()
    process.setup()

    assert isinstance(process.ctx.inputs, AttributeDict)


def test_prepare_inputs_additional_remote_symlink_list(
    generate_workchain_wannier90_base, generate_inputs_wannier90_base
):
    """Test `Wannier90BaseWorkChain.prepare_inputs` for `additional_remote_symlink_list`."""
    from pathlib import Path

    from aiida import orm

    inputs = {"wannier90": generate_inputs_wannier90_base()}
    remote_symlink_files = ["aiida.hkmn", "aiida.hvmn"]
    inputs["settings"] = orm.Dict(dict={"remote_symlink_files": remote_symlink_files})

    process = generate_workchain_wannier90_base(inputs=inputs)
    process.setup()

    assert isinstance(process.ctx.inputs, AttributeDict)
    assert isinstance(process.ctx.inputs.settings, orm.Dict)

    remote_input_folder = inputs["wannier90"]["remote_input_folder"]
    remote_input_folder_path = Path(remote_input_folder.get_remote_path())
    reference_dict = {
        "additional_remote_symlink_list": [
            (
                remote_input_folder.computer.uuid,
                (remote_input_folder_path / _).as_posix(),
                _,
            )
            for _ in remote_symlink_files
        ]
    }
    settings_dict = process.ctx.inputs.settings.get_dict()
    assert settings_dict == reference_dict


@pytest.mark.parametrize(
    "dis_froz_max",
    (
        (1, 2.82242466),  # LUMO + 1
        (8, 3.98687455),  # min((num_wann-1)-th bands) - 1e-4
    ),
)
def test_prepare_inputs_shift_energy_windows(
    generate_inputs_wannier90_base,
    generate_workchain_wannier90_base,
    generate_bands_data,
    dis_froz_max,
):
    """Test `Wannier90BaseWorkChain.prepare_inputs` for `shift_energy_windows`."""
    from aiida.orm import Bool, Dict

    inputs = generate_inputs_wannier90_base()
    parameters = inputs["parameters"].get_dict()

    # Test SCDM fitting is working
    parameters["fermi_energy"] = 1.2
    parameters["dis_froz_max"] = dis_froz_max[0]
    parameters["num_wann"] = 4
    inputs["parameters"] = Dict(parameters)

    inputs = {"wannier90": inputs}
    inputs["bands"] = generate_bands_data()
    inputs["shift_energy_windows"] = Bool(True)

    process = generate_workchain_wannier90_base(inputs=inputs)
    inputs = process.prepare_inputs()

    assert isinstance(inputs, AttributeDict)

    parameters = inputs["parameters"].get_dict()
    assert "dis_froz_max" in parameters, parameters
    assert abs(parameters["dis_froz_max"] - dis_froz_max[1]) < 1e-8, parameters


def test_prepare_inputs_auto_energy_windows(
    generate_inputs_wannier90_base,
    generate_workchain_wannier90_base,
    generate_bands_data,
    generate_projection_data,
):
    """Test `Wannier90BaseWorkChain.prepare_inputs` for `auto_energy_windows`."""
    from aiida.orm import Bool, Dict

    inputs = generate_inputs_wannier90_base()
    parameters = inputs["parameters"].get_dict()

    # Test SCDM fitting is working
    parameters["fermi_energy"] = 1.2
    parameters["num_wann"] = 4
    inputs["parameters"] = Dict(parameters)

    inputs = {"wannier90": inputs}
    inputs["bands"] = generate_bands_data()
    inputs["bands_projections"] = generate_projection_data()
    inputs["auto_energy_windows"] = Bool(True)

    process = generate_workchain_wannier90_base(inputs=inputs)
    inputs = process.prepare_inputs()

    assert isinstance(inputs, AttributeDict)

    parameters = inputs["parameters"].get_dict()
    assert "dis_froz_max" in parameters, parameters
    assert abs(parameters["dis_froz_max"] - 3.98687455) < 1e-8, parameters


@pytest.mark.parametrize(
    "num_procs",
    (
        4,
        2,
    ),
)
def test_handle_output_stdout_incomplete(
    generate_workchain_wannier90_base, generate_inputs_wannier90_base, num_procs
):
    """Test `Wannier90BaseWorkChain.handle_output_stdout_incomplete` for restarting from OOM."""

    inputs = {"wannier90": generate_inputs_wannier90_base()}
    # E.g. when number of MPI procs = 4, the next trial is 2
    inputs["wannier90"]["metadata"]["options"] = {
        "resources": {"num_machines": 1, "num_mpiprocs_per_machine": num_procs},
        "max_wallclock_seconds": 3600,
        "withmpi": True,
        "scheduler_stderr": "_scheduler-stderr.txt",
    }
    process = generate_workchain_wannier90_base(
        exit_code=Wannier90Calculation.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE,
        inputs=inputs,
        test_name="out_of_memory",
    )
    process.setup()

    # Direct call to the handler
    result = process.handle_output_stdout_incomplete(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert result.do_break
    assert result.exit_code.status == 0

    new_num_procs = num_procs // 2
    assert (
        process.ctx.inputs["metadata"]["options"]["resources"][
            "num_mpiprocs_per_machine"
        ]
        == new_num_procs
    )

    # The `inspect_process` will call again the `handle_output_stdout_incomplete` because the
    # `ERROR_OUTPUT_STDOUT_INCOMPLETE` exit code is still there.
    result = process.inspect_process()
    new_num_procs = num_procs // 4
    if new_num_procs == 0:
        assert (
            result == Wannier90BaseWorkChain.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE
        )
        new_num_procs = 1
    else:
        assert result.status == 0
    assert (
        process.ctx.inputs["metadata"]["options"]["resources"][
            "num_mpiprocs_per_machine"
        ]
        == new_num_procs
    )


def test_handle_bvectors(generate_workchain_wannier90_base):
    """Test `Wannier90BaseWorkChain.handle_bvectors`."""

    process = generate_workchain_wannier90_base(
        exit_code=Wannier90Calculation.exit_codes.ERROR_BVECTORS, test_name="bvectors"
    )
    process.setup()

    assert process.ctx.kmeshtol_new == [1e-6, 1e-8, 1e-4]

    # 1st direct call to the handler
    result = process.handle_bvectors(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert result.do_break
    assert result.exit_code.status == 0

    tol = 1e-12
    new_kmesh_tol = 1e-8
    assert (
        abs(process.ctx.inputs["parameters"].get_dict()["kmesh_tol"] - new_kmesh_tol)
        < tol
    )

    # 2nd direct call to the handler
    result = process.handle_bvectors(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert result.do_break
    assert result.exit_code.status == 0

    new_kmesh_tol = 1e-4
    assert (
        abs(process.ctx.inputs["parameters"].get_dict()["kmesh_tol"] - new_kmesh_tol)
        < tol
    )

    # The `inspect_process` will call again the `handle_bvectors` because the
    # `ERROR_BVECTORS` exit code is still there.
    result = process.inspect_process()
    assert result == Wannier90BaseWorkChain.exit_codes.ERROR_BVECTORS


def test_handle_disentanglement_not_enough_states(
    generate_workchain_wannier90_base, generate_inputs_wannier90_base
):
    """Test `Wannier90BaseWorkChain.handle_disentanglement_not_enough_states`."""
    from aiida import orm
    from aiida.cmdline.utils.common import get_workchain_report

    inputs = {"wannier90": generate_inputs_wannier90_base()}
    dis_proj_min = 0.1
    inputs["wannier90"]["parameters"] = orm.Dict(
        dict={
            "dis_proj_min": dis_proj_min,
        }
    )
    process = generate_workchain_wannier90_base(
        exit_code=Wannier90Calculation.exit_codes.ERROR_DISENTANGLEMENT_NOT_ENOUGH_STATES,
        test_name="not_enough_states",
        inputs=inputs,
    )
    process.setup()

    reference_multipliers = [0.5, 0.25, 0.125, 0]
    assert process.ctx.disprojmin_multipliers == reference_multipliers

    # Tolerance for comparing floats
    tol = 1e-12

    # Direct calls to the handler
    new_disprojmin = dis_proj_min
    for ref_multiplier in reference_multipliers:
        result = process.handle_disentanglement_not_enough_states(
            process.ctx.children[-1]
        )
        assert isinstance(result, ProcessHandlerReport)
        assert result.do_break
        assert result.exit_code.status == 0, (
            ref_multiplier,
            get_workchain_report(process, levelname="REPORT"),
        )

        new_disprojmin = new_disprojmin * ref_multiplier
        assert (
            abs(
                process.ctx.inputs["parameters"].get_dict()["dis_proj_min"]
                - new_disprojmin
            )
            < tol
        ), ref_multiplier

    # The `inspect_process` will call again the `handle_disentanglement_not_enough_states` because the
    # `ERROR_DISENTANGLEMENT_NOT_ENOUGH_STATES` exit code is still there.
    result = process.inspect_process()
    assert (
        result
        == Wannier90BaseWorkChain.exit_codes.ERROR_DISENTANGLEMENT_NOT_ENOUGH_STATES
    )


def test_handle_plot_wf_cube(
    generate_workchain_wannier90_base, generate_inputs_wannier90_base
):
    """Test `Wannier90BaseWorkChain.handle_plot_wf_cube`."""
    from aiida import orm
    from aiida.cmdline.utils.common import get_workchain_report

    inputs = {"wannier90": generate_inputs_wannier90_base()}
    inputs["wannier90"]["parameters"] = orm.Dict(
        dict={
            "wannier_plot_supercell": 2,
        }
    )
    process = generate_workchain_wannier90_base(
        exit_code=Wannier90Calculation.exit_codes.ERROR_PLOT_WF_CUBE,
        test_name="plot_wf_cube",
        inputs=inputs,
    )
    process.setup()

    reference_wannier_plot_supercell = [4, 6, 8, 10]
    assert process.ctx.wannier_plot_supercell_new == reference_wannier_plot_supercell

    # Direct calls to the handler
    for new_supercell in reference_wannier_plot_supercell:
        result = process.handle_plot_wf_cube(process.ctx.children[-1])
        assert isinstance(result, ProcessHandlerReport)
        assert result.do_break
        assert result.exit_code.status == 0, (
            new_supercell,
            get_workchain_report(process, levelname="REPORT"),
        )

        assert (
            process.ctx.inputs["parameters"].get_dict()["wannier_plot_supercell"]
            == new_supercell
        )

    # The `inspect_process` will call again the `handle_plot_wf_cube` because the
    # `ERROR_PLOT_WF_CUBE` exit code is still there.
    result = process.inspect_process()
    assert result == Wannier90BaseWorkChain.exit_codes.ERROR_PLOT_WF_CUBE
