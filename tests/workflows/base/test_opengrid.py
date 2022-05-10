"""Tests for the `OpengridBaseWorkChain` class."""
import pytest

from aiida.common import AttributeDict
from aiida.engine import ProcessHandlerReport

from aiida_quantumespresso.calculations.opengrid import OpengridCalculation

from aiida_wannier90_workflows.workflows.base.opengrid import OpengridBaseWorkChain

# pylint: disable=no-member,redefined-outer-name


def test_setup(generate_workchain_opengrid_base):
    """Test `OpengridBaseWorkChain.setup`."""
    process = generate_workchain_opengrid_base()
    process.setup()

    assert isinstance(process.ctx.inputs, AttributeDict)


@pytest.mark.parametrize(
    "npool_value",
    (
        4,
        2,
    ),
)
@pytest.mark.parametrize(
    "npool_key",
    (
        "-nk",
        "-npools",
    ),
)
def test_handle_output_stdout_incomplete(
    generate_workchain_opengrid_base,
    generate_inputs_opengrid_base,
    npool_key,
    npool_value,
):
    """Test `OpengridBaseWorkChain.handle_output_stdout_incomplete` for restarting from OOM."""
    from aiida import orm

    inputs = {"opengrid": generate_inputs_opengrid_base()}
    # E.g. when number of MPI procs = 4, the next trial is 2
    inputs["opengrid"]["metadata"]["options"] = {
        "resources": {"num_machines": 1, "num_mpiprocs_per_machine": npool_value},
        "max_wallclock_seconds": 3600,
        "withmpi": True,
        "scheduler_stderr": "_scheduler-stderr.txt",
    }
    inputs["opengrid"]["settings"] = orm.Dict(
        dict={"cmdline": [npool_key, f"{npool_value}"]}
    )
    process = generate_workchain_opengrid_base(
        exit_code=OpengridCalculation.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE,
        inputs=inputs,
        test_name="out_of_memory",
    )
    process.setup()

    # Direct call to the handler
    result = process.handle_output_stdout_incomplete(process.ctx.children[-1])
    assert isinstance(result, ProcessHandlerReport)
    assert result.do_break
    assert result.exit_code.status == 0

    new_npool_value = npool_value // 2
    assert (
        process.ctx.inputs["metadata"]["options"]["resources"][
            "num_mpiprocs_per_machine"
        ]
        == new_npool_value
    )
    assert process.ctx.inputs["settings"]["cmdline"] == [
        npool_key,
        f"{new_npool_value}",
    ]

    # The `inspect_process` will call again the `handle_output_stdout_incomplete` because the
    # `ERROR_OUTPUT_STDOUT_INCOMPLETE` exit code is still there.
    result = process.inspect_process()
    new_npool_value = npool_value // 4
    if new_npool_value == 0:
        assert result == OpengridBaseWorkChain.exit_codes.ERROR_OUTPUT_STDOUT_INCOMPLETE
        new_npool_value = 1
    else:
        assert result.status == 0
    assert (
        process.ctx.inputs["metadata"]["options"]["resources"][
            "num_mpiprocs_per_machine"
        ]
        == new_npool_value
    )
    assert process.ctx.inputs["settings"]["cmdline"] == [
        npool_key,
        f"{new_npool_value}",
    ]
