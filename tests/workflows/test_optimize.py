"""Tests for the `Wannier90OptimizeWorkChain` class."""

from aiida import orm

from .test_wannier90 import _run_through_wannier90_pp


def test_metadata_survives_wannier90_override(
    generate_workchain,
    generate_inputs_wannier90,
    fixture_localhost,
    generate_remote_data,
    generate_calc_job_node,
):
    """A caller-set ``wannier90.metadata`` label reaches the minimization node.

    ``Wannier90OptimizeWorkChain`` overrides ``run_wannier90``/``_up``/
    ``_down`` and used to reintroduce the same whole-``metadata``-dict
    replacement bug fixed in the base ``Wannier90WorkChain`` -- this pins
    the override too, reusing the base class's own mock chain since
    ``prepare_wannier90_pp_inputs`` and ``run_wannier90_pp`` are inherited
    unchanged.
    """
    inputs = generate_inputs_wannier90()
    # Sidesteps `optimize.validate_inputs`'s `dis_proj_min`/`dis_proj_max`/
    # `optimize_reference_bands` requirements, which are orthogonal to the
    # metadata question this test pins.
    inputs["optimize_disproj"] = orm.Bool(False)
    inputs["wannier90"]["metadata"] = {
        "label": "Minimization",
        "description": "final MLWF run",
    }

    workchain = generate_workchain("wannier90_workflows.optimize", inputs)
    _, captured = _run_through_wannier90_pp(
        workchain, generate_remote_data, fixture_localhost, generate_calc_job_node
    )

    w90_workchain = workchain.run_wannier90()["workchain_wannier90"]

    assert w90_workchain.label == "Minimization"
    assert w90_workchain.description == "final MLWF run"
    assert captured["wannier90"]["call_link_label"] == "wannier90"
