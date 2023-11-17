"""General tests for the protocol methods."""
import pytest

from aiida_wannier90_workflows.workflows.bands import Wannier90BandsWorkChain
from aiida_wannier90_workflows.workflows.open_grid import Wannier90OpenGridWorkChain
from aiida_wannier90_workflows.workflows.optimize import Wannier90OptimizeWorkChain
from aiida_wannier90_workflows.workflows.wannier90 import Wannier90WorkChain


@pytest.mark.parametrize(
    "workchain",
    (
        Wannier90WorkChain,
        Wannier90OpenGridWorkChain,
        Wannier90BandsWorkChain,
        Wannier90OptimizeWorkChain,
    ),
)
@pytest.mark.parametrize(
    "overrides",
    (
        {"scf": {"pw": {"parameters": {"ELECTRONS": {"diagonalization": "paro"}}}}},
        {"nscf": {"pw": {"parallelization": {"npool": 8}}}},
        {"projwfc": {"projwfc": {"metadata": {"options": {"account": "infinite"}}}}},
        {
            "pw2wannier90": {
                "pw2wannier90": {"parameters": {"inputpp": {"scdm_proj": False}}}
            }
        },
        {"wannier90": {"auto_energy_windows_threshold": 0.01}},
    ),
)
def test_overrides(
    generate_builder_inputs, data_regression, serialize_builder, overrides, workchain
):
    """Test specifying parameter ``overrides`` for the ``get_builder_from_protocol()`` method."""
    inputs = generate_builder_inputs("Si")

    overrides_key = next(iter(overrides))

    if overrides_key == "nscf" and workchain == Wannier90OpenGridWorkChain:
        inputs["open_grid_only_scf"] = False

    if overrides_key == "projwfc" and workchain == Wannier90OptimizeWorkChain:
        return

    builder = workchain.get_builder_from_protocol(**inputs, overrides=overrides)
    data_regression.check(serialize_builder(builder).pop(overrides_key))
