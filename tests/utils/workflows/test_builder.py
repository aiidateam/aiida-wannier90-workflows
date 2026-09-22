"""Unit tests for the :py:mod:`~aiida_quantumespresso.utils.workflows.builder` module."""

import numpy as np
import pytest

from aiida import orm


def test_recursive_merge_container():
    """Test the function `recursive_merge_container`."""
    from aiida_wannier90_workflows.utils.workflows.builder.submit import (
        recursive_merge_container,
    )

    left, right = 1, 2
    assert recursive_merge_container(left, right) == 2

    left, right = [1], [2]
    assert recursive_merge_container(left, right) == [1, 2]

    left, right = {"a": 1}, {"b": 2}
    assert recursive_merge_container(left, right) == {"a": 1, "b": 2}

    left, right = {"a": 1}, {"a": 2}
    assert recursive_merge_container(left, right) == {"a": 2}

    left, right = {"a": [1]}, {"a": [2]}
    assert recursive_merge_container(left, right) == {"a": [1, 2]}

    left, right = {"a": {"b": 1}}, {"a": {"b": 2}}
    assert recursive_merge_container(left, right) == {"a": {"b": 2}}

    left, right = orm.List(list=[1]), orm.List(list=[2])
    merged = recursive_merge_container(left, right)
    assert isinstance(merged, orm.List)
    assert merged.get_list() == [1, 2]

    left, right = orm.Dict({"a": 1}), orm.Dict({"a": 2})
    merged = recursive_merge_container(left, right)
    assert isinstance(merged, orm.Dict)
    assert merged.get_dict() == {"a": 2}

    left, right = orm.Dict({"a": [1]}), orm.Dict({"a": [2]})
    merged = recursive_merge_container(left, right)
    assert isinstance(merged, orm.Dict)
    assert merged.get_dict() == {"a": [1, 2]}

    left = orm.Dict({"a": orm.List(list=[1])})
    right = orm.Dict({"a": orm.List(list=[2])})
    merged = recursive_merge_container(left, right)
    assert isinstance(merged, orm.Dict)
    assert isinstance(merged["a"], orm.List)
    assert merged.get_dict()["a"].get_list() == [1, 2]


@pytest.mark.parametrize(
    "parameters",
    (
        {"SYSTEM": {"nbnd": 20}},
        {"ELECTRONS": {"fake_tag": [8, 9]}},
    ),
)
def test_recursive_merge_builder(
    generate_inputs_pw, data_regression, serialize_builder, parameters
):
    """Test the function `recursive_merge_container`."""
    from aiida_quantumespresso.calculations.pw import PwCalculation

    from aiida_wannier90_workflows.utils.workflows.builder.submit import (
        recursive_merge_builder,
    )

    inputs = generate_inputs_pw()

    builder = PwCalculation.get_builder()
    for key, val in inputs.items():
        builder[key] = val
    # I add one fake input parameter to test merge of list
    parameters_dict = builder["parameters"].get_dict()
    parameters_dict["ELECTRONS"]["fake_tag"] = [1, 2]
    builder["parameters"] = orm.Dict(parameters_dict)

    right = {"parameters": orm.Dict(parameters)}

    builder = recursive_merge_builder(builder, right)

    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize(
    "inout",
    (
        ([0, 1, 2, 3], [0, 1, 2, 3]),
        (np.arange(4), [0, 1, 2, 3]),
        (list(np.arange(4)), [0, 1, 2, 3]),
        ({"a": np.arange(4)}, {"a": [0, 1, 2, 3]}),
    ),
)
def test_serializer(inout):
    """Test the function ``serializer``."""
    from aiida_wannier90_workflows.utils.workflows.builder.serializer import serialize

    assert serialize(inout[0]) == inout[1], inout


def test_relax_builder_namespaces(fixture_code, generate_structure):
    """Test relax builder generation and parallelization for AQE 4 and 5."""
    from aiida_quantumespresso.common.types import SpinType
    from aiida_quantumespresso.workflows.pw.relax import PwRelaxWorkChain

    from aiida_wannier90_workflows.utils.workflows.builder.generator import (
        get_relax_builder,
    )
    from aiida_wannier90_workflows.utils.workflows.builder.setter import (
        set_parallelization,
    )

    if "base_relax" in PwRelaxWorkChain.spec().inputs:
        relax_namespaces = ("base_init_relax", "base_relax")
    else:
        relax_namespaces = ("base",)

    builder = get_relax_builder(
        code=fixture_code("quantumespresso.pw"),
        structure=generate_structure(),
        kpoints_distance=0.2,
        pseudo_family="PseudoDojo/0.4/PBE/FR/standard/upf",
        spin_type=SpinType.SPIN_ORBIT,
    )

    for namespace in relax_namespaces:
        assert builder[namespace].kpoints_distance.value == 0.2
        parameters = builder[namespace].pw.parameters.get_dict()
        assert parameters["SYSTEM"]["noncolin"] is True
        assert parameters["SYSTEM"]["lspinorb"] is True

    set_parallelization(
        builder,
        parallelization={"npool": 2},
        process_class=PwRelaxWorkChain,
    )

    pruned_builder = builder._inputs(prune=True)
    parallel_namespaces = {
        "base",
        "base_final_scf",
        "base_init_relax",
        "base_relax",
    }.intersection(pruned_builder)

    assert parallel_namespaces
    for namespace in parallel_namespaces:
        assert builder[namespace].pw.parallelization["npool"] == 2
