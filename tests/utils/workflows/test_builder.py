# -*- coding: utf-8 -*-
"""Unit tests for the :py:mod:`~aiida_quantumespresso.utils.workflows.builder` module."""
import pytest
import numpy as np

from aiida import orm


def test_recursive_merge_container():
    """Test the function `recursive_merge_container`."""
    from aiida_wannier90_workflows.utils.workflows.builder import recursive_merge_container

    left, right = 1, 2
    assert recursive_merge_container(left, right) == 2

    left, right = [1], [2]
    assert recursive_merge_container(left, right) == [1, 2]

    left, right = {'a': 1}, {'b': 2}
    assert recursive_merge_container(left, right) == {'a': 1, 'b': 2}

    left, right = {'a': 1}, {'a': 2}
    assert recursive_merge_container(left, right) == {'a': 2}

    left, right = {'a': [1]}, {'a': [2]}
    assert recursive_merge_container(left, right) == {'a': [1, 2]}

    left, right = {'a': {'b': 1}}, {'a': {'b': 2}}
    assert recursive_merge_container(left, right) == {'a': {'b': 2}}

    left, right = orm.List(list=[1]), orm.List(list=[2])
    merged = recursive_merge_container(left, right)
    assert isinstance(merged, orm.List)
    assert merged.get_list() == [1, 2]

    left, right = orm.Dict(dict={'a': 1}), orm.Dict(dict={'a': 2})
    merged = recursive_merge_container(left, right)
    assert isinstance(merged, orm.Dict)
    assert merged.get_dict() == {'a': 2}

    left, right = orm.Dict(dict={'a': [1]}), orm.Dict(dict={'a': [2]})
    merged = recursive_merge_container(left, right)
    assert isinstance(merged, orm.Dict)
    assert merged.get_dict() == {'a': [1, 2]}

    left = orm.Dict(dict={'a': orm.List(list=[1])})
    right = orm.Dict(dict={'a': orm.List(list=[2])})
    merged = recursive_merge_container(left, right)
    assert isinstance(merged, orm.Dict)
    assert isinstance(merged['a'], orm.List)
    assert merged.get_dict()['a'].get_list() == [1, 2]


@pytest.mark.parametrize('parameters', (
    {
        'SYSTEM': {
            'nbnd': 20
        }
    },
    {
        'ELECTRONS': {
            'fake_tag': [8, 9]
        }
    },
))
def test_recursive_merge_builder(generate_inputs_pw, data_regression, serialize_builder, parameters):
    """Test the function `recursive_merge_container`."""
    from aiida_quantumespresso.calculations.pw import PwCalculation
    from aiida_wannier90_workflows.utils.workflows.builder import recursive_merge_builder

    inputs = generate_inputs_pw()

    builder = PwCalculation.get_builder()
    for key, val in inputs.items():
        builder[key] = val
    # I add one fake input parameter to test merge of list
    parameters_dict = builder['parameters'].get_dict()
    parameters_dict['ELECTRONS']['fake_tag'] = [1, 2]
    builder['parameters'] = orm.Dict(dict=parameters_dict)

    right = {'parameters': orm.Dict(dict=parameters)}

    builder = recursive_merge_builder(builder, right)

    data_regression.check(serialize_builder(builder))


@pytest.mark.parametrize(
    'inout', (
        ([0, 1, 2, 3], [0, 1, 2, 3]),
        (np.arange(4), [0, 1, 2, 3]),
        (list(np.arange(4)), [0, 1, 2, 3]),
        ({
            'a': np.arange(4)
        }, {
            'a': [0, 1, 2, 3]
        }),
    )
)
def test_serializer(inout):
    """Test the function ``serializer``."""
    from aiida_wannier90_workflows.utils.workflows.builder import serializer

    assert serializer(inout[0]) == inout[1], inout
