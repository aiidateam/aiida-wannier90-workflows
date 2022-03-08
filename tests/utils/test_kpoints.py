# -*- coding: utf-8 -*-
"""Unit tests for the :py:mod:`~aiida_quantumespresso.utils.kpoints` module."""
import numpy as np

from aiida import orm


def test_create_kpoints_from_distance(generate_structure):
    """Test the function ``aiida_wannier90_workflows.utils.kpoints.create_kpoints_from_distance``."""
    from aiida_wannier90_workflows.utils.kpoints import create_kpoints_from_distance

    structure = generate_structure('Si')
    kpoints = create_kpoints_from_distance(structure, 0.2)
    mesh, offset = kpoints.get_kpoints_mesh()

    assert np.allclose(mesh, [11, 11, 11])
    assert np.allclose(offset, [0.0, 0.0, 0.0])


def test_get_mesh_from_kpoints():
    """Test the function ``aiida_wannier90_workflows.utils.kpoints.get_mesh_from_kpoints``."""
    from aiida_wannier90_workflows.utils.kpoints import cartesian_product, get_mesh_from_kpoints

    x = np.linspace(0, 1, 3)
    y = np.linspace(0, 1, 4)
    z = np.linspace(0, 1, 5)
    klist = cartesian_product(x, y, z)
    kpoints = orm.KpointsData()
    kpoints.set_kpoints(klist)

    mesh = get_mesh_from_kpoints(kpoints)

    assert np.allclose(mesh, [3, 4, 5])
