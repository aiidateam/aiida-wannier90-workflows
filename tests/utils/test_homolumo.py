# -*- coding: utf-8 -*-
"""Unit tests for the :py:mod:`~aiida_quantumespresso.utils.bands` module."""
import numpy as np

from aiida_wannier90_workflows.workflows.wannier90 import get_homo_lumo


def test_get_homo_lumo():
    """Test the function for aiida_wannier90_workflows.workflows.wannier.get_homo_lumo."""

    bands = np.array([
        [1.0, 2.0, 3.0],
        [1.1, 2.1, 2.1],
    ])

    fermi_energy = 1.5

    homo, lumo = get_homo_lumo(bands, fermi_energy)

    tol = 1e-8
    assert abs(homo - 1.1) < tol
    assert abs(lumo - 2.0) < tol
