"""Unit tests for the :py:mod:`~aiida_quantumespresso.utils.bands` module."""

import numpy as np


def test_get_homo_lumo():
    """Test the function for aiida_wannier90_workflows.workflows.wannier.get_homo_lumo."""
    from aiida_wannier90_workflows.utils.bands import get_homo_lumo

    bands = np.array(
        [
            [1.0, 2.0, 3.0],
            [1.1, 2.1, 2.1],
        ]
    )

    fermi_energy = 1.5

    homo, lumo = get_homo_lumo(bands, fermi_energy)

    tol = 1e-8
    assert abs(homo - 1.1) < tol
    assert abs(lumo - 2.0) < tol


def test_bands_distance(load_bands):
    """Test the function for ``bands_distance``."""
    from aiida_wannier90_workflows.utils.bands.distance import bands_distance

    pw_bands = load_bands("W", "pw.json")
    wan_bands = load_bands("W", "w90.json")

    fermi_energy = 22.753
    exclude_list_dft = [1, 2, 3, 4]

    dist = bands_distance(
        bands_dft=pw_bands,
        bands_wannier=wan_bands,
        fermi_energy=fermi_energy,
        exclude_list_dft=exclude_list_dft,
    )

    ref_dist = np.array(
        [
            [
                2.275300000000000011e01,
                1.036669313927612510e-02,
                5.322753656962231350e-02,
                5.322753656931334537e-02,
            ],
            [
                2.375300000000000011e01,
                1.008729713743582342e-02,
                5.322753656993127469e-02,
                5.322753656993126081e-02,
            ],
            [
                2.475300000000000011e01,
                9.774211496004188773e-03,
                5.322753656993128857e-02,
                5.322753656993128857e-02,
            ],
            [
                2.575300000000000011e01,
                9.008465047456700250e-03,
                5.322753656993128857e-02,
                5.322753656993128857e-02,
            ],
            [
                2.675300000000000011e01,
                8.588777504871471583e-03,
                5.322753656993128857e-02,
                5.322753656993128857e-02,
            ],
            [
                2.775300000000000011e01,
                8.545545737397273328e-03,
                5.322753656993128857e-02,
                5.322753656993128857e-02,
            ],
        ]
    )

    atol = 1e-8
    assert np.allclose(dist, ref_dist, atol=atol)
