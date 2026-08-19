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


def test_bands_distance_fermi_dirac(load_bands):
    """Test the function for ``bands_distance_fermi_dirac``."""
    from aiida_wannier90_workflows.utils.bands.distance import (
        bands_distance_fermi_dirac,
    )

    pw_bands = load_bands("W", "pw.json")
    wan_bands = load_bands("W", "w90.json")

    fermi_energy = 22.753
    exclude_list_dft = [1, 2, 3, 4]

    dist = bands_distance_fermi_dirac(
        bands_dft=pw_bands,
        bands_wannier=wan_bands,
        mu=fermi_energy,
        sigma=0.1,
        exclude_list_dft=exclude_list_dft,
        lower_cutoff=-30,
    )

    # The same weighting as the first row of ``bands_distance``
    assert abs(dist - 1.036669313927612510e-02) < 1e-8


def test_bands_distance_unweighted(load_bands):
    """Test the function for ``bands_distance_unweighted``."""
    import numpy as np

    from aiida_wannier90_workflows.utils.bands.distance import (
        bands_distance_fermi_dirac,
        bands_distance_unweighted,
    )

    pw_bands = load_bands("W", "pw.json")
    wan_bands = load_bands("W", "w90.json")

    exclude_list_dft = [1, 2, 3, 4]

    dist = bands_distance_unweighted(
        bands_dft=pw_bands,
        bands_wannier=wan_bands,
        exclude_list_dft=exclude_list_dft,
    )

    # Every band weighted equally is a plain root-mean-square difference
    num_bands = wan_bands.get_bands().shape[1]
    difference = (
        pw_bands.get_bands()[:, len(exclude_list_dft) :][:, :num_bands]
        - wan_bands.get_bands()
    )
    assert abs(dist - np.sqrt(np.mean(difference**2))) < 1e-8

    # ... and the limit of the Fermi-Dirac weighting for a broadening that
    # reaches every band
    dist_broad = bands_distance_fermi_dirac(
        bands_dft=pw_bands,
        bands_wannier=wan_bands,
        mu=22.753,
        sigma=1e8,
        exclude_list_dft=exclude_list_dft,
    )
    assert abs(dist_broad - dist) < 1e-6


def test_bands_distance_isolated_spin_collinear():
    """Test ``bands_distance_isolated`` on spin-collinear bands.

    Spin-collinear bands arrive as (num_spins, num_kpts, num_bands) and have to
    be moved to (num_kpts, num_bands, num_spins) before the Wannier bands are
    truncated to the DFT ones, otherwise the truncation slices the k-points.
    """
    import numpy as np

    from aiida_wannier90_workflows.utils.bands.distance import bands_distance_isolated

    num_spins, num_kpts, num_dft, num_wann = 2, 3, 5, 4

    dft_bands = np.arange(num_spins * num_kpts * num_dft, dtype=float).reshape(
        num_spins, num_kpts, num_dft
    )
    difference = 0.1 * np.arange(num_spins * num_kpts * num_wann, dtype=float).reshape(
        num_spins, num_kpts, num_wann
    )
    wannier_bands = dft_bands[:, :, :num_wann] - difference

    dist, _, max_dist_2, _, _ = bands_distance_isolated(dft_bands, wannier_bands)

    assert abs(dist - np.sqrt(np.mean(difference**2))) < 1e-8
    assert abs(max_dist_2 - difference.max()) < 1e-8
