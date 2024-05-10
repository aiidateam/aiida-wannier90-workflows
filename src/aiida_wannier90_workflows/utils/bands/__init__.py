"""Utility functions for band structure."""

import typing as ty

import numpy as np


def get_homo_lumo(bands: np.array, fermi_energy: float) -> ty.Tuple[float, float]:
    """Find highest occupied bands and lowest unoccupied bands around Fermi energy.

    :param bands: num_kpoints * num_bands
    :type bands: np.array
    :param fermi_energy: [description]
    :type fermi_energy: float
    :return: [description]
    :rtype: ty.Tuple[float, float]
    """
    occupied = bands <= fermi_energy
    unoccupied = bands > fermi_energy

    bands_occ = bands[occupied]
    if len(bands_occ) == 0:
        raise ValueError("No HOMO found, all bands are unoccupied?")
    homo = np.max(bands_occ)

    bands_unocc = bands[unoccupied]
    if len(bands_unocc) == 0:
        raise ValueError("No LUMO found, all bands are occupied?")
    lumo = np.min(bands_unocc)

    return homo, lumo


def remove_exclude_bands(bands: np.array, exclude_bands: list) -> np.array:
    """Remove bands according the index specified by `exclude_bands`.

    :param bands: num_kpoints x num_bands
    :type bands: np.array
    :param exclude_bands: the index of the to-be-excluded bands, 0-based indexing
    :type exclude_bands: list
    :return: the bands with exclude_bands removed
    :rtype: np.array
    """
    num_kpoints, num_bands = bands.shape  # pylint: disable=unused-variable

    if not set(exclude_bands).issubset(set(range(num_bands))):
        raise ValueError(
            f"exclude_bands {exclude_bands} not in the range of available bands {range(num_bands)}"
        )

    # Remove repetition and sort
    exclude_bands = sorted(set(exclude_bands))

    sub_bands = np.delete(bands, exclude_bands, axis=1)

    return sub_bands
