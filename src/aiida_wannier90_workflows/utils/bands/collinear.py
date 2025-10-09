"""Functions for spin collinear bands claculations."""

import numpy as np

from aiida import orm
from aiida.engine import calcfunction


@calcfunction
def get_spinor_band_structure(  # pylint: disable=unused-argument,inconsistent-return-statements
    band_structure_up: orm.BandsData,
    band_structure_down: orm.BandsData,
    structure: orm.StructureData,
):
    """Combine up and down interpolated_bands into one BandsData."""

    # Both `up` and `down` are in bands_structure.
    # When parsering interpolated_bands, only bands, kpoints and kpoints were set.
    #   bands.set_kpointsdata(k)
    #   ├---k.set_cell <- structure from self.ctx.current_structure, seekpath may refine the structure
    #   └---k.set_kpoints <- kpts from bands_structure["up"]/["down"].get_kpoints()
    #   bands.set_bands(bands_data, unit="eV")
    #   └---bands <- np.array([bands_structure["up"].get_bands(),
    #                          bands_structure["down"].get_bands()])
    #   bands.labels <- bands_structure["up"]/["down"].labels

    band_structure = orm.BandsData()
    k = orm.KpointsData()

    k.set_cell_from_structure(structure)
    if not (band_structure_up.get_kpoints() == band_structure_down.get_kpoints()).all:
        # The BandsData.get_kpoints() will returen np.array, so we need to use .all to return a single bool.
        raise ValueError(
            f"Bands structure for spin up<{band_structure_up.pk}> and "
            + f"down<{band_structure_down.pk}> have different kpath"
        )
    k.set_kpoints(band_structure_up.get_kpoints(), cartesian=False)
    band_structure.set_kpointsdata(k)
    if not np.shape(band_structure_up.get_bands()) == np.shape(
        band_structure_down.get_bands()
    ):
        raise ValueError(
            f"Bands structure for spin up<{band_structure_up.pk}> and "
            + f"down<{band_structure_down.pk}> have different shape"
        )
    band_structure.set_bands(
        np.array([band_structure_up.get_bands(), band_structure_down.get_bands()]),
        units="eV",
    )
    if not band_structure_up.labels == band_structure_down.labels:
        raise ValueError(
            f"Bands structure for spin up<{band_structure_up.pk}> and "
            + f"down<{band_structure_down.pk}> have different labels"
        )
    band_structure.labels = band_structure_up.labels

    return band_structure
