"""Fixtures for testing bands."""
import pytest

# pylint: disable=redefined-outer-name,too-many-statements


@pytest.fixture
def load_bands(filepath_fixtures):
    """Generate a `BandsData` node."""

    def _load_bands(structure: str, file_name: str):
        """Generate a `BandsData` node."""
        import json

        import numpy as np

        from aiida.plugins import DataFactory

        BandsData = DataFactory("core.array.bands")  # pylint: disable=invalid-name
        bands_data = BandsData()

        fname = str(filepath_fixtures / "utils" / "bands" / structure / file_name)
        # aiida-core does not support importing BandsData
        # bands_data.importfile(fname)

        with open(fname) as handle:
            data = json.load(handle)

        data = np.hstack([_["values"] for _ in data["paths"]]).T

        # Use fake kpoints
        bands_data.set_kpoints(np.zeros((data.shape[0], 3)))

        bands_data.set_bands(data)

        return bands_data

    return _load_bands
