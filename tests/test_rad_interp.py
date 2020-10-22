# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import assert_allclose

from bounded_rand_walkers.rad_interp import exact_radii_interp, inv_exact_radii_interp
from bounded_rand_walkers.utils import get_centres


def test_params():
    with pytest.raises(ValueError):
        exact_radii_interp(None, None, None, normalisation="undefined")
    with pytest.raises(ValueError):
        exact_radii_interp(None, None, None, bin_samples=10, bin_width=0.1)
    with pytest.raises(ValueError):
        inv_exact_radii_interp([0, 0], None, None, None)
    with pytest.raises(ValueError):
        inv_exact_radii_interp([0, -1], None, None, None)
    with pytest.raises(ValueError):
        inv_exact_radii_interp([0, 1], None, None, None, normalisation="undefined")


@pytest.mark.parametrize("normalisations", [("divide", "multiply"), ("none",) * 2])
@pytest.mark.parametrize(
    "bin_samples,bin_width", [(20, None), (0.01, None), (None, 0.01)]
)
def test_radii_interp(normalisations, bin_samples, bin_width):
    """Test the conversion between 1D radial and 2D gridded data."""
    # Generate the grid centre coordinates.
    bins = 100
    x_centres = y_centres = get_centres(np.linspace(0, 1, bins + 1))
    # Generate the resulting unique radii.
    coords = np.array(np.meshgrid(x_centres, y_centres, indexing="ij"))
    grid_radii = np.unique(np.linalg.norm(coords, axis=0))
    # Create the original data.
    rad_data = np.sin(grid_radii)

    # Transform the 1D radial data to the 2D grid.
    gridded = inv_exact_radii_interp(
        grid_radii, rad_data, x_centres, y_centres, normalisation=normalisations[0]
    )
    # Sample the 2D grid to retrieve the 1D radial data.
    sampled_radii, sampled = exact_radii_interp(
        gridded,
        x_centres,
        y_centres,
        normalisation=normalisations[1],
        bin_samples=bin_samples,
        bin_width=bin_width,
    )

    # Ensure the data was preserved.
    if bin_samples is None and bin_width is None:
        assert_allclose(sampled, rad_data)
    else:
        assert_allclose(sampled, np.sin(sampled_radii), atol=1e-3)
