# -*- coding: utf-8 -*-
import numpy as np
import pytest

from bounded_rand_walkers.cpp import bound_map
from bounded_rand_walkers.functions import Freehand, Gaussian
from bounded_rand_walkers.position_density import g1D, g2D
from bounded_rand_walkers.rad_interp import exact_radii_interp
from bounded_rand_walkers.utils import get_centres


@pytest.mark.parametrize("pdf", [Freehand(width=2.0), Gaussian(width=0.1)])
def test_1d_density(pdf):
    """Check that the extrema have a lower probability than the centre."""
    pos_prob = g1D(pdf, np.linspace(0, 1, 100))
    assert np.all(pos_prob[[0, -1]] < pos_prob[pos_prob.shape[0] // 2])


@pytest.mark.parametrize(
    "pdf",
    [
        Freehand(centre=np.array([0.0, 0.0]), width=2.0),
        Gaussian(centre=np.array([0.0, 0.0]), width=0.1),
    ],
)
def test_2d_density(pdf):
    """Check that the extrema have a lower probability than the centre."""
    x = y = np.linspace(-2, 2, 30)
    # Calculate the 2D position probably, centre at 0.
    pos_prob = g2D(pdf, x, y, bound_map["circle"]())

    # Extract the radial profile for further testing.
    _, rad_prob = exact_radii_interp(
        pos_prob, get_centres(x), get_centres(y), normalisation="none", bin_width=0.05
    )
    assert rad_prob[-1] < rad_prob[0]
