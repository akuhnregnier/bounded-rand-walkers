# -*- coding: utf-8 -*-
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.interpolate import UnivariateSpline

from bounded_rand_walkers.cpp import bound_map
from bounded_rand_walkers.rad_interp import exact_radii_interp
from bounded_rand_walkers.shaper_generation import (
    adaptive_simpsons,
    gen_rad_shaper_exact,
    gen_shaper2D,
    square_shaper,
)
from bounded_rand_walkers.utils import get_centres, normalise


def test_square():
    xs = np.linspace(-1.2, 1.2, 1000)
    shaper = square_shaper(xs, np.zeros(xs.size), side_length=1)

    assert_allclose(xs[np.argmax(shaper)], 0, atol=1e-2)
    assert_allclose(shaper[[0, -1]], 0, atol=1e-2)


def test_radial_averaging():
    bound_name = "square"

    vertices = bound_map[bound_name]()

    n_bins = 100
    lim = 1.5
    f_t_x_edges = f_t_y_edges = np.linspace(-lim, lim, n_bins + 1)
    f_t_x_centres = f_t_y_centres = get_centres(f_t_x_edges)

    num_2d_shaper = gen_shaper2D(vertices, f_t_x_edges, f_t_y_edges)

    # Extract shaper from 2D shaper values.
    radii, radial_shaper = exact_radii_interp(
        num_2d_shaper,
        f_t_x_centres,
        f_t_y_centres,
        normalisation="multiply",
        bin_width=0.05,
    )

    # Calculate the shaper function explicitly at multiple radii.
    shaper_radii = np.linspace(0, np.max(radii), 100)
    shaper_rad = gen_rad_shaper_exact(
        shaper_radii, vertices=bound_name if bound_name in bound_map else vertices
    )

    # Analytical at `shaper_radii`.
    analytical = normalise(shaper_radii, shaper_rad * shaper_radii)

    # Radially interpolated at `radii`.
    norm_radial_shaper = normalise(radii, radial_shaper)

    spl = UnivariateSpline(radii, norm_radial_shaper, s=0)

    # Spline-shaper at `shaper_radii`.
    spline_shaper = spl(shaper_radii)

    assert_allclose(spline_shaper, analytical, rtol=1e-2, atol=3e-2)


@pytest.mark.parametrize("n_start", [10, 100, 1000])
@pytest.mark.parametrize("vec_func", [False, True])
def test_simpsons(n_start, vec_func):
    assert_allclose(
        adaptive_simpsons(lambda x: x ** 2, 0, 1, n_start=n_start, vec_func=vec_func),
        1 / 3,
    )
