# -*- coding: utf-8 -*-
from tempfile import mkdtemp

import numpy as np
from numpy.testing import assert_allclose

from bounded_rand_walkers.cpp import (
    bound_map,
    generate_data,
    get_binned_data,
    get_cached_filename,
)
from bounded_rand_walkers.data_generation import multi_random_walker
from bounded_rand_walkers.functions import Freehand2
from bounded_rand_walkers.utils import get_centres


def test_data_generation():
    """Test that the Python and C++ implementations agree."""
    N = int(4e4)
    bound_name = "square"
    bounds = np.array(bound_map[bound_name]())
    n_bins = 10  # Nr. of x, y bins.
    g_bounds = (-1, 1)
    f_bounds = (-1, 1)

    ncpus = 2

    # Generate Python data.
    raw_py_steps, raw_py_pos = multi_random_walker(
        n_processes=ncpus,
        f_i=Freehand2(width=2.0),
        bounds=bounds,
        steps=N,
        blocks=2,
        seed=1,
    )
    # Bin Python data.

    # Position binning.
    g_x_edges = g_y_edges = np.linspace(*g_bounds, n_bins + 1)
    g_x_centres = g_y_centres = get_centres(g_x_edges)

    # Step size binning (transformed).
    f_t_x_edges = f_t_y_edges = np.linspace(*f_bounds, n_bins + 1)
    f_t_x_centres = f_t_y_centres = get_centres(f_t_x_edges)

    py_pos = np.histogram2d(*raw_py_pos.T, bins=[g_x_edges, g_y_edges], normed=True)[0]

    py_steps = np.histogram2d(
        *raw_py_steps.T, bins=[f_t_x_edges, f_t_y_edges], normed=True
    )[0]

    # with TemporaryDirectory(prefix="random_walks") as cache_dir:
    cache_dir = mkdtemp(prefix="random_walk")

    # Generate C++ data.
    data_kwargs = dict(
        cache_dir=cache_dir,
        samples=N,
        seed=np.arange(ncpus),
        blocks=2,
        bound_name=bound_name,
        pdf_name="freehand2",
        width=2.0,
    )

    generate_data(squeeze=False, max_workers=ncpus, cache_only=True, **data_kwargs)
    filenames = get_cached_filename(squeeze=False, **data_kwargs)

    # Bin C++ data.
    (cpp_pos, cpp_steps) = get_binned_data(
        filenames=filenames, n_bins=n_bins, g_bounds=g_bounds, f_bounds=f_bounds
    )[-3:-1]

    assert_allclose(py_pos, cpp_pos, atol=0.1)
    assert_allclose(py_steps, cpp_steps, atol=0.1)
