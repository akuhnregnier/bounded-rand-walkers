# -*- coding: utf-8 -*-
from itertools import product

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_equal

from bounded_rand_walkers.cpp import bound_map
from bounded_rand_walkers.utils import (
    DelaunayArray,
    approx_edges,
    cluster_indices,
    get_centres,
    in_bounds,
    label,
    match_ref,
    normalise,
)


def test_get_centres():
    assert_allclose(get_centres([1, 2, 3, 4]), [1.5, 2.5, 3.5])
    assert_allclose(get_centres([-1, 1, 3, 4]), [0, 2, 3.5])


@pytest.mark.parametrize(
    "x,exp_indices,exp_labelled",
    [
        (
            np.array([1, 1, 0, 0, 1, 0]),
            np.array([[0, 2], [4, 5]]),
            np.array([1, 1, 0, 0, 2, 0]),
        ),
        (
            np.array([0, 1, 0, 0, 1, 1]),
            np.array([[1, 2], [4, 6]]),
            np.array([0, 1, 0, 0, 2, 2]),
        ),
        (
            np.array([0, 1, 0, 0, 1, 0]),
            np.array([[1, 2], [4, 5]]),
            np.array([0, 1, 0, 0, 2, 0]),
        ),
        (
            np.array([0, 1, 0, 0, 0, 0]),
            np.array([[1, 2]]),
            np.array([0, 1, 0, 0, 0, 0]),
        ),
    ],
)
def test_labelling(x, exp_indices, exp_labelled):
    assert_equal(cluster_indices(x), exp_indices)
    labelled, n_cluster = label(x)
    assert_equal(labelled, exp_labelled)
    assert n_cluster == np.max(exp_labelled)


def test_approx_edges():
    assert_allclose(approx_edges([1, 2, 3]), [0.5, 1.5, 2.5, 3.5])
    assert_allclose(approx_edges([1, 2, 4]), [0.5, 1.5, 3, 5])
    assert_allclose(approx_edges([-1, 1, 2, 3]), [-2, 0, 1.5, 2.5, 3.5])


def test_normalise():
    centres = [0.5, 1.5]
    y = [1, 1]
    assert_allclose(normalise(centres, y, return_factor=True), 0.5)
    assert_allclose(normalise(centres, y), [0.5, 0.5])

    edges = [0, 1, 2]
    y = [1, 1]
    assert_allclose(normalise(edges, y, return_factor=True), 0.5)
    assert_allclose(normalise(edges, y), [0.5, 0.5])

    edges = [0, 1 / 3, 1]
    y = [1, 1]
    assert_allclose(normalise(edges, y, return_factor=True), 1)
    assert_allclose(normalise(edges, y), y)

    edges = [0, 1, 5]
    y = [1, 1 / 2]
    assert_allclose(normalise(edges, y, return_factor=True), 1 / 3)
    assert_allclose(normalise(edges, y), [1 / 3, 1 / 6])


def test_match_ref():
    x = np.linspace(0, 1, 10)
    y = np.random.RandomState(0).random(10)

    ref_x = x[:8]
    # Scale the original data and add random noise.
    ref_y = 2 * y[:8] + np.random.RandomState(1).random(8) * 0.1

    diff_before = np.mean(np.abs(y[:8] - ref_y))

    scaled = match_ref(x, y, ref_x, ref_y, interpolate=False)
    assert scaled.shape == y.shape

    diff_after_match = np.mean(np.abs(scaled[:8] - ref_y))

    assert diff_after_match / diff_before < 0.05

    scaled_interp = match_ref(x + 0.01, y, ref_x, ref_y, interpolate=True)
    assert scaled_interp.shape == y.shape

    diff_after_interp_match = np.mean(np.abs(scaled_interp[:8] - ref_y))

    assert diff_after_interp_match / diff_before < 0.05

    scaled_factor = match_ref(x, y, ref_x, ref_y, interpolate=False, return_factor=True)
    scaled_interp_factor = match_ref(
        x + 0.01, y, ref_x, ref_y, interpolate=True, return_factor=True
    )

    for factor in (scaled_factor, scaled_interp_factor):
        assert_allclose(factor, 2, rtol=0, atol=0.05)


def test_delaunay_bounds():
    for bound_name, bound_func in bound_map.items():
        bounds = DelaunayArray(bound_func())
        if bound_name == "irregular":
            pos = np.array([0.5, 0.5])
        else:
            pos = np.array([0, 0])
        assert in_bounds(pos, bounds)

        for index, coord in product(range(2), [-3, 3]):
            pos = np.array([0, 0])
            pos[index] = coord
            assert not in_bounds(pos, bounds)
