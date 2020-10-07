#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from tqdm.auto import tqdm

from .data_generation import Delaunay, DelaunayArray, in_bounds
from .utils import get_centres


def _new_shaper2D(x_shift, y_shift, relief_matrix, x0, y0):
    comp_reliefs = relief_matrix & np.roll(
        np.roll(relief_matrix, y_shift, axis=0), x_shift, axis=1
    )
    return (
        np.sum(comp_reliefs)
        * (2 * x0 / float(relief_matrix.shape[0]))
        * (2 * y0 / float(relief_matrix.shape[1]))
    )


def gen_shaper2D(vertices, x_edges, y_edges, verbose=True):
    """Generate shaper function in 2D."""
    bounds = DelaunayArray(vertices, Delaunay(vertices))
    CoM = np.array([np.mean(vertices[:, 0]), np.mean(vertices[:, 1])])

    # All calculations are performed relative to the 'CoM' (centre of the bounds).
    # Therefore the bounds do not have to be 'centred' at 0.
    xs = x_edges + CoM[0]
    ys = y_edges + CoM[1]

    xs_centres = get_centres(xs)
    ys_centres = get_centres(ys)

    x_divisions = xs_centres.shape[0]
    y_divisions = ys_centres.shape[0]

    x0 = np.max(x_edges)
    assert np.isclose(x0, np.abs(np.min(x_edges)))
    y0 = np.max(y_edges)
    assert np.isclose(y0, np.abs(np.min(y_edges)))

    # True (1) when inside boundary.
    relief_matrix = np.zeros((x_divisions, y_divisions), dtype=np.int64)
    for i, yi in enumerate(ys_centres):
        for j, xj in enumerate(xs_centres):
            relief_matrix[i, j] = in_bounds(np.array([xj, yi]), bounds)

    shaper = np.zeros_like(relief_matrix, dtype=np.float64)

    for i, xi in enumerate(
        tqdm(xs_centres, desc="Generating shaper", smoothing=0, disable=not verbose)
    ):
        for j, yi in enumerate(ys_centres):
            shaper[i, j] = _new_shaper2D(
                i - int(x_divisions / 2),
                j - int(y_divisions / 2),
                relief_matrix,
                x0,
                y0,
            )

    return shaper
