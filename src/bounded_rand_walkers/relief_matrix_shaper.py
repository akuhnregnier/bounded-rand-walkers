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


def gen_shaper2D(order_divisions, vertices, verbose=True):
    bounds = DelaunayArray(vertices, Delaunay(vertices))
    CoM = np.array([np.mean(vertices[:, 0]), np.mean(vertices[:, 1])])

    # Upper - lower limits of boundary, as defined in plot.
    x0 = 2
    y0 = 2

    # So that we have no bias in case of asymmetric shape.
    divisions_x = order_divisions
    # divisions_y = order_divisions * int(float(y0) / float(x0))
    divisions_y = order_divisions

    xs = np.linspace(CoM[0] - x0, CoM[0] + x0, divisions_x + 1)
    ys = np.linspace(CoM[1] - y0, CoM[1] + y0, divisions_y + 1)

    xs_centres = get_centres(xs)
    ys_centres = get_centres(ys)

    relief_matrix = np.zeros((divisions_y, divisions_x), dtype=np.int)

    # Replace 0s with 1s in 2d matrix when inside boundary.
    for i, yi in enumerate(ys_centres):
        for j, xj in enumerate(xs_centres):
            relief_matrix[i, j] = in_bounds(np.array([xj, yi]), bounds)

    Z = np.zeros_like(relief_matrix, dtype=np.float64)

    for i, xi in enumerate(
        tqdm(xs_centres, desc="Generating shaper", smoothing=0, disable=not verbose)
    ):
        for j, yi in enumerate(ys_centres):
            Z[i, j] = _new_shaper2D(
                i - int(divisions_x / 2),
                j - int(divisions_y / 2),
                relief_matrix,
                x0,
                y0,
            )

    xsG = xs - CoM[0]
    ysG = ys - CoM[1]
    X, Y = np.meshgrid(xsG, ysG, indexing="ij")

    return X, Y, Z
