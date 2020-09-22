#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import numpy as np

from .data_generation import Delaunay, DelaunayArray, in_bounds
from .utils import get_centres

output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "output"))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


def newShaper2D(x_shift, y_shift, relief_matrix, x0, y0):
    # print('first', np.shape(relief_matrix) )
    # print('second', np.shape(np.roll(np.roll(relief_matrix,y_shift,axis=0),x_shift,axis=1)) )
    comp_reliefs = relief_matrix & np.roll(
        np.roll(relief_matrix, y_shift, axis=0), x_shift, axis=1
    )
    return (
        np.sum(comp_reliefs)
        * (2 * x0 / float(relief_matrix.shape[0]))
        * (2 * y0 / float(relief_matrix.shape[1]))
    )


def gen_shaper2D(order_divisions, vertices):
    bounds = DelaunayArray(vertices, Delaunay(vertices))
    CoM = np.array([np.mean(vertices[:, 0]), np.mean(vertices[:, 1])])

    # upper - lower limits of boundary, as defined in plot
    # x0 = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
    # y0 = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    x0 = 2
    y0 = 2

    # so that we have no bias in case of asymmetric shape
    divisions_x = order_divisions
    # divisions_y = order_divisions * int(float(y0) / float(x0))
    divisions_y = order_divisions

    xs = np.linspace(CoM[0] - x0, CoM[0] + x0, divisions_x + 1)
    # issue with centering of bins?
    ys = np.linspace(CoM[1] - y0, CoM[1] + y0, divisions_y + 1)

    xs_centres = get_centres(xs)
    ys_centres = get_centres(ys)

    relief_matrix = np.zeros((divisions_y, divisions_x), dtype=np.int)

    # now replace 0s with 1s in 2d matrix when inside boundary
    for i, yi in enumerate(ys_centres):
        for j, xj in enumerate(xs_centres):
            relief_matrix[i, j] = in_bounds(np.array([xj, yi]), bounds)

    Z = np.zeros_like(relief_matrix, dtype=np.float64)

    for i, xi in enumerate(xs_centres):
        for j, yi in enumerate(ys_centres):
            # print(str(j + len(ys) * i) + ' over ' + str(len(xs) * len(ys)))

            # Z[i, j] = round(newShaper2D(
            #     i - int(divisions_x / 2),
            #     j - int(divisions_y / 2),
            #     relief_matrix,
            #     x0,
            #     y0),
            #     3
            #     )
            Z[i, j] = newShaper2D(
                i - int(divisions_x / 2),
                j - int(divisions_y / 2),
                relief_matrix,
                x0,
                y0,
            )

    # contour plot

    # xsG = np.linspace(-x0, x0,  divisions_x)
    # ysG = np.linspace(-y0, y0,  divisions_y)
    xsG = xs - CoM[0]
    ysG = ys - CoM[1]
    X, Y = np.meshgrid(xsG, ysG, indexing="ij")

    return X, Y, Z
