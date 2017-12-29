#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 18:42:24 2017

@author: luca
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from data_generation import DelaunayArray, Delaunay, in_bounds
from scipy.signal import correlate2d
from utils import get_centres
import math


def newShaper2D(x_shift, y_shift, relief_matrix, x0, y0):
    #print('first', np.shape(relief_matrix) )
    #print('second', np.shape(np.roll(np.roll(relief_matrix,y_shift,axis=0),x_shift,axis=1)) )
    comp_reliefs = relief_matrix & np.roll(
        np.roll(relief_matrix, y_shift, axis=0), x_shift, axis=1)
    return (np.sum(comp_reliefs) * (2 * x0 / float(relief_matrix.shape[0]))
            * (2 * y0 / float(relief_matrix.shape[1])))


def gen_shaper2D(order_divisions, vertices):
    bounds = DelaunayArray(vertices, Delaunay(vertices))
    CoM = np.array([np.mean(vertices[:, 0]), np.mean(vertices[:, 1])])

    # upper - lower limits of boundary, as defined in plot
    x0 = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
    y0 = np.max(vertices[:, 1]) - np.min(vertices[:, 1])

    # so that we have no bias in case of asymmetric shape
    divisions_x = order_divisions
    divisions_y = order_divisions * int(float(y0) / float(x0))

    xs = np.linspace(CoM[0] - x0,  CoM[0] + x0,  divisions_x + 1)
    # issue with centering of bins?
    ys = np.linspace(CoM[1] - y0,  CoM[1] + y0,  divisions_y + 1)

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
                y0)

    # contour plot

    # xsG = np.linspace(-x0, x0,  divisions_x)
    # ysG = np.linspace(-y0, y0,  divisions_y)
    xsG = xs - CoM[0]
    ysG = ys - CoM[1]
    X, Y = np.meshgrid(xsG, ysG, indexing='ij')

    return X, Y, Z


def gen_shaper2D_alt(order_divisions, vertices):

    bounds = DelaunayArray(vertices, Delaunay(vertices))
    CoM = np.array([np.mean(vertices[:, 0]), np.mean(vertices[:, 1])])
    x0 = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
    y0 = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    divisions_x = order_divisions
    divisions_y = order_divisions * int(float(y0) / float(x0))
    xs = np.linspace(CoM[0] - x0,  CoM[0] + x0,  divisions_x + 1)
    ys = np.linspace(CoM[1] - y0,  CoM[1] + y0,  divisions_y + 1)
    xsG = xs - CoM[0]
    ysG = ys - CoM[1]
    X, Y = np.meshgrid(xsG, ysG, indexing='ij')
    xs_centres = get_centres(xs)
    ys_centres = get_centres(ys)
    relief_matrix = np.zeros((divisions_y, divisions_x), dtype=np.int)
    for i, yi in enumerate(ys_centres):
        for j, xj in enumerate(xs_centres):
            relief_matrix[i, j] = in_bounds(np.array([xj, yi]), bounds)

    # transpose due to the way the correlation is carried out - shifting
    # entries in the matrix 'right' (if printed as a 2d matrix) does not
    # correspond to the x direction, which goes 'down' in 2d matrix
    # notation (numpy refers to it as indexing='ij').
    Z = correlate2d(relief_matrix, relief_matrix).T

    # n -> 2n - 1
    # This relies on rounding down
    trim_x = int(math.floor((order_divisions - 1) / 2.))
    trim_y = int(math.floor((order_divisions - 1) / 2.))
    modifier = (order_divisions + 1) % 2  # 1 if n is even, 0 if n is odd
    # Z = Z[trim_x + modifier:Z.shape[0] - trim_x,
    #         trim_y + modifier:Z.shape[1] - trim_y]
    Z = Z[trim_x:Z.shape[0] - trim_x - modifier,
            trim_y:Z.shape[1] - trim_y - modifier]

    Z = np.asarray(Z, dtype=np.float64)

    cell_area = (xs[1] - xs[0]) * (ys[1] - ys[0])

    # normalise
    Z /= np.sum(Z * cell_area)

    # transpose to make compatible with output above
    return X, Y, Z

if __name__ == '__main__':
    plt.close('all')
    order_divisions = 200

    vertices = np.array([0.1, 0.3, 0.25, 0.98, 0.9, 0.9, 0.7, 0.4, 0.4, 0.05])
    vertices = vertices.reshape(int(len(vertices) / 2), 2)

    #####
    x0 = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
    y0 = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    divisions_x = order_divisions
    divisions_y = order_divisions * int(float(y0) / float(x0))
    xs = np.linspace(- x0, x0, divisions_x + 1)
    ys = np.linspace(- y0, y0, divisions_y + 1)
    cell_area = (xs[1] - xs[0]) * (ys[1] - ys[0])
    #####

    X, Y, Z = gen_shaper2D(order_divisions, vertices)

    # normalise
    Z /= np.sum(Z * cell_area)

    X2, Y2, Z2 = gen_shaper2D_alt(order_divisions, vertices)


    print("shapes")
    print(Z.shape)
    print(Z2.shape)

    plt.figure()
    plt.pcolormesh(X, Y, Z)
    plt.colorbar()

    plt.figure()
    plt.pcolormesh(X, Y, Z2)
    plt.colorbar()

    print("Zs are close")
    print(np.all(np.isclose(Z, Z2)))
    print(np.mean(Z-Z2))
    print(np.min(Z-Z2))
    print(np.max(Z-Z2))
    print(np.std(Z-Z2))
