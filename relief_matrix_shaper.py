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

            Z[i, j] = round(newShaper2D(
                i - int(divisions_x / 2),
                j - int(divisions_y / 2),
                relief_matrix,
                x0,
                y0),
                3
                )

    # contour plot

    # xsG = np.linspace(-x0, x0,  divisions_x)
    # ysG = np.linspace(-y0, y0,  divisions_y)
    xsG = xs - CoM[0]
    ysG = ys - CoM[1]
    X, Y = np.meshgrid(xsG, ysG)

    return X, Y, Z

if __name__ == '__main__':
    order_divisions = 100

    vertices = np.array([0.1, 0.3, 0.25, 0.98, 0.9, 0.9, 0.7, 0.4, 0.4, 0.05])
    vertices = vertices.reshape(int(len(vertices) / 2), 2)

    X, Y, Z = gen_shaper2D(order_divisions, vertices)

    bounds = DelaunayArray(vertices, Delaunay(vertices))
    CoM = np.array([np.mean(vertices[:, 0]), np.mean(vertices[:, 1])])
    x0 = np.max(vertices[:, 0]) - np.min(vertices[:, 0])
    y0 = np.max(vertices[:, 1]) - np.min(vertices[:, 1])
    divisions_x = order_divisions
    divisions_y = order_divisions * int(float(y0) / float(x0))
    xs = np.linspace(CoM[0] - x0,  CoM[0] + x0,  divisions_x + 1)
    ys = np.linspace(CoM[1] - y0,  CoM[1] + y0,  divisions_y + 1)
    xs_centres = get_centres(xs)
    ys_centres = get_centres(ys)
    relief_matrix = np.zeros((divisions_y, divisions_x), dtype=np.int)
    for i, yi in enumerate(ys_centres):
        for j, xj in enumerate(xs_centres):
            relief_matrix[i, j] = in_bounds(np.array([xj, yi]), bounds)

    Z2 = correlate2d(relief_matrix, relief_matrix)
    trim_x = (Z2.shape[0] - Z.shape[0])/2
    trim_y = (Z2.shape[1] - Z.shape[1])/2
    modx = -1 if ((trim_x % 2) == 0) else 0
    mody = -1 if ((trim_y % 2) == 0) else 0
    Z2 = Z2[trim_x - modx:Z2.shape[0] - trim_x, trim_y - mody:Z2.shape[1] - trim_y]


    # plt.figure()
    # CS = plt.contour(X, Y, Z, 7,
    #                  colors='b',
    #                  )
    # plt.clabel(CS, fontsize=9, inline=1)
    plt.figure()
    plt.pcolormesh(X, Y, Z)


    # plt.figure()
    # CS = plt.contour(X, Y, Z2, 7,
    #                  colors='b',
    #                  )
    # plt.clabel(CS, fontsize=9, inline=1)

    plt.figure()
    plt.pcolormesh(X, Y, Z2.T)

    # plt.figure()
    # CS = plt.contour(X, Y, Z2, 7,
    #                  colors='b',
    #                  )
    # plt.clabel(CS, fontsize=9, inline=1)
