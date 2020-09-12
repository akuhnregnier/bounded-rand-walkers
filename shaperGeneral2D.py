#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import time

import numpy as np
from scipy.integrate import dblquad
from scipy.interpolate import RegularGridInterpolator

from data_generation import weird_bounds
from relief_matrix_shaper import gen_shaper2D
from utils import get_centres


def get_weird_shaper(x_centres, y_centres, which="binary", divisions=200):
    """Load a saved shaper function and then interpolate this to the
    desired grid.

    """
    X, Y = np.meshgrid(x_centres, y_centres, indexing="ij")
    if which == "load":
        delta = 2 * np.sqrt(2) / 121.0
        x_centres_orig = np.arange(-np.sqrt(2), np.sqrt(2), delta) + delta / 2.0
        y_centres_orig = np.arange(-np.sqrt(2), np.sqrt(2), delta) + delta / 2.0
        Z_orig = np.load("weird_Z_121.npy")
    if which == "binary":
        X_orig, Y_orig, Z_orig = gen_shaper2D(divisions, weird_bounds)
        x_centres_orig = get_centres(X_orig[:, 0])
        y_centres_orig = get_centres(Y_orig[0, :])
    else:
        raise NotImplementedError("Choice of which:{:}, not supported".format(which))
    interpolate = True
    if (len(x_centres) == len(y_centres)) and (len(x_centres) == divisions):
        print("Not interpolating, as nr. divisions match!")
        interpolate = False
    else:
        print("Interpolating, as nr. divisions don't match!")
    # do the interpolation as a check, since they should return the same
    # result
    interp = RegularGridInterpolator(
        (x_centres_orig, y_centres_orig),
        Z_orig,
        method="linear",
        bounds_error=False,
        fill_value=0.0,
    )
    interp_shaper = interp((X, Y))
    if not interpolate:
        # plt.figure()
        # plt.imshow(interp_shaper)
        # plt.title('Interp')
        # plt.figure()
        # plt.imshow(Z_orig)
        # plt.title('raw')
        assert np.all(np.isclose(interp_shaper, Z_orig))
        return Z_orig
    else:
        return interp_shaper


def disPtLn(m, c, x, y):
    return (-m * x + y - c) / np.sqrt(m ** 2 + 1)


def Theta2D(x, y, m, c, side, k=120):
    """
    Return value of 2D Heaviside Theta with separator being line (m,c)
    """
    if side == "upper":
        return 0.5 + 1 / np.pi * np.arctan(+k * disPtLn(m, c, x, y))  # (-m*x -c +y))
        if side == "lower":
            return 0.5 + 1 / np.pi * np.arctan(
                -k * disPtLn(m, c, x, y)
            )  # (-m*x -c +y))
    else:
        raise Exception("invalid choice of half plane argument for 2d Theta")


def SelectorFn(x, y, vertices):
    """
    Returns 1 for points inside boundary specified by arbitrary vertices and 0 otherwise
    The points are assumed to define a convex bounded space
    vertices := n by 2 array of coordinates
    """
    CoM = np.array([np.mean(vertices[:, 0]), np.mean(vertices[:, 1])])

    flagf = 1
    for nside in range(len(vertices[:, 0]) - 1):

        m = (vertices[nside + 1, 1] - vertices[nside, 1]) / (
            vertices[nside + 1, 0] - vertices[nside, 0]
        )
        c = vertices[nside, 1] - m * vertices[nside, 0]

        if np.sign(-m * CoM[0] - c + CoM[1]) >= 0:
            flagf *= Theta2D(x, y, m, c, "upper")

        else:
            flagf *= Theta2D(x, y, m, c, "lower")

    m = (vertices[0, 1] - vertices[-1, 1]) / (vertices[0, 0] - vertices[-1, 0])
    c = vertices[0, 1] - m * vertices[0, 0]

    if np.sign(-m * CoM[0] - c + CoM[1]) >= 0:
        flagf *= Theta2D(x, y, m, c, "upper")

    else:
        flagf *= Theta2D(x, y, m, c, "lower")

    return flagf


def genShaper(x, y, vertices):
    """
    #rescale x coordinates to fit in 1x1 square
    vertices[:,0] += min(vertices[:,0])
    vertices[:,0] /= max(vertices[:,0])

    #rescale y coordinates to fit in 1x1 square
    vertices[:,1] += min(vertices[:,1])
    vertices[:,1] /= max(vertices[:,1])
    """
    shaper = dblquad(
        lambda a, b: SelectorFn(a, b, vertices) * SelectorFn(x + a, y + b, vertices),
        0,
        1,
        lambda x: 0,
        lambda x: 1,
        epsabs=1e-3,
    )
    return shaper[0]


if __name__ == "__main__":
    # vertices = np.array([0.01,0,0,1,0.99,1,1,0.01]) #squre
    # vertices= np.array([0,0,0.01,1,1,0.5]) #triangle
    vertices = np.array([0.1, 0.3, 0.25, 0.98, 0.9, 0.9, 0.7, 0.4, 0.4, 0.05])
    vertices = vertices.reshape(int(len(vertices) / 2), 2)
    resc_vertices = np.copy(vertices)
    """
    #rescale x coordinates to fit in 1x1 square
    resc_vertices[:,0] += min(vertices[:,0])
    resc_vertices[:,0] /= max(resc_vertices[:,0])

    #rescale y coordinates to fit in 1x1 square
    resc_vertices[:,1] += min(vertices[:,1])
    resc_vertices[:,1] /= max(resc_vertices[:,1])
    """

    delta = 2 * np.sqrt(2) / 121.0
    x = np.arange(-np.sqrt(2), np.sqrt(2), delta) + delta / 2.0
    y = np.arange(-np.sqrt(2), np.sqrt(2), delta) + delta / 2.0
    X, Y = np.meshgrid(x, y)

    Z = np.zeros((len(x), len(y)))
    start = time.time()
    for i, xi in enumerate(x):
        for j, yi in enumerate(y):
            print((str(j + len(y) * i) + " over " + str(len(x) * len(y))))

            if j + len(y) * i == 10:
                stop = time.time()
                print(
                    (
                        "Predicted runtime: "
                        + str(int(len(x) * len(y) / 10.0 * (stop - start) / 60.0 * 5))
                        + " minutes"
                    )
                )

            Z[i, j] = round(
                genShaper(xi, yi, resc_vertices), 3
            )  # gSquare2D(xi+delta/2.,yi+delta/2.,30)
            # Z[i,j] = SelectorFn(xi,yi,resc_vertices)

    print("calculations done")

    np.save("weird_Z_{:}".format(len(x)), Z)

    # matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
    # plt.figure()
    # CS = plt.contour(X, Y, Z, 7,
    #                  colors='b',
    #                  )
    # plt.clabel(CS, fontsize=9, inline=1)
    #
    # plt.figure()
    # plt.contourf(X,Y,Z)
    # plt.colorbar()
    # print(vertices[:,0],vertices[:,1])
    # plt.scatter(vertices[:,0],vertices[:,1])
