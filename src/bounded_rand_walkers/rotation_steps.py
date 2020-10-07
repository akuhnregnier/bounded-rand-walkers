# -*- coding: utf-8 -*-
"""
Code for modifying sequence of positions via rotations to make the
asymmetry clear Valid for 2D case

"""
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from scipy import integrate
from scipy.spatial import Delaunay

from .c_g2D import c_g2D_func
from .data_generation import DelaunayArray, in_bounds
from .utils import get_centres


def get_pdf_transform_shaper(steps, geometry="circle"):
    """Calculate the shaper function at varying radial distances from the centre.

    Radial distances from the centre refer to intrinsic step sizes here.

    pdf f gives probability p(stepsize) of
    transformed pdf

    steps : float

    geometry is a str {'1Dseg', 'circle'}

    """
    if geometry == "1Dseg":
        return (1 - np.abs(steps)) * 0.5 * (np.sign(1 - np.abs(steps)) + 1)

    # old: 1circle

    if geometry == "circle":
        shaper = np.zeros_like(steps, dtype=np.float64)
        mask = steps < 2.0
        shaper[mask] = 2 * np.arccos(steps[mask] / 2) - 0.5 * np.sqrt(
            (4 - steps[mask] ** 2) * steps[mask] ** 2
        )
        return shaper


def g1D(x, f):
    num = integrate.quad(f, -x, 1 - x)
    return num[0]


def g1D_norm(f):
    den = integrate.dblquad(lambda z, y: f(z), 0, 1, lambda z: -z, lambda z: 1 - z)
    return den[0]


def betaCircle(r, l):
    return np.pi - np.arccos((r ** 2 + l ** 2 - 1) / (2 * r * l))


def gRadialCircle(r, f):
    """
    Not yet normalised f is a function of radial distance from starting
    point of step (1D pdf)
    e.g. for a flat infinitely large top hat in 2D, the associated radial
        1D distribution goes as 1/l in which case we expect the probability for
        the position to be uniform within the circle, hence the radial one to
        grow linearly (as observed).

    """
    if np.isclose(r, 0.0):
        return sp.integrate.quad(lambda d: f(d, 0) * 2 * np.pi * d, 0, 1 - r)[0]
    return (
        sp.integrate.quad(lambda d: f(d, 0) * 2 * np.pi * d, 0, 1 - r)[0]
        + sp.integrate.quad(
            lambda l: 2 * np.pi * l * f(l, 0) * (1 - betaCircle(r, l) / np.pi),
            1 - r,
            1 + r,
        )[0]
    )


def g2D(f, xs_edges, ys_edges, bounds):
    """2D position probability."""
    print("G2D")
    bounds = DelaunayArray(bounds, Delaunay(bounds))
    xs_centres = get_centres(xs_edges)
    ys_centres = get_centres(ys_edges)
    # should be True if the region is within the bounds
    position_mask = np.zeros((xs_centres.shape[0], ys_centres.shape[0]), dtype=bool)
    for i, x in enumerate(xs_centres):
        for j, y in enumerate(ys_centres):
            is_in_bounds = in_bounds(np.array([x, y]), bounds)
            position_mask[i, j] = is_in_bounds
    x_indices, y_indices = np.where(position_mask)
    g_values = np.asarray(
        c_g2D_func(f, xs_edges, ys_edges, xs_centres, ys_centres, x_indices, y_indices)
    )
    # now need to normalise
    area = (xs_centres[1] - xs_centres[0]) * (ys_centres[1] - ys_centres[0])
    g_values /= np.sum(area * g_values)
    return g_values


if __name__ == "__main__":
    # must be run from iPython!!
    from IPython import get_ipython

    ipython = get_ipython()
    from functions import Funky

    N = 200
    x = np.linspace(-2, 2, N)
    y = x

    ipython.magic(
        "time "
        "plt.imshow"
        "(g2D(Gaussian(centre=np.array([0.0, "
        "0.0]), width=0.1).pdf, x, y))"
    )


if __name__ == "__main__":
    N = 500000
    pos_data2D = np.random.uniform(0, 1, size=(2, N))
    rot_steps_data = rot_steps(pos_data2D)
    plt.figure()
    plt.hist2d(rot_steps_data[0, :], rot_steps_data[1, :], bins=50)
    plt.plot(np.arange(-2, 2, 0.01), np.array([0 for a in np.arange(-2, 2, 0.01)]))
    plt.title("Observed step-size with fixed incoming direction")
    plt.gca().set_aspect("equal")

    from functions import Funky

    pdf = Funky(centre=(0.0, 0.0)).pdf
    bins = 81
    min_x = -1
    max_x = 1
    min_y = -1
    max_y = 1
    xs_edges = np.linspace(min_x, max_x, bins + 1)
    ys_edges = np.linspace(min_y, max_y, bins + 1)
    g_values = g2D(pdf, xs_edges, ys_edges)
    plt.figure()
    plt.pcolormesh(xs_edges, ys_edges, g_values)
    plt.gca().set_aspect("equal")
    plt.show()
