#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generation of shaper functions given a boundary."""

import numpy as np
from joblib import Memory
from scipy.integrate import dblquad, quad, tplquad
from tqdm.auto import tqdm

from bounded_rand_walkers.cpp import bound_map

from .data_generation import Delaunay, DelaunayArray, in_bounds
from .rad_interp import rotation
from .utils import cache_dir

memory = Memory(cache_dir)


# Define shaper functions that use analytic results.


def square_shaper(x, y, side_length=1):
    """Shaper function for a square with a given side length.

    Parameters
    ----------
    x : float or array-like
        x-component of the step.
    y : float or array-like
        y-component of the step.
    side_length : float
        Square side length.

    Returns
    -------
    shaper : float or array-like
        Value of the shaper function for a square geometry at `x`, `y`.

    """
    # Calculate the relevant quantities for each axis.
    rel_x = side_length - np.abs(x)
    rel_y = side_length - np.abs(y)

    return (rel_x) * (rel_y) * ((rel_x > 0) * (rel_y > 0)).astype("float")


def circle_shaper(x, y, radius=1):
    """Shaper function for a square with a given side length.

    Parameters
    ----------
    x : float or array-like
        x-component of the step.
    y : float or array-like
        y-component of the step.
    radius : float
        Circle radius.

    Returns
    -------
    shaper : float or array-like
        Value of the shaper function for a square geometry at `x`, `y`.

    """
    # Calculate the relevant quantities for each axis.
    x_2 = (x / radius) ** 2
    y_2 = (y / radius) ** 2
    distance = np.sqrt(x_2 + y_2)

    out = 2 * np.arccos(distance / 2) - (1 / 2) * ((4 - x_2 - y_2) * (x_2 + y_2)) ** 0.5
    out[np.isnan(out)] = 0
    return out


shaper_map = {
    "square": {
        "x_y_function": square_shaper,
        "rot_symmetric": False,
    },
    "circle": {
        "x_y_function": circle_shaper,
        "rot_symmetric": True,
    },
}


def _gen_shaper(x, y, bounds):
    """Low-level integration function used to calculate shaper function values.

    Parameters
    ----------
    x : float
        x-coordinate at which to calculate the shaper function.
    y : float
        y-coordinate at which to calculate the shaper function.
    bounds : DelaunayArray
        Boundary for which to calculate the shaper function.

    Returns
    -------
    shaper_function : float
        Shaper function value at (`x`, `y`), given `bounds`.

    """
    shaper = dblquad(
        lambda a, b: in_bounds(np.array([a, b]), bounds)
        * in_bounds(np.array([x + a, y + b]), bounds),
        -1,
        1,
        -1,
        1,
        epsabs=1e-3,
    )
    return shaper[0]


def gen_rad_shaper(shaper_radii, vertices, n_theta, verbose=True):
    """Generate the radially averaged shaper function values.

    Only applicable to radially symmetric step size distributions.

    Parameters
    ----------
    shaper_radii : 1D array
        Radii for which to calculate shaper values.
    vertices : 2D array
        (n, 2) array containing the boundary vertices in a clockwise order. The last
        vertex (last row) should be equal to the first vertex.
    n_theta : int
        Number of angles per radius to compute the shaper function for.
    verbose : bool
        If true, display progress updates.

    Returns
    -------
    shaper : 1D array
        Shaper values at `shaper_radii`.

    """
    bounds = DelaunayArray(vertices, Delaunay(vertices))

    shaper_data = np.empty((shaper_radii.size, n_theta))
    for i, radius in enumerate(
        tqdm(shaper_radii, desc="Shaper radii", smoothing=0, disable=not verbose)
    ):
        for j, theta in enumerate(
            np.linspace(0, 2 * np.pi, n_theta, endpoint=False),
        ):
            shaper_data[i, j] = _gen_shaper(*rotation(radius, 0, theta), bounds)
    return np.mean(shaper_data, axis=1)


def _gen_shaper_exact(l, bounds):
    """Low-level integration function used to calculate shaper function values.

    Parameters
    ----------
    l : float
        Step length.
    bounds : DelaunayArray
        Boundary for which to calculate the shaper function.

    Returns
    -------
    shaper_function : float
        Shaper function value at (`x`, `y`), given `bounds`.

    """
    shaper = tplquad(
        lambda theta, y, x: in_bounds(rotation(l, 0, theta), bounds)
        * in_bounds(rotation(l, 0, theta) + np.array([x, y]), bounds),
        -1,  # x lower bound: -1.
        1,  # x upper bound: 1.
        -1,  # y lower bound: -1.
        1,  # y upper bound: 1.
        0,  # theta lower bound: 0.
        2 * np.pi,  # Theta upper bound: 2 pi.
        epsabs=1e-1,
    )
    return shaper[0]


def gen_rad_shaper_exact(shaper_radii, vertices, verbose=True):
    """Generate the radially averaged shaper function values.

    Only applicable to radially symmetric step size distributions.

    Parameters
    ----------
    shaper_radii : 1D array
        Radii for which to calculate shaper values.
    vertices : 2D array or str
        (n, 2) array containing the boundary vertices in a clockwise order. The last
        vertex (last row) should be equal to the first vertex. If a str is given, an
        analytical function will be retrieved from `shaper_map`. If no matching
        function is found, vertices will be retrieved from
        `bounded_rand_walkers.cpp.bound_map` which will then be used to calculate the
        shaper function.
    verbose : bool
        If true, display progress updates.

    Returns
    -------
    shaper : 1D array
        Shaper values at `shaper_radii`.

    Raises
    ------
    KeyError : If `vertices` is a str, but no matching entry is found in either
        `shaper_map` or `bounded_rand_walkers.cpp.bound_map`.

    """
    shaper_func = None
    if isinstance(vertices, str):
        if vertices in shaper_map:
            # Retrieve shaper function (function of (x, y)).
            shaper_func = shaper_map[vertices]
        else:
            vertices = bound_map[vertices]()

    if shaper_func is None:
        shaper = np.empty(shaper_radii.size)

        # Integrate the full shaper function numerically.
        bounds = DelaunayArray(vertices, Delaunay(vertices))

        for i, radius in enumerate(
            tqdm(
                shaper_radii,
                desc="Shaper radii (over x, y, theta)",
                smoothing=0,
                disable=not verbose,
            )
        ):
            shaper[i] = _gen_shaper_exact(radius, bounds)
        return shaper

    if shaper_func["rot_symmetric"]:
        # We don't need to do any integration, since the function is already
        # rotationally symmetric.
        return shaper_func["x_y_function"](shaper_radii, 0)

    # We already have a shaper function (x, y), so we only have to integrate over
    # [0, 2 pi] for each radius.
    shaper = np.empty(shaper_radii.size)
    for i, radius in enumerate(
        tqdm(
            shaper_radii,
            desc="Shaper radii (over theta only)",
            smoothing=0,
            disable=not verbose,
        )
    ):
        shaper[i] = quad(
            lambda theta: shaper_func["x_y_function"](*rotation(radius, 0, theta)),
            0,
            2 * np.pi,
            epsabs=1e-5,
        )[0]
    return shaper
