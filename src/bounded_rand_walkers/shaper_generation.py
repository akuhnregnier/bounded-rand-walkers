#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generation of shaper functions given a boundary."""
from functools import partial

import numpy as np
from joblib import Memory
from scipy.integrate import dblquad, tplquad
from tqdm.auto import tqdm

from bounded_rand_walkers.cpp import bound_map

from .rad_interp import rotation
from .utils import DelaunayArray, cache_dir, get_centres, in_bounds

memory = Memory(cache_dir, verbose=0)


def _simpsons_inner(f, a, f_a, b, f_b):
    """Calculate the inner term of the adaptive Simpson's method.

    Parameters
    ----------
    f : callable
        Function to integrate.
    a, b : float
        Lower and upper bounds of the interval.
    f_a, f_b : float
        Values of `f` at `a` and `b`.

    Returns
    -------
    m : float
        Midpoint (the mean of `a` and `b`).
    f_m : float
        Value of `f` at `m`.
    whole : float
        Simpson's method result over the interval [`a`, `b`].

    """
    # pprint({k: format(v, '0.3f') for k, v in locals().items() if k != 'f'})
    m = (a + b) / 2
    f_m = f(m)
    return (m, f_m, abs(b - a) / 6 * (f_a + 4 * f_m + f_b))


def _simpsons_outer(f, a, f_a, b, f_b, eps, whole, m, f_m):
    """Recursive implementation of the adaptive Simpson's method.

    Parameters
    ----------
    f : callable
        Function to integrate.
    a, b : float
        Lower and upper bounds of the interval.
    f_a, f_b : float
        Values of `f` at `a` and `b`.
    eps : float
        Maximum integration error.
    whole : float
        Simpson's method result over the interval [`a`, `b`].
    m : float
        Midpoint (the mean of `a` and `b`).
    f_m : float
        Value of `f` at `m`.

    Returns
    -------
    whole : float
        Integral over the given domain.

    """
    l_m, f_lm, left = _simpsons_inner(f, a, f_a, m, f_m)
    r_m, f_rm, right = _simpsons_inner(f, m, f_m, b, f_b)
    new_whole = left + right
    delta = new_whole - whole
    if abs(delta) <= 15 * eps:
        # If we have reached the desired error, return the result, including the
        # correction.
        return new_whole + delta / 15
    # Otherwise split the interval and repeat.
    return _simpsons_outer(
        f, a, f_a, m, f_m, eps / 2, left, l_m, f_lm
    ) + _simpsons_outer(f, m, f_m, b, f_b, eps / 2, right, r_m, f_rm)


def adaptive_simpsons(f, a, b, eps=1e-10, n_start=4, vec_func=False):
    """Adaptive Simpson's rule integration.

    Parameters
    ----------
    f : callable
        Function to integrate.
    a, b : float
        Lower and upper bounds of the domain.
    eps : float
        Maximum integration error.
    n_start : int in [1, inf]
         The number of starting splits.
    vec_func : bool
        If true, `f` may be given an array of points to speed up the calculation.

    Returns
    -------
    result : float
        Integral.

    Raises
    ------
    ValueError
        If `n_start` is not in [1, inf].

    """
    if n_start < 1:
        raise ValueError("'n_start' needs to be in [1, inf].")

    interval_edges = np.linspace(a, b, n_start + 1)

    if vec_func:
        f_vals = f(interval_edges)
    else:
        f_vals = np.array([f(v) for v in interval_edges])

    # Only deal with intervals where the function is non-zero at least one of the
    # edges.
    close_zero = partial(np.isclose, b=0, rtol=0, atol=1e-15)
    non_zero = ~(close_zero(f_vals[1:]) & close_zero(f_vals[:-1]))

    result = 0
    for (start, end, f_start, f_end) in zip(
        interval_edges[:-1][non_zero],
        interval_edges[1:][non_zero],
        f_vals[:-1][non_zero],
        f_vals[1:][non_zero],
    ):
        m, f_m, whole = _simpsons_inner(f, start, f_start, end, f_end)
        result += _simpsons_outer(f, start, f_start, end, f_end, eps, whole, m, f_m)
    return result


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
    "square": {"x_y_function": square_shaper, "rot_symmetric": False},
    "circle": {"x_y_function": circle_shaper, "rot_symmetric": True},
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
    bounds = DelaunayArray(vertices)

    shaper_data = np.empty((shaper_radii.size, n_theta))
    for i, radius in enumerate(
        tqdm(shaper_radii, desc="Shaper radii", smoothing=0, disable=not verbose)
    ):
        for j, theta in enumerate(np.linspace(0, 2 * np.pi, n_theta, endpoint=False)):
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


@memory.cache(ignore=["verbose"])
def gen_rad_shaper_exact(shaper_radii, vertices, verbose=True, n_start=1000):
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
    n_start : int
        Only applies if integration is performed over theta only, and determines the
        number of initial splits for the integration.

    Returns
    -------
    shaper : 1D array
        Shaper values at `shaper_radii`.

    Raises
    ------
    KeyError
        If `vertices` is a str, but no matching entry is found in either `shaper_map`
        or `bounded_rand_walkers.cpp.bound_map`.

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
        bounds = DelaunayArray(vertices)

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
        shaper[i] = adaptive_simpsons(
            lambda theta: shaper_func["x_y_function"](*rotation(radius, 0, theta)),
            0,
            2 * np.pi,
            n_start=n_start,
            vec_func=True,
        )
    return shaper


def gen_shaper2D(vertices, x_edges, y_edges, verbose=True):
    """Approximate the shaper function in 2D.

    Parameters
    ----------
    vertices : array
        Vertices of the boundary.
    x_edges : array of shape (M,)
        x-axis bin edges.
    y_edges : array of shape (N,)
        y-axis bin edges.
    verbose : bool
        If True, show a progress bar.

    Returns
    -------
    shaper : array of shape (M - 1, N - 1)
        Numerically approximated shaper function at the bins given by `x_edges` and
        `y_edges`.

    """
    bounds = DelaunayArray(vertices)
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
            comp_reliefs = relief_matrix & np.roll(
                np.roll(relief_matrix, j - y_divisions // 2, axis=0),
                i - x_divisions // 2,
                axis=1,
            )
            shaper[i, j] = np.sum(comp_reliefs)

    norm_factor = 4 * (x0 / relief_matrix.shape[0]) * (y0 / relief_matrix.shape[1])
    shaper *= norm_factor

    return shaper
