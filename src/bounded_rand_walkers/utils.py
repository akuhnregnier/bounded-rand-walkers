# -*- coding: utf-8 -*-
"""General utility functions used throughout the project."""
from pathlib import Path

import numpy as np
from numba import njit
from scipy.spatial import Delaunay
from statsmodels.stats.weightstats import DescrStatsW

cache_dir = (Path("~") / ".cache" / "bounded_rand_walkers").expanduser()
cache_dir.mkdir(exist_ok=True)


@njit
def cluster_indices(x):
    """Determine the location of adjacent non-zero elements.

    Parameters
    ----------
    x : array of shape (N,)
        Array to label.

    Returns
    -------
    bounds : array of shape (k, 2)
        The starting and end (non-inclusive) indices of the `k` found clusters in the
        first and second column, respectively.

    """
    x = np.abs(x) > 1e-20
    indices = list(np.where(np.diff(x))[0] + 1)

    if x[0]:
        indices = [0] + indices
    if x[-1]:
        indices = indices + [len(x)]

    indices = np.array(indices)
    bounds = np.empty((indices.shape[0] // 2, 2), dtype=np.int32)
    for (i, (a, b)) in enumerate(zip(indices[::2], indices[1::2])):
        bounds[i] = (a, b)
    return bounds


@njit
def label(x):
    """Label adjacent non-zero elements using the same integer.

    Parameters
    ----------
    x : array of shape (N,)
        Array to label.

    Returns
    -------
    labelled : array of shape (N,)
        Labelled array.
    n_clusters : int
        Number of labelled clusters.

    """
    start_indices, end_indices = cluster_indices(x).T
    labelled = np.zeros_like(x, dtype=np.int32)
    counter = 1
    for start, end in zip(start_indices, end_indices):
        labelled[start:end] = counter
        counter += 1
    return labelled, counter - 1


def approx_edges(x):
    """Approximate bin edges from bin centres."""
    diffs = np.diff(x)
    return np.array(
        [x[0] - diffs[0] / 2] + list(x[:-1] + diffs / 2) + [x[-1] + diffs[-1] / 2]
    )


def normalise(x, y, return_factor=False):
    """Normalise y such that it integrates to 1 over x."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == y.size:
        # Approximate bin edges.
        edges = approx_edges(x)
    else:
        edges = x

    integral = np.sum(np.diff(edges) * y)
    if return_factor:
        return 1 / integral
    return y / integral


def match_ref(x, y, ref_x, ref_y, interpolate=False, return_factor=False):
    """Normalise y such that it matches a reference dataset as closely as possible.

    Parameters
    ----------
    x : array-like
        x-coordinates.
    y : array-like
        y-coordinates.
    ref_x : array-like
        Reference dataset x-coordinates.
    ref_y : array-like
        Reference dataset y-coordinates.
    interpolate : bool
        If false, only match `y` and `ref_y` for elements where `x` and `ref_x` agree.
        Otherwise, use linear interpolation to compare elements within the overlap
        between `x` and `ref_x`.
    return_factor : bool
        If true, only return the scaling factor.

    Returns
    -------
    matched : array
        Linearly scaled version (i.e. multiplied by a constant) of `y` such that the
        MSE(`y`, `ref_y`) over the overlap between {`x`, `ref_x`} is minimised.

    """
    x = np.asarray(x)
    y = np.asarray(y)
    ref_x = np.asarray(ref_x)
    ref_y = np.asarray(ref_y)

    orig_y = y.copy()

    min_l = min(len(x), len(ref_x))
    x = x[:min_l]
    y = y[:min_l]
    ref_x = ref_x[:min_l]
    ref_y = ref_y[:min_l]

    def get_scale(x, y):
        """Determine linear scaling factor for y that minimises the MSE."""
        return np.sum(x * y) / np.sum(y ** 2)

    if not interpolate:
        # Only consider exactly matching locations.
        matched = np.isclose(x, ref_x)
        factor = get_scale(ref_y[matched], y[matched])
    else:
        # With interpolation, consider the overlap in general.
        common_x = (max(np.min(x), np.min(ref_x)), min(np.max(x), np.max(ref_x)))
        x_mask = (common_x[0] <= x) & (x < common_x[1])
        m_x = x[x_mask]
        m_y = y[x_mask]

        ref_x_mask = (common_x[0] <= ref_x) & (ref_x < common_x[1])
        m_ref_x = ref_x[ref_x_mask]
        m_ref_y = ref_y[ref_x_mask]

        # Always interpolate `y` onto `ref_x` for comparison with `ref_y`.
        interp_x = np.interp(m_ref_x, xp=m_x, fp=m_y)

        factor = get_scale(m_ref_y, interp_x)

    if return_factor:
        return factor
    return factor * orig_y


def get_centres(bin_edges):
    """Get bin centres from edges."""
    bin_edges = np.asarray(bin_edges)

    left_edges = bin_edges[:-1]
    right_edges = bin_edges[1:]
    bin_centres = (left_edges + right_edges) / 2.0
    return bin_centres


def stats(data1, data2, weights=None):
    """Calculate the mean difference between data sets and its standard deviation.

    Parameters
    ----------
    data1 : array
        Dataset 1.
    data2 : array
        Datast 2.
    weights : array or None
        The weights of each data point.

    Returns
    -------
    mean : float
        Mean difference between data sets.
    std_mean : float
        Standard deviation of the mean difference.

    """
    if len(data1) != len(data2):
        raise Exception("Two data sets have different lengths")

    abs_difference = np.abs(data2 - data1)
    weighted_stats = DescrStatsW(abs_difference, weights=weights)

    return weighted_stats.mean, weighted_stats.std_mean


def plot_name_clean(name):
    """Clean up filename strings."""
    replace_pairs = [
        ("{", "_"),
        ("}", "_"),
        ("(", "_"),
        (")", "_"),
        (":", "_"),
        ("'", ""),
        (",", "_"),
        (".", "_"),
        (" ", "_"),
        ("__", "_"),
        ("__", "_"),
        ("__", "_"),
        ("_png", ".png"),
    ]
    for i, j in replace_pairs:
        name = name.replace(i, j)
    return name


class DelaunayArray(np.ndarray):
    """Array subclass to facilitate 2D bounds checking."""

    def __new__(cls, input_array):
        """Calculation of Delaunay triangulation."""
        # `input_array` is an existing ndarray instance.
        # Use view casting to get the desired type.
        obj = np.asarray(input_array).view(cls)
        # Add the `tri` attribute to our newly created instance.
        obj.tri = Delaunay(input_array)
        # Finally, return this newly created object.
        return obj

    def __array_finalize__(self, obj):
        """Set the `tri` attribute for other scenarios if needed."""
        if obj is None:
            return
        self.tri = getattr(obj, "tri", None)


def in_bounds(position, bounds):
    """Test whether the given `position` is within the given `bounds`.

    Parameters
    ----------
    position : array
        Position to test.
    bounds : DelaunayArray
        The bounds for the random walker, given as an array. The shape of the array
        dictates the dimensionality of the problem. A 1D array containing two values
        represents the lower and upper boundaries (in that order) of the 1D problem.
        For the 2D problem, the boundary is given as a 2D array, where each row
        contains the (x, y) coordinates of a point.

    Returns
    -------
    present : bool
        True if `position` is within `bounds`.

    """
    if bounds.shape[1] > 1:
        # More than 1D.
        return np.all(bounds.tri.find_simplex(position) != -1)
    else:
        return (position >= bounds[0]) and (position < bounds[1])


def circle_points(radius=1.0, samples=20):
    """Generate an array of (x, y) coordinates arranged in a circle.

    Parameters
    ----------
    radius : float
        Circle radius.
    samples : int
        How many points to generate along the circle.

    Returns
    -------
    points : array of shape (`samples`, 2)
        Points along the circle.

    """
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    x = (np.cos(angles) * radius).reshape(-1, 1)
    y = (np.sin(angles) * radius).reshape(-1, 1)
    return np.hstack((x, y))
