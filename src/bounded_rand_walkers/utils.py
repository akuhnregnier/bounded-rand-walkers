# -*- coding: utf-8 -*-
from pathlib import Path

import numpy as np
from statsmodels.stats.weightstats import DescrStatsW

cache_dir = (Path("~") / ".cache" / "bounded_rand_walkers").expanduser()
cache_dir.mkdir(exist_ok=True)


def approx_edges(x):
    """Approximate bin edges from bin centres."""
    diffs = np.diff(x)
    return np.array(
        [x[0] - diffs[0] / 2] + list(x[:-1] + diffs / 2) + [x[-1] + diffs[-1] / 2]
    )


def normalise(x, y, return_factor=False):
    """Normalise y such that it integrates to 1 over x."""
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
        difference |`y` - `ref_y`| over the overlap between {`x`, `ref_x`} is
        minimised.

    """
    orig_y = y.copy()

    min_l = min(len(x), len(ref_x))
    x = x[:min_l]
    y = y[:min_l]
    ref_x = ref_x[:min_l]
    ref_y = ref_y[:min_l]

    if not interpolate:
        # Only consider exactly matching locations.
        matched = np.isclose(x, ref_x)
        factor = np.mean(ref_y[matched] / y[matched])
    else:
        # With interpolation, consider the overlap in general.
        common_x = (
            max(np.min(x), np.min(ref_x)),
            min(np.max(x), np.max(ref_x)),
        )
        x_mask = (common_x[0] <= x) & (x < common_x[1])
        m_x = x[x_mask]
        m_y = y[x_mask]

        ref_x_mask = (common_x[0] <= ref_x) & (ref_x < common_x[1])
        m_ref_x = ref_x[ref_x_mask]
        m_ref_y = ref_y[ref_x_mask]

        # Always interpolate `y` onto `ref_x` for comparison with `ref_y`.
        interp_x = np.interp(m_ref_x, xp=m_x, fp=m_y)

        factor = np.mean(m_ref_y / interp_x)

    if return_factor:
        return factor
    return factor * orig_y


def get_centres(bin_edges):
    left_edges = bin_edges[:-1]
    right_edges = bin_edges[1:]
    bin_centres = (left_edges + right_edges) / 2.0
    return bin_centres


def stats(data1, data2, weights=None):
    """
    This function calculates the mean difference between the input data sets
    and the standard deviation of this mean.

    Args:
        data1: 1D array of dataset 1
        data2: 2D array of dataset 2
        weights: The wheights of each data point. The default are no weights.

    Returns:
        weighted_stats.mean: mean difference between data sets
        weighted_stats.std_mean: standard dev. of mean difference

    """
    if len(data1) != len(data2):
        raise Exception("Two data sets have different lengths")

    abs_difference = np.abs(data2 - data1)
    weighted_stats = DescrStatsW(abs_difference, weights=weights)

    return weighted_stats.mean, weighted_stats.std_mean


def plot_name_clean(name):
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
