# -*- coding: utf-8 -*-
"""Calculation of position probability densities."""
import numpy as np
from numba import njit
from scipy import integrate
from tqdm.auto import tqdm

from .utils import DelaunayArray, get_centres, in_bounds


def g1D(f, x_centres, verbose=True):
    """Calculate 1D position probabilities for the domain [0, 1].

    Parameters
    ----------
    f : callable
        Intrinsic step size distribution.
    x_centres : array of shape (N,)
        Locations at which to calculate the position probability.
    verbose : bool
        If True, display a progress bar.

    Returns
    -------
    probs : array of shape (N,)
        Position probabilities at `x_centres`.

    Raises
    ------
    ValueError
        If any values in `x_centres` are not in [0, 1].

    """
    if np.any((x_centres < 0) | (x_centres > 1)):
        raise ValueError("All `x_centres` must be in [0, 1].")

    num = np.empty((x_centres.shape[0],))
    for i, x in enumerate(
        tqdm(x_centres, desc="Calculating 1D pos. prob.", disable=not verbose)
    ):
        num[i] = integrate.quad(lambda x: f(np.array([x])), -x, 1 - x)[0]
    return num


@njit
def _g2D_func(f, xs_edges, ys_edges, xs_centres, ys_centres, x_indices, y_indices):
    """Calculate the position probability numerically."""
    pos = np.empty((2,))
    x_mod = np.empty((xs_centres.shape[0],))
    y_mod = np.empty((ys_centres.shape[0],))
    g_values = np.zeros((xs_centres.shape[0], ys_centres.shape[0]))

    for k in range(x_indices.shape[0]):
        mask_x_index = x_indices[k]
        mask_y_index = y_indices[k]

        # Evaluate the pdf at each position relative to the current positions. But
        # only iterate over the positions that are actually in the boundary.
        for i in range(xs_centres.shape[0]):
            x_mod[i] = xs_centres[i] - xs_centres[mask_x_index]
        for j in range(ys_centres.shape[0]):
            y_mod[j] = ys_centres[j] - ys_centres[mask_y_index]

        for l in range(x_indices.shape[0]):
            mask_x_index2 = x_indices[l]
            mask_y_index2 = y_indices[l]
            pos[0] = x_mod[mask_x_index2]
            pos[1] = y_mod[mask_y_index2]
            g_values[mask_x_index2, mask_y_index2] += f(pos)

    return g_values


def g2D(f, xs_edges, ys_edges, bounds, verbose=True):
    """Calculate 2D position probabilities.

    Parameters
    ----------
    f : callable
        Intrinsic step size distribution.
    xs_edges : array of shape (M,)
        x-coordinates of the bin edges for the grid on which to calculate the position
        probability.
    ys_edges : array of shape (N,)
        y-coordinates of the bin edges for the grid on which to calculate the position
        probability.
    bounds : array
        Vertices defining the shape boundary.
    verbose : bool
        If True, display a progress bar.

    Returns
    -------
    probs : array of shape (M - 1, N - 1)
        Position probabilities at the centres of the grid defined by `xs_edges` and
        `ys_edges`.

    """
    bounds = DelaunayArray(bounds)
    xs_centres = get_centres(xs_edges)
    ys_centres = get_centres(ys_edges)

    # Should be True if the region is within the bounds.
    position_mask = np.zeros((xs_centres.shape[0], ys_centres.shape[0]), dtype=bool)
    for i, x in enumerate(
        tqdm(xs_centres, desc="Calculating 2D pos. prob.", disable=not verbose)
    ):
        for j, y in enumerate(ys_centres):
            position_mask[i, j] = in_bounds(np.array([x, y]), bounds)
    x_indices, y_indices = np.where(position_mask)

    g_values = np.asarray(
        _g2D_func(f, xs_edges, ys_edges, xs_centres, ys_centres, x_indices, y_indices)
    )
    # Normalise.
    area = (xs_centres[1] - xs_centres[0]) * (ys_centres[1] - ys_centres[0])
    g_values /= np.sum(area * g_values)
    return g_values
