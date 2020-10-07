# -*- coding: utf-8 -*-
"""Averaging of a 2D distribution to a 1D 'slice'."""
from math import ceil
from numbers import Integral, Real

import numpy as np
from scipy.interpolate import griddata
from tqdm.auto import tqdm

from bounded_rand_walkers.utils import get_centres


def rotation(x, y, angle):
    """Rotate x, y position by angle theta."""

    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)

    return x_rot, y_rot


def exact_radii_interp(
    data,
    x_centres,
    y_centres,
    normalisation="multiply",
    bin_samples=None,
    # XXX:
    mode=1,
):
    """Averaging of data at all bin centre radii.

    Parameters
    ----------
    data : 2D array
        Data array.
    x_centres : 1D array
        x-coordinate bin centres.
    y_centres : 1D array
        y-coordinate bin centres.
    normalisation : {'multiply', 'none'}
        If 'multiply' (default), data is multiplied by the radii to yield integrated
        values as opposed to densities.
    bin_samples : None, int, or float
        The number of samples per bin. If None, samples will be returned for all
        unique radii. Giving an int results in bins containing at least this many
        samples (if there are more than `bin_samples` samples in `data`), with the
        last two bins being joined to satisfy this constraint. If a float is given,
        it will be interpreted as a fraction of the total number of samples in `data`,
        where each bin will contain at least this number of samples (see above).

    Returns
    -------
    radii : 1D array
        Radii at which sampling was carried out.
    sampled : 1D array
        Sampled values along the concentric circles.

    Raises
    ------
    ValueError : If `normalisation` is not in {'multiply', 'none'}.

    """
    if not normalisation in ("multiply", "none"):
        raise ValueError("normalisation can be one of {'divide', 'none'}.")

    coords = np.array(np.meshgrid(x_centres, y_centres, indexing="ij"))
    grid_radii = np.linalg.norm(coords, axis=0)
    unique_radii = np.unique(grid_radii)

    sampled = np.empty(unique_radii.size)
    n_samples = np.empty(unique_radii.size, dtype=np.int64)

    total = 0
    for i, radius in enumerate(unique_radii):
        # Alternative normalisation: do not multiply by the radius.
        matched = np.isclose(grid_radii, radius, rtol=0, atol=1e-17)

        n_matched = np.sum(matched)
        n_samples[i] = n_matched
        total += n_matched

        if normalisation == "none":
            sampled[i] = np.average(data[matched])
        else:
            sampled[i] = np.average(data[matched]) * radius

    assert total == grid_radii.size, "Each bin should only be assigned a value once."

    # Now combine bins if requested.
    if isinstance(bin_samples, Real):
        # Use the given fraction.
        bin_samples = ceil(bin_samples * data.size)

    if isinstance(bin_samples, Integral):
        # Use the given number of samples per bin.
        combined = []
        combined_radii = []

        agg_vals = []
        agg_rads = []
        agg_samples = []
        n_agg = 0
        prev_n = 0
        for (i, (value, radius, n_bin)) in enumerate(
            zip(sampled, unique_radii, n_samples)
        ):
            agg_rads.append(radius)
            agg_vals.append(value)
            agg_samples.append(n_bin)
            n_agg += n_bin

            if n_agg >= bin_samples:
                # If the required number of samples has been reached, combine the
                # individual values into a single new bin.

                if mode == 1:
                    combined_radii.append(np.average(agg_rads, weights=agg_samples))
                    combined.append(np.average(agg_vals, weights=agg_samples))
                elif mode == 2:
                    combined_radii.append(np.average(agg_rads, weights=agg_rads))
                    combined.append(np.average(agg_vals, weights=agg_rads))
                elif mode == 3:
                    combined_radii.append(np.mean(agg_rads))
                    combined.append(np.mean(agg_vals))

                # Reset the variables keeping track of the progress.
                agg_rads = []
                agg_vals = []
                agg_samples = []

                # Store the previous count in case the last two bins need to be
                # combined, so that the proper weights can be used.
                prev_n = n_agg
                n_agg = 0

        if n_agg > 0:
            # If data is left over that wasn't combined previously, add this to the
            # last bin with the proper weighting.

            # Compute the new average as usual.
            new_rad = np.average(agg_rads, weights=agg_samples)
            new_val = np.average(agg_vals, weights=agg_samples)

            # Now combine with the last bin.
            combined_radii[-1] = np.average(
                [combined_radii[-1], new_rad], weights=[prev_n, n_agg]
            )
            combined[-1] = np.average([combined[-1], new_val], weights=[prev_n, n_agg])
        return np.array(combined_radii), np.array(combined)

    return unique_radii, sampled


def inv_exact_radii_interp(
    radii, rad_data, x_centres, y_centres, normalisation="divide"
):
    """Inverse of `exact_radii_interp`.

    Conversion of data at given radii to a full 2D grid.

    Parameters
    ----------
    radii : 1D array
        Radii corresponding to `rad_data`. These should be sorted and unique.
    rad_data : 1D array
        Data array.
    x_centres : 1D array
        x-coordinate bin centres.
    y_centres : 1D array
        y-coordinate bin centres.
    normalisation : {'divide', 'none'}
        If 'divide' (default), `rad_data` is divided by `radii`.

    Returns
    -------
    data : 2D array
        Data array containing the values in `rad_data`.

    Raises
    ------
    ValueError : If `radii` are not sorted in ascending order and unique.
    ValueError : If `normalisation` is not in {'divide', 'none'}.

    """
    if not np.all(np.isclose(radii, np.unique(radii))):
        raise ValueError("radii should be sorted in ascending order and unique.")

    if not normalisation in ("divide", "none"):
        raise ValueError("normalisation can be one of {'divide', 'none'}.")

    coords = np.array(np.meshgrid(x_centres, y_centres, indexing="ij"))
    grid_radii = np.linalg.norm(coords, axis=0)

    data = np.empty(grid_radii.shape)

    total = 0
    for (i, (radius, single_rad_data)) in enumerate(zip(radii, rad_data)):
        # Alternative normalisation: do not divide by the radius.

        matched = np.isclose(grid_radii, radius, rtol=0, atol=1e-17)
        total += np.sum(matched)
        if normalisation == "none":
            data[matched] = single_rad_data
        else:
            data[matched] = single_rad_data / radius

    assert total == grid_radii.size, "Each bin should only be assigned a value once."

    return data


def radial_interp(
    data, x_centres, y_centres, num_rad_factor=2.0, num_points_factor=2.0
):
    """Radial averaging of binned 2D data along concentric circles.

    Parameters
    ----------
    data : 2D array
        Data array.
    x_centres : 1D array
        x-coordinate bin centres.
    y_centres : 1D array
        y-coordinate bin centres.
    num_rad_factor : float
        Ratio between the maximum radius divided by the minimum bin width (i.e. the
        smallest difference between bin centres out of the x and y bins) and the
        number of concentric circles used to sample the distribution.
    num_points_factor : float
        Ratio between the radius of the sampling circles divided by the minimum bin
        width (i.e. the smallest difference between bin centres out of the x and y
        bins) and the number of points along each circle.

    Returns
    -------
    radii : 1D array
        Radii at which sampling was carried out.
    sampled : 1D array
        Sampled values along the concentric circles.

    """
    fill_val = -9.0

    # Create x and y arrays.
    pos_arrays = np.meshgrid(x_centres, y_centres, indexing="ij")

    pos = np.empty((data.size, 2))
    pos[:, 0] = pos_arrays[0].ravel()  # x positions.
    pos[:, 1] = pos_arrays[1].ravel()  # y positions.

    max_x = np.max(np.abs(pos[:, 0]))
    max_y = np.max(np.abs(pos[:, 1]))
    max_rad = np.sqrt(max_x ** 2 + max_y ** 2)

    # Calculate the points at which to interpolate the binned data.
    smallest_width = min(np.min(np.diff(x_centres)), np.min(np.diff(y_centres)))
    # Number of concentric circles.
    num_rad = ceil((max_rad / smallest_width) * num_rad_factor)
    radii = get_centres(np.linspace(0, max_rad, num_rad + 1))
    # Number of points along each of those circles.
    num_points = np.ceil(
        (2 * np.pi * radii / smallest_width) * num_points_factor
    ).astype(np.int64)

    points = []
    for radius, n_point in zip(radii, num_points):
        # Rotate around z axis.
        for angle in np.linspace(0, 2 * np.pi, n_point, endpoint=False):
            points.append(rotation(radius, 0.0, angle))
    points = np.array(points)

    # Carry out interpolation along the concentric circles.
    interp_points = griddata(pos, data.ravel(), points, fill_value=fill_val)

    # Aggregate points at the same radius over concentric circles.
    avg = np.empty(num_rad)
    circle_boundaries = np.append(0, np.cumsum(num_points))
    for (i, (circ_start, circ_end)) in enumerate(
        zip(circle_boundaries[:-1], circle_boundaries[1:])
    ):
        values = interp_points[circ_start:circ_end]
        valid = ~np.isclose(values, fill_val)
        if np.any(valid):
            # Alternative normalisation: do not multiply by the radius.

            # Average here instead of summing in order to avoid the effect of invalid
            # elements.
            avg[i] = np.average(values[valid]) * radii[i]

    # Normalising averages weighted by area.
    # total = np.sum(avg) * max_rad / float(num_rad)
    # avg /= total

    return radii, avg


def radial_interp_circ(data, num_radii, verbose=True):
    """Radial interpolation from 2D distribution.

    Averaging is carried out using concentric circles and unbinned data.

    If the number of data points is large, this function may take very long to
    execute.

    """
    # Calculate distances from the origin.
    distances = np.linalg.norm(data, axis=1)

    radii_edges = np.linspace(0, np.max(distances), num_radii + 1)
    radii = get_centres(radii_edges)

    avg = np.zeros_like(radii)

    for (i, (mean_rad, rad_u)) in enumerate(
        zip(
            tqdm(radii, desc="Averaging over radii", disable=not verbose, smoothing=0),
            radii_edges[1:],
        )
    ):
        # Selected all samples that are between the two concentric circles.
        # Since we are using concentric circles centred at the origin, we can simply
        # test the number of samples that are within the outer circle, removing these
        # elements from the array before the next iteration.
        selection = distances <= rad_u
        avg[i] = np.sum(selection)

        if i < (num_radii - 1):
            # Discard these distances for the next iteration.
            distances = distances[~selection]

    return radii, avg
