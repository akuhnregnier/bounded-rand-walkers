# -*- coding: utf-8 -*-
"""Averaging of 2D distributions to 1D radial distributions."""
from math import ceil
from numbers import Integral, Real

import numpy as np


def rotation(x, y, angle):
    """Rotate the x, y position by angle theta.

    Parameters
    ----------
    x, y : float
        x-coord and y-coord.
    angle : float
        Angle (radians).

    Returns
    -------
    x_rot, y_rot : float
        Rotated position.

    Examples
    --------
    >>> import numpy as np
    >>> np.allclose(rotation(0, 1, np.pi), [0, -1])
    True
    >>> np.allclose(rotation(0, 1, np.pi / 2), [-1, 0])
    True

    """
    sin_angle = np.sin(angle)
    cos_angle = np.cos(angle)

    x_rot = x * cos_angle - y * sin_angle
    y_rot = x * sin_angle + y * cos_angle

    return x_rot, y_rot


def exact_radii_interp(
    data,
    x_centres,
    y_centres,
    normalisation="multiply",
    bin_samples=None,
    bin_width=None,
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
    bin_width : None or float
        Alternative binning mechanism to the above option. If a float is given, this
        determines the bin width to be used, starting at 0. If no bins are found in an
        interval, this interval will be ignored. Otherwise, all samples found within
        an interval will be combined, weighted by the number of matching radii.

    Returns
    -------
    radii : 1D array
        Radii at which sampling was carried out.
    sampled : 1D array
        Sampled values along the concentric circles.

    Raises
    ------
    ValueError
        If `normalisation` is not in {'multiply', 'none'}.
    ValueError
        If a value is given for both `bin_samples` and `bin_width`.

    """
    if not normalisation in ("multiply", "none"):
        raise ValueError("normalisation can be one of {'divide', 'none'}.")

    if bin_samples is not None and bin_width is not None:
        raise ValueError("Only one of 'bin_samples' and 'bin_width' may be given.")

    coords = np.array(np.meshgrid(x_centres, y_centres, indexing="ij"))
    grid_radii = np.linalg.norm(coords, axis=0)
    unique_radii = np.unique(grid_radii)

    sampled = np.empty(unique_radii.size)
    n_samples = np.empty(unique_radii.size, dtype=np.int64)

    total = 0
    for i, radius in enumerate(unique_radii):
        # Alternative normalisation: do not multiply by the radius.
        matched = np.isclose(grid_radii, radius, rtol=0, atol=1e-19)

        n_matched = np.sum(matched)
        n_samples[i] = n_matched
        total += n_matched

        if normalisation == "none":
            sampled[i] = np.average(data[matched])
        else:
            sampled[i] = np.average(data[matched]) * radius

    assert total == grid_radii.size, "Each bin should only be assigned a value once."

    # Now combine bins if requested.
    if isinstance(bin_samples, Real) and not isinstance(bin_samples, Integral):
        # Use the given fraction.
        bin_samples = ceil(bin_samples * data.size)

    if isinstance(bin_samples, Integral):
        # Use the given number of samples per bin.

        # Keep track of combined radii and values.
        combined_radii = []
        combined = []

        # Aggregation of individual bins until the desired number of samples per new
        # bin has been reached.
        agg_vals = []
        agg_rads = []
        agg_samples = []
        n_agg = prev_n = 0

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

                # The bins are averaged, weighted by the number of samples (number of
                # matched cells based on their radii) - this makes intuitive sense,
                # especially for asymmetrical distributions where the pdf is not
                # circularly symmetric, and has been empirically verified to work
                # well against the known analytical distribution.
                combined_radii.append(np.average(agg_rads, weights=agg_samples))
                combined.append(np.average(agg_vals, weights=agg_samples))

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

    if bin_width is not None:
        # Keep track of combined radii and values.
        combined_radii = []
        combined = []

        bin_edges = np.arange(0, np.max(unique_radii) + bin_width, bin_width)
        for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
            sel = (lower < unique_radii) & (unique_radii < upper)
            if not np.any(sel):
                continue

            combined_radii.append(np.average(unique_radii[sel], weights=n_samples[sel]))
            combined.append(np.average(sampled[sel], weights=n_samples[sel]))

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
    ValueError
        If `radii` are not sorted in ascending order and unique.
    ValueError
        If `normalisation` is not in {'divide', 'none'}.
    RuntimeError
        If a bin was assigned a value multiple times. This should never happen as it
        invalidates the output data.

    """
    unique_radii = np.unique(radii)  # Also sorted in ascending order.
    if not np.allclose(radii, unique_radii) or unique_radii.shape[0] == 1:
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

        # Ensure these radii can never be matched again.
        grid_radii[matched] = -1

    if total != grid_radii.size:
        raise RuntimeError("Each bin should only be assigned a value once.")

    return data
