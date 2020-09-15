# -*- coding: utf-8 -*-
"""Averaging of a 2D distribution to a 1D 'slice'."""
import cv2
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm

from bounded_rand_walkers.utils import get_centres


def rotation(x, y, angle):
    """Rotate x, y position by angle theta."""

    x_rot = x * np.cos(angle) - y * np.sin(angle)
    y_rot = x * np.sin(angle) + y * np.cos(angle)

    return x_rot, y_rot


def radial_interp(
    data, xcentre, ycentre, num_radii, num_points_per_radius, dtype="float"
):
    """Radial interpolation from 2D distribution."""
    if dtype == "float":
        data_copy = np.zeros(data.shape, dtype=np.int32)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if data[row, col] > 1e-6:
                    data_copy[row, col] = 1
    elif dtype == "int32":
        data_copy = data.copy()
    else:
        raise Exception("Error: not correct integer type of input data")

    # generate mask
    filled_array = np.zeros((data_copy.shape[0] + 2, data_copy.shape[1] + 2), np.uint8)
    cv2.floodFill(data_copy, filled_array, (0, 0), newVal=255)

    if np.all(data_copy == 255):
        mask = np.ones((ycentre.size, xcentre.size), dtype=bool)
    else:
        mask = np.zeros((ycentre.size, xcentre.size), dtype=bool)
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                if data_copy[row, col] != 255:
                    mask[row, col] = True
                else:
                    mask[row, col] = False

    data_array = data[np.where(mask)]

    pos = np.zeros((data_array.size, 2)) - 9

    # create x and y arrays
    pos_array = np.zeros((data.shape[0], data.shape[1], 2)) - 9
    for col in range(data.shape[1]):
        pos_array[:, col, 0] = xcentre
    for row in range(data.shape[0]):
        pos_array[row, :, 1] = ycentre

    pos[:, 0] = pos_array[mask, 0]  # x positions
    pos[:, 1] = pos_array[mask, 1]  # y positions

    max_x = np.max(np.abs(pos[:, 0]))
    max_y = np.max(np.abs(pos[:, 1]))
    max_rad = np.sqrt(max_x ** 2 + max_y ** 2)

    # interpolate on grid encompassing these points
    points = []
    radii = np.zeros(num_radii)
    index = 0
    for r in np.linspace(0, max_rad, num_radii):
        x = r
        y = 0.0
        radii[index] = r
        for angle in np.arange(0, 2 * np.pi, 2 * np.pi / num_points_per_radius):
            # rotate around z axis
            x, y = rotation(x, y, angle)
            points.append([x, y])
        index += 1

    interp_points = griddata(pos, data_array, points, fill_value=-9.0)

    # average number of points at same radius
    avg = np.zeros(num_radii) - 9
    start = 0
    for i in range(num_radii):
        values = interp_points[
            start * num_points_per_radius : (start + 1) * num_points_per_radius
        ]
        index = np.where(values != -9.0)[0]
        if index.size != 0:
            avg[i] = np.average(values[index])
        else:
            avg[i] = 0.0
        start += 1

    # normalising averages weighted by area
    total = np.sum(avg) * max_rad / float(num_radii)
    avg /= total

    return avg, radii


def radial_interp_circ(data, num_radii, dtype="float", verbose=True):
    """Radial interpolation from 2D distribution.

    Averaging is carried out using concentric circles and unbinned data.

    If the number of data points is large, this function may take very long to
    execute.

    TODO: Compare speed to straight binning of distances, which is equivalent.

    """
    min_coord = np.min(data)
    max_coord = np.max(data)

    max_abs = max(np.abs(min_coord), np.abs(max_coord))

    radii_edges = np.linspace(0, max_abs, num_radii + 1)
    radii = get_centres(radii_edges)

    avg = np.zeros_like(radii)

    # Calculate distances from the origin.
    distances = np.linalg.norm(data, axis=1)

    for (i, (mean_rad, rad_u)) in enumerate(
        zip(
            tqdm(
                radii,
                desc="Averaging over radii",
                disable=not verbose,
                smoothing=0,
            ),
            radii_edges[1:],
        )
    ):
        # Selected all samples that are between the two concentric circles.
        # Since we are using concentric circles centred at the origin, we can simply
        # test the number of samples that are within the outer circle, removing these
        # elements from the array before the next iteration.
        selection = distances < rad_u
        avg[i] = np.sum(selection)

        # Discard these distances for the next iteration.
        distances = distances[~selection]

    return avg, radii
