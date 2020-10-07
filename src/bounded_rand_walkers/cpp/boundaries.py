# -*- coding: utf-8 -*-
"""Generate boundary vertices.

Vertices are returned in a clockwise order with x coords in the first and y coords in
the second column. The last returned vertex will always equal the first.

"""

import numpy as np


def square(centre=(0, 0), side=1):
    """Square boundary.

    Parameters
    ----------
    centre : array-like
        Centre coordinates.
    side : float
        Side length.

    Returns
    -------
    vertices : array
        (5, 2) array containing the boundary vertices arranged in a clockwise order.
        The last vertex matches the first.

    """
    side = np.abs(side)

    xlim = np.array([-side / 2, side / 2]) + centre[0]
    ylim = np.array([-side / 2, side / 2]) + centre[1]

    return np.array(
        [
            [xlim[0], ylim[0]],
            [xlim[0], ylim[1]],
            [xlim[1], ylim[1]],
            [xlim[1], ylim[0]],
            [xlim[0], ylim[0]],
        ]
    )


def triangle(centre=(0, 0), side=2):
    """Equilateral triangular boundary.

    Parameters
    ----------
    centre : array-like
        Centre coordinates.
    side : float
        Side length.

    Returns
    -------
    vertices : array
        (4, 2) array containing the boundary vertices arranged in a clockwise order.
        The last vertex matches the first.

    """
    bottom_y = centre[1] - (side / (2 * 3 ** 0.5))
    top_y = centre[1] + side / 3 ** 0.5

    return np.array(
        [
            [centre[0] - side / 2, bottom_y],
            [centre[0], top_y],
            [centre[0] + side / 2, bottom_y],
            [centre[0] - side / 2, bottom_y],
        ]
    )


def circle(centre=(0, 0), radius=1.0, N=30):
    """Circular boundary.

    Parameters
    ----------
    centre : array-like
        Centre coordinates.
    radius : float
        Circle radius.
    N : int
        Number of boundary points.

    Returns
    -------
    vertices : array
        (N + 1, 2) array containing the boundary vertices arranged in a clockwise
        order with the last vertex matching the first.

    """
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    xs = radius * np.cos(angles)
    ys = radius * np.sin(angles)

    vertices = np.hstack((xs[:, None], ys[:, None]))
    vertices = np.vstack((vertices, vertices[:1]))

    return vertices


def weird():
    return np.array([[0.1, 0.3], [0.25, 0.98], [0.9, 0.9], [0.7, 0.4], [0.4, 0.05]])


bound_map = {"square": square, "triangle": triangle, "circle": circle, "weird": weird}
