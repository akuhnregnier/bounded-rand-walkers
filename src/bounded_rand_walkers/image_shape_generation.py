#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import Delaunay

if __name__ == "__main__":
    image = Image.open("shape1.jpg")
    data = np.asarray(image)
    data = np.sum(data, axis=2)

    mask = (10 < data) & (data < 80)
    xs = np.arange(data.shape[0])
    ys = np.arange(data.shape[1])

    x_mask = np.where((xs > 800) & (xs < 2600))
    y_mask = np.where((ys > 880) & (ys < 3300))

    d2 = np.ma.array(data.copy(), mask=np.zeros_like(data))
    d2.mask[~mask] = True

    d2 = d2[x_mask]
    d2 = d2.T[y_mask].T

    d2[~d2.mask] = 1
    d2[d2.mask] = 0

    a = 4
    b = 2
    d2 = ndimage.morphology.binary_erosion(
        d2,
        np.ones((b, b)),
        iterations=1,
    ).astype(d2.dtype)
    d2 = ndimage.morphology.binary_dilation(
        d2,
        np.ones((a, a)),
        iterations=1,
    ).astype(d2.dtype)

    plt.figure()
    plt.imshow(d2)

    points = 100

    xy_indices = np.array(np.where(d2.T)).T

    chosen_points = xy_indices[np.random.choice(xy_indices.shape[0], points)]

    x_indices = chosen_points[:, 0].astype(np.float64)
    y_indices = chosen_points[:, 1].astype(np.float64)

    x_range = np.max(x_indices) - np.min(x_indices)
    y_range = np.max(y_indices) - np.min(y_indices)

    max_range = np.max([x_range, y_range])

    x_indices -= np.min(x_indices)
    y_indices -= np.min(y_indices)

    x_indices /= max_range
    y_indices /= max_range

    y_indices *= -1
    y_indices -= np.min(y_indices)

    points = np.array([x_indices, y_indices]).T
    tri = Delaunay(points)

    plt.figure()
    plt.triplot(points[:, 0], points[:, 1], tri.simplices.copy())
    plt.plot(points[:, 0], points[:, 1], "o")
    plt.gca().set_aspect("equal")
