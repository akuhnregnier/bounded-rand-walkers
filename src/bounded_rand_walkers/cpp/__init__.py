# -*- coding: utf-8 -*-
"""Compiled C++ code that interfaces with Python.

Mainly, the code speeds up random walker data generation.

"""
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from _bounded_rand_walkers_cpp import *


def get_centres(x):
    return (x[1:] + x[:-1]) / 2.0


def test_1d(
    pdf_name="gauss",
    bins=200,
    xlim=(-0.5, 0.5),
    N=1000000,
    n_sample=10,
    blocks=100,
    **params
):
    pdf = globals()[pdf_name]
    assert callable(pdf)

    x_edges = np.linspace(*xlim, bins)
    x_centres = get_centres(x_edges)

    out = np.histogram(
        testing_1d(start_pos=0.5, pdf_name=pdf_name, blocks=blocks, N=N, **params),
        bins=x_edges,
        density=True,
    )[0]
    exp = np.zeros(bins - 1)

    # params['width'] *= 1

    for (i, (x_0, x_1)) in enumerate(
        zip(tqdm(x_edges[:-1], desc="x bins"), x_edges[1:])
    ):
        if n_sample > 1:
            sum_val = 0
            for p_x in get_centres(np.linspace(x_0, x_1, n_sample + 1)):
                sum_val += pdf([p_x], **params)
            exp[i] = sum_val / n_sample
        else:
            exp[i] = pdf([(x_0 + x_1) / 2.0], **params)

    # Normalise `exp`.
    # sum(bin_area * frequency) = 1
    exp /= np.sum(exp) * np.diff(x_edges)[0]

    fig, axes = plt.subplots(2, 1, sharex=True)
    axes[0].plot(x_centres, out, label="Sampled")
    axes[0].plot(x_centres, exp, label="Analytical", linestyle="--")
    axes[0].legend(loc="best")
    axes[1].plot(x_centres, out - exp, label="Sampled - Analytical")
    axes[1].legend(loc="best")


def test_2d(
    pdf_name="gauss",
    bins=200,
    xlim=(-1, 1),
    ylim=(-1, 1),
    N=100000,
    n_sample=4,
    blocks=100,
    **params
):
    pdf = globals()[pdf_name]
    assert callable(pdf)

    x_edges = np.linspace(*xlim, bins)
    y_edges = np.linspace(*ylim, bins)

    out = np.histogram2d(
        *testing_2d(pdf_name=pdf_name, blocks=blocks, N=N, **params).T,
        bins=[x_edges, y_edges],
        density=True
    )[0]
    exp = np.zeros((bins - 1, bins - 1))

    for (i, (x_0, x_1)) in enumerate(
        zip(tqdm(x_edges[:-1], desc="x bins"), x_edges[1:])
    ):
        for (j, (y_0, y_1)) in enumerate(zip(y_edges[:-1], y_edges[1:])):
            sum_val = 0
            for p_x, p_y in product(
                get_centres(np.linspace(x_0, x_1, n_sample + 1)),
                get_centres(np.linspace(y_0, y_1, n_sample + 1)),
            ):
                sum_val += pdf([p_x, p_y], **params)
            exp[i, j] = sum_val / (n_sample ** 2.0)

    # Normalise `exp`.
    # sum(bin_area * frequency) = 1
    exp /= np.sum(exp) * np.diff(x_edges)[0] * np.diff(y_edges)[0]

    cbar_kwargs = dict(shrink=0.5)

    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(11, 6))
    im0 = axes[0].pcolormesh(x_edges, y_edges, out)
    axes[0].set_title("Sampler Output")
    fig.colorbar(im0, ax=axes[0], **cbar_kwargs)

    im1 = axes[1].pcolormesh(x_edges, y_edges, exp)
    axes[1].set_title("Analytical (bin centres)")
    fig.colorbar(im1, ax=axes[1], **cbar_kwargs)

    im2 = axes[2].pcolormesh(x_edges, y_edges, out - exp)
    axes[2].set_title("Sampler - Analytical")
    fig.colorbar(im2, ax=axes[2], **cbar_kwargs)

    for ax in axes:
        ax.axis("scaled")

    plt.figure()
    plt.pcolormesh(x_edges, y_edges, out - exp)
    plt.colorbar()
    plt.axis("scaled")
