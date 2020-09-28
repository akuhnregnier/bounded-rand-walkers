# -*- coding: utf-8 -*-
"""Compiled C++ code that interfaces with Python.

Mainly, the code speeds up random walker data generation.

"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from numbers import Number
from pathlib import Path
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from joblib import Memory
from tqdm.auto import tqdm

from _bounded_rand_walkers_cpp import *

from ..utils import cache_dir, get_centres
from .boundaries import *

# Rename the original function since we will be defining our own version here.
cpp_generate_data = generate_data


memory = Memory(cache_dir, verbose=0)


def get_cached_filename(
    cache_dir=None,
    samples=1000,
    pdf_name="funky",
    bound_name="square",
    centre=(0, 0),
    width=0.5,
    decay_rate=1,
    exponent=1,
    binsize=0.1,
    blocks=50,
    seed=-1,
    squeeze=True,
):
    """Get filename for cached random walker data.

    Parameters
    ----------
    cache_dir : pathlib.Path, str, or None
        Cache directory. By default, no caching is done. Caching also requires a valid
        seed.
    samples : int
        Number of random walk steps.
    pdf_name : str
        Name of the step-size pdf.
    bound_name : str
        Name of the boundary (for 2D).
    seed : int or array-like
        Random number generator seed: a non-negative int. If multiple seeds are given,
        multiple filenames will be returned.
    squeeze : bool
        If true, return a single filename if a single `seed` is given. Otherwise,
        always return a tuple of filenames.

    Returns
    -------
    filename : pathlib.Path or tuple of pathlib.Path
        Cache filename(s).

    PDF parameters
    --------------
    centre : array-like
    width : float
    decay_rate : float
    exponent : float
    binsize : float

    """
    if isinstance(seed, Number):
        seeds = np.array([seed])
    else:
        seeds = np.asarray(seed)

    if cache_dir is None:
        if squeeze and seeds.shape[0] == 1:
            return None
        return (None,) * seeds.shape[0]

    # Conditions satisfied for caching.
    filenames = [
        (
            Path(cache_dir)
            / (
                (
                    str(samples)
                    + pdf_name
                    + bound_name
                    + f"{centre[0]:0.2f}_{centre[1]:0.2f}"
                    + f"{width:0.2f}"
                    + f"{decay_rate:0.2f}"
                    + f"{exponent:0.2f}"
                    + f"{binsize:0.2f}"
                    + str(blocks)
                    + str(seed)
                ).replace(".", "_")
                + ".npz"
            )
        )
        for seed in seeds
    ]

    if squeeze and len(filenames) == 1:
        return filenames[0]
    return tuple(filenames)


def generate_data(
    samples=1000,
    pdf_name="funky",
    bound_name="square",
    centre=(0, 0),
    width=0.5,
    decay_rate=1,
    exponent=1,
    binsize=0.1,
    blocks=50,
    seed=-1,
    cache_dir=None,
    max_workers=1,
    squeeze=True,
    cache_only=False,
    verbose=True,
):
    """Generate random walker data.

    Parameters
    ----------
    samples : int
        Number of random walk steps.
    pdf_name : str
        Name of the step-size pdf.
    bound_name : str
        Name of the boundary (for 2D).
    blocks : int
        Number of blocks (along each dimension) to use in the adaptive sampling
        algorithm to speed up sampling.
    seed : int or array-like
        Random number generator seed. By default, the seed is derived using the
        current time and thus runs will not be repeatable. To yield repeatable runs, a
        non-negative int must be given. This also applied if multiple seeds are given,
        where data may be generated in parallel depending on `max_workers`.
    cache_dir : pathlib.Path, str, or None
        Cache directory. By default, no caching is done. Caching also requires a valid
        seed.
    max_workers : int or None
        Number of concurrent data generation tasks to run. If None, all detected cpus
        are used.
    squeeze : bool
        If true and a single `seed` is given, return positions and steps. Otherwise, a
        tuple of tuples of positions and steps will be returned.
    cache_only : bool
        If true, do not return data and only cache it (depending on `cache_dir`).
    verbose : bool
        If true, output progress information.

    Returns
    -------
    data : 2-tuple of array or tuple of 2-tuple of array
        If squeeze and a single `seed` is given, two arrays are returned. The first is
        a (`samples + 1`, 2) array containing the random walker positions. The second
        is a (`samples`, 2) array containing the random walker steps. Otherwise, a
        tuple of such arrays is returned depending on the number of seeds given.

    PDF parameters
    --------------
    centre : array-like
    width : float
    decay_rate : float
    exponent : float
    binsize : float

    """
    if isinstance(seed, Number):
        seeds = np.array([seed])
    else:
        seeds = np.asarray(seed)

    # Deal with multiple seeds.
    if seeds.shape[0] > 1:
        # Concurrent processing.
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            fs = [
                executor.submit(
                    generate_data,
                    samples=samples,
                    pdf_name=pdf_name,
                    bound_name=bound_name,
                    centre=centre,
                    width=width,
                    decay_rate=decay_rate,
                    exponent=exponent,
                    binsize=binsize,
                    blocks=blocks,
                    seed=seed,
                    cache_dir=cache_dir,
                    max_workers=1,
                    squeeze=False,
                    cache_only=cache_only,
                )
                for seed in seeds
            ]

            results = []
            for f in tqdm(
                as_completed(fs),
                total=len(fs),
                desc="Calculating data",
                smoothing=0,
                disable=not verbose,
            ):
                results.extend(f.result())
        return results

    # Since there is only a single seed.
    seed = seeds[0]

    if cache_dir is not None and seed >= 0:
        # Cached data may be available.
        filename = get_cached_filename(
            cache_dir,
            samples,
            pdf_name,
            bound_name,
            centre,
            width,
            decay_rate,
            exponent,
            binsize,
            blocks,
            seed,
        )
    else:
        # No caching is possible/requested.
        filename = None

    if filename is not None and filename.is_file() and not cache_only:
        # Cached data exists.
        saved = np.load(filename)
        positions = saved["positions"]
        steps = saved["steps"]
        saved.close()
    else:
        # Generate new data.
        positions, steps = cpp_generate_data(
            bounds_vertices=bound_map[bound_name](),
            samples=samples,
            pdf_name=pdf_name,
            centre=centre,
            width=width,
            decay_rate=decay_rate,
            exponent=exponent,
            binsize=binsize,
            blocks=blocks,
            seed=seed,
        )
        if filename is not None:
            # Save the data.
            np.savez_compressed(filename, positions=positions, steps=steps)
    if squeeze:
        if cache_only:
            return None
        return positions, steps
    if cache_only:
        return (None,)
    return ((positions, steps),)


def get_binned_2D(
    filenames,
    g_x_edges,
    g_y_edges,
    f_t_x_edges,
    f_t_y_edges,
    f_t_r_edges,
    verbose=True,
    ignore_initial=1000,
):
    """Bin data from `generate_data`.

    The data is binned iteratively and finally normalised.

    Parameters
    ----------
    filenames : iterable of pathlib.Path
        Filenames as generated by `get_cached_filename`.
    g_x_edges : array
        Position distribution x bin edges.
    g_y_edges : array
        Position distribution y bin edges.
    f_t_x_edges : array
        Step size distribution x bin edges.
    f_t_y_edges : array
        Step size distribution y bin edges.
    f_t_r_edges : array
        Step size distribution radial bin edges.
    verbose : bool
        If true, output progress information.
    ignore_initial : int
        Number of initial samples to ignore.

    Returns
    -------
    g_numerical : array
        Binned positions distribution.
    f_t_numerical : array
        Binned (transformed) step size distribution.
    f_t_r_numerical : array
        Radially binned (transformed) step size distribution.

    """
    # Use the cell areas to normalise the data later
    g_cell_area = np.abs((g_x_edges[1] - g_x_edges[0]) * (g_y_edges[1] - g_y_edges[0]))
    f_t_cell_area = np.abs(
        (f_t_x_edges[1] - f_t_x_edges[0]) * (f_t_y_edges[1] - f_t_y_edges[0])
    )
    f_t_r_length = np.abs(f_t_r_edges[1] - f_t_r_edges[0])

    g_numerical = np.zeros((g_x_edges.size - 1, g_y_edges.size - 1), dtype=np.float64)
    f_t_numerical = np.zeros(
        (f_t_x_edges.size - 1, f_t_y_edges.size - 1), dtype=np.float64
    )
    f_t_r_numerical = np.zeros(f_t_r_edges.size - 1, dtype=np.float64)

    for filename in tqdm(
        filenames, desc="Binning data", smoothing=0, disable=not verbose
    ):
        if filename is not None and filename.is_file():
            # Cached data exists.
            saved = np.load(filename)
            positions = saved["positions"][ignore_initial:]
            steps = saved["steps"][ignore_initial:]
            saved.close()
        else:
            warn(f"Filename '{filename}' was not found.")
            continue

        # Bin the position data.
        g_numerical += np.histogram2d(
            *positions.T, bins=[g_x_edges, g_y_edges], normed=False
        )[0]

        # Bin the step size data.
        f_t_numerical += np.histogram2d(
            *steps.T, bins=[f_t_x_edges, f_t_y_edges], normed=False
        )[0]

        # Radially bin the step size data.
        radial_lengths = np.linalg.norm(steps, axis=1)
        f_t_r_numerical += np.histogram(radial_lengths, bins=f_t_r_edges)[0]

    # Normalise the binned data generated above.
    g_numerical /= np.sum(g_numerical * g_cell_area)
    f_t_numerical /= np.sum(f_t_numerical * f_t_cell_area)
    f_t_r_numerical /= np.sum(f_t_r_numerical * f_t_r_length)

    return g_numerical, f_t_numerical, f_t_r_numerical


@memory.cache
def get_binned_data(
    filenames, n_bins, max_step_length=8 ** 0.5, g_bounds=(-2, 2), f_bounds=(-2, 2)
):
    """Get binned data.

    Parameters
    ----------
    filenames : iterable of pathlib.Path
        Filenames to load data from.
    n_bins : int
        Number of bins (per axis).
    max_step_length : float
        Maximum expected step length.
    g_bounds : tuple of float
        Position bounds.
    f_bounds : tuple of float
        Step size bounds.

    Returns
    -------
    g_x_edges : 1D array
        Position x-axis bin edges.
    g_y_edges : 1D array
        Position y-axis bin edges.
    g_x_centres : 1D array
        Position x-axis bin centres.
    g_y_centres : 1D array
        Position y-axis bin centres.
    f_t_x_edges : 1D array
        Step size distribution x-axis bin edges.
    f_t_y_edges : 1D array
        Step size distribution y-axis bin edges.
    f_t_x_centres : 1D array
        Step size distribution x-axis bin centres.
    f_t_y_centres : 1D array
        Step size distribution y-axis bin centres.
    f_t_r_edges : 1D array
        Step size distribution radial bin edges.
    f_t_r_centres : 1D array
        Step size distribution radial bin centres.
    g_numerical : 2D array
        Binned positions.
    f_t_numerical : 2D array
        Binned step sizes.
    f_t_r_numerical : 1D array
        Radially binned step sizes.

    """
    # Position binning.
    g_x_edges = g_y_edges = np.linspace(*g_bounds, n_bins + 1)
    g_x_centres = g_y_centres = get_centres(g_x_edges)
    # Step size binning (transformed).
    f_t_x_edges = f_t_y_edges = np.linspace(*f_bounds, n_bins + 1)
    f_t_x_centres = f_t_y_centres = get_centres(f_t_x_edges)
    # Step size binning (transformed) - radially.
    f_t_r_edges = np.linspace(0, max_step_length, n_bins + 1)
    f_t_r_centres = get_centres(f_t_r_edges)

    g_numerical, f_t_numerical, f_t_r_numerical = get_binned_2D(
        filenames, g_x_edges, g_y_edges, f_t_x_edges, f_t_y_edges, f_t_r_edges
    )
    return (
        g_x_edges,
        g_y_edges,
        g_x_centres,
        g_y_centres,
        f_t_x_edges,
        f_t_y_edges,
        f_t_x_centres,
        f_t_y_centres,
        f_t_r_edges,
        f_t_r_centres,
        g_numerical,
        f_t_numerical,
        f_t_r_numerical,
    )


def test_1d(
    pdf_name="gauss",
    bins=200,
    xlim=(-0.5, 0.5),
    N=1000000,
    n_sample=10,
    blocks=100,
    **params,
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
    **params,
):
    pdf = globals()[pdf_name]
    assert callable(pdf)

    x_edges = np.linspace(*xlim, bins)
    y_edges = np.linspace(*ylim, bins)

    out = np.histogram2d(
        *testing_2d(pdf_name=pdf_name, blocks=blocks, N=N, **params).T,
        bins=[x_edges, y_edges],
        density=True,
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
