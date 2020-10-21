# -*- coding: utf-8 -*-
"""Testing of numerical pdf samplers."""
from itertools import product
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm, trange

from . import cpp
from .rejection_sampling import Sampler
from .utils import get_centres


def test_1d(
    cpp_pdf_name="gauss",
    python_pdf=None,
    bins=200,
    xlim=(-0.5, 0.5),
    N=1000000,
    n_sample=10,
    blocks=100,
    seed=-1,
    verbose=True,
    axes=None,
    **params,
):
    """1D testing of numerical samplers.

    Results are compared using comparative plots.

    Parameters
    ----------
    cpp_pdf_name : str
        Name of the C++ step-size pdf.
    python_pdf : callable or None
        If a callable is given, this pdf will be sampled using the native Python
        sampler and compared in addition to the C++ sampler.
    bins : int
        Number of bins used to visualise the results.
    xlim : tuple of float
        Maximum and minimum step sizes along the x-axis to bin within.
    N : int
        Number of samples.
    n_sample : int
        Number of samples averaged for every analytical bin to improve its discrete
        representation.
    blocks : int or None
        Number of blocks (along each dimension) to use in the adaptive sampling
        algorithm to speed up sampling. If None, this will be set to 100 or 2 for 1D
        or 2D sampling, respectively. Should equal 2 for 2D sampling.
    seed : int
        Random number generator seed. By default, the seed is derived using the
        current time and thus runs will not be repeatable. To yield repeatable runs, a
        non-negative int must be given.
    verbose : bool
        If True, display a progress bar for Python operations (sampling and binning).
    axes : 2-iterable of matplotlib Axes
        Axes to plot onto. If None, a new Figure and new Axes will be created.

    C++ PDF parameters
    ------------------
    centre : array-like
    width : float
    decay_rate : float
    exponent : float
    binsize : float

    Raises
    ------
    ValueError
        If the C++ pdf is not found, or not a callable.

    """
    if not hasattr(cpp, cpp_pdf_name) or not callable(getattr(cpp, cpp_pdf_name)):
        raise ValueError(
            f"The C++ pdf '{cpp_pdf_name}' could not be found or was not a callable."
        )

    pdf = getattr(cpp, cpp_pdf_name)

    x_edges = np.linspace(*xlim, bins + 1)
    x_centres = get_centres(x_edges)

    # C++ pdf sampling and binning.
    out = np.histogram(
        cpp.testing_1d(
            start_pos=0.5,
            pdf_name=cpp_pdf_name,
            blocks=blocks,
            N=N,
            seed=seed,
            **params,
        ),
        bins=x_edges,
        density=True,
    )[0]

    # Python pdf sampling.
    if python_pdf is not None:
        py_sampler = Sampler(python_pdf, dimensions=1, blocks=blocks, seed=seed)
        py_samples = np.empty((N,))
        py_pos = np.array([0.5])
        for i in trange(N, desc="Python sampling", disable=not verbose):
            py_samples[i] = py_sampler.sample(position=py_pos)
        py_out = np.histogram(py_samples, bins=x_edges, density=True)[0]

    # Generation of the expected pdf.
    exp = np.zeros(bins)
    for (i, (x_0, x_1)) in enumerate(
        zip(tqdm(x_edges[:-1], desc="Expected pdf", disable=not verbose), x_edges[1:])
    ):
        sum_val = 0
        for p_x in get_centres(np.linspace(x_0, x_1, n_sample + 1)):
            sum_val += pdf([p_x], **params)
        exp[i] = sum_val / n_sample

    # Normalise `exp`.
    exp /= np.sum(exp) * np.diff(x_edges)[0]

    # Plot the comparisons.
    if axes is None:
        fig, axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(x_centres, out, label="Sampled (C++)", c="C0", zorder=2)
    axes[0].plot(x_centres, exp, label="Analytical", linestyle="--", c="C2", zorder=3)
    axes[1].plot(x_centres, out - exp, label="Sampled (C++) - Analytical", zorder=2)

    if python_pdf is not None:
        axes[0].plot(x_centres, py_out, label="Sampled (Py)", c="C1", zorder=1)
        axes[1].plot(
            x_centres, py_out - exp, label="Sampled (Py) - Analytical", c="C1", zorder=1
        )

    for ax in axes:
        ax.legend()


def test_2d(
    cpp_pdf_name="gauss",
    python_pdf=None,
    bins=200,
    xlim=(-1, 1),
    ylim=(-1, 1),
    N=1000000,
    n_sample=4,
    blocks=2,
    seed=-1,
    verbose=True,
    fig=None,
    axes=None,
    **params,
):
    """2D testing of numerical samplers.

    Results are compared using comparative plots.

    Parameters
    ----------
    cpp_pdf_name : str
        Name of the C++ step-size pdf.
    python_pdf : callable or None
        If a callable is given, this pdf will be sampled using the native Python
        sampler and compared in addition to the C++ sampler.
    bins : int
        Number of bins used to visualise the results.
    xlim : tuple of float
        Maximum and minimum step sizes along the x-axis to bin within.
    ylim : tuple of float
        Maximum and minimum step sizes along the y-axis to bin within.
    N : int
        Number of samples.
    n_sample : int
        Number of samples averaged for every analytical bin to improve its discrete
        representation, along each axis (i.e. `n_samples` ** 2 samples are taken for
        every grid cell).
    blocks : int or None
        Number of blocks (along each dimension) to use in the adaptive sampling
        algorithm to speed up sampling. If None, this will be set to 100 or 2 for 1D
        or 2D sampling, respectively. Should equal 2 for 2D sampling.
    seed : int
        Random number generator seed. By default, the seed is derived using the
        current time and thus runs will not be repeatable. To yield repeatable runs, a
        non-negative int must be given.
    verbose : bool
        If True, display a progress bar for Python operations (sampling and binning).
    fig : matplotlib Figure
        Figure to plot onto. Must be given in combination with `axes`.
    axes : 2-iterable of matplotlib Axes
        Axes to plot onto. If None, a new Figure and new Axes will be created.

    C++ PDF parameters
    ------------------
    centre : array-like
    width : float
    decay_rate : float
    exponent : float
    binsize : float

    Raises
    ------
    ValueError
        If the C++ pdf is not found, or not a callable.

    """
    if not hasattr(cpp, cpp_pdf_name) or not callable(getattr(cpp, cpp_pdf_name)):
        raise ValueError(
            f"The C++ pdf '{cpp_pdf_name}' could not be found or was not a callable."
        )

    if blocks > 2:
        warn(f"'blocks' > 2 encountered ({blocks}). Sampling may be inaccurate.")

    pdf = getattr(cpp, cpp_pdf_name)

    x_edges = np.linspace(*xlim, bins + 1)
    y_edges = np.linspace(*ylim, bins + 1)

    # C++ pdf sampling and binning.
    out = np.histogram2d(
        *cpp.testing_2d(
            pdf_name=cpp_pdf_name, blocks=blocks, N=N, seed=seed, **params
        ).T,
        bins=[x_edges, y_edges],
        density=True,
    )[0]

    plot_data = [out]

    # Python pdf sampling.
    if python_pdf is not None:
        py_sampler = Sampler(python_pdf, dimensions=2, blocks=blocks, seed=seed)
        py_samples = np.empty((N, 2))
        py_pos = np.array([0.0, 0.0])
        for i in trange(N, desc="Python sampling", disable=not verbose):
            py_samples[i] = py_sampler.sample(py_pos).ravel()
        py_out = np.histogram2d(*py_samples.T, bins=[x_edges, y_edges], density=True)[0]
        plot_data.append(py_out)

    # Generation of the expected pdf.
    exp = np.zeros((bins, bins))
    for (i, (x_0, x_1)) in enumerate(
        zip(tqdm(x_edges[:-1], desc="Expected pdf", disable=not verbose), x_edges[1:])
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
    exp /= np.sum(exp) * np.diff(x_edges)[0] * np.diff(y_edges)[0]

    cbar_kwargs = dict(shrink=0.5)

    # Plot the comparisons.
    if fig is None or axes is None:
        fig, axes = plt.subplots(
            1 if python_pdf is None else 2,
            3,
            sharex=True,
            sharey=True,
            figsize=(11, 3 * (1 if python_pdf is None else 2)),
            squeeze=False,
        )

    for (row, (out, title)) in enumerate(zip(plot_data, ("C++", "Py"))):
        im0 = axes[row, 0].pcolormesh(x_edges, y_edges, out)
        axes[row, 0].set_title(f"Sampler ({title}) Output")
        fig.colorbar(im0, ax=axes[row, 0], **cbar_kwargs)

        im1 = axes[row, 1].pcolormesh(x_edges, y_edges, exp)
        axes[row, 1].set_title("Analytical (bin centres)")
        fig.colorbar(im1, ax=axes[row, 1], **cbar_kwargs)

        im2 = axes[row, 2].pcolormesh(x_edges, y_edges, out - exp)
        axes[row, 2].set_title(f"Sampler ({title}) - Analytical")
        fig.colorbar(im2, ax=axes[row, 2], **cbar_kwargs)

    for ax in np.asarray(axes).flatten():
        ax.axis("scaled")
