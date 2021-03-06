#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Simulation of random walks.

From an arbitrary intrinsic step size distribution and arbitrary bounds in 1D or 2D,
generate the modified step size distributions numerically.

"""
import logging
import multiprocessing
from time import time

import numpy as np

from .rejection_sampling import Sampler
from .utils import DelaunayArray, in_bounds


def format_time(time_value):
    """Sensible string formatting of a time variable.

    Output units range from seconds ('s') to minutes ('m') and hours ('h'), depending
    on the magnitude of the input `time_value`.

    Parameters
    ----------
    time_value : float
        The time value in seconds that is to be formatted.

    Returns
    -------
    formatted : tuple of float, str
        The converted time value and its associated units.

    """
    units = "s"
    rounded_time = int(round(time_value))
    nr_digits = len(str(rounded_time))
    if nr_digits > 2:
        # convert to minutes
        units = "m"
        time_value /= 60.0
        rounded_time = int(round(time_value))
        nr_digits = len(str(rounded_time))
        if nr_digits > 2:
            # convert to hours
            units = "h"
            time_value /= 60.0
    return time_value, units


def random_walker(f_i, bounds, steps=int(1e2), sampler=None, blocks=50, seed=None):
    """Generate random walker steps.

    Trace a random walker given the `bounds` and the given intrinsic step size
    distribution `f_i`. If a step would lead outside of the boundaries, reject this
    step and choose another one until the boundaries are satisfied (this effectively
    normalises the pdf given the truncation at the boundaries).

    The walker is started at a random position.

    Parameters
    ----------
    f_i : callable
        The intrinsic step size distribution, given as a pdf (probability distribution
        function). The function should return a probability for each step size `l`,
        where `l` should be between 0 and 1. For a 1D problem, `f_i` should take one
        argument, whereas for a 2D problem, it should take two arguments.
    bounds : array
        The bounds for the random walker, given as an array. The shape of the array
        dictates the dimensionality of the problem. A 1D array containing two values
        represents the lower and upper boundaries (in that order) of the 1D problem.
        For the 2D problem, the boundary is given as a 2D array, where each row
        contains the (x, y) coordinates of a point.
    steps : int
        The number of steps to take. The function will return `steps` number of step
        sizes. Defaults to `int(1e4)`.
    seed : int or None
        Seed for the random number generator. By default, runs are not repeatable.

    Returns
    -------
    steps : array of shape (`steps`,)
        The step sizes obtained by executing the random walker simulation.
    positions : array of shape (`steps` + 1,)
        The positions of the random walker.

    Notes
    -----
    All lengths and bounds are to be given in the range [0, 1] for the 1D case!

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> import numpy as np
    >>> from bounded_rand_walkers.functions import (
    ...     Tophat_1D, Tophat_2D, Gaussian, Power, Exponential
    ... )
    >>> step_values, positions = random_walker(
    ...     f_i=Tophat_1D(),
    ...     bounds=np.array([0, 1]),
    ...     steps=int(1e3),
    ... )
    >>> fig, axes = plt.subplots(1, 2, squeeze=True)  # doctest: +SKIP
    >>> axes[0].hist(step_values, bins='auto')  # doctest: +SKIP
    >>> axes[0].set_title('Step Values')  # doctest: +SKIP
    >>> axes[1].hist(positions, bins='auto')  # doctest: +SKIP
    >>> axes[1].set_title('Positions')  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP
    >>> bounds = np.array([
    ...     [0, 0],
    ...     [0, 1],
    ...     [1, 1],
    ...     [1, 0]]
    ... )
    >>> step_values, positions = random_walker(
    ...     f_i=Tophat_2D(),
    ...     bounds=bounds,
    ...     steps=int(1e3),
    ... )
    >>> fig, axes = plt.subplots(1, 2, squeeze=True)  # doctest: +SKIP
    >>> fig.subplots_adjust(right=0.8)  # doctest: +SKIP
    >>> # bin first time to get the maximum bin counts,
    >>> # which are used below
    >>> steps_bin = axes[0].hexbin(*step_values.T)  # doctest: +SKIP
    >>> positions_bin = axes[1].hexbin(*positions.T)  # doctest: +SKIP
    >>> axes[0].cla()  # doctest: +SKIP
    >>> axes[1].cla()  # doctest: +SKIP
    >>> # use this max value with the hexbin vmax option
    >>> # in order to have the same colour scaling for both
    >>> # hexbin plots, such that the same colorbar may be used
    >>> max_value = np.max([np.max(steps_bin.get_array()),  # doctest: +SKIP
    ...                     np.max(positions_bin.get_array())  # doctest: +SKIP
    ...                     ])  # doctest: +SKIP
    >>> steps_bin = axes[0].hexbin(*step_values.T, vmin=0, vmax=max_value)  # doctest: +SKIP
    >>> steps_bin = axes[0].hexbin(*step_values.T, vmin=0, vmax=max_value)  # doctest: +SKIP
    >>> positions_bin = axes[1].hexbin(*positions.T, vmin=0,  # doctest: +SKIP
    ...                                vmax=max_value)  # doctest: +SKIP
    >>> positions_bin = axes[1].hexbin(*positions.T, vmin=0,  # doctest: +SKIP
    ...                                vmax=max_value)  # doctest: +SKIP
    >>> axes[0].set_title('Step Values')  # doctest: +SKIP
    >>> axes[1].set_title('Positions')  # doctest: +SKIP
    >>> for ax in axes:  # doctest: +SKIP
    >>>     ax.set_aspect('equal')  # doctest: +SKIP
    >>> cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])  # doctest: +SKIP
    >>> fig.colorbar(positions_bin, cax=cbar_ax)  # doctest: +SKIP
    >>> plt.show()  # doctest: +SKIP

    """
    logger = logging.getLogger(__name__)
    if bounds.size == bounds.shape[0]:
        bounds = bounds.reshape(-1, 1)
    else:
        bounds = DelaunayArray(bounds)

    dimensions = bounds.shape[1]
    if sampler is None:
        sampler = Sampler(f_i, dimensions, blocks=blocks, seed=seed)
    positions = np.zeros((steps + 1, dimensions), dtype=np.float64)
    step_values = np.zeros((steps, dimensions), dtype=np.float64)
    # give random initial position
    positions[0] = np.random.uniform(low=0.0, high=1.0, size=dimensions)
    while not in_bounds(positions[0], bounds):
        logger.debug("creating new initial position")
        positions[0] = np.random.uniform(low=0.0, high=1.0, size=dimensions)
    logger.debug("Initial walker position:{:}".format(positions[0]))
    start_time = time()
    for position_index in range(1, steps + 1):
        logger.debug("position index:{:} {:}".format(position_index, steps + 1))
        if ((position_index % ((steps + 1) / 10)) == 0 or (steps < 10)) or (
            position_index == int(4e3)
        ):
            elapsed_time = time() - start_time
            elapsed_time_per_step = elapsed_time / position_index
            remaining_time = (steps - position_index) * elapsed_time_per_step
            elapsed_time, elapsed_time_units = format_time(elapsed_time)
            remaining_time, remaining_time_units = format_time(remaining_time)
            logger.info(
                "Position index:{:.1e} out of {:.1e}".format(position_index, steps + 1)
            )
            logger.info(
                "Time elapsed:{:>4.1f} {:}, remaining:{:0.1f} {:}".format(
                    elapsed_time,
                    elapsed_time_units,
                    remaining_time,
                    remaining_time_units,
                )
            )
        logger.debug("Current position:{:}".format(positions[position_index - 1]))
        step_index = position_index - 1
        found = False
        while not found:
            step = sampler.sample(positions[position_index - 1]).reshape(
                -1,
            )
            next_position = positions[position_index - 1] + step
            if in_bounds(next_position, bounds):
                positions[position_index] = next_position
                step_values[step_index] = step
                found = True
    elapsed_time = time() - start_time
    elapsed_time_per_step = elapsed_time / position_index
    logger.info("time per step:{:.1e} s".format(elapsed_time_per_step))
    return step_values, positions


def multi_random_walker(n_processes, f_i, bounds, steps=int(1e2), blocks=50, seed=None):
    """Generate random walks in multiple processes concurrently.

    If the `n_processes==1`, the `random_walker` function is called in the standard
    way, as if it was called directly.

    Parameters
    ----------
    n_processes : int
        The number of processes to run in parallel. The generated data will be joined
        together at the end.
    **kwargs : random_walker parameters
        For an explanation of the other arguments, see `random_walker`.

    Notes
    -----
    Note that the generated position series have no causal relationship - they are
    completely random, since each random_walker execution has no information about any
    of the other executions run in parallel. Therefore, generating steps from the
    returned positions produces an error at the point where the different datasets are
    joined together. However, this error will approach 0 as `steps` is increased,
    since the ratio of 'contiguous' data to erroneous data (at the boundary of the
    contiguous random walks) increases.

    """
    logger = logging.getLogger(__name__)
    assert n_processes >= 1
    if n_processes == 1:
        # simply execute random_walker while ignoring the multiprocessing
        step_values, positions = random_walker(
            f_i=f_i, bounds=bounds, steps=steps, blocks=blocks
        )
        return step_values, positions

    if bounds.size == bounds.shape[0]:
        bounds = bounds.reshape(-1, 1)
    dimensions = bounds.shape[1]

    sampler = Sampler(f_i, dimensions, blocks=blocks, seed=seed)

    def rand_walk_worker(procnum, return_dict):
        """Worker function which executes the random_walker"""
        logger.info(f"New process: {procnum}.")
        step_values, positions = random_walker(
            f_i=f_i, bounds=bounds, steps=int(steps / n_processes), sampler=sampler
        )
        return_dict[procnum] = (step_values, positions)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(n_processes):
        np.random.seed(i)
        p = multiprocessing.Process(target=rand_walk_worker, args=(i, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    step_values = [i[0] for i in list(return_dict.values())]
    positions = [i[1] for i in list(return_dict.values())]
    step_values = np.vstack(step_values)
    positions = np.vstack(positions)

    return step_values, positions
