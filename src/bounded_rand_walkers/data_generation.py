#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
From an arbitrary intrinsic step size distribution and
arbitrary bounds in 1D or 2D, generate the modified step
size distributions numerically.

"""
import logging
import multiprocessing
from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay

from .functions import Tophat_1D, Tophat_2D
from .rejection_sampling import Sampler

# bounds with x coords in the first column and y coords in the second
weird_bounds = np.array([[0.1, 0.3], [0.25, 0.98], [0.9, 0.9], [0.7, 0.4], [0.4, 0.05]])


def circle_points(radius=1.0, samples=20):
    """Generate an array of (x, y) coordinates arranged in a circle.

    Parameters
    ----------
    radius : float
        Circle radius.
    samples : int
        How many points to generate along the circle.

    Returns
    -------
    points : array of shape (`samples`, 2)
        Points along the circle.

    """
    angles = np.linspace(0, 2 * np.pi, samples, endpoint=False)
    x = (np.cos(angles) * radius).reshape(-1, 1)
    y = (np.sin(angles) * radius).reshape(-1, 1)
    return np.hstack((x, y))


class DelaunayArray(np.ndarray):
    def __new__(cls, input_array, tri=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        # add the new attribute to the created instance
        obj.tri = tri
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.tri = getattr(obj, "tri", None)


def in_bounds(position, bounds):
    """Test whether the given `position` is within the given `bounds`.

    Parameters
    ----------
    position : array
        Position to test.
    bounds : DelaunayArray
        The bounds for the random walker, given as an array. The shape of the array
        dictates the dimensionality of the problem. A 1D array containing two values
        represents the lower and upper boundaries (in that order) of the 1D problem.
        For the 2D problem, the boundary is given as a 2D array, where each row
        contains the (x, y) coordinates of a point.

    Returns
    -------
    present : bool
        True if `position` is within `bounds`.

    """
    if bounds.shape[1] > 1:
        # more than 1D
        return np.all(bounds.tri.find_simplex(position) != -1)
    else:
        return (position >= bounds[0]) and (position < bounds[1])


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


def random_walker(f_i, bounds, steps=int(1e2), sampler=None, blocks=50):
    """
    Trace a random walker given the `bounds` and the given
    intrinsic step size distribution `f_i`.
    If a step would lead outside of the boundaries, reject this step
    and choose another one until the boundaries are satisfied.
    (This effectively normalises the pdf given the truncation at the
    boundaries.)
    The walker is started at a random position.

    note::

        All lengths and bounds are to be given in the range [0, 1] for the
        1D case!

    Args:
        f_i (function): The intrinsic step size distribution,
            given as a pdf (probability distribution function).
            The function should return a probability for
            each step size `l`, where `l` should be
            between 0 and 1. For a 1D problem, `f_i` should take
            one argument, whereas for a 2D problem, it should take
            two arguments.
        bounds (numpy.ndarray): The bounds for the random walker,
            given as an array. The shape of the array dictates
            the dimensionality of the problem.
            A 1D array containing two values represents the
            lower and upper boundaries (in that order) of the
            1D problem.
            For the 2D problem, the boundary is given as a 2D array,
            where each row contains the (x, y) coordinates of a point.
        steps (int): The number of steps to take. The function will return
            `steps` number of step sizes. Defaults to `int(1e4)`.

    Returns:
        numpy.ndarray: 1D array of length `steps`, containing the
        step sizes obtained by executing the random walker simulation.
        If `return_positions` is True, returns a tuple of the above
        array and an array containing the positions of the random
        walker.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from bounded_rand_walkers.functions import (
        ...     Tophat_1D, Tophat_2D, Gaussian, Power, Exponential
        ... )
        >>> step_values, positions = random_walker(
        ...     f_i=Tophat_1D().pdf,
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
        ...     f_i=Tophat_2D().pdf,
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
        bounds = DelaunayArray(bounds, Delaunay(bounds))

    dimensions = bounds.shape[1]
    if sampler is None:
        sampler = Sampler(f_i, dimensions, blocks=blocks)
    positions = np.zeros((steps + 1, dimensions), dtype=np.float64)
    step_values = np.zeros((steps, dimensions), dtype=np.float64)
    # give random initial position
    positions[0] = np.random.uniform(low=0.0, high=1.0, size=dimensions)
    while not in_bounds(positions[0], bounds):
        logger.debug("creating new initial position")
        positions[0] = np.random.uniform(low=0.0, high=1.0, size=dimensions)
    logger.info("Initial walker position:{:}".format(positions[0]))
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


def multi_random_walker(n_processes, f_i, bounds, steps=int(1e2), blocks=50):
    """Generate random walks in multiple processes concurrently.

    If the `n_processes==1`, the `random_walker` function is called in
    the standard way, as if it was called directly.

    Args:
        n_processes (int): The number of processes to run in parallel. The
            generated data will be joined together at the end.

    For an explanation of the other arguments, see `random_walker`.

    Note that the generated position series have no causal relationship -
    they are completely random, since each random_walker execution has no
    information about any of the other executions run in parallel.
    Therefore, generating steps from the returned positions produces an
    error at the point where the different datasets are joined together.
    However, this error will approach 0 as `steps` is increased, since
    the ratio of 'contiguous' data to erroneous data (at the boundary of
    the contiguous random walks) increases.

    """
    assert n_processes >= 1
    if n_processes == 1:
        # simply execute random_walker while ignoring the multiprocessing
        step_values, positions = random_walker(
            f_i=f_i, bounds=bounds, steps=steps, blocks=blocks
        )
        print("data shapes (steps, positions)")
        print((step_values.shape, positions.shape))
        return step_values, positions

    if bounds.size == bounds.shape[0]:
        bounds = bounds.reshape(-1, 1)
    dimensions = bounds.shape[1]

    sampler = Sampler(f_i, dimensions, blocks=blocks)

    def rand_walk_worker(procnum, return_dict):
        """Worker function which executes the random_walker"""
        print(("process " + str(procnum) + " has started!"))
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

    print("data shapes (steps, positions)")
    print((step_values.shape, positions.shape))

    return step_values, positions


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ONE_D = False
    TWO_D = True
    # test the sampling function
    # import scipy.stats
    # plt.figure()
    # plt.hist(generate_random_samples(scipy.stats.norm(0.5, 0.1).pdf, 1000))
    # plt.show()

    if ONE_D:

        step_values, positions = multi_random_walker(
            n_processes=4,
            f_i=Tophat_1D(width=0.5, centre=0.2).pdf,
            bounds=np.array([0, 1]),
            steps=int(1e3),
        )

        print("data shapes")
        print((step_values.shape, positions.shape))

        fig, axes = plt.subplots(1, 2, squeeze=True)
        axes[0].hist(step_values, bins="auto")
        axes[0].set_title("Step Values")
        axes[1].hist(positions, bins="auto")
        axes[1].set_title("Positions")
        plt.show()

    if TWO_D:
        # bounds = np.array([
        #     [0, 0],
        #     [0, 1],
        #     [1, 1],
        #     [1, 0]],
        #     dtype=np.float64
        #     )
        bounds = np.array([[1, 0], [-1, 1], [-1, -1]])

        step_values, positions = multi_random_walker(
            n_processes=4,
            f_i=Tophat_2D(extent=1.5).pdf,
            bounds=bounds,
            steps=int(1e4),
            blocks=3,
        )

        fig, axes = plt.subplots(1, 2, squeeze=True)
        fig.subplots_adjust(right=0.8)
        # bin first time to get the maximum bin counts,
        # which are used below
        steps_bin = axes[0].hexbin(*step_values.T)
        positions_bin = axes[1].hexbin(*positions.T)
        axes[0].cla()
        axes[1].cla()

        # use this max value with the hexbin vmax option
        # in order to have the same colour scaling for both
        # hexbin plots, such that the same colorbar may be used
        max_value = np.max(
            [np.max(steps_bin.get_array()), np.max(positions_bin.get_array())]
        )
        steps_bin = axes[0].hexbin(*step_values.T, vmin=0, vmax=max_value)
        steps_bin = axes[0].hexbin(*step_values.T, vmin=0, vmax=max_value)
        positions_bin = axes[1].hexbin(*positions.T, vmin=0, vmax=max_value)
        positions_bin = axes[1].hexbin(*positions.T, vmin=0, vmax=max_value)

        axes[0].set_title("Step Values")
        axes[1].set_title("Positions")
        for ax in axes:
            ax.set_aspect("equal")

        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(positions_bin, cax=cbar_ax)

        plt.show()
