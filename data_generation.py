#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
From an arbitrary intrinsic step size distribution and
arbitrary bounds in 1D or 2D, generate the modified step
size distributions numerically.

"""
import logging
import multiprocessing
import numpy as np
import scipy.optimize
import scipy.stats
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from time import time
from functions import Tophat_1D, Tophat_2D, Gaussian, Power, Exponential


# bounds with x coords in the first column and y coords in the second
weird_bounds = np.array([
    [0.1, 0.3],
    [0.25, 0.98],
    [0.9, 0.9],
    [0.7, 0.4],
    [0.4, 0.05]]
    )


def circle_points(radius=1., samples=20):
    angles = np.linspace(0, 2*np.pi, samples, endpoint=False)
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
        self.tri = getattr(obj, 'tri', None)


def generate_random_samples(f_i, position, nr_samples, dimensions=1):
    """
    Use the rejection sampling method in order to generate
    samples distributed according to the pdf f_i.

    The random sampling is done within the interval [-pos_i, 1 - pos_i]
    with either ``pos_0 = x`` (1D) or ``pos_0 = x`` and ``pos_1 = y`` (2D)!

    Args:
        f_i (function): The intrinsic step size distribution,
            given as a pdf (probability distribution function).
            The function should return a probability for
            each step size ``l``, where ``l`` should be
            between 0 and 1. For a 1D problem, ``f_i`` should take
            one argument, whereas for a 2D problem, it should take
            two aguments.
        position (numpy.ndarray): Current position of the random
            walker. The number of elements in this array dictate the
            dimensionality of the program, ie. 1 element -> 1D,
            2 elements -> 2D.
        nr_samples (int): The number of samples to return
        dimensions (int): The dimensionality of the problem. Defaults to 1.

    Returns:
        numpy.ndarray: Array of samples distributed according
            to ``f_i``.

    Examples:
        >>> import scipy.stats
        >>> import matplotlib.pyplot as plt
        >>> from data_generation import generate_random_samples
        >>> plt.figure()
        >>> plt.hist(generate_random_samples(
        ...     scipy.stats.norm(0.5, 0.1).pdf, 1000))
        >>> plt.show()

    """
    logger = logging.getLogger(__name__)
    if generate_random_samples.max_fn_value != 0:
        logger.debug('Retrieving max fn value')
        max_fn = generate_random_samples.max_fn_value
        logger.debug('Got:{:}'.format(max_fn))
    else:
        logger.info('Finding maximum of f_i')
        # need to be careful, since the minimiser might never find the
        # minimum if the function is like a tophat function - ie. if it is
        # flat in some regions!
        if dimensions == 1:
            # look at the function value at many positions and then find
            # the minimum around the minimum of the points discovered so
            # far
            trial_x_coords = np.linspace(0, 1, 1000)
            fn_values = [f_i(x) for x in trial_x_coords]
            trial_max_x = trial_x_coords[np.argmax(fn_values)]
            max_x = scipy.optimize.fmin(lambda x: -f_i(x), trial_max_x)
            max_fn = f_i(max_x)
        elif dimensions == 2:
            # look at the function value at many positions and then find
            # the minimum around the minimum of the points discovered so
            # far
            N = 100
            trial_coords = np.linspace(-1, 1, N)
            fn_values = np.zeros((N, N), dtype=np.float64)
            for i in range(N):
                for j in range(N):
                    fn_values[i, j] = f_i(trial_coords[i], trial_coords[j])

            max_index = np.unravel_index(np.argmax(fn_values),
                                         fn_values.shape)
            trial_max_x = trial_coords[max_index[0]]
            trial_max_y = trial_coords[max_index[1]]
            max_args = scipy.optimize.fmin(lambda args: -f_i(*args),
                                           (trial_max_x, trial_max_y)
                                           )
            max_fn = f_i(*max_args)
        # fix this value so it does not need to be calculated next time
        logger.info('storing maximum fn value')
        generate_random_samples.max_fn_value = max_fn
        logger.debug('max_fn:{:}'.format(generate_random_samples.max_fn_value))

    # imagine as throwing darts, with a 'height' randomly distributed
    # from 0 to ``max_fn``, and a position randomly along x (1D)
    # or (x, y) (2D). If the 'height' is below the return value
    # of ``f_i`` at the randomly chosen point, keep the point.
    # If not, randomly sample position and 'height' again.

    logger.debug('position:{:}'.format(position))

    distribution = np.zeros((nr_samples, dimensions), dtype=np.float64) - 9999
    for sample in range(nr_samples):
        found = False
        tries = 0
        while not found:
            tries += 1
            logger.debug('tries:{:}'.format(tries))
            # get random position and height
            random_p = np.random.uniform(low=0.0, high=max_fn, size=1)
            # bounded between -pos_i and 1 - pos_i
            if dimensions == 1:
                random_args = [
                        np.random.uniform(
                            low=-pos_i, high=1-pos_i, size=1)
                        for pos_i in position
                        ]
            elif dimensions == 2:
                random_args = [
                        np.random.uniform(
                            low=-(1 + pos_i), high=1-pos_i, size=1)
                        for pos_i in position
                        ]
            else:
                raise NotImplementedError('{:} dimensions not implemented'
                                          .format(dimensions))

            # now test the actual p value at the position
            # given by ``random_args``

            actual_p = f_i(*random_args)
            logger.debug('args: {:} actual p: {:} random p:{:}'
                         .format(random_args, actual_p, random_p))
            if random_p < actual_p:
                # add to record
                distribution[sample] = random_args
                found = True
    return distribution


def in_bounds(position, bounds):
    """
    Test whether the given ``position`` is within the given ``bounds``.

    Args:
        position (numpy.ndarray): Position to test.
        bounds (DelaunayArray): The bounds for the random walker,
            given as an array. The shape of the array dictates
            the dimensionality of the problem.
            A 1D array containing two values represents the
            lower and upper boundaries (in that order) of the
            1D problem.
            For the 2D problem, the boundary is given as a 2D array,
            where each row contains the (x, y) coordinates of a point.

    """
    if bounds.shape[1] > 1:
        # more than 1D
        return np.all(bounds.tri.find_simplex(position) != -1)
    else:
        return ((position >= bounds[0]) and (position < bounds[1]))


def format_time(time_value):
    """Sensible string formatting of a time variable.

    Output units range from seconds ('s') to minutes ('m')
    and hours ('h'), depending on the magnitude of the
    input ``time_value``.

    Args:
        time_value (float): The time value in seconds
            that is to be formatted.

    Returns:
        tuple of float, str: The converted time value,
            and its associated units are returned.

    """
    units = 's'
    rounded_time = int(round(time_value))
    nr_digits = len(str(rounded_time))
    if nr_digits > 2:
        # convert to minutes
        units = 'm'
        time_value /= 60.
        rounded_time = int(round(time_value))
        nr_digits = len(str(rounded_time))
        if nr_digits > 2:
            # convert to hours
            units = 'h'
            time_value /= 60.
    return time_value, units


def random_walker(f_i, bounds, steps=int(1e2), sampler=None, blocks=50):
    """
    Trace a random walker given the ``bounds`` and the given
    intrinsic step size distribution ``f_i``.
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
            each step size ``l``, where ``l`` should be
            between 0 and 1. For a 1D problem, ``f_i`` should take
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
            ``steps`` number of step sizes. Defaults to ``int(1e4)``.

    Returns:
        numpy.ndarray: 1D array of length ``steps``, containing the
        step sizes obtained by executing the random walker simulation.
        If ``return_positions`` is True, returns a tuple of the above
        array and an array containing the positions of the random
        walker.

    Examples:
        >>> import matplotlib.pyplot as plt
        >>> import numpy as np
        >>> from functions import (Tophat_1D, Tophat_2D, Gaussian,
        ...                        Power, Exponential)
        >>> step_values, positions = random_walker(
        ...         f_i=Tophat_1D().pdf,
        ...         bounds=np.array([0, 1]),
        ...         steps=int(1e5),
        ...         )
        >>> fig, axes = plt.subplots(1, 2, squeeze=True)
        >>> axes[0].hist(step_values, bins='auto')
        >>> axes[0].set_title('Step Values')
        >>> axes[1].hist(positions, bins='auto')
        >>> axes[1].set_title('Positions')
        >>> plt.show()
        >>> #
        >>> bounds = np.array([
        ...     [0, 0],
        ...     [0, 1],
        ...     [1, 1],
        ...     [1, 0]]
        ...     )
        >>> step_values, positions = random_walker(
        ...         f_i=Tophat_2D().pdf,
        ...         bounds=bounds,
        ...         steps=int(1e6),
        ...         )
        >>> fig, axes = plt.subplots(1, 2, squeeze=True)
        >>> fig.subplots_adjust(right=0.8)
        >>> # bin first time to get the maximum bin counts,
        >>> # which are used below
        >>> steps_bin = axes[0].hexbin(*step_values.T)
        >>> positions_bin = axes[1].hexbin(*positions.T)
        >>> axes[0].cla()
        >>> axes[1].cla()
        >>> # use this max value with the hexbin vmax option
        >>> # in order to have the same colour scaling for both
        >>> # hexbin plots, such that the same colorbar may be used
        >>> max_value = np.max([np.max(steps_bin.get_array()),
        ...                     np.max(positions_bin.get_array())
        ...                     ])
        >>> steps_bin = axes[0].hexbin(*step_values.T, vmin=0, vmax=max_value)
        >>> steps_bin = axes[0].hexbin(*step_values.T, vmin=0, vmax=max_value)
        >>> positions_bin = axes[1].hexbin(*positions.T, vmin=0,
        ...                                vmax=max_value)
        >>> positions_bin = axes[1].hexbin(*positions.T, vmin=0,
        ...                                vmax=max_value)
        >>> axes[0].set_title('Step Values')
        >>> axes[1].set_title('Positions')
        >>> for ax in axes:
        >>>     ax.set_aspect('equal')

        >>> cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        >>> fig.colorbar(positions_bin, cax=cbar_ax)
        >>> plt.show()

    """
    logger = logging.getLogger(__name__)
    generate_random_samples.max_fn_value = 0
    if bounds.size == bounds.shape[0]:
        bounds = bounds.reshape(-1, 1)
    else:
        bounds = DelaunayArray(bounds, Delaunay(bounds))

    dimensions = bounds.shape[1]
    positions = np.zeros((steps + 1, dimensions), dtype=np.float64)
    step_values = np.zeros((steps, dimensions), dtype=np.float64)
    # give random initial position
    positions[0] = np.random.uniform(low=0.0, high=1.0, size=dimensions)
    while not in_bounds(positions[0], bounds):
        logger.debug('creating new initial position')
        positions[0] = np.random.uniform(low=0.0, high=1.0, size=dimensions)
    logger.info('Initial walker position:{:}'.format(positions[0]))
    start_time = time()
    for position_index in range(1, steps + 1):
        logger.debug('position index:{:} {:}'
                     .format(position_index, steps + 1))
        if ((position_index % ((steps + 1) / 10)) == 0 or (steps < 10)) or (
                position_index == int(4e3)):
            elapsed_time = time() - start_time
            elapsed_time_per_step = elapsed_time / position_index
            remaining_time = (steps - position_index) * elapsed_time_per_step
            elapsed_time, elapsed_time_units = format_time(elapsed_time)
            remaining_time, remaining_time_units = format_time(remaining_time)
            logger.info('Position index:{:.1e} out of {:.1e}'
                        .format(position_index, steps + 1))
            logger.info('Time elapsed:{:>4.1f} {:}, remaining:{:0.1f} {:}'
                        .format(elapsed_time, elapsed_time_units,
                                remaining_time, remaining_time_units))
        logger.debug('Current position:{:}'
                     .format(positions[position_index - 1]))
        step_index = position_index - 1
        found = False
        while not found:
            step = generate_random_samples(
                    f_i=f_i,
                    position=positions[position_index - 1],
                    nr_samples=1,
                    dimensions=dimensions
                    )
            next_position = positions[position_index - 1] + step
            if in_bounds(next_position, bounds):
                positions[position_index] = next_position
                step_values[step_index] = step
                found = True
    elapsed_time = time() - start_time
    elapsed_time_per_step = elapsed_time / position_index
    logger.info('time per step:{:.1e} s'.format(elapsed_time_per_step))
    return step_values, positions


def multi_random_walker(n_processes, f_i, bounds, steps=int(1e2), blocks=50):
    """Generate random walks in multiple processes concurrently.

    If the ``n_processes==1``, the ``random_walker`` function is called in
    the standard way, as if it was called directly.

    Args:
        n_processes (int): The number of processes to run in parallel. The
            generated data will be joined together at the end.

    For an explanation of the other arguments, see ``random_walker``.

    Note that the generated position series have no causal relationship -
    they are completely random, since each random_walker execution has no
    information about any of the other executions run in parallel.
    Therefore, generating steps from the returned positions produces an
    error at the point where the different datasets are joined together.
    However, this error will approach 0 as ``steps`` is increased, since
    the ratio of 'contiguous' data to erroneous data (at the boundary of
    the contiguous random walks) increases.

    """
    assert n_processes >= 1
    if n_processes == 1:
        # simply execute random_walker while ignoring the multiprocessing
        return random_walker(
                f_i=f_i,
                bounds=bounds,
                steps=steps,
                )
    def rand_walk_worker(procnum, return_dict):
        """Worker function which executes the random_walker"""
        print('process ' + str(procnum) + ' has started!')
        step_values, positions = random_walker(
                f_i=f_i,
                bounds=bounds,
                steps=int(steps / n_processes),
                )
        return_dict[procnum] = (step_values, positions)

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i in range(n_processes):
        np.random.seed(i)
        p = multiprocessing.Process(target=rand_walk_worker,
                                    args=(i, return_dict))
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    step_values = [i[0] for i in return_dict.values()]
    positions = [i[1] for i in return_dict.values()]
    step_values = np.vstack(step_values)
    positions = np.vstack(positions)

    print('data shapes (steps, positions)')
    print(step_values.shape, positions.shape)

    return step_values, positions


if __name__ == '__main__':
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

        print('data shapes')
        print(step_values.shape, positions.shape)

        fig, axes = plt.subplots(1, 2, squeeze=True)
        axes[0].hist(step_values, bins='auto')
        axes[0].set_title('Step Values')
        axes[1].hist(positions, bins='auto')
        axes[1].set_title('Positions')
        plt.show()

    if TWO_D:
        # bounds = np.array([
        #     [0, 0],
        #     [0, 1],
        #     [1, 1],
        #     [1, 0]],
        #     dtype=np.float64
        #     )
        bounds = np.array([
            [1, 0],
            [-1, 1],
            [-1, -1]]
            )

        step_values, positions = multi_random_walker(
                n_processes=4,
                f_i=Tophat_2D(extent=1.).pdf,
                bounds=bounds,
                steps=int(5e3),
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
        max_value = np.max([np.max(steps_bin.get_array()),
                            np.max(positions_bin.get_array())
                            ])
        steps_bin = axes[0].hexbin(*step_values.T, vmin=0, vmax=max_value)
        steps_bin = axes[0].hexbin(*step_values.T, vmin=0, vmax=max_value)
        positions_bin = axes[1].hexbin(*positions.T, vmin=0, vmax=max_value)
        positions_bin = axes[1].hexbin(*positions.T, vmin=0, vmax=max_value)

        axes[0].set_title('Step Values')
        axes[1].set_title('Positions')
        for ax in axes:
            ax.set_aspect('equal')

        cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
        fig.colorbar(positions_bin, cax=cbar_ax)

        plt.show()
