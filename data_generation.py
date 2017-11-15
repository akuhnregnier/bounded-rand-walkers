#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
From an arbitrary intrinsic step size distribution and
arbitrary bounds in 1D or 2D, generate the modified step
size distributions numerically.

"""
import logging
import numpy as np
import scipy.optimize
import scipy.stats
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt


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
        >>> plt.figure()
        >>> from data_generation import generate_random_samples
        >>> plt.hist(generate_random_samples(
        ...     scipy.stats.norm(0.5, 0.1).pdf, 1000))
        >>> plt.show()

    """
    logger = logging.getLogger(__name__)
    if hasattr(f_i, 'max_fn_value'):
        logger.debug('Retrieving max fn value')
        max_fn = f_i.max_fn_value
    else:
        logger.debug('Finding maximum of f_i')
        if dimensions == 1:
            max_x = scipy.optimize.fmin(lambda x: -f_i(x), 0)
            max_fn = f_i(max_x)
        elif dimensions == 2:
            max_args = scipy.optimize.fmin(lambda args: -f_i(*args), (0, 0))
            max_fn = f_i(*max_args)
        # fix this value so it does not need to be calculated next time
        f_i.max_fn_value = max_fn

    # imagine as throwing darts, with a 'height' randomly distributed
    # from 0 to ``max_fn``, and a position randomly along x (1D)
    # or (x, y) (2D). If the 'height' is below the return value
    # of ``f_i`` at the randomly chosen point, keep the point.
    # If not, randomly sample position and 'height' again.

    distribution = np.zeros((nr_samples, dimensions), dtype=np.float64)
    for sample in range(nr_samples):
        found = False
        while not found:
            # get random position and height
            random_p = np.random.uniform(low=0.0, high=max_fn, size=1)
            # bounded between -pos_i and 1 - pos_i
            random_args = [
                    np.random.uniform(
                        low=-pos_i, high=1-pos_i, size=1)
                    for pos_i in position
                    ]

            # now test the actual p value at the position
            # given by ``random_args``

            actual_p = f_i(*random_args)
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


def random_walker(f_i, bounds, steps=int(1e2), return_positions=False):
    """
    Trace a random walker given the ``bounds`` and the given
    intrinsic step size distribution ``f_i``.
    If a step would lead outside of the boundaries, reject this step
    and choose another one until the boundaries are satisfied.
    (This effectively normalises the pdf given the truncation at the
    boundaries.)
    The walker is started at a random position.

    note::

        All lengths and bounds are to be given in the range [0, 1]!

    Args:
        f_i (function): The intrinsic step size distribution,
            given as a pdf (probability distribution function).
            The function should return a probability for
            each step size ``l``, where ``l`` should be
            between 0 and 1. For a 1D problem, ``f_i`` should take
            one argument, whereas for a 2D problem, it should take
            two aguments.
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
        return_positions (bool): If False (default), return an array
            of step sizes taken by the random walker. If True,
            return a tuple of the step size array, as well as an array
            containing the positions the random walker visited.

    Returns:
        numpy.ndarray: 1D array of length ``steps``, containing the
        step sizes obtained by executing the random walker simulation.
        If ``return_positions`` is True, returns a tuple of the above
        array and an array containing the positions of the random
        walker.

    Examples:
        >>> # 1D examples
        >>> def f_i(x):
        ...     '''very crude uniform distribution'''
        ...     return 1.
        >>> def f2_i(x):
        ...     '''tophat uniform distribution'''
        ...     if abs(x) < 0.25:
        ...         return 1.
        >>> step_values, positions = random_walker(
        ...         f_i=f2_i,
        ...         bounds=np.array([0, 1]),
        ...         steps=int(1e5),
        ...         return_positions=True,
        ...         )
        >>> fig, axes = plt.subplots(1, 2, squeeze=True)
        >>> axes[0].hist(step_values, bins='auto')
        >>> axes[0].set_title('Step Values')
        >>> axes[1].hist(positions, bins='auto')
        >>> axes[1].set_title('Positions')
        >>> plt.show()
        >>> # 2D examples
        >>> def f_2D(x, y):
        ...     '''sample 2D pdf function'''
        ...     # x is length 2
        ...     return (scipy.stats.norm(0, 0.25).pdf(x)
        ...             + scipy.stats.norm(0, 0.25).pdf(y))
        >>> bounds = np.array([
        ...     [0, 0],
        ...     [0, 1],
        ...     [1, 1],
        ...     [1, 0]]
        ...     )
        >>> step_values, positions = random_walker(
        ...         f_i=f_2D,
        ...         bounds=bounds,
        ...         steps=int(1e4),
        ...         return_positions=True,
        ...         )

        >>> fig, axes = plt.subplots(1, 2, squeeze=True)
        >>> axes[0].hexbin(*step_values.T)
        >>> axes[0].set_title('Step Values')
        >>> axes[1].hexbin(*positions.T)
        >>> axes[1].set_title('Positions')
        >>> axes[0].set_aspect('equal')
        >>> axes[1].set_aspect('equal')
        >>> plt.show()
    """
    logger = logging.getLogger(__name__)
    bounds = DelaunayArray(bounds, Delaunay(bounds))
    if bounds.size == bounds.shape[0]:
        bounds = bounds.reshape(-1, 1)

    dimensions = bounds.shape[1]
    positions = np.zeros((steps + 1, dimensions), dtype=np.float64)
    step_values = np.zeros((steps, dimensions), dtype=np.float64)
    # give random initial position
    positions[0] = np.random.uniform(low=0.0, high=1.0, size=dimensions)
    for position_index in range(1, steps + 1):
        if (position_index % ((steps + 1) / 10)) == 0 or (steps < 10):
            logger.info('Position index:{:} out of {:}'.format(position_index,
                                                               steps + 1))
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
    if return_positions:
        return step_values, positions
    else:
        return step_values


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    # test the sampling function
    # import scipy.stats
    # plt.figure()
    # plt.hist(generate_random_samples(scipy.stats.norm(0.5, 0.1).pdf, 1000))
    # plt.show()

    # def f_i(x):
    #     """very crude uniform distribution"""
    #     return 1.

    # def f2_i(x):
    #     """tophat uniform distribution"""
    #     if abs(x) < 0.25:
    #         return 1.

    # step_values, positions = random_walker(
    #         f_i=f2_i,
    #         bounds=np.array([0, 1]),
    #         steps=int(1e5),
    #         return_positions=True,
    #         )

    # fig, axes = plt.subplots(1, 2, squeeze=True)
    # axes[0].hist(step_values, bins='auto')
    # axes[0].set_title('Step Values')
    # axes[1].hist(positions, bins='auto')
    # axes[1].set_title('Positions')
    # plt.show()

    def f_2D(x, y):
        """sample 2D pdf function"""
        # x is length 2
        return (scipy.stats.norm(0, 0.25).pdf(x)
                + scipy.stats.norm(0, 0.25).pdf(y))

    bounds = np.array([
        [0, 0],
        [0, 1],
        [1, 1],
        [1, 0]]
        )

    step_values, positions = random_walker(
            f_i=f_2D,
            bounds=bounds,
            steps=int(1e5),
            return_positions=True,
            )

    fig, axes = plt.subplots(1, 2, squeeze=True)
    axes[0].hexbin(*step_values.T)
    axes[0].set_title('Step Values')
    axes[1].hexbin(*positions.T)
    axes[1].set_title('Positions')
    axes[0].set_aspect('equal')
    axes[1].set_aspect('equal')
    plt.show()

