#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Intrinsic step size distributions."""
import numpy as np
from numba import njit


def Tophat_1D(width=0.5, centre=0.0):
    """Tophat uniform pdf.

    Note that the distribution is not normalised.

    Parameters
    ----------
    width : float
        Total tophat width.
    centre : float
        Tophat centre.

    Returns
    -------
    pdf : callable
        The pdf to be called with the position.

    """

    def pdf(x):
        """Sample the pdf at `x`."""
        if np.abs(x[0] - centre) < (width / 2.0):
            return 1.0
        else:
            return 0.0

    return pdf


def Tophat_2D(extent=0.5, x_centre=0.0, y_centre=0.0, type_2D=0):
    """Tophat uniform pdf.

    Note that the distribution is not normalised.

    The distribution has a diameter `extent` in the
    'circularly-symmetric' 2D case, and a square shape with side length
    `extent` in the 'square' case.

    Parameters
    ----------
    extent : float
        The extent of the distribution. See above for details.
    x_centre : float
        The centre of the distribution in the x-dimension.
    y_centre : float
        The centre of the distribution in the y-dimension.
    type_2D : int
        If 0 is given ('circularly-symmetric'), the distribution is circular in
        the 2D plane. If 1 is given ('square'), the distribution is square in the
        2D plane.

    Returns
    -------
    pdf : callable
        The pdf to be called with the position.

    """

    @njit
    def pdf(pos):
        """Calculate the probability at a point.

        Parameters
        ----------
        pos : array
            Point (x, y) coordinates.

        Returns
        -------
        prob : float
            Probability at the given point.

        """
        x = pos[0]
        y = pos[1]
        if type_2D == 0:
            if ((x - x_centre) ** 2.0 + (y - y_centre) ** 2.0) < (extent / 2.0) ** 2.0:
                return 1.0
            else:
                return 0.0
        elif type_2D == 1:
            if (np.abs(x - x_centre) < (extent / 2.0)) and (
                np.abs(y - y_centre) < (extent / 2.0)
            ):
                return 1.0
            else:
                return 0.0
        else:
            return -1.0

    return pdf


def Power(centre=np.array([0.0]), exponent=1.0, binsize=0.001):
    """A rotationally symmetric power law distribution.

    Parameters
    ----------
    centre : array
        Center of the power law. For a 2D distribution, give a numpy array of x, y
        position (ie. [x, y]) of the centre of the exponential. For the 1D
        distribution, give a 1-element numpy array.
    exponent : float
        Characteristic exponent of the probability decay.
    binsize : float
        Scale of UV cutoff, should have scale of hist binsize.

    Returns
    -------
    pdf : callable
        The pdf to be called with the position.

    """
    centre = np.asarray(centre)

    @njit
    def pdf(pos):
        """Calculate the probability at a point.

        Parameters
        ----------
        pos : array
            Point (x, y) coordinates.

        Returns
        -------
        prob : float
            Probability at the given point.

        """
        d = np.linalg.norm(np.asarray(pos) - centre[: len(pos)])
        return (d + binsize) ** (-exponent) / (2 * binsize ** (1 - exponent))

    return pdf


def Gaussian(centre=np.array([0.0]), width=1.0):
    """A Gaussian distribution.

    Parameters
    ----------
    centre : array
        Centre of the Gaussian. For a 2D distribution, give a list of x, y
        position (ie. [x, y]) of the centre of the Gaussian.
    width : float
        Used to scale the Gaussian.

    Returns
    -------
    pdf : callable
        The pdf to be called with the position.

    """

    @njit
    def pdf(pos):
        """Calculate the probability at a point.

        Parameters
        ----------
        pos : array
            Point (x, y) coordinates.

        Returns
        -------
        prob : float
            Probability at the given point.

        """
        # Symmetric in radius.
        d2 = np.sum(np.square(pos - centre[: len(pos)]))
        return np.exp(-d2 / (2 * width ** 2))

    return pdf


def Exponential(centre=0.0, decay_rate=1.0):
    """A rotationally symmetric exponential distribution.

    Parameters
    ----------
    centre : array
        Center of the exponential. For a 2D distribution, give a list of x, y
        position (ie. [x, y]) of the centre of the exponential.
    decay_rate : float
        The constant governing the decay of the exponential.

    Returns
    -------
    pdf : callable
        The pdf to be called with the position.

    """

    @njit
    def pdf(pos):
        """Calculate the probability at a point.

        Parameters
        ----------
        pos : array
            Point (x, y) coordinates.

        Returns
        -------
        prob : float
            Probability at the given point.

        """
        # Symmetric in radius.
        d = np.linalg.norm(np.asarray(pos) - centre[: len(pos)])
        return np.exp(-d * decay_rate)

    return pdf


def Freehand(centre=None, width=1.0):
    """The pdf used for the numerical results presented in the paper.

    Notes
    -----
    Note that this width is 3x the width in the analytical expression for the pdf
    in the paper.

    """
    if centre is None:
        centre = np.array([0.0])
    centre = np.asarray(centre)
    frequency = 3.7

    third = (1 / 3) * width  # Width of the pdf sections.

    power_law = Power(np.array([2 * third]), 0.25, 0.001)

    @njit
    def pdf(pos):
        """Calculate the probability at a point.

        Parameters
        ----------
        pos : array
            Point (x, y) coordinates.

        Returns
        -------
        prob : float
            Probability at the given point.

        """
        position = np.linalg.norm(pos - centre[: len(pos)])

        scale = power_law(np.array([2 * third]))

        const1 = np.abs(np.sinc(third * frequency)) * (1 + 5 * third)
        const2 = const1 * (1 + third)

        if position == 0:
            prob = 1.0
        elif position > 0 and position <= third:
            prob = np.abs(np.sinc(position * frequency))
            # multiply by linear factor
            prob = prob * (1 + 5 * position)
        elif position > third and position <= 2 * third:
            prob = const1 * (1 + position - third)
        elif position > 2 * third:
            prob = const2 * power_law(np.array([position])) / scale

        return prob

    return pdf


def Freehand2(*args, **kwargs):
    """Scaled version of the `Freehand` pdf."""
    pdf = Freehand(*args, **kwargs)

    def scaled(pos):
        """Calculate the probability at a point.

        Parameters
        ----------
        pos : array
            Point (x, y) coordinates.

        Returns
        -------
        prob : float
            Probability at the given point.

        """
        return pdf(2 * pos)

    return scaled
