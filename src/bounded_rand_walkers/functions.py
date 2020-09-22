#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from numba import float64, int64, njit
from numba.experimental import jitclass


# Re-define `get_centres` here since we need a jitted version.
@njit
def get_centres(edges):
    """Get bin centres from edges."""
    return (edges[1:] + edges[:-1]) / 2.0


spec = [
    ("width", float64),
    ("centre", float64),
]


@jitclass(spec)
class Tophat_1D(object):
    def __init__(self, width=0.5, centre=0.0):
        """Tophat uniform pdf.

        The distribution is not normalised.
        The distribution has a total width ``extent`` in the 1D case,

        Args:
            x (float): In the 1D case, the x-values to
                calculate the function value for.
        """
        self.width = width
        self.centre = centre

    def pdf(self, x):
        if np.abs(x[0] - self.centre) < (self.width / 2.0):
            return 1.0
        else:
            return 0.0


spec = [
    ("extent", float64),
    ("x_centre", float64),
    ("y_centre", float64),
    ("type_2D", int64),
]


@jitclass(spec)
class Tophat_2D(object):
    def __init__(self, extent=0.5, x_centre=0.0, y_centre=0.0, type_2D=0):
        """Tophat uniform pdf.

        The distribution is not normalised.
        The distribution has a diameter ``extent`` in the
        'circularly-symmetric' 2D case, and a square shape with side length
        ``extent`` in the 'square' case.

        Args:
            extent (float): The extent of the distribution. See above for
                details.
            x_centre (float): The centre of the distribution in the
                x-dimension.
            y_centre (float): The centre of the distribution in the
                y-dimension.
            type_2D (int): If ``type_2D is 0 ('circularly-symmetric')``, the
                distribution is circular in the 2D plane.
                If ``type_2D is 1 ('square')``, the
                distribution is square in the 2D plane.

        """
        self.extent = extent
        self.x_centre = x_centre
        self.y_centre = y_centre
        self.type_2D = type_2D

    def pdf(self, pos):
        """Tophat uniform pdf.

        The distribution is not normalised.
        The distribution has a diameter ``extent`` in the
        'circularly-symmetric' 2D case, and a square shape with side length
        ``extent`` in the 'square' case.

        Args:
            x (float): The (x, y) value for which to calculate
                the pdf value.

        Returns:
            p (float): The probability at the point ``(x, y)``.

        """
        x = pos[0]
        y = pos[1]
        if self.type_2D == 0:
            if ((x - self.x_centre) ** 2.0 + (y - self.y_centre) ** 2.0) < (
                self.extent / 2.0
            ) ** 2.0:
                return 1.0
            else:
                return 0.0
        elif self.type_2D == 1:
            if (np.abs(x - self.x_centre) < (self.extent / 2.0)) and (
                np.abs(y - self.y_centre) < (self.extent / 2.0)
            ):
                return 1.0
            else:
                return 0.0
        else:
            return -1.0
            # raise NotImplementedError('type_2D value {:} not implemented.'
            #                           .format(self.type_2D))


spec = [
    ("centre", float64[:]),
    ("exponent", float64),
    ("binsize", float64),
]


@jitclass(spec)
class Power(object):
    def __init__(self, centre=np.array([0.0]), exponent=1.0, binsize=0.001):
        """A rotationally symmetric power law distribution.

        Args:
            centre: Center of the power law. For a 2D distribution, give a
                numpy array of x, y position (ie. [x, y]) of the centre of the
                exponential. For the 1D distribution, give a 1-element
                numpy array.
            exponent: characteristic exponent of the probability decay.
            binsize= scale of UV cutoff, should have scale of hist binsize

        """
        self.centre = centre
        self.exponent = exponent
        self.binsize = binsize

    def pdf(self, pos):
        """Calculate the probability at a point.

        Args:
            *args: The coordinates of the points of interest. For the 1D
                case, call like pdf(x). In the 2D case, give coordinates
                like pdf(x, y).

        Returns:
            prob: probability at the given point

        """

        if len(pos) == 1:
            x = pos[0]
            prob = 0.5 * (
                (np.abs(x - self.centre[0]) + self.binsize) ** (-self.exponent)
                / (self.binsize) ** (1 - self.exponent)
            )

        elif len(pos) == 2:
            x, y = pos
            radius = (
                (x - self.centre[0]) ** 2 + (y - self.centre[1]) ** 2
            ) ** 0.5 + self.binsize
            prob = 0.5 * (
                radius ** (-self.exponent) / (self.binsize) ** (1 - self.exponent)
            )

        return prob


spec = [
    ("centre", float64[:]),
    ("width", float64),
]


@jitclass(spec)
class Gaussian(object):
    def __init__(self, centre=np.array([0.0]), width=1.0):
        """A Gaussian distribution.

        Args:
            centre: Centre of the Gaussian. For a 2D distribution, give a
                list of x, y position (ie. [x, y]) of the centre of the
                Gaussian.
            width: Used to scale the Gaussian.

        """
        self.centre = centre
        self.width = width

    def pdf(self, pos):
        """Calculate the probability at a point.

        Args:
            *args: The coordinates of the points of interest. For the 1D
                case, call like pdf(x). In the 2D case, give coordinates
                like pdf(x, y).

        Returns:
            prob: probability at the given point

        """
        if len(pos) == 1:
            x = pos[0]
            expon = -((x - self.centre[0]) ** 2) / (2 * self.width ** 2.0)
        elif len(pos) == 2:
            x, y = pos
            # symmetric in radius
            expon = -(
                ((x - self.centre[0]) ** 2 + (y - self.centre[1]) ** 2)
                / (2 * self.width ** 2.0)
            )
        return np.exp(expon)


spec = [
    ("centre", float64[:]),
    ("decay_rate", float64),
]


@jitclass(spec)
class Exponential(object):
    def __init__(self, centre=0.0, decay_rate=1.0):
        """
        A rotationally symmentric exponential distribution.

        Args:
            centre: Center of the exponential. For a 2D distribution, give a
                list of x, y position (ie. [x, y]) of the centre of the
                exponential.
            decay_rate: the constant governing the decay of the exponential.

        """

        self.centre = centre
        self.decay_rate = decay_rate

    def pdf(self, pos):
        """
        Calculates the probability at a point.

        Args:
            *args: The coordinates of the points of interest. For the 1D
                case, call like pdf(x). In the 2D case, give coordinates
                like pdf(x, y).

        Returns:
            prob: probability at the given point

        """

        if len(pos) == 1:
            x = pos[0]
            prob = self.decay_rate * np.exp(
                -np.abs(x - self.centre[0]) * self.decay_rate
            )
        elif len(pos) == 2:
            x, y = pos
            # symmetric in radius
            prob = np.exp(
                -np.sqrt((x - self.centre[0]) ** 2 + (y - self.centre[1]) ** 2)
                * self.decay_rate
            )
        return prob


spec = [
    ("centre", float64[:]),
    ("width", float64),
    ("frequency", float64),
    ("grad", float64),
]


@jitclass(spec)
class Funky(object):
    def __init__(self, centre=0.0, width=1.0):
        """"""
        self.centre = centre
        self.width = width
        self.frequency = 3.7
        self.grad = 1.0

    def pdf(self, pos):

        if len(pos) == 1:
            x = pos[0]
            position = np.abs(x - self.centre[0])
            power_law = Power(np.array([2 / 3.0 * self.width]), 0.25, 0.001)
            scale = power_law.pdf(np.array([2 / 3.0 * self.width]))

            const1 = np.abs(np.sinc(1 / 3.0 * self.width * self.frequency)) * (
                1 + 5 * 1 / 3.0 * self.width
            )
            const2 = const1 * (1 + self.grad * 1 / 3.0 * self.width)

            if position == 0:
                prob = 1.0
            elif position > 0 and position <= 1 / 3.0 * self.width:
                prob = np.abs(np.sinc(position * self.frequency))

                # multiply by linear factor
                prob = prob * (1 + 5 * position)

            elif position > 1 / 3.0 * self.width and position <= 2 / 3.0 * self.width:
                prob = const1 * (1 + self.grad * (position - 1 / 3.0 * self.width))
            elif position > 2 / 3.0 * self.width:
                prob = power_law.pdf(np.array([position])) / scale * const2

        if len(pos) == 2:
            x, y = pos
            position = np.sqrt((x - self.centre[0]) ** 2 + (y - self.centre[1]) ** 2)
            power_law = Power(np.array([2 / 3.0 * self.width]), 0.25, 0.001)
            scale = power_law.pdf(np.array([2 / 3.0 * self.width]))

            const1 = np.abs(np.sinc(1 / 3.0 * self.width * self.frequency)) * (
                1 + 5 * 1 / 3.0 * self.width
            )
            const2 = const1 * (1 + self.grad * 1 / 3.0 * self.width)

            if position == 0:
                prob = 1.0
            elif position > 0 and position <= 1 / 3.0 * self.width:
                prob = np.abs(np.sinc(position * self.frequency))

                # multiply by linear factor
                prob = prob * (1 + 5 * position)

            elif position > 1 / 3.0 * self.width and position <= 2 / 3.0 * self.width:
                prob = const1 * (1 + self.grad * (position - 1 / 3.0 * self.width))
            elif position > 2 / 3.0 * self.width:
                prob = power_law.pdf(np.array([position])) / scale * const2

        return prob


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # plt.close("all")
    # plt.ion()
    # 1D case
    pdfs_args_1D = [
        (Tophat_1D, {"width": 0.7, "centre": 0.3}),
        (Gaussian, {"centre": np.array([0.7]), "width": 0.3}),
        (Power, {"centre": np.array([0.5]), "exponent": 1.0, "binsize": 0.5}),
        (Exponential, {"centre": np.array([0.0]), "decay_rate": 1.0}),
    ]
    x = np.linspace(-1, 1, 10000)
    for PDFClass, kwargs in pdfs_args_1D:
        print((PDFClass, kwargs))
        p = [PDFClass(**kwargs).pdf(np.array([v])) for v in x]
        plt.figure()
        plt.title(PDFClass.__name__)
        plt.plot(x, p)
    plt.show()
    plt.figure()
    a = Funky(centre=np.array([0.5, 0.5]), width=2.0)
    x = [np.sqrt(2) * i / 100.0 for i in range(200)]
    y = [a.pdf(np.array([i / 100.0, i / 100.0])) for i in range(200)]
    # y = [a.pdf(i/100) for i in range(100)]
    plt.plot(x, y)
    plt.title("funky")

    # 2D case
    pdfs_args_2D = [
        (Tophat_2D, {"extent": 0.7, "x_centre": 0.3, "y_centre": 0.1, "type_2D": 0}),
        (Gaussian, {"centre": np.array((0.0, 0.5)), "width": 1.0}),
        (
            Power,
            {
                "centre": np.array((0.5, -0.5)),
                "exponent": 0.2,
                "binsize": 0.8,
            },
        ),
        (
            Exponential,
            {
                "centre": np.array([0.5, -0.5]),
                "decay_rate": 0.5,
            },
        ),
    ]
    x_edges = np.linspace(-1, 1, 200)
    y_edges = np.linspace(-1, 1, 200)
    x_centres = get_centres(x_edges)
    y_centres = get_centres(y_edges)
    x, y = np.meshgrid(x_centres, y_centres)
    C = np.zeros_like(x, dtype=np.float64)
    for PDFClass, kwargs in pdfs_args_2D:
        print((PDFClass, kwargs))
        instance = PDFClass(**kwargs)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                C[i, j] = instance.pdf(np.array([x[i, j], y[i, j]]))
        plt.figure()
        plt.title(PDFClass.__name__)
        plt.pcolormesh(x, y, C)
        plt.gca().set_aspect("equal")
    plt.show()
