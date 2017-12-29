#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm


def get_centres(edges):
    """Get bin centres from edges."""
    return (edges[1:] + edges[:-1]) / 2.


class Tophat_1D(object):
    def __init__(self, width=0.5, centre=0.):
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
        if abs(x - self.centre) < (self.width/2.):
            return 1.
        else:
            return 0.


class Tophat_2D(object):
    def __init__(self, extent=0.5, x_centre=0., y_centre=0.,
                 type_2D='circularly-symmetric'):
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
            type_2D (str): If ``type_2D is 'circularly-symmetric'``, the
                distribution is circular in the 2D plane.
                If ``type_2D is 'square'``, the
                distribution is square in the 2D plane.

        """
        self.extent = extent
        self.x_centre = x_centre
        self.y_centre = y_centre
        self.type_2D = type_2D

    def pdf(self, x, y):
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
        if self.type_2D == 'circularly-symmetric':
            if (((x - self.x_centre)**2. + (y - self.y_centre)**2.)
                    < (self.extent / 2.)**2.):
                return 1.
            else:
                return 0.
        elif self.type_2D == 'square':
            if ((abs(x - self.x_centre) < (self.extent / 2.))
                    and (abs(y - self.y_centre) < (self.extent / 2.))):
                return 1.
            else:
                return 0.
        else:
            raise NotImplementedError('type_2D value {:} not implemented.'
                                      .format(self.type_2D))


class Power(object):
    def __init__(self, centre=0., exponent=1., binsize=0.001):
        """A rotationally symmetric power law distribution.

        Args:
            centre: Center of the power law. For a 2D distribution, give a
                list of x, y position (ie. [x, y]) of the centre of the
                exponential.
            exponent: characteristic exponent of the probability decay.
            binsize= scale of UV cutoff, should have scale of hist binsize

        """
        self.centre = centre
        self.exponent = exponent
        self.binsize = binsize

    def pdf(self, *args):
        """Calculate the probability at a point.

        Args:
            *args: The coordinates of the points of interest. For the 1D
                case, call like pdf(x). In the 2D case, give coordinates
                like pdf(x, y).

        Returns:
            prob: probability at the given point

        """

        if len(args) == 1:
            x = args[0]
            prob = 0.5*(
                (np.abs(x-self.centre)+self.binsize)
                **(-self.exponent)
                / (self.binsize)**(1-self.exponent)
                )

        elif len(args) == 2:
            x, y = args
            radius = ((x - self.centre[0])**2
                      + (y - self.centre[1])**2)**0.5 + self.binsize
            prob = 0.5*(radius**(-self.exponent)
                        / (self.binsize)**(1-self.exponent))

        return prob


class Gaussian(object):
    def __init__(self, centre=0., width=1.):
        """A Gaussian distribution.

        Args:
            centre: Centre of the Gaussian. For a 2D distribution, give a
                list of x, y position (ie. [x, y]) of the centre of the
                Gaussian.
            width: Used to scale the Gaussian.

        """
        self.centre = centre
        self.width = width

    def pdf(self, *args):
        """Calculate the probability at a point.

        Args:
            *args: The coordinates of the points of interest. For the 1D
                case, call like pdf(x). In the 2D case, give coordinates
                like pdf(x, y).

        Returns:
            prob: probability at the given point

        """
        if len(args) == 1:
            x = args[0]
            prob = norm.pdf(x, loc=self.centre, scale=self.width)
        elif len(args) == 2:
            x, y = args
            # symmetric in radius
            prob = norm.pdf(np.sqrt((x - self.centre[0])**2
                            + (y - self.centre[1])**2),
                            scale=self.width)
        return prob

class Exponential(object):
    def __init__(self, centre=0., decay_rate=1.):
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

    def pdf(self, *args):
        """
        Calculates the probability at a point.

        Args:
            *args: The coordinates of the points of interest. For the 1D
                case, call like pdf(x). In the 2D case, give coordinates
                like pdf(x, y).

        Returns:
            prob: probability at the given point

        """

        if len(args) == 1:
            x = args[0]
            prob = self.decay_rate * np.exp(
                    - np.abs(x - self.centre) * self.decay_rate
                    )
        elif len(args) == 2:
            x, y = args
            # symmetric in radius
            prob = np.exp(
                     - np.sqrt((x - self.centre[0])**2
                            + (y - self.centre[1])**2)
                    * self.decay_rate
                    )
        return prob


class Funky(object):
    def __init__(self, centre=0., width=1.):
        """

        """
        self.centre = centre
        self.width = width
        self.frequency = 3.7
        self.grad = 1.

    def pdf(self, *args):

        if len(args) == 1:
            x = args[0]
            position = np.abs(x - self.centre)
            power_law = Power(centre=(2/3. * self.width), exponent=0.25)
            scale = power_law.pdf(2/3. * self.width)

            const1 = np.abs(np.sinc(1/3. * self.width * self.frequency)) * (1 + 5 * 1/3. * self.width)
            const2 = const1 * ( 1 + self.grad *  1/3. * self.width)

            if position == 0:
                prob = 1.
            elif position > 0 and position <= 1/3. * self.width:
                prob = np.abs(np.sinc(position * self.frequency))

                # multiply by linear factor
                prob = prob * ( 1 + 5 * position)

            elif position > 1/3. * self.width and position <= 2/3. * self.width:
                prob = const1 * ( 1 + self.grad * (position - 1/3. * self.width))
            elif position > 2/3. * self.width:
                prob = power_law.pdf(position) / scale * const2

        if len(args) == 2:
            x,y = args
            position = np.sqrt((x - self.centre[0])**2 + (y - self.centre[1])**2)
            power_law = Power(centre=(2/3. * self.width), exponent=0.25)
            scale = power_law.pdf(2/3. * self.width)

            const1 = np.abs(np.sinc(1/3. * self.width * self.frequency)) * (1 + 5 * 1/3. * self.width)
            const2 = const1 * ( 1 + self.grad *  1/3. * self.width)

            if position == 0:
                prob = 1.
            elif position > 0 and position <= 1/3. * self.width:
                prob = np.abs(np.sinc(position * self.frequency))

                 # multiply by linear factor
                prob = prob * ( 1 + 5 * position)

            elif position > 1/3. * self.width and position <= 2/3. * self.width:
                prob = const1 * ( 1 + self.grad * (position - 1/3. * self.width))
            elif position > 2/3. * self.width:
                prob = power_law.pdf(position) / scale * const2

        return prob

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 1D case
    pdfs_args_1D = [
            (Tophat_1D, {'width': 0.7, 'centre': 0.3}),
            (Gaussian, {'centre': 0.7, 'scale': 0.3}),
            (Power, {
                'centre': 0.5,
                'exponent': 1.,
                'binsize': 0.5
                }),
            (Exponential, {
                'centre': 0.,
                'decay_rate': 1.
                }),
            ]
    x = np.linspace(-1, 1, 10000)
    #for PDFClass, kwargs in pdfs_args_1D:
    ##    p = [PDFClass(**kwargs).pdf(v) for v in x]
    #    plt.figure()
    #    plt.title(PDFClass.__name__)
    #    plt.plot(x, p)
    #plt.show()
    a = Funky([0.5,0.5], width=2.)
    x = [np.sqrt(2) * i/100. for i in range(200)]
    y = [a.pdf(*[i/100., i/100.]) for i in range(200)]
    #y = [a.pdf(i/100) for i in range(100)]
    plt.plot(x,y)
'''
    # 2D case
    pdfs_args_2D = [
            (Tophat_2D, {
                'x_centre': 0.3,
                'y_centre': -0.4,
                'extent': 0.6,
                'type_2D': 'circularly-symmetric'
                }),
            (Tophat_2D, {
                'x_centre': 0.3,
                'y_centre': -0.4,
                'extent': 0.6,
                'type_2D': 'square'
                }),
            (Gaussian, {'centre': (0., 0.5), 'scale': 1.}),
            (Power, {
                'centre': (0.5, -0.5),
                'exponent': 0.2,
                'binsize': 0.8,
                }),
            (Exponential, {
                'centre': (0.5, -0.5),
                'decay_rate': 0.5,
                }),
            ]
    x_edges = np.linspace(-1, 1, 200)
    y_edges = np.linspace(-1, 1, 200)
    x_centres = get_centres(x_edges)
    y_centres = get_centres(y_edges)
    x, y = np.meshgrid(x_centres, y_centres)
    C = np.zeros_like(x, dtype=np.float64)
    for PDFClass, kwargs in pdfs_args_2D:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                C[i, j] = (PDFClass(**kwargs)
                           .pdf(x[i, j], y[i, j]))
        plt.figure()
        plt.title(PDFClass.__name__)
        plt.pcolormesh(x, y, C)
        plt.gca().set_aspect('equal')
    plt.show()
'''
