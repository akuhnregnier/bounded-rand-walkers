# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import norm


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
    def __init__(self, centre=0., exponent=1.,binsize=0.001):
        self.centre = centre
        self.exponent = exponent
        self.binsize = binsize
        
    def pdf(self, *args):
        
        if len(args) == 1:
            x = args[0]
            prob = 0.5*( (np.abs(x+self.binsize-self.centre))**(-self.exponent) / (self.binsize)**(1-self.exponent))
            
        elif len(args) == 2:
            x, y = args
            radius = (x+self.binsize-self.centre[0])**2 + (y+self.binsize-self.centre[1])**2
            prob = 0.5*( radius**(-self.exponent) / (self.binsize)**(1-self.exponent))
            
        return prob
            
class Gaussian(object):
    def __init__(self, centre=0., scale=1.):
        """A Gaussian distribution.

        Args:
            centre: Centre of the Gaussian. For a 2D distribution, give a
                list of x, y position (ie. [x, y]) of the centre of the
                Gaussian.
            scale: Used to scale the Gaussian.

        """
        self.centre = centre
        self.scale = scale

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
            prob = norm.pdf(x, loc=self.centre, scale=self.scale)
        elif len(args) == 2:
            x, y = args
            # symmetric in radius
            prob = norm.pdf(np.sqrt((x - self.centre[0])**2
                            + (y - self.centre[1])**2),
                            scale=self.scale)
        return prob


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # Tophat tests
    # 1D case
    x = np.linspace(-1, 1, 100)
    p = []
    for v in x:
        p.append(Tophat_1D(width=0.7, centre=0.3).pdf(v))
    plt.figure()
    plt.plot(x, p)
    plt.show()
    # 2D case
    x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    C = np.zeros_like(x, dtype=np.float64)
    for i in range(100):
        for j in range(100):
            C[i, j] = (Tophat_2D(x_centre=0.3, y_centre=-0.4, extent=0.6)
                       .pdf(x[i, j], y[i, j]))

    plt.figure()
    plt.pcolormesh(x, y, C)
    plt.gca().set_aspect('equal')
    plt.show()

    # Gaussian tests
    # 1D case
    x = np.linspace(-1, 1, 100)
    p = []
    for v in x:
        p.append(Gaussian(centre=0.7, scale=0.3).pdf(v))
    plt.figure()
    plt.plot(x, p)
    plt.show()
    # 2D case
    x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    C = np.zeros_like(x, dtype=np.float64)
    for i in range(100):
        for j in range(100):
            C[i, j] = (Gaussian(centre=(0., 0.5), scale=1.)
                       .pdf(x[i, j], y[i, j]))

    plt.figure()
    plt.pcolormesh(x, y, C)
    plt.gca().set_aspect('equal')
    plt.show()
