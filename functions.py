# -*- coding: utf-8 -*-
import numpy as np


def tophat_1D(x, width=0.4, centre=0.):
    """Tophat uniform pdf.

    The distribution is not normalised.
    The distribution has a total width ``extent`` in the 1D case,

    Args:
        x (float): In the 1D case, the x-values to
            calculate the function value for.
    """
    if abs(x - centre) < (width/2.):
        return 1.
    else:
        return 0.


def tophat_2D(x, y, extent=0.4, x_centre=0., y_centre=0.,
              type_2D='circularly-symmetric'):
    """Tophat uniform pdf.

    The distribution is not normalised.
    The distribution has a diameter ``extent`` in the 'circularly-symmetric'
    2D case, and a square shape with side length ``extent`` in the 'square'
    case.

    Args:
        x (float): In the 2D case, the (x, y) value for which to calculate
            the pdf value.
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
    if type_2D == 'circularly-symmetric':
        if ((x - x_centre)**2. + (y - y_centre)**2.) < (extent / 2.)**2.:
            return 1.
        else:
            return 0.
    elif type_2D == 'square':
        if ((abs(x - x_centre) < (extent / 2.))
                and (abs(y - y_centre) < (extent / 2.))):
            return 1.
        else:
            return 0.
    else:
        raise NotImplementedError('type_2D value {:} not implemented.'
                                  .format(type_2D))


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    # 1D case
    x = np.linspace(-1, 1, 100)
    p = []
    for v in x:
        p.append(tophat_1D(v, width=0.7, centre=0.3))
    plt.figure()
    plt.plot(x, p)
    plt.show()
    # 2D case
    x, y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    C = np.zeros_like(x, dtype=np.float64)
    for i in range(100):
        for j in range(100):
            C[i, j] = tophat_2D(x[i, j], y[i, j], x_centre=0.3,
                                y_centre=-0.4, extent=0.6)

    plt.figure()
    plt.pcolormesh(x, y, C)
    plt.gca().set_aspect('equal')
    plt.show()

