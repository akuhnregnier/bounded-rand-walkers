# -*- coding: utf-8 -*-
"""
Code for modifying sequence of positions via rotations to make the
asymmetry clear Valid for 2D case

"""
import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
from data_generation import in_bounds, DelaunayArray, weird_bounds
from scipy.spatial import Delaunay
from utils import get_centres
from numba import njit, jit
from c_rot_steps import rot_steps as c_rot_steps


@njit
def rot_steps_fast(data):
    """
    Feed data containing stopping positions in 2D - shape [2 , n-stops]

    Examples:
        >>> N = 1000
        >>> pos_data2D = rand.uniform(0, 1, size=(2, N))
        >>> rot_steps_data = rot_steps(pos_data2D)
        >>> plt.figure()
        >>> plt.hist2d(rot_steps[0, :], rot_steps[1, :], bins=50)
        >>> plt.plot(np.arange(-2, 2, 0.01),
        ...          np.array([0 for a in np.arange(-2, 2, 0.01)]))
        >>> plt.title('Observed step-size with fixed incoming direction')
        >>> plt.gca().set_aspect('equal')
        >>> plt.show()

    """
    print("This is rot_steps_fast")
    rot_steps_data = np.zeros((2, data.shape[1] - 2))

    for i in range(data.shape[1] - 2):

        a = data[:, i]
        b = data[:, i + 1]
        c = data[:, i + 2]

        # dot_prod = np.dot(a - b, np.array([0, -1]))
        left = a - b
        dot_prod = - left[1]

        phi = np.arccos(
                dot_prod / np.linalg.norm(a - b))

        if a[0] > b[0]:
            theta = -phi
        else:
            theta = phi

        # R = np.array([[np.cos(theta), -np.sin(theta)],
        #               [np.sin(theta), np.cos(theta)]],
        #              dtype=np.float64
        #              )

        R_0 = np.array([np.cos(theta), -np.sin(theta)])
        R_1 = np.array([np.sin(theta), np.cos(theta)])

        right = c - b
        c_rot = np.zeros((2,))
        c_rot[0] = np.sum(R_0 * right)
        c_rot[1] = np.sum(R_1 * right)

        rot_steps_data[:, i] = c_rot

    return rot_steps_data


def rot_steps(data):
    """
    Feed data containing stopping positions in 2D - shape [2 , n-stops]

    Examples:
        >>> N = 1000
        >>> pos_data2D = rand.uniform(0, 1, size=(2, N))
        >>> rot_steps_data = rot_steps(pos_data2D)
        >>> plt.figure()
        >>> plt.hist2d(rot_steps[0, :], rot_steps[1, :], bins=50)
        >>> plt.plot(np.arange(-2, 2, 0.01),
        ...          np.array([0 for a in np.arange(-2, 2, 0.01)]))
        >>> plt.title('Observed step-size with fixed incoming direction')
        >>> plt.gca().set_aspect('equal')
        >>> plt.show()

    """
    rot_steps_data = np.zeros([2, np.shape(data)[1] - 2])

    for i in range(np.shape(data)[1] - 2):

        a = data[:, i]
        b = data[:, i + 1]
        c = data[:, i + 2]

        phi = np.arccos(
            np.dot(a - b, np.array([0, -1])) / np.linalg.norm(a - b))

        if a[0] > b[0]:
            theta = -phi
        else:
            theta = phi

        R = np.matrix([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])

        # print(R.dot())
        c_rot = R.dot(c - b)

        rot_steps_data[:, i] = c_rot

    return rot_steps_data


def Corr_spatial_1D(data, binnumber):

    binwidth = 1. / binnumber
    print(binwidth)
    correlations = []

    for i in range(binnumber):

        prev = []
        post = []
        prod = []

        for j in range(np.shape(data)[1] - 2):
            if i * binwidth < data[0, j + 1] < (i + 1) * binwidth:

                prev.append(data[0, j + 1] - data[0, j])
                post.append(data[0, j + 2] - data[0, j + 1])
                prod.append((data[0, j + 1] - data[0, j]) *
                            (data[0, j + 2] - data[0, j + 1]))

        corr = np.mean(prod) - np.mean(post) * np.mean(prev)
        correlations.append(corr)

    print(correlations)
    plt.bar([binwidth * i for i in range(binnumber)],
            correlations, width=binwidth, alpha=0.4)


def Pdf_Transform(step, f, geometry):
    """
    For a given intrinsic step size pdf f gives probability p(stepsize) of
    transformed pdf Geometry is a str, choices atm '1Dseg' (takes float
    steps) & '1circle' (takes 2 element arrays)

    """
    if geometry == '1Dseg':
        if not isinstance(step, float):
            raise TypeError('for 1D pdf use float for step')

        return f(step) * (1 - np.abs(step)) * 0.5 * (np.sign(1 - np.abs(step)) + 1)

    if geometry == '1circle':
        if type(step) != np.ndarray:
            raise TypeError('for 2D pdf use 1d, 2 entry array, for step')

        l = np.linalg.norm(step)
        return f(step, 0) * (2 * np.arccos(l / 2) - 0.5 * np.sqrt((4 - l**2) * l**2))


def get_pdf_transform_shaper(steps, geometry):
    """
    For given intrinsic step sizes, get the shaper function at those radial
    distances from the centre.
    pdf f gives probability p(stepsize) of
    transformed pdf Geometry is a str, choices atm '1Dseg' (takes float
    steps) & '1circle' (takes 2 element arrays)

    """
    if geometry == '1Dseg':
        return (1 - np.abs(steps)) * 0.5 * (np.sign(1 - np.abs(steps)) + 1)

    if geometry == '1circle':
        shaper = np.zeros_like(steps, dtype=np.float64)
        mask = steps < 2.
        shaper[mask] = (2 * np.arccos(steps[mask] / 2)
                        - 0.5 * np.sqrt((4 - steps[mask]**2) * steps[mask]**2))
        return shaper


def g1D(x, f):
    num = integrate.quad(f, -x, 1 - x)
    return num[0]


def g1D_norm(f):
    den = integrate.dblquad(lambda z, y: f(
        z), 0, 1, lambda z: -z, lambda z: 1 - z)
    return den[0]


def betaCircle(r, l):
    return np.pi - np.arccos((r**2 + l**2 - 1)/(2*r*l))


def gRadialCircle(r, f):
    """
    Not yet normalised f is a function of radial distance from starting
    point of step (1D pdf)
    e.g. for a flat infinitely large top hat in 2D, the associated radial
        1D distribution goes as 1/l in which case we expect the probability for
        the position to be uniform within the circle, hence the radial one to
        grow linearly (as observed).

    """
    if np.isclose(r, 0.):
        return (
            sp.integrate.quad(lambda d: f(d,0)*2*np.pi*d, 0, 1-r)[0]
            )
    return (
        sp.integrate.quad(lambda d: f(d,0)*2*np.pi*d, 0, 1-r)[0]
         + sp.integrate.quad(
            lambda l: 2*np.pi*l*f(l,0)*(1-betaCircle(r, l)/np.pi), 1-r, 1+r)[0]
        )

def g2D(f, xs_edges, ys_edges, bounds=weird_bounds):
    """2D position probability."""
    print('G2D')
    bounds = DelaunayArray(bounds, Delaunay(bounds))
    xs_centres = get_centres(xs_edges)
    ys_centres = get_centres(ys_edges)
    g_values = np.zeros((xs_centres.shape[0], ys_centres.shape[0]),
                        dtype=np.float64)
    # should be True if the region is within the bounds
    position_mask = np.zeros_like(g_values, dtype=bool)
    for i, x in enumerate(xs_centres):
        for j, y in enumerate(ys_centres):
            is_in_bounds = in_bounds(np.array([x, y]), bounds)
            position_mask[i, j] = is_in_bounds
    x_indices, y_indices = np.where(position_mask)

    counter = 0
    max_counter = len(x_indices)
    for mask_x_index, mask_y_index in zip(x_indices, y_indices):
        # evaluate the pdf at each position relative to the current
        # positions. But only iterate over the positions that are
        # actually in the boundary.
        if max_counter < 20:
            print('counter:', counter + 1, 'out of:', max_counter)
        else:
            if ((counter) % (max_counter / 10)) == 0:
                print('{:07.2%}'.format(float(counter) / (max_counter - 1)))
        counter += 1
        x, y = (xs_centres[mask_x_index],
                ys_centres[mask_y_index])
        x_mod = xs_centres - x
        y_mod = ys_centres - y

        for mask_x_index, mask_y_index in zip(x_indices, y_indices):
            relative_position = (x_mod[mask_x_index],
                                 y_mod[mask_y_index])
            g_values[mask_x_index, mask_y_index] += f(*relative_position)

    cell_areas = ((xs_edges[1:] - xs_edges[:-1]).reshape(-1, 1)
                  * (ys_edges[1:] - ys_edges[:-1]).reshape(1, -1)
                  )
    total_prob = cell_areas * g_values
    g_mask = np.isclose(g_values, 0)
    # return the normalised g values
    g_values[~g_mask] /= np.sum(total_prob[~g_mask])
    return g_values


if __name__ == '__main__':
    # must be run from iPython!!
    from IPython import get_ipython
    ipython = get_ipython()

    d = np.random.randn(int(1e4), 2)
    assert np.all(np.isclose(rot_steps(d.T), rot_steps_fast(d.T)))
    assert np.all(np.isclose(rot_steps(d.T), c_rot_steps(d.T)))
    print("tests passed")
    print("rot_steps test")
    ipython.magic('time rot_steps(d.T)')
    print("rot_steps_fast test")
    ipython.magic('time rot_steps_fast(d.T)')
    print("c_rot_steps test")
    ipython.magic('time c_rot_steps(d.T)')


if __name__ == '__main__1':
    N = 100000
    pos_data2D = rand.uniform(0, 1, size=(2, N))
    rot_steps_data = rot_steps(pos_data2D)
    plt.figure()
    plt.hist2d(rot_steps_data[0, :], rot_steps_data[1, :], bins=50)
    plt.plot(np.arange(-2, 2, 0.01),
             np.array([0 for a in np.arange(-2, 2, 0.01)]))
    plt.title('Observed step-size with fixed incoming direction')
    plt.gca().set_aspect('equal')

    from functions import Funky

    pdf = Funky(centre=(0., 0.)).pdf
    bins = 81
    min_x = -1
    max_x = 1
    min_y = -1
    max_y = 1
    xs_edges = np.linspace(min_x, max_x, bins + 1)
    ys_edges = np.linspace(min_y, max_y, bins + 1)
    g_values = g2D(pdf, xs_edges, ys_edges)
    plt.figure()
    plt.pcolormesh(xs_edges, ys_edges, g_values)
    plt.gca().set_aspect('equal')
    plt.show()

