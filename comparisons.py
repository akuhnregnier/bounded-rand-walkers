#!/usr/bin/env python2
# -*- conding: utf-8 -*-
"""
Compare analytical and numerical stepsize and positions distributions.

"""
import logging
import matplotlib.pyplot as plt
import numpy as np
from rotation_steps import g1D, gRadialCircle
from functions import Tophat_1D, Tophat_2D
from binning import estimate_fi
from data_generation import random_walker, circle_points


def get_centres(bin_edges):
    left_edges = bin_edges[:-1]
    right_edges = bin_edges[1:]
    bin_centres = (left_edges + right_edges) / 2.
    return bin_centres


def compare_1D(pdf, analytical_bins, numerical_bins, num_samples=int(1e4),
               bounds=np.array([0, 1])):
    logger = logging.getLogger(__name__)
    # analytical result
    xs = np.linspace(0, 1, analytical_bins)
    g_analytical = []
    for x in xs:
        g_analytical.append(g1D(x, pdf))
    g_analytical = np.asarray(g_analytical)

    # numerical result
    step_values, positions = random_walker(
            f_i=pdf,
            bounds=bounds,
            steps=int(num_samples),
            return_positions=True,
            )
    logger.debug('{:} {:}'.format(step_values.shape, positions.shape))
    probs, bin_edges = np.histogram(positions, bins=numerical_bins,
                                    density=True)
    bin_centres = get_centres(bin_edges)
    return xs, g_analytical, bin_centres, probs


def compare_2D(pdf, analytical_bins, numerical_bins, num_samples=int(1e4),
               bounds=circle_points(samples=40)):
    logger = logging.getLogger(__name__)

    xs = np.linspace(-1, 1, analytical_bins, endpoint=True)
    ys = np.linspace(-1, 1, analytical_bins, endpoint=True)

    xcoords, ycoords = np.meshgrid(xs, ys)
    rads = np.sqrt(xcoords**2. + ycoords**2.)
    g_analytical = np.zeros_like(rads)
    unique_rads = np.unique(rads)
    nr_unique_rads = len(unique_rads)

    logger.info('integrating for {:} unique radii'
                .format(len(unique_rads)))
    for i, rad in enumerate(unique_rads):
        logger.info('{:>5d} out of {:>5d}'
                    .format(i + 1, nr_unique_rads))
        g_analytical_value = gRadialCircle(rad, pdf) / (2*np.pi*rad)
        mask = np.where(np.isclose(rad, rads))
        g_analytical[mask] = g_analytical_value

    logger.info('Finished analytical result, starting numerical run')

    # numerical result
    step_values, positions = random_walker(
            f_i=pdf,
            bounds=bounds,
            steps=int(num_samples),
            return_positions=True,
            )
    logger.info('Finished numerical run')
    logger.debug('{:} {:}'.format(step_values.shape, positions.shape))
    probs, xedges, yedges = np.histogram2d(*positions.T, bins=numerical_bins,
                                           normed=True)
    x_centres = get_centres(xedges)
    y_centres = get_centres(yedges)
    return xs, ys, g_analytical, x_centres, y_centres, probs


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    ONE_D = False
    TWO_D = True

    if ONE_D:
        # 1D case
        widths = [0.3, 0.7]
        for width in widths:
            pdf = Tophat_1D(width=width, centre=0.).pdf

            analytical_bins = 5
            numerical_bins = 5

            (analytical_bin_centres, g_analytical,
             numerical_bin_centres, g_numerical) = (
                compare_1D(pdf, analytical_bins, numerical_bins,
                           num_samples=int(1e4))
                )

            fig, axes = plt.subplots(1, 2, squeeze=True)
            axes[0].set_title(r'$Analytical g(x)$')
            axes[0].plot(analytical_bin_centres, g_analytical)
            axes[1].set_title(r'$Numerical g(x)$')
            axes[1].plot(numerical_bin_centres, g_numerical)
            plt.show()

    if TWO_D:
        # 2D case
        pdf = Tophat_2D(extent=20., x_centre=0, y_centre=0,
                        type_2D='circularly-symmetric').pdf

        analytical_bins = 40
        numerical_bins = 40

        (analytical_bin_centres_x, analytical_bin_centres_y, g_analytical,
         numerical_bin_centres_x, numerical_bin_centres_y, g_numerical) = (
                 compare_2D(pdf, analytical_bins, numerical_bins,
                            num_samples=int(1e5),
                            bounds=circle_points(samples=20)
                            ))

        fig, axes = plt.subplots(1, 2, squeeze=True)
        axes[0].set_title(r'$Analytical g(x)$')
        axes[0].pcolormesh(analytical_bin_centres_x,
                           analytical_bin_centres_y,
                           g_analytical
                           )
        axes[1].set_title(r'$Numerical g(x)$')
        axes[1].pcolormesh(numerical_bin_centres_x,
                           numerical_bin_centres_y,
                           g_numerical
                           )
        for ax in axes:
            ax.set_aspect('equal')
        plt.show()
