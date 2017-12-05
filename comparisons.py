#!/usr/bin/env python2
# -*- conding: utf-8 -*-
"""
Compare analytical and numerical stepsize and positions distributions.

"""
import logging
import matplotlib.pyplot as plt
import numpy as np
from rotation_steps import g1D, gRadialCircle, Pdf_Transform, rot_steps
from functions import Tophat_1D, Tophat_2D, Power, Exponential, Gaussian, Funky
import scipy.integrate
from data_generation import random_walker, circle_points
from statsmodels.stats.weightstats import DescrStatsW


def get_centres(bin_edges):
    left_edges = bin_edges[:-1]
    right_edges = bin_edges[1:]
    bin_centres = (left_edges + right_edges) / 2.
    return bin_centres


def stats(data1, data2, weights=None):
    """
    This function calculates the mean difference between the input data sets
    and the standard deviation of this mean.

    Args:
        data1: 1D array of dataset 1
        data2: 2D array of dataset 2
        weights: The wheights of each data point. The default are no weights.

    Returns:
        weighted_stats.mean: mean difference between data sets
        weighted_stats.std_mean: standard dev. of mean difference

    """
    if len(data1) != len(data2):
        raise Exception('Two data sets have different lengths')

    abs_difference = np.abs(data2 - data1)
    weighted_stats = DescrStatsW(abs_difference, weights=weights)

    return weighted_stats.mean, weighted_stats.std_mean


def compare_1D(pdf, nr_bins, num_samples=int(1e4),
               bounds=np.array([0, 1])):
    logger = logging.getLogger(__name__)
    # analytical result
    pos_bin_edges = np.linspace(0, 1, nr_bins + 1)
    pos_bin_centres = get_centres(pos_bin_edges)
    g_analytical = []
    for x in pos_bin_centres:
        g_analytical.append(g1D(x, pdf))

    step_bin_edges = np.linspace(-1, 1, nr_bins + 1)
    step_bin_centres = get_centres(step_bin_edges)
    f_t_analytical = []
    for x in step_bin_centres:
        f_t_analytical.append(Pdf_Transform(x, pdf, '1Dseg'))

    g_analytical = np.asarray(g_analytical)
    f_t_analytical = np.asarray(f_t_analytical)

    # numerical result
    step_values, positions = random_walker(
            f_i=pdf,
            bounds=bounds,
            steps=int(num_samples),
            )
    logger.debug('{:} {:}'.format(step_values.shape, positions.shape))
    g_numerical, pos_bin_edges = np.histogram(
            positions,
            bins=pos_bin_edges,
            density=True
            )
    f_t_numerical, step_bin_edges = np.histogram(
            step_values,
            bins=step_bin_edges,
            density=True
            )

    return (
        pos_bin_centres,
        g_analytical,
        g_numerical,
        step_bin_centres,
        f_t_analytical,
        f_t_numerical
        )


def compare_2D(pdf, nr_bins, num_samples=int(1e4),
               bounds=circle_points(samples=40)):
    logger = logging.getLogger(__name__)

    x_edges = np.linspace(-1, 1, nr_bins + 1, endpoint=True)
    y_edges = np.linspace(-1, 1, nr_bins + 1, endpoint=True)
    x_centres = get_centres(x_edges)
    y_centres = get_centres(y_edges)
    xcoords, ycoords = np.meshgrid(x_centres, y_centres)
    rads = np.sqrt(xcoords**2. + ycoords**2.)
    unique_rads = np.unique(rads)

    total_mask = np.zeros_like(rads, dtype=bool)

    g_analytical = np.zeros_like(rads)

    nr_unique_rads = len(unique_rads)
    logger.info('integrating for {:} unique radii'
                .format(len(unique_rads)))

    for i, rad in enumerate(unique_rads):
        logger.info('{:>5d} out of {:>5d}'
                    .format(i + 1, nr_unique_rads))
        g_analytical_value = gRadialCircle(rad, pdf)
        mask = np.where(np.isclose(rad, rads))
        if not np.isnan(g_analytical_value):
            total_mask |= np.isclose(rad, rads)
        g_analytical[mask] = g_analytical_value

    g_analytical /= scipy.integrate.quad(
            lambda x: (2*np.pi*x) * gRadialCircle(x, pdf), 0, 1
            )[0]

    g_analytical[~total_mask] = 0.

    ft_xs = np.linspace(-2, 2, nr_bins + 1, endpoint=True)
    ft_ys = np.linspace(-2, 2, nr_bins + 1, endpoint=True)
    ft_x_values = get_centres(ft_xs)
    ft_y_values = get_centres(ft_ys)
    ft_xcoords, ft_ycoords = np.meshgrid(ft_x_values, ft_y_values)
    ft_rads = np.sqrt(ft_xcoords**2. + ft_ycoords**2.)
    ft_total_mask = np.zeros_like(ft_rads, dtype=bool)
    f_t_analytical = np.zeros_like(ft_rads)
    ft_unique_rads = np.unique(ft_rads)

    for i, rad in enumerate(ft_unique_rads):
        mask = np.isclose(rad, ft_rads)
        f_t_analytical_value = Pdf_Transform(np.array([rad]), pdf, '1circle')
        f_t_analytical[mask] = f_t_analytical_value
        if not np.isnan(f_t_analytical_value):
            ft_total_mask |= mask

    f_t_analytical[~ft_total_mask] = 0.

    """
    do normalisation of the f_t_analytical probabilities
    sum of area * value should add up to 1 afterwards

    in this case, the cells have equal areas
    """
    cell_area = np.abs((ft_xs[1] - ft_xs[0]) * (ft_ys[1] - ft_ys[0]))
    total_integral = np.sum(f_t_analytical) * cell_area
    f_t_analytical /= total_integral

    logger.info('Finished analytical result, starting numerical run')

    # numerical result
    step_values, positions = random_walker(
            f_i=pdf,
            bounds=bounds,
            steps=int(num_samples),
            )
    logger.info('Finished numerical run')
    logger.debug('{:} {:}'.format(step_values.shape, positions.shape))
    g_numerical, _, _ = np.histogram2d(
            *positions.T, bins=[x_edges, y_edges],
            normed=True
            )

    f_t_numerical, _, _ = np.histogram2d(
            *step_values.T,
            bins=[ft_xs, ft_ys],
            normed=True
            )

    rot_steps_data = rot_steps(positions.T)
    rot_probs, _, _ = np.histogram2d(
        rot_steps_data[0, :],
        rot_steps_data[1, :],
        bins=[ft_xs, ft_ys],
        normed=True
        )

    return ((x_edges, y_edges),
            g_analytical,
            g_numerical,
            (ft_xs, ft_ys),
            f_t_analytical,
            f_t_numerical,
            rot_probs
            )


def compare_1D_plotting(pdf, nr_bins, steps=int(1e3)):

    (pos_bin_centres,
     g_analytical,
     g_numerical,
     step_bin_centres,
     f_t_analytical,
     f_t_numerical) = (
        compare_1D(pdf, nr_bins,
                   num_samples=steps)
        )

    print('g stats (mean, std)')
    print(stats(g_analytical, g_numerical))
    print('f_t stats (mean, std)')
    print(stats(f_t_analytical, f_t_numerical))

    fig, axes = plt.subplots(1, 2, squeeze=True)
    axes[0].set_title(r'$Analytical \ g(x)$')
    axes[0].plot(pos_bin_centres, g_analytical)
    axes[1].set_title(r'$Numerical \ g(x)$')
    axes[1].plot(pos_bin_centres, g_numerical)
    fig2, axes2 = plt.subplots(1, 2, squeeze=True)
    axes2[0].set_title(r'$Analytical \ f_t(x)$')
    axes2[0].plot(step_bin_centres,
                  f_t_analytical,
                  )
    axes2[1].set_title(r'$Numerical \ f_t(x)$')
    axes2[1].plot(step_bin_centres,
                  f_t_numerical,
                  )

    # plot figures on top of each other
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111)
    ax3.set_title(r'$Comparing \ Analytical \ and \ Numerical \ g(x)$')
    ax3.plot(pos_bin_centres, g_analytical, label='Analytical Solution')
    ax3.plot(pos_bin_centres, g_numerical, label='Numerical Solution')
    ax3.legend()

    fig4 = plt.figure()
    ax4 = fig4.add_subplot(111)
    ax4.set_title(r'$Comparing \ Analytical \ and \ Numerical \ f_t(x)$')
    ax4.plot(step_bin_centres, f_t_analytical, label='Analytical Solution')
    ax4.plot(step_bin_centres, f_t_numerical, label='Numerical Result')
    ax4.legend()

    plt.show()


def compare_2D_plotting(pdf, nr_bins, steps=int(1e3)):

    ((pos_x_edges, pos_y_edges),
     g_analytical,
     g_numerical,
     (ft_xs, ft_ys),
     f_t_analytical,
     f_t_numerical,
     rot_probs
     ) = (
        compare_2D(pdf, nr_bins,
                   num_samples=steps,
                   bounds=circle_points(samples=20)
                   )
        )

    print('g stats (mean, std)')
    print(stats(g_analytical.flatten(), g_numerical.flatten()))
    print('f_t stats (mean, std)')
    print(stats(f_t_analytical.flatten(), f_t_numerical.flatten()))

    """
    Plot of analytical and numerical g distributions
    """
    fig, axes = plt.subplots(1, 2, squeeze=True)
    fig.subplots_adjust(right=0.8)
    # use this max/min value with the hexbin vmax/vmin option
    # in order to have the same colour scaling for both
    # hexbin plots, such that the same colorbar may be used
    max_value = np.max([np.max(g_analytical[~np.isnan(g_analytical)]),
                        np.max(g_numerical[~np.isnan(g_numerical)])
                        ])
    min_value = np.min([np.min(g_analytical[~np.isnan(g_analytical)]),
                        np.min(g_numerical[~np.isnan(g_numerical)])
                        ])
    axes[0].set_title(r'$Analytical \ g(x, y)$')
    analytical_mesh = axes[0].pcolormesh(
                       pos_x_edges,
                       pos_y_edges,
                       g_analytical,
                       vmin=min_value,
                       vmax=max_value,
                       )
    axes[1].set_title(r'$Numerical \ g(x, y)$')
    numerical_mesh = axes[1].pcolormesh(
                       pos_x_edges,
                       pos_y_edges,
                       g_numerical,
                       vmin=min_value,
                       vmax=max_value,
                       )
    for ax in axes:
        ax.set_aspect('equal')
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(numerical_mesh, cax=cbar_ax)

    """
    Plot of analytical and numerical f_t distributions
    """
    fig, axes = plt.subplots(1, 2, squeeze=True)
    fig.subplots_adjust(right=0.8)
    # use this max/min value with the hexbin vmax/vmin option
    # in order to have the same colour scaling for both
    # hexbin plots, such that the same colorbar may be used
    max_value = np.max([np.max(f_t_analytical[~np.isnan(f_t_analytical)]),
                        np.max(f_t_numerical[~np.isnan(f_t_numerical)])
                        ])
    min_value = np.min([np.min(f_t_analytical[~np.isnan(f_t_analytical)]),
                        np.min(f_t_numerical[~np.isnan(f_t_numerical)])
                        ])
    axes[0].set_title(r'$Analytical \ f_t(x, y)$')
    analytical_mesh = axes[0].pcolormesh(
                       ft_xs,
                       ft_ys,
                       f_t_analytical,
                       vmin=min_value,
                       vmax=max_value,
                       )
    axes[1].set_title(r'$Numerical \ f_t(x, y)$')
    numerical_mesh = axes[1].pcolormesh(
                       ft_xs,
                       ft_ys,
                       f_t_numerical,
                       vmin=min_value,
                       vmax=max_value,
                       )
    for ax in axes:
        ax.set_aspect('equal')
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(numerical_mesh, cax=cbar_ax)

    """
    'teardrop' f_t plot
    """
    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.set_title(r'$Orientationally\  normalised\  f_t(x, y)$')
    rot_probs_plot = ax.pcolormesh(ft_xs, ft_ys, rot_probs.T)
    fig.colorbar(rot_probs_plot)
    ax.hlines((0.,), np.min(ft_xs), np.max(ft_xs),
              colors='b')
    ax.set_aspect('equal')

    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    ONE_D = False
    TWO_D = True

    if ONE_D:
        # 1D case
        pdfs_args_1D = [
                (Tophat_1D, {
                    'width': 0.3,
                    'centre': 0.
                    }),
                # (Gaussian, {'centre': 0.7, 'scale': 0.3}),
                # (Power, {
                #     'centre': 0.5,
                #     'exponent': 1.,
                #     'binsize': 0.5
                #     }),
                # (Exponential, {
                #     'centre': 0.,
                #     'decay_rate': 1.
                #     }),
                ]
        bins = 11
        for PDFClass, kwargs in pdfs_args_1D:
            pdf = PDFClass(**kwargs).pdf
            compare_1D_plotting(pdf, bins, steps=int(1e4))

    if TWO_D:
        # 2D case
        pdfs_args_2D = [
                # (Tophat_2D, {
                #     'x_centre': 0.,
                #     'y_centre': 0.,
                #     'extent': 1.2,
                #     'type_2D': 'circularly-symmetric'
                #     }),
                (Funky, {
                    'centre': (0., 0.),
                    'width': 2.
                    }),
                # (Tophat_2D, {
                #     'x_centre': 0.3,
                #     'y_centre': -0.4,
                #     'extent': 0.6,
                #     'type_2D': 'square'
                #     }),
                # (Gaussian, {'centre': (0., 0.5), 'scale': 1.}),
                # (Power, {
                #     'centre': (0.5, -0.5),
                #     'exponent': 0.2,
                #     'binsize': 0.8,
                #     }),
                # (Exponential, {
                #     'centre': (0.5, -0.5),
                #     'decay_rate': 0.5,
                #     }),
                ]
        bins = 61
        for PDFClass, kwargs in pdfs_args_2D:
            pdf = PDFClass(**kwargs).pdf

            compare_2D_plotting(pdf, bins, steps=int(1e3))

