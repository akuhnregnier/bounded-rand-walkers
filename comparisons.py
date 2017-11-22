#!/usr/bin/env python2
# -*- conding: utf-8 -*-
"""
Compare analytical and numerical stepsize and positions distributions.

"""
import logging
import matplotlib.pyplot as plt
import numpy as np
from rotation_steps import g1D, gRadialCircle, Pdf_Transform
from functions import Tophat_1D, Tophat_2D
import scipy.integrate
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

    ft_xs = np.linspace(-1, 1, analytical_bins)
    f_t_analytical = []
    for x in ft_xs:
        f_t_analytical.append(Pdf_Transform(x, pdf, '1Dseg'))

    g_analytical = np.asarray(g_analytical)
    g_analytical = np.asarray(g_analytical)

    # numerical result
    step_values, positions = random_walker(
            f_i=pdf,
            bounds=bounds,
            steps=int(num_samples),
            return_positions=True,
            )
    logger.debug('{:} {:}'.format(step_values.shape, positions.shape))
    probs, bin_edges = np.histogram(
            positions,
            bins=numerical_bins,
            density=True
            )
    step_probs, step_bin_edges = np.histogram(
            step_values,
            bins=numerical_bins,
            density=True
            )
    bin_centres = get_centres(bin_edges)
    step_bin_centres = get_centres(step_bin_edges)
    return (xs, g_analytical,
            ft_xs, f_t_analytical,
            bin_centres, probs,
            step_bin_centres, step_probs
            )


def compare_2D(pdf, analytical_bins, numerical_bins, num_samples=int(1e4),
               bounds=circle_points(samples=40)):
    logger = logging.getLogger(__name__)

    xs = np.linspace(-1, 1, analytical_bins, endpoint=True)
    ys = np.linspace(-1, 1, analytical_bins, endpoint=True)
    x_values = (xs[1:] + xs[:-1]) / 2.
    y_values = (ys[1:] + ys[:-1]) / 2.
    xcoords, ycoords = np.meshgrid(x_values, y_values)
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

    ft_xs = np.linspace(-2, 2, analytical_bins, endpoint=True)
    ft_ys = np.linspace(-2, 2, analytical_bins, endpoint=True)
    ft_x_values = (ft_xs[1:] + ft_xs[:-1]) / 2.
    ft_y_values = (ft_ys[1:] + ft_ys[:-1]) / 2.
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
            return_positions=True,
            )
    logger.info('Finished numerical run')
    logger.debug('{:} {:}'.format(step_values.shape, positions.shape))
    probs, xedges, yedges = np.histogram2d(*positions.T, bins=numerical_bins,
                                           normed=True)

    step_probs, step_xedges, step_yedges = np.histogram2d(
            *step_values.T,
            bins=numerical_bins,
            normed=True
            )

    return (xs, ys, g_analytical,
            ft_xs, ft_ys, f_t_analytical,
            xedges, yedges, probs,
            step_xedges, step_yedges, step_probs
            )

def compare_1D_plotting(pdf, analytical_bins,
        numerical_bins=None, steps=int(1e3)):

    if numerical_bins is None:
        numerical_bins == analytical_bins

    (analytical_bin_centres, g_analytical,
     analytical_ft_centres, f_t_analytical,
     numerical_position_bin_centres, g_numerical,
     numerical_f_t_bin_centres, f_t_numerical) = (
        compare_1D(pdf, analytical_bins, numerical_bins,
                   num_samples=steps)
        )

    fig, axes = plt.subplots(1, 2, squeeze=True)
    axes[0].set_title(r'$Analytical \ g(x)$')
    axes[0].plot(analytical_bin_centres, g_analytical)
    axes[1].set_title(r'$Numerical \ g(x)$')
    axes[1].plot(numerical_f_t_bin_centres, g_numerical)
    fig2, axes2 = plt.subplots(1, 2, squeeze=True)
    axes2[0].set_title(r'$Analytical \ f_t(x)$')
    axes2[0].plot(analytical_ft_centres,
                  f_t_analytical,
                  )
    axes2[1].set_title(r'$Numerical \ f_t(x)$')
    axes2[1].plot(numerical_f_t_bin_centres,
                  f_t_numerical,
                  )
    plt.show()


def compare_2D_plotting(pdf, analytical_bins, numerical_bins=None,
                        steps=int(1e3)):
    if numerical_bins is None:
        numerical_bins == analytical_bins

    (analytical_bin_centres_x, analytical_bin_centres_y, g_analytical,
     ft_analytical_x, ft_analytical_y, f_t_analytical,
     numerical_edges_x, numerical_edges_y, g_numerical,
     numerical_step_xedges, numerical_step_yedges, numerical_f_t) = (
             compare_2D(pdf, analytical_bins, numerical_bins,
                        num_samples=steps,
                        bounds=circle_points(samples=20)
                        ))

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
    axes[0].set_title(r'$Analytical \ g(x)$')
    analytical_mesh = axes[0].pcolormesh(
                       analytical_bin_centres_x,
                       analytical_bin_centres_y,
                       g_analytical,
                       vmin=min_value,
                       vmax=max_value,
                       )
    axes[1].set_title(r'$Numerical \ g(x)$')
    numerical_mesh = axes[1].pcolormesh(
                       numerical_edges_x,
                       numerical_edges_y,
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
                        np.max(numerical_f_t[~np.isnan(numerical_f_t)])
                        ])
    min_value = np.min([np.min(f_t_analytical[~np.isnan(f_t_analytical)]),
                        np.min(numerical_f_t[~np.isnan(numerical_f_t)])
                        ])
    axes[0].set_title(r'$Analytical \ f_t(x)$')
    analytical_mesh = axes[0].pcolormesh(
                       ft_analytical_x,
                       ft_analytical_y,
                       f_t_analytical,
                       vmin=min_value,
                       vmax=max_value,
                       )
    axes[1].set_title(r'$Numerical \ f_t(x)$')
    numerical_mesh = axes[1].pcolormesh(
                       numerical_step_xedges,
                       numerical_step_yedges,
                       numerical_f_t,
                       vmin=min_value,
                       vmax=max_value,
                       )
    for ax in axes:
        ax.set_aspect('equal')
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
    fig.colorbar(numerical_mesh, cax=cbar_ax)
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    ONE_D = False
    TWO_D = True

    if ONE_D:
        # 1D case
        widths = [0.7]
        for width in widths:
            pdf = Tophat_1D(width=width, centre=0.).pdf

            analytical_bins = 30
            numerical_bins = 30

            compare_1D_plotting(pdf, analytical_bins, numerical_bins,
                                steps=int(1e4))

    if TWO_D:
        # 2D case
        pdf = Tophat_2D(extent=20., x_centre=0, y_centre=0,
                        type_2D='circularly-symmetric').pdf

        analytical_bins = 60
        numerical_bins = 60

        compare_2D_plotting(pdf, analytical_bins, numerical_bins,
                            steps=int(1e4))
