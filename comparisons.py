#!/usr/bin/env python2
# -*- conding: utf-8 -*-
"""
Compare analytical and numerical stepsize and positions distributions.

"""
import logging
import os
import numpy as np
import matplotlib as mpl
import matplotlib.colors as colors
try:
    import cPickle as pickle
except ImportError:
    import pickle

N_PROCESSES = 1
SHOW = True
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rc('text', usetex=True)
mpl.rc('font', family='serif', size=15)
if not SHOW:
    mpl.use('Agg')
import matplotlib.pyplot as plt

from rotation_steps import (g1D, gRadialCircle, Pdf_Transform,
                            rot_steps, g1D_norm, g2D,
                            get_pdf_transform_shaper)
from functions import Tophat_1D, Tophat_2D, Power, Exponential, Gaussian, Funky
from data_generation import multi_random_walker, circle_points, weird_bounds
from utils import get_centres, stats
from shaperGeneral2D import genShaper
from shaperGeneral2D import get_weird_shaper
from rad_interp import radial_interp


output_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__),
    'output'
    ))
if not os.path.isdir(output_dir):
    os.makedirs(output_dir)


def compare_1D(pdf, nr_bins, num_samples=int(1e4),
               bounds=np.array([0, 1]),
               pdf_name='tophat',
               pdf_kwargs={'test': 10},
               load=True):
    logger = logging.getLogger(__name__)
    logger.info('Starting 1D')

    pickle_name = ('1D_compare_data_{:}_{:}_{:}_{:}.pickle'
                   .format(pdf_name, pdf_kwargs, nr_bins,
                           num_samples))
    pickle_path = os.path.join(output_dir, pickle_name)
    if os.path.isfile(pickle_path) and load:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        logger.info('read pickle data from {:}'.format(pickle_path))
        return data
    else:
        logger.info('not loading {:}'.format(pickle_path))

    # analytical result
    pos_bin_edges = np.linspace(0, 1, nr_bins + 1)
    pos_bin_centres = get_centres(pos_bin_edges)
    g_analytical = []
    logger.debug('Getting analytical g')
    for x in pos_bin_centres:
        g_analytical.append(g1D(x, pdf))
    logger.debug('Got analytical g')

    g_analytical = np.asarray(g_analytical)
    # now normalise g_analytical
    norm_const = g1D_norm(pdf)
    g_analytical /= norm_const
    logger.debug('normalised analytical g')

    step_bin_edges = np.linspace(-1, 1, nr_bins + 1)
    step_bin_centres = get_centres(step_bin_edges)
    f_t_analytical = []
    logger.debug('getting analytical f_t')
    shaper = get_pdf_transform_shaper(step_bin_centres, '1Dseg')
    f_i_analytical = np.array(
            [pdf(x_coord) for x_coord in step_bin_centres],
            dtype=np.float64
            )
    f_t_analytical = f_i_analytical * shaper
    logger.debug('got analytical f_t')

    f_t_analytical = np.asarray(f_t_analytical)

    # normalise f_t_analytical
    # using the bins specified in ``step_bin_edges``
    f_t_analytical /= np.sum(f_t_analytical
                             * (step_bin_edges[1:]
                                - step_bin_edges[:-1])
                             )

    # numerical result
    step_values, positions = multi_random_walker(
            n_processes=N_PROCESSES,
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

    data = (
        pos_bin_centres,
        g_analytical,
        g_numerical,
        step_bin_centres,
        f_t_analytical,
        f_t_numerical
        )

    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f, protocol=-1)
    logger.info('wrote pickle data to {:}'.format(pickle_path))

    return data


def compare_2D(pdf, nr_bins, num_samples=int(1e4),
               bounds=circle_points(samples=40),
               bounds_name='circle',
               pdf_name='tophat',
               pdf_kwargs={'test': 10},
               load=True):

    logger = logging.getLogger(__name__)

    pickle_name = ('2D_compare_data_{:}_{:}_{:}_{:}_{:}.pickle'
                   .format(pdf_name, pdf_kwargs, bounds_name,
                           nr_bins, num_samples))
    pickle_path = os.path.join(output_dir, pickle_name)
    if os.path.isfile(pickle_path) and load:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        logger.info('read pickle data from {:}'.format(pickle_path))
        return data
    else:
        logger.info('not loading {:}'.format(pickle_path))

    x_edges = np.linspace(-1, 1, nr_bins + 1, endpoint=True)
    y_edges = np.linspace(-1, 1, nr_bins + 1, endpoint=True)
    x_centres = get_centres(x_edges)
    y_centres = get_centres(y_edges)
    xcoords, ycoords = np.meshgrid(x_centres, y_centres)

    g_analytical = g2D(pdf, x_edges, y_edges, bounds=bounds)

    ft_xs = np.linspace(-2, 2, nr_bins + 1, endpoint=True)
    ft_ys = np.linspace(-2, 2, nr_bins + 1, endpoint=True)
    ft_x_values = get_centres(ft_xs)
    ft_y_values = get_centres(ft_ys)
    ft_xcoords, ft_ycoords = np.meshgrid(ft_x_values, ft_y_values)
    ft_rads = np.sqrt(ft_xcoords**2. + ft_ycoords**2.)
    f_t_analytical = np.zeros_like(ft_rads)

    # Analytical: multiply f_i with shaper to get f_t
    ft_total_mask = np.zeros_like(ft_rads, dtype=bool)
    ft_unique_rads = np.unique(ft_rads)

    for rad in ft_unique_rads:
        mask = np.isclose(rad, ft_rads)
        f_t_analytical_value = pdf(rad, 0)
        f_t_analytical[mask] = f_t_analytical_value
        if not np.isnan(f_t_analytical_value):
            ft_total_mask |= mask

    f_t_analytical[~ft_total_mask] = 0.

    if bounds_name == "weird":
        shaper = get_weird_shaper(ft_x_values, ft_y_values)
    else:
        shaper = np.zeros_like(f_t_analytical, dtype=np.float64)
        rad_shaper_values = get_pdf_transform_shaper(ft_unique_rads, '1circle')
        for rad, shaper_value in zip(ft_unique_rads, rad_shaper_values):
            mask = np.isclose(rad, ft_rads)
            shaper[mask] = shaper_value
            if not np.isnan(shaper_value):
                ft_total_mask |= mask

        f_t_analytical[~ft_total_mask] = 0.

    f_t_analytical *= shaper

    """
    do normalisation of the f_t_analytical probabilities
    sum of area * value should add up to 1 afterwards

    in this case, the cells have equal areas
    """
    cell_area = np.abs((ft_xs[1] - ft_xs[0]) * (ft_ys[1] - ft_ys[0]))
    total_p = np.sum(f_t_analytical) * cell_area
    f_t_analytical /= total_p

    logger.info('Finished analytical result, starting numerical run')

    # numerical result
    step_values, positions = multi_random_walker(
            n_processes=N_PROCESSES,
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

    # normalise g_numerical
    g_mask = np.isclose(g_numerical, 0)  # avoid dividing by 0
    g_numerical[~g_mask] /= np.sum(
            (g_numerical
             * ((x_edges[1:] - x_edges[:-1]).reshape(-1, 1))
                * (y_edges[1:] - y_edges[:-1]).reshape(1, -1)
             )[~g_mask]
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

    # try radial calculation
    avg, radii = radial_interp(f_t_numerical, ft_x_values, ft_y_values,
                               400, 200, dtype='float')

    data = ((x_edges, y_edges),
            g_analytical,
            g_numerical,
            (ft_xs, ft_ys),
            f_t_analytical,
            f_t_numerical,
            rot_probs,
            radii,
            avg
            )

    with open(pickle_path, 'wb') as f:
        pickle.dump(data, f, protocol=-1)
    logger.info('wrote pickle data to {:}'.format(pickle_path))

    return data


def compare_1D_plotting(pdf, nr_bins, steps=int(1e3), pdf_name='tophat',
                        pdf_kwargs={'test': 10}, load=True):

    (pos_bin_centres,
     g_analytical,
     g_numerical,
     step_bin_centres,
     f_t_analytical,
     f_t_numerical) = (
        compare_1D(pdf, nr_bins,
                   num_samples=steps,
                   pdf_name=pdf_name,
                   load=load,
                   pdf_kwargs=pdf_kwargs)
        )

    print('g stats (mean, std)')
    print(stats(g_analytical, g_numerical))
    print('f_t stats (mean, std)')
    print(stats(f_t_analytical, f_t_numerical))

    fig1, axes = plt.subplots(1, 2, squeeze=True)
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

    # save all figures
    suffix = ('{:} {:} {:} {:}.png'
              .format(pdf_name, pdf_kwargs, nr_bins, steps))
    name = '1D analytical vs numerical g ' + suffix
    fig1.savefig(os.path.join(output_dir, name))
    name = '1D analytical vs numerical f_t ' + suffix
    fig2.savefig(os.path.join(output_dir, name))
    name = '1D overplot analytical vs numerical g ' + suffix
    fig3.savefig(os.path.join(output_dir, name))
    name = '1D overplot analytical vs numerical f_t ' + suffix
    fig4.savefig(os.path.join(output_dir, name))

    if SHOW:
        plt.show()
    else:
        plt.close('all')


def compare_2D_plotting(pdf, nr_bins, steps=int(1e3),
                        bounds=circle_points(samples=20),
                        bounds_name='circle', pdf_name='tophat',
                        pdf_kwargs={'test': 10},
                        load=True):

    ((pos_x_edges, pos_y_edges),
     g_analytical,
     g_numerical,
     (ft_xs, ft_ys),
     f_t_analytical,
     f_t_numerical,
     rot_probs,
     radii,
     avg
     ) = (
        compare_2D(pdf, nr_bins,
                   num_samples=steps,
                   bounds=bounds,
                   bounds_name=bounds_name,
                   pdf_name=pdf_name,
                   load=load,
                   pdf_kwargs=pdf_kwargs
                   )
        )

    print('g stats (mean, std)')
    print(stats(g_analytical.flatten(), g_numerical.flatten()))
    print('f_t stats (mean, std)')
    print(stats(f_t_analytical.flatten(), f_t_numerical.flatten()))

    """
    Plot of analytical and numerical g distributions
    """
    fig1, axes = plt.subplots(1, 2, squeeze=True)
    fig1.subplots_adjust(right=0.8)
    # rewrite the values close to 0 to nan, such that they are ignored when
    # performing the colormapping.
    analytical_mask = g_analytical < 1e-2
    numerical_mask = g_numerical < 1e-2
    g_analytical[analytical_mask] = np.nan
    g_numerical[numerical_mask] = np.nan

    # get the bounding box from the masks defined above
    overall_mask = ~analytical_mask | ~numerical_mask
    mask_x_indices, mask_y_indices = np.where(overall_mask)
    x_lims = (np.min(mask_x_indices), np.max(mask_x_indices))
    y_lims = (np.min(mask_y_indices), np.max(mask_y_indices))

    # use this max/min value with the hexbin vmax/vmin option
    # in order to have the same colour scaling for both
    # hexbin plots, such that the same colorbar may be used
    max_value = np.max([np.max(g_analytical[~np.isnan(g_analytical)]),
                        np.max(g_numerical[~np.isnan(g_numerical)])
                        ])
    min_value = np.min([np.min(g_analytical[~np.isnan(g_analytical)]),
                        np.min(g_numerical[~np.isnan(g_numerical)])
                        ])
    axes[0].set_title(r'Analytical $g(x, y)$')

    analytical_mesh = axes[0].pcolormesh(
            pos_x_edges[x_lims[0]: x_lims[1] + 2],
            pos_y_edges[y_lims[0]: y_lims[1] + 2],
            g_analytical[x_lims[0]: x_lims[1] + 1,
                         y_lims[0]: y_lims[1] + 1].T,
            norm=colors.PowerNorm(gamma=2.),
            vmin=min_value,
            vmax=max_value,
            )
    axes[1].set_title(r'Numerical $g(x, y)$')
    numerical_mesh = axes[1].pcolormesh(
            pos_x_edges[x_lims[0]: x_lims[1] + 2],
            pos_y_edges[y_lims[0]: y_lims[1] + 2],
            g_numerical[x_lims[0]: x_lims[1] + 1,
                        y_lims[0]: y_lims[1] + 1].T,
            norm=colors.PowerNorm(gamma=2.),
            vmin=min_value,
            vmax=max_value,
            )
    for ax in axes:
        ax.set_aspect('equal')
    cbar_ax = fig1.add_axes([0.85, 0.15, 0.02, 0.7])

    axes[0].set_ylabel('y')
    axes[0].set_xlabel('x')
    axes[1].set_xlabel('x')

    fig1.colorbar(numerical_mesh, cax=cbar_ax, label='P(x,y)')

    """
    Plot of analytical and numerical f_t distributions
    """
    fig2, axes = plt.subplots(1, 2, squeeze=True)
    fig2.subplots_adjust(right=0.8)
    # use this max/min value with the hexbin vmax/vmin option
    # in order to have the same colour scaling for both
    # hexbin plots, such that the same colorbar may be used
    max_value = np.max([np.max(f_t_analytical[~np.isnan(f_t_analytical)]),
                        np.max(f_t_numerical[~np.isnan(f_t_numerical)])
                        ])
    min_value = np.min([np.min(f_t_analytical[~np.isnan(f_t_analytical)]),
                        np.min(f_t_numerical[~np.isnan(f_t_numerical)])
                        ])
    axes[0].set_title(r'Analytical $f_t(x, y)$')
    analytical_mesh = axes[0].pcolormesh(
                       ft_xs,
                       ft_ys,
                       f_t_analytical.T,
                       norm=colors.PowerNorm(gamma=0.75),
                       vmin=min_value,
                       vmax=max_value,
                       )
    axes[1].set_title(r'Numerical $f_t(x, y)$')
    numerical_mesh = axes[1].pcolormesh(
                       ft_xs,
                       ft_ys,
                       f_t_numerical.T,
                       norm=colors.PowerNorm(gamma=0.75),
                       vmin=min_value,
                       vmax=max_value,
                       )
    for ax in axes:
        ax.set_aspect('equal')
    cbar_ax = fig2.add_axes([0.85, 0.15, 0.02, 0.7])

    axes[0].set_ylabel('y (step size)')
    axes[0].set_xlabel('x (step size)')
    axes[1].set_xlabel('x (step size)')

    fig2.colorbar(numerical_mesh, cax=cbar_ax, label='P(x,y)')

    """
    'teardrop' f_t plot
    """
    fig3, ax = plt.subplots(1, 1, squeeze=True)
    ax.set_title(r'Orientationally normalised $f_t(x, y)$')
    rot_probs_plot = ax.pcolormesh(
            ft_xs,
            ft_ys,
            rot_probs.T,
            norm=colors.PowerNorm(gamma=0.5)
            )
    fig3.colorbar(rot_probs_plot, label='P(x,y)')
    ax.hlines((0.,), np.min(ft_xs), np.max(ft_xs),
              colors='b')
    ax.set_aspect('equal')
    ax.set_ylabel('y (step size)')
    ax.set_xlabel('x (step size)')

    """
    Plot Malte's most incredible, astonishing, brilliant, amazong, Im running
    out of adjacctives, cool radial function stuff
    """
    fig4, axis = plt.subplots(1,1, squeeze=True)
    plt.plot(radii, avg)
    plt.title(r'Average Radial Distribution of the numerical $f_t$')
    plt.xlabel('x (step size)')
    plt.ylabel('y (step size)')

    # save all figures
    suffix = ('{:} {:} {:} {:} {:}.png'
              .format(pdf_name, pdf_kwargs, bounds_name,
                      nr_bins, steps)
              )
    name = '2D analytical vs numerical g ' + suffix
    fig1.savefig(os.path.join(output_dir, name))
    name = '2D analytical vs numerical f_t ' + suffix
    fig2.savefig(os.path.join(output_dir, name))
    name = '2D orientationally normalised f_t ' + suffix
    fig3.savefig(os.path.join(output_dir, name))
    name = '2D cross section f_t ' + suffix
    fig4.savefig(os.path.join(output_dir, name))

    if SHOW:
        plt.show()
    else:
        plt.close('all')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    ONE_D = False
    TWO_D = True

    if ONE_D:
        # 1D case
        pdfs_args_1D = [
                (Gaussian, 'gauss', {
                    'scale': 2.0,
                    'centre': 0.
                    }),
                (Gaussian, 'gauss', {
                    'scale': 1.5,
                    'centre': 0.
                    }),
                (Gaussian, 'gauss', {
                    'scale': 1.0,
                    'centre': 0.
                    }),
                (Gaussian, 'gauss', {
                    'scale': 0.5,
                    'centre': 0.
                    }),
                ]
        bins = 31
        for PDFClass, pdf_name, kwargs in pdfs_args_1D:
            pdf = PDFClass(**kwargs).pdf
            compare_1D_plotting(pdf, bins, steps=int(1e6),
                                pdf_name=pdf_name,
                                pdf_kwargs=kwargs)

    if TWO_D:
        # 2D case

        pdfs_args_2D = [
                (Funky, 'funky', {
                    'centre': (0., 0.),
                    'width': 2.
                    }),
                ]

        bins = 61
        for PDFClass, pdf_name, kwargs in pdfs_args_2D:
            pdf = PDFClass(**kwargs).pdf

            compare_2D_plotting(pdf, bins, steps=int(1e3),
                                pdf_name=pdf_name,
                                pdf_kwargs=kwargs,
                                bounds=weird_bounds,
                                bounds_name='weird',
                                load=False)

