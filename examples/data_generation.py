# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from bounded_rand_walkers.cpp import (
    bound_map,
    funky,
    generate_data,
    get_cached_filename,
)
from bounded_rand_walkers.rad_interp import exact_radii_interp, inv_exact_radii_interp
from bounded_rand_walkers.relief_matrix_shaper import gen_shaper2D
from bounded_rand_walkers.rotation_steps import get_pdf_transform_shaper
from bounded_rand_walkers.utils import cache_dir, get_centres, normalise

if __name__ == "__main__":
    # Generate a single set of data.

    pdf_kwargs = dict(width=2.0)

    def get_f_i(r):
        """Calculate f_i for given radii."""
        return np.array([funky([c, 0], **pdf_kwargs) for c in r])

    bound_name = "square"

    data_kwargs = dict(
        cache_dir=cache_dir,
        samples=int(1e7),
        seed=np.arange(10),
        blocks=2,
        bound_name=bound_name,
        **pdf_kwargs
    )

    filenames = get_cached_filename(squeeze=False, **data_kwargs)
    if not all(p.is_file() for p in filenames):
        generate_data(squeeze=False, max_workers=2, cache_only=True, **data_kwargs)

    n_bins = 200

    (
        g_x_edges,
        g_y_edges,
        g_x_centres,
        g_y_centres,
        f_t_x_edges,
        f_t_y_edges,
        f_t_x_centres,
        f_t_y_centres,
        f_t_r_edges,
        f_t_r_centres,
        g_numerical,
        f_t_numerical,
        f_t_r_numerical,
    ) = get_binned_data(filenames=filenames, n_bins=n_bins)

    # avg_f_t_num_radii, avg_f_t_num = radial_interp(
    #     data=f_t_numerical, x_centres=f_t_x_centres, y_centres=f_t_y_centres
    # )

    # avg_f_t_num_radii2, avg_f_t_num2 = exact_radii_interp(
    #     data=f_t_numerical, x_centres=f_t_x_centres, y_centres=f_t_y_centres
    # )

    # saved = np.load(filenames[0])
    # steps = saved["steps"]
    # saved.close()

    # avg_r_rad, avg_r_num = radial_interp_circ(steps, 100)

    # axes[4].plot(avg_r_rad, avg_r_num / np.max(avg_r_num), label="1D circ",
    #         linestyle='--')

    # axes[4].plot(avg_f_t_num_radii, avg_f_t_num / np.max(avg_f_t_num), label="2D circ")

    # axes[4].plot(
    #     avg_f_t_num_radii2,
    #     avg_f_t_num2 / np.max(avg_f_t_num2),
    #     label="2D circ non interp",
    # )

    f_i_r_analytical = get_f_i(f_t_r_centres)

    # Calculate shaper function.
    f_radii = np.linalg.norm(
        np.meshgrid(f_t_x_centres, f_t_y_centres, indexing="ij"), axis=0
    )
    f_i_analytical = np.zeros_like(f_radii)

    # Analytical: multiply f_i with shaper to get f_t.
    f_i_total_mask = np.zeros_like(f_radii, dtype=np.bool_)
    f_unique_radii = np.unique(f_radii)

    f_i_analytical = inv_exact_radii_interp(
        f_unique_radii,
        get_f_i(f_unique_radii),
        f_t_x_centres,
        f_t_y_centres,
        normalisation="none",
    )

    order_divisions = 100  # Bump to 400 does not improve things visibly.

    if bound_name == "square":
        vertices = bound_map[bound_name]()
        raw_shaper_X, raw_shaper_Y, raw_shaper = gen_shaper2D(order_divisions, vertices)

        x0 = y0 = 2
        divisions_x = order_divisions
        divisions_y = divisions_x
        # divisions_y = order_divisions * int(float(y0) / float(x0))

        interp = RegularGridInterpolator(
            (
                get_centres(np.linspace(-x0, x0, divisions_x + 1)),
                get_centres(np.linspace(-y0, y0, divisions_y + 1)),
            ),
            raw_shaper,
            method="linear",
            bounds_error=False,
            fill_value=0.0,
        )
        f_t_X_grid, f_t_Y_grid = np.meshgrid(
            f_t_x_centres, f_t_y_centres, indexing="ij"
        )
        interp_shaper = interp(
            np.hstack((f_t_X_grid.ravel()[:, None], f_t_Y_grid.ravel()[:, None]))
        ).reshape(f_t_X_grid.shape)
    elif bound_name == "circle":
        shaper = np.zeros_like(f_i_analytical, dtype=np.float64)
        radial_shaper_values = get_pdf_transform_shaper(f_unique_radii, "circle")

        interp_shaper = inv_exact_radii_interp(
            f_unique_radii,
            radial_shaper_values,
            f_t_x_centres,
            f_t_y_centres,
            normalisation="none",
        )

    f_t_analytical = f_i_analytical * interp_shaper

    # Reconstruct f_i from the numerics and the shaper function (2D).
    f_i_numerical = np.ma.MaskedArray(np.zeros_like(f_t_numerical), mask=True)
    valid_radii = ~np.isclose(interp_shaper, 0)
    f_i_numerical[valid_radii] = f_t_numerical[valid_radii] / interp_shaper[valid_radii]

    f_i_r_num_radii, f_i_r_num = exact_radii_interp(
        data=f_i_numerical,
        x_centres=f_t_x_centres,
        y_centres=f_t_y_centres,
        normalisation="none",
    )
    valid_f_i_r_num = ~np.isnan(f_i_r_num)
    f_i_r_num_radii = f_i_r_num_radii[valid_f_i_r_num]
    f_i_r_num = f_i_r_num[valid_f_i_r_num]

    f_t_analytical_radii, f_t_analytical_num = exact_radii_interp(
        f_t_analytical, f_t_x_centres, f_t_y_centres
    )

    # # Visualise the data.
    # fig, axes = plt.subplots(2, 3, figsize=(15, 5))
    # axes = axes.flatten()

    # axes[0].set_title("Positions")
    # axes[0].pcolormesh(g_x_edges, g_y_edges, g_numerical)

    # axes[1].set_title("Steps")
    # axes[1].pcolormesh(f_t_x_edges, f_t_y_edges, f_t_numerical)

    # axes[2].set_title("f_t_analytical")
    # axes[2].pcolormesh(f_t_x_edges, f_t_y_edges, f_t_analytical)

    # axes[5].set_title("f_i_analytical")
    # axes[5].pcolormesh(f_t_x_edges, f_t_y_edges, f_i_analytical)

    # f_ta_selection = f_t_r_centres < 1.8
    # axes[3].plot(
    #     f_t_r_centres[f_ta_selection], normalise(f_t_r_centres[f_ta_selection], f_i_r_analytical[f_ta_selection]), label="analytical f_i",
    #     zorder=2,
    #     linestyle='--',
    # )

    # f_t_selection = f_i_r_num_radii < 1.8
    # axes[3].plot(
    #     f_i_r_num_radii[f_t_selection],
    #     normalise(f_i_r_num_radii[f_t_selection], f_i_r_num[f_t_selection]),
    #     label="f_i rec",
    #     zorder=1,
    # )

    # axes[4].plot(
    #     f_t_analytical_radii,
    #     normalise(f_t_analytical_radii, f_t_analytical_num),
    #     label="analytical f_t",
    #     zorder=1,
    # )
    # axes[4].plot(
    #     f_t_r_centres,
    #     normalise(f_t_r_edges, f_t_r_numerical),
    #     label="f_t num",
    #     zorder=2,
    #     linestyle='--',
    # )

    # for ax in list(axes[:3]) + list(axes[5:]):
    #     ax.axis("scaled")

    # for ax in axes[3:5]:
    #     ax.set_title("Step lengths")

    # for ax in axes[3:5]:
    #     ax.legend(loc="best")
    #     ax.grid(linestyle="--", alpha=0.4)

    # fig.tight_layout()

    # Plot f_i.
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$r$ (step size)")
    ax.set_ylabel(r"$f_i(r)$")

    plot_r = 3
    f_ta_selection = f_t_r_centres < plot_r
    ax.plot(
        f_t_r_centres[f_ta_selection],
        normalise(f_t_r_centres[f_ta_selection], f_i_r_analytical[f_ta_selection]),
        label="analytical f_i",
        zorder=2,
        linestyle="--",
    )

    f_t_selection = f_i_r_num_radii < plot_r
    ax.plot(
        f_i_r_num_radii[f_t_selection],
        normalise(f_i_r_num_radii[f_t_selection], f_i_r_num[f_t_selection]),
        label="f_i rec",
        zorder=1,
    )
    ax.legend(loc="best")
    ax.grid(linestyle="--", alpha=0.4)

    # Plot f_t.
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$r$ (step size)")
    ax.set_ylabel(r"$f_t(r)$")

    ax.plot(
        f_t_analytical_radii,
        normalise(f_t_analytical_radii, f_t_analytical_num),
        label="analytical f_t",
        zorder=1,
    )
    ax.plot(
        f_t_r_centres,
        normalise(f_t_r_edges, f_t_r_numerical),
        label="f_t num",
        zorder=2,
        linestyle="--",
    )
    ax.legend(loc="best")
    ax.grid(linestyle="--", alpha=0.4)
