#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.pardir))
from time import time

# from rotation_steps import rot_steps_fast as rot_steps
from c_rot_steps import rot_steps

data_dir = os.path.join(os.path.dirname(__file__), "data")


def get_cpp_binned_2D(
    samples, bounds_name, pdf_name, pdf_kwargs, x_edges, y_edges, ft_xs, ft_ys
):
    """Bin data generated by "data_generation.cpp"

    Due to memory constraints, it can't all be loaded at the same time -
    therefore bin it iteratively, and normalise it at the end.

    centre (iterable): Eg. [0.2, 0.2] - the centre of the pdf
    data_type (str): 'positions' or 'steps'

    """
    limit = 4000
    # use the cell areas to normalise the data later
    g_cell_area = np.abs((x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0]))
    ft_cell_area = np.abs((ft_xs[1] - ft_xs[0]) * (ft_ys[1] - ft_ys[0]))

    g_numerical_counts = np.zeros(
        (x_edges.size - 1, y_edges.size - 1), dtype=np.float64
    )
    f_t_numerical_counts = np.zeros(
        (x_edges.size - 1, y_edges.size - 1), dtype=np.float64
    )
    rot_probs_counts = np.zeros((x_edges.size - 1, y_edges.size - 1), dtype=np.float64)
    counter = 0
    for positions in load_cpp_data_iter(
        samples,
        2,
        bounds_name,
        pdf_name,
        pdf_kwargs["centre"],
        pdf_kwargs["width"],
        "positions",
    ):
        # use x_edges and y_edges to bin the position data
        g_numerical_raw, _, _ = np.histogram2d(
            *positions.T, bins=[x_edges, y_edges], normed=False
        )
        g_numerical_counts += g_numerical_raw
        counter += 1
        if counter == limit:
            print("Reached limit in data reading!")
            break

    counter = 0
    for steps in load_cpp_data_iter(
        samples,
        2,
        bounds_name,
        pdf_name,
        pdf_kwargs["centre"],
        pdf_kwargs["width"],
        "steps",
    ):
        start = time()
        # use x_edges and y_edges to bin the position data
        f_t_numerical_raw, _, _ = np.histogram2d(
            *steps.T, bins=[ft_xs, ft_ys], normed=False
        )
        f_t_numerical_counts += f_t_numerical_raw
        print(("binning steps:", time() - start))

        rot_steps_raw = rot_steps(steps.T)
        print(("rot steps:", time() - start))
        rot_probs_raw_binned, _, _ = np.histogram2d(
            rot_steps_raw[0, :], rot_steps_raw[1, :], bins=[ft_xs, ft_ys], normed=False
        )
        rot_probs_counts += rot_probs_raw_binned
        print(("rot steps binning:", time() - start))
        counter += 1
        if counter == limit:
            print("Reached limit in data reading!")
            break

    # now normalise the binned data generated above
    g_numerical = g_numerical_counts / np.sum(g_numerical_counts * g_cell_area)
    f_t_numerical = f_t_numerical_counts / np.sum(f_t_numerical_counts * ft_cell_area)
    rot_probs = rot_probs_counts / np.sum(rot_probs_counts * ft_cell_area)

    return g_numerical, f_t_numerical, rot_probs


def load_cpp_data_iter(
    samples, dims, bounds_name, pdf_string, centre, width, data_type="positions"
):
    """Load data generated by "data_generation.cpp"

    centre (iterable): Eg. [0.2, 0.2] - the centre of the pdf
    data_type (str): 'positions' or 'steps'

    """
    version_int = 0
    centre_str = ""
    for c in centre:
        centre_str += "{:.3e},".format(c)
    while True:
        filename = (
            "{:}{:}_samples_{:0.3e}_dims_{:}_bounds_{:}_pdf_"
            "{:}_centre_{:}_width_{:0.3e}.npy"
        ).format(
            version_int,
            data_type,
            float(samples),
            dims,
            bounds_name,
            pdf_string,
            centre_str,
            float(width),
        )
        filename = os.path.join(data_dir, filename)
        if os.path.isfile(filename):
            print(("Found data for version:{:}".format(version_int)))
            yield np.load(filename)
            version_int += 1
        else:
            print(("Did not find:{:}".format(filename)))
            break


def load_cpp_data(
    samples, dims, bounds_name, pdf_string, centre, width, data_type="positions"
):
    """Load data generated by "data_generation.cpp"

    centre (iterable): Eg. [0.2, 0.2] - the centre of the pdf
    data_type (str): 'positions' or 'steps'

    """
    data_arrays = []
    for data in load_cpp_data_iter(
        samples, dims, bounds_name, pdf_string, centre, width, data_type
    ):
        data_arrays.append(data)
    if data_arrays:
        return np.vstack(data_arrays)
    else:
        raise Exception(
            "Data for {:} not found".format(
                str(
                    (samples, dims, bounds_name, pdf_string, centre, width, "positions")
                )
            )
        )


if __name__ == "__main__":
    args = [1e7, 2, "weird", "gauss", [0.0, 0.0], 0.2, "positions"]
    positions = load_cpp_data(*args)
    args[-1] = "steps"
    steps = load_cpp_data(*args)

    fig, axes = plt.subplots(1, 2, squeeze=True)
    axes[0].hexbin(*positions.T)
    axes[0].set_aspect("equal")
    axes[0].set_title("Positions")
    axes[1].hexbin(*steps.T)
    axes[1].set_aspect("equal")
    axes[1].set_title("Steps")
    plt.show()
