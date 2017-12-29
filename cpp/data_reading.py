#!/usr/bin/env python2
import numpy as np
import matplotlib.pyplot as plt
import os


def load_cpp_data(samples, dims, pdf_string, centre, width,
                  data_type="positions"):
    """Load data generated by "data_generation.cpp"

    centre (iterable): Eg. [0.2, 0.2] - the centre of the pdf
    data_type (str): 'positions' or 'steps'

    """
    data_arrays = []
    version_int = 0
    centre_str = ''
    for c in centre:
        centre_str += '{:.3e},'.format(c)
    while True:
        filename = (
                ("data/{:}{:}_samples_{:0.3e}_dims_{:}_pdf_"
                 "{:}_centre_{:}_width_{:0.3e}.npy")
                .format(
                    version_int,
                    data_type,
                    float(samples),
                    dims,
                    pdf_string,
                    centre_str,
                    float(width))
                    )
        if os.path.isfile(filename):
            data_arrays.append(np.load(filename))
            version_int += 1
        else:
            break
    if data_arrays:
        return np.vstack(data_arrays)
    else:
        raise Exception('{:} not found'.format(filename))


if __name__ == '__main__':
    args = [1e6, 2, "gauss", [0., 0.], 0.2, "positions"]
    positions = load_cpp_data(*args)
    args[-1] = "steps"
    steps = load_cpp_data(*args)

    fig, axes = plt.subplots(1, 2, squeeze=True)
    axes[0].hexbin(*positions.T)
    axes[0].set_aspect('equal')
    axes[0].set_title('Positions')
    axes[1].hexbin(*steps.T)
    axes[1].set_aspect('equal')
    axes[1].set_title('Steps')
    plt.show()