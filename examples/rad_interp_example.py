# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

from bounded_rand_walkers.cpp import *
from bounded_rand_walkers.rad_interp import *

if __name__ == "__main__":
    # N = 100
    # x_edges = y_edges = np.linspace(-1, 1, N + 1)
    # x_centres = get_centres(x_edges)
    # y_centres = get_centres(y_edges)

    # data = (
    #     np.ones((N, N))
    #     + np.random.random((N, N)) * 0.2
    #     - 0.5 * (x_centres.reshape(-1, 1) + y_centres.reshape(1, -1)) ** 2.0
    # )
    # fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    # im = axes[0].pcolormesh(x_edges, y_edges, data)
    # axes[0].axis("scaled")
    # fig.colorbar(im, ax=axes[0], shrink=0.5)

    # axes[1].plot(*radial_interp(data, x_centres, y_centres, 20, 10)[::-1], label="orig")
    # axes[1].plot(
    #     *radial_interp_circ(data, x_centres, y_centres, 20)[::-1], label="circ"
    # )
    # axes[1].legend(loc='best')

    # plt.show()

    N = int(1e6)

    # Generate x, y coords.
    # data = (np.random.random((N, 2)) - 0.5)

    lengths = np.random.random(N)
    angles = np.random.random(N) * 2 * np.pi

    x = lengths * np.cos(angles)
    y = lengths * np.sin(angles)

    data = np.hstack((x[:, None], y[:, None]))

    # data = testing_2d(pdf_name='gauss', N=N)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    im = axes[0].hexbin(*data.T)
    axes[0].axis("scaled")
    fig.colorbar(im, ax=axes[0], shrink=0.5)

    axes[1].plot(*radial_interp_circ(data, 10)[::-1], label="circ")
    axes[1].legend(loc="best")

    plt.show()
