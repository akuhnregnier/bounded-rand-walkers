# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

import bounded_rand_walkers.cpp.boundaries as bounds

if __name__ == "__main__":
    all_vertices = (bounds.square(), bounds.triangle(), bounds.circle())
    fig, axes = plt.subplots(1, len(all_vertices), sharex=True, sharey=True)
    for ax, vertices in zip(axes, all_vertices):
        ax.plot(*vertices.T, marker="o")
        ax.axis("scaled")
        ax.set_xlabel("x")

    axes[0].set_ylabel("y")
