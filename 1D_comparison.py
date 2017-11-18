#!/usr/bin/env python2
# -*- conding: utf-8 -*-
"""
Compare analytical and numerical stepsize and positions distributions.

"""
import logging
import matplotlib.pyplot as plt
import numpy as np
from rotation_steps import g1D
from binning import estimate_fi
from data_generation import random_walker
from functions import tophat_1D

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
xs = np.linspace(0, 1, 10)


widths = [0.3, 0.5, 0.7]
g_analytical = []
for x in xs:
    g_analytical.append(g1D(x, tophat_1D))
g_analytical = np.asarray(g_analytical)

step_values, positions = random_walker(
        f_i=tophat_1D,
        bounds=np.array([0, 1]),
        steps=int(1e4),
        return_positions=True,
        )
logger.debug('{:} {:}'.format(step_values.shape, positions.shape))
probs, bin_edges = np.histogram(positions)
left_edges = bin_edges[:-1]
right_edges = bin_edges[1:]
bin_centres = (left_edges + right_edges) / 2.

fig, axes = plt.subplots(1, 2, squeeze=True)

axes[0].set_title(r'$Analytical g(x)$')
axes[0].plot(xs, g_analytical)

axes[1].set_title(r'$Numerical g(x)$')
axes[1].plot(bin_centres, probs)

plt.show()
