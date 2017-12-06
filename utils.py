# -*- coding: utf-8 -*-
#
import numpy as np
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
