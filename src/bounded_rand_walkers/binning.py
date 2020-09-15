# -*- coding: utf-8 -*-
import copy
import logging

import numpy as np


def binning1D(binsize, data, normalising):
    """
    This function bins the data in constant binsizes.

    Args:
        data: 1d array of data that should be binned
        binsize: constant, which is the size of each bin
        normalising: Boolean, if true, the histogram is normalised

    returns:
        histrogram: list of list with [centre of bin, frequency]

    """

    if isinstance(data, list):
        data = np.asarray(data)
    if len(data) == 0:
        return []

    n_elements = float(data.size)
    histogram = []

    # shift everything by number binsizes upwards and then shift it down
    min_value = np.min(data)
    bin_num = np.ceil(np.abs(min_value / float(binsize)))

    data = data + bin_num * binsize

    # Bin the positive values
    flag = True
    counter = 1.0
    while flag:
        indeces = np.where((data - counter * binsize) < 0)[0]

        # Normalise it if desired
        if normalising:
            freq = indeces.size / n_elements
        else:
            freq = indeces.size

        data = np.delete(data, indeces)

        if freq > 0:
            # binsize / 2 so get centre bin
            histogram.append([binsize * (counter - 0.5 - bin_num), freq])

        if not data.size:
            flag = False
        else:
            counter += 1
    return convert_array_1D(histogram, bin_size=binsize)


def binning2D(binsize, data, normalising):
    """
    This function bins the data in constant binsizes in 2 dimensions

    Args:
        data: 2d array of data that should be binned
        binsize: list of constant x and y binsize ie [x_binsize, y_binsize]
        normalising: Boolean, if true, the histogram is normalised

    returns:
        histrogram: list of list with [ [x,y centre of bin], frequency]

    """
    if isinstance(data, list):
        data = np.asarray(data)
    if len(data.shape) == 0:
        raise Exception("Passed empty data set to 2D binning function")

    min_value = np.min(data[:, 0])
    bin_num = np.ceil(np.abs(min_value / float(binsize[0])))

    copy_data = copy.deepcopy(data)

    copy_data[:, 0] = data[:, 0] + bin_num * binsize[0]

    flag = True
    n_elements = data.size / 2.0
    histogram2D = []
    counter = 1.0

    while flag:
        indeces = np.where((copy_data[:, 0] - counter * binsize[0]) < 0)[0]

        ybin = binning1D(binsize[1], copy_data[indeces, 1], False)

        copy_data = np.delete(copy_data, indeces, axis=0)

        if ybin > 0:
            for l in range(len(ybin)):
                if normalising:
                    histogram2D.append(
                        [
                            [binsize[0] * (counter - 0.5 - bin_num), ybin[l][0]],
                            ybin[l][1] / n_elements,
                        ]
                    )
                else:
                    histogram2D.append(
                        [
                            [binsize[0] * (counter - 0.5 - bin_num), ybin[l][0]],
                            ybin[l][1],
                        ]
                    )

        if not copy_data.size:
            flag = False
        else:
            counter += 1

    return histogram2D


def estimate_f_t(length_list, binsize):
    """
    This function uses all lengths lists to estimate the f_t(y)
    distribution.

    Args:
        length_list: list of all step sizes

    Returns:
        histogram of f_t(y)

    """
    histogram = binning1D(binsize, length_list, True)
    return histogram


def estimate_gx(binsize, length_list):
    """
    This function estimates G(x) by first taking the cumulative sum of the
    length list and then binning the result.

    Args:
        length_list: list or array of all step sizes
        binsize: size of the bins in which the data should be put

    note::

        both input arguments can either be 1 or 2D

    """
    # Calculate cumulative list
    length_list = np.squeeze(length_list)
    cum_array = np.cumsum(length_list, axis=0)
    print((np.min(cum_array), np.max(cum_array)))

    if len(cum_array.shape) == 1:
        binned = binning1D(binsize, cum_array, True)

        return binned

    elif len(cum_array.shape) == 2:
        binned = binning2D(binsize, cum_array, True)
        xpos, ypos, freq_array = convert_array_2D(binned, binsize)

        return xpos, ypos, freq_array

    else:
        raise NotImplementedError(
            "The input array is neither 1 or 2D which " "this process cannot handle"
        )
    return binned


def convert_array_1D(histogram1D, bin_size=None):
    """
    Convert [[bin_center1, freq1], ...] lists into an array containing the
    bin centres and an array containing the frequencies.

    Examples:
        >>> from bounded_rand_walkers.binning import convert_array_1D
        >>> histogram1D = [(.1, 0.2), (.2, 0.3), (.4, 0.1), (.5, 0.2)]
        >>> centres, frequencies = convert_array_1D(histogram1D)
        >>> np.testing.assert_allclose(centres, np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        >>> np.testing.assert_allclose(frequencies, np.array([0.2, 0.3, 0., 0.1, 0.2]))

    """
    centres = np.asarray([element[0] for element in histogram1D])
    frequencies = np.asarray([element[1] for element in histogram1D])
    if bin_size == None:
        # try to determine automatically
        centre_diffs = np.diff(centres)
        unique_centre_diffs = np.unique(centre_diffs)
        occurrences = [
            (centre_diff, np.sum(np.isclose(centre_diffs, centre_diff)))
            for centre_diff in unique_centre_diffs
        ]
        sorted_occurrences = sorted(occurrences, key=lambda x: x[1], reverse=True)
        bin_size = sorted_occurrences[0][0]
        ratio = (np.max(centres) - np.min(centres)) / bin_size
        assert np.isclose(np.round(ratio), ratio), "{:} {:} {:}".format(
            bin_size, np.round(ratio), ratio
        )

    bin_centres = np.arange(np.min(centres), np.max(centres) + bin_size, bin_size)
    out_frequencies = np.zeros_like(bin_centres, dtype=np.float64)
    for i, bin_centre in enumerate(bin_centres):
        matching_index = np.where(np.isclose(centres, bin_centre))[0]
        if len(matching_index):
            out_frequencies[i] = frequencies[matching_index[0]]
    if np.isclose(out_frequencies[-1], 0):
        bin_centres = bin_centres[:-1]
        out_frequencies = out_frequencies[:-1]
    return bin_centres, out_frequencies


def convert_array_2D(histogram2D, binsize):
    """
    This function converts an input list of list, of the form [[[x,y], freq],
    [[x2,y2], freq2], ...] into the one list of all xpositions, one list of all
    ypositions and an 2D array of shape (len(ypositions), len(xpositions))
    of the freq values. This function would be usually be used to convert
    the output of Binning2D into an usable array format.

    Args:
        histogram2D: a list of list of the format described above
        binsize: a 1D array or list of the binsizes in the x,y direction

    Returns:
        all_xpos: list of all xpositions starting from the minimum x position,
                  ending at the max xposition in integer binsize steps
        all_ypos: same as all_xpos just with y values and y binsizes
        freq_array: 2D array of shape (y,x) with their frequencies as entries

    """
    logger = logging.getLogger(__name__)
    xpos = [histogram2D[i][0][0] for i in range(len(histogram2D))]
    ypos = [histogram2D[i][0][1] for i in range(len(histogram2D))]

    min_x = np.min(xpos)
    max_x = np.max(xpos)

    min_y = np.min(ypos)
    max_y = np.max(ypos)

    freq_array = np.zeros(
        (
            int(np.round(np.floor((max_y - min_y) / float(binsize[1])))) + 2,
            int(np.round(np.floor((max_x - min_x) / float(binsize[0])))) + 2,
        )
    )

    logger.debug(freq_array.shape)

    for i, row in enumerate(histogram2D):
        xindex = (row[0][0] - min_x) / float(binsize[0])
        yindex = (row[0][1] - min_y) / float(binsize[1])
        freq_array[int(np.round(yindex)), int(np.round(xindex))] = row[1]

    all_xpos = []
    all_ypos = []

    for i in range(freq_array.shape[1]):
        all_xpos.append(min_x + i * binsize[0])
    for j in range(freq_array.shape[0]):
        all_ypos.append(min_y + j * binsize[1])

    return all_xpos, all_ypos, freq_array
