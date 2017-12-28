#!/usr/bin/env python2
import matplotlib.pyplot as plt
import numpy as np
import sys


if __name__ == '__main__':
    data = np.load(sys.argv[1])
    plt.figure()
    if len(sys.argv) == 2:
        if np.squeeze(data).ndim == 1:
            plt.plot(data, linestyle='--', marker='o')
        elif np.squeeze(data).ndim == 2:
            plt.imshow(data)
            plt.gca().set_aspect('equal')
            plt.colorbar()
    elif len(sys.argv) == 3:
        if sys.argv[2] == 'hist':
            if np.squeeze(data).ndim == 1:
                plt.hist(data)
            elif np.squeeze(data).ndim == 2:
                plt.hexbin(*data.T)
                plt.gca().set_aspect('equal')
                plt.colorbar()

    plt.show()
