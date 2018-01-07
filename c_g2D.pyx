from __future__ import division
import numpy as np
cimport numpy as np
DTYPE = np.float64
ctypedef np.float64_t DTYPE_t
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef DTYPE_t[:, :] c_g2D_func(
        f, DTYPE_t[:] xs_edges, DTYPE_t[:] ys_edges, DTYPE_t[:] xs_centres,
        DTYPE_t[:] ys_centres, np.int64_t[:] x_indices, np.int64_t[:] y_indices):

    cdef int mask_x_index
    cdef int mask_y_index
    cdef int mask_x_index2
    cdef int mask_y_index2
    cdef int i
    cdef int j
    cdef int k
    cdef int l
    cdef DTYPE_t [:] pos = np.empty((2,))
    cdef DTYPE_t [:] x_mod = np.empty((xs_centres.shape[0],))
    cdef DTYPE_t [:] y_mod = np.empty((ys_centres.shape[0],))
    cdef DTYPE_t [:, :] g_values = np.zeros((xs_centres.shape[0], ys_centres.shape[0]))

    # counter = 0
    # max_counter = len(x_indices)
    for k in range(x_indices.shape[0]):
        mask_x_index = x_indices[k]
        mask_y_index = y_indices[k]
        # evaluate the pdf at each position relative to the current
        # positions. But only iterate over the positions that are
        # actually in the boundary.

        # if max_counter < 20:
        #     print('counter:', counter + 1, 'out of:', max_counter)
        # else:
        #     if ((counter) % (max_counter / 10)) == 0:
        #         print('{:07.2%}'.format(float(counter) / (max_counter - 1)))
        # counter += 1

        for i in range(xs_centres.shape[0]):
            x_mod[i] = xs_centres[i] - xs_centres[mask_x_index]
        for j in range(ys_centres.shape[0]):
            y_mod[j] = ys_centres[j] - ys_centres[mask_y_index]

        for l in range(x_indices.shape[0]):
            mask_x_index2 = x_indices[l]
            mask_y_index2 = y_indices[l]
            pos[0] = x_mod[mask_x_index2]
            pos[1] = y_mod[mask_y_index2]
            g_values[mask_x_index2, mask_y_index2] += f(pos)

    return g_values
