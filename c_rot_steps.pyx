from __future__ import division
import numpy as np
from libc.math cimport sin, cos, acos
# "cimport" is used to import special compile-time information
# about the numpy module (this is stored in a file numpy.pxd which is
# currently part of the Cython distribution).
cimport numpy as np
# We now need to fix a datatype for our arrays. I've used the variable
# DTYPE for this, which is assigned to the usual NumPy runtime
# type info object.
DTYPE = np.float64
# "ctypedef" assigns a corresponding compile-time type to DTYPE_t. For
# every type in the numpy module there's a corresponding compile-time
# type with a _t-suffix.
ctypedef np.float64_t DTYPE_t
ctypedef DTYPE_t DTYPE_m
# "def" can type its arguments but not have a return type. The type of the
# arguments for a "def" function is checked at run-time when entering the
# function.
#
# The arrays f, g and h is typed as "np.ndarray" instances. The only effect
# this has is to a) insert checks that the function arguments really are
# NumPy arrays, and b) make some attribute access like f.shape[0] much
# more efficient. (In this example this doesn't matter though.)
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef DTYPE_t[:, :] rot_steps(DTYPE_t[:, :] data):
    """
    Feed data containing stopping positions in 2D - shape [2 , n-stops]

    Examples:
        >>> N = 1000
        >>> pos_data2D = rand.uniform(0, 1, size=(2, N))
        >>> rot_steps_data = rot_steps(pos_data2D)
        >>> plt.figure()
        >>> plt.hist2d(rot_steps[0, :], rot_steps[1, :], bins=50)
        >>> plt.plot(np.arange(-2, 2, 0.01),
        ...          np.array([0 for a in np.arange(-2, 2, 0.01)]))
        >>> plt.title('Observed step-size with fixed incoming direction')
        >>> plt.gca().set_aspect('equal')
        >>> plt.show()

    """
    # print("This is c_rot_steps")
    # The "cdef" keyword is also used within functions to type variables. It
    # can only be used at the top indentation level (there are non-trivial
    # problems with allowing them in other places, though we'd love to see
    # good and thought out proposals for it).
    #
    # For the indices, the "int" type is used. This corresponds to a C int,
    # other C types (like "unsigned int") could have been used instead.
    # Purists could use "Py_ssize_t" which is the proper Python type for
    # array indices.
    # assert data.dtype == DTYPE
    cdef int i
    cdef int j
    cdef int columns = data.shape[1]
    cdef DTYPE_t [:, :] rot_steps_data = (
        np.zeros([2, columns - 2], dtype=DTYPE))
    cdef DTYPE_t [:] a = np.zeros([2,], dtype=DTYPE)
    cdef DTYPE_t [:] b = np.zeros([2,], dtype=DTYPE)
    cdef DTYPE_t [:] c = np.zeros([2,], dtype=DTYPE)
    cdef DTYPE_t [:] left = np.zeros([2,], dtype=DTYPE)
    cdef DTYPE_t [:] R_0 = np.zeros([2,], dtype=DTYPE)
    cdef DTYPE_t [:] R_1 = np.zeros([2,], dtype=DTYPE)
    cdef DTYPE_t [:] right = np.zeros([2,], dtype=DTYPE)
    # cdef np.ndarray[DTYPE_t, ndim=1] c_rot = np.zeros([2,], dtype=DTYPE)
    cdef DTYPE_m dot_prod
    cdef DTYPE_m phi
    cdef DTYPE_m theta
    cdef DTYPE_m a_b_norm

    # It is very important to type ALL your variables. You do not get any
    # warnings if not, only much slower code (they are implicitly typed as
    # Python objects).

    # For the value variable, we want to use the same data type as is
    # stored in the array, so we use "DTYPE_t" as defined above.
    # NB! An important side-effect of this is that if "value" overflows its
    # datatype size, it will simply wrap around like in C, rather than raise
    # an error like in Python.

    for i in range(columns - 2):

        for j in range(2):
            a[j] = data[j, i]
            b[j] = data[j, i + 1]
            c[j] = data[j, i + 2]

        # left = a - b
        # dot_prod = - left[1]
        dot_prod = b[1] - a[1]

        a_b_norm = 0
        for j in range(2):
            a_b_norm += (a[j] - b[j])**2.
        a_b_norm = a_b_norm**0.5

        phi = acos(
                dot_prod
                # / np.linalg.norm(a - b)
                / a_b_norm
                )

        if a[0] > b[0]:
            theta = -phi
        else:
            theta = phi

        # R = np.array([[np.cos(theta), -np.sin(theta)],
        #               [np.sin(theta), np.cos(theta)]],
        #              dtype=np.float64
        #              )

        # R_0 = np.array([np.cos(theta), -np.sin(theta)], dtype=DTYPE)
        # R_1 = np.array([np.sin(theta), np.cos(theta)], dtype=DTYPE)

        for j in range(2):
            right[j] = c[j] - b[j]
        # # c_rot[0] = np.sum(R_0 * right)
        # # c_rot[1] = np.sum(R_1 * right)
        # # rot_steps_data[:, i] = c_rot
        # rot_steps_data[0, i] = np.sum(R_0 * right)
        # rot_steps_data[1, i] = np.sum(R_1 * right)
        rot_steps_data[0, i] = (cos(theta) * right[0]) - (sin(theta) * right[1])
        rot_steps_data[1, i] = (sin(theta) * right[0]) + (cos(theta) * right[1])

    return rot_steps_data
