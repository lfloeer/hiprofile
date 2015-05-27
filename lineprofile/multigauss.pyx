#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False
#cython: embedsignature=True

STUFF = "Hi"

import numpy as np
cimport numpy as np

cimport cython

from numpy.math cimport INFINITY as inf
from libc.math cimport log, fabs, exp


cdef class MultiGauss:

    def __init__(self, x_coord, y_coord, n_sources, n_baseline):

        self.x_coord = np.array(x_coord, dtype=np.double, copy=True)
        self.y_coord = np.array(y_coord, dtype=np.double, copy=True)

        self.n_sources = n_sources
        self.n_baseline = n_baseline

    def model(self, double[:] p):
        """
        Parameters
        ----------
            p : double buffer
                Parameters for each of the gaussian components and baseline
                terms.
        """
        pass

    cdef eval_model(self, double[:] p):
        pass