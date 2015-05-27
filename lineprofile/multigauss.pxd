
cdef class MultiGauss:

    cdef:
        double[:] x_coord
        double[:] y_coord

        readonly int n_sources, n_baseline

