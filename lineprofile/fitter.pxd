from model cimport LineModel

cdef class FitGaussian(LineModel):

    cdef:
        public double[:] data
        public double[:] weights

        # Prior parameters

        # Mean and center of the prior radial velocity
        # of each profile and gaussian.
        public double[:] v_center_mean
        public double[:] v_center_std

        public double min_turbulence

    cdef double ln_bounds(self, double[:] p)
    cdef double ln_prior(self, double[:] p)
    cdef double ln_likelihood(self, double[:] p)