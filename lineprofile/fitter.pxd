from model cimport LineModel

cdef class FitGaussian(LineModel):

    cdef:
        double[:] data
        double[:] weights

        # Prior parameters

        # Mean and center of the prior radial velocity
        # of each profile and gaussian.
        double[:] v_center_mean
        double[:] v_center_std

        # Rotation
        public double v_rot_k
        public double v_rot_theta

        # Solid body rotation
        public double fsolid_p
        public double fsolid_q

        # Asymmetry
        public double asym_p
        public double asym_q

        # Turbulent motions
        public double turbulence_min
        public double turbulence_k
        public double turbulence_theta

        # Baseline
        public double baseline_std

    cdef int model_params_offset(self)
    cdef double ln_bounds_model(self, double[:] p)
    cdef double ln_prior_model(self, double[:] p)
    cdef double ln_bounds_components(self, double[:] p)
    cdef double ln_prior_components(self, double[:] p)
    cdef double ln_likelihood(self, double[:] p)

cdef class FitLaplacian(FitGaussian):
    pass
