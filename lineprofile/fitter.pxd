from model cimport LineModel

cdef class FitGaussian(LineModel):

    cdef:
        double[:] data
        double[:] weights

        # Prior parameters
        public double fint_min
        public double fint_max

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

        # Gaussian dispersion
        public double gauss_disp_min
        public double gauss_disp_max

        # Baseline
        public double baseline_std

        # Enforce ordering
        public bint enforce_ordering
        public bint normalize_priors

    cdef int likelihood_params_offset(self)
    cdef double ln_bounds_likelihood(self, double[::1] p)
    cdef double ln_prior_likelihood(self, double[::1] p)
    cdef double ln_prior_likelihood_normalization(self)
    cdef double ln_bounds_components(self, double[::1] p)
    cdef double ln_prior_components(self, double[::1] p)
    cdef double ln_prior_components_normalization(self)
    cpdef double ln_likelihood(self, double[::1] p)

cdef class FitLaplacian(FitGaussian):
    pass

cdef class FitMixture(FitGaussian):

    cdef:
        public double fraction_min
        public double fraction_max
        public double std_in_min
        public double std_in_max
        public double std_out_min
        public double std_out_max
        public double mu_out_std
