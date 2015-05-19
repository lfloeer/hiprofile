cimport fitter

cdef class SpectralDeblender(fitter.FitMixture):

    cdef:
        double[:, ::1] all_data, all_weights
        double[::1] formatted_p

        public double d_vsys_std
        public double d_asym_std
        public double d_turbulence_std
        public double d_v_rot_std
        public double d_fsolid_std

        readonly bint fix_parameters

    cpdef double deblending_ln_prior(self, double[::1] p)
    cpdef double deblending_ln_likelihood(self, double[::1] p)
    cpdef double deblending_ln_posterior(self, double[::1] p)

    cdef void setup_spectrum(self, double[::1] full_p, int spectrum)