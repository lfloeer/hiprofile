from fitter cimport FitMixture

cdef class GaussianModel:

    cdef:
        double[:] x_coord, y_coord
        double[:] model_array

        readonly int n_sources, n_baseline

    cdef void eval_model(self, double[::1] p)
    cdef void eval_gaussians(self, double[::1] p)
    cdef void eval_baseline(self, double[::1] p)


cdef class Deblender(GaussianModel):

    cdef:
        double[:] data
        double[:] x_coord
        double[:] y_coord

        public double amp_min, amp_max
        public double disp_min, disp_max
        public double std_baseline
        public double fraction_min, fraction_max
        public double std_in_min, std_in_max
        public double std_out_min, std_out_max
        public double mu_out_std

        public double[:] src_x_coord
        public double[:] src_y_coord
        public double[:] src_coord_std

        readonly int n_sources, n_baseline

    cdef double metric(self, double x1, double y1, double x2, double y2)

    cdef int likelihood_params_offset(self)
    cdef double ln_bounds_likelihood(self, double[::1] p)
    cdef double ln_prior_likelihood(self, double[::1] p)
    
    cdef double ln_bounds_components(self, double[::1] p)
    cdef double ln_prior_components(self, double[::1] p)
    
    cpdef double ln_likelihood(self, double[::1] p)
    cpdef double ln_prior(self, double[::1] p)
    cpdef double ln_posterior(self, double[::1] p)


cdef class SphericalDeblender(Deblender):
    pass
