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
from libc.math cimport log, exp, M_PI

cimport ln_likes

cdef class FitGaussian(LineModel):
    "Fit a LineModel to data assuming a Gaussian likelihood"

    def __init__(self, velocities, data, weights=None, **kwargs):

        super(FitGaussian, self).__init__(velocities, **kwargs)
        self.data = np.array(data, dtype=np.double, copy=True)

        if weights is None:
            self.weights = np.ones_like(data)
        else:
            self.weights = weights

        self.fint_min = 0.0
        self.fint_max = 4.0

        self.v_rot_k = 4.38
        self.v_rot_theta = 55.40

        self.fsolid_p = 1.0
        self.fsolid_q = 5.0

        self.asym_p = 10.0
        self.asym_q = 10.0

        self.turbulence_min = 5.0
        self.turbulence_k = 5.0
        self.turbulence_theta = 2.0

        self.gauss_disp_min = -1
        self.gauss_disp_max = 2

        self.baseline_std = 0.05

    property data:

        def __get__(self):
            return np.asarray(self.data)

        def __set__(self, value):
            self.data = np.array(value, dtype=np.double, copy=True)

    property weights:

        def __get__(self):
            return np.asarray(self.weights)

        def __set__(self, value):
            self.weights = np.array(value, dtype=np.double, copy=True)

    property v_center_mean:

        def __get__(self):
            return np.asarray(self.v_center_mean)

        def __set__(self, value):
            self.v_center_mean = np.array(value, dtype=np.double, copy=True)

    property v_center_std:

        def __get__(self):
            return np.asarray(self.v_center_std)

        def __set__(self, value):
            self.v_center_std = np.array(value, dtype=np.double, copy=True)

    property n_dim:

        def __get__(self):
            return self.n_disks * 6 + self.n_gaussians * 3 + self.n_baseline + 1

    cdef int likelihood_params_offset(self):
        return 6 * self.n_disks + 3 * self.n_gaussians + self.n_baseline

    cdef double ln_bounds_components(self, double[::1] p):
        "Evaluate the hard bounds for each parameter of the model components"
        cdef:
            int i, offset, component
            double vmin, vmax

        offset = 0
        component = 0
        vmin = self.velocities[0]
        vmax = self.velocities[self.velocities.shape[0] - 1]

        for i in range(self.n_disks):
            # Positive integrated flux density
            if p[offset + 0] < self.fint_min or p[offset + 0] > self.fint_max:
                return -inf
            # Positive rotation
            if p[offset + 2] <= 0.:
                return -inf
            # Positive dispersion
            if p[offset + 3] <= self.turbulence_min:
                return -inf
            # Bounded solid rotating fraction
            if p[offset + 4] <= 0. or p[offset + 4] >= 1.:
                return -inf
            # Bounded asymmetry
            if p[offset + 5] <= -1.0 or p[offset + 5] >= 1.0:
                return -inf

            offset += 6
            component += 1

        for i in range(self.n_gaussians):
            # Positive amplitude
            if p[offset + 0] < self.fint_min or p[offset + 0] > self.fint_max:
                return -inf
            # Positive dispersion
            if p[offset + 2] < self.gauss_disp_min or p[offset + 2] > self.gauss_disp_max:
                return -inf

            offset += 3
            component += 1

        for i in range(self.n_baseline):
            offset += 1

        return 0.0

    cdef double ln_bounds_likelihood(self, double[::1] p):
        "Evaluate the hard bounds for each parameter"
        cdef:
            int i, offset

        offset = self.likelihood_params_offset()

        # Positive likelihood stddev
        if p[offset] <= -3.0 and p[offset] >= 3.0:
            return -inf

        return 0.0

    cdef double ln_prior_components(self, double[::1] p):
        cdef:
            int i, offset, component
            double ln_value = 0.0

        offset = 0
        component = 0

        for i in range(self.n_disks):
            # Normal prior on radial velocity
            ln_value += ln_likes.ln_normal(p[offset + 1],
                                           self.v_center_mean[component],
                                           self.v_center_std[component])
            # Gamma prior on profile width
            # Prior parameters are from a ML fit to HIPASS data
            ln_value += ln_likes.ln_gamma(p[offset + 2], self.v_rot_k, self.v_rot_theta)
            # Gamma prior on turbulent motion. Parameters roughly guided
            # by the paper of Stewart et al. 2014
            ln_value += ln_likes.ln_gamma(p[offset + 3] - self.turbulence_min, self.turbulence_k, self.turbulence_theta)
            # Beta distribution on fraction of solid body rotation
            ln_value += ln_likes.ln_beta(p[offset + 4], self.fsolid_p, self.fsolid_q)
            # Beta distribution on asymmetry
            ln_value += ln_likes.ln_beta(0.5 * (p[offset + 5] + 1.0), self.asym_p, self.asym_q)

            offset += 6
            component += 1

        for i in range(self.n_gaussians):
            # Normal prior on center
            ln_value += ln_likes.ln_normal(p[offset + 1],
                                           self.v_center_mean[component],
                                           self.v_center_std[component])

            offset += 3
            component += 1

        for i in range(self.n_baseline):
            # Normal prior on baseline coefficients
            ln_value += ln_likes.ln_normal(p[offset], 0.0, self.baseline_std)
            offset += 1

        return ln_value

    cdef double ln_prior_likelihood(self, double[::1] p):
        return 0.0

    cpdef double ln_likelihood(self, double[::1] p):
        cdef:
            int i, offset
            double ln_value = 0.0
            double stddev

        self.eval_model(p)
        
        offset = self.likelihood_params_offset()
        stddev = 10.0 ** p[offset]

        for i in range(self.data.shape[0]):
            ln_value += ln_likes.ln_normal(self.data[i],
                                           self.model_array[i],
                                           stddev * self.weights[i])

        return ln_value

    def ln_posterior(self, double[::1] p):
        cdef double ln_value = self.ln_bounds_likelihood(p)

        if ln_value == 0.0:

            ln_value = self.ln_bounds_components(p)

            if ln_value == 0.0:
                
                ln_value += self.ln_prior_components(p)
                ln_value += self.ln_prior_likelihood(p)
                ln_value += self.ln_likelihood(p)

        return ln_value

    def ln_prior(self, double[::1] p):
        cdef double ln_value = self.ln_bounds_likelihood(p)

        if ln_value == 0.0:

            ln_value = self.ln_bounds_components(p)

            if ln_value == 0.0:
                
                ln_value += self.ln_prior_components(p)
                ln_value += self.ln_prior_likelihood(p)

        return ln_value

cdef class FitLaplacian(FitGaussian):

    cpdef double ln_likelihood(self, double[::1] p):
        cdef:
            int i, offset
            double ln_value = 0.0
            double stddev

        self.eval_model(p)

        offset = self.likelihood_params_offset()
        stddev = 10.0 ** p[offset]

        for i in range(self.data.shape[0]):
            ln_value += ln_likes.ln_laplace(self.data[i],
                                            self.model_array[i],
                                            stddev * self.weights[i])

        return ln_value

    cdef double ln_prior_likelihood(self, double[::1] p):
        return 0.0

cdef class FitMixture(FitGaussian):
    """
    Parameters of the likelihood (offset + x):
    0: fraction of bad samples
    1: stddev of good samples
    2: offset of bad samples
    3: stddev of bad samples
    """

    def __init__(self, *args, **kwargs):

        super(FitMixture, self).__init__(*args, **kwargs)

        self.fraction_min = -3
        self.fraction_max = 0
        self.std_in_min = -3
        self.std_in_max = 0
        self.std_out_min = -3
        self.std_out_max = 1
        self.mu_out_std = 0.1

    property n_dim:

        def __get__(self):
            return self.n_disks * 6 + self.n_gaussians * 3 + self.n_baseline + 4
    
    cpdef double ln_likelihood(self, double[::1] p):
        cdef:
            int i, j, offset
            double diff_dm, tmp
            double fraction, std_in, mu_out, std_out
            double scaled_std_in, scaled_var_out
            double ln_value = 0.0
            double ln_in, ln_out
            double ln_frac_in, ln_frac_out

        self.eval_model(p)

        offset = self.likelihood_params_offset()

        fraction = 10.0 ** p[offset + 0]
        ln_frac_in = log(1. - fraction)
        ln_frac_out = log(fraction)

        std_in = 10.0 ** p[offset + 1]

        mu_out = p[offset + 2]
        std_out = 10.0 ** p[offset + 3]

        # The individual data point likelihoods are added using the
        # "log-sum-exp trick" to prevent underflow.
        for i in range(self.data.shape[0]):
            scaled_std_in = std_in * self.weights[i]
            scaled_var_out = std_out * std_out + scaled_std_in * scaled_std_in
            
            diff_dm = self.data[i] - self.model_array[i]
            
            tmp = diff_dm / scaled_std_in
            tmp *= tmp
            ln_in = ln_frac_in - 0.5 * tmp - log(scaled_std_in)

            tmp = (diff_dm + mu_out)
            tmp = tmp * tmp / scaled_var_out
            ln_out = ln_frac_out - 0.5 * tmp - 0.5 * log(scaled_var_out)

            if ln_out > ln_in:
                ln_value += ln_out + log(1. + exp(ln_in - ln_out))
            else:
                ln_value += ln_in + log(1. + exp(ln_out - ln_in))

        return ln_value

    cdef double ln_prior_likelihood(self, double[::1] p):
        cdef int offset
        cdef double mu_out
        cdef double ln_value = 0.0

        offset = self.likelihood_params_offset()
        mu_out = p[offset + 2]

        ln_value += ln_likes.ln_normal(mu_out, 0., self.mu_out_std)

        return ln_value

    cdef double ln_bounds_likelihood(self, double[::1] p):
        cdef int offset
        cdef double fraction, std_in, std_out

        offset = self.likelihood_params_offset()
        fraction = p[offset + 0]
        std_in = p[offset + 1]
        std_out = p[offset + 3]

        if fraction <= self.fraction_min or fraction >= self.fraction_max:
            return -inf

        if std_in <= self.std_in_min or std_in >= self.std_in_max:
            return -inf

        if std_out <= self.std_out_min or std_out >= self.std_out_max:
            return -inf

        return 0.0
