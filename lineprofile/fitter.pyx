#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False

STUFF = "Hi"

import numpy as np
cimport numpy as np

cimport cython

from numpy.math cimport INFINITY as inf

cimport ln_likes

cdef class FitGaussian(LineModel):
    "Fit a LineModel to data assuming a Gaussian posterior"

    def __init__(self, velocities, data, weights=None, **kwargs):

        super(FitGaussian, self).__init__(velocities, **kwargs)
        self.data = data

        if weights is None:
            self.weights = np.ones_like(data)
        else:
            self.weights = weights

        self.v_rot_k = 4.38
        self.v_rot_theta = 55.40

        self.fsolid_p = 1.0
        self.fsolid_q = 5.0

        self.asym_p = 10.0
        self.asym_q = 10.0

        self.turbulence_min = 5.0
        self.turbulence_k = 5.0
        self.turbulence_theta = 2.0

        self.baseline_std = 1.0

    property data:

        def __get__(self):
            return np.asarray(self.data)

        def __set__(self, value):
            self.data = np.asarray(value, dtype=np.double)

    property weights:

        def __get__(self):
            return np.asarray(self.weights)

        def __set__(self, value):
            self.weights = np.asarray(value, dtype=np.double)

    property v_center_mean:

        def __get__(self):
            return np.asarray(self.v_center_mean)

        def __set__(self, value):
            self.v_center_mean = np.asarray(value, dtype=np.double)

    property v_center_std:

        def __get__(self):
            return np.asarray(self.v_center_std)

        def __set__(self, value):
            self.v_center_std = np.asarray(value, dtype=np.double)

    cdef double ln_bounds(self, double[:] p):
        "Evaluate the hard bounds for each parameter"
        cdef:
            int i, offset
            double vmin, vmax

        offset = 0
        vmin = self.velocities[0]
        vmax = self.velocities[self.velocities.shape[0] - 1]

        for i in range(self._n_profiles):
            # Positive integrated flux density
            if p[offset + 0] < 0.:
                return -inf
            # Profile fully in bounds
            if (p[offset + 1] - p[offset + 2] / 2.) < vmin or \
               (p[offset + 1] + p[offset + 2] / 2.) > vmax:
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

        for i in range(self._n_gaussians):
            # Positive amplitude
            if p[offset + 0] < 0.:
                return -inf
            # In bounds
            if p[offset + 1] - p[offset + 2] < vmin or \
               p[offset + 1] + p[offset + 2] > vmax:
               return -inf
            # Positive dispersion
            if p[offset + 2] <= 0.:
                return -inf

            offset += 3

        for i in range(self._n_baseline):
            offset += 1

        # Positive likelihood stddev
        if p[offset] <= 0.:
            return -inf

        return 0.0

    cdef double ln_prior(self, double[:] p):
        cdef:
            int i, offset, component
            double ln_value = 0.0

        offset = 0
        component = 0

        for i in range(self._n_profiles):
            # Log-prior on integrated flux density
            ln_value += ln_likes.ln_log(p[offset + 0])
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

        for i in range(self._n_gaussians):
            # Log-prior on amplitude
            ln_value += ln_likes.ln_log(p[offset + 0])
            # Normal prior on center
            ln_value += ln_likes.ln_normal(p[offset + 1],
                                           self.v_center_mean[component],
                                           self.v_center_std[component])
            # Log-prior on dispersion
            ln_value += ln_likes.ln_log(p[offset + 2])

            offset += 3
            component += 1

        for i in range(self._n_baseline):
            # Normal prior on baseline coefficients
            ln_value += ln_likes.ln_normal(p[offset], 0.0, self.baseline_std)
            offset += 1

        # Log prior on posterior variance
        ln_value += ln_likes.ln_log(p[offset] * p[offset])

        return ln_value

    cdef double ln_likelihood(self, double[:] p):
        cdef:
            int i, offset
            double ln_value = 0.0

        self.reset_model()
        self.eval_profiles(p)
        self.eval_gaussians(p)
        self.eval_baseline(p)

        offset = 6 * self._n_profiles + 3 * self._n_gaussians + self._n_baseline

        for i in range(self.data.shape[0]):
            ln_value += ln_likes.ln_normal(self.data[i],
                                           self.model_array[i],
                                           p[offset])

        return ln_value

    def ln_posterior(self, double[:] p):
        cdef double ln_value = self.ln_bounds(p)
        
        if ln_value == 0.0:
            ln_value += self.ln_prior(p)
            ln_value += self.ln_likelihood(p)

        return ln_value