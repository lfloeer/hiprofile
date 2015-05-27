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
from libc.math cimport log, fabs, exp, sqrt, sin, cos, asin

cimport ln_likes

cdef class GaussianModel:

    def __init__(self, x_coord, y_coord, n_sources=2, n_baseline=2):
        self.x_coord = np.array(x_coord, dtype=np.double, copy=True)
        self.y_coord = np.array(y_coord, dtype=np.double, copy=True)

        assert self.x_coord.shape[0] == self.y_coord.shape[0]

        self.model_array = np.empty_like(x_coord, dtype=np.double)

    cdef void eval_model(self, double[::1] p):
        cdef int i

        for i in range(self.model_array.shape[0]):
            self.model_array[i] = 0.0

        self.eval_gaussians(p)
        self.eval_baseline(p)

    cdef void eval_gaussians(self, double[::1] p):
        pass

    cdef void eval_baseline(self, double[::1] p):
        pass

    def model(self, double[::1] p, copy=True):
        self.eval_model(p)
        return np.array(self.model_array, copy=copy)


cdef class Deblender(GaussianModel):

    def __init__(self, data, x_coord, y_coord, **kwargs):

        super(Deblender, self).__init__(x_coord, y_coord **kwargs)
        self.data = np.array(data, dtype=np.double, copy=True)

        self.amp_min = -3
        self.amp_max = 3
        
        self.disp_min = -3
        self.disp_max = 3

        self.std_baseline = 1

        self.fraction_min = -3
        self.fraction_max = 0
        
        self.std_in_min = -3
        self.std_in_max = 0

        self.std_out_min = -2
        self.std_out_max = 1

        self.mu_out_std = 1

    cdef double metric(self, double x1, double y1, double x2, double y2):
        """Euclidian metric"""
        cdef double dx = x1 - x2
        cdef double dy = y1 - y2
        return sqrt(dx * dx + dy * dy)

    cdef int likelihood_params_offset(self):
        return self.n_sources * 4 + self.n_baseline

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

    cdef double ln_prior_likelihood(self, double[::1] p):
        cdef int offset
        cdef double mu_out
        cdef double ln_value = 0.0

        offset = self.likelihood_params_offset()
        mu_out = p[offset + 2]

        ln_value += ln_likes.ln_normal(mu_out, 0., self.mu_out_std)

        return ln_value


cdef class SphericalDeblender(Deblender):

    cdef double metric(self, double lon1, double lat1, double lon2, double lat2):
        """Haversine formula"""
        cdef double sindlon = sin(fabs(lon1 - lon2) / 2.0)
        cdef double sindlat = sin(fabs(lat1 - lat2) / 2.0)
        return 2.0 * asin(sqrt(sindlat * sindlat + cos(lat1) * cos(lat2) * sindlon * sindlon))
