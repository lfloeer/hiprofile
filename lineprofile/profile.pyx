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

cdef extern from 'math.h':
    double M_PI
    double j0(double a)
    double j1(double a)
    double exp(double a)

cdef extern from 'complex.h':
    complex cexp(complex a)

cdef extern from 'fftw3.h':
    
    ctypedef struct fftw_plan:
        pass

    double *fftw_alloc_real(size_t n) nogil
    complex *fftw_alloc_complex(size_t n) nogil
    void fftw_free(void *p) nogil
    void fftw_cleanup() nogil
    
    void fftw_destroy_plan(fftw_plan plan) nogil
    fftw_plan fftw_plan_dft_c2r_1d(
        int n, complex *input, double *output, unsigned flags) nogil
    void fftw_execute(const fftw_plan plan) nogil


cdef class LineModel:
    """
    Model for the global profiles of HI in galaxies
    after

    Stewart et al., 2014
    A simple model for global H i profiles of galaxies
    http://adsabs.harvard.edu/abs/2014A%26A...567A..61S
    """

    cdef:
        # FFTW3 related members
        double *_output_array
        complex *_input_array
        fftw_plan _plan

        double _dtau,
        double _v_high, _v_low, _v_chan
        int _supersample, _N
        int _n_profiles, _n_gaussians, _n_baseline

        np.ndarray _model_array
        np.ndarray _velo_array

        double[:] _model_array_view
        double[:] _velo_array_view
        complex[:] _input_view
        double[:] _output_view


    def __init__(self, velocities, n_profiles=1, n_gaussians=0, n_baseline=0, supersample=2):
        """
        Create a new model object on the given velocity grid

        Parameters
        ----------

        velocities : (N,) ndarray
            The velocities on which the model is to be sampled.
            The algorithm assumes the velocities to be
            sorted in increasing order.

        supersample : int, optional
            The degree to which the model is oversampled.
            Default: 2
        """
        
        self._v_high = velocities[velocities.shape[0] - 1]
        self._v_low = velocities[0]
        self._supersample = supersample
        self._v_chan = (velocities[1] - velocities[0]) / float(self._supersample)
        self._N = velocities.shape[0] * self._supersample
        
        self._n_profiles = n_profiles
        self._n_gaussians = n_gaussians
        self._n_baseline = n_baseline

        # Dtau is twice as large as given in the paper
        self._dtau = M_PI / (self._N * self._v_chan)

        self._output_array = fftw_alloc_real(self._N)
        self._input_array = fftw_alloc_complex(self._N / 2 + 1)

        self._input_view = <complex[:self._N / 2 + 1]> self._input_array
        self._output_view = <double[:self._N]> self._output_array

        self._plan = fftw_plan_dft_c2r_1d(
            self._N,
            self._input_array,
            self._output_array,
            1) #FFTW MEASURE | DESTROY INPUT

        self._velo_array = np.asarray(velocities, dtype=np.float64)
        self._velo_array_view = self._velo_array

        self._model_array = np.zeros(velocities.shape[0], dtype=np.float64)
        self._model_array_view = self._model_array


    def __dealloc__(self):
        """
        Free the input and output array
        during deallocation of the object.
        """

        fftw_destroy_plan(self._plan)
        fftw_free(self._output_array)
        fftw_free(self._input_array)
        fftw_cleanup()

    property model_array:

        def __get__(self):
            return self._model_array
    
    def model(self, double[:] p):
        """
        Evaluate the profile model for the given parameters
        and return the real-space representation

        p : (6,)-ndarray
            The parameters of the line model in the following
            order: 0) total_flux, 1) v_center, 2) v_width, 3) v_random,
            4) f_solid and 5) asymmetry.
        """
        self.reset_model()

        self.eval_profiles(p)
        self.eval_gaussians(p)
        self.eval_baseline(p)
        
        return self._model_array

    cdef void reset_model(self):
        cdef int i
        for i in range(self._model_array_view.shape[0]):
            self._model_array_view[i] = 0.0

    cdef void eval_profiles(self, double[:] p):
        cdef int i
        if self._n_profiles > 0:
            self.make_ft_model(p)
            self.transform_model()

            for i in range(self._model_array_view.shape[0]):
                self._model_array_view[i] += self._output_view[i * self._supersample]

    cdef void eval_gaussians(self, double[:] p):
        
        cdef:
            int gaussian, i, offset
            double tmp

        for gaussian in range(self._n_gaussians):

            offset = self._n_profiles * 6 + gaussian * 3

            for i in range(self._model_array_view.shape[0]):
                tmp = (self._velo_array_view[i] - p[offset + 1]) / p[offset + 2]
                tmp *= tmp
                self._model_array_view[i] += p[offset + 0] * exp(-0.5 * tmp)

    cdef void eval_baseline(self, double[:] p):
        
        cdef:
            int order, i, offset
            double tmp, x, dx

        offset = self._n_profiles * 6 + self._n_gaussians * 3
        x = -1.0
        dx = 2. / self._velo_array_view.shape[0]

        for i in range(self._model_array_view.shape[0]):

            tmp = 0.

            for order in range(self._n_baseline):
                tmp = tmp * x + p[offset + order]

            self._model_array_view[i] += tmp
            x += dx

    cdef void make_ft_model(self, double[:] p):
        """
        Populate the input array of the FFT with the
        fourier transform of the profile model given
        the parameters.
        """
        
        cdef:
            int i, profile, offset
            double phi, j0tau, j1tau, tau, j_tau, e
            complex bvalue, tmp

        for profile in range(self._n_profiles):

            offset = profile * 6

            phi = 2. * (p[offset + 1] - self._v_low) / p[offset + 2]

            for i in range(self._N / 2 + 1):
                
                if profile == 0:
                    self._input_view[i] = 0.

                tau = self._dtau * p[offset + 2] * i * -1.0
                j0tau = j0(tau)
                j1tau = j1(tau)
                
                # Approximations for singularities at tau == 0
                if tau == 0.:
                    j_tau = 0.5
                    e = 0.
                else:
                    j_tau = j1tau / tau
                    e = 1. / tau * (2. / tau * j1tau - j0tau)
                
                tmp = p[offset + 0] / self._v_chan * cexp(1.0j * phi * tau)
                
                bvalue = (1 - p[offset + 4]) * j0tau + 2. * p[offset + 4] * j_tau
                bvalue += 1.0j * p[offset + 5] * ((1 - p[offset + 4]) * j1tau + 2. * p[offset + 4] * e)

                tmp *= bvalue

                tmp *= exp(-2. * (p[offset + 3] / p[offset + 2] * tau) ** 2)

                self._input_view[i] += tmp

    cdef void transform_model(self):
        """
        Execute the fourier transform of the model
        and apply the necessary normalization.
        """

        fftw_execute(self._plan)
        
        # Normalize FFT
        for i in range(self._N):
            self._output_view[i] /= self._N


cdef class FitGaussian(LineModel):
    "Fit a LineModel to data assuming a Gaussian posterior"

    cdef:
        public double[:] data
        public double[:] weights

        # Prior parameters

        # Mean and center of the prior radial velocity
        # of each profile and gaussian.
        public double[:] v_center_mean
        public double[:] v_center_std

        public double min_turbulence


    def __init__(self, velocities, data, weights=None, **kwargs):

        super(self, FitGaussian).__init__(velocities, **kwargs)
        self.data = data

        if weights is None:
            self.weights = np.ones_like(data)
        else:
            self.weights = weights

        self.min_turbulence = 5.0

    cdef double ln_bounds(self, double[:] p):
        "Evaluate the hard bounds for each parameter"
        cdef:
            int i, offset
            double vmin, vmax

        offset = 0
        vmin = self._velo_array_view[0]
        vmax = self._velo_array_view[self._velo_array_view.shape[0] - 1]

        for i in range(self._n_profiles):
            # Positive integrated flux density
            if p[offset + 0] < 0.:
                return -inf
            # Profile fully in bounds
            if (p[offset + 1] - p[offset + 2] / 2.) < vmin or \
               (p[offset + 1] + p[offset + 2] / 2.) > vmax:
               return -inf
            # Positive rotation
            if p[offset + 2] < 0.:
                return -inf
            # Positive dispersion
            if p[offset + 3] <= self.min_turbulence:
                return -inf
            # Bounded solid rotating fraction
            if p[offset + 4] < 0. or p[offset + 4] > 1.:
                return -inf
            # Bounded asymmetry
            if p[offset + 5] < -1.0 or p[offset + 5] > 1.0:
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
            ln_value += ln_likes.ln_gamma(p[offset + 2], 4.38, 55.40)
            # Gamma prior on turbulent motion. Parameters roughly guided
            # by the paper of Stewart et al. 2014
            ln_value += ln_likes.ln_gamma(p[offset + 3] - self.min_turbulence, 5, 2)
            # Beta distribution on fraction of solid body rotation
            ln_value += ln_likes.ln_beta(p[offset + 4], 1, 5)
            # Beta distribution on asymmetry
            ln_value += ln_likes.ln_beta(0.5 * (p[offset + 5] + 1.0), 10, 10)

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
            ln_value += ln_likes.ln_normal(p[offset], 0.0, 1.0)
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
                                           self._model_array_view[i],
                                           p[offset])

        return ln_value

    def ln_posterior(self, double[:] p):
        cdef double ln_value = self.ln_bounds(p)
        
        if ln_value == 0.0:
            ln_value += self.ln_prior(p)
            ln_value += self.ln_likelihood(p)

        return ln_value
