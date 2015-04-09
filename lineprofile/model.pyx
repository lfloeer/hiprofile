#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False
#cython: embedsignature=True

import numpy as np
cimport numpy as np
cimport cython

cdef class LineModel:
    """
    Model for the global profiles of HI in galaxies
    after

    Stewart et al., 2014
    A simple model for global H i profiles of galaxies
    http://adsabs.harvard.edu/abs/2014A%26A...567A..61S
    """

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

        self.velocities = velocities
        
        self._v_high = self.velocities[self.velocities.shape[0] - 1]
        self._v_low = self.velocities[0]
        self._supersample = supersample
        self._v_chan = (self.velocities[1] - self.velocities[0]) / float(self._supersample)
        self._N = self.velocities.shape[0] * self._supersample
        
        self._n_profiles = n_profiles
        self._n_gaussians = n_gaussians
        self._n_baseline = n_baseline

        # Dtau is twice as large as given in the paper
        self._dtau = M_PI / (self._N * self._v_chan)

        self._fft_output = fftw_alloc_real(self._N)
        self._fft_input = fftw_alloc_complex(self._N / 2 + 1)

        self.fft_input = <complex[:self._N / 2 + 1]> self._fft_input
        self.fft_output = <double[:self._N]> self._fft_output

        self._plan = fftw_plan_dft_c2r_1d(
            self._N,
            self._fft_input,
            self._fft_output,
            1) #FFTW MEASURE | DESTROY INPUT

        self.model_array = np.zeros(velocities.shape[0], dtype=np.float64)


    def __dealloc__(self):
        """
        Free the input and output array
        during deallocation of the object.
        """

        fftw_destroy_plan(self._plan)
        fftw_free(self._fft_output)
        fftw_free(self._fft_input)
        fftw_cleanup()

    property model_array:

        def __get__(self):
            return np.asarray(self.model_array)

    property velocities:

        def __get__(self):
            return np.asarray(self.velocities)

    property n_profiles:
        def __get__(self):
            return self._n_profiles

    property n_gaussians:
        def __get__(self):
            return self._n_gaussians

    property n_baseline:
        def __get__(self):
            return self._n_baseline
    
    def model(self, double[:] p):
        """
        Evaluate the profile model for the given parameters
        and return the real-space representation

        p : (6,)-ndarray
            The parameters of the line model in the following
            order: 0) total_flux, 1) v_center, 2) v_width, 3) v_random,
            4) f_solid and 5) asymmetry.
        """
        self.eval_model(p)
        
        return np.asarray(self.model_array)

    cdef void eval_model(self, double[:] p):
        self.reset_model()
        self.eval_profiles(p)
        self.eval_gaussians(p)
        self.eval_baseline(p)

    cdef void reset_model(self):
        cdef int i
        for i in range(self.model_array.shape[0]):
            self.model_array[i] = 0.0

    cdef void eval_profiles(self, double[:] p):
        cdef int i
        if self._n_profiles > 0:
            self.make_ft_model(p)
            self.transform_model()

            for i in range(self.model_array.shape[0]):
                self.model_array[i] += self.fft_output[i * self._supersample]

    cdef void eval_gaussians(self, double[:] p):
        
        cdef:
            int gaussian, i, offset
            double tmp

        for gaussian in range(self._n_gaussians):

            offset = self._n_profiles * 6 + gaussian * 3

            for i in range(self.model_array.shape[0]):
                tmp = (self.velocities[i] - p[offset + 1]) / p[offset + 2]
                tmp *= tmp
                self.model_array[i] += p[offset + 0] * exp(-0.5 * tmp)

    cdef void eval_baseline(self, double[:] p):
        
        cdef:
            int order, i, offset
            double tmp, x, dx

        offset = self._n_profiles * 6 + self._n_gaussians * 3
        x = -1.0
        dx = 2. / (self.velocities.shape[0] - 1.)

        for i in range(self.model_array.shape[0]):

            tmp = 0.

            for order in range(self._n_baseline):
                tmp = tmp * x + p[offset + order]

            self.model_array[i] += tmp
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
            double fint, tmp2
            complex bvalue, tmp

        for profile in range(self._n_profiles):

            offset = profile * 6

            phi = 2. * (p[offset + 1] - self._v_low) / p[offset + 2]
            fint = 10.0 ** p[offset + 0]

            for i in range(self._N / 2 + 1):
                
                if profile == 0:
                    self.fft_input[i] = 0.

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
                
                tmp = fint / self._v_chan * cexp(1.0j * phi * tau)
                
                bvalue = (1 - p[offset + 4]) * j0tau + 2. * p[offset + 4] * j_tau
                bvalue += 1.0j * p[offset + 5] * ((1 - p[offset + 4]) * j1tau + 2. * p[offset + 4] * e)

                tmp *= bvalue

                tmp2 = (p[offset + 3] / p[offset + 2] * tau)
                tmp2 *= tmp2

                tmp *= exp(-2. * tmp2)

                self.fft_input[i] += tmp

    cdef void transform_model(self):
        """
        Execute the fourier transform of the model
        and apply the necessary normalization.
        """

        fftw_execute(self._plan)
        
        # Normalize FFT
        for i in range(self._N):
            self.fft_output[i] /= self._N
