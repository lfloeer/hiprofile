#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False

STUFF = "Hi"

import numpy as np

cimport numpy as np
cimport cython

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
        
        np.ndarray _model_array
        double _dtau,
        double _v_high, _v_low, _v_chan
        int _supersample, _N
        int _n_profiles, _n_gaussians, _n_baseline

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
        
        self._v_high = velocities[velocities.size - 1]
        self._v_low = velocities[0]
        self._supersample = supersample
        self._v_chan = (velocities[1] - velocities[0]) / float(self._supersample)
        self._N = velocities.size * self._supersample
        
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

        self._model_array = np.asarray(<double[:self._N]>self._output_array)


    def __dealloc__(self):
        """
        Free the input and output array
        during deallocation of the object.
        """

        fftw_free(self._output_array)
        fftw_free(self._input_array)
        fftw_destroy_plan(self._plan)
        fftw_cleanup()

    
    def model(self, double[:] p):
        """
        Evaluate the profile model for the given parameters
        and return the real-space representation

        p : (6,)-ndarray
            The parameters of the line model in the following
            order: 0) total_flux, 1) v_center, 2) v_width, 3) v_random,
            4) f_solid and 5) asymmetry.
        """
        self.reset_output()
        self.eval_profiles(p)
        self.eval_gaussians(p)
        self.eval_baseline(p)
        
        return self._model_array[::self._supersample]

    cdef void reset_output(self):
        cdef int i
        # Reset output array if no FFT is performed
        if self._n_profiles == 0:
            for i in range(self._N):
                self._output_array[i] = 0.0

    cdef void eval_profiles(self, double[:] p):
        if self._n_profiles > 0:
            self.make_ft_model(p)
            self.transform_model()

    cdef void eval_gaussians(self, double[:] p):
        
        cdef:
            int gaussian, i, offset
            double velocity, tmp

        for gaussian in range(self._n_gaussians):

            offset = self._n_profiles * 6 + gaussian * 3

            for i from 0 <= i < self._N by self._supersample:

                velocity = i * self._v_chan * self._supersample + self._v_low
                tmp = velocity - p[offset + 1]
                tmp *= tmp
                self._output_array[i] += p[offset + 0] * exp(-0.5 * tmp / p[offset + 2] / p[offset + 2])

    cdef void eval_baseline(self, double[:] p):
        pass

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
