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

        complex[:] _input_view
        double[:] _output_view


    def __init__(self, velocities, supersample=2):
        """
        Create a new model object on the given velocity grid

        Parameters
        ----------

        velocities : (N,) ndarray
            The velocities on which the model is to be sampled.
            Currently, the algorithm assumes the velocities to be
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
        self.make_ft_model(p)
        self.transform_model()
        
        return self._model_array[::self._supersample]


    cdef void make_ft_model(self, double[:] p):
        """
        Populate the input array of the FFT with the
        fourier transform of the profile model given
        the parameters.
        """
        
        cdef:
            int i, profile, n_profiles
            double phi, j0tau, j1tau, tau, j_tau, e
            complex bvalue, tmp

        n_profiles = p.shape[0] / 6

        for profile in range(n_profiles):

            phi = 2. * (p[1] - self._v_low) / p[2]

            for i in range(self._N / 2 + 1):
                
                if profile == 0:
                    self._input_view[i] = 0.

                tau = self._dtau * p[2] * i * -1.0
                j0tau = j0(tau)
                j1tau = j1(tau)
                
                # Approximations for singularities at tau == 0
                if tau == 0.:
                    j_tau = 0.5
                    e = 0.
                else:
                    j_tau = j1tau / tau
                    e = 1. / tau * (2. / tau * j1tau - j0tau)
                
                tmp = p[0] / self._v_chan * cexp(1.0j * phi * tau)
                
                bvalue = (1 - p[4]) * j0tau + 2. * p[4] * j_tau
                bvalue += 1.0j * p[5] * ((1 - p[4]) * j1tau + 2. * p[4] * e)

                tmp *= bvalue

                tmp *= exp(-2. * (p[3] / p[2] * tau) ** 2)

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
