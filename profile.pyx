#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

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


cdef class LineModel:

    cdef:
        np.ndarray _ft_model_values
        double _dtau,
        double _v_high, _v_low, _v_chan
        int _supersample, _N


    def __init__(self, velocities, supersample=2):
        
        self._v_high = velocities[velocities.size - 1]
        self._v_low = velocities[0]
        self._supersample = supersample
        self._v_chan = (velocities[1] - velocities[0]) / float(self._supersample)
        self._N = velocities.size * self._supersample

        self._ft_model_values = np.empty(self._N + 1, dtype=complex)
        self._dtau = M_PI / (2. * self._N * self._v_chan)


    property ft_model_values:
        def __get__(self):
            return self._ft_model_values

    
    def model(self, double[:] p):
        """
        Evaluate the profile model for the given parameters
        and return the real-space representation

        p : (6,)-ndarray
            The parameters of the line model in the following
            order: 0) total_flux, 1) v_center, 2) v_width, 3) v_random,
            4) f_solid and 5) asymmetry.
        """
        self.ft_model(p)
        m = np.fft.irfft(self._ft_model_values)
        
        return m[ : self._N : self._supersample]


    cdef void ft_model(self, double[:] p):
        
        cdef:
            complex[:] model = self._ft_model_values
            int i
            double phi, j0tau, j1tau, tau, j_tau, e
            complex bvalue

        phi = 2. * (p[1] - self._v_low) / p[2]

        for i in range(self._N + 1):
            
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
            
            model[i] = p[0] / self._v_chan * cexp(1.0j * phi * tau)
            
            bvalue = (1 - p[4]) * j0tau + 2. * p[4] * j_tau
            bvalue += 1.0j * p[5] * ((1 - p[4]) * j1tau + 2. * p[4] * e)

            model[i] *= bvalue

            model[i] *= exp(-2. * (p[3] / p[2] * tau) ** 2)