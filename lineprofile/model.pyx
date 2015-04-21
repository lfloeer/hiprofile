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

    def __init__(self, velocities, n_disks=1, n_gaussians=0, n_baseline=0, supersample=2):
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

        self.velocities = np.array(velocities, dtype=np.double, copy=True)

        self._v_low = self.velocities[0]
        self._supersample = supersample
        self._v_chan = (self.velocities[1] - self.velocities[0]) / float(self._supersample)
        self._N = self.velocities.shape[0] * self._supersample
        
        self.n_disks = n_disks
        self.n_gaussians = n_gaussians
        self.n_baseline = n_baseline

        # Dtau is twice as large as given in the paper
        self._dtau = M_PI / (self._N * self._v_chan)

        self._fft_output = fftw_alloc_real(self._N)
        self._fft_input = fftw_alloc_complex(self._N / 2 + 1)

        self.fft_output = <double[:self._N]> self._fft_output

        self._plan = fftw_plan_dft_c2r_1d(
            self._N,
            self._fft_input,
            self._fft_output,
            1) #FFTW MEASURE | DESTROY INPUT

        self.model_array = np.zeros(velocities.shape[0], dtype=np.double)


    def __dealloc__(self):
        """
        Free the input and output array
        during deallocation of the object.
        """
        fftw_free(self._fft_output)
        fftw_free(self._fft_input)
        fftw_destroy_plan(self._plan)
        fftw_cleanup()

    property model_array:

        def __get__(self):
            return np.asarray(self.model_array)

    property velocities:

        def __get__(self):
            return np.asarray(self.velocities)
    
    def model(self, double[::1] p, copy=True):
        self.eval_model(p)
        return np.array(self.model_array, copy=copy)

    cdef void eval_model(self, double[:] p):
        self.reset_model()
        self.eval_disks(p)
        self.eval_gaussians(p)
        self.eval_baseline(p)

    cdef void reset_model(self):
        cdef int i
        for i in range(self.model_array.shape[0]):
            self.model_array[i] = 0.0

    cdef void eval_disks(self, double[:] p):
        cdef int i
        if self.n_disks > 0:

            make_model(self._fft_input, self._N,
                       &p[0], self.n_disks,
                       self._dtau, self._v_chan, self._v_low)

            fftw_execute(self._plan)

            for i in range(self.model_array.shape[0]):
                self.model_array[i] += self.fft_output[i * self._supersample] / self._N

    cdef void eval_gaussians(self, double[:] p):
        
        cdef:
            int gaussian, i, offset
            double tmp, normalization, dispersion

        for gaussian in range(self.n_gaussians):

            offset = self.n_disks * 6 + gaussian * 3
            dispersion = 10 ** p[offset + 2]
            normalization = 10 ** p[offset + 0] / sqrt(2. * M_PI) / dispersion

            for i in range(self.model_array.shape[0]):
                tmp = (self.velocities[i] - p[offset + 1]) / dispersion
                tmp *= tmp
                self.model_array[i] += normalization * exp(-0.5 * tmp)

    cdef void eval_baseline(self, double[:] p):
        
        cdef:
            int order, i, offset
            double tmp, x, dx

        offset = self.n_disks * 6 + self.n_gaussians * 3
        x = -1.0
        dx = 2. / (self.velocities.shape[0] - 1.)

        for i in range(self.model_array.shape[0]):

            tmp = 0.

            for order in range(self.n_baseline):
                tmp = tmp * x + p[offset + order]

            self.model_array[i] += tmp
            x += dx
