cdef extern from 'math.h':
    double M_PI
    double j0(double a) nogil
    double j1(double a) nogil
    double exp(double a) nogil

cdef extern from 'complex.h':
    complex cexp(complex a) nogil

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
    
    cdef:
        # FFTW3 related members
        fftw_plan _plan

        complex *_fft_input
        complex[:] fft_input

        double *_fft_output
        double[:] fft_output

        double _dtau,
        double _v_high, _v_low, _v_chan
        int _supersample, _N
        int _n_profiles, _n_gaussians, _n_baseline

        double[:] model_array
        double[:] velocities

    cdef void eval_model(self, double[:] p)
    cdef void reset_model(self)
    cdef void eval_profiles(self, double[:] p)
    cdef void eval_gaussians(self, double[:] p)
    cdef void eval_baseline(self, double[:] p)

    cdef void make_ft_model(self, double[:] p)
    cdef void transform_model(self)
