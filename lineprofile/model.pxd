cdef extern from 'math.h' nogil:
    double M_PI
    double j0(double a)
    double j1(double a)
    double exp(double a)
    double sqrt(double a)

cdef extern from 'complex.h' nogil:
    complex cexp(complex a)

cdef extern from 'fftw3.h' nogil:

    ctypedef struct fftw_plan:
        pass

    double *fftw_alloc_real(size_t n)
    complex *fftw_alloc_complex(size_t n)
    void fftw_free(void *p)
    void fftw_cleanup()
    
    void fftw_destroy_plan(fftw_plan plan)
    fftw_plan fftw_plan_dft_c2r_1d(
        int n, complex *input, double *output, unsigned flags)
    void fftw_execute(const fftw_plan plan)

cdef extern from 'make_model.h' nogil:
    void make_model(complex *fft_input, int fft_size,
                    const double *parameters, int n_disks,
                    double d_tau, double v_chan, double v_low)

cdef class LineModel:
    
    cdef:
        # FFT related members
        fftw_plan _plan
        complex *_fft_input
        double *_fft_output
        double _dtau, _v_low, _v_chan
        int _supersample, _N
        
        readonly int n_disks, n_gaussians, n_baseline

        double[:] fft_output
        double[:] model_array
        double[:] velocities

    cdef void eval_model(self, double[:] p)
    cdef void reset_model(self)
    cdef void eval_disks(self, double[:] p)
    cdef void eval_gaussians(self, double[:] p)
    cdef void eval_baseline(self, double[:] p)
