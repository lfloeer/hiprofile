cdef extern from 'fftw3.h':
    
    ctypedef struct fftw_plan:
        pass

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

        readonly double[:] model_array
        readonly double[:] velocities

    cdef void reset_model(self)
    cdef void eval_profiles(self, double[:] p)
    cdef void eval_gaussians(self, double[:] p)
    cdef void eval_baseline(self, double[:] p)

    cdef void make_ft_model(self, double[:] p)
    cdef void transform_model(self)
