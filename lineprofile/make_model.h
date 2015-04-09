#include "math.h"
#include "complex.h"
#include "fftw3.h"

void make_model(_Complex double *fft_input, double *fft_output, int fft_size, fftw_plan plan,
                double *parameters, int n_profiles,
                double d_tau, double v_chan, double v_low);
