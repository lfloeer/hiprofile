#include "math.h"
#include "complex.h"
#include "fftw3.h"

void make_model(double complex *fft_input, int fft_size,
                const double *parameters, int n_profiles,
                double d_tau, double v_chan, double v_low);
