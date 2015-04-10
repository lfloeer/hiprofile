#include "make_model.h"

void make_model(double complex *fft_input, int fft_size,
                double *parameters, int n_profiles,
                double d_tau, double v_chan, double v_low) {

    int offset;
    double phi, j0tau, j1tau, tau, j_tau, e;
    double fint, vsys, vrot, vturb, fsolid, asym, tmp2;
    double b_real, b_imag;
    double complex tmp;

    int n_values = fft_size / 2 + 1;

    for (int i = 0; i < n_values; ++i)
    {
        fft_input[i] = 0;
    }

    for (int profile = 0; profile < n_profiles; ++profile)
    {
        offset = profile * 6;
        fint = pow(10.0, parameters[offset + 0]);
        vsys = parameters[offset + 1];
        vrot = parameters[offset + 2];
        vturb = parameters[offset + 3];
        fsolid = parameters[offset + 4];
        asym = parameters[offset + 5];

        phi = 2. * (vsys - v_low) / vrot;

        fft_input[0] += fint / v_chan;

        for (int i = 1; i < n_values; ++i)
        {

            tau = d_tau * vrot * i * -1.0;
            j0tau = j0(tau);
            j1tau = j1(tau);

            j_tau = j1tau / tau;
            e = 1. / tau * (2. / tau * j1tau - j0tau);

            tmp = fint / v_chan * cexp(_Complex_I * phi * tau);

            b_real = (1 - fsolid) * j0tau + 2. * fsolid * j_tau;
            b_imag = asym * ((1 - fsolid) * j1tau + 2. * fsolid * e);

            tmp *= b_real + _Complex_I * b_imag;

            tmp2 = (vturb / vrot * tau);
            tmp2 *= tmp2;

            tmp *= exp(-2. * tmp2);

            fft_input[i] += tmp;
        }
    }
}