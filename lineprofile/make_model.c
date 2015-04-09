#include "make_model.h"

void make_model(_Complex double *fft_input, double *fft_output, int fft_size, fftw_plan plan,
                double *parameters, int n_profiles,
                double d_tau, double v_chan, double v_low) {

    int offset;
    double phi, j0tau, j1tau, tau, j_tau, e;
    double fint, vsys, vrot, vturb, fsolid, asym, tmp2;
    _Complex double bvalue, tmp;

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

        for (int i = 0; i < fft_size / 2 + 1; ++i)
        {
            if (profile == 0)
            {
                fft_input[i] = 0;
            }

            tau = d_tau * vrot * i * -1.0;
            j0tau = j0(tau);
            j1tau = j1(tau);

            if (tau == 0.0)
            {
                j_tau = 0.5;
                e = 0.;
            } else {
                j_tau = j1tau / tau;
                e = 1. / tau * (2. / tau * j1tau - j0tau);
            }

            tmp = fint / v_chan * cexp(_Complex_I * phi * tau);

            bvalue = (1 - fsolid) * j0tau + 2. * fsolid * j_tau;
            bvalue += _Complex_I * asym * ((1 - fsolid) * j1tau + 2. * fsolid * e);

            tmp *= bvalue;

            tmp2 = (vturb / vrot * tau);
            tmp2 *= tmp2;

            tmp *= exp(-2. * tmp2);

            fft_input[i] += tmp;
        }
    }

    fftw_execute(plan);

    for (int i = 0; i < fft_size; ++i)
    {
        fft_output[i] /= fft_size;
    }
}