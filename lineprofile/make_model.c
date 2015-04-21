#include "make_model.h"

void make_model(double complex *fft_input, int fft_size,
                const double *parameters, int n_disks,
                double d_tau, double v_chan, double v_low) {

    int offset;
    double j0tau, j1tau, tau, j_tau, e;
    double fint, vsys, vrot, vturbrot, fsolid, asym, tmp2;
    double two_fsolid, fdiff;
    double b_real, b_imag;
    double complex tmp, phi;

    int n_values = fft_size / 2 + 1;

    for (int i = 0; i < n_values; ++i)
    {
        fft_input[i] = 0;
    }

    for (int profile = 0; profile < n_disks; ++profile)
    {
        offset = profile * 6;
        fint = pow(10.0, parameters[offset + 0]);
        vsys = parameters[offset + 1];
        vrot = parameters[offset + 2];
        vturbrot = parameters[offset + 3] / vrot;
        fsolid = parameters[offset + 4];
        asym = parameters[offset + 5];
        
        two_fsolid = 2. * fsolid;
        fdiff = 1 - fsolid;
        phi = _Complex_I * 2. * (vsys - v_low) / vrot;

        fft_input[0] += fint / v_chan;

        for (int i = 1; i < n_values; ++i)
        {
            tau = d_tau * vrot * i * -1.0;
            j0tau = j0(tau);
            j1tau = j1(tau);

            j_tau = j1tau / tau;
            e = 1. / tau * (2. / tau * j1tau - j0tau);

            tmp = fint / v_chan * cexp(phi * tau);

            b_real = fdiff * j0tau + two_fsolid * j_tau;
            b_imag = asym * (fdiff * j1tau + two_fsolid * e);

            tmp *= b_real + _Complex_I * b_imag;

            tmp2 = vturbrot * tau;
            tmp2 *= tmp2;

            tmp *= exp(-2. * tmp2);

            fft_input[i] += tmp;
        }
    }
}