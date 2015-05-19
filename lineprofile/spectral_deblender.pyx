#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
#cython: initializedcheck=False
#cython: embedsignature=True

STUFF = "Hi"

import numpy as np
cimport numpy as np

cimport cython

cdef class SpectralDeblender(fitter.FitMixture):
    """
    Deblend two or more sources from their point-spectra.

    The class uses multiple, constrained FitMixture classes to separate blended
    galaxy spectra. The constraints are as follows:
        1. The integrated flux density of each source has to be lower than the
           the integrated flux density in the spectrum that is exactly
           "on source". This assumes that the sources are mostly unresolved, as
           this assumption is violated for very extended sources.
        2. The difference in systemic velocity of each source between the
           multiple spectra is constrained to be small by using a Normal prior
           on the difference.
        3. Likewise, other parameters are constrained to be similar in each of
           the spectra. The strictness of these priors can be adjusted by the
           parameters.
    """

    def __init__(self, velocities, all_data, all_weights, **kwargs):

        super(SpectralDeblender, self).__init__(velocities,
                                                all_data[0],
                                                n_disks=all_data.shape[0],
                                                **kwargs)

        self.all_data = np.array(all_data, dtype=np.double, copy=True)
        self.all_weights = np.array(all_weights, dtype=np.double, copy=True)
        self.formatted_p = np.zeros(4 + self.n_disks * 6 + self.n_baseline,
                                    dtype=np.double)

        self.d_vsys_std = 10.0
        self.d_asym_std = 0.1
        self.d_turbulence_std = 1.0
        self.d_v_rot_std = 10.0
        self.d_fsolid_std = 0.1

        self.fix_parameters = 1

    cdef void setup_spectrum(self, double[::1] full_p, int spectrum):
        """
        Extract the parameter set for the given source spectrum.
        The first n_disks ** 2 values are the flux density mixing matrix in row
        order. The flux density for source i is at position (n_disks + 1) * i.
        The other entries give the flux of source i in spectrum j.

        Depending on the fix_parameters setting, the next parameters are the
        shape parameters of the sources.

        If fix_parameters is true, the next 5 * n_disks parameters are
            1. v_sys
            2. v_rot
            3. v_turb
            4. fsolid
            5. asym
        for each source. fix_parameters == False is not implemented at the
        moment but in this case, there would be 5 * n_disks ** 2 parameters,
        describing the shape parameters for each source in each spectrum. In this
        case, we put a prior on the difference between the parameters for the
        same source in each spectrum.

        The next n_baseline * n_disks parameters describe the baseline for
        each spectrum.

        The last n_disks * 4 parameters are the likelihood parameters
            1. fout
            2. std_in
            3. mu_out
            4. std_out
        for each spectrum.

        Calling this function sets up the formatted_p array to represent the
        parameters for a single spectrum with multiple sources. This is used
        in the functions deblending_ln_* to calculate prior, likelihood, and
        posterior probabilities by repeatedly calling the member functions
        ln_* and summing their results.
        """
        cdef:
            int flux_offset = spectrum * self.n_disks
            int n_fluxes = self.n_disks * self.n_disks
            int baseline_offset = n_fluxes + 5 * self.n_disks + spectrum * self.n_baseline
            int likelihood_offset = n_fluxes + 5 * self.n_disks + self.n_disks * self.n_baseline + 4 * spectrum
            int parameter = 0
            int source, i

        self.data = self.all_data[spectrum]
        self.weights = self.all_weights[spectrum]

        for source in range(self.n_disks):
            self.formatted_p[source * 6 + 0] = full_p[flux_offset + source]
            for i in range(5):
                self.formatted_p[source * 6 + i] = full_p[n_fluxes + source * 5 + i]

        for i in range(self.n_baseline):
            self.formatted_p[self.n_disks * 6 + i] = full_p[baseline_offset + i]

        for i in range(4):
            self.formatted_p[self.n_disks * 6 + self.n_baseline + i] = full_p[likelihood_offset + i]

    cpdef double deblending_ln_prior(self, double[::1] p):
        cdef:
            double ln_value = 0.0
            int spectrum

        for spectrum in range(self.n_disks):
            self.setup_spectrum(p, spectrum)
            ln_value += self.ln_prior(self.formatted_p)
        
        return ln_value

    cpdef double deblending_ln_likelihood(self, double[::1] p):
        cdef:
            double ln_value = 0.0
            int spectrum

        for spectrum in range(self.n_disks):
            self.setup_spectrum(p, spectrum)
            ln_value += self.ln_likelihood(self.formatted_p)

        return ln_value

    cpdef double deblending_ln_posterior(self, double[::1] p):
        cdef:
            double ln_value = 0.0
            int spectrum

        for spectrum in range(self.n_disks):
            self.setup_spectrum(p, spectrum)
            ln_value += self.ln_prior(self.formatted_p)
            ln_value += self.ln_likelihood(self.formatted_p)

        return ln_value


