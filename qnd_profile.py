import numpy as np
import scipy.special as spsp
import pylab as pl
from numba import njit, jit
import math

def e_approx(tau, j0tau, j1tau):
    return np.where(tau == 0., 0., 1. / tau * (2. / tau * j1tau - j0tau))


def j_tau_approx(tau, j1tau):
    return np.where(tau == 0., 0.5, j1tau / tau)


def z_intrinsic(v_chan, v_low, v_center, v_width, f_solid, flux, asymmetry):

    ### Paper says v_high - v_center!

    phi = 2. * (v_center - v_low) / v_width

    def ztau(tau):

        j0tau = spsp.j0(tau).astype(np.complex)
        j1tau = spsp.j1(tau).astype(np.complex)

        ### Paper says 2 * flux !
        
        value = flux / v_chan * np.exp(1.0j * phi * tau)
        bvalue = (1 - f_solid) * j0tau + 2. * f_solid * j_tau_approx(tau, j1tau)
        bvalue += 1.0j * asymmetry * ((1 - f_solid) * j1tau + 2. * f_solid * e_approx(tau, j0tau, j1tau))
        value *= bvalue

        return value

    return ztau


def dispersive_motion(v_random, v_width):

    def gtau(tau):

        return np.exp(-2. * (v_random / v_width * tau)**2)

    return gtau


def line_model(velocities, supersample=2):

    v_high = velocities[-1]
    v_low = velocities[0]
    v_chan = abs(velocities[1] - velocities[0]) / float(supersample)
    N = float(velocities.size) * supersample

    _dtau = np.pi / (2. * N * v_chan)
    sample_points = np.arange(N + 1) * -1

    def model(p, full=False):
        # p: total_flux, v_center, v_width, v_random, f_solid, asymmetry

        dtau = _dtau * p[2]
        tau_samples = sample_points * dtau

        ztau = z_intrinsic(v_chan, v_low, p[1], p[2], p[4], p[0], p[5])
        gtau = dispersive_motion(p[3], p[2])

        ftvals = ztau(tau_samples) * gtau(tau_samples)
        m = np.fft.irfft(ftvals)[:N:supersample]
        
        if full:
            return m, ftvals
        else:
            return m 

    return model


if __name__ == '__main__':

    channels = 512
    velocities = np.arange(channels) * 10.24 / 10.

    v_center = np.mean(velocities)
    v_width = 100.
    v_random = 10.
    total_flux = 5
    f_solid = 0.1
    asymmetry = -0.2

    p0 = np.array([total_flux, v_center, v_width, v_random, f_solid, asymmetry])

    model = line_model(velocities)

    data = model(p0) + np.random.normal(0, 0.023, channels)

    pltdict = {'drawstyle' : 'steps-mid'}
    pl.plot(velocities, model(p0), **pltdict)
    pl.plot(velocities, data, **pltdict)

    from scipy.optimize import leastsq, fmin_l_bfgs_b, minimize

    ln_likelihood = lambda p: -0.5 * np.power((data - model(p)) / 0.023, 2).sum()
    
    @jit(nopython=True)
    def ln_prior(p):
        l = 0
        if not p[0] > 0:
            return -np.inf
        else:
            l += p[0] ** -1
        
        if not 250 < p[1] < 750:
            return -np.inf
        
        if not 0 < p[2] < 1000:
            return -np.inf
        
        if not 5 < p[3] < 30:
            return -np.inf
        
        if not 0 <= p[4] <= 1:
            return -np.inf
        else:
            l += p[4] ** (1. - 1.) * (1 - p[4]) ** (3. - 1.)
        
        if not -1 <= p[5] <= 1:
            return -np.inf
        else:
            pt = (p[5] + 1) * 0.5
            l += pt ** (2.5 - 1) * (1 - pt) ** (2.5 - 1)
        
        return l

    
    def ln_posterior(p):
        prior = ln_prior(p)
        if np.isfinite(prior):
            return prior + ln_likelihood(p)
        else:
            return -np.inf

    pstart = np.array(p0)
    pstart[1] -= 70
    pstart[2] = 100.
    pstart[-2:] = 0.

    result = minimize(
        lambda p: -1 * ln_likelihood(p),
        pstart,
        bounds=[(0,None),(250,750),(0,1000),(5,30),(0,0.5),(-0.5,0.5)],
        method='l-bfgs-b')

    import emcee

    n_dim, n_walkers = 6, 500
    sampler_p0 = [np.random.randn(n_dim) * 0.01 + result.x for i in xrange(n_walkers)]

    print result
    pf = result.x

    pl.plot(velocities, model(pf), **pltdict)

    if True:
        sampler = emcee.EnsembleSampler(n_walkers, n_dim, ln_posterior, threads=4)

        sampler.run_mcmc(sampler_p0, 750)

        chain_r = sampler.chain[:,600:,:].reshape((-1, n_dim))
        
        for s in np.random.permutation(chain_r)[:100]:
            pl.plot(velocities, model(s), color='k', alpha=0.1)


        for i in xrange(n_dim):
            pl.figure()
            pl.plot(sampler.chain[:,:,i].T, alpha=0.1, color='k')
            pl.axhline(p0[i], color='r')

        pl.figure()
        pl.plot(sampler.lnprobability.T, alpha=0.1, color='k')


    pl.show()
