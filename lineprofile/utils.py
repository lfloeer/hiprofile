import numpy as np
import itertools as it


def sample_prior(n_sampler, fitter):
    def sample_components():
        """Get samples from prior on line profile"""
        for component_idx in range(fitter.n_disks):
            yield np.random.uniform(fitter.fint_min, fitter.fint_max, n_sampler)
            yield np.random.normal(fitter.v_center_mean[component_idx],
                                   fitter.v_center_std[component_idx],
                                   n_sampler)
            yield np.random.gamma(fitter.v_rot_k,
                                  fitter.v_rot_theta,
                                  n_sampler)
            yield fitter.turbulence_min + np.random.gamma(fitter.turbulence_k,
                                                          fitter.turbulence_theta,
                                                          n_sampler)
            yield np.random.beta(fitter.fsolid_p, fitter.fsolid_q, n_sampler)
            yield 2 * np.random.beta(fitter.asym_p, fitter.asym_q, n_sampler) - 1.0

    def sample_gaussians():
        """Get samples from prior on gaussians"""
        for component_idx in range(fitter.n_disks, fitter.n_disks + fitter.n_gaussians):
            yield np.random.uniform(fitter.fint_min, fitter.fint_max, n_sampler)
            yield np.random.normal(fitter.v_center_mean[component_idx],
                                   fitter.v_center_std[component_idx],
                                   n_sampler)
            yield np.random.uniform(fitter.gauss_disp_min, fitter.gauss_disp_max, n_sampler)

    def sample_baseline():
        """Get samples from prior on baseline"""
        for _ in range(fitter.n_baseline):
            yield np.random.normal(0, 0.1, n_sampler)

    def sample_likelihood():
        """Get samples from prior on posterior parameters"""
        yield np.random.uniform(fitter.fraction_min, fitter.fraction_max, n_sampler)
        yield np.random.uniform(fitter.std_in_min, fitter.std_in_max, n_sampler)
        yield np.random.normal(0., fitter.mu_out_std, n_sampler)
        yield np.random.uniform(fitter.std_out_min, fitter.std_out_max, n_sampler)

    prior_it = it.chain(sample_components(), sample_gaussians(), sample_baseline(), sample_likelihood())
    return np.array([samples for samples in prior_it]).T.copy()


def resample_position(position, n_walkers, n_dim, fitter, ball_size=1e-2):
    """Use rejection sampling to resample the walker positions"""
    scale_factors = np.ones(n_dim)
    scale_factors[3:6 * fitter.n_disks:6] = 10
    scale_factors[2:6 * fitter.n_disks:6] = 100
    scale_factors[1:6 * fitter.n_disks:6] = 10
    scale_factors *= ball_size

    new_positions = np.array([position + scale_factors * np.random.randn(n_dim)
                              for _ in xrange(n_walkers)])
    valid = np.array([np.isfinite(fitter.ln_prior(p))
                      for p in new_positions])

    for _ in xrange(20):
        n_invalid = np.sum(~valid)
        if n_invalid == 0:
            break
        new_positions[~valid] = np.array([position + ball_size * np.random.randn(n_dim)
                                          for _ in xrange(n_invalid)])
        valid[~valid] = np.array([np.isfinite(fitter.ln_prior(p))
                                  for p in new_positions[~valid]])

    return new_positions
