import numpy as np
import emcee

from lineprofile.model import LineModel
from lineprofile.fitter import FitMixture
from lineprofile.utils import sample_prior

import matplotlib.pyplot as plt


def make_data():
    """
    Generate some fake spectral line data using LineModel.
    """
    n_channels = 512
    dv_channel = 10.24

    velocities = np.arange(n_channels) * dv_channel

    parameters = np.array([
        np.log10(30.0),  # 30. Jy.km/s integrated flux density
        np.mean(velocities),  # Center the profile in the data
        230.0,  # 230.0 km/s rotational broadening
        15.0,  # 15.0 km/s turbulent broadening
        0.2,  # 20 % solid body rotation
        -0.1,  # Slight asymmetry to lower velocities
    ])

    model = LineModel(velocities, n_disks=1, n_baseline=0)

    data = model.model(parameters)

    # Add some thermal noise to the data
    data += np.random.normal(0, 0.023, n_channels)

    # Insert 10 corrupted channels
    corrupted_channels = np.random.random_integers(0, n_channels, 10)
    data[corrupted_channels] += np.random.rayleigh(0.1, 10)

    return velocities, data


def plot_data(velocities, data):
    """Plot the data"""
    plt.plot(velocities, data, drawstyle='steps-mid', color='k')
    plt.show()


def plot_chains(sampler):
    """Visualize the MCMC chains from the sampler object"""
    fig, axes = plt.subplots(2, 5, sharex=True,
                             figsize=(10, 5),
                             tight_layout=True)

    for param, ax in enumerate(axes.flat):
        ax.plot(
            sampler.chain[::10, ::10, param],
            drawstyle='steps-mid',
            color='k',
            alpha=0.1)

    plt.show()


def plot_model(velocities, data, sampler):
    """Visualize the posterior draws of the model"""
    model = LineModel(velocities)
    model_samples = np.array([model.model(p) for p in sampler.chain[-1, :, :]])

    model_p16, model_p50, model_p84 = np.percentile(model_samples,
                                                    [16., 50., 84.],
                                                    axis=0)

    plt.plot(velocities, data, drawstyle='steps-mid', color='k')
    plt.plot(velocities, model_p50, color='r')
    plt.fill_between(velocities, model_p16, model_p84, color='r', alpha=.25)

    plt.show()


def fit_data(velocities, data):
    """Fit the model to the data using the emcee sampler"""
    fitter = FitMixture(velocities, data, n_disks=1, n_baseline=0)

    # Set the prior parameters for each disk to be fit
    fitter.v_center_mean = [np.mean(velocities)]
    fitter.v_center_std = [30.]

    sampler = emcee.EnsembleSampler(
        nwalkers=500,
        dim=10,
        lnpostfn=fitter.ln_posterior)

    pos0 = sample_prior(500, fitter)

    sampler.run_mcmc(pos0, 500)

    return sampler


def main():

    velocities, data = make_data()

    plot_data(velocities, data)

    sampler = fit_data(velocities, data)

    plot_chains(sampler)
    plot_model(velocities, data, sampler)


if __name__ == '__main__':
    main()
