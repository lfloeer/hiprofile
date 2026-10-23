# Configuration reference

Generated from `hiprofile.config` by `docs/gen_config_reference.py`. Every key is optional.
The validated configuration, the code version and the seed are written into every result file.

## Fit configuration (`hiprofile fit --config`)

| key | type | default | description |
|---|---|---|---|
| `model.n_max` | `int` | `2` | Largest number of profile components |
| `model.k_min` | `int` | `0` |  |
| `model.k_max` | `int` | `3` | Largest Chebyshev baseline order |
| `model.n_rfi` | `int` | `0` | Narrow RFI components added to every model |
| `model.poisson_rate` | `float` | `0.5` | Poisson prior mean of N |
| `model.parametrization` | `Literal['direct', 'effective']` | `direct` |  |
| `model.response` | `str` | `boxcar` |  |
| `model.supersample` | `int \| None` | `null` |  |
| `priors.v_ctr` | `Range` | `{'kind': 'uniform'}` | Line centre [km/s]. Uniform bounds default to the band minus `v_margin`. |
| `priors.v_margin` | `float` | `0.05` | Band fraction excluded at each edge |
| `priors.delta_v` | `Range` | `{'kind': 'uniform', 'low': 0.0, 'high': 600.0}` |  |
| `priors.delta_v_rand` | `Range` | `{'kind': 'loguniform', 'low': 4.0, 'high': 60.0}` |  |
| `priors.alpha` | `Range` | `{'kind': 'uniform', 'low': -1.0, 'high': 1.0}` |  |
| `priors.amplitude_scale` | `Union[float, Literal['auto']]` | `auto` | Std of the half-normal prior on a_ring and a_disk [flux * km/s]. Fix it for a survey so that evidences are comparable between spectra. |
| `priors.baseline_scale` | `Union[float, Literal['auto']]` | `auto` | Std of the Chebyshev coefficients; auto = 10 x noise rms |
| `priors.baseline_mean` | `list[float] \| None` | `null` |  |
| `priors.baseline_cov` | `list[list[float]] \| None` | `null` |  |
| `priors.rfi_sigma_channels` | `tuple[float, float]` | `(0.3, 1.0)` | Width range of RFI components in channels |
| `priors.noise_scale` | `tuple[float, float] \| None` | `null` | Fit a global noise scale factor with log-uniform prior in this range |
| `noise.sigma` | `Literal['auto', 'input', 'estimate']` | `auto` | auto: use the input sigma column if present, else estimate |
| `noise.correlation` | `str \| list[float]` | `auto` | white \| hanning \| hanning-decimated \| [rho_1, rho_2, ...]; auto: hanning if the response contains hanning |
| `noise.signal_free` | `list[tuple[float, float]] \| None` | `null` | Velocity ranges used to estimate sigma |
| `rfi.prefit` | `bool` | `True` |  |
| `rfi.background` | `Literal['positive', 'symmetric']` | `positive` |  |
| `rfi.threshold` | `float` | `0.5` |  |
| `rfi.action` | `Literal['mask', 'inflate']` | `mask` |  |
| `rfi.n_iter` | `int` | `8` |  |
| `rfi.flag_fraction` | `float` | `0.05` |  |
| `sampler.mode` | `Literal['quick', 'full', 'check']` | `full` |  |
| `sampler.backend` | `Literal['nautilus', 'dynesty', 'ultranest', 'smc', 'quick']` | `nautilus` |  |
| `sampler.check_backend` | `Literal['nautilus', 'dynesty', 'ultranest', 'smc']` | `dynesty` |  |
| `sampler.n_live` | `int` | `400` |  |
| `sampler.baseline_evidence` | `Literal['reweight', 'rerun']` | `reweight` | reweight: sample once per N at the preferred K and importance-reweight to the other K (rerun when the effective sample size is too small) |
| `sampler.reweight_min_ess` | `float` | `0.2` |  |
| `sampler.options` | `dict[str, Any]` | `{}` |  |
| `sampler.likelihood_backend` | `Literal['numpy', 'jax']` | `numpy` |  |
| `diagnostics.ppp_min` | `float` | `0.01` |  |
| `diagnostics.wiggle_max` | `float` | `0.1` |  |
| `diagnostics.acf_sigma` | `float` | `4.0` |  |
| `diagnostics.edge_mass` | `float` | `0.2` |  |
| `diagnostics.min_ess` | `float` | `500.0` |  |
| `diagnostics.logz_sigma` | `float` | `3.0` |  |
| `diagnostics.negative_fraction` | `float` | `0.5` | Flag when this fraction of amplitude draws is negative. Narrow profiles make ring and disk nearly degenerate, so 10 % (roadmap value) flags most of them. |
| `diagnostics.multimodal_weight` | `float` | `0.1` |  |
| `output.n_posterior` | `int` | `2000` | Posterior samples stored per model |
| `output.plots` | `Literal['none', 'flagged', 'sample', 'all']` | `sample` |  |
| `output.plot_fraction` | `float` | `0.02` |  |
| `output.catalog_format` | `Literal['fits', 'parquet', 'csv']` | `fits` |  |
| `seed` | `int` | `20141` |  |

Prior ranges (`priors.v_ctr`, `priors.delta_v`, ...) take `kind` (`uniform`, `loguniform`,
`normal`), `low`/`high` or `mean`/`std`.

## Simulation configuration (`hiprofile simulate --config`)

| key | type | default | description |
|---|---|---|---|
| `n_chan` | `int` | `512` |  |
| `dv` | `float` | `5.0` |  |
| `v0` | `float` | `500.0` |  |
| `response` | `str` | `boxcar` |  |
| `supersample` | `int` | `8` |  |
| `n_components` | `int \| list[float]` | `1` |  |
| `profiles` | `Literal['model', 'template', 'mixed']` | `model` |  |
| `template_files` | `list[str]` | `[]` |  |
| `priors` | `Priors` | `{'v_ctr': {'kind': 'uniform'}, 'v_margin': 0.05, 'delta_v': {'kind': 'uniform', 'low': 0.0, 'high': 600.0}, 'delta_v_rand': {'kind': 'loguniform', 'low': 4.0, 'high': 60.0}, 'alpha': {'kind': 'uniform', 'low': -1.0, 'high': 1.0}, 'amplitude_scale': 'auto', 'baseline_scale': 'auto', 'rfi_sigma_channels': (0.3, 1.0)}` |  |
| `flux` | `Literal['prior', 'snr']` | `snr` |  |
| `snr_range` | `tuple[float, float]` | `(3.0, 100.0)` |  |
| `f_range` | `tuple[float, float]` | `(0.0, 1.0)` |  |
| `noise_sigma` | `float` | `0.01` |  |
| `noise_correlation` | `Literal['white', 'hanning']` | `white` |  |
| `baseline` | `Literal['none', 'chebyshev', 'sine']` | `none` |  |
| `baseline_order` | `int` | `1` |  |
| `baseline_scale` | `float` | `0.01` |  |
| `sine_amplitude` | `float` | `0.5` |  |
| `sine_period` | `tuple[float, float]` | `(100.0, 600.0)` |  |
| `rfi` | `Literal['none', 'spikes', 'broad']` | `none` |  |
| `rfi_rate` | `float` | `0.01` |  |
| `rfi_amplitude` | `tuple[float, float]` | `(5.0, 50.0)` |  |
| `write_sigma` | `bool` | `True` |  |
