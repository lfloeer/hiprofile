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
| `priors.delta_v` | `Range` | `{'kind': 'lognormal', 'low': 0.0, 'high': 800.0, 'mean': 2.2, 'std': 0.4}` | Line width [km/s]: log-normal around 160 km/s, 0.4 dex (THINGS: 0.1-566 km/s, median 153) |
| `priors.delta_v_rand` | `Range` | `{'kind': 'lognormal', 'low': 4.0, 'high': 60.0, 'mean': 1.1, 'std': 0.2}` | Random motions [km/s]: log-normal around 12.6 km/s (THINGS: 8-32 km/s) |
| `priors.alpha` | `Range` | `{'kind': 'normal', 'low': -1.0, 'high': 1.0, 'mean': 0.0, 'std': 0.25}` | Asymmetry: normal(0, 0.25) truncated to [-1, 1] (THINGS: median 0.00, robust std 0.14) |
| `priors.f` | `Range` | `{'kind': 'beta', 'low': 0.0, 'high': 1.0, 'a': 1.0, 'b': 3.0}` | Solid-body fraction: Beta(1, 3), density 3 (1 - f)^2, median 0.21, P(f > 0.7) = 2.7 %; fitted to the 34 THINGS galaxies (median 0.15, 3 % above 0.7), which reject a uniform prior |
| `priors.amplitude_scale` | `Union[float, Literal['auto']]` | `auto` | Std of the half-normal prior on the integrated flux S of each component [flux * km/s]. Fix it for a survey so that evidences are comparable between spectra. |
| `priors.baseline_scale` | `Union[float, Literal['auto']]` | `auto` | Std of the Chebyshev coefficients; auto = 10 x noise rms |
| `priors.baseline_mean` | `list[float] \| None` | `null` |  |
| `priors.baseline_cov` | `list[list[float]] \| None` | `null` |  |
| `priors.rfi_sigma_channels` | `tuple[float, float]` | `(0.3, 1.0)` | Width range of RFI components in channels |
| `priors.noise_scale` | `tuple[float, float] \| None` | `null` | Fit a global noise scale factor with log-uniform prior in this range |
| `noise.sigma` | `Literal['auto', 'input', 'estimate']` | `auto` | auto: use the input sigma column if present, else estimate |
| `noise.correlation` | `str \| list[float]` | `auto` | white \| hanning \| hanning-decimated \| [rho_1, rho_2, ...]; a '+white' suffix (e.g. hanning+white) adds a white component whose fraction is estimated from the data; auto: hanning+white if the response contains hanning |
| `noise.signal_free` | `list[tuple[float, float]] \| None` | `null` | Velocity ranges used to estimate sigma |
| `rfi.prefit` | `bool` | `True` |  |
| `rfi.background` | `Literal['positive', 'symmetric']` | `positive` |  |
| `rfi.threshold` | `float` | `0.5` |  |
| `rfi.action` | `Literal['mask', 'inflate']` | `mask` |  |
| `rfi.n_iter` | `int` | `8` |  |
| `rfi.flag_fraction` | `float` | `0.05` |  |
| `rfi.min_sigma_channels` | `float` | `2.0` | Smallest Delta v_rand (in channels) of the line model in the pre-stage, so that a spike of one or two channels cannot be absorbed as a line |
| `rfi.close_holes` | `bool` | `True` | Also mask a single channel between two masked ones |
| `rfi.max_width_channels` | `int \| None` | `3` | Longest run of outlier channels (gaps of up to two channels included) treated as RFI; a longer run is signal the one-component pre-stage cannot explain, typically a second line, and stays in the fit; null: no limit |
| `sampler.mode` | `Literal['quick', 'full', 'check']` | `full` |  |
| `sampler.backend` | `Literal['nautilus', 'dynesty', 'ultranest', 'smc', 'quick']` | `smc` | smc: fastest (vectorised tempered SMC); nautilus: most precise ln Z but ~10x slower |
| `sampler.check_backend` | `Literal['nautilus', 'dynesty', 'ultranest', 'smc']` | `nautilus` |  |
| `sampler.n_live` | `int` | `400` |  |
| `sampler.gate_margin` | `float \| None` | `5.0` | Skip nested sampling for N when its quick evidence is this many nats below the best model with fewer components (null: always sample) |
| `sampler.n_seeds` | `int` | `3` | Warm starts of the quick stage for N >= 2: the N-1 solution plus a new component on each of the strongest residual features (random starts rarely hit a second line) |
| `sampler.init_fraction` | `float` | `0.25` | Fraction of the SMC particles drawn from the quick-stage posterior instead of the prior (defensive mixture with importance weights, the evidence stays exact); 0 disables |
| `sampler.options` | `dict[str, Any]` | `{}` |  |
| `sampler.likelihood_backend` | `Literal['numpy', 'jax']` | `numpy` |  |
| `diagnostics.ppp_min` | `float` | `0.01` |  |
| `diagnostics.wiggle_max` | `float` | `0.1` |  |
| `diagnostics.acf_sigma` | `float` | `4.0` |  |
| `diagnostics.edge_mass` | `float` | `0.2` |  |
| `diagnostics.min_ess` | `float` | `500.0` |  |
| `diagnostics.logz_sigma` | `float` | `3.0` |  |
| `diagnostics.min_w50_channels` | `float` | `2.0` | Flag `unresolved` when a component's W50 is narrower than this many channels |
| `diagnostics.min_constraint` | `float` | `0.3` | Flag `prior_dominated` when the posterior 16-84 % width of delta_v is wider than (1 - min_constraint) times the prior width, i.e. the line width is not measured |
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
| `priors` | `Priors` | `{'v_ctr': {'kind': 'uniform'}, 'v_margin': 0.05, 'delta_v': {'kind': 'lognormal', 'low': 0.0, 'high': 800.0, 'mean': 2.2, 'std': 0.4}, 'delta_v_rand': {'kind': 'lognormal', 'low': 4.0, 'high': 60.0, 'mean': 1.1, 'std': 0.2}, 'alpha': {'kind': 'normal', 'low': -1.0, 'high': 1.0, 'mean': 0.0, 'std': 0.25}, 'f': {'kind': 'beta', 'low': 0.0, 'high': 1.0, 'a': 1.0, 'b': 3.0}, 'amplitude_scale': 'auto', 'baseline_scale': 'auto', 'rfi_sigma_channels': (0.3, 1.0)}` |  |
| `flux` | `Literal['prior', 'snr']` | `snr` |  |
| `snr_range` | `tuple[float, float]` | `(3.0, 100.0)` |  |
| `noise_sigma` | `float` | `0.01` |  |
| `noise_correlation` | `Literal['white', 'hanning']` | `white` |  |
| `baseline` | `Literal['none', 'chebyshev', 'sine']` | `none` |  |
| `baseline_order` | `int` | `1` |  |
| `baseline_order_min` | `int \| None` | `null` |  |
| `baseline_scale` | `float` | `0.01` |  |
| `sine_amplitude` | `float` | `0.5` |  |
| `sine_period` | `tuple[float, float]` | `(100.0, 600.0)` |  |
| `rfi` | `Literal['none', 'spikes', 'broad']` | `none` |  |
| `rfi_rate` | `float` | `0.01` |  |
| `rfi_amplitude` | `tuple[float, float]` | `(5.0, 50.0)` |  |
| `write_sigma` | `bool` | `True` |  |
