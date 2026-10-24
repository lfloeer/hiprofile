# Priors

The default priors are calibrated on the fits of the 34 THINGS galaxies in
Table 1 of Stewart, Blyth & de Blok (2014) and on the priors the authors used
for their low-S/N fits (their Appendix B.2). They are slightly wider than
THINGS because a survey contains more dwarfs, more disturbed systems and
lower S/N.

![Default priors against the THINGS fits](img/priors_things.png)

| parameter | default | THINGS fits | other references |
|---|---|---|---|
| Δv | log-normal, log₁₀ = 2.2 ± 0.4 (≈ 160 km/s), ≤ 800 km/s | 0.1–566 km/s, median 153; two dwarfs with Δv ≈ 0 | Stewart B.2: log-normal 2.2 ± 0.3; ALFALFA width function: modified Schechter with w* ≈ 300 km/s, α ≈ −0.56 (Oman 2022) |
| Δv_rand | log-normal, log₁₀ = 1.1 ± 0.2 (≈ 12.6 km/s), 4–60 km/s | 8–32 km/s, median 12.4, log₁₀ 1.13 ± 0.16 | Stewart B.2: 1.17 ± 0.14; resolved HI dispersions ≈ 10 ± 2 km/s (Tamburro et al. 2009); narrow / broad super-profile components 6.5 ± 1.5 / 16.8 ± 4.3 km/s (Ianjamasimanana et al. 2012) |
| α | normal(0, 0.25) truncated to [−1, 1] | median 0.00, robust σ 0.14; 3 of 34 with \|α\| > 0.6 (NGC 3077, NGC 1569, Ho I) | Stewart B.2: normal(0, 0.2) |
| f | half-normal amplitudes with σ(a_disk) = σ(a_ring) / 3, prior median f ≈ 0.25 | median 0.15, mean 0.23 | Stewart B.2: exponential, scale ~0.3 |
| v_ctr | uniform over the band minus 5 % margins; normal(v_cat, 20–50 km/s) for targeted spectra | – | Stewart B.2: normal, σ = 20 km/s |
| S | half-normal on a_ring (`amplitude_scale`, fix per survey) | – | Stewart: flat (improper, unusable for evidences) |

All priors are simple one-parameter-family shapes (log-normal, truncated
normal, half-normal); Δv uses 0.4 dex instead of the 0.3 dex of Stewart et
al. so that narrow dwarf profiles keep prior mass. The amplitudes keep Gaussian (half-normal) priors so that they can be
marginalised analytically; the prior on f follows from the ratio of the two
scales. Positivity enters the sampling target exactly through the orthant
probability of each component (see [Inference](inference.md)).

The simulator draws from the same priors (`hiprofile simulate`, all
validation stages), so simulated galaxies look like the ones the fit expects.

## Survey-specific settings

* **Single-dish surveys** (HIPASS, ALFALFA, EBHIS, FAST): standing waves need
  higher baseline orders; Stewart et al. needed K = 6 for EBHIS. Use
  `model.k_max: 6` and a baseline prior from signal-free neighbour spectra
  (`hiprofile.likelihood.baseline.baseline_prior_from_spectra`).
* **Interferometric surveys** (WALLABY, THINGS-like cubes): K ≤ 1 is usually
  enough.
* **Targeted spectra** with an optical redshift: `priors.v_ctr: {kind: normal,
  mean: <v_cat>, std: 30}`.
* **Evidence comparisons across a survey**: set `priors.amplitude_scale` to a
  fixed value (e.g. the 95th percentile of the expected integrated fluxes).

At high S/N the data dominate and the priors hardly matter; at low S/N they
regularise the degeneracies between Δv and Δv_rand and between ring and
disk. Objects outside the THINGS range (very disturbed or very narrow
profiles) are pulled towards typical values; the `prior_edge` flag marks
posteriors that pile up at a prior boundary.

References: Stewart, Blyth & de Blok (2014), A&A 567, A61 (arXiv:1405.1838);
Ianjamasimanana et al. (2012), arXiv:1207.5041; Oman (2022), arXiv:2108.08856;
Tamburro et al. (2009), AJ.
