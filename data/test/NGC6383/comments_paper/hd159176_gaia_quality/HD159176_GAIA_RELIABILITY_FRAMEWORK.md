# HD 159176 Gaia DR3 reliability framework

This note addresses the referee-level question:

> If HD 159176 is a bright O+O binary and its Gaia DR3 astrometry may be
> biased, how can we decide whether the Gaia parallax and proper motions are
> reliable enough to infer membership?

## Short answer

For HD 159176, the Gaia DR3 values should be treated as informative but not as a
clean, decisive single-star astrometric solution.

The source agrees reasonably with the NGC 6383 reference distribution in
parallax and `pmra`, but not in `pmdec`. That one-component disagreement is not
automatically invalid. Gaia solves the five astrometric parameters using
along-scan observations with a non-uniform scanning law, and unmodelled source
structure, bright-star calibration effects, or binary photocentre motion can
project anisotropically into one astrometric component.

However, the present evidence is not strong enough to claim that Gaia DR3 alone
proves HD 159176 is physically unrelated to NGC 6383. The defensible statement is
that HD 159176 is excluded from the secure Gaia-selected reference sample.

## Direct checks already performed

Gaia DR3 source used:

```text
source_id = 4054618559611164288
G = 5.710343
parallax = 1.166596 +/- 0.070692 mas
pmra = 2.621238 +/- 0.083297 mas/yr
pmdec = -0.797612 +/- 0.058472 mas/yr
```

Quality diagnostics:

```text
RUWE = 0.936933
astrometric_excess_noise = 0.404612 mas
astrometric_excess_noise_sig = 135.619766
visibility_periods_used = 13
duplicated_source = False
astrometric_params_solved = 31
astrometric_n_good_obs_al = 159
astrometric_n_bad_obs_al = 1
astrometric_matched_transits = 18
astrometric_gof_al = -1.080805
astrometric_sigma5d_max = 0.115833 mas
ipd_gof_harmonic_amplitude = 0.085097
ipd_frac_multi_peak = 39
ipd_frac_odd_win = 0
```

Relevant covariance terms:

```text
parallax_pmdec_corr = 0.326
pmra_pmdec_corr = 0.040
parallax_pmra_corr = 0.059
dec_pmdec_corr = -0.380
```

This means the `pmdec` offset is not explained by a strong `pmra-pmdec`
correlation. The moderate `parallax-pmdec` and `dec-pmdec` correlations should
be mentioned if a full covariance Mahalanobis comparison is later computed.

Gaia DR3 non-single-star table checks for this source:

```text
gaiadr3.nss_two_body_orbit: 0 rows
gaiadr3.nss_acceleration_astro: 0 rows
gaiadr3.nss_non_linear_spectro: 0 rows
gaiadr3.nss_vim_fl: 0 rows
gaiadr3.vari_eclipsing_binary: 0 rows
```

The absence of a Gaia DR3 NSS solution does not prove the source is astrometri-
cally clean. Gaia DR3 NSS is incomplete by construction and binaries can remain
undetected depending on orbit, brightness ratio, scanning law, and mission time
baseline.

## How to decide what is trustworthy

Use a tiered reliability statement, not a binary "trust/don't trust" flag.

### Tier A: safe to use as ordinary Gaia member

Use this only when all of the following are true:

- the star has a normal quality profile compared with a magnitude-matched
  control sample;
- `RUWE`, `astrometric_excess_noise_sig`, `ipd_frac_multi_peak`,
  `ipd_frac_odd_win`, bad-observation fraction, and photometric-excess fields
  are not outliers;
- the source has enough visibility periods, preferably well above the `10`
  caution region discussed in the Gaia documentation;
- the source is not a known close binary, or a Gaia NSS/orbital solution shows
  that the barycentric motion is consistent with the cluster.

HD 159176 does not meet this tier because its excess-noise significance and IPD
multi-peak fraction are extreme.

### Tier B: useful astrometric evidence with caveats

This is the correct tier for HD 159176.

Use the Gaia parallax and proper motions as evidence, but do not treat the
formal DR3 uncertainties as the whole error budget. In this tier:

- parallax and `pmra` agreement with the cluster should be reported as agreement;
- the `pmdec` discrepancy should be reported as the main Gaia-space tension;
- the conclusion should be operational: not retained in the secure Gaia-selected
  member sample;
- the physical association should remain open unless external evidence closes
  the case.

### Tier C: unreliable for membership inference

Use this if additional checks show severe source-level problems, for example:

- very low visibility-period count;
- duplicated-source flag;
- strong odd-window or contamination indicators;
- nearby Gaia sources in the same window that plausibly explain the IPD
  multi-peak value;
- a Gaia NSS or external orbit showing that the five-parameter proper motion is
  dominated by photocentre/orbital motion;
- disagreement with independent long-baseline astrometry at the same level as
  the cluster/non-cluster decision.

HD 159176 is not currently in this tier because RUWE is normal, it has 13
visibility periods, no duplicated-source flag, and only 1 bad AL observation.
The issue is not "Gaia is unusable"; the issue is "Gaia DR3 is not clean enough
for an absolute non-membership claim."

## What would make the claim stronger

The strongest next checks are:

1. Magnitude-matched control sample: compare HD 159176 with Gaia DR3 sources at
   similar `G`, colour, sky region, and crowding, not only with faint PMS
   members.
2. Full covariance membership test: compute the 3D astrometric distance
   `(parallax, pmra, pmdec)` using the Gaia covariance matrix and the empirical
   cluster covariance, then repeat with inflated uncertainties for bright-source
   systematics.
3. Bright-source correction: apply or at least bound the Cantat-Gaudin & Brandt
   magnitude-dependent proper-motion correction. If the correction is much
   smaller than the observed `pmdec` offset, it cannot by itself explain the
   discrepancy.
4. Neighbour/window audit: query nearby Gaia sources and image-level crowding
   indicators to test whether `ipd_frac_multi_peak=39` is caused by a nearby
   source, source splitting, or bright-source processing.
5. External astrometry: compare Hipparcos/Tycho/HGCA-style long-baseline proper
   motions where available. A proper-motion anomaly between long-baseline and
   Gaia DR3 would support orbital/photocentre contamination.
6. Spectroscopic/orbital context: because HD 159176 is a short-period SB2 O+O
   binary, test whether the known orbit could produce a photocentre displacement
   large enough to affect Gaia DR3 at the required level. For a nearly equal
   O+O pair, the expected photocentre wobble may be small even though the system
   is physically binary; this must be estimated rather than assumed.
7. Independent membership evidence: radial velocity/systemic velocity, age,
   extinction, position at the cluster centre, and historical PMS/HD age
   comparisons must be discussed separately from Gaia-only astrometry.

## Referee-facing wording

Recommended:

```text
HD 159176 agrees with the Gaia-selected cluster distribution in parallax and
proper motion in right ascension within the limitations of the comparison, but
its proper motion in declination remains discrepant. Because HD 159176 is a very
bright O+O binary and its Gaia DR3 solution has significant excess-noise and IPD
multi-peak diagnostics, we do not interpret the five-parameter Gaia solution with
the same weight as for the fainter PMS population. We therefore exclude
HD 159176 from the secure Gaia-selected reference sample, while avoiding the
stronger claim that Gaia DR3 alone definitively rules out a physical association
with NGC 6383.
```

Avoid:

```text
HD 159176 is not a member of NGC 6383.
```

Avoid:

```text
RUWE is below 1.4, therefore the Gaia astrometry is reliable.
```

## Literature basis

- Gaia DR3 `gaia_source` documentation defines excess noise as the disagreement
  between observations and the best-fitting five-parameter model, and explicitly
  notes that unresolved binaries can contribute to positive excess noise.
- The same Gaia documentation states that `astrometric_excess_noise_sig > 2`
  is significant for well-behaved sources; HD 159176 has 135.6.
- The Gaia documentation also defines `ipd_frac_multi_peak` as the fraction of
  successful IPD windows with more than one peak, linked to visual doubles or
  real binaries, while warning that bright-binary interpretation is complicated.
- Cantat-Gaudin & Brandt (2021) show that bright Gaia EDR3 proper motions have
  magnitude-dependent systematics, supporting an explicit bright-source caveat.
- Gaia DR3 content/validation papers note that DR3 includes non-single-star
  products, but the absence of an NSS row is not a general proof of singleness
  or clean five-parameter astrometry.
- Castro-Ginard et al. (2024) show that RUWE detectability for unresolved
  binaries depends on binary properties and Gaia's time baseline; many binaries
  can remain below a simple RUWE threshold.
- Rybizki et al. (2022) supports multi-indicator Gaia-quality reasoning rather
  than one-flag cuts.

