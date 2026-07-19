# HD 159176 Gaia DR3 quality comparison

This report compares the Gaia DR3 astrometric-quality indicators of HD 159176
against the NGC 6383 likely-candidate catalogue used in the A&A revision.

## Main result

HD 159176 is much brighter than the member sample (`G=5.710` mag),
so the comparison must be interpreted as a quality-context check rather than
a direct like-for-like member test. Its `ruwe` is not alarming, but its
`astrometric_excess_noise_sig` and `ipd_frac_multi_peak` are extreme relative
to the NGC 6383 reference members and are physically plausible warning signs
for an O+O binary outside the magnitude range of the Gaia-selected reference
sample.

The percentile column below uses a midrank convention for tied values. The
last column reports the smaller of the lower-tail and upper-tail fractions,
which avoids over-interpreting metrics where most reference stars have the
same value, such as `ipd_frac_odd_win=0`.

## HD 159176 values

| quantity | value |
| --- | ---: |
| `source_id` | 4054618559611164288 |
| `G` | 5.710343 |
| `parallax` | 1.166596 +/- 0.070692 mas |
| `pmra` | 2.621238 +/- 0.083297 mas/yr |
| `pmdec` | -0.797612 +/- 0.058472 mas/yr |
| `ruwe` | 0.936933 |
| `astrometric_excess_noise` | 0.404612 mas |
| `astrometric_excess_noise_sig` | 135.619766 |
| `visibility_periods_used` | 13 |
| `duplicated_source` | False |
| `astrometric_params_solved` | 31 |
| `ipd_frac_multi_peak` | 39.000000 |
| `ipd_frac_odd_win` | 0.000000 |

## Against the 254-source reference sample

| metric | HD 159176 | ref median | ref p16-p84 | HD midrank percentile | min tail [%] |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ruwe` | 0.936933 | 0.999983 | 0.939911-1.05606 | 14.2 | 14.2 |
| `astrometric_excess_noise` | 0.404612 | 0 | 0-0.258595 | 90.6 | 9.4 |
| `astrometric_excess_noise_sig` | 135.62 | 0 | 0-1.53052 | 99.6 | 0.4 |
| `visibility_periods_used` | 13 | 16 | 15-16 | 4.1 | 6.7 |
| `astrometric_gof_al` | -1.08081 | 0.0308983 | -1.24462-1.17469 | 19.7 | 19.7 |
| `ipd_gof_harmonic_amplitude` | 0.0850965 | 0.0498585 | 0.0227791-0.0844632 | 84.6 | 15.4 |
| `ipd_frac_multi_peak` | 39 | 0 | 0-0 | 100.0 | 0.0 |
| `ipd_frac_odd_win` | 0 | 0 | 0-0 | 49.0 | 98.0 |

## Magnitude-aware caveat

The candidate catalogue contains 31 likely candidates with `G<13`.
That subset is still much fainter than HD 159176 in most cases, so it does
not remove the bright-source caveat. It only shows that the paper member
sample does not provide a well-populated control set at `G~5.7`.

## Recommended manuscript interpretation

Keep the current HD 159176 membership statement conservative. The Gaia DR3
declination proper-motion offset remains the main reason not to use HD 159176
as a secure member, but the quality diagnostics show that the astrometric
solution is not clean enough to make an overconfident claim from Gaia alone.
Do not argue from RUWE alone.

## Generated files

- `ngc6383_members_gaia_quality.csv`
- `ngc6383_members_gaia_quality.ecsv`
- `hd159176_gaia_quality_summary.json`
- `hd159176_gaia_quality_comparison.pdf`
- `hd159176_ipd_vs_gmag.pdf`
