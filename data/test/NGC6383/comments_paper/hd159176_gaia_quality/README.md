# HD 159176 Gaia DR3 astrometric-quality audit

This directory contains the reproducible Gaia DR3 astrometric-quality comparison
for HD 159176 against the NGC 6383 likely-candidate catalogue used in the A&A
revision.

## Main products

- `HD159176_GAIA_QUALITY_COMPARISON.md`: human-readable comparison report.
- `HD159176_GAIA_RELIABILITY_FRAMEWORK.md`: referee-facing decision framework
  for how much weight to give the Gaia DR3 astrometry of HD 159176.
- `LITERATURE_NOTES.md`: source-backed interpretation and paper/thesis wording
  guidance.
- `hd159176_gaia_quality_summary.json`: machine-readable metric summary.
- `ngc6383_members_gaia_quality.csv` / `.ecsv`: Gaia-quality fields for the 321
  likely candidates.
- `hd159176_gaiadr3_quality.csv` / `.ecsv`: Gaia-quality row for HD 159176.
- `hd159176_gaia_quality_comparison.pdf`: diagnostic histograms.
- `hd159176_ipd_vs_gmag.pdf`: magnitude-aware IPD diagnostic plot.

## Source queried

The Gaia DR3 source matching HD 159176 in the 0.01 deg cone around the cluster
centre is:

```text
source_id = 4054618559611164288
G = 5.710343 mag
parallax = 1.166596 +/- 0.070692 mas
pmra = 2.621238 +/- 0.083297 mas/yr
pmdec = -0.797612 +/- 0.058472 mas/yr
```

Query output:

```text
comments_paper/hd159176_gaia_quality/hd159176_gaiadr3_quality.csv
```

## Current diagnostic values

| quantity | HD 159176 value | first interpretation |
| --- | ---: | --- |
| `ruwe` | 0.936933 | Below the common `RUWE < 1.4` quality cut, so the reduced single-source fit statistic alone is not alarming. |
| `astrometric_excess_noise` | 0.404612 mas | Positive excess noise; Gaia defines zero as statistically well behaved and positive values as residuals larger than expected. |
| `astrometric_excess_noise_sig` | 135.61977 | Very significant excess noise; Gaia documentation treats values above 2 as significant. |
| `visibility_periods_used` | 13 | Above the common caution boundary of 10; Gaia notes that small values below about 10 make parallax more vulnerable to unmodelled errors. |
| `duplicated_source` | false | No Gaia duplicated-source flag. This does not rule out binary/bright-source complications. |
| `astrometric_params_solved` | 31 | Five-parameter solution. |
| `astrometric_gof_al` | -1.080805 | Along-scan goodness of fit is not high by itself. |
| `ipd_frac_multi_peak` | 39 | Large fraction of windows with more than one IPD peak; Gaia documentation links this to visually resolved doubles or real binaries, with bright-binary interpretation explicitly complicated. |
| `ipd_frac_odd_win` | 0 | No odd-window fraction warning in this field. |

## Practical rule for manuscript wording

Do not reduce the HD 159176 astrometric-quality argument to RUWE alone. For this
source, RUWE looks good, but `astrometric_excess_noise_sig` and
`ipd_frac_multi_peak` are strongly suspicious and are physically plausible for an
O+O binary outside the magnitude range of the Gaia-selected reference sample. The robust comparison now:

1. queries the same Gaia DR3 quality columns for the 254 reference members and the
   321 likely candidates.
2. compares HD 159176 against the member distributions in `G`, `ruwe`,
   `astrometric_excess_noise`, `astrometric_excess_noise_sig`,
   `visibility_periods_used`, `duplicated_source`, `astrometric_gof_al`,
   `ipd_frac_multi_peak`, and `ipd_frac_odd_win`.
3. includes a magnitude-aware comparison, because HD 159176 is far brighter
   (`G ~= 5.7`) than the low-mass/PMS candidate population.
4. keeps the manuscript conclusion conservative: the declination proper-motion
   offset is the main reason to exclude HD 159176 from the secure member sample,
   while the quality diagnostics prevent overclaiming from Gaia alone.

## Sources checked

- Gaia DR3 data model, `gaia_source` column definitions:
  `https://gea.esac.esa.int/archive/documentation/GDR3/Gaia_archive/chap_datamodel/sec_dm_main_source_catalogue/ssec_dm_gaia_source.html`
- Gaia EDR3 astrometric quality documentation:
  `https://gea.esac.esa.int/archive/documentation/GEDR3/Data_processing/chap_cu3ast/sec_cu3ast_quality/ssec_cu3ast_quality_properties.html`
- Everall & Boubert (2022), selection functions in Gaia EDR3, for the commonly
  used `RUWE < 1.4` cut:
  `https://academic.oup.com/mnras/article/509/4/6205/6426194`
- Rybizki et al. (2022), spurious astrometric-solution classifier and fidelity:
  `https://academic.oup.com/mnras/article/510/2/2597/6460502`
- Cantat-Gaudin & Brandt (2021), bright-source proper-motion bias:
  `https://www.aanda.org/articles/aa/full_html/2021/05/aa40807-21/aa40807-21.html`
- Maiz Apellaniz (2022), bright-source parallax-bias context:
  `https://www.aanda.org/articles/aa/full_html/2022/01/aa42365-21/aa42365-21.html`
- Belokurov et al. (2020), unresolved companions and Gaia astrometric residuals:
  `https://academic.oup.com/mnras/article/496/2/1922/5849452`
- Castro-Ginard et al. (2024), Gaia DR3 detectability of unresolved binaries:
  `https://arxiv.org/abs/2404.14127`
