# Literature verification of Pierluigi's substantive comments

Checked via ADS/SciX + arXiv, 2026-06-02.

## 1. Cameron (2011) — #51 ✅ TRUE, accept rewrite
- Cameron, E. 2011, PASA 28, 128, "On the Estimation of Confidence Intervals for Binomial Population Proportions in Astronomy" (`2011PASA...28..128C`, arXiv:1012.0566, 425 cites).
- Method = Bayesian quantiles of the Beta posterior for binomial proportions; argues against Wald (normal approx) and Clopper-Pearson. = `astropy.stats.binom_conf_interval` (jeffreys/wilson).
- Paper uses B(1,1) prior + Beta posterior 95% CI → exactly Cameron's method. Pierluigi's rewrite ("We followed the method of Cameron 2011") is the correct attribution.
- ACTION: accept rewrite, only if code truly used beta-quantile (it states B(1,1) prior, so yes).

## 2. Buckner & Froebrich (2013) — #53 ⚠️ COMPARISON IS VALID (corrects earlier wrong claim)
- Buckner & Froebrich 2013, MNRAS 436, 1465, "Properties of star clusters - I." (`2013MNRAS.436.1465B`, arXiv:1309.0708).
- B&F Y_frac = **YSO fraction** (Q-parameter / NIR-excess), age proxy. Sample = 378 known OCs + 397 FSR candidates. Values mostly ~0, tail rarely >0.15.
- **KEY**: this paper's Y_frac is ALSO a YSO fraction, computed with the SAME B&F Q<-0.05 method (Sect "Identification of YSOs", lines 381-395). So the comparison is apples-to-apples — NOT a category error.
  - (My earlier session claim of "binary vs YSO category error" was WRONG — caused by a wrong premise in my research prompt. The paper's Y_frac is unambiguously YSO fraction.)
- NGC6383 not being in B&F sample is fine: text says "higher compared to the clusters analyzed in B&F", a comparison TO their sample, not a claim of membership in it.
- REMAINING: the specific "18 of 397 had Y_frac>0.1, none >0.2" count is plausible (right-skewed, mostly 0) but was NOT independently confirmed line-by-line in B&F — author should double-check the exact numbers against B&F Table.
- ACTION: keep comparison; reword per Pierluigi; verify the 18/397 figures.

## 3. Kalari (2019) — #70 🟡 his rewrite INVERTS the method, do not paste verbatim
- Kalari, V.M. 2019, MNRAS 484, 5102 (`2019MNRAS.484.5102K`, arXiv:1901.07511).
- Real workflow: PRIMARY selection = photometric Halpha equivalent width (r-Halpha vs r-i, Barrado y Navascues criteria) -> 156 CTTS. SECONDARY = double-peaked Gaussian fit to Gaia proper-motion distribution, keep IQR -> 55 kinematic members.
- Pierluigi's rewrite "selected cluster members by fitting a double-peaked Gaussian to proper motions" = the PM step was secondary outlier-rejection, not the selection method. Inverts the order.
- Note: the paper's CURRENT text (line 448) ALSO says "Kalari determined membership by modeling the distribution of proper motions with a double-peaked Gaussian" — same imprecision already present.
- ACTION: rewrite as "Kalari (2019) selected CTTS primarily from photometric Halpha equivalent widths, then refined membership with a double-peaked Gaussian fit to the Gaia proper-motion distribution." Fix BOTH the existing text and any new sentence.

## 4. Rauw et al. (2010) offset — #69 🟢 valid question, minor fix
- Rauw, Manfroid & De Becker 2010, A&A 511, A25 (`2010A&A...511A..25R`).
- 0.12-0.24 is the (Rc-Halpha) COLOUR INDEX above the MS Rc-Halpha relation (EW_Halpha ~6-10 A), candidate criterion; >0.24 = emitter. NOT a vertical CMD offset; band = custom Halpha vs Rc, not V.
- Paper text (line 446) already establishes "Rc-Halpha index" in the same paragraph, so it's mostly OK — could tighten "0.12-0.24 mag above the main-sequence (Rc-Halpha) relation".
- ACTION: add "(Rc-Halpha)" qualifier; answer Pierluigi: not the CMD band, it's the Halpha-excess colour.

## 5. Halpha absence non-decisive — #75 (strikeout) 🟡 content is CORRECT, don't lose it
- Astrophysically sound: WTTS are bona-fide PMS with weak/no Halpha; Halpha variable; Rauw's own X-ray PMS show weak Halpha. Absence != not-PMS.
- Pierluigi struck it (likely as verbose). Keep the idea, compress wording.
- ACTION: keep a one-line version; do not delete the scientific point.

## 6. PMS CMD -> ASteCA binary/mass unreliable — #62/#64 🟢 correct, add citation
- Da Rio, Gouliermis & Gennaro 2010, ApJ 723, 166 (`2010ApJ...723..166D`): PMS CMD broadening from binarity, differential extinction, variability, accretion, age spread; isochrone-based mass/binarity unreliable; results sensitive to assumed binary fraction & model grid.
- ASteCA = Perren, Vazquez & Piatti 2015, A&A 576, A6 (`2015A&A...576A...6P`) — yes derives binarity from CMD position vs isochrone.
- ACTION: cite Da Rio et al. (2010) to back the caution.

## 7. KS test report D + p — #66 🟢 standard
- Best practice: report KS statistic D + p-value + N1,N2. `scipy.stats.ks_2samp` returns both.
- ACTION: ensure every KS result reports D, p, and N.

## 8. Bin-dependent p-values = red flag — #67 🔴 most important
- Slicing many mass ranges/quartiles and reporting the significant one = multiple-comparisons / forking-paths; nominal p invalid.
- Standard mass-segregation alternative: Allison et al. 2009, MNRAS 395, 1449 (`2009MNRAS.395.1449A`), MST Lambda_MSR — bin-agnostic, continuous segregation profile with significance.
- ACTION: use a physically-motivated mass cut, report ALL bins tried, and/or add Allison et al. Lambda_MSR as cross-check.

## 9. Sagitta per-star histograms — #47 🟡 legitimate
- Sagitta = McBride et al. 2021, AJ 162, 282 (`2021AJ....162..282M`); per-star point estimates from NN, not a calibrated posterior; histogram reflects NN dispersion, not cluster-age posterior.
- ACTION: keep only if explicitly framed as distribution of individual point estimates (paper already says this, line 370); consider appendix.
