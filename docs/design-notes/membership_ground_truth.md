# Ground truth for membership calibration — what exists, and what it costs

_2026-07-26. Independent (non-astrometric) member catalogs usable to calibrate a membership
classifier, and the limits of each. From a sourced multi-agent review; every bibcode resolved via
ADS with its title echoed. `[S]`=verified, `[I]`=inferred/own arithmetic._

## 0. The organising rule: **publication date ≈ independence**

Calibrating an astrometric classifier against labels that were themselves astrometrically filtered
is **circular**. Pre-Gaia-DR2 catalogs *could not* have used Gaia astrometry, which makes the date a
better first filter than the indicator type.

- **Independent:** COUP (2005), MYStIX (2013), SFiNCs (2017), Kashyap Cyg OB2 (2023 — verified
  Gaia-free feature list), IPHAS/VPHAS+/IGAPS Hα (2008–2020), SPICY (2021, IR-only classifier).
- **Circular — do not use as truth:** Kalari 2019 NGC 6383 (§4), Damiani 2019 NGC 6530, Fratta 2021,
  Salvato 2025 eRASS1 counterparts, Fritzewski 2025 NGC 2516.

## 1. The crux: does anything independent reach **G > 18** *and* **M < 0.3 M☉**?

**G > 18 — yes, ~30 clusters.** Kuhn+2019 (`2019ApJ...870...32K`) matched MYStIX+SFiNCs to Gaia DR2
and reports a median matched magnitude of G≈18.1 (IQR 16.6–19.1), ~17,509 sources with 5-parameter
solutions, and a ~13% contamination rate.

```{attention}
**These three Kuhn+2019 figures are `[UNVERIFIED]`** — median G=18.1, 17,509 usable labels, and the
~13% label-noise rate. The bibcode is right and the numbers came through an automated full-text
extractor, but a second agent could not confirm them (they are not in the abstract). **Open the PDF
before any of them goes in a manuscript.** They are quoted here as a lead, not as evidence.
```

**M < 0.3 M☉ *with* a Gaia counterpart — only ≲500 pc, plus two Hα fields at ~1 kpc:**

| Region | Catalog | Bibcode | Reach |
|---|---|---|---|
| ONC / OMC-1 (414 pc) | **COUP** | `2005ApJS..160..319G` | **0.1–2 M☉**, G≈20 |
| IC 348, NGC 1333, Serpens… (235–500 pc) | **SFiNCs** | `2017ApJS..229...28G` | ~0.3 M☉ |
| IC 1396 / Tr 37 (870 pc) | IPHAS Hα | `2011MNRAS.415..103B` | **0.2 M☉**, 13<r′<20 |
| M8 / NGC 6530 (1250 pc) | VPHAS+ Hα | `2015MNRAS.453.1026K` | 0.2–0.4 M☉ @ 60% |

```{warning}
**Beyond ~1 kpc, G>18 labels are faint SOLAR-mass stars** — faint from distance and extinction, not
low-mass. They are still useful for testing the faint-parallax-error regime, but **must not be sold
as an M-dwarf test.** The two conditions decouple with distance, and the deeper an indicator reaches
in mass the less likely the star is in Gaia at all (JWST reaches substellar YSOs at 2.35 kpc with no
Gaia counterpart — nothing to calibrate). The usable window is narrow by construction.
```

## 1b. The structural insight: independence is a property of the *product*, not the survey

Every published OC member list is purity-only, because all of them are Cantat-Gaudin / H&R
crossmatches. But the **parent archives are astrometrically unselected**. Query the archive and
**ignore the member flag**, and false negatives become measurable. This reframes the whole search.

## 2a. The best set found: **OCCAM DR17** `2022AJ....164...85M`

Ranks above the X-ray sets for *bright-end purity* because of how it was selected `[S]`:

- **Target selection is a spatial cone** (2×R_CG) — **not a member list**. That is what makes false
  negatives measurable.
- **Parallax is never used**; RV and [Fe/H] are independent discriminants (PM is Gaia's).
- **26,699 rows, of which ~24,700 are non-members** — real negatives at scale.
- Purity result, directly usable: of Gaia astrometric members at `CG_PROB>0.7`, **9.3% are rejected
  by independent RV, 19.8% by independent [Fe/H], 25.1% by either.** DR19 gives 9.0% on RV+[Fe/H].
- Limit: giant-tip, H ≲ 12.2–13.8. Join on `APOGEE_ID`. Pull `occam_member-DR17.fits` (4 MB) from
  the SDSS SAS — **it is not on VizieR**.

```{danger}
**Newest is worst here — use DR17, not DR19.** OCCAM **DR19 hard-gates on Cantat-Gaudin P > 0.70**
(verified empirically: no values below 0.70 exist in the file). So DR19 contains **no negatives at
all** and cannot support a reliability diagram — the reflex of grabbing the latest release destroys
the one property that makes this set valuable. DR19 contributes exactly one usable number (9.0% of
CG2020 P>0.7 members fail an independent RV+[Fe/H] test); everything else must come from **DR17**.
```
- ⚠ **Split on `APOGEE2_TARGET1` bit 9 first** — some late APOGEE-2 targets *were* Gaia-PM
  preselected, so the union is optimistically biased.

**And the single most on-point measurement anywhere — Poovelil+2020** (`2020ApJ...903...55P`) `[S]`:
~5% false positives and ~4% false negatives against Cantat-Gaudin 2018, where the false negatives
arise because *"These stars do not have measured Gaia-DR2 parallaxes."* **That is exactly our failure
mode, measured independently, by someone else.** Cite it.

## 2. X-ray

- **Kashyap Cyg OB2 2023** (`2023ApJS..269...10K`) — **start here for calibration.** All **7,924**
  X-ray sources carry `CLASSIFICATION` = member / background / foreground with per-class accuracy
  **96 / 93 / 80%** (~95% overall), from a verified **Gaia-free** feature list (SDSS *riz*, INT Hα,
  UKIDSS/2MASS *JHK*, X-ray quantiles, A_V). The only product here shipping **ready-made negatives**
  with a Gaia-free discriminant. VizieR `J/ApJS/269/10`, join on `CXO_ID`. Caveat: only 6 citations —
  too new to have been audited; a known unknown, not a clean bill. `[S]`
- **MYStIX** (`2013ApJS..209...32B`, ~31,784 members, 20 regions 0.41–3.6 kpc) — classifier inputs
  are median energy, *J*, X-ray variability, spectral type, 4.5 µm, IR SED: **no PM, no parallax**.
  Ships four-class posteriors (H1 foreground / H2 young / H3 background / H4 extragalactic), though
  VizieR `J/ApJS/209/32` exposes only the discrete class — **continuous posteriors are in the ApJS
  electronic edition only**. `[S]`
- **SFiNCs** (`2017ApJS..229...28G`, 8,492 members, 22 SFRs 235–1400 pc) — *"roughly complete down to
  ∼0.3 M⊙ and ∼2 M⊙ for the nearest and most distant"*. VizieR ships **both** `xsources` (all 15,364
  X-ray sources) and `spcms` (members) → negatives available. `[S]`
- **COUP** (`2005ApJS..160..319G`) — 838 ks, the deepest look at any young cluster; 1,616 sources →
  1,315 members, with foreground/AGN/spurious all labelled. Detection limit
  *"L_X,min~10^27.3 erg s⁻¹"*, detecting *">97% of the optically visible late-type cluster stars"*,
  spanning **0.1–2 M☉**. **No paper has ever re-derived COUP membership against Gaia** (40 of 369
  citing papers checked) — a genuine literature gap, i.e. an opening. `[S]`

**Label-noise floor — assume labels are impure, but verify the number.** Kuhn+2019 §3.3 reportedly
gives ~13% contamination for MYStIX/SFiNCs (~87% pure) `[UNVERIFIED — see the box in §1]`. Two
figures that *were* independently verified and can be used now: **Jadhav+2026 WD audit**, >13%
contamination in cluster cores and **>48% in tidal tails** `[S]`; and **OCCAM DR17** (§2b), where
9.3% of Cantat-Gaudin members at P>0.7 fail an independent RV test and 19.8% fail on [Fe/H] `[S]`.

**eROSITA — dead end for this purpose.** Only **DR1** is public (2024-01-31, western hemisphere,
930,203 sources). Its 50%-completeness flux limit implies log L_X ≈ 30.0 at 414 pc, ≈31.4 at 2 kpc
`[I]` — so ~1 M☉ only inside ~500 pc, and it cannot compete with Chandra on any MYStIX-class target.
**No dedicated eROSITA young-cluster (<50 Myr) member catalog exists**; the 2025 counterpart
catalogues are explicitly Gaia-matched → circular. `[S]`

## 3. Lithium — a *dating* tool, not a membership discriminant

**EAGLES** (`2023MNRAS.523..802J`): empirical Li-depletion model, **6,200 stars / 52 clusters**,
2–6000 Myr, **3000 < T_eff/K < 6500**, best precision 0.1 dex in log age. `[S]`

```{important}
Two limits that constrain the Sagitta two-arm plan (see `../design-notes/`, and
`~/phd/sagitta-followup.md`):
1. **EAGLES answers "is this star young?", not "is this star a member."** It returns an age posterior
   for a star already selected by other means.
2. **It needs a Li 6708 Å equivalent width → spectroscopy → it structurally cannot reach G > 18.**
   The per-star age-calibration arm is therefore magnitude-limited and cannot cover the faint regime
   where our parallax-error problem lives.
```

**Successor found — use this one:** `2024MNRAS.534.2014W` (Weaver, Jeffries & Jackson 2024), a neural
network on the same training set, with *"better modelling of the 'lithium dip' at ages <50 Myr and
T_eff∼3500 K"* — **directly in our age window**. Ships as **EAGLES software v2.0**. No later
successor found. `[S]`

**LDB ages are not infallible ground truth** `[S]`: `2023MNRAS.526.1260J` overturned IC 4665's
published LDB age of 32⁺⁴₋₅ Myr — *"the LDB … has not been detected and … the published LDB age should
be interpreted as a lower limit"* — revising it to **55 ± 3 Myr** (~70% error), for one of only ~12
clusters with an LDB age. Note also that modern Li work is itself *"astrometrically and
spectroscopically filtered"* → circular for calibrating an astrometric classifier.

## 4. Hα / accretion — best depth, hard ceiling on recall

```{warning}
**Hα bounds purity, never recall.** Only ~40% of genuine members accrete: *"38%─41% are actively
accreting"* (Venuti+2024, Lagoon, `2024AJ....167..120V`); ~40% CTTS in NGC 2264. So **~60% of true
members are Hα-negative even at 1–3 Myr** (weak-line T Tauri stars). Unusable above ~5–10 Myr.
```

- **IGAPS** (`2020A&A...638A..18M`, VizieR `V/165/igapsdr1`) — **the most calibration-ready product
  found.** 295,365,268 rows, northern plane, with an `emitter` column: **2** = good Hα-excess
  candidate, **1** = marginal, **0** = *tested and in main locus*. Value 0 is a **published negative
  with a graded score, at ~300M scale** — nothing else offers this. Tested to r < 19.5. `[S]`
- **Barentsen IC 1396** (`2011MNRAS.415..103B`) — 158 PMS candidates, 0.2–2.0 M☉ (**56% between
  0.2–0.5 M☉**), 13 < r′ < 20, ≤15% contamination, discriminant is (r′−i′, r′−Hα) only. VizieR
  `J/MNRAS/415/103` even includes a 30-row **explicit-rejects** table. `[S]`
- **Kalari M8** (`2015MNRAS.453.1026K`) — 235 CTTS; the **only mass-binned completeness figure
  anywhere**: *"complete up to 60, 90 and 50 per cent in the 0.2–0.4, 0.4–1, 1–2 M☉ ranges"*.
  Contamination 10–30%. The quotable independence statement: *"we cannot explicitly select for
  cluster membership (using for example proper motions)."* `[S]`
- **Realistic completeness prior:** Fratta+2021 (`2021MNRAS.505.1135F`) — Hα-outlier selection has
  *"completeness … between 3 and 5 per cent … a purity fraction of 81.9 per cent."* High purity,
  catastrophic completeness. `[S]`
- **Practical rule:** use (r−i, r−Hα), never a *g*-based colour plane — Drew+2026
  (`2026MNRAS.545f2137D`) find *"almost 10 per cent of the EB sit above the unreddened MS"* from
  filter-acquisition delays in variable stars. `[S]`

## 5. ⚠️ Directly relevant to our NGC 6383 work

**Kalari 2019** (`2019MNRAS.484.5102K`, *CTTS with VPHAS+ II: NGC 6383*) identified **156 CTTS
photometrically** — independent of astrometry. But VizieR `J/MNRAS/484/5102/table1` **publishes only
the 55 that survived a Gaia DR2 proper-motion cut.** `[S]`

> **The published 55 are CIRCULAR — do not use them as ground truth for NGC 6383.**

The independent 156 are not available machine-readable; they would have to be re-derived from VPHAS+
DR2 (VizieR `II/341`) using the paper's stated colour criteria — estimated **half a day**, and
probably worth it given our NGC 6383 paper. This also bears on the backlog item about re-adjudicating
the "missing Kalari CTTS".

## 6. An independent ceiling on our own method

**Vioque+2023** (`2023AJ....166..183V`) ran HDBSCAN on Gaia DR3 and recovered only **55–60% of
confirmed young stars at 1.5–4 M☉, and 27% at 4–10 M☉** `[S]`. That is a completeness ceiling on
astrometric clustering measured *without* our method — cite it rather than discovering it in review.

## 6b. Attribution and independence corrections `[S]`

- **`2026A&A...706A.341H` is NOT "Hunt & Reffert."** Authors: Hunt, Cantat-Gaudin, Anders, Malhotra,
  Spina, Castro-Ginard, Cavallo. **Reffert is not on it.** Cite accordingly.
- **Jackson+2022 is not "independent of astrometry."** Its likelihood is `L_RV × L_pm` — it *uses*
  proper motion; what it omits is **per-star parallax** (parallax enters only as a cluster-level
  scaling `d_c`, which the authors say has no direct effect on membership). So it is orthogonal to
  *half* our discriminant. Worse for our purpose: **parallax is a pre-screen** — ≥5,764 rows (13.7%)
  were cut on a parallax criterion and carry `P = −1`. Those are **censored, not negative**, so the
  stars our clustering would reject *on parallax* are largely absent from the scored sample.
  Filter to **Λ=665 (GIRAFFE)** only (UVES fibres were preselected on prior membership) and **drop
  all 7 globular clusters** — the authors state non-members cannot be cleanly identified there.
- **"Jackson et al. 2018" does not exist** — no such membership catalog.
- **The ABYSS bibcodes are not OC membership catalogs** (`2023ApJS..266...10K` = targeting strategy;
  `2024AJ....167..125S` = young stars in SDSS spectra), and ABYSS selection *includes phase-space
  position*. Drop unless working <30 Myr.
- **Chemical tagging is dead as a discriminant** — Sinha+2024: field stars matched in
  R_GC/[M/H]/[α/M]/T_eff/log g show only **+0.012 dex** more intrinsic scatter than members. Useful
  only as a veto against a grossly wrong radial MDF.
- **The whole spectroscopic axis is a giant-tip instrument** — APOGEE 7≲H≲13.8, Gaia RVS
  G_RVS≲14. Against DR3 parallax errors of 107 µas at G=18 → 462 µas at G=20, **spectroscopy cannot
  reach our problem.**

**Caveats reported by users of the sets we plan to lean on** `[S]`:

| Set | Reported failure mode |
|---|---|
| H&R 2023 | Their own Paper III: only 79% bound, **11% within 250 pc**. Alfonso+2024: at G>18 it *"reaches σϖ up to ~1.6 mas, an order of magnitude larger errors."* |
| Jackson+2022 | The literature **routinely misdescribes it as parallax-informed** — don't trust second-hand descriptions. Only **5.2% of scored rows** land in the eight interior probability bins (110–340 stars/bin pooled over 69 clusters) → **per-cluster reliability diagrams are not feasible**; pooled ones have wide error bars exactly where calibration fails. |
| OCCAM | No third-party critique found (top 50 of 155, top 40 of 76 checked); citing papers use its gradients, not its membership. |
| Tarricq+2021 | OCCAM IV: NGC 2266's RV came from **one star**. Rule: distrust means from ≤2 members. |
| LAMOST / Fu+2022 | Zhang+2024: systematic [Fe/H] bias below 5000 K; *">10% of clusters exhibit a metallicity dispersion greater than 0.25 dex."* |
| SPICY | Marton+2023: only **753 objects** in common with the Gaia DR3 YSO sample — nearly disjoint populations. |
| WISE-only YSO | Silverberg+2018: false-positive rates **>70%** — disqualifying. |
| Asteroseismic anchors | Sandquist+2013: masses *"systematically too high by as much as 8%"*; in M67 **the two independent anchors disagree with each other**. |
| `ocelot` simulator | Draws errors as **independent univariate Gaussians per dimension** — no Gaia 5-parameter covariance — with an author TODO confirming **no parallax zero-point or correlated systematics**. Both bite hardest at G>18. No independent critique exists; caveats are entirely author-supplied. |

## 6c. Action items

1. **Email `emily.lauren.hunt@univie.ac.at`.** Hunt+2026 Table B.2 — **50,873,539 rows with observed
   *and* true astrometry** — is documented at CDS `J/A+A/706/A341` but **not served** (confirmed four
   ways: HTTP 403, VizieR "table does not exist", FTP listing, full tarball). The paper explicitly
   invites our use case: *"a tool for a future work to derive direct calibration factors for HR24's
   membership lists."* This is the best-shaped truth set in existence for us.
2. **Pull `occam_member-DR17.fits`** from the SDSS SAS and split on `APOGEE2_TARGET1` bit 9.
3. **Re-derive the Kalari+2019 NGC 6383 sample** from VPHAS+ DR2 (§5) — bears on the P07 thread about
   CTTS removed by the 2σ parallax clip.
4. **Audit COUP against Gaia** — nobody has, and it sits exactly on our low-mass regime.

**Framing gift:** the field's reference benchmark, **HR21, states *"we cut all stars fainter than
G = 18"*** — it stops exactly where our question begins. And the 2026 review (`2026arXiv260713711R`)
calls membership lists *"very unsatisfactory"* and demands *"a list of standard star clusters to test
and verify all known methods."* That lane is open. `[S]`

## 7. Tiered recommendation

| Purpose | Use | Why |
|---|---|---|
| **Calibration with negatives** | Kashyap Cyg OB2 `J/ApJS/269/10` | ready-made 3-class labels + accuracies, Gaia-free |
| **Bright-end purity** | **OCCAM DR17** `occam_member-DR17.fits` | spatial-cone selection, ~24,700 real negatives, parallax never used |
| **Scale + negatives** | MYStIX `table7`, SFiNCs `xsources` | large label set; assume impure (Kuhn figure unverified) |
| **Faint Hα negatives at scale** | IGAPS `V/165/igapsdr1` `emitter=0` | ~300M tested sources, graded |
| **M < 0.3 M☉** | COUP, nearest SFiNCs, Barentsen IC 1396, Kalari M8 | only sets reaching the low-mass regime |
| **Purity only** | SPICY `J/ApJS/254/33`, Li/EAGLES | no completeness; spectroscopic sets die at G≈18 |
| **Avoid** | eROSITA, WISE-only YSO catalogs (>70% FPR), 2022–26 IR+astrometry censuses | shallow, impure, or circular |

## Verification caveats
Bibcodes resolved via ADS with titles echoed. Explicit nulls (searched, nothing found): no citing
critique for COUP (40/369 checked), Dunham 2015 (30/317), Richert 2018 (30/126), Marton 2019 (25/86);
Kashyap 2023 has only 6 citations. Quoted sentences came through an automated full-text extractor —
faithful in substance, but **eyeball against the PDFs before manuscript use**, and section numbers
refer to arXiv versions. The M_G↔mass conversions and the eROSITA / Cyg OB2 log L_X figures are own
arithmetic `[I]`; eROSITA's survey PSF could not be verified.
