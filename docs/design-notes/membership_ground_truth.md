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

**G > 18 — yes, ~30 clusters.** Kuhn+2019 (`2019ApJ...870...32K`) matched MYStIX+SFiNCs to Gaia DR2:
*"The median magnitude of these sources is G=18.1 mag (inter-quartile range: 16.6–19.1 mag)"* `[S]`.
Half the X-ray-selected members are fainter than 18.1 — and that is the *Gaia-matched subset*, so it
understates the true depth (of 30,839 YSOs, 20,716 matched, **17,509 with 5-parameter solutions** —
that last number is the usable label count).

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

**Label-noise floor — quote this rather than assuming clean labels:** Kuhn+2019 §3.3, an
independent Gaia-based audit of MYStIX/SFiNCs: *"contamination rates were about 13%, with … individual
systems mostly falling into the range 7–15%"* → **~87% pure**. `[S]`

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

## 7. Tiered recommendation

| Purpose | Use | Why |
|---|---|---|
| **Calibration with negatives** | Kashyap Cyg OB2 `J/ApJS/269/10` | ready-made 3-class labels + accuracies, Gaia-free |
| **Scale + negatives** | MYStIX `table7`, SFiNCs `xsources` | 17,509 usable labels, **~13% noise floor** |
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
