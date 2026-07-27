# The OC census landscape — audits of Hunt & Reffert, and the DR4 opening

_2026-07-26. Recursive citation sweep of the Hunt & Reffert / Emily Hunt lineage (2 full rounds +
a partial third, **not saturated** — see §6). Every bibcode resolved via ADS with its title echoed.
`[S]`=verified, `[I]`=inferred._

## 0. The strategic finding — corrected

```{danger}
**Gaia DR4 is NOT released.** An earlier version of this note claimed it was. Verified against ESA's
own release page (2026-07-27): **"Gaia DR4 (based on 66 months of data) — 2 December 2026"**, listed
as *not yet released*; DR5 not before end of 2030. `[S]`

The error came from seeing DPAC "DR4" papers (Creevey et al. `2026arXiv260700264C`, FLAME masses and
ages; Jamal et al. `2026A&A...711A..62J`) and inferring the data was out. **DPAC pre-publishes
content-description papers ahead of the release** — ESA's page says *"a first summary of the Gaia DR4
contents is published now"*. Papers ≠ data.
```

**What survives, restated honestly:** no DR4 open-cluster catalogue exists — but that is *trivially*
true, since nobody has the data. The real content is a **schedule**, not an open slot:

- **~4 months of runway** (from 2026-07-27) to have a pipeline validated and ready for DR4 day.
- Ramezani+2026 frames its whole review as preparation *"for the forthcoming Gaia DR4"* — the field is
  visibly positioning for the same date. `[S]`
- The competitive advantage is being **ready**, not being first to notice the gap. Everything in this
  note that improves membership on DR3 transfers directly.

Related: **there is no Paper IV or V.** `title:"Improving the open cluster census"` returns exactly
6 records = the 3 papers + their 3 VizieR entries. Paper III (`2024A&A...686A..42H`) remains the
current cleaned catalogue; Hunt+2025/2026 are the completeness/selection-function branch, not a new
census. `[S]`

## 1. The seed lineage (author lists verified)

| Bibcode | Paper | Authors | Cites |
|---|---|---|---|
| `2021A&A...646A.104H` | Census I — clustering-algorithm comparison, DR2 | Hunt & Reffert | 132 |
| `2023A&A...673A.114H` | Census II — all-sky DR3 catalogue | Hunt & Reffert | 357 |
| `2024A&A...686A..42H` | Census III — cleaned catalogue via mass/radius/dynamics | Hunt & Reffert | 194 |
| `2025A&A...699A.273H` | Anticentre completeness | Hunt, Cantat-Gaudin, Anders, **+ Reffert** | 7 |
| `2026A&A...706A.341H` | Selection function | Hunt, Cantat-Gaudin, Anders — **no Reffert** | 2 |

⚠ Reffert **is** an author on the 2025 paper and **is not** on the 2026 one. Do not generalise either
way. `[S]`

## 2. H&R's own numbers — the baseline any claim must beat `[S]`

- 7,167 clusters; 2,387 new; 4,105 in the "highly reliable" cut; **1,152 MWSC clusters that should
  have been detectable were not** — tentatively *"may not be real"*.
- Only **5,647/7,167 (79%) compatible with bound clusters**; **just 11% within 250 pc**.
- Census complete within 1.8 kpc **only above 230 M☉**; total MW population ~1.3×10⁵, of which
  **~4% is currently known**.
- Selection function: **80,590 injection-and-retrievals**, logistic detectability model at 94.53%
  training accuracy; a 25 km/s orbital-speed boost can give ~3× higher detection probability.

## 3. Independent audits — every measured error rate `[S]`

| Source | Measurement | Data H&R did not use |
|---|---|---|
| **Jadhav+2026** `2026arXiv260711299J` | Of 235 OC–WD pairs in 80 clusters: **>28% spurious overall, >48% in tidal tails, >13% in cores** — *"significant field-star contamination in current Gaia-based catalogues"* | UV–IR SEDs + N-body grids |
| **Yu+2026** `2026Univ...12...78Y` | Virial reclassification of 4,809 candidates → 3,897 bound; **93.60% precision, 80.04% recall** vs the prior classification (83.55% on the high-quality subset) | virial theorem |
| **Jadhav+2025** `2025A&A...704A..50J` | 122 tidal-tail catalogues graded: 15 gold / 55 silver / **51 bronze — 42% fail the majority of diagnostics** | N-body expectations |
| **Ritter+2026** `2026arXiv260719735R` | HSC 2686 and Lynga 3 — **neither is real**; *"considerable doubt on the veracity of many newly identified OCs with modest numbers of stellar members"* | object-level spectroscopy |
| **Barth+2025** `2025ApJ...985..129B` | HDBSCAN in chemo-dynamical space gives **low OC recovery** for every parameter combination; dynamics beats chemistry | GALAH DR4 abundances |
| **Chi+2025** `2025PASJ...77.1050C` | Own pipeline: 2,932 candidates → 872 → **739 high-confidence (25% survival)** — a purity prior for any blind Gaia search | — |
| **Neumannová+2026** `2026arXiv260715149N` | Masses/radii for ~7,000 OCs; literature values *"widely disagree"*; **DR3 parallaxes make 3D shapes untrustworthy beyond 500 pc** (spurious needle elongation) | — |
| **Gagné+2026** `2026arXiv260215695G` | **~4 km/s residual systematics in Gaia DR3 RVs for A-type stars** — a floor on any RV-based validation | MOCA/BANYAN Σ |

**The debate is not one-directional.** Counter-evidence that H&R *under*-counts: Donada+2026
(`2026A&A...710A.381D`) chemically confirms UBC 1052 as genuine (<0.03 dex across 20 elements);
Ferreira+2026 (`2026MNRAS.548ag621F`) finds **31 additional** anticentre clusters (+31% at 3–4 kpc);
Liu+2025 (`2025AJ....169..326L`) finds **more** members than existing catalogues in nearby OCs. The
credible reading: **H&R over-counts *bound objects* and under-counts *members per object*.** `[I]`

## 4. Reusable artefacts

- **MiMO** `2025AJ....170..288L` — Bayesian CMD mixture with field contamination as an explicit
  component; **per-star photometric membership probabilities + full likelihood chains + open code**,
  1,232 clusters. **The most likely calibration testbed we have** — it publishes a *posterior*, and
  nobody has validated its reliability curve. `[S]`
- **LAMOST-MRS-O** `2026RAA....26e5001Z` — **RVs for 1,033 OCs**, [Fe/H]+abundances for 446, ~7,000
  member stars. Internal accuracy: RV offset <1 km/s (σ<10), [Fe/H] offset 0.02–0.04 dex. The largest
  independent-data resource for validating a membership list, and **not yet used as an audit**. `[S]`
- **Li+2025 Bayes factors** `2025ApJ...994..227L` — BF>100 separates genuine clusters from chance
  field overdensities. A ready-made *cluster-level* reliability statistic. `[S]`
- **C-4** `2026MNRAS.550g1219T` — NN selection function via artificial-cluster injection
  (extragalactic, LEGUS/NGC 628); cites Hunt+2025 as precedent.
- Competing catalogues: Dias+2026 (`2026AJ....171...24D`, 178 new), CANDiSC (`2026A&A...705A.244O`,
  NIR consensus detector, self-reported **FPR < 5%**), Li Zhong-Mu 2024 (`2024RAA....24e5014L`, BSEC,
  83 new), Chi+2025 (739 beyond 5 kpc).

## 5. The calibration gap — now documented by a review

**No counter-example found.** No paper performs a *calibration* check on an OC membership catalogue
(does the set assigned p=0.8 actually contain 80% true members?). What exists is adjacent: MiMO
publishes posteriors but never validates their reliability curve; Li+2025 validates *clusters*, not
stars; Malhotra+2026 (`2026A&A...706A..62M`) gives per-star uncertainties on **masses**, not
membership. `[S]`

**The citation that justifies the project in one sentence** — Ramezani+2026
(`2026arXiv260713711R`), a review: *"the current situation of membership lists is very
unsatisfactory"*, *"most methods are based on poor statistical numerics"*, and it **calls for a
standard benchmark list of clusters spanning age/distance/reddening/metallicity/mass to test and
verify all methods — which does not exist.** `[S]`

**The Hunt injection-recovery machinery is author-only.** Hunt+2026 has **2 citations total**
(Donada+2026, Li Lu+2026) — neither reuses it; Hunt+2025's 7 citations include none that run
injection-recovery on the H&R catalogue. `[S]`

## 6. Saturation — honest accounting

**The sweep did NOT reach saturation.** Two full rounds plus a partial third. Round 1 enumerated
Hunt+2025 (7 cites) and Hunt+2026 (2 cites) completely, but the three large seeds (357+194+132
citations) were sampled via **8 topic-sliced queries**, not read exhaustively. **Round 2 was still
producing new names** — Ramezani's tidal-tails companion (`2026arXiv260713747R`), Sharma+2025
(`2025A&A...704A.167S`), Ghasemi+2026, Biswas+2026, Rosen+2026 — so a third full round would
very likely add more.

**Not examined:** the pre-2025 slice of Census II's citations beyond the topic filters; the full
194-citation list of Census III outside the bound/unbound filter; and any sub-literature reachable
only through non-astrometric keywords. Treat this note as a strong sample, not a census of the census.
