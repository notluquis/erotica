<!-- DRAFT — NOT SUBMITTABLE. JOSS hard-requires automated tests + CI + an
archived DOI, none of which COSMIC has yet (see ~/phd/cosmic-package.md
release-blockers). This is scaffolding for P02, not a ready submission.
Author list, order, and ORCIDs are PLACEHOLDERS — confirm before use. -->
---
title: 'COSMIC: an integrated, calibrated Bayesian pipeline for Gaia-era open-cluster analysis'
tags:
  - Python
  - astronomy
  - open clusters
  - Gaia
  - Bayesian inference
  - stellar populations
authors:
  # TODO(author): confirm final author list + ORDER + ORCIDs before submission.
  # Affiliation confirmed: Universidad de Concepción (work developed during the first
  # author's MSc there; udec email lescobar2019@udec.cl is canonical).
  - name: Lucas Pulgar-Escobar
    orcid: 0000-0000-0000-0000  # TODO: add ORCID
    affiliation: 1
  - name: Nicolás Henríquez Salgado
    orcid: 0000-0000-0000-0000  # TODO: add ORCID
    affiliation: 1
affiliations:
  - name: Universidad de Concepción, Chile
    index: 1
date: 19 July 2026
bibliography: paper.bib
---

# Summary

`COSMIC` (Characterization Of Star clusters using Machine-learning Inference and
Clustering) is a Python package for identifying and characterizing open clusters
in Gaia data. It combines density-based membership (an HDBSCAN pseudo-probability
sweep with Bayesian proper-motion/parallax refinement), Bayesian structural
fitting (King profile), gradient-based isochrone fitting (a No-U-Turn Sampler over
a Poisson Hess-diagram likelihood; under validation), and dynamical diagnostics — in a single API
that emits standardized ArviZ `InferenceData` posteriors for every fit. It was
developed for and validated on the young cluster NGC 6383 [@pulgar2024a; @pulgar2024b].

# Statement of need

Gaia DR3 has made open clusters a large-sample statistical enterprise, but the
open-source tooling is fragmented. Membership finders (`UPMASK`/`pyUPMASK`
[@pera2021], HDBSCAN pipelines [@hunt2021]) and parameter fitters (`ASteCA`
[@perren2015], `BASE-9`, `isochrones`) are separate tools with separate data
models. More decisively, a 2026 review of every open-cluster membership method
[@ramezani2026] — spatial, kinematic, photometric, maximum-likelihood, Bayesian,
and machine-learning — never once asks whether the probabilities they produce are
*calibrated*: whether a stated membership probability of 0.7 corresponds to a
~70% true-member frequency. Their outputs are rankings or heuristic scores, not
probabilities validated against observed member frequencies. `COSMIC` addresses two gaps: (1) a single pipeline
from raw Gaia catalog to characterized cluster with per-fit posterior provenance,
and (2) membership probabilities accompanied by calibration diagnostics
(reliability diagram / Hosmer–Lemeshow / Brier score) and, where they are
miscalibrated, an explicit recalibration recipe. To our knowledge no released
open-cluster pipeline reports per-star membership calibration, though probability
calibration is standard practice for photometric-redshift PDFs [@myles2021].
Completeness-dependent results (luminosity/mass functions, spatial statistics) are
corrected with the Gaia DR3 open-cluster selection function [@hunt2026sf].

# State of the field

The closest integrated tool is `ASteCA` [@perren2015], which since v0.5.0 is
sampler-agnostic (a Poisson likelihood-ratio object; the user supplies the
sampler) and, in its current version, provides no gradient sampler, no King/radius
fit, and no automated test suite. `COSMIC`'s membership statistic
$\tilde{p} = f_i\,p_\mathrm{HDBSCAN}$ was introduced in @pulgar2024b; the new
contribution here is its *calibration* (the closest open-cluster work,
@jackson2022, validates by discrimination, not calibration). For isochrone
fitting, `COSMIC` samples the binned Poisson Hess-diagram likelihood [@dolphin2002]
with a No-U-Turn Sampler; Bayesian single-cluster CMD fitting is itself established
(`BASE-9` [@vonhippel2006], non-gradient MCMC), and the gradient-based combination
is recent and cited head-on: @chi2026 apply a No-U-Turn Sampler with differentiable
PARSEC isochrones to an open cluster, but as a per-star rotation+binarity model
rather than a binned Hess-diagram likelihood; @garling2025 sample a Poisson
Hess-diagram likelihood with Hamiltonian Monte Carlo, but over *linear*
star-formation-history coefficients rather than nonlinear single-population
isochrone parameters; and Bayesian fundamental-parameter fitting of open clusters
is now performed at survey scale [5056 clusters via nested sampling, @plevne2026].
`COSMIC` therefore frames its isochrone module as a *capability* within an
integrated, calibrated pipeline — NUTS over nonlinear single-population isochrone
parameters through a differentiable Poisson–Hess forward model — not as a
standalone advance; its principal novelty is the membership-calibration protocol
and the end-to-end integration with reproducible provenance.

`COSMIC` is built to *complement* this ecosystem, not to displace it. `ASteCA`
[@perren2015] — whose synthetic-CMD forward-modelling design directly inspired
this work — and `pyUPMASK` [@pera2021] remain references for their respective
tasks, and `COSMIC` ships an `ASteCA` adapter (`cosmic.analysis.external.asteca`)
so the tools compose rather than fragment the ecosystem further. Following field
practice, the companion methods paper reports an honest cross-comparison on shared
clusters — including NGC 6383, one of the 20 validation clusters of @perren2015 —
verifying that `COSMIC` *agrees* with `ASteCA`, `pyUPMASK`, and the survey-scale
fits of @plevne2026 on the axes they share (membership assignment,
fundamental-parameter recovery, runtime), and contributing the one axis none of
them currently reports: per-star membership calibration. The contribution is the
missing calibration layer and its integration into a reproducible pipeline — a
step toward *less* fragmentation — not a claim of superiority.

# Acknowledgements

<!-- TODO: funding, Gaia/DPAC acknowledgement, collaborators. -->

# References
