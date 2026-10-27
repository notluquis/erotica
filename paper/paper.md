---
title: 'EROTICA: an integrated Bayesian pipeline for Gaia-era open-cluster analysis'
tags:
  - Python
  - astronomy
  - open clusters
  - Gaia
  - Bayesian inference
  - stellar populations
authors:
  # L.P.-E.: sole implementer; ORCID verified against the ORCID public API 2026-08-03.
  # P.C.: added 2026-09-22 for project direction -- JOSS (submitting.html) counts "active
  # project direction" as authorship, distinct from the purely financial/organizational
  # contributions it excludes. Affiliation confirmed against \institute in the co-authored
  # A&A manuscript (aa52082-24.tex); ORCID verified against the public ORCID registry.
  - name: Lucas Pulgar-Escobar
    orcid: 0009-0008-4359-2444
    affiliation: 1
  - name: Pierluigi Cerulo
    orcid: 0000-0003-0703-3123
    affiliation: 2
affiliations:
  - name: Universidad de Concepción, Chile
    index: 1
  - name: Departamento de Ingeniería Informática y Ciencias de la Computación, Universidad de Concepción, Chile
    index: 2
date: 26 October 2026
bibliography: paper.bib
---

# Summary

`EROTICA` (Estimation, Recovery & Optimization, together with Inference, for Cluster
Analysis) is a Python package for identifying and characterizing open clusters in *Gaia*
data. It combines density-based membership (an HDBSCAN pseudo-probability sweep with
Bayesian proper-motion and parallax refinement), Bayesian structural fitting of King, EFF
and corona profiles, gradient-based isochrone fitting, and dynamical diagnostics behind a
single API that returns an ArviZ `InferenceData` posterior for every fit. It also ships
diagnostics for asking whether its own membership probabilities are calibrated. It was
developed for, and used in, the NGC 6383 studies [@pulgar2024a; @pulgar2024b], the second
now accepted at Astronomy & Astrophysics.

# Statement of need

*Gaia* DR3 has made open clusters a large-sample statistical enterprise, but the open-source
tooling is fragmented. Membership finders (`UPMASK` [@kronemartins2014], `pyUPMASK`
[@pera2021], HDBSCAN pipelines [@hunt2021]) and parameter fitters (`ASteCA` [@perren2015],
`BASE-9` [@vonhippel2006]) are separate tools with separate data models, so an end-to-end
analysis means glue code that discards each stage's uncertainty before the next consumes it.
`EROTICA` is written for astronomers who derive membership, structure and ages of Galactic open
clusters from *Gaia* astrometry and photometry, and addresses two gaps for them.

The first is integration: one path from a *Gaia* catalogue to a characterized cluster, with
every intermediate posterior retained rather than collapsed to a point estimate.

The second is calibration, stated precisely because most of the adjacent ground is already
occupied. Scoring membership output is not new here — `pyUPMASK` grades
itself with six probabilistic-classification metrics on 600 synthetic clusters [@pera2021] — and @olivares2018 propagate
observational uncertainties into per-star membership probabilities. The check `EROTICA` adds is
reliability: whether a stated probability of 0.7 corresponds to a ~70%
true-member frequency. The open-cluster validations we read on 26 October 2026 report
discrimination [ROC AUC, @jackson2022] or the completeness and contamination of recovered member
lists [@shen2026], and a 2026 review recommends that calibration curves accompany member lists
[@ramezani2026b]. The check is routine for photometric-redshift distributions [@disanto2018;
@myles2021] and, for *galaxy* clusters, at the percent level [@rozo2015].
`EROTICA` ships reliability diagrams, the Hosmer–Lemeshow statistic, Brier score and
expected/maximum calibration error, with isotonic and Platt recalibration returning a fitted
callable — on `numpy`/`scipy`/`scikit-learn` alone, so they apply to any pipeline's output.

# State of the field

The closest integrated tool is `ASteCA` [@perren2015], whose synthetic-CMD forward-modeling
design directly inspired this work; since v0.5.0 it is sampler-agnostic, exposing a Poisson
likelihood-ratio object for a user-supplied sampler. On the structural axis `EROTICA` fits
King, EFF and corona profiles as an unbinned Bayesian point process. The antecedents are by the
`ASteCA` group: an elliptical rotated King profile fitted to spatial data by Bayesian inference
[@pera2021king], and a generalized King profile fitted through `ASteCA` to over 1000 clusters
[@pera2024]. In the released `ASteCA` (v0.7.0) the King routine is private and unwired, a
two-parameter least-squares fit to a binned radial profile, and `pyUPMASK` has no structural or
isochrone stage, so this pipeline was built rather than added to either.

For isochrones, `EROTICA` evaluates an unbinned per-star likelihood against MIST isochrones
interpolated at fixed equivalent evolutionary phase [@dotter2016; @choi2016], with stellar mass
and unresolved binaries marginalized along the track, and samples it with the No-U-Turn Sampler
of `nutpie`. `IsochroneFitterKW` adds a free scale $k$ on the catalog errors, so each star's
width is $\sqrt{(k e_i)^2 + \sigma_{\mathrm{int}}^2 + \sigma_{\mathrm{floor}}^2}$, and fits inside a
magnitude window, which isolates the pre-main sequence when one isochrone cannot describe both
sequences. On the 201 NGC 6383 members in G 14–19 a slow test reruns that fit under CCM89 extinction and asserts $\hat R \le 1.01$ and
bulk ESS $> 400$ on log age, no divergences, and the committed age quantiles. The age is that of
the window, and the distance prior selects it along an age–distance ridge; in one injection cell
the chains did not converge from the default starts. `BASE-9` established Bayesian single-cluster
CMD fitting with non-gradient MCMC [@vonhippel2006]. Gradient-based fitting is recent: @chi2026
apply a No-U-Turn Sampler to differentiable PARSEC isochrones with per-star masses sampled as
latent parameters, and @garling2025 sample a binned Poisson Hess-diagram likelihood
[@dolphin2002] with Hamiltonian Monte Carlo. The isochrone module is a capability
within an integrated pipeline, not a standalone advance.

`ASteCA`, `pyUPMASK` and the survey-scale nested-sampling fits of @plevne2026 are the
reference implementations in this field, and they are the baselines `EROTICA` has to be
measured against. That comparison ships as software: an `ASteCA` adapter
(`erotica.analysis.external.asteca`) wrapping its isochrone and synthetic-cluster machinery,
and a harness scoring `EROTICA`, `ASteCA` and `pyUPMASK` on membership agreement, parameter
recovery, runtime and calibration. Benchmark results belong to a
companion methods paper. What `EROTICA` adds is a capability, not a
measured win: one path that keeps every intermediate posterior, with calibration reported per
star; the harness computes calibration for the baselines too.

# Software design

`EROTICA` is a library. Its default sampler, `nutpie` with `pymc` models compiled through
`jax`, is a core dependency; the alternatives (`numpyro`, `blackjax`) sit behind a `bayes` extra.
Submodules resolve lazily through a module-level `__getattr__` (PEP 562), so `import erotica`
loads none of the sampling stack; the cost is that the dependency graph is no longer readable
from the import statements.

Both fitting modules use unbinned likelihoods, because binning was tested and failed in each. The
structural fit treats sky positions as an inhomogeneous Poisson point process with intensity
$\lambda(r) = 2\pi r\,\Sigma(r)$, giving $\log L = \sum_i \log \lambda(r_i) - \Lambda$ — the
continuous form of the Cash statistic [@cash1979] — with $\Lambda$ integrated over the actual
footprint. Under the *approximately* equal-count annuli the earlier implementation used, the count per bin is nearly fixed by
construction, and the Poisson dispersion index measures 0.045 against the 1.0 a Poisson
likelihood asserts — a roughly 25-fold mis-specification. The isochrone module's earlier binned
Hess-diagram likelihood locked posteriors onto its precomputed grid's reference point, and a
binned model rebuilt at every evaluation biased the distance modulus by 0.2–0.3 mag on synthetic
clusters (`docs/design-notes/decisions.md`).

Quantities carry `astropy` units on output, and each fit returns `InferenceData` rather than a
summary row, so convergence diagnostics stay attached to the numbers they describe. When a
trace is saved, a sidecar record captures the git commit and dirty flag, the random seeds, the
tracked dependency versions, and a blake2b checksum of every input file. The dependency list
is curated rather than locked, because the package is imported, not deployed.

# Research impact statement

`EROTICA` is new software: it has no downstream dependents, and its use to date is
Pulgar-Escobar's own — the NGC 6383 studies [@pulgar2024a; @pulgar2024b], the second of
which, co-authored with Cerulo, is now accepted at Astronomy & Astrophysics. The evidence
offered here is of the other admissible kind: reproducible materials demonstrating
capability.

<!-- 49 / 27: tools/release/validation_json_count.py at 6387d5a and at this branch (2026-10-26).
It counts top-level scripts whose source names a .json tracked under tools/validation/; it does not
check for a falsification criterion or a negative control, so the text does not claim them (it read
"40 ... 26 ...", a figure nothing reproduced). TODO(v0.3.0): re-count at the release commit. -->
The repository carries a validation program of 49 scripts under `tools/validation/`, 27 of
which commit their result as a JSON file. The yield is largely negative results, which is the
point:

- The EFF slope estimator is biased high at the sample sizes typical of the *Gaia* cluster
  census, and the bias shrinks as $N$ grows. A survey-scale comparison of slopes would read a
  selection-independent artifact as physics, with sparse clusters appearing systematically
  steeper than rich ones.
- A free background term fabricates a background where the injected truth is exactly zero, and
  biases the slope upward by 5.8$\sigma$; pinning the term removes both effects. The size of
  the artifact depends on the shape being measured, so no single offset corrects it.
- Fitting a circular profile to an elliptical cluster biases the slope *downward*, toward the
  value at which EFF and an untruncated King profile coincide — so an apparent pile-up there
  can be an artifact of the assumed geometry.

In continuous integration on Python 3.14, with a separate job for the `bayes`
extra: **562 tests at the v0.2.0 release**, 558 without the extra.

The suite is audited by mutation rather than by coverage: 39 deliberate bugs were re-applied
to the shipping source one at a time, and 18 survived, falsifying this project's own repeated
claim that every test carried an independent oracle.

# AI usage disclosure

`EROTICA` was developed with generative-AI assistance throughout.

**Tools and versions.** Anthropic's Claude, via the Claude Code command-line agent, applied
to code, tests, documentation and this paper's text.
<!-- TODO(v0.3.0): the counts below are pinned to commit 9e61814 on this PR's branch, not to the
release: the v0.3.0 release commit does not exist yet. Before tagging, regenerate at the parent of
the commit that writes them (tools/release/RELEASE_INSTRUCTIONS.md, "AI-usage disclosure") and
replace this paragraph's counts; --check must pass at that sha. -->
As of commit `9e61814`, 486 of the 544 commits carry a `Co-Authored-By` trailer naming the model:
Claude Opus 5 (301), Claude Opus 5.5 (104), Claude Opus 4.8 (38), Claude Sonnet 5 (20), Claude
Sonnet 5.5 (14) and Claude Sonnet 4.6 (9). A further 13, from 19–20 October 2026, carry none but
were AI-assisted, with the model unrecorded; they are listed in
`tools/release/ai_assisted_without_trailer.txt`. Of the rest, 38 are merge commits (31 of pull
requests) and 7 are early notebook-removal and housekeeping work, not reconstructed as assisted.
The git history and that list are the authoritative record.

**Scope.** The assistance was substantive — implementation, test and experiment scaffolding,
literature search, and drafting of design notes, documentation and this paper's prose — but
under review throughout: models proposed, implemented and drafted; the authors decided what
to measure and what to adopt or publish, including narrowing or withdrawing claims a model
had drafted (below).

**Verification, because it is the material point.** AI-generated code produced several
results that were plausible and wrong, and the project adopted a verification discipline in
response: every generator is checked against an external, parameter-free oracle — a closed
form, an analytic limit, or an independent implementation — and every test is verified by
re-applying the bug it is meant to catch. That discipline caught, among others, a
synthetic-cluster generator that filled a cube rather than a sphere; a unit error that made a profile wrong by a factor of 480 with no
exception raised; and several novelty claims — including an earlier, broader version of this
paper's own calibration claim — that were falsified on checking and then narrowed or
withdrawn.

**Responsibility.** The authors assert that human authors reviewed, edited and validated all
AI-assisted outputs and made the core design decisions, and take responsibility for every
scientific claim and the decision to publish; every number reported here comes from a script
committed to the repository, not a model's assertion.

# Acknowledgements

We gratefully acknowledge support from the ANID BASAL project FB210003 and the SOCHIAS GEMINI
project 32230014. L.P.-E. acknowledges financial support from the Dirección de Postgrado,
Universidad de Concepción, through its MSc scholarship program, under which this software
was developed. P.C. acknowledges support from the UdeC VRID Iniciación fund 2025001386INI.
The funders had no role in the design of the software, the analyses, or the writing of this
paper.

# References
