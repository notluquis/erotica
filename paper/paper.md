<!-- SUBMITTABLE as of 2026-08-03. Every JOSS hard requirement is met:
  archived DOI  10.5281/zenodo.21769959  (CONCEPT DOI — always resolves to the latest
                version; the v0.1.0 version DOI is 10.5281/zenodo.21769960)
  author        sole author, ORCID 0009-0008-4359-2444 verified against the ORCID public API
  licence       AGPL-3.0-or-later, consistent across LICENSE, pyproject.toml and CITATION.cff
  tests + CI    488 tests, GitHub Actions on Python 3.13 and 3.14
Read the state-of-the-field section against ~/phd/software-landscape.md before submitting:
several novelty claims in this programme have been falsified, and the surviving claim here is
deliberately narrow. -->
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
  # Sole author, confirmed by the author 2026-08-03. The repository has 158 commits from a
  # single contributor, and JOSS holds that "purely financial ... and organizational (such as
  # general supervision of a research group) contributions are not considered sufficient for
  # co-authorship". Those who supported the companion science paper are acknowledged instead.
  - name: Lucas Pulgar-Escobar
    orcid: 0009-0008-4359-2444
    affiliation: 1
affiliations:
  - name: Universidad de Concepción, Chile
    index: 1
date: 3 August 2026
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
developed for and validated on the young cluster NGC 6383 [@pulgar2024a; @pulgar2024b].

# Statement of need

*Gaia* DR3 has made open clusters a large-sample statistical enterprise, but the open-source
tooling is fragmented. Membership finders (`UPMASK` [@kronemartins2014], `pyUPMASK`
[@pera2021], HDBSCAN pipelines [@hunt2021]) and parameter fitters (`ASteCA` [@perren2015],
`BASE-9` [@vonhippel2006]) are separate tools with separate data models, so an end-to-end
analysis means glue code that discards each stage's uncertainty before the next consumes it.
`EROTICA` addresses two gaps.

The first is integration: one path from a *Gaia* catalogue to a characterized cluster, with
every intermediate posterior retained rather than collapsed to a point estimate.

The second is calibration, and it needs stating precisely, because most of the adjacent
ground is already occupied. Scoring membership output is not new here — `pyUPMASK` grades
itself with six metrics for the accuracy of probabilistic classification across 600 synthetic
clusters [@pera2021] — and @olivares2018 propagate
observational uncertainties into per-star membership probabilities. What we could not find
reported for stellar clusters is the reliability check itself: whether a stated probability
of 0.7 corresponds to a ~70%
true-member frequency. A 2026 review of membership methods [@ramezani2026] does not raise it,
and the closest validation work [@jackson2022] measures discrimination rather than
calibration. The check is routine for photometric-redshift distributions [@disanto2018;
@myles2021] and, for *galaxy* clusters, at the percent level [@rozo2015] — so the claim is
scoped to stellar clusters, and concerns what is reported rather than what is possible.
`EROTICA` ships reliability diagrams, the Hosmer–Lemeshow statistic, Brier score and
expected/maximum calibration error, with isotonic and Platt recalibration returning a fitted
callable — on `numpy`/`scipy`/`scikit-learn` alone, so they apply to any pipeline's output.

# State of the field

The closest integrated tool is `ASteCA` [@perren2015], whose synthetic-CMD forward-modelling
design directly inspired this work; since v0.5.0 it is sampler-agnostic, exposing a Poisson
likelihood-ratio object for a user-supplied sampler. It does contain a King-radius routine,
private and unwired: a two-parameter least-squares fit to a binned radial profile. On the
structural axis `EROTICA` fits King, EFF and corona profiles as an unbinned Bayesian point
process; the closest prior art is @pera2021king, who fit an elliptical rotated King profile
to spatial data by Bayesian inference.

For isochrones, `EROTICA` samples the binned Poisson Hess-diagram likelihood [@dolphin2002]
with a No-U-Turn Sampler. Bayesian single-cluster CMD fitting is established (`BASE-9`,
@vonhippel2006, non-gradient MCMC), and the gradient-based combination is recent and cited
head-on: @chi2026 apply a No-U-Turn Sampler to differentiable PARSEC isochrones for an open
cluster, and @garling2025 sample a Poisson Hess-diagram likelihood with Hamiltonian Monte
Carlo — both on different likelihoods from the one used here. The isochrone module is
therefore a capability within an integrated pipeline, not a standalone advance.

`ASteCA`, `pyUPMASK` and the survey-scale nested-sampling fits of @plevne2026 are the
reference implementations in this field, and they are the baselines `EROTICA` has to be
measured against. That comparison ships as software: an `ASteCA` adapter
(`erotica.analysis.external.asteca`) wrapping its isochrone and synthetic-cluster machinery,
and a harness scoring `EROTICA`, `ASteCA` and `pyUPMASK` on membership agreement, parameter
recovery against synthetic truth, runtime and calibration. Benchmark results belong to a
companion methods paper and are not reported here.

Where `EROTICA` goes beyond the baselines is a claim about capability, not a measured win: an
end-to-end path that retains every intermediate posterior; an unbinned point-process
structural fit, where the released `ASteCA` routine fits a binned profile by least squares;
and per-star calibration as a first-class, reported output. Calibration is computable for any
method that emits membership probabilities, and the harness computes it for the baselines too
— they are not incapable of reporting it.

# Software design

`EROTICA` is a library, and its governing constraint is that the expensive parts stay
optional. The core install needs only `numpy`, `scipy`, `scikit-learn`, `astropy` and
`hdbscan`; everything requiring a probabilistic-programming stack sits behind a `bayes` extra,
and `import erotica` succeeds with `pymc`, `pytensor`, `arviz`, `numpyro`, `jax` and
`blackjax` all absent. This is enforced rather than asserted: submodules resolve lazily
through a module-level `__getattr__` (PEP 562), each sampler entry point is guarded by a check
that names the missing extra, and continuous integration runs the suite in a job that installs
the package *without* it. The cost is that the dependency graph is no longer readable from the
import statements.

The two fitting modules make different likelihood choices, for a measured reason. The
structural fit treats sky positions as an inhomogeneous Poisson point process with intensity
$\lambda(r) = 2\pi r\,\Sigma(r)$, giving $\log L = \sum_i \log \lambda(r_i) - \Lambda$ — the
continuous form of the Cash statistic [@cash1979] — with $\Lambda$ integrated over the actual
footprint. It is unbinned because binning was tested and failed: under the equal-count annuli
the earlier implementation used, the count per bin is fixed by construction, and the Poisson
dispersion index measures 0.045 against the 1.0 a Poisson likelihood asserts — a roughly
25-fold mis-specification. The isochrone module samples a *binned* Hess-diagram likelihood
[@dolphin2002]; the asymmetry is deliberate, since the colour–magnitude and sky planes pose
different problems, but it is an asymmetry rather than a unified formulation.

Quantities carry `astropy` units on output, and each fit returns `InferenceData` rather than a
summary row, so convergence diagnostics stay attached to the numbers they describe. When a
trace is saved, a sidecar record captures the git commit and dirty flag, the random seeds, the
tracked dependency versions, and a blake2b checksum of every input file. The dependency list
is curated rather than locked, deliberately: the package is imported rather than deployed, and
bit-identical cross-machine results are unattainable once BLAS threading and XLA fusion
reorder floating-point summation.

# Research impact statement

`EROTICA` is new software and its realized external impact is limited: no downstream
dependants, and the two NGC 6383 studies it was built for [@pulgar2024a; @pulgar2024b] are
the authors' own. The evidence offered is of the other admissible kind — reproducible
materials demonstrating capability.

The repository carries a validation programme of 22 scripts under `tools/validation/`. The 13
that produce quoted numbers each commit a JSON sidecar holding the full result, a docstring
stating what would falsify the conclusion, and their negative controls run and reported rather
than assumed. The yield is largely negative results, which is the point:

- The EFF slope estimator is biased high at the sample sizes typical of the *Gaia* cluster
  census, and the bias shrinks as $N$ grows. A survey-scale comparison of slopes would read a
  selection-independent artefact as physics, with sparse clusters appearing systematically
  steeper than rich ones.
- A free background term fabricates a background where the injected truth is exactly zero, and
  biases the slope upward by 5.8$\sigma$; pinning the term removes both effects. The size of
  the artefact depends on the shape being measured, so no single offset corrects it. It
  surfaced in the null cell of an experiment where that control was present as a formality.
- Fitting a circular profile to an elliptical cluster biases the slope *downward*, toward the
  value at which EFF and an untruncated King profile coincide — so an apparent pile-up there
  can be an artefact of the assumed geometry.

The test suite (488 tests) runs in continuous integration on Python 3.13 and 3.14, with a
separate job that installs the `bayes` extra, and is audited by mutation rather than by
coverage: 39
deliberate bugs were re-applied to the shipping source one at a time, and 18 survived. That
falsified this project's own repeated claim that every test carried an oracle independent of
the code under test. Each repair is verified by re-applying the mutation it was written to
catch.

# AI usage disclosure

Development of `EROTICA` was assisted by large language models throughout; this section states
the tools, the scope and the review process.

**Tools.** Anthropic's Claude, accessed through the Claude Code command-line agent. Of the 158
commits on the default branch, 151 carry a `Co-Authored-By` trailer naming the model: Claude
Opus 5 (104), Claude Opus 4.8 (38) and Claude Sonnet 4.6 (9). The trailers are in the git
history and are the authoritative record. The seven without one are early notebook-removal and
housekeeping commits; we have not reconstructed whether they were assisted.

**Scope.** The assistance was substantive rather than confined to language editing:
implementation of the statistical models and their normalisations, design of the validation
experiments, literature search, drafting of the design notes and API documentation, and the
framing of several results including the recoverability findings above.

**Verification, because it is the material point.** AI-generated code produced several results
that were plausible and wrong, and the project adopted a verification discipline in response:
every generator is checked against an external, parameter-free oracle — a closed form, an
analytic limit, or an independent implementation — and every test is verified by re-applying
the bug it is meant to catch. That discipline caught, among others, a synthetic-cluster
generator that filled a cube rather than a sphere; an experiment degenerate with its own
control, which would have supported a clean but false conclusion; a unit error that made a
profile wrong by a factor of 480 with no exception raised; and several novelty claims —
including an earlier and broader version of this paper's own calibration claim — that were
falsified on checking and then narrowed or withdrawn. Each is recorded in the design notes,
which keep the wrong number beside the correction.

**Responsibility.** All scientific claims, the interpretation of every result, and the
decision to publish rest with the authors. Every number reported here is produced by a script
committed to the repository; none originates from a model's assertion.

# Acknowledgements

I gratefully acknowledge support from the ANID BASAL project FB210003 and the SOCHIAS GEMINI
project 32230014, and financial support from the Dirección de Postgrado, Universidad de
Concepción, through its MSc scholarship programme, under which this software was developed.
I thank P. Cerulo for guidance on machine-learning methods and for discussions of Bayesian
modelling with PyMC during the work that led to this package.

# References
