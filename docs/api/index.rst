API reference
=============

EROTICA's public API — a **curated** selection, not an exhaustive one, and the difference is
stated here rather than left for a reader to discover.

This page lists **32** names: the 12 in ``erotica.__all__`` plus 20 more drawn from
``erotica.analysis.__all__``, which itself exports **118**. So roughly 90 public names in the
analysis layer are *importable and supported but not listed on this page* — among them
``distance_model``, ``fit_parallax_model``, ``lambda_msr``, ``half_mass_relaxation_time``,
``coulomb_argument_from_mass_function`` and ``identifiability_report``.

.. warning::

   An earlier version of this paragraph called these *"the classes and functions re-exported at
   the top level of the package (``erotica.__all__``)"*. That was wrong in both directions at
   once: the page lists **more** than ``erotica.__all__`` contains, and far **less** than the
   analysis layer exports. Do not read absence from this page as evidence that a name is private —
   check ``__all__``. Genuinely private names are prefixed with an underscore.

Clustering & data
-----------------

.. currentmodule:: erotica

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   core.Clustering
   core.HDBSCANEstimator
   io.DataLoader
   preprocess.DataPreprocessor

.. autosummary::
   :toctree: generated
   :nosignatures:

   utils.compare_datasets

Analysis
--------

.. currentmodule:: erotica.analysis

.. autosummary::
   :toctree: generated
   :template: autosummary/class.rst
   :nosignatures:

   ClusterAnalyzer
   ClusterDynamicsAnalyzer
   ClusterInferenceAnalyzer
   ClusterStructureAnalyzer
   ClusterFigureBuilder
   IsochroneFitter
   PhotometricMassEstimator

Reproducibility
---------------

A result is reproducible only if you can say *what code, what inputs and what randomness*
produced it. :func:`~erotica.analysis.build_metadata` records all three; saving a trace with
:func:`~erotica.analysis.store_trace_results` writes it as a JSON sidecar automatically. See
:doc:`/design-notes/decisions` for what each field is for and why.

.. currentmodule:: erotica.analysis

.. autosummary::
   :toctree: generated
   :nosignatures:

   build_metadata
   write_metadata
   git_provenance
   file_checksum
   dependency_versions
   store_trace_results
   load_results
   summarize_trace
   posterior_mode

Radial structure
----------------

Radial-profile fitting is an **unbinned inhomogeneous Poisson point process**: the likelihood is
``sum(log lambda(r_i)) - Lambda``, with no binning step to choose and no Gaussian approximation to
break in the sparse outer bins that set ``R_t``. King and EFF share that machinery, so
:func:`compare_radial_profiles` can put a Bayes factor between them. See
:doc:`/design-notes/king_model_validity` for what the fits can and cannot identify.

.. currentmodule:: erotica.analysis

.. autosummary::
   :toctree: generated
   :nosignatures:

   king_unbinned
   eff_unbinned
   compare_radial_profiles
   king_profile
   eff_surface_density
   king_expected_count
   king_expected_count_weighted
   eff_expected_count
   radial_density_profile

Synthetic clusters
------------------

Validation data. Drawing from a smooth King or EFF profile begs the question when the thing being
validated is a profile fit, so :func:`fractal_cluster` generates substructure by the Goodwin &
Whitworth (2004) box-fractal construction instead.

.. currentmodule:: erotica.analysis

.. autosummary::
   :toctree: generated
   :nosignatures:

   fractal_cluster
   radial_profile_of
