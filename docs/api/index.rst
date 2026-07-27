API reference
=============

EROTICA's public API. These are the classes and functions re-exported at the top level of
the package (``erotica.__all__``); the modules they live in carry additional internals not
listed here.

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
