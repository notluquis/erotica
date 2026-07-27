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
