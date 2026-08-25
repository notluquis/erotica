"""High-level clustering utilities for EROTICA."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence

import numpy as np
from astropy.table import QTable
from tqdm.auto import tqdm

from ._constants import DEFAULT_SCORE_METHOD, DEFAULT_SEARCH_METHOD, SUPPORTED_SEARCH_METHODS
from ._estimator import HDBSCANEstimator
from ._plots import (
    plot_cluster_members,
    plot_cluster_persistence,
    plot_condensed_tree,
    plot_grid_search_results,
    plot_members_vs_persistence,
    plot_pm_scatter,
    plot_probability_histogram,
)
from ._search import run_grid_search, run_optuna_search
from ._style import apply_default_style
from ._summary import build_cluster_summary, clustering_statistics, combine_datasets


class NoCandidateClusters(RuntimeError):
    """La búsqueda no encontró ningún candidato. **Es una respuesta, no un fallo.**

    Sobre un campo sin estructura no hay nada que encontrar, y decirlo es correcto. Lo que estaba mal
    era el tipo: llegaba como `RuntimeError` pelado, indistinguible de que la búsqueda se rompiera,
    así que cada consumidor tenía que decidirlo leyendo el texto del mensaje — y uno de ellos lo hacía
    con un comentario en vez de con código.

    Medido en el hilo CTRL-seeds: **82 ocurrencias sobre 24 semillas de campo liso** —70 celdas de
    los arms 5D y 12 de los 3D— y **cero** en cuanto hay estructura. La rama no es rara: es la
    respuesta normal a un cielo vacío. El prerregistro de ese hilo llegó a escribir que era «una
    caída puntuada como acierto» antes de leer el mensaje, que es justo el costo de no poder
    distinguirlas.

    Hereda de `RuntimeError` a propósito: quien ya lo captaba sigue funcionando, y quien quiera
    separar «no hay nada» de «me rompí» ahora puede hacerlo sin mirar una cadena.
    """


class Clustering:
    """Perform HDBSCAN clustering with optional hyper-parameter search."""

    def __init__(
        self,
        data: QTable,
        bad_data: QTable | None = None,
        *,
        search_method: str = DEFAULT_SEARCH_METHOD,
        sqlite_path: str = "optuna_study.db",
        study_name: str | None = None,
        legacy_cluster_selection: bool = False,
    ):
        """Construct a clustering run.

        Parameters
        ----------
        legacy_cluster_selection
            **Exists solely to reproduce results published before 2026-08-03**, and there
            is no other reason to set it. When ``True``, ``search_pseudoprobability``
            picks the reported cluster with :meth:`_cluster_label_for_size`, the defective
            size-matching selector of issue #7: it compares a count of *condensed-tree
            rows* against *flat-cluster point counts* and, when they fail to coincide,
            falls back to the largest non-noise cluster — which on a contaminated field is
            the field. The default ``False`` uses :meth:`_cluster_label_from_tree`, which
            resolves the tree node to the label its leaves actually carry.

            This is a keyword rather than a monkey-patched attribute because reproducing
            an old result must not require reaching inside the object.
        """
        if search_method not in SUPPORTED_SEARCH_METHODS:
            raise ValueError("search_method must be 'grid' or 'optuna'.")
        self.search_method = search_method
        self.data = data
        self.bad_data = bad_data
        self.storage_url = f"sqlite:///{sqlite_path}" if sqlite_path else None
        self.study_name = study_name
        self.legacy_cluster_selection = bool(legacy_cluster_selection)

        self.clusterer = None
        self.cv_results_ = None
        self.best_params_ = None
        self.best_score_ = None
        self.combined_data = None
        self.pseudoprobability_results_ = None
        self.pseudoprobability_selected_ = None
        self.pseudoprobability_sweep_track_ = None
        self._study = None
        self._pareto_trials = []

        apply_default_style()

    # ------------------------------------------------------------------
    # Main search routine
    # ------------------------------------------------------------------
    def search(
        self,
        columns: Sequence[str],
        *,
        persistence_threshold: float = 0.0,
        param_grid: dict[str, list] | None = None,
        grid_kwargs: dict | None = None,
        optuna_search_space: dict[str, dict] | None = None,
        n_trials: int = 50,
        n_jobs: int = -1,
        sampler: str = "TPESampler",
        sampler_kwargs: dict | None = None,
        score_method: str | Iterable[str] = DEFAULT_SCORE_METHOD,
        hdbscan_kwargs: dict | None = None,
    ) -> None:
        """Run the requested hyper-parameter search and annotate results."""
        hdbscan_kwargs = hdbscan_kwargs or {}
        X = self.data[list(columns)].to_pandas().values

        if self.search_method == "grid":
            results = run_grid_search(
                X,
                persistence_threshold=persistence_threshold,
                param_grid=param_grid,
                grid_kwargs=grid_kwargs,
                hdbscan_kwargs=hdbscan_kwargs,
            )
            self.cv_results_ = results["cv_results"]
        else:
            methods = [score_method] if isinstance(score_method, str) else list(score_method)
            results = run_optuna_search(
                X,
                persistence_threshold=persistence_threshold,
                search_space=optuna_search_space,
                n_trials=n_trials,
                n_jobs=n_jobs,
                sampler_name=sampler,
                sampler_kwargs=sampler_kwargs,
                score_methods=methods,
                hdbscan_kwargs=hdbscan_kwargs,
                storage_url=self.storage_url,
                study_name=self.study_name,
            )
            self._study = results["study"]
            self._pareto_trials = results.get("pareto", [])

        self.clusterer = results["clusterer"]
        self.best_params_ = results.get("best_params")
        self.best_score_ = results.get("best_score")

        self._annotate_results()

    # ------------------------------------------------------------------
    # Pseudoprobability search
    # ------------------------------------------------------------------
    def search_pseudoprobability(
        self,
        columns: Sequence[str] = ("pmra", "pmdec"),
        *,
        min_cluster_size_samples: Iterable[int] = range(10, 300),
        min_samples: int | None = None,
        probability_threshold: float = 0.5,
        min_cluster_members: int | None = None,
        max_cluster_members: int | None = None,
        selection: str = "max_persistence",
        select_cluster: bool = True,
        probability_method: str = "hdbscan",
        recovery_frequency: str = "any",
        approx_min_span_tree: bool = False,
        match_reference_implementation: bool = True,
        hdbscan_kwargs: dict | None = None,
    ) -> None:
        """Sweep min_cluster_size, build pseudoprobability, select best cluster.

        ``probability_times`` = fraction of sweep steps in which each source was in any cluster.
        ``probability`` = per-star membership score x ``probability_times``.

        Parameters
        ----------
        probability_method
            Which per-star score multiplies ``probability_times``.

            ``"hdbscan"`` (default) uses ``probabilities_``. ``"soft"`` uses the
            ``all_points_membership_vectors`` column of the selected cluster.

            .. note::
               **``"soft"`` measures far better and is still not the default, deliberately.**

               ``probabilities_`` is ``min(lambda_i, lambda_death(C)) / lambda_death(C)``. Under
               ``cluster_selection_method="eom"``, which this method hardcodes, a parent selected
               over its sub-clusters makes the ``min()`` clamp: measured **83.6% of the cluster
               receives exactly 1.0**, so the score cannot rank those points at all. That is the
               mechanism behind this package's benchmark AUC of 0.776 against ASteCA's 0.917.

               Soft membership has no clamp. Measured over 2700 fits, 15 seeds with 5 held out::

                   metric                probabilities_    soft
                   ROC-AUC                    0.7706      0.9867
                   average precision          0.2339      0.9126
                   reliability (miscal.)      0.2088      0.0155
                   resolution                 0.0381      0.0621
                   held-out ROC-AUC           0.7644      0.9883

               Held-out exceeds train, so it is not fitted to the seeds. And it wins on
               calibration as well as ranking: after out-of-sample isotonic recalibration --
               which repairs reliability but *cannot* create resolution -- soft retains 1.9x the
               resolution, so its advantage survives the transformation that could have erased it.

               ⚠ **The validated benchmark has now run, and it cuts that gain by ~8x.** Those
               numbers score ``probabilities_`` ALONE; what ships is the product with
               ``probability_times``. On ``benchmark_erotica_vs_asteca.py``, 108 cells with half
               the seeds held out, paired per cell::

                   arm                  dROC (held-out)      dAP (held-out)     wins
                   3d soft - 3d       +0.0249 +- 0.0189   +0.0413 +- 0.0223    27/54
                   5d soft - 5d       +0.0309 +- 0.0133   +0.0740 +- 0.0160    27/54

               The gain is **real** -- dAP on 5d is 4.6 standard errors, and every held-out delta
               exceeds its train counterpart, the opposite of overfitting. But it is +0.03 ROC,
               not +0.22: the ``f_i`` factor multiplies both scores and washes most of the
               advantage out. **Scoring one factor of a product and reporting it as the product's
               improvement was the error**, and it is why this stayed off the default until the
               harness ran.

               It also does **not** close the gap it was meant to: 0.5883 average precision
               against ASteCA's 0.8615. Soft membership recovers roughly a quarter of that
               distance. The rest is not in the per-star score.

               ``wins 27/54`` is exactly half, so the mean gain comes from large wins in some
               cells rather than a uniform shift -- an improvement in expectation, not in every
               case. Worth enabling on a 5D feature set; not worth a claim.

               Cost: ``prediction_data=True`` on the final fit only. Measured 4.9 s for n=12000
               with 3 clusters -- ``all_points_membership_vectors`` does linear scans per
               (point, cluster) pair, so it is superlinear in cluster count.
        recovery_frequency
            What the sweep term counts. ``"any"`` (default) is the shipped
            ``probability_times``: the fraction of swept ``min_cluster_size`` values in which
            the source landed in **any** cluster. ``"target"`` counts only the steps in which
            it landed in the step's **target** cluster, identified without ground truth by
            maximum Jaccard overlap with the finally selected member set. See
            :meth:`_target_recovery_frequency`.

            ``probability_times`` is written on both paths and always means "any". The
            target-aware vector is written as ``probability_times_target`` **only** when
            ``recovery_frequency="target"``, so recovering both products means running the
            sweep once per setting; a default run leaves the target column absent.
        match_reference_implementation
            Match the original Java HDBSCAN* reference implementation. Default ``True``, which
            is what this package has always used — but it was hardcoded and unexplained until
            2026-08-04, and it is the flag that actually moves the numbers.

            .. important::
               **It does FOUR things, not the three the hdbscan source lists**
               (``hdbscan_.py:743-746``)::

                   min_samples        -= 1
                   min_cluster_size   += 1
                   approx_min_span_tree = False
                   + an extra -1 assignment in do_labelling (_hdbscan_tree.pyx:508-512)

               The fourth was found because ``mri=True`` is **not** reproducible by applying the
               three documented shifts by hand.

               ⚠ **The +/-1 shifts mean the effective hyperparameters are not the ones passed.**
               ``min_cluster_size=N`` runs at ``N+1`` and ``min_samples=M`` at ``M-1``. Anything
               quoting a swept ``min_cluster_size`` from a run with this flag is off by one, and
               the legacy NGC 6383 notebook names a variable ``effective_mcs`` while storing the
               *requested* value — so the published "optimal min cluster size" labels are the
               requested numbers, not the effective ones. ``pseudoprobability_selected_`` now
               carries ``effective_min_cluster_size`` and ``effective_min_samples`` so a caller
               can report what actually ran.

               **On real data it is not cosmetic.** Largest-cluster membership on the NGC 6383
               catalogues, ``mri=True`` versus ``False``::

                   radius  mcs   mriT    mriF   Jaccard
                     40'    50   6241    8926    0.6992
                     50'    50  24236   15439    0.6370
                     60'    50  28041   40138    0.6986
                     70'   150  29329   47134    0.6222
                     (the other four configurations)  0.99+

               Four of eight configurations disagree on ~35% of the membership, and one differs
               by 60% in count. The pattern is erratic in both radius and ``mcs``, which is the
               signature of the +/-1 landing on either side of a cluster-selection boundary.

               **Kept True because it is measurably better, held-out.** Paired against injected
               truth, identical frames, 8 seeds held out of 16::

                   metric   block      delta (False - True)   wins False
                   ROC      HELD-OUT      -0.0208 +- 0.0053      9/32
                   AP       HELD-OUT      -0.0202 +- 0.0062     11/32
                   purity   HELD-OUT      +0.0116 +- 0.0333     15/32
                   recall   HELD-OUT      -0.0001 +- 0.0625      2/32

               Turning it off costs ~4 standard errors of ROC and ~3 of average precision, and
               the effect is LARGER held-out than in training. Purity and recall are unchanged.
               So the flag earns its place on discrimination, and that is now measured rather
               than inherited.
        approx_min_span_tree
            Whether to let hdbscan build an **approximate** minimum spanning tree. Default
            ``False``, i.e. exact — which is *not* hdbscan's own default.

            .. important::
               **This is a named parameter and not a buried default because it can change
               results, and because hdbscan's documentation is wrong about a related knob.**

               hdbscan says ``leaf_size`` "does not alter the resulting clustering"
               (``hdbscan_.py:977-981``). False on the path used here: ``algorithm="best"`` on 2D
               euclidean data dispatches to Borůvka, whose approximate MST resets distance bounds
               only when a pass makes no progress (``_hdbscan_boruvka.pyx:585-598``), so leaf
               geometry decides tie-breaks and therefore labels.

               Measured on synthetic contaminated frames, 12 seeds, n=1500, contamination 0.9::

                   approx=True   leaf_size    5    10    20    40   100   200   400
                                 ROC-AUC   .8887 .8944 .9006 .9009 .9020 .9043 .9043
                   approx=False  ROC-AUC   .9043 at EVERY leaf_size, labels byte-identical

               So the approximation is **strictly lossy**, not neutral noise: large ``leaf_size``
               "wins" only by converging on the answer the exact tree computes directly. Tuning
               ``leaf_size`` would be tuning the approximation error.

               ⚠ **CORRECTION, 2026-08-04: on this package's path the switch is a NO-OP, and the
               real-data table below is vacuous rather than reassuring.**
               ``base_kwargs`` hardcodes ``match_reference_implementation=True``, and that flag
               *forces* ``approx_min_span_tree = False`` (``hdbscan_.py:743-746``). So the exact
               tree was already in use, both arms of the comparison below were exact, and
               Jaccard 1.0000 was guaranteed by construction. Verified directly: with
               ``match_reference_implementation=True``, setting ``approx_min_span_tree`` either
               way returns byte-identical labels.

               The parameter is kept because it makes the exactness explicit instead of an
               accident of another flag, and because it becomes live the moment
               ``match_reference_implementation`` is turned off. The synthetic ``leaf_size``
               numbers above were measured *without* that flag and remain valid for that case.

               **The flag that actually changes results here is
               ``match_reference_implementation``**, which is hardcoded ``True`` and does four
               things, not the three its own docs list: ``min_samples -= 1``,
               ``min_cluster_size += 1``, ``approx_min_span_tree = False``, and an extra ``-1``
               assignment in ``do_labelling`` (``_hdbscan_tree.pyx:508-512``). Measured on a
               contaminated frame: largest cluster 236 vs 228, noise 1011 vs 1027 — and it is
               **not** reproducible by applying the three documented shifts by hand, which is how
               the fourth effect was found.

               Original table follows, retained as the record of a measurement that could not
               have come out any other way. Largest-cluster membership, every radius and two
               ``min_cluster_size``:

                   radius   n       mcs   approx   exact   Jaccard   speedup
                     40'    23740    50     6241    6241   1.0000     12.3x
                     50'    38677    50    24236   24236   1.0000      5.0x
                     50'    38677   150    13974   13974   1.0000      1.1x
                     60'    56784    50    28041   28041   1.0000      0.9x
                     60'    56784   150    20620   20620   1.0000      1.2x
                     70'    78893    50    53024   53024   1.0000      1.6x

               **Jaccard 1.0000 in all seven configurations** — not "close", identical. So
               switching the default does **not** move any published NGC 6383 number, while it
               does remove the pathological case on contaminated synthetic frames. Cost is
               neutral to favourable: the exact tree ranged from 0.9x to 12x *faster*, never
               meaningfully slower.

               ⚠ Note the direction of the earlier synthetic timing: at n=1500 the exact tree was
               1.4x SLOWER, which is the regime that does not matter here. Timing the wrong scale
               would have argued against the better default.

               Set ``True`` only to reproduce a result produced before 2026-08-04.

               It also makes ``relative_validity_`` trustworthy: that score is computed from
               ``minimum_spanning_tree_`` and silently inherited the approximation.
        selection
            Which sweep step to keep. See :meth:`_select_pseudoprobability_result`.
        """
        base_kwargs = {
            "algorithm": "best",
            "cluster_selection_method": "eom",
            "allow_single_cluster": False,
            "metric": "euclidean",
            "match_reference_implementation": bool(match_reference_implementation),
            # EXACT minimum spanning tree by default, not hdbscan's approximate one. This is a
            # named parameter rather than a buried default BECAUSE IT CHANGES RESULTS -- see the
            # `approx_min_span_tree` entry in this method's docstring.
            #
            # hdbscan's docstring claims leaf_size "does not alter the resulting clustering"
            # (hdbscan_.py:977-981). False here: algorithm="best" on 2D euclidean data dispatches
            # to Boruvka, whose APPROXIMATE MST resets distance bounds only when a pass makes no
            # progress (_hdbscan_boruvka.pyx:585-598), so leaf geometry decides tie-breaks and
            # therefore labels.
            #
            # The first response to that was to pin leaf_size=40. That was wrong, and measuring it
            # properly says why -- 12 seeds, n=1500, contamination 0.9:
            #
            #   approx=True    leaf_size    5   10   20   40  100  200  400
            #                  ROC-AUC   .8887 .8944 .9006 .9009 .9020 .9043 .9043
            #   approx=False   ROC-AUC    .9043 at EVERY leaf_size, labels byte-identical
            #
            # So the approximation is not neutral noise -- it is strictly LOSSY, and the pinned 40
            # was measurably worse than exact (0.9009 vs 0.9043). Large leaf_size "wins" only by
            # converging on the answer the exact MST computes directly. Tuning leaf_size would
            # have been tuning the approximation error.
            #
            # Cost of exactness: 1.16x, 30.0 ms -> 35.0 ms per fit at n=1500. At ~290 sweep steps
            # that is a few seconds, against a sweep already measured at 3 s of NUTS sampling for
            # a full structural fit. Worth it.
            #
            # It also makes `relative_validity_` trustworthy: that score is computed from
            # minimum_spanning_tree_ and silently inherited the approximation.
            "approx_min_span_tree": bool(approx_min_span_tree),
            **(hdbscan_kwargs or {}),
        }
        if probability_method not in ("hdbscan", "soft"):
            raise ValueError("probability_method must be 'hdbscan' or 'soft'.")
        if recovery_frequency not in ("any", "target"):
            raise ValueError("recovery_frequency must be 'any' or 'target'.")
        if recovery_frequency == "target" and not select_cluster:
            # The target-aware term is defined RELATIVE to the selected cluster, so with no
            # selection there is no target to match sweep steps against. Accepting the argument
            # and quietly computing the "any" product instead left the caller no way to tell it
            # had been ignored, and produced a `probability` column indistinguishable from a
            # genuine target-aware run.
            raise ValueError(
                "recovery_frequency='target' requires select_cluster=True: the target-aware "
                "sweep term is defined relative to the selected cluster."
            )
        self._probability_method = probability_method

        # MST not needed during sweep — skip to save ~20% per iteration
        sweep_kwargs = {**base_kwargs, "gen_min_span_tree": False}
        final_kwargs = {**base_kwargs, "gen_min_span_tree": True}
        if probability_method == "soft":
            # Only the FINAL fit needs it: prediction_data builds a tree, a k-NN query and the
            # exemplar arrays, and the sweep discards every intermediate model anyway.
            final_kwargs = {**final_kwargs, "prediction_data": True}

        samples = list(min_cluster_size_samples)
        X = self.data[list(columns)].to_pandas().values
        self._warn_on_coincident_rows(X, columns, samples, min_samples, selection)
        n_sources = len(self.data)
        n_samples = len(samples)
        labels_matrix = np.full((n_sources, n_samples), -1, dtype=np.int32)
        results: list[dict] = []
        sweep_track: list[dict] = []  # all mcs → cluster size, for plotting

        for i, min_cluster_size in enumerate(tqdm(samples, desc="mcs sweep", unit="mcs")):
            estimator = HDBSCANEstimator(
                min_cluster_size=int(min_cluster_size),
                min_samples=min_samples,
                **sweep_kwargs,
            ).fit(X)
            model = estimator.model_
            labels = np.asarray(model.labels_, dtype=np.int32)
            labels_matrix[:, i] = labels

            probability_times = np.mean(labels_matrix[:, : i + 1] != -1, axis=1)
            probability = np.asarray(model.probabilities_, dtype=float) * probability_times
            tree = model.condensed_tree_.to_pandas()
            if tree.empty or "lambda_val" not in tree or "parent" not in tree:
                continue

            lambda_value = float(tree["lambda_val"].max())
            desired_len = self._desired_tree_branch_size(tree)
            sweep_track.append(
                {"min_cluster_size": int(min_cluster_size), "desired_len": int(desired_len)}
            )
            if min_cluster_members is not None and desired_len < min_cluster_members:
                continue
            if max_cluster_members is not None and desired_len > max_cluster_members:
                continue
            if len(np.unique(labels)) <= 1:
                continue
            if np.count_nonzero(probability > probability_threshold) < 1:
                continue

            # Score this sweep step by the persistence of the cluster the selector would
            # actually return, not by a count of condensed-tree rows. `cluster_persistence_`
            # is HDBSCAN's own stability measure, is indexed by flat-cluster label, and was
            # already being collected here and discarded. See `selection=` in the docstring.
            persistence = np.asarray(getattr(model, "cluster_persistence_", []), dtype=float)
            step_label = self._cluster_label_from_tree(tree, labels, len(labels))
            step_persistence = (
                float(persistence[step_label]) if 0 <= step_label < persistence.size else 0.0
            )

            results.append(
                {
                    "min_cluster_size": int(min_cluster_size),
                    "desired_len": int(desired_len),
                    "lambda_value": lambda_value,
                    "selected_label": int(step_label),
                    "selected_persistence": step_persistence,
                    "relative_validity": float(getattr(model, "relative_validity_", np.nan)),
                    "cluster_persistence": persistence,
                }
            )

        if not results:
            raise NoCandidateClusters("Pseudo-probability search did not find candidate clusters.")

        selected = self._select_pseudoprobability_result(results, selection)
        final_probability_times = np.mean(labels_matrix != -1, axis=1)
        final_estimator = HDBSCANEstimator(
            min_cluster_size=selected["min_cluster_size"],
            min_samples=min_samples,
            **final_kwargs,
        ).fit(X)

        self.clusterer = final_estimator.model_
        self.best_params_ = {
            "min_cluster_size": selected["min_cluster_size"],
            "min_samples": min_samples,
            **final_kwargs,
        }
        self.best_score_ = selected["lambda_value"]
        self.pseudoprobability_results_ = results
        self.pseudoprobability_sweep_track_ = sweep_track
        # Record the EFFECTIVE hyperparameters, not just the requested ones.
        # match_reference_implementation silently shifts both (hdbscan_.py:743-746), so
        # `min_cluster_size=N` runs at N+1 and `min_samples=M` at M-1. Reporting the requested
        # value is how a published "optimal min_cluster_size" ends up off by one -- the legacy
        # NGC 6383 notebook even named a variable `effective_mcs` while storing the requested
        # number. Both are kept so a caller can quote whichever it means, and say which.
        _req_mcs = int(selected["min_cluster_size"])
        _req_ms = int(min_samples) if min_samples is not None else _req_mcs
        _shift = 1 if match_reference_implementation else 0
        self.pseudoprobability_selected_ = {
            **selected,
            "probability_times": final_probability_times.copy(),
            "requested_min_cluster_size": _req_mcs,
            "requested_min_samples": _req_ms,
            "effective_min_cluster_size": _req_mcs + _shift,
            "effective_min_samples": _req_ms - _shift,
            "match_reference_implementation": bool(match_reference_implementation),
        }
        # `labels_matrix` is passed by reference and NOT stored on the instance. It is
        # (n_sources x n_sweep_steps) int32 -- 92 MB for the 70' NGC 6383 catalogue at 290
        # steps -- and the only thing downstream needs from it is one length-n vector, which
        # `_annotate_pseudoprobability_results` computes while the matrix is still in scope.
        self._annotate_pseudoprobability_results(
            probability_times=final_probability_times,
            desired_len=selected["desired_len"],
            probability_threshold=probability_threshold,
            select_cluster=select_cluster,
            sweep_labels=labels_matrix,
            recovery_frequency=recovery_frequency,
        )

    # ------------------------------------------------------------------
    # Error-aware pseudoprobability (Monte-Carlo over Gaia errors)
    # ------------------------------------------------------------------
    def search_pseudoprobability_error_aware(
        self,
        columns: Sequence[str] = ("pmra", "pmdec"),
        *,
        error_columns: Sequence[str] | None = None,
        corr_columns: dict[tuple[str, str], str] | None = None,
        n_mc: int = 100,
        min_cluster_size_samples: Iterable[int] = range(10, 300, 10),
        min_samples: int | None = None,
        random_state: int = 0,
        hdbscan_kwargs: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Error-aware recovery frequency, folding Gaia astrometric errors into f_i.

        For each of ``n_mc`` Monte-Carlo draws every source is perturbed by its
        (correlated) Gaia covariance, the ``min_cluster_size`` sweep is re-run, and
        cluster membership is recorded. The returned frequency is the mean clustered
        fraction over draws x resolutions; its spread is the error-induced membership
        uncertainty. Faint sources with large parallax/PM errors flicker in and out
        across draws and earn a correctly lower, hedged frequency instead of a hard cut.

        Adds ``pFreqMC`` and ``pFreqMC_std`` columns to ``self.data`` and returns
        ``(f_mean, f_std)``. This is the frequentist, error-aware layer; the calibrated
        Bayesian posterior is a separate step (see the membership guide). The validated
        :meth:`search_pseudoprobability` is untouched.

        Note that, like the base sweep, features are used on their raw scale -- mixing
        units (parallax with PM) still requires standardizing ``columns`` beforehand.
        """
        from ._error_aware import error_aware_pseudoprobability, gaia_covariance

        cols = list(columns)
        X = self.data[cols].to_pandas().values
        cov = gaia_covariance(self.data, cols, error_columns, corr_columns)
        f_mean, f_std = error_aware_pseudoprobability(
            X,
            cov=cov,
            n_mc=n_mc,
            min_cluster_size_samples=min_cluster_size_samples,
            min_samples=min_samples,
            random_state=random_state,
            hdbscan_kwargs=hdbscan_kwargs,
        )
        self.pfreq_error_aware_ = f_mean
        self.pfreq_error_aware_std_ = f_std
        self.data["pFreqMC"] = f_mean
        self.data["pFreqMC_std"] = f_std
        return f_mean, f_std

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def show_results(self) -> None:
        if self._study is None:
            print("Run .search(...) first.")
            return
        trial = self._pareto_trials[0] if self._pareto_trials else self._study.best_trial
        print("Params:", trial.params)
        if hasattr(trial, "values"):
            print("Objectives:", tuple(f"{v:.6f}" for v in trial.values))
        else:
            print("Objective:", f"{trial.value:.6f}")
        if self._pareto_trials:
            print(f"Pareto front size: {len(self._pareto_trials)}")

    def get_best_params(self) -> dict | None:
        return self.best_params_

    def save_results(self, filename: str, format: str = "csv") -> None:
        if self.combined_data is None:
            raise ValueError("No results to save; run search first.")
        self.combined_data.write(filename, format=format, overwrite=True)

    def clustering_statistics(self, show_outliers: bool = False) -> None:
        table = self.combined_data or self.data
        stats = clustering_statistics(table, include_outliers=show_outliers)
        if stats["clusters"] == 0 and not np.isfinite(stats["mean"]):
            print("No clusters found")
            return
        print("Clustering Statistics:", stats)

    # ------------------------------------------------------------------
    # Plot wrappers
    # ------------------------------------------------------------------
    def plot_grid_search_results(self) -> None:
        if not self.cv_results_:
            raise ValueError("No CV results; run grid search first.")
        plot_grid_search_results(self.cv_results_)

    def plot_pm_scatter(self, pm_columns=("pmra", "pmdec"), show_outliers=False, clusters=None):
        table = self.data.copy()
        plot_pm_scatter(
            table,
            pm_columns=pm_columns,
            show_outliers=show_outliers,
            clusters=clusters,
        )

    def plot_probability_histogram(self) -> None:
        table = self.data if self.combined_data is None else self.combined_data
        plot_probability_histogram(table)

    def plot_cluster_members(self, show_outliers=False) -> None:
        table = self.combined_data or self.data
        plot_cluster_members(table, show_outliers=show_outliers)

    def plot_cluster_persistence(self) -> None:
        summary = self.get_cluster_summary(include_noise=True)
        plot_cluster_persistence(summary)

    def plot_condensed_tree(
        self,
        figsize: tuple[float, float] = (8, 6),
        cmap: str = "viridis",
        select_clusters: bool = True,
        label_clusters: bool = False,
        save_path: str | None = None,
    ) -> None:
        if self.clusterer is None:
            raise RuntimeError("Run search or search_pseudoprobability first.")
        plot_condensed_tree(
            self.clusterer,
            figsize=figsize,
            cmap=cmap,
            select_clusters=select_clusters,
            label_clusters=label_clusters,
            save_path=save_path,
        )

    def plot_mcs_sweep(self, figsize=(7, 5), save_path: str | None = None) -> None:
        """Plot cluster size vs min_cluster_size from pseudoprobability sweep."""
        if self.pseudoprobability_sweep_track_ is None:
            raise RuntimeError("Run search_pseudoprobability first.")
        import matplotlib.pyplot as plt

        mcs_vals = [r["min_cluster_size"] for r in self.pseudoprobability_sweep_track_]
        sizes = [r["desired_len"] for r in self.pseudoprobability_sweep_track_]
        best_mcs = self.pseudoprobability_selected_["min_cluster_size"]
        max_size = max(sizes)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(mcs_vals, sizes, color="steelblue")
        ax.axvline(
            best_mcs,
            color="steelblue",
            linestyle="--",
            label=f"Optimal Min Cluster Size: {best_mcs}",
        )
        ax.axhline(
            max_size, color="olivedrab", linestyle="--", label=f"Max Cluster Size: {max_size}"
        )
        ax.set_xlabel("Min Cluster Size")
        ax.set_ylabel("Cluster Size")
        ax.legend()
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
        plt.show()

    def plot_members_vs_persistence(self, show_outliers: bool = False) -> None:
        summary = self.get_cluster_summary(include_noise=show_outliers)
        plot_members_vs_persistence(summary)

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def get_cluster_summary(
        self,
        pm_columns: Sequence[str] = ("pmra", "pmdec"),
        include_noise: bool = True,
    ):
        table = self.combined_data or self.data
        if table is None:
            raise ValueError("No data available. Run search() or assign .data first.")
        persistence = np.asarray(
            getattr(getattr(self, "clusterer", None), "cluster_persistence_", []),
            dtype=float,
        )
        return build_cluster_summary(
            table,
            pm_columns=pm_columns,
            include_noise=include_noise,
            persistence_array=persistence,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _annotate_pseudoprobability_results(
        self,
        *,
        probability_times: np.ndarray,
        desired_len: int,
        probability_threshold: float,
        select_cluster: bool,
        sweep_labels: np.ndarray | None = None,
        recovery_frequency: str = "any",
    ) -> None:
        if self.clusterer is None:
            raise RuntimeError("No clustering model available.")
        labels = np.asarray(self.clusterer.labels_, dtype=int)
        self.data["cluster_hdbscan"] = labels
        self.data["probability_hdbscan"] = self.clusterer.probabilities_
        self.data["probability_times"] = probability_times
        self.data["probability"] = (
            np.asarray(self.clusterer.probabilities_, dtype=float) * probability_times
        )
        self.data["outlier_score"] = self.clusterer.outlier_scores_

        if select_cluster:
            # Resolve the label from the condensed tree, not by matching a row count against a
            # cluster size. The size-matching path succeeded 23 times out of 83 and fell back to
            # "largest cluster" -- the field -- on the rest. See issue #7 and
            # _cluster_label_for_size's danger note. `Clustering(..., legacy_cluster_selection=True)`
            # restores the old behaviour for reproducing results published before 2026-08-03.
            if self.legacy_cluster_selection:
                selected_label = self._cluster_label_for_size(labels, desired_len)
            else:
                selected_label = self._cluster_label_from_tree(
                    self.clusterer.condensed_tree_.to_pandas(), labels, len(labels)
                )

            # Optional: replace the HDBSCAN membership strength with the soft-clustering column
            # of the SELECTED cluster. Off by default -- see the method's docstring for why the
            # measured advantage is not yet sufficient grounds to move a default.
            if getattr(self, "_probability_method", "hdbscan") == "soft":
                self._soft_column_info = None
                soft = self._soft_membership_column(selected_label, len(labels))
                if self._soft_column_info is not None:
                    self.pseudoprobability_selected_ = {
                        **self.pseudoprobability_selected_,
                        "soft_column": self._soft_column_info,
                    }
                if soft is not None:
                    self.data["probability_soft"] = soft
                    self.data["probability"] = soft * probability_times

            # Target-aware sweep term. Computed here rather than in the sweep loop because it
            # needs `selected_label`, and here rather than on the instance because it needs
            # the sweep label matrix, which must not outlive this call.
            if sweep_labels is not None and recovery_frequency == "target":
                f_target, info = self._target_recovery_frequency(
                    sweep_labels, labels == selected_label
                )
                self.data["probability_times_target"] = f_target
                score = np.asarray(
                    self.data["probability_soft"]
                    if "probability_soft" in self.data.colnames
                    and getattr(self, "_probability_method", "hdbscan") == "soft"
                    else self.data["probability_hdbscan"],
                    dtype=float,
                )
                self.data["probability"] = score * f_target
                self.pseudoprobability_selected_ = {
                    **self.pseudoprobability_selected_,
                    "target_recovery": info,
                }

            retained = np.asarray(self.data["probability"], dtype=float) > probability_threshold
            self.data["cluster"] = np.where((labels == selected_label) & retained, labels, -1)
            self.pseudoprobability_selected_ = {
                **self.pseudoprobability_selected_,
                "selected_cluster": selected_label,
                "probability_threshold": probability_threshold,
            }
        else:
            self.data["cluster"] = labels

        self.combined_data = combine_datasets(self.data, self.bad_data)

    def _soft_membership_column(self, selected_label: int, n_rows: int):
        """Soft-membership score for the SELECTED cluster, or ``None`` if unavailable.

        Why this exists: ``probabilities_`` is ``min(lambda_i, lambda_death(C)) / lambda_death(C)``
        (``_hdbscan_tree.pyx:519-557``). Under ``cluster_selection_method="eom"`` -- which this
        package uses -- when a parent is selected over its sub-clusters, ``lambda_i`` is measured
        against the *sub*-cluster while the denominator is the *parent's* lower death lambda, so
        the ``min()`` clamps. Measured: **83.6% of an EOM-merged cluster gets exactly 1.0**, against
        15.7% under leaf selection. A score that is identical for most of its cluster cannot rank
        those points at all, and no threshold repairs it.

        ``all_points_membership_vectors`` is computed from distance-to-exemplars and has no such
        clamp.

        Three properties of that function, read from the source, that this method has to respect:

        * **Rows do not sum to 1.** They sum to ``in_cluster_probs[i]`` (``prediction.py:760-762``),
          measured mean 0.63. The residual is the implicit "belongs to no cluster" mass. So a
          column is a score in [0, 1], not a normalised posterior, and must not be renormalised.
        * **Column j is NOT ``labels_ == j``, and this method therefore resolves the mapping
          rather than assuming it.** ``all_points_membership_vectors`` builds its columns from
          ``sorted(condensed_tree_._select_clusters())`` (``prediction.py:658``), and
          ``_select_clusters`` (``plots.py:235-245``) returns one entry per label **present in**
          ``labels_``, in ascending label order, as ``groups[label].min()``. So

              column c  <->  present_labels[argsort(raw)[c]]

          and reading ``soft[:, selected_label]`` is right only when that is the identity, which
          needs *both* an ascending ``raw`` *and* a gap-free label range. The second condition
          fails whenever ``do_labelling`` reassigns every point of a selected cluster to noise --
          ordinary here, because ``match_reference_implementation=True`` adds an extra ``-1``
          assignment (``_hdbscan_tree.pyx:508-512``) beyond its three documented effects. The
          missing label is skipped and **every higher label's column shifts down by one**.

          Measured on 46 synthetic cells that produced a cluster (``soft_column_alignment.py``):
          ``raw`` ascending 46/46, label range gap-free 44/46, so the mapping was the identity in
          44 and shifted in 2. In both shifted cells the old code's bounds check happened to fire
          and the run fell back to ``probabilities_``; the 108-cell benchmark logs the same event
          5 / 1 / 1 times across its three selectors as "soft membership unavailable".

          **The bounds check was never sufficient**: it is a check on the index, not on identity.
          With three or more present labels and a gap below the top -- present ``[0, 2, 3]``,
          selected ``2`` -- the naive index is in range and silently returns label ``3``'s column.
          That case was not observed in 46 cells, and it is reachable, which is why the mapping is
          now computed instead of assumed.
        * It is **not** interchangeable with ``membership_vector``, which uses a different formula
          (``max_l/(max_l - h)`` and ``dist**0.5 * outlier**2.0`` versus ``exp(-max_l/h)`` and
          ``dist * outlier``); measured max abs difference 0.13 on identical points.

        Returns ``None`` rather than raising: an unavailable soft vector is a reason to fall back
        to ``probabilities_``, not to lose the run.
        """
        if selected_label < 0:
            return None
        try:
            import hdbscan as _hdbscan

            soft = np.atleast_2d(
                np.asarray(_hdbscan.all_points_membership_vectors(self.clusterer), dtype=float)
            )
        except Exception as exc:  # unavailable without prediction_data=True, among others
            warnings.warn(
                f"soft membership unavailable ({type(exc).__name__}); "
                "falling back to probabilities_.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        column = self._soft_column_for_label(selected_label)
        self._soft_column_info = {
            "selected_cluster": int(selected_label),
            "column": None if column is None else int(column),
            "n_columns": int(soft.shape[1]),
            "remapped": bool(column is not None and column != selected_label),
        }
        if soft.shape[0] != n_rows or column is None or column >= soft.shape[1]:
            warnings.warn(
                f"soft membership shape {soft.shape} does not admit label {selected_label} "
                f"over {n_rows} rows; falling back to probabilities_.",
                RuntimeWarning,
                stacklevel=3,
            )
            return None
        return np.clip(soft[:, column], 0.0, 1.0)

    def _soft_column_for_label(self, selected_label: int) -> int | None:
        """Column of ``all_points_membership_vectors`` that holds ``selected_label``.

        ``None`` when the mapping cannot be resolved, which is a reason to fall back to
        ``probabilities_`` rather than to guess. See :meth:`_soft_membership_column` for why
        the identity mapping is not safe to assume.
        """
        try:
            labels = np.asarray(self.clusterer.labels_, dtype=int)
            present = sorted({int(v) for v in np.unique(labels) if v >= 0})
            raw = [int(v) for v in self.clusterer.condensed_tree_._select_clusters()]
        except Exception:  # a private hdbscan API; a version change must not lose the run
            return None
        if len(raw) != len(present) or selected_label not in present:
            return None
        label_of_column = [present[int(j)] for j in np.argsort(np.asarray(raw))]
        return int(label_of_column.index(selected_label))

    @staticmethod
    def _target_recovery_frequency(
        sweep_labels: np.ndarray, member_mask: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        """Fraction of sweep steps in which each source landed in the step's TARGET cluster.

        The shipped ``probability_times`` counts steps in which a source was clustered into
        **anything**, so a field star that sits firmly inside a field cluster at every step
        scores 1.0. Nothing in that quantity refers to the cluster the pipeline finally
        returns, which is why it cannot respond to a better choice of cluster. This counts the
        same steps against the target instead.

        The target at step *i* is chosen **without ground truth**, as the cluster of that step
        whose member set has the largest Jaccard overlap with the finally selected member set
        ``M``. Jaccard rather than raw overlap because raw overlap is maximised by whichever
        cluster is largest, which at high contamination is the field; Jaccard divides by the
        union and so penalises a cluster that swallows ``M`` inside something much bigger.

        **The tie-break is "no match", not "closest match".** A step whose clusters share no
        source at all with ``M`` contributes zero to every source rather than contributing its
        argmax, which would otherwise be an arbitrary cluster. There is no Jaccard floor above
        that: any floor is a tuned quantity, and requiring a non-empty intersection is the
        only threshold the definition forces.

        The denominator is the **total** number of sweep steps, identical to the one
        ``probability_times`` uses. That is what makes the two directly comparable and makes
        ``f_target <= f_any`` hold pointwise: the steps counted here are a subset of the steps
        counted there. It also means no comparison between them can be confounded by a
        different sweep grid.

        Returns ``(frequency, diagnostics)``. The diagnostics carry the number of steps that
        matched at all and the mean Jaccard and size ratio of the matched clusters -- the
        quantities that reveal the failure mode of the rule, which is a step where the only
        cluster is one giant structure that happens to contain ``M``.
        """
        sweep = np.asarray(sweep_labels, dtype=np.int64)
        if sweep.ndim != 2:
            raise ValueError("sweep_labels must be (n_sources, n_steps).")
        mask = np.asarray(member_mask, dtype=bool)
        n_rows, n_steps = sweep.shape
        if mask.size != n_rows:
            raise ValueError("member_mask length does not match sweep_labels rows.")
        freq = np.zeros(n_rows, dtype=float)
        n_target = int(mask.sum())
        info = {
            "n_steps": int(n_steps),
            "n_steps_matched": 0,
            "n_target_members": n_target,
            "mean_matched_jaccard": None,
            "mean_matched_size_ratio": None,
        }
        if n_steps == 0 or n_target == 0:
            return freq, info

        jaccards: list[float] = []
        ratios: list[float] = []
        for i in range(n_steps):
            col = sweep[:, i]
            clustered = col >= 0
            if not clustered.any():
                continue
            n_labels = int(col[clustered].max()) + 1
            sizes = np.bincount(col[clustered], minlength=n_labels).astype(float)
            hit = clustered & mask
            inter = np.bincount(col[hit], minlength=n_labels).astype(float) if hit.any() else None
            if inter is None or not inter.any():
                continue  # no cluster at this step overlaps the target at all
            union = sizes + n_target - inter
            jac = np.where(union > 0, inter / union, 0.0)
            best = int(np.argmax(jac))
            freq += col == best
            jaccards.append(float(jac[best]))
            ratios.append(float(sizes[best] / n_target))

        info["n_steps_matched"] = len(jaccards)
        if jaccards:
            info["mean_matched_jaccard"] = float(np.mean(jaccards))
            info["mean_matched_size_ratio"] = float(np.mean(ratios))
        return freq / float(n_steps), info

    @staticmethod
    def _warn_on_coincident_rows(X, columns, samples, min_samples, selection) -> dict:
        """Warn when duplicate rows can degrade ``cluster_persistence_``.

        .. danger::
           **Coincident rows silently destroy ``cluster_persistence_``, and the failure is graded
           rather than binary, so there is no exception to catch.**

           Traced through the hdbscan source. ``min_samples`` or more identical rows give a core
           distance of 0, hence a mutual reachability of 0, hence ``lambda = INFTY``
           (``_hdbscan_tree.pyx:112``). ``get_stability_scores`` (``:635``) then takes the
           ``np.isinf(max_lambda)`` branch and assigns **every** cluster a persistence of exactly
           1.0.

           Sub-threshold duplicates are the nastier case: they never trigger the infinity, they
           just inflate the *global* ``max(tree['lambda_val'])`` that every persistence is divided
           by. Measured on a synthetic frame, 39 near-coincident rows at ``min_samples=40`` moved
           ``max_lambda`` from 7.06 to 141.3 and pushed every persistence toward zero:

               n_dup=39: max_lambda=141.3  persistence=[0.0235 0.0144 0.7225 0.0004]
               n_dup=40: max_lambda=inf    persistence=[1. 1. 1. 1.]

           This matters because ``selection="max_persistence"`` is the default sweep-step rule, and
           it is an argmax over exactly that quantity. Ties at 1.0 make the argmax arbitrary;
           crushed values make it noise. EROTICA runs on cross-matched catalogues, where repeated
           coordinates are an ordinary artefact rather than a pathology.


        Returns the diagnostic dict so a caller can record it in provenance.
        """
        arr = np.asarray(X, dtype=float)
        finite = np.isfinite(arr).all(axis=1)
        clean = arr[finite]
        if clean.size == 0:
            return {"n_rows": 0, "n_distinct": 0, "max_multiplicity": 0}

        _, counts = np.unique(clean, axis=0, return_counts=True)
        max_mult = int(counts.max())
        n_in_dups = int(counts[counts > 1].sum())
        # min_samples defaults to min_cluster_size, so the smallest sweep step is the lowest
        # threshold at which the infinity can fire.
        #
        # The trigger is min_samples + 1 duplicates, NOT min_samples, and the +1 is load-bearing:
        # the core distance is the distance to the k-th nearest *other* point (every fit path uses
        # k=min_samples+1), so exactly min_samples coincident rows still leave a non-zero core
        # distance. Measured, and it is sharp:
        #     n_dup=40, min_samples=40 -> max_lambda 3.113, persistence [0.7645 0.5795]
        #     n_dup=41, min_samples=40 -> max_lambda inf,   persistence [1. 1. 1.]
        # An earlier version of this guard used >= min_samples and would have fired one row early.
        threshold = int(min_samples) if min_samples else int(min(samples)) if samples else 0

        info = {
            "n_rows": int(clean.shape[0]),
            "n_distinct": int(len(counts)),
            "max_multiplicity": max_mult,
            "n_rows_in_duplicate_groups": n_in_dups,
            "min_samples_threshold": threshold,
            "degenerate": bool(threshold and max_mult > threshold),
        }
        if info["degenerate"]:
            warnings.warn(
                f"{max_mult} identical rows in {list(columns)} at min_samples={threshold} "
                f"(the trigger is min_samples+1): "
                "mutual reachability is 0 for that group, so lambda is infinite and EVERY "
                "cluster_persistence_ will be exactly 1.0. "
                + (
                    "selection='max_persistence' is an argmax over that quantity and its result "
                    "is therefore arbitrary here -- pass selection='max_members' or deduplicate."
                    if selection == "max_persistence"
                    else "cluster_persistence_ is unusable on this input."
                ),
                RuntimeWarning,
                stacklevel=3,
            )
        elif max_mult > 1 and threshold and max_mult >= max(2, threshold // 4):
            warnings.warn(
                f"{n_in_dups} rows lie in duplicate groups in {list(columns)} "
                f"(largest {max_mult}, min_samples={threshold}). Below the threshold that makes "
                "lambda infinite, but duplicates still inflate the global max(lambda_val) that "
                "every cluster_persistence_ is divided by, which biases the sweep-step choice.",
                RuntimeWarning,
                stacklevel=3,
            )
        return info

    @staticmethod
    def _build_pseudoprobability(labels_storage: list[list[int]]) -> np.ndarray:
        return np.array(
            [
                np.count_nonzero(np.asarray(labels, dtype=int) != -1) / len(labels)
                if labels
                else 0.0
                for labels in labels_storage
            ],
            dtype=float,
        )

    @staticmethod
    def _desired_tree_branch_size(tree) -> int:
        max_lambda_val_row = tree["lambda_val"].idxmax()
        desired_parent = tree.at[max_lambda_val_row, "parent"]
        return int(len(tree[tree["parent"] == desired_parent]))

    @staticmethod
    def _select_pseudoprobability_result(results: list[dict], selection: str) -> dict:
        """Choose which sweep step to keep.

        .. danger::
           **``"max_members"`` and ``"max_lambda"`` are both defective above ~0.8
           contamination, measured.** They are retained only to reproduce results published
           before 2026-08-04.

           ``"max_members"`` is the argmax of ``desired_len``, which is a count of
           **condensed-tree rows** — the same quantity behind issue #7. At high
           contamination it selects a ``min_cluster_size`` whose densest branch lies *inside
           the field*, and no label chosen from that branch can be right. Measured selected
           purity 0.638 against an oracle of 0.940.

           ``"max_lambda"`` is not the alternative: it chose ``mcs_range.start`` in **12 of
           12** cells, i.e. it ignores the sweep entirely and returns whatever the smallest
           ``min_cluster_size`` produced.

           ``"max_persistence"`` (the default) scores each step by
           ``cluster_persistence_`` **of the cluster the selector actually returns** —
           HDBSCAN's own stability measure for that specific cluster, rather than a proxy
           for how big something is. The value was already being collected here and thrown
           away.

        .. important::
           **The default is justified on PRINCIPLE, not on measured superiority, and an
           earlier version of this docstring overstated it.**

           It was adopted on 54 cells from one generator with no held-out block. Re-measured
           on 108 cells with 3 of 6 realisations held out, the paired per-cell difference
           against ``"max_members"`` is within one standard error of zero on **every**
           metric, and the sign flips between blocks:

               metric      block       delta (persistence - members)   wins
               ROC-AUC     train              -0.0144 +- 0.0101       20/54
               ROC-AUC     HELD-OUT           +0.0103 +- 0.0170       22/54
               purity      HELD-OUT           +0.0112 +- 0.0182       16/54
               recall      HELD-OUT           +0.0026 +- 0.0189       16/54
               sigma_pm    HELD-OUT           -0.0206 +- 0.0360           -

           So the two rules are **not distinguishable by performance** on this generator.

           It remains the default for a reason that does not depend on that: ``"max_members"``
           is the argmax of ``desired_len``, a count of **condensed-tree rows** matched against
           flat-cluster point counts — the units mismatch behind issue #7. Ranking sweep steps
           by a quantity that is known to be the wrong kind of number is indefensible even when
           it happens to score the same. ``cluster_persistence_`` is at least the quantity it
           claims to be.

           ⚠ But see the coincident-row guard: ``cluster_persistence_`` collapses to exactly 1.0
           for every cluster when the input contains more than ``min_samples`` duplicate rows,
           which makes this argmax arbitrary. Measured zero duplicates in the NGC 6383
           catalogues at all four radii, so it is not currently firing.
        """
        if selection == "max_persistence":
            return max(
                results,
                key=lambda item: (item["selected_persistence"], item["lambda_value"]),
            )
        if selection == "max_members":
            return max(results, key=lambda item: (item["desired_len"], item["lambda_value"]))
        if selection == "max_lambda":
            return max(results, key=lambda item: (item["lambda_value"], item["desired_len"]))
        raise ValueError("selection must be 'max_persistence', 'max_members' or 'max_lambda'.")

    @staticmethod
    def _cluster_label_for_size(labels: np.ndarray, desired_len: int) -> int:
        """Match a flat cluster by size. **Kept only to reproduce pre-2026-08-03 results.**

        .. danger::
           **This is the defect measured in issue #7, and it selects the field.** ``desired_len``
           comes from :meth:`_desired_tree_branch_size`, which counts **rows of the condensed
           tree** whose parent is a given node — its immediate children, sub-clusters or falling
           points. That is not the number of points in the flat cluster. The two coincide only by
           accident: **23 matches out of 83** across the benchmark.

           On the other 60 the fallback below returns the **largest non-noise cluster**, which at
           contamination 0.8–0.95 *is the field*. Measured consequence: HDBSCAN isolates the
           cluster at ≥0.8 purity in 96% of cells, and this selector then returns purity
           0.394 ± 0.052, dragging AUC to 0.70–0.80 against pyUPMASK's 0.998.

           Use :meth:`_cluster_label_from_tree`. This is retained only because ``select_cluster``
           results produced before 2026-08-03 used it.
        """
        clusters, counts = np.unique(labels, return_counts=True)
        matching = clusters[counts == desired_len]
        if matching.size:
            return int(matching[0])
        non_noise = clusters[clusters != -1]
        if non_noise.size == 0:
            return -1
        return int(non_noise[np.argmax(counts[clusters != -1])])

    @staticmethod
    def _cluster_label_from_tree(tree, labels: np.ndarray, n_samples: int) -> int:
        """Label of the flat cluster that the densest branch of the condensed tree resolves to.

        The right question is *"which cluster does this tree node become?"*, and it is answered by
        descending the node to its leaves and reading their label — not by hoping a row count
        equals a cluster size (see :meth:`_cluster_label_for_size`).

        The starting node is the parent of the maximum-``lambda_val`` row: the branch whose points
        persist to the highest density. Descend it to leaf children (``child < n_samples`` are data
        points, anything larger is another tree node), then take the modal non-noise label.

        **The fallback is noise, deliberately.** If the branch resolves to no labelled points the
        answer is "no cluster selected", not "the largest cluster" — on a contaminated field the
        largest cluster *is* the contamination, which is precisely how the previous implementation
        failed.
        """
        if tree is None or len(tree) == 0:
            return -1
        labels = np.asarray(labels, dtype=int)
        parent = int(tree.at[tree["lambda_val"].idxmax(), "parent"])

        parents = tree["parent"].to_numpy(dtype=int)
        children = tree["child"].to_numpy(dtype=int)
        frontier, seen, leaves = [parent], set(), []
        while frontier:
            node = frontier.pop()
            if node in seen:  # condensed trees are acyclic; a malformed one would otherwise hang
                continue
            seen.add(node)
            for child in children[parents == node]:
                (leaves if child < n_samples else frontier).append(int(child))

        if not leaves:
            return -1
        member_labels = labels[np.asarray(leaves, dtype=int)]
        member_labels = member_labels[member_labels != -1]
        if member_labels.size == 0:
            return -1
        values, counts = np.unique(member_labels, return_counts=True)
        return int(values[np.argmax(counts)])

    def _annotate_results(self) -> None:
        if self.clusterer is None:
            raise RuntimeError("No clustering model available.")
        self.data["cluster"] = self.clusterer.labels_
        self.data["probability_hdbscan"] = self.clusterer.probabilities_
        self.combined_data = combine_datasets(self.data, self.bad_data)


__all__ = ["Clustering", "HDBSCANEstimator"]
