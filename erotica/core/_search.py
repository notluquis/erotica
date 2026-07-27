"""Hyper-parameter search helpers for EROTICA clustering."""

from __future__ import annotations

import datetime
from typing import Any, Iterable

import numpy as np
import optuna
from hdbscan.validity import validity_index
from sklearn.model_selection import GridSearchCV

from ._constants import DEFAULT_PARAM_GRID
from ._estimator import FullSplit, HDBSCANEstimator


def run_grid_search(
    X,
    *,
    persistence_threshold: float,
    param_grid: dict[str, list] | None = None,
    grid_kwargs: dict | None = None,
    hdbscan_kwargs: dict | None = None,
):
    """Execute a :class:`GridSearchCV` over :class:`HDBSCANEstimator`."""
    estimator = HDBSCANEstimator(
        persistence_threshold=persistence_threshold,
        **(hdbscan_kwargs or {}),
    )
    search = GridSearchCV(
        estimator,
        param_grid or DEFAULT_PARAM_GRID,
        scoring=lambda est, _: est.score(X),
        cv=FullSplit(),
        n_jobs=-1,
        verbose=1,
        error_score=float("-inf"),
        **(grid_kwargs or {}),
    )
    search.fit(X)
    return {
        "clusterer": search.best_estimator_.model_,
        "cv_results": search.cv_results_,
        "best_params": search.best_params_,
        "best_score": search.best_score_,
    }


def run_optuna_search(
    X,
    *,
    persistence_threshold: float,
    search_space: dict[str, dict] | None,
    n_trials: int,
    n_jobs: int = 1,
    sampler_name: str,
    sampler_kwargs: dict | None,
    score_methods: Iterable[str],
    hdbscan_kwargs: dict,
    storage_url: str | None,
    study_name: str | None,
):
    """Execute an Optuna study and return fitted artifacts."""
    search_space = search_space or {}
    sampler_kwargs = sampler_kwargs or {}
    methods = list(score_methods)

    sampler = _build_sampler(sampler_name, search_space, sampler_kwargs, n_jobs=n_jobs)
    study = _create_study(methods, sampler, storage_url, study_name)

    def objective(trial):
        params = _suggest_hyperparameters(trial, search_space)
        try:
            estimator = HDBSCANEstimator(
                min_cluster_size=params.get("min_cluster_size"),
                min_samples=params.get("min_samples"),
                persistence_threshold=persistence_threshold,
                **hdbscan_kwargs,
            )
            estimator.fit(X)
        except Exception as exc:
            print(f"  [ERROR] al entrenar HDBSCAN con params {params}: {exc!r}")
            raise

        results: list[float] = []
        for method in methods:
            if method == "relative_validity":
                results.append(_score_relative_validity(estimator, X, trial))
            elif method == "cluster_persistence":
                results.append(_score_cluster_persistence(estimator, trial))
            elif method == "dbcv":
                results.append(_score_dbcv(estimator, X, persistence_threshold))
            else:
                raise ValueError(f"Unknown metric '{method}'")

        if len(results) != len(methods):
            raise RuntimeError(
                f"Trial {trial.number} devolvió {len(results)} valores, pero esperábamos {len(methods)}."
            )
        return tuple(results) if len(results) > 1 else results[0]

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    if len(methods) > 1:
        pareto_trials = study.best_trials
        best_trial = pareto_trials[0]
    else:
        pareto_trials = []
        best_trial = study.best_trial

    best_params = best_trial.params
    final_params = {**hdbscan_kwargs, **best_params}
    estimator = HDBSCANEstimator(
        persistence_threshold=persistence_threshold,
        **final_params,
    ).fit(X)

    best_score = getattr(best_trial, "values", None)
    if best_score is None:
        best_score = best_trial.value

    return {
        "clusterer": estimator.model_,
        "best_params": best_params,
        "best_score": best_score,
        "study": study,
        "pareto": pareto_trials,
    }


def _build_sampler(name: str, search_space: dict[str, dict], sampler_kwargs: dict, *, n_jobs: int = 1):
    import optuna.samplers as samplers_module

    available = [
        candidate
        for candidate, obj in vars(samplers_module).items()
        if isinstance(obj, type) and candidate.endswith("Sampler")
    ]

    try:
        sampler_cls = getattr(samplers_module, name)
    except AttributeError as exc:
        raise ValueError(
            f"Unknown sampler '{name}'. Available samplers are: {', '.join(available)}"
        ) from exc

    if name == "GridSampler":
        grid_space: dict[str, list] = {}
        for param, info in search_space.items():
            if "choices" in info:
                grid_space[param] = info["choices"]
            elif info.get("type") == "int":
                low, high = info["low"], info["high"]
                grid_space[param] = list(range(low, high + 1))
            elif info.get("type") == "float":
                low, high = info["low"], info["high"]
                grid_space[param] = list(np.linspace(low, high, num=10))
            else:
                raise ValueError(
                    f"Cannot infer grid for parameter '{param}' of type '{info.get('type')}'"
                )
        sampler_kwargs.setdefault("search_space", grid_space)

    if name == "GPSampler":
        from optuna.samplers import RandomSampler

        seed = sampler_kwargs.get("seed")
        sampler_kwargs.setdefault("independent_sampler", RandomSampler(seed=seed))

    # constant_liar evita que workers paralelos sugieran los mismos params.
    # Soportado en TPESampler siempre, y en GPSampler desde v4.8.0.
    if n_jobs != 1 and name in {"TPESampler", "GPSampler"}:
        sampler_kwargs.setdefault("constant_liar", True)

    return sampler_cls(**sampler_kwargs)


def _create_study(methods, sampler, storage_url, study_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = study_name or f"study_{timestamp}"

    directions = None
    if len(methods) > 1:
        directions = ["maximize"] * len(methods)

    if storage_url:
        if directions is not None:
            return optuna.create_study(
                study_name=name,
                storage=storage_url,
                load_if_exists=True,
                directions=directions,
                sampler=sampler,
            )
        return optuna.create_study(
            study_name=name,
            storage=storage_url,
            load_if_exists=True,
            direction="maximize",
            sampler=sampler,
        )

    if directions is not None:
        return optuna.create_study(
            study_name=name,
            directions=directions,
            sampler=sampler,
            load_if_exists=False,
        )
    return optuna.create_study(
        study_name=name,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )


def _suggest_hyperparameters(trial, search_space):
    params = {}
    for name, space in (search_space or {}).items():
        if "choices" in space:
            params[name] = trial.suggest_categorical(name, space["choices"])
        elif space.get("type") == "int":
            params[name] = trial.suggest_int(
                name,
                space["low"],
                space["high"],
                log=space.get("log", False),
            )
        elif space.get("type") == "float":
            params[name] = trial.suggest_float(
                name,
                space["low"],
                space["high"],
                log=space.get("log", False),
            )
        else:
            raise ValueError(f"Unknown parameter specification for '{name}': {space}")
    return params


def _score_relative_validity(estimator, X, trial):
    try:
        value = estimator.score(X)
        if np.isnan(value):
            print(f"  [RV] Trial #{trial.number}: score() devolvió NaN. Lo convierto a -inf.")
            return float("-inf")
        return float(value)
    except Exception as exc:
        print(f"  [RV] Trial #{trial.number}: excepción en score(): {exc!r}. Devolviendo -inf.")
        return float("-inf")


def _score_cluster_persistence(estimator, trial):
    try:
        persistence = getattr(estimator.model_, "cluster_persistence_", [])
        if len(persistence) == 0:
            print("  [CP] No hay clusters (lista vacía). max persistence → 0.0")
            max_value = 0.0
        else:
            max_value = float(max(persistence))
            print(f"  [CP] Persistencia máxima entre clusters: {round(max_value, 6)}")
        if np.isnan(max_value):
            print("  [CP] max_p es NaN. Lo convierto a -inf.")
            return float("-inf")
        return max_value
    except Exception as exc:
        print(
            f"  [CP] Trial #{trial.number}: excepción al obtener cluster_persistence_: {exc!r}. Devolviendo -inf."
        )
        return float("-inf")


def _score_dbcv(estimator, X, persistence_threshold):
    labels = estimator.model_.labels_.copy()
    persistence = estimator.model_.cluster_persistence_
    if persistence_threshold > 0.0:
        for cid, value in enumerate(persistence):
            if value < persistence_threshold:
                labels[labels == cid] = -1

    unique_labels = set(labels)
    unique_labels.discard(-1)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        if n_clusters == 0:
            print("    → DBCV no se puede calcular: 0 clusters (todo ruido).")
        else:
            only_one = next(iter(unique_labels))
            print(f"    → DBCV no se puede calcular: solo 1 cluster ({only_one}).")
        return float("-inf")

    X_array = np.asarray(X)
    try:
        score = validity_index(X_array, labels, metric="euclidean")
    except Exception as exc:
        print(f"    → Excepción en validity_index: {exc!r}. Uso -inf.")
        return float("-inf")

    if np.isnan(score):
        print("    → validity_index devolvió NaN. Uso -inf.")
        return float("-inf")
    return float(score)


__all__ = ["run_grid_search", "run_optuna_search"]
