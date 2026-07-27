"""Summary helpers for EROTICA clustering outputs."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from astropy.table import QTable, vstack
from scipy.stats import iqr


def combine_datasets(data: QTable, bad_data: QTable | None) -> QTable:
    if bad_data is None:
        return data.copy()
    combo = vstack([data, bad_data])
    combo["cluster"].fill_value = -1
    combo["probability_hdbscan"].fill_value = 0.0
    return combo


def clustering_statistics(table: QTable, *, include_outliers: bool = False) -> dict[str, float]:
    labels = np.array(table["cluster"].data, dtype=int)
    total = labels.size
    outliers = int((labels == -1).sum())
    working = labels if include_outliers else labels[labels != -1]
    if working.size == 0:
        return {
            "total_stars": total,
            "outliers": outliers,
            "clusters": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    unique, counts = np.unique(working, return_counts=True)
    return {
        "total_stars": total,
        "outliers": outliers,
        "clusters": int(len(unique)),
        "mean": float(np.mean(counts)),
        "median": float(np.median(counts)),
        "std": float(np.std(counts)),
        "min": float(np.min(counts)),
        "max": float(np.max(counts)),
    }


def build_cluster_summary(
    table: QTable,
    *,
    pm_columns: Sequence[str] = ("pmra", "pmdec"),
    include_noise: bool = True,
    persistence_array: np.ndarray | None = None,
) -> pd.DataFrame:
    df = table.to_pandas()
    if "cluster" not in df or "probability_hdbscan" not in df:
        raise KeyError("Required columns 'cluster' and 'probability_hdbscan' were not found.")

    if not include_noise:
        df = df[df["cluster"] != -1]

    total = len(df)
    if total == 0:
        return pd.DataFrame(
            columns=[
                "cluster",
                "count",
                "fraction",
                "persistence",
                "mean_prob",
                "median_prob",
                "min_prob",
                "max_prob",
                "iqr_prob",
                *(f"centroid_{c}" for c in pm_columns),
                "mean_dist2centroid",
                "std_dist2centroid",
                *(f"{c}_range" for c in pm_columns),
            ]
        )

    persistence_array = np.asarray(persistence_array, dtype=float)
    have_pm = all(col in df.columns for col in pm_columns)

    records: list[dict] = []
    for label, group in df.groupby("cluster", sort=True):
        probs = group["probability_hdbscan"].astype(float).to_numpy()
        probs = probs[np.isfinite(probs)]
        if probs.size:
            p_mean = float(np.nanmean(probs))
            p_med = float(np.nanmedian(probs))
            p_min = float(np.nanmin(probs))
            p_max = float(np.nanmax(probs))
            p_iqr = float(iqr(probs, nan_policy="omit"))
        else:
            p_mean = p_med = p_min = p_max = p_iqr = np.nan

        persistence = np.nan
        if (
            persistence_array.size
            and isinstance(label, (int, np.integer))
            and 0 <= int(label) < persistence_array.size
        ):
            persistence = float(persistence_array[int(label)])

        record = {
            "cluster": label,
            "count": int(len(group)),
            "fraction": float(len(group) / total) if total else 0.0,
            "persistence": persistence,
            "mean_prob": p_mean,
            "median_prob": p_med,
            "min_prob": p_min,
            "max_prob": p_max,
            "iqr_prob": p_iqr,
        }

        if have_pm:
            pts = group[list(pm_columns)].astype(float).to_numpy()
            finite_rows = np.all(np.isfinite(pts), axis=1)
            if np.any(finite_rows):
                pts_f = pts[finite_rows]
                centroid = np.nanmean(pts_f, axis=0)
                dists = np.linalg.norm(pts_f - centroid, axis=1)
                record.update(
                    {
                        **{
                            f"centroid_{pm_columns[i]}": float(centroid[i])
                            for i in range(len(pm_columns))
                        },
                        "mean_dist2centroid": float(np.nanmean(dists)),
                        "std_dist2centroid": float(np.nanstd(dists)),
                        **{
                            f"{pm_columns[i]}_range": float(
                                np.nanmax(pts_f[:, i]) - np.nanmin(pts_f[:, i])
                            )
                            for i in range(len(pm_columns))
                        },
                    }
                )
            else:
                record.update(
                    {
                        **{f"centroid_{c}": np.nan for c in pm_columns},
                        "mean_dist2centroid": np.nan,
                        "std_dist2centroid": np.nan,
                        **{f"{c}_range": np.nan for c in pm_columns},
                    }
                )
        else:
            record.update(
                {
                    **{f"centroid_{c}": np.nan for c in pm_columns},
                    "mean_dist2centroid": np.nan,
                    "std_dist2centroid": np.nan,
                    **{f"{c}_range": np.nan for c in pm_columns},
                }
            )

        records.append(record)

    columns = [
        "cluster",
        "count",
        "fraction",
        "persistence",
        "mean_prob",
        "median_prob",
        "min_prob",
        "max_prob",
        "iqr_prob",
        *(f"centroid_{c}" for c in pm_columns),
        "mean_dist2centroid",
        "std_dist2centroid",
        *(f"{c}_range" for c in pm_columns),
    ]

    return (
        pd.DataFrame.from_records(records, columns=columns)
        .sort_values("cluster")
        .reset_index(drop=True)
    )


__all__ = ["combine_datasets", "clustering_statistics", "build_cluster_summary"]
