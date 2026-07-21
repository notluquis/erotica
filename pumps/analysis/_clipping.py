"""Sigma-clipping utilities for PUMPS cluster analysis."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from astropy.stats import biweight_location, biweight_scale, mad_std, sigma_clip
from astropy.table import QTable


def sigma_clip_parallax(
    table: QTable,
    *,
    cluster: int,
    sigma: float,
    use_biweight: bool,
    in_place: bool,
    mark_label: int,
    print_results: bool,
    return_mask: bool,
    preselector_mask: Sequence[bool] | None,
):
    """Apply robust sigma clipping on the parallax column of ``table``."""
    cluster_col = table["cluster"]
    parallax_col = table["parallax"]

    base_mask = cluster_col == cluster
    if not np.any(base_mask):
        raise ValueError(f"Cluster {cluster} has zero rows in the provided table.")

    if preselector_mask is not None:
        preselector_mask = np.asarray(preselector_mask, dtype=bool)
        if preselector_mask.shape != base_mask.shape:
            raise ValueError("preselector_mask must have the same length as the table.")
        selection_mask = base_mask & preselector_mask
    else:
        selection_mask = base_mask

    parallax_values = parallax_col[selection_mask]
    if getattr(parallax_values, "size", len(parallax_values)) == 0:
        raise ValueError("No rows selected for sigma clipping (mask is empty).")
    n_finite = np.count_nonzero(np.isfinite(parallax_values))
    if n_finite < 3:
        raise ValueError(f"Not enough finite parallax values for sigma clipping: {n_finite}")

    cenfunc = biweight_location
    stdfunc = biweight_scale if use_biweight else mad_std

    _, lower, upper = sigma_clip(
        parallax_values,
        sigma=sigma,
        cenfunc=cenfunc,
        stdfunc=stdfunc,
        return_bounds=True,
    )

    if not (np.isfinite(lower) and np.isfinite(upper)):
        raise RuntimeError(f"sigma_clip returned non-finite bounds: lower={lower}, upper={upper}")

    within_bounds = (parallax_col >= lower) & (parallax_col <= upper)
    if preselector_mask is not None:
        final_keep = base_mask & preselector_mask & within_bounds
    else:
        final_keep = base_mask & within_bounds
    to_noise = base_mask & ~final_keep

    if in_place:
        if mark_label < 0 and not np.issubdtype(cluster_col.dtype, np.signedinteger):
            raise TypeError(
                f"'cluster' dtype must be signed int to assign {mark_label}, got {cluster_col.dtype}"
            )
        table["cluster"][to_noise] = mark_label
        if print_results:
            print(f"{int(base_mask.sum())} before sigma clipping")
            print(f"{int(final_keep.sum())} after sigma clipping")
        if return_mask:
            return lower, upper, final_keep, to_noise
        return lower, upper

    data_copy = table.copy(copy_data=True)
    data_copy["cluster"][to_noise] = mark_label
    if print_results:
        print(f"{int(base_mask.sum())} before sigma clipping")
        print(f"{int(final_keep.sum())} after sigma clipping")
    if return_mask:
        return lower, upper, data_copy, final_keep, to_noise
    return lower, upper, data_copy


__all__ = ["sigma_clip_parallax"]
