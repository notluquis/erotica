import numpy as np
from tqdm import tqdm


def _row_discrepancies(left, right):
    """Yield the columns on which two single-row slices disagree.

    Shared by both comparison branches so the *rule* for "these differ" cannot
    drift between them; each branch keeps its own print format.

    Yields
    ------
    tuple
        ``(column, left_value, right_value)`` for every shared column whose
        values differ.
    """
    for col in left.colnames:
        if col not in right.colnames:
            continue
        val_left, val_right = left[col], right[col]
        if np.issubdtype(val_left.dtype, np.number):
            if not np.allclose(val_left, val_right, equal_nan=True):
                yield col, val_left, val_right
        elif np.any(val_left != val_right):
            yield col, val_left, val_right


def compare_datasets(*datasets, verbose: bool = True):
    """
    Compare one, two, or four QTable datasets to identify overlaps,
    missing entries, and value discrepancies by `source_id`.

    Parameters
    ----------
    *datasets : QTable
        Exactly one, two, or four tables; any other count raises. Each must have
        a ``source_id`` column, which is the only key used to match rows.

        With **two**, they are compared directly as ``(data1, data2)``. With
        **four**, they are read positionally as
        ``(good_data, bad_data, good_data_test, bad_data_test)``: the first two
        are checked for mutual overlap (they are meant to be disjoint fidelity
        splits, so any overlap is reported as an error), then each is compared
        against its ``_test`` counterpart. With **one**, nothing is compared.
    verbose : bool, default True
        Whether to also print the report to stdout. The printed text is
        unchanged from the print-only version of this function, so existing
        interactive use and any stdout-scraping caller keep working; set
        ``False`` to use the return value alone.

    Returns
    -------
    dict
        A record of what was compared. Always carries ``"n_datasets"`` and
        ``"comparisons"``; the latter is keyed by the comparison performed and
        is **empty for a single dataset**, because nothing was compared.

        Each entry of ``"comparisons"`` has the same shape whichever arity
        produced it::

            {
                "overlap": [source_id, ...],  # in both, sorted
                "missing_in_right": [source_id, ...],  # in left only
                "missing_in_left": [source_id, ...],  # in right only
                "discrepancies": [
                    {"source_id": ..., "column": ..., "left": ..., "right": ...},
                    ...,
                ],
            }

        "left"/"right" are ``(data1, data2)`` for two datasets and
        ``(program, test)`` for four. With four datasets the result also carries
        ``"good_bad_overlap"``, the sorted ``source_id`` values found in
        **both** ``good_data`` and ``bad_data`` -- empty is the healthy case,
        since those are meant to be disjoint.

        Counts are deliberately **not** stored: they are ``len()`` of the lists,
        and a stored count is a second source of truth that can disagree with
        the ids it claims to count.

    Raises
    ------
    ValueError
        If the number of datasets is not 1, 2, or 4.

    Notes
    -----
    Comparison is **row-by-row inside a Python loop** with a boolean mask per
    ``source_id``, wrapped in :mod:`tqdm`, so cost grows as the product of table
    length and overlap size. It is an interactive diagnostic, not something to
    put in a pipeline over full Gaia catalogues.

    Only columns present in *both* tables are compared; a column in one and not
    the other is skipped silently rather than reported as a difference. Numeric
    columns are compared with :func:`numpy.allclose` at its default tolerances
    and with ``equal_nan=True``, so two NaNs count as equal and differences
    below ``rtol=1e-5`` are not reported. Non-numeric columns are compared for
    exact inequality.

    Units are honoured, because :func:`numpy.allclose` converts Astropy
    Quantities: the same physical value stored as arcmin in one table and
    degrees in the other compares **equal**, and the same *number* under
    different units compares **different**. Columns whose units are not
    convertible at all (mass against angle, say) raise
    :class:`~astropy.units.UnitConversionError` out of this function rather than
    being reported as a discrepancy, so a schema mismatch aborts the comparison
    instead of appearing in the output.

    The values stored under ``"left"`` and ``"right"`` are the **table slices**,
    not scalars -- length-1 columns carrying their units, exactly what was
    printed before. Take ``[0]`` to get the value.
    """
    if len(datasets) not in {1, 2, 4}:
        raise ValueError(f"Only 1, 2, or 4 datasets supported, got {len(datasets)}.")

    result: dict = {"n_datasets": len(datasets), "comparisons": {}}

    if len(datasets) == 1:
        if verbose:
            print("Only one dataset provided; no comparison can be performed.")
        return result

    if len(datasets) == 2:
        # Compare two datasets by source_id
        data1, data2 = datasets
        ids1, ids2 = set(data1["source_id"]), set(data2["source_id"])
        overlap = ids1 & ids2
        missing_in_data2 = ids1 - ids2
        missing_in_data1 = ids2 - ids1

        if verbose:
            print(f"Overlap between datasets: {len(overlap)}")
            print(f"Missing in data2: {len(missing_in_data2)}")
            print(f"Missing in data1: {len(missing_in_data1)}")

        discrepancies: list[dict] = []
        for source_id in tqdm(overlap, desc="Checking discrepancies"):
            row1 = data1[data1["source_id"] == source_id]
            row2 = data2[data2["source_id"] == source_id]
            for col, val1, val2 in _row_discrepancies(row1, row2):
                discrepancies.append(
                    {"source_id": source_id, "column": col, "left": val1, "right": val2}
                )
                if verbose:
                    print(f"Discrepancy for source_id {source_id} in column {col}:")
                    print(f"  Data1: {val1}")
                    print(f"  Data2: {val2}")

        result["comparisons"]["data1_vs_data2"] = {
            "overlap": sorted(overlap),
            "missing_in_right": sorted(missing_in_data2),
            "missing_in_left": sorted(missing_in_data1),
            "discrepancies": discrepancies,
        }

    else:
        # Four datasets: (good_data, bad_data, good_data_test, bad_data_test)
        good_data, bad_data, good_data_test, bad_data_test = datasets

        # Check overlap between good and bad in program
        overlap = set(good_data["source_id"]) & set(bad_data["source_id"])
        result["good_bad_overlap"] = sorted(overlap)
        if verbose:
            if overlap:
                print(f"Error: Overlap detected between good_data and bad_data: {len(overlap)}")
            else:
                print("No overlap between good_data and bad_data.")

        # Compare program vs test for both good and bad sets
        for label, prog, test in [
            ("good", good_data, good_data_test),
            ("bad", bad_data, bad_data_test),
        ]:
            ids_prog, ids_test = set(prog["source_id"]), set(test["source_id"])
            common = ids_prog & ids_test
            if verbose:
                print(f"Missing in {label}_test: {len(ids_prog - ids_test)}")
                print(f"Missing in {label}_prog: {len(ids_test - ids_prog)}")
                print(f"Common source_ids in {label}_data: {len(common)}")

            discrepancies = []
            for source_id in tqdm(common, desc=f"Checking {label}_data discrepancies"):
                row_prog = prog[prog["source_id"] == source_id]
                row_test = test[test["source_id"] == source_id]
                for col, v_prog, v_test in _row_discrepancies(row_prog, row_test):
                    discrepancies.append(
                        {"source_id": source_id, "column": col, "left": v_prog, "right": v_test}
                    )
                    if verbose:
                        print(f"Discrepancy in {label}_data for source_id {source_id} col {col}:")
                        print(f"  Program: {v_prog}")
                        print(f"  Test:    {v_test}")

            result["comparisons"][label] = {
                "overlap": sorted(common),
                "missing_in_right": sorted(ids_prog - ids_test),
                "missing_in_left": sorted(ids_test - ids_prog),
                "discrepancies": discrepancies,
            }

    return result
