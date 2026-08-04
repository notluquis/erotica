import numpy as np
from tqdm import tqdm


def compare_datasets(*datasets):
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

    Returns
    -------
    None
        This function **reports rather than returns**: every result is written
        to stdout with :func:`print`, and there is no return value to assert on.
        Counts of overlapping and missing ``source_id`` values, and every
        differing cell, are printed as they are found. Capture stdout if the
        output is needed programmatically.

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
    """
    if len(datasets) not in {1, 2, 4}:
        raise ValueError(f"Only 1, 2, or 4 datasets supported, got {len(datasets)}.")

    if len(datasets) == 1:
        print("Only one dataset provided; no comparison can be performed.")
        return

    if len(datasets) == 2:
        # Compare two datasets by source_id
        data1, data2 = datasets
        ids1, ids2 = set(data1["source_id"]), set(data2["source_id"])
        overlap = ids1 & ids2
        missing_in_data2 = ids1 - ids2
        missing_in_data1 = ids2 - ids1

        print(f"Overlap between datasets: {len(overlap)}")
        print(f"Missing in data2: {len(missing_in_data2)}")
        print(f"Missing in data1: {len(missing_in_data1)}")

        for source_id in tqdm(overlap, desc="Checking discrepancies"):
            row1 = data1[data1["source_id"] == source_id]
            row2 = data2[data2["source_id"] == source_id]
            for col in data1.colnames:
                if col in data2.colnames:
                    val1, val2 = row1[col], row2[col]
                    if np.issubdtype(val1.dtype, np.number):
                        if not np.allclose(val1, val2, equal_nan=True):
                            print(f"Discrepancy for source_id {source_id} in column {col}:")
                            print(f"  Data1: {val1}")
                            print(f"  Data2: {val2}")
                    elif np.any(val1 != val2):
                        print(f"Discrepancy for source_id {source_id} in column {col}:")
                        print(f"  Data1: {val1}")
                        print(f"  Data2: {val2}")

    else:
        # Four datasets: (good_data, bad_data, good_data_test, bad_data_test)
        good_data, bad_data, good_data_test, bad_data_test = datasets

        # Check overlap between good and bad in program
        overlap = set(good_data["source_id"]) & set(bad_data["source_id"])
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
            print(f"Missing in {label}_test: {len(ids_prog - ids_test)}")
            print(f"Missing in {label}_prog: {len(ids_test - ids_prog)}")
            common = ids_prog & ids_test
            print(f"Common source_ids in {label}_data: {len(common)}")
            for source_id in tqdm(common, desc=f"Checking {label}_data discrepancies"):
                row_prog = prog[prog["source_id"] == source_id]
                row_test = test[test["source_id"] == source_id]
                for col in prog.colnames:
                    if col in test.colnames:
                        v_prog, v_test = row_prog[col], row_test[col]
                        if np.issubdtype(v_prog.dtype, np.number):
                            if not np.allclose(v_prog, v_test, equal_nan=True):
                                print(
                                    f"Discrepancy in {label}_data for source_id {source_id} col {col}:"
                                )
                                print(f"  Program: {v_prog}")
                                print(f"  Test:    {v_test}")
                        elif np.any(v_prog != v_test):
                            print(
                                f"Discrepancy in {label}_data for source_id {source_id} col {col}:"
                            )
                            print(f"  Program: {v_prog}")
                            print(f"  Test:    {v_test}")
