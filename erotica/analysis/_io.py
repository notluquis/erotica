"""Input/output helpers for EROTICA cluster analysis."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import dill
import pandas as pd
from astropy.table import QTable, Table

from erotica.core.clustering import Clustering
from erotica.io._helpers import apply_unit_corrections, handle_masked_columns
from erotica.io.loader import DataLoader


@contextmanager
def optuna_safe_unpickle():
    """Context manager that prevents Optuna from reopening SQLite engines during ``dill.load``."""
    try:
        from optuna.storages._rdb.storage import RDBStorage  # type: ignore
    except Exception:
        # No Optuna installation or different layout
        yield
        return

    original_setstate = getattr(RDBStorage, "__setstate__", None)
    if original_setstate is None:
        yield
        return

    def _patched(self, state):  # pragma: no cover - behaviour copied from legacy code
        for key, value in state.items():
            setattr(self, key, value)
        for attr in ("engine", "_engine", "scoped_session"):
            if hasattr(self, attr):
                setattr(self, attr, None)

    RDBStorage.__setstate__ = _patched
    try:
        yield
    finally:
        RDBStorage.__setstate__ = original_setstate


def load_dataset(
    file_obj: Any,
    *,
    dataloader_kwargs: dict | None = None,
    dill_cache: bool = True,
    output_dir: str | None = None,
    verbose: int = logging.INFO,
    debug_mode: bool = False,
):
    """Load input data for :class:`ClusterAnalyzer` from a path or in-memory object."""
    dataloader_kwargs = dataloader_kwargs or {}

    if isinstance(file_obj, str):
        return _load_from_path(
            file_obj,
            dataloader_kwargs=dataloader_kwargs,
            dill_cache=dill_cache,
            output_dir=output_dir,
            verbose=verbose,
            debug_mode=debug_mode,
        )

    if isinstance(file_obj, pd.DataFrame):
        table = QTable.from_pandas(file_obj)
        return _load_from_table(
            table,
            source="<DataFrame>",
            output_dir=output_dir,
            verbose=verbose,
            debug_mode=debug_mode,
        )

    if isinstance(file_obj, (QTable, Table)):
        table = file_obj if isinstance(file_obj, QTable) else QTable(file_obj)
        return _load_from_table(
            table,
            source="<Table>",
            output_dir=output_dir,
            verbose=verbose,
            debug_mode=debug_mode,
        )

    raise TypeError(
        "file_obj must be a path, pandas.DataFrame, astropy QTable/Table, or compatible object."
    )


def _load_from_path(
    path_like: str,
    *,
    dataloader_kwargs: dict,
    dill_cache: bool,
    output_dir: str | None,
    verbose: int,
    debug_mode: bool,
):
    path = Path(path_like).expanduser().resolve()
    base_dir = path.parent
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else base_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = None
    clustering = None
    data: QTable | None = None

    if path.suffix.lower() == ".dill" and path.exists():
        payload = _read_dill(path)
        if isinstance(payload, Clustering):
            clustering = payload
            data = payload.data
        elif isinstance(payload, dict):
            for key in ("data", "table", "qtable"):
                if key in payload:
                    value = payload[key]
                    if isinstance(value, QTable):
                        data = value
                    elif isinstance(value, Table):
                        data = QTable(value)
                    elif isinstance(value, pd.DataFrame):
                        data = QTable.from_pandas(value)
                    else:
                        data = None
                    if data is not None:
                        break
            if data is None:
                raise KeyError(
                    f"The dill payload at {path} does not contain recognised table keys."
                )
        elif isinstance(payload, (QTable, Table)):
            data = payload if isinstance(payload, QTable) else QTable(payload)
        elif isinstance(payload, pd.DataFrame):
            data = QTable.from_pandas(payload)
        else:
            raise TypeError(
                f"Unsupported payload type in {path}: {type(payload)!r}. Expected Clustering, dict, QTable/Table or pandas.DataFrame."
            )

        loader = DataLoader(str(path), verbose=verbose, debug_mode=debug_mode)
        if data is None:
            raise RuntimeError(f"No data recovered from dill file {path}.")
        _normalise_table(data, loader.logger)
        loader.data = data
        return {
            "data": data,
            "loader": loader,
            "clustering": clustering,
            "base_dir": base_dir,
            "output_dir": out_dir,
            "source": str(path),
            "dill_path": path,
        }

    # Fallback: regular file to be read by DataLoader
    loader = DataLoader(str(path), verbose=verbose, debug_mode=debug_mode)
    data = loader.load_data(**dataloader_kwargs)
    dill_path = path.with_suffix(".dill")
    if dill_cache and path.suffix.lower() != ".dill":
        payload = {
            "data": data,
            "source": str(path),
            "loader_kwargs": dataloader_kwargs,
        }
        with dill_path.open("wb") as handle:
            dill.dump(payload, handle, protocol=dill.HIGHEST_PROTOCOL)

    return {
        "data": data,
        "loader": loader,
        "clustering": None,
        "base_dir": base_dir,
        "output_dir": out_dir,
        "source": str(path),
        "dill_path": dill_path if dill_cache else None,
    }


def _load_from_table(
    table: QTable,
    *,
    source: str,
    output_dir: str | None,
    verbose: int,
    debug_mode: bool,
):
    base_dir = Path.cwd()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else base_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    loader = DataLoader(source, verbose=verbose, debug_mode=debug_mode)
    data = table.copy(copy_data=True)
    _normalise_table(data, loader.logger)
    loader.data = data

    return {
        "data": data,
        "loader": loader,
        "clustering": None,
        "base_dir": base_dir,
        "output_dir": out_dir,
        "source": source,
        "dill_path": None,
    }


def _read_dill(path: Path):
    try:
        with path.open("rb") as handle:
            return dill.load(handle)
    except Exception:
        with optuna_safe_unpickle():
            with path.open("rb") as handle:
                return dill.load(handle)


def _normalise_table(table: QTable, logger: logging.Logger) -> None:
    handle_masked_columns(table)
    apply_unit_corrections(table, logger=logger)


__all__ = ["load_dataset", "optuna_safe_unpickle"]
