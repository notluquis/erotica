"""``paper/figures/make_pipeline.py`` must not leave its figure open once it has saved it."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def test_main_closes_the_figure_it_drew(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "make_pipeline", ROOT / "paper" / "figures" / "make_pipeline.py"
    )
    mp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mp)
    monkeypatch.setattr(mp, "HERE", tmp_path)
    plt.close("all")
    mp.main()
    assert plt.get_fignums() == []
    assert (tmp_path / "pipeline.png").exists()
