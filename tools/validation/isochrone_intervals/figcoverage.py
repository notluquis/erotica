#!/usr/bin/env python3
"""Figure of ``coverage.json``: per replicate, the error (estimate - truth) of log t and [Fe/H] with
the 68 % and 90 % intervals of each estimator (control = MAP +- Laplace; NUTS = sample quantiles),
side by side. A replicate covered by an interval is one whose bar crosses zero.
Writes ``coverage_<cell>.png``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main(cell: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads((HERE / "coverage.json").read_text())[cell]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
    for a, p in zip(ax, ("loga", "feh"), strict=True):
        for i, r in enumerate(d["rows"]):
            c = r[f"ctrl_{p}"]
            a.errorbar(i - 0.12, c["err"], yerr=1.645 * c["sd"], fmt="none", ecolor="0.6", lw=1)
            a.errorbar(i - 0.12, c["err"], yerr=c["sd"], fmt="o", color="0.3", ms=4, lw=2.5,
                       label="MAP + Laplace" if i == 0 else None)
            if f"nuts_{p}" in r:
                n = r[f"nuts_{p}"]
                t = n["median"] - n["err"]
                conv = r["nuts_gate"]["converged"]
                col = "C0" if conv else "C3"
                a.plot([i + 0.12] * 2, [n["q05"] - t, n["q95"] - t], color=col, lw=1, alpha=0.6)
                a.plot([i + 0.12] * 2, [n["q16"] - t, n["q84"] - t], color=col, lw=2.5)
                a.plot(i + 0.12, n["err"], "s", color=col, ms=4,
                       label=("NUTS (converged)" if conv else "NUTS (not converged)")
                       if i == 0 else None)
        a.axhline(0, color="k", lw=0.8)
        a.set_xlabel("replicate")
        a.set_ylabel(f"estimate - truth, {p}")
        a.set_title(f"cell {cell}: {p} (thick 68 %, thin 90 %)", fontsize=9)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(HERE / f"coverage_{cell}.png", dpi=90)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "A")
