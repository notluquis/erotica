#!/usr/bin/env python3
"""The CMD the author saw (registry run ``map-ngc6383-mist-v12``: max likelihood, C1 priors, NO parallax
dm prior) against the same grid with the parallax prior (``colour-ngc6383-r1``) and PARSEC's map run,
over the 254 members (hub finding ``isochrone-ms-vs-pms.md`` §2). Reads the registry JSONs only.
Output: ``fig_registry_overlay.png``."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "isochrone_runs/runs"
SETS = [
    ("map-ngc6383-mist-v12-cf596a9.json", "MIST v1.2, sin paralaje (map)", "C0"),
    ("map-ngc6383-mist-v25-cf596a9.json", "MIST v2.5, sin paralaje (map)", "C2"),
    ("map-ngc6383-parsec-v12s-cf596a9.json", "PARSEC v1.2S, sin paralaje (map)", "C3"),
    ("colour-ngc6383-r1-b82effc.json", "MIST v1.2, con paralaje (R1)", "k"),
]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5))
    for ax, ylim in zip(axes, [(21, 3), (14.5, 8)], strict=True):
        r = json.loads((RUNS / SETS[0][0]).read_text())
        ax.scatter(r["cmd"]["color"], r["cmd"]["mag"], s=6, c="0.5", zorder=1)
        for fn, lab, c in SETS:
            r = json.loads((RUNS / fn).read_text())
            m = r["isochrone"]["median"]
            p = r["posterior"]
            ax.plot(
                m["color"],
                m["mag"],
                c=c,
                lw=1.2,
                label=f"{lab}: log t {p['loga']['q50']:.2f}, dm {p['dm']['q50']:.2f}, A_V {p['Av']['q50']:.2f}",
            )
        ax.axhline(14, ls=":", c="0.3")
        ax.set_ylim(*ylim)
        ax.set_xlim(-0.2, 3.3 if ylim[0] > 15 else 1.6)
        ax.set_xlabel("BP-RP")
        ax.set_ylabel("G")
    axes[0].legend(fontsize=7, loc="lower left")
    axes[1].set_title("G < 14 (35 miembros)")
    fig.tight_layout()
    fig.savefig(HERE / "fig_registry_overlay.png", dpi=120)


if __name__ == "__main__":
    main()
