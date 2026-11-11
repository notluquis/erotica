#!/usr/bin/env python3
"""Per-window best isochrones (arm ``a``: [Fe/H], dm, A_V at the global fit) over the 254 members, one
panel per grid (hub finding ``isochrone-ms-vs-pms.md`` §2). Reads the registry runs written by
``export_run.py phases`` only. Output: ``fig_phases.png``."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "isochrone_runs/runs"
HEAD = "20a6901"
GRIDS = [("mist12", "MIST v1.2"), ("mist25", "MIST v2.5"), ("parsec", "PARSEC v1.2S")]
PH = [
    ("global", "global", "k"),
    ("pms-a", "pms (G >= 14)", "C0"),
    ("ms-a", "ms (G < 14)", "C3"),
    ("ums-a", "ums (G < 12)", "C1"),
    ("to-a", "to (12-14)", "C2"),
]


def main():
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.5), sharey=True)
    for ax, (g, lab) in zip(axes, GRIDS, strict=True):
        r0 = json.loads((RUNS / f"phases-ngc6383-{g}-global-{HEAD}.json").read_text())
        ax.scatter(r0["cmd"]["color"], r0["cmd"]["mag"], s=6, c="0.55", zorder=1)
        for tag, name, c in PH:
            r = json.loads((RUNS / f"phases-ngc6383-{g}-{tag}-{HEAD}.json").read_text())
            m = r["isochrone"]["median"]
            ax.plot(
                m["color"],
                m["mag"],
                c=c,
                lw=1.3 if tag != "global" else 1.8,
                ls="--" if tag == "global" else "-",
                label=f"{name}: log t {r['posterior']['loga']['q50']:.2f}",
            )
        ax.axhline(14, ls=":", c="0.3")
        ax.axhline(12, ls=":", c="0.6")
        ax.set_ylim(21, 6)
        ax.set_xlim(-0.2, 3.3)
        ax.set_title(f"{lab} (brazo a)")
        ax.set_xlabel("BP-RP")
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("G")
    fig.tight_layout()
    fig.savefig(HERE / "fig_phases.png", dpi=110)


if __name__ == "__main__":
    main()
