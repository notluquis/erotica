#!/usr/bin/env python3
"""What PARSEC v1.2S draws above the brightest member in the run registry's ``map-ngc6383-parsec``
(hub finding ``isochrone-ms-vs-pms.md`` §3): the two [M/H] nodes that bracket the fit's +0.64 at
log t 6.99, raw, against the curve the fitter interpolates at pseudo-EEP, in the post-MS part.

If the interpolated curve leaves the envelope of its two nodes (a colour neither node reaches at that
point), the pseudo-EEP matched different phases between the nodes; if it stays inside, the red
supergiant branch is the nodes' own physics at that age. Output: ``parsec_top.json``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phases import SAMPLE  # noqa: E402
from real_phases import fitter_on, grid_and_pri  # noqa: E402

FIT = {
    "feh": 0.6425169135622059,
    "loga": 6.989999999269621,
    "dm": 10.390751928424441,
    "Av": 1.3849827230696397,
}


def main() -> None:
    from astropy.table import QTable, Table

    grid, pri = grid_and_pri("parsec")
    pri = {**pri, "Av_range": (0.5, 2.0), "dm_mu": 10.3, "dm_sigma": 0.2}
    f = fitter_on(grid, "parsec", QTable(Table.read(SAMPLE)), pri)
    X = f._interp_isochrone(FIT["feh"], FIT["loga"], np)
    mass, G, col = X[0], X[1] + FIT["dm"] + f._kG * FIT["Av"], X[2] + f._k_col1 * FIT["Av"]
    nodes = np.asarray(grid.feh_nodes, float)
    lo, hi = nodes[nodes <= FIT["feh"]].max(), nodes[nodes >= FIT["feh"]].min()
    a = float(grid.loga_nodes[np.argmin(np.abs(np.asarray(grid.loga_nodes) - FIT["loga"]))])
    out = {"fit": FIT, "bracket_feh": [float(lo), float(hi)], "loga_node": a, "nodes": {}}
    for z in (lo, hi):
        n = grid.node(float(z), a)
        m = np.asarray(n.mass)
        g = np.asarray(n.mags["Gmag"]) + FIT["dm"] + f._kG * FIT["Av"]
        c = np.asarray(n.mags["G_BPmag"]) - np.asarray(n.mags["G_RPmag"]) + f._k_col1 * FIT["Av"]
        out["nodes"][f"{z:.3f}"] = {
            "max_mass": float(m.max()),
            "G_min": float(g.min()),
            "colour_at_G_min": float(c[np.argmin(g)]),
            "n_points_mass_gt_14": int((m > 14).sum()),
            "colour_range_mass_gt_14": [float(c[m > 14].min()), float(c[m > 14].max())]
            if (m > 14).any()
            else None,
            "label": [str(v) for v in np.unique(n.extra.get("label", []))][:12]
            if hasattr(n, "extra") and "label" in n.extra
            else None,
        }
    k = mass > 14
    out["interp"] = {
        "max_mass": float(mass.max()),
        "G_min": float(G.min()),
        "colour_range_mass_gt_14": [float(col[k].min()), float(col[k].max())],
        "n_points_mass_gt_14": int(k.sum()),
        "max_jump_colour_between_consecutive": float(np.max(np.abs(np.diff(col[k]))))
        if k.sum() > 1
        else None,
    }
    # same at solar composition and at the global MIST-like age (log t 6.0, 6.4) for context
    for feh, la in ((0.0, 6.0), (0.0, 6.4), (0.0, 6.99)):
        Y = f._interp_isochrone(float(np.clip(feh, nodes.min(), nodes.max())), la, np)
        out.setdefault("context", []).append(
            {
                "feh": feh,
                "loga": la,
                "max_mass": float(Y[0].max()),
                "G_min_at_dm10.223_Av0.66": float((Y[1] + 10.223 + f._kG * 0.66).min()),
            }
        )
    (HERE / "parsec_top.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
