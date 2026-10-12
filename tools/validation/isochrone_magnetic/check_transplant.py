"""V1 (BC hold-out error) and V2 (identity) of the transplant; writes check_transplant.json."""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "isochrone_phases"))
from phases import mist  # noqa: E402
from transplant import feiden_shift, spots_shift, transplant, v1_bc_holdout  # noqa: E402

t0 = time.time()
grid = mist(feh=[-0.5, -0.25, 0.0])
out = {"v1": v1_bc_holdout(grid)}
gid, st = transplant(grid, None)
dmax = max(
    float(np.max(np.abs(gid.node(f, a).mags[b] - grid.node(f, a).mags[b])))
    for f, a in grid.nodes()
    for b in grid.bands
)
out["v2_identity_max_abs"] = dmax
out["v2_stats"] = st
for nm, src, tt in (
    ("spots034", spots_shift(0.34), False),
    ("spots034_2T", spots_shift(0.34), True),
    ("feiden", feiden_shift(), False),
):
    g2, st = transplant(grid, src, two_temp=tt)
    # the shift in G and BP-RP at log t 6.0 and 6.3, [Fe/H] -0.5, for a few masses
    rows = {}
    for a in (6.0, 6.3, 6.7):
        n0, n1 = grid.node(-0.5, a), g2.node(-0.5, a)
        r = {}
        for m in (0.2, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 3.0):
            k = int(np.argmin(np.abs(n0.mass - m)))
            b = grid.bands
            r[str(m)] = [
                round(float(n1.mags[b[0]][k] - n0.mags[b[0]][k]), 4),
                round(float((n1.mags[b[1]][k] - n1.mags[b[2]][k]) - (n0.mags[b[1]][k] - n0.mags[b[2]][k])), 4),
            ]
        rows[str(a)] = r
    out[nm] = {"stats": st, "dG_dcol_feh-0.5": rows}
out["seconds"] = round(time.time() - t0, 1)
(HERE / "check_transplant.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
