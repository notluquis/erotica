"""R38-08 (prerregistro hub §4.3): ruido de la compuerta G1/G2 con los tamanos por cumulo del POOL fuera de muestra, ANTES de aplicar
el modelo (solo cuenta estrellas; misma simulacion que exp2_power.py: binomial pura y beta-binomial con CV = 1 entre cumulos).
Corre: ~/miniforge3/envs/erotica-2026-10/bin/python r38_power.py > r38_power_r113_measured.txt
r38_power_measured.txt es la corrida anterior al arreglo R113-01 (concentracion de la Beta sin el factor 1/p, env erotica;
ver power_sim.py y README, Defecto conocido): queda como el registro que citaba el hallazgo.
"""
import json
from pathlib import Path

import numpy as np

from power_sim import cluster_fpr

import exp2_common as E
from r38_oos_data import load_oos

rng = np.random.default_rng(7)
o = load_oos()
old, grp, k = o["old"], o["grp"], E.tramo_index(o["mg0"])
print(f"pool: {len(old)} estrellas en {len(np.unique(grp))} cumulos; etiqueta 0 (viejas) {old.sum()} en {len(np.unique(grp[old]))} cumulos; "
      f"etiqueta 1 (PMS por §0.2, quedan fuera de la FPR) {(~old).sum()}")


def boot_upper(fp, n, R=400):
    return np.percentile([fp[i].sum() / max(n[i].sum(), 1) for i in (rng.integers(0, len(n), len(n)) for _ in range(R))], 97.5)


def sim(nc, p, cv, S=300):
    out = []
    for _ in range(S):
        pc = cluster_fpr(p, cv, len(nc), rng)
        out.append(boot_upper(rng.binomial(nc, pc), nc) > 0.10)
    return np.mean(out)


rows = {}
for name, s in [("todos", old)] + [(E.TRAMO_NAMES[i], old & (k == i)) for i in range(len(E.TRAMOS))]:
    u, nc = np.unique(grp[s], return_counts=True)
    # same call order as the f-string of the pre-R113 version, so the random stream is the same
    r = {"n_stars": int(s.sum()), "n_clusters": int(len(u)), "max_per_cluster": int(nc.max()) if len(nc) else 0}
    r["fail03_disp"] = float(sim(nc, 0.03, 1))
    r["fail05_binom"] = float(sim(nc, 0.05, 0))
    r["fail05_disp"] = float(sim(nc, 0.05, 1))
    r["pass15_binom"] = float(1 - sim(nc, 0.15, 0))
    r["pass15_disp"] = float(1 - sim(nc, 0.15, 1))
    rows[name] = r
    print(f"  {name:8s} n {s.sum():5d} en {len(u):3d} cumulos (max {nc.max()}): P(falla | 0,03) disp {r['fail03_disp']:.3f} | "
          f"P(falla | 0,05) binom {r['fail05_binom']:.3f} disp {r['fail05_disp']:.3f} | P(pasa | 0,15) binom {r['pass15_binom']:.3f} "
          f"disp {r['pass15_disp']:.3f}", flush=True)

# JSON sidecar (unrounded; S = 300 simulations per cell, bootstrap R = 400, gate: 97.5 % bound > 0.10)
Path("results").mkdir(exist_ok=True)
Path("results/r38_power_r113.json").write_text(json.dumps({
    "script": "r38_power.py", "fix": "R113-01: Beta concentration (1 - p(1 + cv^2)) / (p cv^2)", "seed": 7, "S": 300,
    "bootstrap_R": 400, "gate": "97.5 % cluster-bootstrap bound of the FPR > 0.10", "cv_disp": 1.0,
    "sample": "pool fuera de muestra", "rows": rows}, indent=1) + "\n")
