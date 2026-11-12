"""Ruido de la compuerta del segundo prerregistro (§2.4), calculado ANTES de entrenar la grilla: con los tamanos reales por cumulo de la
mitad CONFIRMATORIA (B) de los viejos, en cada tramo, ¿con que probabilidad la regla "cota 95 % del bootstrap por cumulo <= 0,10" falla
con un FPR verdadero de 0,05 (falso fallo) y pasa con uno de 0,15 (falso pase)? Dos mundos: binomial pura por cumulo, y beta-binomial
con dispersion entre cumulos (coeficiente de variacion 1 del FPR por cumulo: el modelo del F2 da 0,27-1,08 por tramo y Sagitta
0,0-2,25, momentos sobre cumulos con n >= 10; medido antes de escribir §2).
Corre: ~/miniforge3/envs/erotica-2026-10/bin/python exp2_power.py > exp2_power_r113_measured.txt
exp2_power_measured.txt es la corrida anterior al arreglo R113-01 (concentracion de la Beta sin el factor 1/p, env erotica;
ver power_sim.py y README, Defecto conocido): queda como el registro que citaba el hallazgo.
"""
import json
from pathlib import Path

import numpy as np

from power_sim import cluster_fpr

import core
import exp2_common as E
from train_validate import build, load, f

rng = np.random.default_rng(7)
o = np.load("cache/oof.npz", allow_pickle=True)
m = load()
m = m[m["role"] == "train"]
X, y, ph = build(m)
grp = o["grp"].astype(str)
mg0 = X[:, core.FEATURES.index("MG0")]
old = (o["logt"] >= 8.5) & (o["dist"] >= 600) & (o["dist"] <= 2500) & ~o["y"]
A, B = E.old_split(grp, old)
print(f"viejos: {old.sum()} estrellas en {len(np.unique(grp[old]))} cumulos; ajuste A {A.sum()} ({len(np.unique(grp[A]))} cum), "
      f"confirmatoria B {B.sum()} ({len(np.unique(grp[B]))} cum)")
k = E.tramo_index(mg0)


def boot_upper(fp, n, R=400):
    up = []
    for _ in range(R):
        i = rng.integers(0, len(n), len(n))
        up.append(fp[i].sum() / max(n[i].sum(), 1))
    return np.percentile(up, 97.5)


def sim(nc, p, cv, S=300):
    out = []
    for _ in range(S):
        pc = cluster_fpr(p, cv, len(nc), rng)
        fp = rng.binomial(nc, pc)
        out.append(boot_upper(fp, nc) > 0.10)
    return np.mean(out)


rows = {}
for name, s in [("todos", B)] + [(E.TRAMO_NAMES[i], B & (k == i)) for i in range(len(E.TRAMOS))]:
    u, nc = np.unique(grp[s], return_counts=True)
    # same call order as the f-string of the pre-R113 version, so the random stream is the same
    r = {"n_stars": int(s.sum()), "n_clusters": int(len(u)), "max_per_cluster": int(nc.max()) if len(nc) else 0}
    r["fail03_disp"] = float(sim(nc, 0.03, 1))
    r["fail05_binom"] = float(sim(nc, 0.05, 0))
    r["fail05_disp"] = float(sim(nc, 0.05, 1))
    r["pass15_binom"] = float(1 - sim(nc, 0.15, 0))
    r["pass15_disp"] = float(1 - sim(nc, 0.15, 1))
    rows[name] = r
    print(f"  {name:8s} n {s.sum():5d} en {len(u):3d} cumulos (max {nc.max() if len(nc) else 0}): "
          f"P(falla | FPR 0,03) disp {r['fail03_disp']:.3f} | P(falla | FPR 0,05) binom {r['fail05_binom']:.3f} disp {r['fail05_disp']:.3f} | "
          f"P(pasa | FPR 0,15) binom {r['pass15_binom']:.3f} disp {r['pass15_disp']:.3f}", flush=True)

# JSON sidecar (unrounded; S = 300 simulations per cell, bootstrap R = 400, gate: 97.5 % bound > 0.10)
Path("results").mkdir(exist_ok=True)
Path("results/exp2_power_r113.json").write_text(json.dumps({
    "script": "exp2_power.py", "fix": "R113-01: Beta concentration (1 - p(1 + cv^2)) / (p cv^2)", "seed": 7, "S": 300,
    "bootstrap_R": 400, "gate": "97.5 % cluster-bootstrap bound of the FPR > 0.10", "cv_disp": 1.0,
    "sample": "confirmatoria B", "rows": rows}, indent=1) + "\n")
