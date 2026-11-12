"""R38-08 (prerregistro hub §4.3): ruido de la compuerta G1/G2 con los tamanos por cumulo del POOL fuera de muestra, ANTES de aplicar
el modelo (solo cuenta estrellas; misma simulacion que exp2_power.py: binomial pura y beta-binomial con CV = 1 entre cumulos).
Corre: ~/miniforge3/envs/erotica-2026-10/bin/python r38_power.py > r38_power_r113_measured.txt
r38_power_measured.txt es la corrida anterior al arreglo R113-01 (concentracion de la Beta sin el factor 1/p, env erotica;
ver power_sim.py y README, Defecto conocido): queda como el registro que citaba el hallazgo.
"""
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


for name, s in [("todos", old)] + [(E.TRAMO_NAMES[i], old & (k == i)) for i in range(len(E.TRAMOS))]:
    u, nc = np.unique(grp[s], return_counts=True)
    print(f"  {name:8s} n {s.sum():5d} en {len(u):3d} cumulos (max {nc.max()}): P(falla | 0,03) disp {sim(nc, .03, 1):.3f} | "
          f"P(falla | 0,05) binom {sim(nc, .05, 0):.3f} disp {sim(nc, .05, 1):.3f} | P(pasa | 0,15) binom {1 - sim(nc, .15, 0):.3f} "
          f"disp {1 - sim(nc, .15, 1):.3f}", flush=True)
