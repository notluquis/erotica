"""Ruido de la compuerta del segundo prerregistro (§2.4), calculado ANTES de entrenar la grilla: con los tamanos reales por cumulo de la
mitad CONFIRMATORIA (B) de los viejos, en cada tramo, ¿con que probabilidad la regla "cota 95 % del bootstrap por cumulo <= 0,10" falla
con un FPR verdadero de 0,05 (falso fallo) y pasa con uno de 0,15 (falso pase)? Dos mundos: binomial pura por cumulo, y beta-binomial
con dispersion entre cumulos (coeficiente de variacion 1 del FPR por cumulo: el modelo del F2 da 0,27-1,08 por tramo y Sagitta
0,0-2,25, momentos sobre cumulos con n >= 10; medido antes de escribir §2).
Corre: ~/miniforge3/envs/erotica/bin/python exp2_power.py > exp2_power_measured.txt
"""
import numpy as np

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
        if cv > 0:
            a = (1 - p * (1 + cv ** 2)) / cv ** 2
            pc = rng.beta(p * a, (1 - p) * a, len(nc)) if a > 0 else np.full(len(nc), p)
        else:
            pc = np.full(len(nc), p)
        fp = rng.binomial(nc, pc)
        out.append(boot_upper(fp, nc) > 0.10)
    return np.mean(out)


for name, s in [("todos", B)] + [(E.TRAMO_NAMES[i], B & (k == i)) for i in range(len(E.TRAMOS))]:
    u, nc = np.unique(grp[s], return_counts=True)
    print(f"  {name:8s} n {s.sum():5d} en {len(u):3d} cumulos (max {nc.max() if len(nc) else 0}): "
          f"P(falla | FPR 0,03) disp {sim(nc, .03, 1):.3f} | P(falla | FPR 0,05) binom {sim(nc, .05, 0):.3f} disp {sim(nc, .05, 1):.3f} | "
          f"P(pasa | FPR 0,15) binom {1 - sim(nc, .15, 0):.3f} disp {1 - sim(nc, .15, 1):.3f}", flush=True)
