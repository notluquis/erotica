"""POST HOC (escrito despues de ver que F2 fallo; no estaba prerregistrado): donde caen los falsos positivos de los cumulos viejos,
y si ese regimen es el de NGC 6383. Entradas: cache/oof.npz (train_validate.py), cache/ngc6383_pred.npz (apply_ngc6383.py).
Env: erotica. Corre: nice -n 19 ~/miniforge3/envs/erotica/bin/python posthoc_f2.py > posthoc_f2_measured.txt
"""
from pathlib import Path

import numpy as np
from astropy.table import Table

import core
from train_validate import build, cp, f, load

C = Path(__file__).parent / "cache"
o = np.load(C / "oof.npz", allow_pickle=True)
m = load()
m = m[m["role"] == "train"]
X, y, ph = build(m)
assert np.all(np.asarray(m["GaiaDR3"], np.int64) == o["sid"])
MG0 = X[:, core.FEATURES.index("MG0")]
old = (o["logt"] >= 8.5) & (o["dist"] >= 600) & (o["dist"] <= 2500) & ~o["y"]
print("[P1] FPR (propio >= 0,5 / Sagitta >= 0,6) en cumulos viejos con etiqueta 0, por M_G,0 y por G")
for lo, hi in ((-5, 1), (1, 3), (3, 5), (5, 7), (7, 12)):
    s = old & (MG0 >= lo) & (MG0 < hi)
    k1, k2 = int(np.sum(o["oof"][s] >= .5)), int(np.sum(o["sag"][s] >= .6))
    print(f"  M_G,0 {lo:+d} a {hi:+d}: n {s.sum():5d}  propio {k1 / max(s.sum(), 1):.3f} {np.round(cp(k1, s.sum()), 3)}  "
          f"Sagitta {k2 / max(s.sum(), 1):.3f}")
for lo, hi in ((0, 15), (15, 17), (17, 18), (18, 19), (19, 21)):
    s = old & (ph["G"] >= lo) & (ph["G"] < hi)
    k1 = int(np.sum(o["oof"][s] >= .5))
    print(f"  G {lo}-{hi}: n {s.sum():5d}  propio {k1 / max(s.sum(), 1):.3f} {np.round(cp(k1, s.sum()), 3)}")

t = Table.read(Path.home() / "phd/agent-findings/scripts/virac2_ngc6383_members_clean.ecsv")
t = t[np.argsort(np.asarray(t["GaiaDR3"], np.int64))]
pr = np.load(C / "ngc6383_pred.npz")
h = Table.read(C / "ngc6383_hr23.ecsv")[0]
G = f(t["G"])
mg0 = G - float(h["MOD50"]) - core.EXT["G"] * float(h["AV50"])
lab = core.label(G, float(h["MOD50"]), float(h["AV50"]), float(h["logAge50"]))
pc = pr["p_cal"] >= .5
print(f"[P2] NGC 6383: M_G,0 de los 254 [p5 p50 p95] {np.percentile(mg0, [5, 50, 95]).round(2)}; G {np.percentile(G, [5, 50, 95]).round(2)}")
print(f"  sobre el turn-on (etiqueta 0): {np.sum(~lab)}; de ellas el propio llama PMS a {np.sum(~lab & pc)}; Sagitta {np.sum(~lab & (pr['sag_cal'] >= .6))}")
print(f"  bajo el turn-on (etiqueta 1): {lab.sum()}; propio PMS {np.sum(lab & pc)} ({np.sum(lab & pc) / lab.sum():.3f}); "
      f"Sagitta {np.sum(lab & (pr['sag_cal'] >= .6))} ({np.sum(lab & (pr['sag_cal'] >= .6)) / lab.sum():.3f})")
for lo, hi in ((-5, 3), (3, 5), (5, 7), (7, 12)):
    s = (mg0 >= lo) & (mg0 < hi)
    print(f"  M_G,0 {lo:+d} a {hi:+d}: n {s.sum():3d}, propio PMS {np.sum(pc[s]):3d}, Sagitta PMS {np.sum(pr['sag_cal'][s] >= .6):3d}")

print("[P3] discriminacion por tramo de M_G,0: recall en cumulos jovenes (logAge < 6,8, etiqueta 1) contra FPR en viejos (de [P1])")
yng = (o["logt"] < 6.8) & o["y"]
for lo, hi in ((-5, 1), (1, 3), (3, 5), (5, 7), (7, 12)):
    s, so = yng & (MG0 >= lo) & (MG0 < hi), old & (MG0 >= lo) & (MG0 < hi)
    r1, r2 = np.mean(o["oof"][s] >= .5) if s.any() else np.nan, np.mean(o["sag"][s] >= .6) if s.any() else np.nan
    f1, f2 = np.mean(o["oof"][so] >= .5) if so.any() else np.nan, np.mean(o["sag"][so] >= .6) if so.any() else np.nan
    print(f"  M_G,0 {lo:+d} a {hi:+d}: jovenes n {s.sum():5d} recall propio {r1:.3f} Sagitta {r2:.3f} | viejos FPR propio {f1:.3f} "
          f"Sagitta {f2:.3f} | recall - FPR: propio {r1 - f1:+.3f} Sagitta {r2 - f2:+.3f}")
