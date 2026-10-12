"""R38-02, paso 0 (mirar antes de medir): CMD de OC_0011 con la isocrona HR23 encima y los falsos positivos marcados.
Env: erotica. Corre: cd ~/erotica-wt-pms/tools/pms_classifier && OMP_NUM_THREADS=2 ~/miniforge3/envs/erotica/bin/python r38_oc0011_cmd.py
Salida: figs/r38_oc0011_cmd.png
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import core
import exp2_common as E
import exp2_grid as GR

C = Path("cache")
d = GR.prepare()
oof = np.load(C / "exp2_oof.npz")
th = np.array(json.load(open(C / "exp2_selected.json"))["th"])
dec = E.decide(oof["T0P1K0F0"].astype(float), d["mg0"], th)
m = d["m"]
name = "OC_0011"
s = d["grp"] == name
G, BP, RP = (np.asarray(m[c], float)[s] for c in ("Gmag", "BPmag", "RPmag"))
mod, av = float(np.median(np.asarray(m["MOD50"], float)[s])), float(np.median(np.asarray(m["AV50"], float)[s]))
lt = {k: float(np.median(np.asarray(m[k], float)[s])) for k in ("logAge16", "logAge50", "logAge84")}
mi = core._read_mist()
fig, ax = plt.subplots(1, 2, figsize=(11, 6))
fp = dec[s]
ax[0].scatter(BP[~fp] - RP[~fp], G[~fp], s=5, c="0.5", label=f"no FP ({(~fp).sum()})")
ax[0].scatter(BP[fp] - RP[fp], G[fp], s=7, c="r", label=f"FP elegido ({fp.sum()})")
for k, ls in (("logAge16", ":"), ("logAge50", "-"), ("logAge84", "--")):
    ages = np.unique(mi["log10_isochrone_age_yr"])
    t0 = ages[np.argmin(np.abs(ages - lt[k]))]
    z = (mi["log10_isochrone_age_yr"] == t0) & (mi["phase"] <= 3)
    ax[0].plot(mi["Gaia_BP_EDR3"][z] - mi["Gaia_RP_EDR3"][z] + (1.002 - 0.589) * av, mi["Gaia_G_EDR3"][z] + mod + 0.789 * av,
               "b", ls=ls, lw=1, label=f"MIST log t {t0:.2f}")
ax[0].arrow(0.6, 11.5, 0.413, 0.789, color="k", head_width=0.05)
ax[0].text(0.6, 11.3, "A_V = 1", fontsize=8)
ax[0].invert_yaxis(); ax[0].set_xlabel("BP-RP"); ax[0].set_ylabel("G")
ax[0].set_title(f"{name}: HR23 MOD {mod:.2f} A_V {av:.2f} log t {lt['logAge50']:.2f}")
ax[0].legend(fontsize=7)
# dereddened, with young-cluster PMS training density for context
y = d["young"]
X = d["X"]
ax[1].hexbin(X[y, 1], X[y, 0], gridsize=60, cmap="Greens", bins="log", mincnt=1)
ax[1].scatter(X[s, 1][~fp], X[s, 0][~fp], s=5, c="0.4")
ax[1].scatter(X[s, 1][fp], X[s, 0][fp], s=7, c="r")
ax[1].invert_yaxis(); ax[1].set_xlabel("(BP-RP)_0"); ax[1].set_ylabel("M_G,0")
ax[1].set_title("desenrojecido con A_V HR23; verde = PMS de jovenes (log t < 6,8)")
fig.tight_layout(); Path("figs").mkdir(exist_ok=True)
fig.savefig("figs/r38_oc0011_cmd.png", dpi=110)
print("FP", fp.sum(), "de", s.sum(), "MG0 de FP: min %.2f med %.2f max %.2f" % tuple(np.percentile(X[s, 0][fp], [0, 50, 100])))
print("BP-RP obs de FP: med %.2f; no FP: med %.2f" % (np.median((BP - RP)[fp]), np.median((BP - RP)[~fp])))
