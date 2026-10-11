"""Figure for agent-findings/isochrone-sed-external.md: (left) A_V of the spectral-type anchors vs
radius, against the CMD level and the map's foreground; (right) synthetic posterior width of A_V
by true T_eff, configs a and b, and the atmosphere-mismatch bias."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.table import Table
HERE = Path(__file__).resolve().parent
OUT = Path("/Users/notluquis/phd/agent-findings/scripts/isochrone_sed_external.png")
a = json.load(open(HERE / "anchors.json"))
m = Table.read(HERE.parent / "isochrone_external" / "av_maps_perstar.ecsv")
idx = {int(s): i for i, s in enumerate(m["source_id"])}
r = [x for x in a["rows"] if x["model"] == "mist" and x["config"] == "b"]
rad = np.array([m["radius_arcmin"][idx[x["source_id"]]] for x in r])
av = np.array([x["av50"] for x in r]); lo = av - np.array([x["av16"] for x in r]); hi = np.array([x["av84"] for x in r]) - av
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].axhspan(0.59, 0.71, color="tab:red", alpha=0.25, label="CMD (X1–X3)")
ax[0].axhline(0.81, color="tab:green", ls="--", label="Edenhofer+24 a d_c")
h = a["hier_mist_b_all"]
ax[0].axhspan(h["mu"][0], h["mu"][2], color="tab:blue", alpha=0.2, label=f"μ anclas {h['mu'][1]:.2f}, τ {h['tau'][1]:.2f}")
ax[0].errorbar(rad, av, yerr=[lo, hi], fmt="o", color="k", ms=4)
ax[0].set_xlabel("radio (arcmin)"); ax[0].set_ylabel("A_V (SED con T_eff del tipo espectral, config b)")
ax[0].set_title("17 miembros con tipo espectral"); ax[0].legend(fontsize=8)
s = json.load(open(HERE / "synth_by_teff.json"))
for key, c, mk in (("ctl_a", "tab:blue", "o"), ("ctl_b", "tab:orange", "s")):
    bins = list(s[key]); x = np.arange(len(bins))
    ax[1].plot(x, [s[key][b]["med_sd_av"] for b in bins], marker=mk, color=c, label=f"sd(A_V) {key}")
    ax[1].plot(x, [s["eem-truth_" + key[-1]].get(b, {}).get("bias_av", np.nan) for b in bins], marker=mk, ls=":", color=c,
               label=f"sesgo A_V verdad EEM, ajuste MIST ({key[-1]})")
ax[1].axhline(0.15, color="gray", ls="--", label="umbral 0,15")
ax[1].set_xticks(x); ax[1].set_xticklabels([b.replace("T", "") for b in bins], rotation=20, fontsize=8)
ax[1].set_xlabel("T_eff verdadera (K)"); ax[1].set_ylabel("mag"); ax[1].set_title("Sintético: ancho y sesgo de A_V")
ax[1].legend(fontsize=7)
fig.tight_layout(); fig.savefig(OUT, dpi=130)
print(OUT)
