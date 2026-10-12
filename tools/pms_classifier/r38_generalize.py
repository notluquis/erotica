"""R38-02, generalizacion (prerregistro hub §4.2): ¿el FPR de los viejos crece con la A_V por extincion del cumulo mal estimada o por
extincion diferencial? Cumulo por cumulo en A u B (OOF de la elegida, n >= 10): Spearman con AV50, diffAV50 y |AV50 - A_V mapa|,
parciales, con y sin OC_0011, leave-one-out; y re-prediccion de los viejos con la A_V del mapa (Vergely+2022) usando los modelos de
cada particion (r38_folds.py). POST HOC (no prerregistrado, rotulado): lo mismo sobre el pool fuera de muestra con el modelo final.
Env: erotica. Corre: cd ~/erotica-wt-pms/tools/pms_classifier && OMP_NUM_THREADS=2 nice -n 5 ~/miniforge3/envs/erotica/bin/python \
     r38_generalize.py > r38_generalize_measured.txt 2> cache/r38_generalize.log
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
from xgboost import XGBClassifier

import core
import exp2_common as E
import exp2_grid as GR
import r38_common as R
from r38_oos_data import load_oos
from train_validate import f

C = Path("cache")
th = np.array(json.load(open(C / "exp2_selected.json"))["th"])


def partial_spearman(y, x, z):
    ry, rx, rz = rankdata(y), rankdata(x), rankdata(z)
    res = lambda a: a - np.polyval(np.polyfit(rz, a, 1), rz)  # noqa: E731
    return pearsonr(res(ry), res(rx))


def per_cluster(dec, grp, sel, extra):
    ug, inv = np.unique(grp[sel], return_inverse=True)
    n = np.bincount(inv)
    x = np.bincount(inv, weights=dec[sel].astype(float))
    cols = {k: np.array([np.median(v[sel][inv == j]) for j in range(len(ug))]) for k, v in extra.items()}
    return ug, n, x, cols


def report(tag, ug, n, x, cols):
    m = n >= 10
    fpr = x / np.maximum(n, 1)
    print(f"  [{tag}] cumulos con n >= 10: {m.sum()}")
    for k in cols:
        ok = m & np.isfinite(cols[k])
        r = spearmanr(fpr[ok], cols[k][ok])
        print(f"    Spearman FPR vs {k:10s}: rho {r.statistic:+.3f} p {r.pvalue:.3g} (n {ok.sum()})")
    ok = m & np.isfinite(cols["AV50"]) & np.isfinite(cols["diffAV50"])
    for a, b in (("diffAV50", "AV50"), ("AV50", "diffAV50")):
        r = partial_spearman(fpr[ok], cols[a][ok], cols[b][ok])
        print(f"    parcial FPR vs {a} | {b}: r {r.statistic:+.3f} p {r.pvalue:.3g}")
    print(f"    Spearman AV50 vs diffAV50: {spearmanr(cols['AV50'][ok], cols['diffAV50'][ok]).statistic:+.3f}")


d = GR.prepare()
oof = np.load(C / "exp2_oof.npz")["T0P1K0F0"].astype(float)
dec = E.decide(oof, d["mg0"], th)
m = d["m"]
old = d["old"]
L, B_, av, dav = f(m["GLON"]), f(m["GLAT"]), f(m["AV50"]), f(m["diffAV50"])
# A_V del mapa por cumulo (posicion y distancia medianas)
ug = np.unique(d["grp"][old])
avmap_c = {}
for c in ug:
    s = d["grp"] == c
    avmap_c[c] = R.vergely_av(np.median(L[s]), np.median(B_[s]), np.median(d["dist"][s]))[0]
avmap = np.array([avmap_c.get(g, np.nan) for g in d["grp"]])
print(f"[V0] mapa: {np.sum(np.isfinite(list(avmap_c.values())))} de {len(ug)} viejos dentro del cubo; AV50 - mapa: mediana "
      f"{np.nanmedian(np.array([np.median(av[d['grp'] == c]) - avmap_c[c] for c in ug])):+.2f}")
print(f"    OC_0011: AV50 {np.median(av[d['grp'] == 'OC_0011']):.2f}, mapa {avmap_c['OC_0011']:.2f}")
extra = dict(AV50=av, diffAV50=dav, dAVmap=np.abs(av - avmap), dist=d["dist"])
print("[V1] A u B, OOF de la elegida, A_V de HR23")
ugc, n, x, cols = per_cluster(dec, d["grp"], old, extra)
report("todos", ugc, n, x, cols)
keep = ugc != "OC_0011"
report("sin OC_0011", ugc[keep], n[keep], x[keep], {k: v[keep] for k, v in cols.items()})
m10 = n >= 10
fpr = x / n
loo = []
for j in np.where(m10)[0]:
    k = m10.copy(); k[j] = False
    loo.append(spearmanr(fpr[k], cols["AV50"][k]).statistic)
loo2 = []
for j in np.where(m10)[0]:
    k = m10.copy(); k[j] = False
    loo2.append(partial_spearman(fpr[k], cols["diffAV50"][k], cols["AV50"][k]).statistic)
print(f"    leave-one-out: rho(AV50) en [{min(loo):+.3f}; {max(loo):+.3f}]; parcial(diffAV50 | AV50) en [{min(loo2):+.3f}; {max(loo2):+.3f}]")
for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4)):
    s = (cols["AV50"] >= lo) & (cols["AV50"] < hi)
    print(f"    AV50 {lo}-{hi}: cumulos {s.sum():3d} FPR {x[s].sum() / max(n[s].sum(), 1):.3f}")
for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 9)):
    s = (cols["diffAV50"] >= lo) & (cols["diffAV50"] < hi)
    print(f"    diffAV50 {lo}-{hi}: cumulos {s.sum():3d} FPR {x[s].sum() / max(n[s].sum(), 1):.3f}")

print("[V2] re-prediccion de los viejos con la A_V del MAPA (modelos de particion; entrenamiento sin cambios)")
fold = np.load(C / "r38_folds.npz")["fold"]
ph = {}
q = np.array([str(z) if (z is not None and not np.ma.is_masked(z)) else "XXX" for z in m["ph_qual"]])
q = np.array([z if len(z) == 3 else "XXX" for z in q])
J = core.nir_clean(f(m["j_m"]), [z[0] for z in q])
H = core.nir_clean(f(m["h_m"]), [z[1] for z in q])
K = core.nir_clean(f(m["ks_m"]), [z[2] for z in q])
G, BP, RP, mod = f(m["Gmag"]), f(m["BPmag"]), f(m["RPmag"]), f(m["MOD50"])
avuse = np.where(np.isfinite(avmap), avmap, av)
Xm = core.features(G, BP, RP, J, H, K, mod, avuse)
pm_ = np.full(len(G), np.nan)
for k in range(5):
    s = old & (fold == k)
    mdl = XGBClassifier()
    mdl.load_model(C / f"r38_fold{k}.json")
    pm_[s] = mdl.predict_proba(Xm[s])[:, 1]
decm = np.zeros(len(G), bool)
decm[old] = E.decide(pm_[old], Xm[old, 0], th)
print(f"    FPR viejos: HR23 {dec[old].mean():.4f} -> mapa {decm[old].mean():.4f}; B: {dec[d['B']].mean():.4f} -> {decm[d['B']].mean():.4f}; "
      f"OC_0011 FP {dec[d['grp'] == 'OC_0011'].sum()} -> {decm[d['grp'] == 'OC_0011'].sum()}")
ugc2, n2, x2, cols2 = per_cluster(decm, d["grp"], old, dict(AV50=av, diffAV50=dav, AVmap=avmap))
for lo, hi in ((0, 1), (1, 2), (2, 3), (3, 4)):
    s = (cols2["AV50"] >= lo) & (cols2["AV50"] < hi)
    print(f"    por AV50 de HR23 {lo}-{hi}: FPR HR23 {x[s].sum() / max(n[s].sum(), 1):.3f} -> mapa {x2[s].sum() / max(n2[s].sum(), 1):.3f}")
m10 = n2 >= 10
r = spearmanr(x2[m10] / n2[m10], cols2["AV50"][m10])
print(f"    con el mapa: Spearman FPR vs AV50 rho {r.statistic:+.3f} p {r.pvalue:.3g}; parcial vs diffAV50 | AV50 "
      f"{partial_spearman(x2[m10] / n2[m10], cols2['diffAV50'][m10], cols2['AV50'][m10]).statistic:+.3f}")

print("[V3] POST HOC, no prerregistrado: el pool fuera de muestra con el modelo final (n mayor)")
o = load_oos()
mdl = XGBClassifier()
mdl.load_model(C / "exp2_model_T0P1K0F0D1.json")
deco = E.decide(mdl.predict_proba(o["X"])[:, 1], o["mg0"], th)
ugp, nP, xP, colsP = per_cluster(deco, o["grp"], o["old"], dict(AV50=o["av"], diffAV50=o["dav"], dist=o["dist"]))
report("pool", ugp, nP, xP, colsP)
