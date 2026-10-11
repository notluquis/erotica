"""Segundo experimento, paso 2: CONFIRMACION de la configuracion elegida por exp2_grid.py (G1-G4 de §2.4) y sintetico de verdad
conocida. Primera vez que se mira la mitad B de los viejos y NGC 6530 con la grilla nueva.

Prerregistro: ~/phd/agent-findings/pms-classifier-non-circular.md §2 (hub 42b93a6).
Env: erotica. Corre:
  cd ~/erotica-wt-pms/tools/pms_classifier && SAGITTA_SRC=~/.local/lib/sagitta-1.2.3 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 nice -n 5 \
      ~/miniforge3/envs/erotica/bin/python exp2_confirm.py > exp2_confirm_measured.txt 2> cache/exp2_confirm.log
Salida: exp2_confirm_measured.txt; cache/exp2_model_<config>.json (modelo final con todo el entrenamiento) y cache/exp2_confirm.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

import core
import exp2_common as E
import exp2_grid as GR
import sagitta_ref as S
from train_validate import cp, f

C = Path(__file__).parent / "cache"
RNG = np.random.default_rng(20261013)


def boot_fpr(pr, grp, R=1000):
    ug, inv = np.unique(grp, return_inverse=True)
    n = np.bincount(inv, minlength=len(ug)).astype(float)
    k = np.bincount(inv, weights=pr.astype(float), minlength=len(ug))
    v = []
    for _ in range(R):
        i = RNG.integers(0, len(ug), len(ug))
        v.append(k[i].sum() / max(n[i].sum(), 1))
    return np.percentile(v, [2.5, 97.5])


def gate_table(tag, pr, d, sel):
    k = E.tramo_index(d["mg0"])
    out = {}
    n, x = int(sel.sum()), int(pr[sel].sum())
    lo, hi = cp(x, n)
    blo, bhi = boot_fpr(pr[sel], d["grp"][sel])
    out["all"] = dict(fpr=x / n, cp=[lo, hi], boot=[blo, bhi])
    row = [f"  {tag:22s} global {x}/{n} = {x / n:.4f} CP [{lo:.4f}; {hi:.4f}] boot [{blo:.4f}; {bhi:.4f}]"]
    for i, nm in enumerate(E.TRAMO_NAMES):
        s = sel & (k == i)
        xs = int(pr[s].sum())
        blo, bhi = boot_fpr(pr[s], d["grp"][s])
        out[nm] = dict(fpr=xs / max(s.sum(), 1), n=int(s.sum()), boot=[blo, bhi], est=bool(bhi <= 0.10))
        row.append(f"    {nm:8s} {xs:4d}/{int(s.sum()):5d} = {xs / max(s.sum(), 1):.4f} boot [{blo:.4f}; {bhi:.4f}] "
                   f"{'<= 0,10' if bhi <= 0.10 else '>  0,10'}")
    print("\n".join(row), flush=True)
    return out


def fit_final(d, cfg):
    X = d["X1"] if cfg["F"] == "F1" else d["X"]
    y, stab = d["tg"][cfg["T"]]
    w = E.config_weights(cfg, y, stab, d["dobs"], d["dth"], d["logt"])
    trivial = cfg["P"] == "P0" and cfg["K"] == "K0"
    return GR.fit_any(X, y, seed=20261011, w=None if trivial else w, nfeat=X.shape[1])


def feats(cfg, G, BP, RP, J, H, K, mod, av):
    X = core.features(G, BP, RP, J, H, K, mod, av)
    return E.extra_features(X) if cfg["F"] == "F1" else X


# ---------------------------------------------------------------- sintetico de verdad conocida
def iso_interp(logt, phase_sel):
    m = core._read_mist()
    s = (np.abs(m["log10_isochrone_age_yr"] - logt) < 1e-6) & phase_sel(m)
    g = m["Gaia_G_EDR3"][s]
    o = np.argsort(g)
    cols = {b: m[c][s][o] for b, c in (("G", "Gaia_G_EDR3"), ("BP", "Gaia_BP_EDR3"), ("RP", "Gaia_RP_EDR3"),
                                         ("J", "2MASS_J"), ("H", "2MASS_H"), ("Ks", "2MASS_Ks"))}
    return cols


def synth(kind, n_per, mod, av, binary=False):
    """kind 'ms' = MIST log t 9,0 phase 0; 'pms' = log t 6,6 phase -1 bajo el turn-on. Uniforme en M_G,0 por tramo."""
    if kind == "ms":
        iso = iso_interp(9.0, lambda m: m["phase"] == 0)
    else:
        mon = float(core.mg_turn_on(6.6))
        iso = iso_interp(6.6, lambda m: (m["phase"] == -1) & (m["Gaia_G_EDR3"] > mon))
    rows = []
    for lo, hi in E.TRAMOS:
        lo2, hi2 = max(lo, iso["G"].min()), min(hi, iso["G"].max())
        if hi2 <= lo2:
            continue
        mg = RNG.uniform(lo2, hi2, n_per)
        rows.append({b: np.interp(mg, iso["G"], iso[b]) for b in iso})
    abs_ = {b: np.concatenate([r[b] for r in rows]) for b in rows[0]}
    if binary:
        abs_ = {b: v - 0.753 for b, v in abs_.items()}
    N = len(abs_["G"])
    at = np.clip(av + RNG.normal(0, 0.5, N), 0, None)  # A_V por estrella: extincion diferencial de NGC 6383 [I: sd 0,5]
    app = {b: abs_[b] + mod + core.EXT[b] * at for b in abs_}
    sig = {"G": np.full(N, 0.003), "BP": 0.005 + 0.01 * 10 ** (0.4 * (app["BP"] - 17)),
           "RP": 0.005 + 0.01 * 10 ** (0.4 * (app["RP"] - 17)),
           "J": 0.02 + 0.05 * 10 ** (0.4 * (app["J"] - 15.5)), "H": 0.02 + 0.05 * 10 ** (0.4 * (app["H"] - 15.0)),
           "Ks": 0.02 + 0.05 * 10 ** (0.4 * (app["Ks"] - 14.5))}
    obs = {b: app[b] + RNG.normal(0, 1, N) * sig[b] for b in app}
    for b, lim in (("J", 16.5), ("H", 16.0), ("Ks", 15.5)):  # limite de 2MASS [I: aproximado]
        obs[b] = np.where(app[b] > lim, np.nan, obs[b])
    return obs


if __name__ == "__main__":
    selj = json.load(open(C / "exp2_selected.json"))
    key = selj["chosen"]
    cfg = selj["cfg"]
    th = np.array(selj["th"])
    print(f"[C0] configuracion elegida en exp2_grid: {key}; umbrales por tramo {th.round(2).tolist()}")
    d = GR.prepare()
    oofs = np.load(C / "exp2_oof.npz")
    p = oofs[GR.tag(cfg)].astype(float)
    dec = E.decide(p, d["mg0"], th)
    o = np.load(C / "oof.npz", allow_pickle=True)
    assert np.array_equal(o["sid"], d["sid"])

    print("[G1/G2] FPR en la mitad CONFIRMATORIA B de los viejos (bootstrap por cumulo B = 1000)")
    res = {"chosen": gate_table(key, dec, d, d["B"])}
    res["control_F2"] = gate_table("control T0P0K0F0D0", oofs["T0P0K0F0"] >= 0.5, d, d["B"])
    res["sagitta06_token"] = gate_table("Sagitta(0,6) U->token", o["sag"] >= 0.6, d, d["B"])
    mB = d["m"][d["B"]]
    q = np.array([str(x) if (x is not None and not np.ma.is_masked(x) and len(str(x)) == 3) else "XXX" for x in mB["ph_qual"]])
    lst = {b: np.where(np.array([s != "XXX" for s in q]), f(mB[c]), np.nan) for b, c in (("J", "j_m"), ("H", "h_m"), ("K", "ks_m"))}
    sU, _ = S.sagitta_pms(f(mB["GLON"]), f(mB["GLAT"]), f(mB["Plx"]), f(mB["Gmag"]), f(mB["BPmag"]), f(mB["RPmag"]),
                          lst["J"], lst["H"], lst["K"])
    sU_full = np.zeros(len(d["y0"]), bool)
    sU_full[np.where(d["B"])[0]] = sU >= 0.6
    res["sagitta06_listed"] = gate_table("Sagitta(0,6) U listada", sU_full, d, d["B"])
    g1 = res["chosen"]["all"]["cp"][1] <= 0.10 and res["chosen"]["all"]["boot"][1] <= 0.10
    print(f"  G1 {'PASA' if g1 else 'FALLA'}: cota CP {res['chosen']['all']['cp'][1]:.4f}, boot {res['chosen']['all']['boot'][1]:.4f}")
    allold = d["old"]
    print(f"  (referencia, viejos completos A+B, sin compuerta: {key} FPR {np.mean(dec[allold]):.4f}; F2 0,0973)")

    print("[C1] modelo final (todo el entrenamiento) de la configuracion elegida", flush=True)
    mdl = fit_final(d, cfg)
    mdl.save_model(C / f"exp2_model_{key}.json")

    h = Table.read(C / "ngc6383_hr23.ecsv")[0]
    MOD83, AV83 = float(h["MOD50"]), float(h["AV50"])
    print(f"[G2-sint] sintetico MIST a la MOD {MOD83:.3f} y A_V {AV83:.3f} de NGC 6383, A_V por estrella ~ N(A_V, 0,5); n por tramo 2000")
    synth_ok = {}
    for kind in ("ms", "pms"):
        for binary in (False, True):
            ob = synth(kind, 2000, MOD83, AV83, binary)
            X = feats(cfg, ob["G"], ob["BP"], ob["RP"], ob["J"], ob["H"], ob["Ks"], MOD83, AV83)
            pp = mdl.predict_proba(X)[:, 1]
            dd = E.decide(pp, X[:, 0], th)
            k = E.tramo_index(X[:, 0])
            fr = [float(np.mean(dd[k == i])) if np.any(k == i) else np.nan for i in range(len(E.TRAMOS))]
            lab = "FPR (MS log t 9,0)" if kind == "ms" else "recall (PMS log t 6,6)"
            print(f"  {lab:24s} {'binarias q=1' if binary else 'simples     '}: " +
                  "  ".join(f"{nm} {v:.3f}" for nm, v in zip(E.TRAMO_NAMES, fr)), flush=True)
            if kind == "ms" and not binary:
                synth_ok = {nm: bool(v <= 0.10) for nm, v in zip(E.TRAMO_NAMES, fr)}
    est = {nm: res["chosen"][nm]["est"] and synth_ok.get(nm, False) for nm in E.TRAMO_NAMES}
    print(f"  TRAMOS ESTABLECIDOS (G2: boot B <= 0,10 y sintetico <= 0,10): {[nm for nm, v in est.items() if v]}")

    # ------------------------------------------------------------ G3 NGC 6530
    cl = Table.read(C / "hr23_clusters_sel.ecsv")
    r = cl[cl["Name"] == "NGC_6530"][0]
    MOD, AV, LOGT = float(r["MOD50"]), float(r["AV50"]), float(r["logAge50"])
    g = Table.read(C / "ngc6530_dr3.ecsv")
    GAL = SkyCoord(f(g["RAdeg"]) * u.deg, f(g["DEdeg"]) * u.deg).galactic
    q6 = np.array([str(x) if (x is not None and not np.ma.is_masked(x) and len(str(x)) == 3) else "XXX" for x in g["ph_qual"]])
    J = core.nir_clean(f(g["j_m"]), [s[0] for s in q6])
    H = core.nir_clean(f(g["h_m"]), [s[1] for s in q6])
    K = core.nir_clean(f(g["ks_m"]), [s[2] for s in q6])
    G, BP, RP, plx = f(g["Gmag"]), f(g["BPmag"]), f(g["RPmag"]), f(g["Plx"])
    ok = np.isfinite(G) & np.isfinite(BP) & np.isfinite(RP)
    ydef = core.label(G, MOD, AV, LOGT)
    X6 = feats(cfg, G, BP, RP, J, H, K, MOD, AV)
    p6 = mdl.predict_proba(X6)[:, 1]
    d6 = E.decide(p6, X6[:, 0], th)
    has = np.array([s != "XXX" for s in q6])
    L = {b: np.where(has, f(g[c]), np.nan) for b, c in (("J", "j_m"), ("H", "h_m"), ("K", "ks_m"))}
    s6, _ = S.sagitta_pms(GAL.l.deg, GAL.b.deg, plx, G, BP, RP, J, H, K)
    s6L, _ = S.sagitta_pms(GAL.l.deg, GAL.b.deg, plx, G, BP, RP, L["J"], L["H"], L["K"])
    X6L = feats(cfg, G, BP, RP, L["J"], L["H"], L["K"], MOD, AV)
    d6L = E.decide(mdl.predict_proba(X6L)[:, 1], X6L[:, 0], th)
    sel = ok & ydef
    n6 = int(sel.sum())
    print(f"[G3] NGC 6530 (Wright+2019, etiqueta PMS, n {n6}); NGC 6530 es in-sample para Sagitta")
    for nm, pr in (("elegida", d6), ("elegida U listada", d6L), ("Sagitta(0,6) U->token", s6 >= .6),
                   ("Sagitta(0,6) U listada", s6L >= .6)):
        x = int(pr[sel].sum())
        print(f"  {nm:24s} {x}/{n6} = {x / n6:.3f} {np.round(cp(x, n6), 3).tolist()}")
    k6 = E.tramo_index(X6[:, 0])
    for i, nm in enumerate(E.TRAMO_NAMES):
        s = sel & (k6 == i)
        print(f"    {nm:8s} n {s.sum():4d}: elegida {np.mean(d6[s]):.3f}  Sagitta(0,6) token {np.mean(s6[s] >= .6):.3f}  "
              f"listada {np.mean(s6L[s] >= .6):.3f}")
    rec_t = [float(np.mean(d6[sel & (k6 == i)])) for i in range(len(E.TRAMOS))]
    print(f"  recall de la elegida reponderado a NGC 6383 (pesos {E.W6383.tolist()}): {np.dot(E.W6383, rec_t):.3f}")
    idx = np.where(sel)[0]
    a, b = d6[idx].astype(float), (s6[idx] >= .6).astype(float)
    dd = [np.mean(a[s] - b[s]) for s in (RNG.integers(0, len(idx), len(idx)) for _ in range(2000))]
    lo, hi = np.percentile(dd, [2.5, 97.5])
    g3 = not (hi < -0.05)
    print(f"  diferencia pareada elegida - Sagitta(0,6) token: {np.mean(a - b):+.3f} [{lo:+.3f}; {hi:+.3f}] -> G3 "
          f"{'pasa' if g3 else 'FALLA'}")

    # ------------------------------------------------------------ G4 ablacion de NIR
    t83 = Table.read(Path.home() / "phd/agent-findings/scripts/virac2_ngc6383_members_clean.ecsv")
    q83 = np.array([str(x) if (x is not None and not np.ma.is_masked(x) and len(str(x)) == 3) else "XXX"
                    for x in t83["tmass_ph_qual"]])
    miss83 = np.stack([~np.isfinite(core.nir_clean(f(t83[f"{bd}_2mass"]), [s[i] for s in q83]))
                       for i, bd in enumerate(("J", "H", "Ks"))], 1)
    G83 = f(t83["G"])
    gb, gb83 = np.clip(np.floor(G), 8, 20), np.clip(np.floor(G83), 8, 20)
    fl = []
    for _ in range(50):
        Jm, Hm, Km = J.copy(), H.copy(), K.copy()
        for i in np.where(ok)[0]:
            pool = np.where(gb83 == gb[i])[0]
            if not len(pool):
                pool = np.argsort(np.abs(G83 - G[i]))[:5]
            mk = miss83[RNG.choice(pool)]
            Jm[i] = np.nan if mk[0] else Jm[i]
            Hm[i] = np.nan if mk[1] else Hm[i]
            Km[i] = np.nan if mk[2] else Km[i]
        Xa = feats(cfg, G, BP, RP, Jm, Hm, Km, MOD, AV)
        da = E.decide(mdl.predict_proba(Xa)[:, 1], Xa[:, 0], th)
        fl.append(np.mean(da[ok] != d6[ok]))
    g4 = np.mean(fl) <= 0.15
    print(f"[G4] ablacion de NIR (patron de NGC 6383, 50 sorteos): cambia de clase {np.mean(fl):.3f} (max {np.max(fl):.3f}) -> "
          f"{'pasa' if g4 else 'FALLA'}")
    verdict = g1 and g3 and g4
    print(f"[VEREDICTO §2.4] G1 {'pasa' if g1 else 'FALLA'}, G3 {'pasa' if g3 else 'FALLA'}, G4 {'pasa' if g4 else 'FALLA'} -> "
          f"{'EL ARREGLO SIRVE' if verdict else 'EL ARREGLO NO SIRVE (falsificador)'}; tramos establecidos "
          f"{[nm for nm, v in est.items() if v]}")
    json.dump(dict(key=key, cfg=cfg, th=th.tolist(), established=est, verdict=bool(verdict), gates=res,
                   rec6530_tramo=rec_t), open(C / "exp2_confirm.json", "w"), indent=1, default=float)
