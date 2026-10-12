"""El 116 de P01 (Sagitta 1.2.3 sobre el CDS de P01, umbral 0,6) con las `U` de 2MASS recodificadas, TODO LO DEMAS FIJO.

Entradas: CDS final de P01 (~/paper-ngc6383-aa52082-24/cds_final/ngc6383_members.ecsv, Ref = 1; SOLO LECTURA) y `tmass_ph_qual` de la
tabla limpia (~/phd/agent-findings/scripts/virac2_ngc6383_members_clean.ecsv), emparejadas por GaiaDR3.
Control: con las magnitudes del CDS tal cual debe dar 116 (= el oraculo de sagitta_ref.py); y se verifica que la magnitud del CDS es
la de 2MASS de la tabla limpia (si no, la recodificacion no seria "todo lo demas fijo").
Brazos: (0) CDS tal cual (U como valor) ; (1) U -> faltante (token 18) por banda ; (2) U -> faltante solo si ademas el error es nulo.
Env: erotica. Corre: SAGITTA_SRC=~/.local/lib/sagitta-1.2.3 OMP_NUM_THREADS=2 ~/miniforge3/envs/erotica/bin/python r38_p01_u.py \
     > r38_p01_u_measured.txt
"""
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

import sagitta_ref as S

f = lambda t, k: np.ma.filled(np.ma.asarray(t[k]).astype(float), np.nan)  # noqa: E731
cds = Table.read(Path.home() / "paper-ngc6383-aa52082-24/cds_final/ngc6383_members.ecsv")
cds = cds[cds["Ref"] == 1]
cds = cds[np.argsort(np.asarray(cds["GaiaDR3"], np.int64))]
tab = Table.read(Path.home() / "phd/agent-findings/scripts/virac2_ngc6383_members_clean.ecsv")
tab = tab[np.argsort(np.asarray(tab["GaiaDR3"], np.int64))]
assert np.all(np.asarray(cds["GaiaDR3"], np.int64) == np.asarray(tab["GaiaDR3"], np.int64))
q = np.array([str(x) if (x is not None and not np.ma.is_masked(x) and len(str(x)) == 3) else "---" for x in tab["tmass_ph_qual"]])
sc = SkyCoord(f(cds, "RAdeg") * u.deg, f(cds, "DEdeg") * u.deg).galactic
plx, G, BP, RP = f(cds, "Plx"), f(cds, "Gmag"), f(cds, "BPmag"), f(cds, "RPmag")
M = {b: f(cds, c) for b, c in (("J", "Jmag"), ("H", "Hmag"), ("Ks", "Ksmag"))}
M2 = {b: f(tab, c) for b, c in (("J", "J_2mass"), ("H", "H_2mass"), ("Ks", "Ks_2mass"))}
E2 = {b: f(tab, c) for b, c in (("J", "eJ_2mass"), ("H", "eH_2mass"), ("Ks", "eKs_2mass"))}
print(f"n {len(cds)}; con J,H,Ks en el CDS: {int(np.sum(np.isfinite(M['J']) & np.isfinite(M['H']) & np.isfinite(M['Ks'])))}")
for i, b in enumerate(("J", "H", "Ks")):
    both = np.isfinite(M[b]) & np.isfinite(M2[b])
    isU = np.array([s[i] == "U" for s in q])
    print(f"  {b}: CDS finita {np.isfinite(M[b]).sum()}, 2MASS limpia {np.isfinite(M2[b]).sum()}, ambas {both.sum()}, "
          f"|CDS - 2MASS| max {np.nanmax(np.abs(M[b][both] - M2[b][both])):.4f}; ph_qual U {isU.sum()} (con magnitud en el CDS "
          f"{np.sum(isU & np.isfinite(M[b]))}; con error nulo en la limpia {np.sum(isU & ~np.isfinite(E2[b]))})")
p0, av = S.sagitta_pms(sc.l.deg, sc.b.deg, plx, G, BP, RP, M["J"], M["H"], M["Ks"])
print(f"[0] CDS tal cual (U como valor): {np.sum(p0 >= .6)} (control: 116); |PMSProb CDS - red| max "
      f"{np.nanmax(np.abs(p0 - f(cds, 'PMSProb'))):.4f}")
for tag, cond in (("U -> token", lambda i, b: np.array([s[i] == "U" for s in q])),
                  ("U y error nulo -> token", lambda i, b: np.array([s[i] == "U" for s in q]) & ~np.isfinite(E2[b]))):
    Mx = {b: np.where(cond(i, b), np.nan, M[b]) for i, b in enumerate(("J", "H", "Ks"))}
    p = S.predict_pms(plx, av, G, BP, RP, Mx["J"], Mx["H"], Mx["Ks"])
    anyU = np.array([("U" in s) for s in q])
    print(f"[{tag}] PMS >= 0,6: {np.sum(p >= .6)}; cambian a PMS {np.sum((p >= .6) & (p0 < .6))}, dejan de serlo "
          f"{np.sum((p < .6) & (p0 >= .6))}; estrellas con alguna U {anyU.sum()}")
    for thr in (0.5, 0.6, 0.7):
        print(f"    umbral {thr}: {np.sum(p0 >= thr)} -> {np.sum(p >= thr)}")
# solo una banda a la vez
for i, b in enumerate(("J", "H", "Ks")):
    Mx = dict(M)
    Mx[b] = np.where(np.array([s[i] == "U" for s in q]), np.nan, M[b])
    print(f"  solo {b} U -> token: {np.sum(S.predict_pms(plx, av, G, BP, RP, Mx['J'], Mx['H'], Mx['Ks']) >= .6)}")

# Revision adversarial (M3): el CDS de P01 deja 61 filas sin JHKs; de ellas, las que tienen 2MASS en la tabla limpia se rellenan, y
# entonces la U entra como valor en TODAS las estrellas (la codificacion del entrenamiento) o como token.
fill = ~np.isfinite(M["J"]) & np.isfinite(M2["J"]) & np.isfinite(M2["H"]) & np.isfinite(M2["Ks"])
print(f"[M3] filas sin JHKs en el CDS con 2MASS en la tabla limpia: {fill.sum()} (de ellas con alguna U {np.sum(fill & np.array(['U' in s for s in q]))})")
Mv = {b: np.where(fill, M2[b], M[b]) for b in ("J", "H", "Ks")}
pv = S.predict_pms(plx, av, G, BP, RP, Mv["J"], Mv["H"], Mv["Ks"])
Mt = {b: np.where(np.array([s[i] == "U" for s in q]), np.nan, Mv[b]) for i, b in enumerate(("J", "H", "Ks"))}
pt = S.predict_pms(plx, av, G, BP, RP, Mt["J"], Mt["H"], Mt["Ks"])
print(f"  2MASS completo, U como valor: {np.sum(pv >= .6)} | U -> token: {np.sum(pt >= .6)}")
