"""Baja el conjunto de entrenamiento del clasificador PMS no circular (prerregistro: ~/phd/agent-findings/pms-classifier-non-circular.md §0.3).

Fuentes:
  - Hunt & Reffert 2023 (2023A&A...673A.114H), TAPVizieR J/A+A/673/A114/{clusters,members}. COUNT(*) medido 2026-10-11:
    clusters 7167, members 1 291 929. maxrec explicito en cada consulta (skill astro-data-access: truncado silencioso).
  - 2MASS PSC por CDS X-Match a 1" con epoca llevada a 2000 (ver fetch_tmass; el cruce oficial de Gaia no respondio). Antes:
    designation), subiendo los source_id por lotes a la TAP de ESA (launch_job_async; el sincrono trunca a 2000 sin aviso).

Seleccion (declarada antes de bajar):
  Type='o', CST>5, dist50 300-3000 pc, AV50<4. Todos los logAge50<7.8; de los mas viejos 25 % al azar (semilla 20261011).
  Excluye: < 3 deg de NGC 6383; < 6 deg con |plx-0.886|<0.15 y |pm-(2.53,-1.70)|<1 (Sh 2-012 / cinematica de NGC 6383);
  < 2 deg de NGC 6530 (validacion, se baja aparte con su etiqueta 'val_ngc6530'). Miembros Prob >= 0.7.

Env: ~/miniforge3/envs/erotica (astroquery 0.4.11, pyvo 1.9.1). Corre:
  cd ~/erotica-wt-pms/tools/pms_classifier && nice -n 19 ~/miniforge3/envs/erotica/bin/python fetch_training.py
Salida: cache/ (no versionada): hr23_clusters_sel.ecsv, hr23_members.ecsv, tmass.ecsv, fetch_log.txt
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pyvo
from astropy.coordinates import SkyCoord
from astropy.table import Table, unique, vstack
import astropy.units as u

HERE = Path(__file__).parent
C = HERE / "cache"
C.mkdir(exist_ok=True)
LOG = open(C / "fetch_log.txt", "a")
VZ = pyvo.dal.TAPService("https://tapvizier.cds.unistra.fr/TAPVizieR/tap")

N6383 = SkyCoord(263.673 * u.deg, -32.577 * u.deg)
N6530 = SkyCoord(271.102 * u.deg, -24.370 * u.deg)
PLX6383, PM6383 = 0.886, (2.53, -1.70)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def vz(q, maxrec=2_000_000, tries=5):
    for k in range(tries):
        try:
            return VZ.run_async(q, maxrec=maxrec).to_table()
        except Exception as e:  # TAPVizieR "too busy": reintenta
            log(f"  reintento {k + 1}: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(10 * (k + 1))
    raise RuntimeError(q[:200])


def select_clusters():
    t = vz("""SELECT Name, ID, N, RA_ICRS, DE_ICRS, GLON, GLAT, dist50, MOD50, logAge16, logAge50, logAge84, AV50, diffAV50,
                     CST, Plx, pmRA, pmDE, Type
              FROM "J/A+A/673/A114/clusters" WHERE Type='o' AND CST>5 AND dist50 BETWEEN 300 AND 3000 AND AV50 < 4""",
           maxrec=100000)
    log(f"HR23 clusters con Type=o CST>5 300-3000 pc AV<4: {len(t)}")
    sc = SkyCoord(np.asarray(t["RA_ICRS"]) * u.deg, np.asarray(t["DE_ICRS"]) * u.deg)
    d6383, d6530 = sc.separation(N6383).deg, sc.separation(N6530).deg
    kin = ((d6383 < 6) & (np.abs(np.asarray(t["Plx"]) - PLX6383) < 0.15)
           & (np.hypot(np.asarray(t["pmRA"]) - PM6383[0], np.asarray(t["pmDE"]) - PM6383[1]) < 1.0))
    excl = (d6383 < 3) | kin
    val = (d6530 < 2) & ~excl
    log(f"excluidos por NGC 6383/Sh 2-012: {excl.sum()} -> {sorted(np.asarray(t['Name'][excl]).tolist())}")
    log(f"apartados para validacion (NGC 6530 < 2 deg): {val.sum()} -> {sorted(np.asarray(t['Name'][val]).tolist())}")
    rng = np.random.default_rng(20261011)
    old = np.asarray(t["logAge50"]) >= 7.8
    keep_old = rng.random(len(t)) < 0.25
    train = ~excl & ~val & (~old | keep_old)
    t["role"] = np.where(val, "val_ngc6530", np.where(train, "train", "drop"))
    t["sep_ngc6383"] = d6383
    sel = t[t["role"] != "drop"]
    log(f"seleccion: train {np.sum(sel['role'] == 'train')} cumulos (jovenes <7.8: "
        f"{np.sum((sel['role'] == 'train') & (sel['logAge50'] < 7.8))}), val {np.sum(sel['role'] == 'val_ngc6530')}; "
        f"N declarado {int(np.sum(sel['N']))}")
    sel.write(C / "hr23_clusters_sel.ecsv", overwrite=True)
    return sel


def fetch_members(sel):
    out = C / "hr23_members.ecsv"
    if out.exists():
        return Table.read(out)
    names = list(np.asarray(sel["Name"]))
    parts = []
    for i in range(0, len(names), 60):
        chunk = names[i:i + 60]
        inl = ",".join("'" + n.replace("'", "''") + "'" for n in chunk)
        q = f"""SELECT Name, GaiaDR3, Prob, RA_ICRS, DE_ICRS, GLON, GLAT, Plx, e_Plx, pmRA, pmDE, RUWE, FG, e_FG, FBP, e_FBP, FRP,
                       e_FRP, Gmag, BPmag, RPmag FROM "J/A+A/673/A114/members" WHERE Prob >= 0.7 AND Name IN ({inl})"""
        cnt = vz(f"""SELECT COUNT(*) AS n FROM "J/A+A/673/A114/members" WHERE Prob >= 0.7 AND Name IN ({inl})""")["n"][0]
        t = vz(q)
        log(f"  miembros lote {i // 60}: {len(t)} filas (COUNT {cnt})")
        if len(t) != cnt:
            raise RuntimeError(f"truncado: {len(t)} != {cnt}")
        parts.append(t)
    m = vstack(parts)
    m["GaiaDR3"] = np.asarray(m["GaiaDR3"], dtype=np.int64)
    m.write(out, overwrite=True)
    log(f"miembros Prob>=0.7: {len(m)}; source_id unicos {len(np.unique(m['GaiaDR3']))}")
    return m


def fetch_tmass(m):
    """2MASS PSC por CDS X-Match (vizier:II/246/out), radio 1''. La TAP de ESA (cruce oficial best_neighbour) corto las subidas
    con ConnectionResetError y no respondio ni a un TOP 3 el 2026-10-11: se cambio de camino y se declara. Cada miembro se lleva
    de la epoca de Gaia DR3 (2016,0) a 2000,0 con su movimiento propio HR23 antes de cruzar (2MASS: 1997-2001); se queda el mas
    cercano. Qfl = ph_qual."""
    from astroquery.xmatch import XMatch
    out = C / "tmass.ecsv"
    if out.exists():
        return Table.read(out)
    sid = np.asarray(m["GaiaDR3"], np.int64)
    _, first = np.unique(sid, return_index=True)
    ra, de = np.asarray(m["RA_ICRS"], float)[first], np.asarray(m["DE_ICRS"], float)[first]
    pmr = np.nan_to_num(np.ma.filled(np.ma.asarray(m["pmRA"]).astype(float), 0.0))[first]
    pmd = np.nan_to_num(np.ma.filled(np.ma.asarray(m["pmDE"]).astype(float), 0.0))[first]
    dt = 2000.0 - 2016.0
    up = Table({"source_id": sid[first], "ra2000": ra + pmr * dt / 3.6e6 / np.cos(np.radians(de)),
                "de2000": de + pmd * dt / 3.6e6})
    raw = C / "tmass_xmatch_raw.ecsv"
    parts = [Table.read(raw)] if raw.exists() else []
    for i in ([] if parts else range(0, len(up), 40000)):
        for k in range(5):
            try:
                r = XMatch.query(cat1=up[i:i + 40000], cat2="vizier:II/246/out", max_distance=1 * u.arcsec,
                                 colRA1="ra2000", colDec1="de2000")
                break
            except Exception as e:
                log(f"  reintento X-Match {k + 1}: {type(e).__name__}: {str(e)[:150]}")
                time.sleep(20 * (k + 1))
        else:
            raise RuntimeError("X-Match")
        log(f"  2MASS X-Match lote {i // 40000}: {len(r)} pares para {len(up[i:i + 40000])} fuentes")
        parts.append(r)
    t = vstack(parts)
    if not raw.exists():
        t.write(raw, overwrite=True)
    log(f"  columnas X-Match: {t.colnames}")
    t = t[np.lexsort((np.asarray(t["angDist"], float), np.asarray(t["source_id"], np.int64)))]
    nmult = len(t) - len(np.unique(np.asarray(t["source_id"], np.int64)))
    t = unique(t, keys="source_id", keep="first")
    keep = ["source_id", "angDist", "2MASS", "Jmag", "e_Jmag", "Hmag", "e_Hmag", "Kmag", "e_Kmag", "Qfl", "MeasureJD"]
    t = t[[c for c in keep if c in t.colnames]]
    t.rename_columns(["Jmag", "Hmag", "Kmag", "Qfl"], ["j_m", "h_m", "ks_m", "ph_qual"])
    t.write(out, overwrite=True)
    log(f"2MASS: {len(t)} de {len(up)} source_id con contraparte a <= 1''; pares extra descartados (no el mas cercano) {nmult}")
    return t


if __name__ == "__main__":
    log(f"=== fetch_training {time.strftime('%Y-%m-%d %H:%M:%S')}")
    sel = select_clusters()
    m = fetch_members(sel)
    fetch_tmass(m)
