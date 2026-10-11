"""Validacion externa: NGC 6530 (Laguna), miembros jovenes confirmados de Wright+2019 (2019MNRAS.486.2477W, VizieR
J/MNRAS/486/2477/table, 819 filas; seleccion por diagnosticos espectroscopicos GES -- gravedad, Li -- y rayos X). Es VERDAD
independiente de la etiqueta del clasificador (que sale de la isocrona a la edad HR23). Solo trae miembros: mide recall.

Camino (la TAP de ESA no respondio el 2026-10-11; se usa CDS X-Match, el mismo servicio que el entrenamiento):
  Wright (posiciones Gaia DR2, ep. 2015,5) -> vizier:I/355/gaiadr3 a <= 0,5'', y se exige que el DR3 cruzado sea el mismo objeto:
  |G_DR3 - G_Wright| < 0,3 mag; se queda el mas cercano. Luego 2MASS (vizier:II/246/out) a <= 1'' con la posicion DR3 llevada a
  2000,0 por su movimiento propio, igual que fetch_training.fetch_tmass.
Env: ~/miniforge3/envs/erotica. Corre: nice -n 19 ~/miniforge3/envs/erotica/bin/python fetch_validation.py
Salida: cache/wright2019.ecsv, cache/ngc6530_dr3.ecsv
"""
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.table import Table, join, unique
from astroquery.vizier import Vizier
from astroquery.xmatch import XMatch

C = Path(__file__).parent / "cache"
C.mkdir(exist_ok=True)

wp = C / "wright2019.ecsv"
if wp.exists():
    w = Table.read(wp)
else:
    w = Vizier(columns=["**"], row_limit=-1).get_catalogs("J/MNRAS/486/2477/table")[0]
    w.write(wp, overwrite=True)
print("Wright+2019 filas:", len(w), "(esperado 819)")
up = Table({"recno": np.asarray(w["recno"]), "ra": np.asarray(w["RAJ2000"], float), "dec": np.asarray(w["DEJ2000"], float),
            "Gw": np.ma.filled(np.ma.asarray(w["Gmag"]).astype(float), np.nan)})
x = XMatch.query(cat1=up, cat2="vizier:I/355/gaiadr3", max_distance=0.5 * u.arcsec, colRA1="ra", colDec1="dec")
print("pares Wright-DR3 a <= 0,5'':", len(x), "; columnas:", x.colnames[:40])
gcol = "Gmag" if "Gmag" in x.colnames else [c for c in x.colnames if c.lower().startswith("gmag")][0]
dG = np.abs(np.asarray(x[gcol], float) - np.asarray(x["Gw"], float))
x = x[~(dG >= 0.3)]
x = x[np.lexsort((np.asarray(x["angDist"], float), np.asarray(x["recno"])))]
x = unique(x, keys="recno", keep="first")
print("un DR3 por miembro, |dG| < 0,3 (o sin G en Wright):", len(x))
ra, de = np.asarray(x["RAdeg"], float), np.asarray(x["DEdeg"], float)
pmr = np.nan_to_num(np.ma.filled(np.ma.asarray(x["pmRA"]).astype(float), 0.0))
pmd = np.nan_to_num(np.ma.filled(np.ma.asarray(x["pmDE"]).astype(float), 0.0))
dt = 2000.0 - 2016.0
u2 = Table({"recno": np.asarray(x["recno"]), "ra2000": ra + pmr * dt / 3.6e6 / np.cos(np.radians(de)),
            "de2000": de + pmd * dt / 3.6e6})
t = XMatch.query(cat1=u2, cat2="vizier:II/246/out", max_distance=1 * u.arcsec, colRA1="ra2000", colDec1="de2000")
t = t[np.lexsort((np.asarray(t["angDist"], float), np.asarray(t["recno"])))]
t = unique(t, keys="recno", keep="first")["recno", "Jmag", "Hmag", "Kmag", "Qfl"]
t.rename_columns(["Jmag", "Hmag", "Kmag", "Qfl"], ["j_m", "h_m", "ks_m", "ph_qual"])
print("con 2MASS a <= 1'':", len(t))
x.rename_column("angDist", "angDist_dr3")
g = join(x, t, keys="recno", join_type="left")
g.write(C / "ngc6530_dr3.ecsv", overwrite=True)
print("escrito cache/ngc6530_dr3.ecsv:", len(g), "filas; columnas", g.colnames)
