"""Carga el pool de viejos fuera de muestra (fetch_oos.py) con el mismo build() del entrenamiento. Sin predicciones."""
from pathlib import Path

import numpy as np
from astropy.table import Table, join

import core
import exp2_common as E
from train_validate import build, f

C = Path(__file__).parent / "cache"


def load_oos():
    cl = Table.read(C / "oos_clusters.ecsv")
    m = Table.read(C / "oos_members.ecsv")
    tm = Table.read(C / "oos_tmass.ecsv")
    tm.rename_column("source_id", "GaiaDR3")
    m = join(m, cl["Name", "MOD50", "AV50", "diffAV50", "dist50", "logAge16", "logAge50", "logAge84", "role"], keys="Name")
    m = join(m, tm["GaiaDR3", "ph_qual", "j_m", "h_m", "ks_m"], keys="GaiaDR3", join_type="left")
    X, y, ph = build(m)
    grp = np.asarray(m["Name"]).astype(str)
    old = ~y  # etiqueta 0 por §0.2 (en log t >= 8,5 casi todas)
    return dict(m=m, X=X, y=y, ph=ph, grp=grp, old=old, mg0=X[:, core.FEATURES.index("MG0")],
                av=f(m["AV50"]), dav=f(m["diffAV50"]), logt=f(m["logAge50"]), dist=f(m["dist50"]))
