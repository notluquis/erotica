"""R38-08: pool de cumulos VIEJOS fuera de muestra (prerregistro: ~/phd/agent-findings/pms-classifier-non-circular.md §4).

Mismos cortes que fetch_training.py (Type='o', CST>5, AV50<4, exclusiones de NGC 6383/Sh 2-012 y de NGC 6530) y los de "viejo" del
segundo experimento (logAge50 >= 8,5; dist50 600-2500 pc), MENOS todo cumulo que este en cache/hr23_clusters_sel.ecsv (los que
fetch_training.py dejo en 'train' o 'val_ngc6530'). Esos son el 75 % de los viejos que el sorteo con semilla 20261011 descarto: ninguno
entro a ningun entrenamiento, ni al modelo final. Se toman TODOS (N fijado por la regla, no por el resultado).
Miembros Prob >= 0,7 y 2MASS por CDS X-Match (II/246/out, 1'', epoca 2000) con el mismo codigo.
Env: erotica. Corre: cd ~/erotica-wt-pms/tools/pms_classifier && nice -n 10 ~/miniforge3/envs/erotica/bin/python fetch_oos.py \
     > cache/fetch_oos.log 2>&1
Salida (cache/, no versionada): oos_clusters.ecsv, oos_members.ecsv, oos_tmass.ecsv
"""
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

import fetch_training as FT

C = Path(__file__).parent / "cache"
FT.LOG = open(C / "fetch_oos_log.txt", "a")

t = FT.vz("""SELECT Name, ID, N, RA_ICRS, DE_ICRS, GLON, GLAT, dist50, MOD50, logAge16, logAge50, logAge84, AV50, diffAV50,
                 CST, Plx, pmRA, pmDE, Type, AV16, AV84, MOD16, MOD84
          FROM "J/A+A/673/A114/clusters" WHERE Type='o' AND CST>5 AND dist50 BETWEEN 600 AND 2500 AND AV50 < 4
          AND logAge50 >= 8.5""", maxrec=100000)
FT.log(f"HR23 viejos candidatos (o, CST>5, 600-2500 pc, AV<4, logAge>=8,5): {len(t)}")
sc = SkyCoord(np.asarray(t["RA_ICRS"]) * u.deg, np.asarray(t["DE_ICRS"]) * u.deg)
d6383, d6530 = sc.separation(FT.N6383).deg, sc.separation(FT.N6530).deg
kin = ((d6383 < 6) & (np.abs(np.asarray(t["Plx"]) - FT.PLX6383) < 0.15)
       & (np.hypot(np.asarray(t["pmRA"]) - FT.PM6383[0], np.asarray(t["pmDE"]) - FT.PM6383[1]) < 1.0))
excl = (d6383 < 3) | kin | (d6530 < 2)
seen = set(np.asarray(Table.read(C / "hr23_clusters_sel.ecsv")["Name"]).astype(str))
inseen = np.array([str(n) in seen for n in t["Name"]])
FT.log(f"excluidos por posicion/cinematica {excl.sum()}; ya usados (train/val) {inseen.sum()}")
t = t[~excl & ~inseen]
t["role"] = "oos_old"
FT.log(f"pool fuera de muestra: {len(t)} cumulos, N declarado {int(np.sum(t['N']))}")
t.write(C / "oos_clusters.ecsv", overwrite=True)

# reutiliza los descargadores con rutas propias
import fetch_training
orig = fetch_training.C
mfile, tfile, rawfile = C / "oos_members.ecsv", C / "oos_tmass.ecsv", C / "oos_tmass_xmatch_raw.ecsv"
if mfile.exists():
    m = Table.read(mfile)
else:
    import shutil, tempfile
    tmp = Path(tempfile.mkdtemp(dir=C))
    fetch_training.C = tmp
    m = fetch_training.fetch_members(t)
    shutil.move(tmp / "hr23_members.ecsv", mfile)
    fetch_training.C = orig
if not tfile.exists():
    import shutil, tempfile
    tmp = Path(tempfile.mkdtemp(dir=C))
    fetch_training.C = tmp
    fetch_training.fetch_tmass(m)
    shutil.move(tmp / "tmass.ecsv", tfile)
    shutil.move(tmp / "tmass_xmatch_raw.ecsv", rawfile)
    fetch_training.C = orig
FT.log("fetch_oos listo")
