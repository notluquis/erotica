"""R38 (prerregistro hub §4.1 d y §4.2): reentrena las 5 particiones GroupKFold de la configuracion elegida (T0P1K0F0) exactamente
como exp2_grid.oof_for, guarda los modelos y CONTROLA que reproducen cache/exp2_oof.npz (max |dp| < 1e-5); si no, se detiene.
Env: erotica. Corre: cd ~/erotica-wt-pms/tools/pms_classifier && OMP_NUM_THREADS=4 nice -n 10 ~/miniforge3/envs/erotica/bin/python \
     r38_folds.py > cache/r38_folds.log 2>&1
Salida: cache/r38_fold{k}.json, cache/r38_folds.npz (indice de particion por estrella)
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

import exp2_common as E
import exp2_grid as GR

C = Path("cache")
d = GR.prepare()
cfg = dict(T="T0", P="P1", K="K0", F="F0")
y, stab = d["tg"]["T0"]
w = E.config_weights(cfg, y, stab, d["dobs"], d["dth"], d["logt"])
ref = np.load(C / "exp2_oof.npz")["T0P1K0F0"].astype(float)
fold = np.full(len(y), -1)
oof = np.full(len(y), np.nan)
for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(d["X"], y, d["grp"])):
    mdl = GR.fit_any(d["X"][tr], y[tr], seed=20261011 + k, w=w[tr], nfeat=d["X"].shape[1])
    oof[te] = mdl.predict_proba(d["X"][te])[:, 1]
    fold[te] = k
    mdl.save_model(C / f"r38_fold{k}.json")
    print(f"fold {k}: max |p - p_ref| = {np.max(np.abs(oof[te] - ref[te])):.2e}", flush=True)
np.savez_compressed(C / "r38_folds.npz", fold=fold, sid=d["sid"])
dmax = np.max(np.abs(oof - ref))
print(f"CONTROL: max |p - p_ref| global = {dmax:.2e}")
if dmax > 1e-5:
    sys.exit("CONTROL FALLA: las particiones no reproducen exp2_oof.npz")
