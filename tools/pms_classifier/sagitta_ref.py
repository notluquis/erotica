"""Sagitta 1.2.3 (McBride+2021, 2021AJ....162..282M) evaluado directo, como CONTROL del clasificador propio.

No se instala Sagitta: se lee del sdist de PyPI extraido en $SAGITTA_SRC (sagitta-1.2.3.tar.gz,
sha256 5141b5b92a9a1eed10071acd6c79354e8c4095bbdacd90330f04d6a2e983660d, verificado al bajarlo 2026-10-11).
Los pesos se cargan con ``weights_only=True``. Mismas redes y normalizaciones que ``SagittaPipeline.predict_av/predict_pms``,
igual que ``~/phd/agent-findings/scripts/p01_sagitta_imputed_j.py``; faltantes: g 20, bp 21, rp 19, j/h/k 18 (el token que la
red aprendio), parallax 0.

``python sagitta_ref.py`` corre el ORACULO: reproducir el PMSProb y el AvSag del CDS de P01 (116 = 87 con JHKs + 29 sin).
Si no reproduce, ningun numero de Sagitta de este directorio sirve de control.
Env: ~/miniforge3/envs/erotica (torch 2.12.0 instalado con mamba el 2026-10-11 para esto).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

SRC = Path(os.environ.get("SAGITTA_SRC", "")).expanduser()
if not (SRC / "sagitta" / "model_code.py").exists():
    raise SystemExit("defina SAGITTA_SRC = directorio extraido de sagitta-1.2.3.tar.gz (ver docstring)")
sys.path.insert(0, str(SRC))

import torch  # noqa: E402
from sagitta.data_tools import DataTools  # noqa: E402
from sagitta.model_code import Sagitta  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
SDIR = SRC / "sagitta" / "model_state_dicts"
FILL = dict(parallax=0.0, av=5.0, g=20.0, bp=21.0, rp=19.0, j=18.0, h=18.0, k=18.0)


def _model(name, **kw):
    m = Sagitta(**kw)
    m.load_state_dict(torch.load(SDIR / name, map_location="cpu", weights_only=True))
    m.eval()
    return m


AV_NET = _model("av_model.pt", connectShape=40)
PMS_NET = _model("pms_model.pt")


def predict_av(l, b, plx):
    plx = np.where(np.isfinite(plx), plx, 0.0)
    x = np.stack([l / 360, b / 180 + 0.5, plx / 5], 1).astype(np.float32).reshape(-1, 1, 3)
    with torch.no_grad():
        return AV_NET(torch.from_numpy(x)).numpy().ravel() * 5


def predict_pms(plx, av, g, bp, rp, j, h, k):
    """Faltantes (NaN) -> el token de Sagitta, como ``DataTools.fix_and_mark_nans``."""
    cols = dict(parallax=plx, av=av, g=g, bp=bp, rp=rp, j=j, h=h, k=k)
    x = np.stack([DataTools.normalize_gaia(np.where(np.isfinite(cols[n]), cols[n], FILL[n]).astype(float), n)
                  for n in ("parallax", "av", "g", "bp", "rp", "j", "h", "k")], 1)
    out = []
    with torch.no_grad():
        for s in range(0, len(x), 20000):
            xb = torch.from_numpy(x[s:s + 20000].astype(np.float32)).reshape(-1, 1, 8)
            out.append(torch.sigmoid(PMS_NET(xb)).numpy().ravel())
    return np.concatenate(out) if out else np.array([])


def sagitta_pms(l, b, plx, g, bp, rp, j, h, k):
    av = predict_av(l, b, plx)
    return predict_pms(plx, av, g, bp, rp, j, h, k), av


if __name__ == "__main__":
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    import astropy.units as u

    cds = Table.read(Path.home() / "paper-ngc6383-aa52082-24/cds_final/ngc6383_members.ecsv")
    cds = cds[cds["Ref"] == 1]
    f = lambda k: np.ma.filled(np.ma.asarray(cds[k]).astype(float), np.nan)  # noqa: E731
    sc = SkyCoord(f("RAdeg") * u.deg, f("DEdeg") * u.deg).galactic
    p, av = sagitta_pms(sc.l.deg, sc.b.deg, f("Plx"), f("Gmag"), f("BPmag"), f("RPmag"), f("Jmag"), f("Hmag"), f("Ksmag"))
    has = np.isfinite(f("Jmag")) & np.isfinite(f("Hmag")) & np.isfinite(f("Ksmag"))
    dp = np.abs(p - f("PMSProb"))
    print(f"oraculo Sagitta: n = {len(p)}; |AvSag - red| max {np.nanmax(np.abs(av - f('AvSag'))):.4f}; "
          f"|PMSProb - red| max {dp.max():.4f}, mediana {np.median(dp):.5f}")
    print(f"PMS >= 0,6: CDS {np.sum(f('PMSProb') >= .6)}, aqui {np.sum(p >= .6)} = {np.sum(has & (p >= .6))} con JHKs "
          f"+ {np.sum(~has & (p >= .6))} sin  (esperado 116 = 87 + 29)")
