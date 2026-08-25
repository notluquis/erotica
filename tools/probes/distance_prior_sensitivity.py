#!/usr/bin/env python3
"""D5 + D7: las cifras que el manuscrito imprime sobre el prior de ``sigma_r``, reproducibles.

El manuscrito afirma ser *"a released, reusable pipeline"*, y esa frase es justo la razón por la que
D7 importaba. Las cifras que métodos imprime sobre la sensibilidad al prior se midieron primero en
un scratchpad de sesión: **reproducían, tenían las unidades buenas, y nada en el repositorio las
podía volver a producir**. Eso es la misma clase de agujero que F2, creada dentro del arreglo de F2.
Este fichero es el arreglo: un árbitro corre esto y obtiene los números del párrafo.

Tres mediciones, y el orden importa porque la tercera es el control de la segunda:

1. **D7 — los dos priores.** El del manuscrito, ``s_r`` = ``nanstd`` de las distancias ajustadas,
   contra el del paquete liberado, 0,05 kpc fijo. El manuscrito describe el primero; quien instale
   el paquete hoy obtiene el segundo.
2. **D5 — el barrido.** Siete escalas de 0,025 a 0,20 kpc, ocho veces, que enmarcan a las dos.
3. **El nulo.** Misma escala, cuatro semillas. Sin él, el rango del barrido podría ser ruido del
   muestreador y no respuesta al prior — y las dos cosas se ven idénticas en una tabla.

⚠ Lo que se publica **no** es la media de ``mu_r``. El par impreso es el **modo de la predictiva
Gamma** y ``std_r``; la media de ``mu_r`` es 1,1168 y redondea a 1,12, que no aparece en el paper.
Por eso la salida imprime los tres y no sólo el que uno esperaría.

Requiere el extra ``bayes``. Unos ~15 s por ajuste; doce ajustes.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

RAIZ = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ))

import astropy.units as u  # noqa: E402
from astropy.table import QTable, Table  # noqa: E402

from erotica.analysis.inference import (  # noqa: E402
    DistancePriors,
    SamplingConfig,
    distance_model,
)

NGC = RAIZ / "data/test/NGC6383"
# La ruta y el corte del ajuste publicado, iguales a `comments_paper/review_repo/convergence_audit.py`.
REF = NGC / "comments_paper/radius_robustness/generated/40/paperfaithful_reference_p06.ecsv"
CLU = NGC / "data/40/clustering_results.ecsv"
ESCALAS = (0.025, 0.0375, 0.05, 0.075, 0.10, 0.15, 0.20)
SEMILLAS = (42, 7, 2024, 99)


def distancias() -> np.ndarray:
    ref = Table.read(REF)
    sub = ref[np.abs(ref["parallax_error"] / ref["parallax"]) < 0.1]
    cl = Table.read(CLU)
    bj = pd.DataFrame(
        {
            "source_id": np.asarray(cl["source_id"]),
            "r_med_geo": np.asarray(cl["r_med_geo"], dtype=float),
        }
    )
    rg = (
        pd.DataFrame({"source_id": np.asarray(sub["source_id"])})
        .merge(bj, on="source_id", how="left")["r_med_geo"]
        .values
    )
    return rg[np.isfinite(rg)] / 1000.0  # pc -> kpc


def ajusta(r: np.ndarray, escala: float, semilla: int) -> dict[str, float]:
    cfg = SamplingConfig(
        draws=2000,
        tune=2000,
        target_accept=0.9,
        chains=4,
        random_seed=semilla,
        nuts_sampler="pymc",
        progressbar=False,
        extra_kwargs={"cores": 1},
    )
    t = QTable()
    t["r_med_geo"] = r * u.kpc
    out = distance_model(
        t, return_trace=True, sampling=cfg, priors=DistancePriors(sigma_scale=escala)
    )
    post = out.trace.posterior
    mu = post["mu_r"].values.ravel()
    sd = post["std_r"].values.ravel()
    # El modo de la predictiva Gamma, que es la mitad izquierda del `1.11 +- 0.06` publicado.
    # Analitico y no por KDE: medido, el modo por KDE se aleja del analitico 0,0045-0,0152 kpc
    # segun cuantos draws se le den, o sea mas que cualquiera de los efectos que esto mide.
    k = (mu / sd) ** 2
    th = sd**2 / mu
    return {
        "mu_r": float(mu.mean()),
        "mu_r_err": float(mu.std()),
        "std_r": float(sd.mean()),
        "modo_predictiva": float(np.median((k - 1) * th)),
        "divergencias": int(out.trace.sample_stats.diverging.sum()),
    }


def main() -> int:
    r = distancias()
    print(f"N = {r.size} distancias BJ, nanstd = {np.nanstd(r):.5f} kpc\n")

    print("1. D7 -- los dos priores")
    dos = {}
    for nombre, esc in (("manuscrito", float(np.nanstd(r))), ("liberado", 0.05)):
        dos[nombre] = ajusta(r, esc, 42)
        d = dos[nombre]
        print(
            f"   {nombre:11s} escala={esc:.5f}  mu_r={d['mu_r']:.5f}+-{d['mu_r_err']:.5f}  "
            f"std_r={d['std_r']:.5f}  modo_pred={d['modo_predictiva']:.5f}  div={d['divergencias']}"
        )
    d_mu = abs(dos["manuscrito"]["mu_r"] - dos["liberado"]["mu_r"])
    d_sd = abs(dos["manuscrito"]["std_r"] - dos["liberado"]["std_r"])
    print(f"   -> |d mu_r| = {d_mu:.5f} kpc, |d std_r| = {d_sd:.5f} kpc")

    print(
        f"\n2. D5 -- barrido de {ESCALAS[0]} a {ESCALAS[-1]} kpc ({ESCALAS[-1] / ESCALAS[0]:.0f}x)"
    )
    barrido = [ajusta(r, e, 42) for e in ESCALAS]
    sd = np.array([b["std_r"] for b in barrido])
    mo = np.array([b["modo_predictiva"] for b in barrido])
    for e, b in zip(ESCALAS, barrido, strict=True):
        print(f"   escala={e:.4f}  std_r={b['std_r']:.5f}  modo_pred={b['modo_predictiva']:.5f}")
    rango_sd = float(sd.max() - sd.min())
    rango_prior = ESCALAS[-1] - ESCALAS[0]
    print(
        f"   -> std_r recorre {rango_sd:.5f} kpc = {100 * rango_sd / sd.mean():.1f}% de su valor "
        f"y {100 * rango_sd / rango_prior:.1f}% del rango barrido"
    )
    print(
        f"   -> el modo de la predictiva redondea a {sorted({f'{x:.2f}' for x in mo})} en las {len(ESCALAS)} escalas"
    )

    print(f"\n3. El nulo -- misma escala (0,05), {len(SEMILLAS)} semillas")
    nulo = np.array([ajusta(r, 0.05, s)["std_r"] for s in SEMILLAS])
    rango_nulo = float(nulo.max() - nulo.min())
    print(f"   {', '.join(f'{x:.5f}' for x in nulo)}")
    print(
        f"   -> rango del nulo {rango_nulo:.5f} kpc; el del barrido es {rango_sd / max(rango_nulo, 1e-9):.0f}x mayor"
    )

    fallos = []
    if rango_sd <= rango_nulo:
        fallos.append(
            "el barrido no supera al nulo: `0,0013` seria ruido del muestreador, no el prior"
        )
    if len({f"{x:.2f}" for x in mo}) != 1:
        fallos.append("el modo de la predictiva ya no redondea igual en todas las escalas")
    if any(b["divergencias"] for b in barrido):
        fallos.append("hubo divergencias: los numeros no son de un posterior fiable")
    for f in fallos:
        print(f"\n  - {f}")
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
