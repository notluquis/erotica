#!/usr/bin/env python3
"""J.2(b)3: cuanto del `1,11 +- 0,06 kpc` publicado lo pone el catalogo y no el cumulo.

El ajuste publicado (codigo a 2026-07-19, `a2c6ab9`) era

    prior_mu_r = nanmean([nanmean(1/parallax), nanmean(r_med_geo)])
    mu_r  = Uniform(0.5*prior_mu_r, 1.5*prior_mu_r)
    std_r = HalfNormal(sigma=nanstd(r_med_geo))
    Gamma("r", mu=mu_r, sigma=std_r, observed=r_med_geo)

o sea una poblacion Gamma ajustada a los estimadores PUNTUALES de Bailer-Jones. §J.2(b)3 de
`methodology.md` dice que eso hace dos cosas a la vez, y son dos defectos con dos victimas:

  (P) doble-cuenta el prior galactico de Bailer-Jones  -> contamina `mu_r`, o sea el 1,11
  (E) descarta las incertidumbres por estrella          -> contamina `std_r`, o sea el +-0,06

Mezclarlas hace irrecuperable la atribucion, asi que T3 corre y se lee ANTES de interpretar T2:
T3 quita (E) y deja (P) en pie, de modo que la diferencia que mide T2 tiene un significado.

  T1  aritmetica, sin muestreo: la mediana de sigma_i = (r_hi_geo - r_lo_geo)/2 contra el
      std_r = 0,0600 publicado, y la particion en quadratura.
  T3  `distance_model` de hoy CON r_lo_geo/r_hi_geo -- la rama normal marginalizada de F3.
      Sigue ajustando distancias de Bailer-Jones: (P) sigue, (E) no.
  T2  `fit_parallax_model` de hoy con sigma_varpi por estrella y el cero nuisance de
      Maiz Apellaniz+2021 (10,3 uas). Espacio de paralaje: ni (P) ni (E).
  T5  el presupuesto de error de la media, que es lo que el ajuste publicado nunca calculo.
      Descompone la sigma(mu_varpi) de T2 en su parte estadistica y el suelo sistematico del cero
      residual (10,3 uas, Maiz Apellaniz+2021). NO es una critica al tratamiento del cero: la
      correccion de Lindegren+2021 ESTA aplicada -- se verifico que `parallax` = `parallax_observed`
      - `zpvals` exacto, con `zpvals` variando por estrella. Lo que T5 mide es que la INCERTIDUMBRE
      residual de esa correccion domina la incertidumbre correcta de la media, y que el
      Gamma-sobre-`r_med_geo` publicado no la contiene por ninguna parte: su +-0,06 es dispersion
      de catalogo, con cero sistematica de cero dentro.
  T4  el control que impide repetir el defecto dentro del arreglo (K.1.34). La clase de defecto
      que T1-T3 diagnostican es "una anchura reportada como cantidad fisica cuando es sobre todo
      error de medicion". Declarar la profundidad de T2/T3 sin aplicarle el mismo criterio seria
      cometerlo un nivel mas abajo. **La conclusion la carga la geometria, que es interna al
      paper**: el propio manuscrito mide R_t = 17,4 pc, y una esfera uniforme de radio R tiene
      sigma_LOS = R/sqrt(5) ~ 7,8 pc. Para un perfil concentrado -- que es lo que un King con
      r_c = 0,63 pc describe -- R/sqrt(5) es una COTA SUPERIOR de sigma_LOS, asi que el exceso
      que se imprime es un piso, no una estimacion.

      ⚠ Una segunda via se probo y se DEGRADO a nota, porque conflaba regimenes: la
      subestimacion de 10-20 % de los errores formales que mide Vasiliev & Baumgardt 2021
      (`2021MNRAS.505.5978V`) es para CUMULOS GLOBULARES EN CAMPOS APINADOS, y la propia §J.2
      marca ese caveat con todas las letras -- para cumulos abiertos la referencia es Riess+2022,
      -3 +/- 4 uas. Un binario O brillante dentro del campo no lo convierte en apinado. La
      coincidencia numerica se imprime como plausibilidad y NO sostiene ninguna conclusion.

Umbrales prerregistrados en `~/phd/agent-findings/p01-distance-from-bj-quantiles.md`, escritos
antes de correr esto. Este fichero falla si alguno deja de cumplirse.

Requiere el extra `bayes`. Unos ~20 s por ajuste; ocho ajustes.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ))

import astropy.units as u  # noqa: E402
from astropy.table import QTable, Table  # noqa: E402

from erotica.analysis.inference import (  # noqa: E402
    DistancePriors,
    ParallaxPriors,
    SamplingConfig,
    distance_model,
    fit_parallax_model,
)

NGC = RAIZ / "data/test/NGC6383"
# La misma ruta y el mismo corte que `distance_prior_sensitivity.py`, que es el ajuste publicado.
REF = NGC / "comments_paper/radius_robustness/generated/40/paperfaithful_reference_p06.ecsv"

# Lo que el manuscrito imprime, contra lo que se compara todo lo de abajo.
PUB_D, PUB_STD = 1.110, 0.0600  # kpc
SEMILLAS = (42, 7, 2024, 99)

#: R_t del paper (Sect. estructural), en pc. Es la unica escala fisica independiente del paralaje
#: con la que se puede confrontar una profundidad de linea de vision.
R_T_PC = 17.4
#: Techo de la subestimacion de errores formales que mide Vasiliev & Baumgardt 2021 -- para
#: CUMULOS GLOBULARES EN CAMPOS APINADOS. Se imprime como plausibilidad y no sostiene ninguna
#: conclusion; §J.2 marca explicitamente el caveat de regimen.
SUBESTIMACION = 0.20


def muestra() -> QTable:
    """Las 130 fuentes del ajuste publicado, con distancias y paralajes."""
    ref = Table.read(REF)
    sub = ref[np.abs(ref["parallax_error"] / ref["parallax"]) < 0.1]
    t = QTable()
    for col, unidad, escala in (
        ("r_med_geo", u.kpc, 1e-3),
        ("r_lo_geo", u.kpc, 1e-3),
        ("r_hi_geo", u.kpc, 1e-3),
        ("parallax", u.mas, 1.0),
        ("parallax_observed", u.mas, 1.0),
        ("parallax_error", u.mas, 1.0),
    ):
        t[col] = np.asarray(sub[col], dtype=float) * escala * unidad
    bueno = np.all([np.isfinite(np.asarray(t[c])) for c in t.colnames], axis=0)
    return t[bueno]


def cfg(semilla: int) -> SamplingConfig:
    return SamplingConfig(
        draws=2000,
        tune=2000,
        target_accept=0.9,
        chains=4,
        random_seed=semilla,
        nuts_sampler="pymc",
        progressbar=False,
        extra_kwargs={"cores": 1},
    )


def modo_gamma(mu: np.ndarray, sd: np.ndarray) -> float:
    """El modo de la predictiva Gamma: la mitad izquierda del par publicado.

    Analitico y no por KDE. Medido en D7: el modo por KDE se aleja del analitico 0,0045-0,0152 kpc
    segun cuantas extracciones se le den, o sea mas que cualquiera de los efectos que esto mide.
    """
    k = (mu / sd) ** 2
    return float(np.median((k - 1) * (sd**2 / mu)))


def t1(t: QTable) -> dict[str, float]:
    lo = np.asarray(t["r_lo_geo"].to_value(u.kpc), dtype=float)
    hi = np.asarray(t["r_hi_geo"].to_value(u.kpc), dtype=float)
    r = np.asarray(t["r_med_geo"].to_value(u.kpc), dtype=float)
    sig = (hi - lo) / 2.0
    med = float(np.median(sig))
    var_resto = PUB_STD**2 - med**2
    return {
        "mediana_sigma_i": med,
        "cuota_varianza": med**2 / PUB_STD**2,
        "profundidad_implicita": float(np.sqrt(var_resto)) if var_resto > 0 else float("nan"),
        "dispersion_r_med_geo": float(np.std(r, ddof=1)),
    }


def t3(t: QTable, semilla: int) -> dict[str, float]:
    """La Gamma-sobre-Bailer-Jones, pero con los errores por estrella dentro."""
    out = distance_model(
        t,
        distance_lo_column="r_lo_geo",
        distance_hi_column="r_hi_geo",
        return_trace=True,
        sampling=cfg(semilla),
        priors=DistancePriors(),
    )
    post = out.trace.posterior
    mu = post["mu_r"].values.ravel()
    sd = post["std_r"].values.ravel()
    return {
        "mu_r": float(mu.mean()),
        "std_r": float(sd.mean()),
        "modo": modo_gamma(mu, sd),
        "divergencias": float(out.trace.sample_stats["diverging"].values.sum()),
    }


def t2(t: QTable, semilla: int, columna: str = "parallax") -> dict[str, float]:
    """Espacio de paralaje: sigma_varpi por estrella y el cero residual como nuisance."""
    out = fit_parallax_model(
        t,
        parallax_column=columna,
        parallax_error_column="parallax_error",
        zero_point=True,
        return_trace=True,
        sampling=cfg(semilla),
        priors=ParallaxPriors(),
    )
    post = out.trace.posterior
    mu = post["mu_parallax"].values.ravel()
    d = 1.0 / mu  # mas -> kpc
    return {
        "mu_parallax": float(mu.mean()),
        "d": float(np.mean(d)),
        "d_err": float(np.std(d)),
        "sigma_int": float(post["sigma_parallax"].values.mean()),
        # La profundidad que implica la dispersion intrinseca de paralaje, para comparar con T1.
        "profundidad": float(np.mean(post["sigma_parallax"].values) / np.mean(mu) ** 2),
        "divergencias": float(out.trace.sample_stats["diverging"].values.sum()),
    }


def t5(t: QTable, sigma_int_mas: float, mu_varpi: float) -> dict[str, float]:
    """El presupuesto de error de la media: cuanto es estadistico y cuanto es el suelo del cero."""
    e = np.asarray(t["parallax_error"].to_value(u.mas), dtype=float)
    total = np.sqrt(sigma_int_mas**2 + e**2)
    var_estadistica = 1.0 / np.sum(1.0 / total**2)  # media pesada por precision
    zp = ParallaxPriors().zero_point_scale
    var_total = var_estadistica + zp**2
    # sigma_d = sigma_varpi / varpi^2
    conv = 1.0 / mu_varpi**2
    return {
        "sigma_estadistica": float(np.sqrt(var_estadistica)),
        "suelo_zp": float(zp),
        "sigma_total": float(np.sqrt(var_total)),
        "cuota_zp": float(zp**2 / var_total),
        "d_estadistica": float(np.sqrt(var_estadistica) * conv),
        "d_zp": float(zp * conv),
        "d_total": float(np.sqrt(var_total) * conv),
    }


def t4(t: QTable, sigma_int_mas: float, profundidad_kpc: float) -> dict[str, float]:
    """El mismo criterio, un nivel mas abajo: la anchura que queda, ?es fisica?"""
    e = np.asarray(t["parallax_error"].to_value(u.mas), dtype=float)
    med = float(np.median(e))
    # Varianza que aparece como dispersion "intrinseca" si los errores formales estan
    # subestimados en `SUBESTIMACION`: sqrt((e*(1+f))^2 - e^2).
    equivalente = float(np.sqrt((med * (1 + SUBESTIMACION)) ** 2 - med**2))
    sigma_los_geom = R_T_PC / np.sqrt(5.0) / 1000.0  # pc -> kpc
    return {
        "mediana_sigma_varpi": med,
        "equivalente_subestimacion": equivalente,
        "sigma_int": sigma_int_mas,
        "sigma_los_geometrico": sigma_los_geom,
        "exceso_sobre_geometria": profundidad_kpc / sigma_los_geom,
        "exceso_publicado": PUB_STD / sigma_los_geom,
    }


def promedia(filas: list[dict[str, float]]) -> dict[str, float]:
    return {k: float(np.mean([f[k] for f in filas])) for k in filas[0]}


def main() -> int:
    t = muestra()
    print(f"muestra: {len(t)} fuentes (el ajuste publicado usa 130)\n")

    a = t1(t)
    print("== T1 - aritmetica, sin muestreo ==")
    print(f"  mediana sigma_i (Bailer-Jones) : {a['mediana_sigma_i']:.5f} kpc")
    print(f"  cuota de la varianza impresa   : {a['cuota_varianza'] * 100:.1f} %")
    print(f"  profundidad implicita          : {a['profundidad_implicita']:.5f} kpc")
    print(f"  dispersion de r_med_geo        : {a['dispersion_r_med_geo']:.5f} kpc")
    print(f"  std_r publicado                : {PUB_STD:.5f} kpc\n")

    c = promedia([t3(t, s) for s in SEMILLAS])
    print("== T3 - Gamma sobre Bailer-Jones, con errores por estrella ==")
    print(f"  mu_r  : {c['mu_r']:.5f} kpc")
    print(f"  std_r : {c['std_r']:.5f} kpc   (publicado {PUB_STD:.4f})")
    print(f"  modo  : {c['modo']:.5f} kpc   (publicado {PUB_D:.3f})")
    print(f"  divergencias: {c['divergencias']:.1f}\n")

    b = promedia([t2(t, s) for s in SEMILLAS])
    print("== T2 - espacio de paralaje, sigma_varpi por estrella + cero nuisance ==")
    print(f"  mu_parallax : {b['mu_parallax']:.5f} mas")
    print(f"  d = 1/mu    : {b['d']:.5f} +- {b['d_err']:.5f} kpc   (publicado {PUB_D:.3f})")
    print(f"  sigma_int   : {b['sigma_int']:.5f} mas -> profundidad {b['profundidad']:.5f} kpc")
    print(f"  divergencias: {b['divergencias']:.1f}\n")

    z = t5(t, b["sigma_int"], b["mu_parallax"])
    print("== T5 - el presupuesto de error de la media, que el ajuste publicado no calcula ==")
    print(f"  parte estadistica : {z['sigma_estadistica']:.5f} mas -> {z['d_estadistica']:.5f} kpc")
    print(f"  suelo del cero    : {z['suelo_zp']:.5f} mas -> {z['d_zp']:.5f} kpc")
    print(f"  total             : {z['sigma_total']:.5f} mas -> {z['d_total']:.5f} kpc")
    print(f"  cuota del cero en la varianza : {z['cuota_zp'] * 100:.1f} %")
    print(f"  (T2 muestreado da {b['d_err']:.5f} kpc)\n")

    # La escala de la correccion aplicada, para leer lo anterior. NO es un error: la correccion
    # de Lindegren+2021 esta puesta (`parallax` = `parallax_observed` - `zpvals`, exacto).
    zz = promedia([t2(t, s_, columna="parallax_observed") for s_ in SEMILLAS])
    print("== T5b - el tamano de la correccion que SI esta aplicada ==")
    print(f"  d sin corregir el cero : {zz['d']:.5f} kpc")
    print(f"  o sea la correccion vale : {abs(zz['d'] - b['d']):.5f} kpc")
    print(
        f"  contra el defecto de §J.2 ({abs(b['d'] - PUB_D):.5f} kpc): "
        f"{abs(zz['d'] - b['d']) / max(abs(b['d'] - PUB_D), 1e-9):.0f}x -- el paper acierta "
        "en la decision grande\n"
    )

    d = t4(t, b["sigma_int"], b["profundidad"])
    print("== T4 - la anchura que queda, sometida al mismo criterio (K.1.34) ==")
    print(f"  mediana sigma_varpi formal          : {d['mediana_sigma_varpi']:.5f} mas")
    print(
        f"  equivalente a subestimarlos {SUBESTIMACION * 100:.0f} %     : "
        f"{d['equivalente_subestimacion']:.5f} mas   <- NOTA, no conclusion: otro regimen"
    )
    print(f"  sigma_int medido en T2              : {d['sigma_int']:.5f} mas")
    print(f"  sigma_LOS que implica R_t = {R_T_PC} pc  : {d['sigma_los_geometrico']:.5f} kpc")
    print(f"  exceso de la profundidad medida     : {d['exceso_sobre_geometria']:.1f}x")
    print(f"  exceso de la profundidad publicada  : {d['exceso_publicado']:.1f}x\n")

    # Los umbrales del prerregistro. Cada uno falla por separado y dice cual.
    fallos = []
    if not a["mediana_sigma_i"] > PUB_STD / np.sqrt(2):
        fallos.append(
            f"T1: (E) ya no queda establecido -- mediana sigma_i {a['mediana_sigma_i']:.5f} "
            f"<= {PUB_STD / np.sqrt(2):.5f} kpc"
        )
    if not abs(a["dispersion_r_med_geo"] - PUB_STD) < 0.002:
        fallos.append(
            f"T1: std_r publicado ya no reproduce la dispersion muestral de r_med_geo "
            f"({a['dispersion_r_med_geo']:.5f} contra {PUB_STD:.5f})"
        )
    if not c["std_r"] < 0.8 * PUB_STD:
        fallos.append(
            f"T3: meter los errores por estrella ya no encoge std_r ({c['std_r']:.5f} kpc)"
        )
    if not abs(b["d"] - PUB_D) <= PUB_STD:
        fallos.append(
            f"T2: el ajuste en espacio de paralaje se sale de +-{PUB_STD:.3f} kpc del valor "
            f"publicado ({b['d']:.5f} contra {PUB_D:.3f}) -- ESCALAR AL AUTOR, renumera el paper"
        )
    # T4 no tiene umbral prerregistrado -- se anadio despues de ver T1-T3, y va marcado como tal.
    # Lo que si se afirma es la conclusion que sostiene, y esa falla si deja de sostenerse.
    if not d["exceso_sobre_geometria"] > 2.0:
        fallos.append(
            f"T4: la profundidad medida ya no excede la geometria de R_t por mas de 2x "
            f"({d['exceso_sobre_geometria']:.1f}x) -- podria ser extension fisica"
        )
    # Lo que D10 afirma: el suelo del cero residual DOMINA la incertidumbre correcta de la media.
    # Puede fallar -- se da vuelta si N crece, si los errores por estrella encogen, o si alguien
    # baja `zero_point_scale`. No es un margen de 12x sin mecanismo, que es lo que era antes.
    if not z["suelo_zp"] ** 2 > z["sigma_estadistica"] ** 2:
        fallos.append(
            f"T5: el suelo del cero ya no domina la varianza de la media "
            f"({z['suelo_zp']:.5f} contra {z['sigma_estadistica']:.5f} mas) -- D10 hay que "
            "reescribirlo"
        )
    for nombre, res in (("T3", c), ("T2", b)):
        if res["divergencias"] > 0:
            fallos.append(f"{nombre}: {res['divergencias']:.0f} divergencias")

    if fallos:
        print("FALLA:")
        for f in fallos:
            print(f"  - {f}")
        return 1
    print("OK: las cuatro afirmaciones prerregistradas se sostienen.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
