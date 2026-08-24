#!/usr/bin/env python3
"""B6: que el generador pueda inyectar un vecino COMÓVIL, y que sea el que B1 midió.

**Escrita antes que la rama que prueba**, y por eso ahora falla. La lección 29 del `CLAUDE.md`
global: un parámetro opcional cuyo default apaga la rama nueva deja toda la suite existente verde
**sin ejercitar nada de lo nuevo, por construcción** — su contraejemplo fue un bug que vivió semanas
dentro de una rama que ningún test podía alcanzar.

Lo que se prueba, y por qué así:

- El contaminante NO cambia de tamaño, cambia de **composición**. `contamination` sigue siendo la
  fracción de no-miembros; lo que el parámetro nuevo reparte es cuánto de eso es un **segundo cúmulo**
  en vez de campo liso. Es exactamente lo que B1 midió —179 de 271 miembros externos (66,1%) caen en
  cúmulos comóviles catalogados— y hace que los brazos A y C difieran **sólo** en esa composición, que
  es la comparación que B6 necesita.
- La separación se declara en unidades de la **dispersión propia del cúmulo**, no en mas/yr, porque
  eso es lo que hace comparable un barrido: `delta_pm=1.25` es Theia 1645 (0,250 mas/yr contra
  σ=0,20), `delta_plx=0.8` su profundidad (0,004 mas contra σ=0,005).
- El vecino tiene que ser **recuperable como cúmulo**: si se inyecta con la dispersión del campo no
  es un vecino, es campo con otro centroide, y el brazo no probaría nada.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ / "tools" / "validation"))

from benchmark_erotica_vs_asteca import (  # noqa: E402
    PLX_C,
    PMDEC_C,
    PMRA_C,
    SIGMA_PLX_INT,
    SIGMA_PM_INT,
    generate,
)

# Lo que B1 midió sobre NGC 6383, en unidades de la dispersión propia del cúmulo.
#   Theia 1645: Δpm 0,250 mas/yr = 1,25 σ ; Δϖ 0,004 mas = 0,8 σ
#   Antalova 2: Δpm 0,549 mas/yr = 2,75 σ ; Δϖ 0,011 mas = 2,2 σ
THEIA = dict(delta_pm=1.25, delta_plx=0.8)
FRAC_VECINO = 0.661  # 179 de 271 miembros externos


def main() -> int:
    fallos: list[str] = []
    comun = dict(n_members=300, contamination=0.5, fractal_dimension=2.0, seed=7)

    # --- el brazo A tiene que seguir siendo exactamente lo que era ---
    a = generate(**comun)
    a2 = generate(**comun, neighbour_fraction=0.0)
    if not np.array_equal(a.pmra, a2.pmra) or not np.array_equal(a.truth, a2.truth):
        fallos.append(
            "con neighbour_fraction=0 la realización cambia: el brazo A dejaría de reproducir "
            "la tabla publicada, y el control deja de ser control"
        )

    # --- el brazo C: el vecino existe, y está donde B1 lo midió ---
    c = generate(**comun, neighbour_fraction=FRAC_VECINO, **THEIA)
    if not hasattr(c, "neighbour"):
        fallos.append(
            "la realización no expone qué estrellas son del vecino: sin eso no hay verdad"
        )
        # sin la máscara no se puede seguir
        for f in fallos:
            print(f"  - {f}")
        return 1

    vec = np.asarray(c.neighbour, dtype=bool)
    n_vec = int(vec.sum())
    n_no_miembro = int((~np.asarray(c.truth, dtype=bool)).sum())
    esperado = round(FRAC_VECINO * n_no_miembro)
    if abs(n_vec - esperado) > max(3, 0.05 * esperado):
        fallos.append(f"el vecino tiene {n_vec} estrellas y se pidió ~{esperado} del contaminante")

    # El contaminante TOTAL no cambia: sólo su composición. Si cambia, A y C difieren en dos cosas.
    if abs(c.contamination - a.contamination) > 0.02:
        fallos.append(
            f"la contaminación cambió al meter el vecino ({a.contamination:.3f} -> "
            f"{c.contamination:.3f}): los brazos diferirían en más de una cosa"
        )

    # --- el vecino está a la distancia declarada, en unidades de la dispersión del cúmulo ---
    if n_vec:
        dpm = np.hypot(np.mean(c.pmra[vec]) - PMRA_C, np.mean(c.pmdec[vec]) - PMDEC_C)
        pedido_pm = THEIA["delta_pm"] * SIGMA_PM_INT
        # Tolerancia MEDIDA, no elegida. La primera versión era
        # `5*SIGMA_PM_INT/sqrt(n) + 0.05` = 0,121 mas/yr, dominada por un 0,05 puesto a ojo, y una
        # mutación que anulaba el desplazamiento en UN eje pasó en verde: dejaba ΔPM en 0,177
        # contra los 0,250 pedidos, dentro de esa holgura.
        #
        # Medido sobre 12 semillas: ΔPM recuperado = 0,2458 ± 0,0148, desvío máximo 0,0253. La
        # mutación da 0,1768, a 0,0732 del pedido. La ventana que discrimina es (0,0253, 0,0732);
        # 0,25 σ = 0,050 cae en medio, a 2x el ruido y 0,7x el error que tiene que cazar.
        tol_pm = 0.25 * SIGMA_PM_INT
        if abs(dpm - pedido_pm) > tol_pm:
            fallos.append(f"ΔPM del vecino {dpm:.3f} mas/yr, se pidió {pedido_pm:.3f}")
        # Δϖ: se comprueba que el parámetro esté CABLEADO, no que 0,8σ sea recuperable —
        # porque está medido que no lo es, y eso es un resultado, no un fallo del test.
        #
        # La tolerancia que había —`5*σ_plx/√n + 0.01`— repetía el defecto de la de arriba dos
        # líneas más abajo: el `+0.01` es **2,5× el pedido** (0,0040 mas), así que una mutación que
        # anulaba el desplazamiento entero pasaba verde. Al medirlo, la ventana que discriminaría
        # resultó **VACÍA**: haría falta tolerancia > 0,0116 (ruido de muestreo, 12 semillas) y
        # < 0,0040 (el pedido) a la vez.
        #
        # La causa es física y es el hallazgo: **el error de paralaje del catálogo es 12× la
        # profundidad intrínseca** —20 a 124 µas contra σ_plx = 5 µas— así que la media observada
        # del vecino la domina el error de medida. Δϖ = 0,8σ son 4 µas bajo ese ruido. Sólo abre a
        # n_vec ≈ 3300, y ahí la ventana es del 25%: demasiado frágil para un guardia.
        #
        # Medido a N de trabajo: el desplazamiento se recupera con desvío máximo 0,0116 mas sea cual
        # sea el pedido, así que a `delta_plx=10` (0,0500) la señal es 4× el ruido. Eso prueba lo
        # que este guardia puede probar — que el parámetro llega al generador — y deja lo otro
        # dicho como medición. → `methodology.md` §K.1.6, "un test de recuperación sólo es un test
        # donde el parámetro es recuperable".
        SONDA_PLX = 10.0
        lejos_plx = generate(
            **comun, neighbour_fraction=FRAC_VECINO, delta_pm=THEIA["delta_pm"], delta_plx=SONDA_PLX
        )
        vp = np.asarray(lejos_plx.neighbour, dtype=bool)
        dplx_p = np.mean(lejos_plx.plx[vp]) - PLX_C
        ped_p = SONDA_PLX * SIGMA_PLX_INT
        if abs(dplx_p - ped_p) > 0.5 * ped_p:
            fallos.append(
                f"Δϖ del vecino {dplx_p:.4f} mas, se pidió {ped_p:.4f}: el desplazamiento en "
                "paralaje no llega al generador"
            )

        # --- y es un CÚMULO, no campo con otro centroide ---
        disp = np.std(
            np.hypot(c.pmra[vec] - np.mean(c.pmra[vec]), c.pmdec[vec] - np.mean(c.pmdec[vec]))
        )
        if disp > 4 * SIGMA_PM_INT:
            fallos.append(
                f"la dispersión de PM del vecino es {disp:.3f} mas/yr, del orden del campo: "
                "eso no es un vecino comóvil, es campo con otro centroide"
            )

    # --- el barrido tiene que poder alejarlo: sin eso no hay brazo B ---
    lejos = generate(**comun, neighbour_fraction=FRAC_VECINO, delta_pm=12.0, delta_plx=0.8)
    v2 = np.asarray(lejos.neighbour, dtype=bool)
    if v2.sum():
        d2 = np.hypot(np.mean(lejos.pmra[v2]) - PMRA_C, np.mean(lejos.pmdec[v2]) - PMDEC_C)
        if d2 < 6 * SIGMA_PM_INT:
            fallos.append(
                f"con delta_pm=12 el vecino quedó a {d2:.3f} mas/yr: el barrido no separa"
            )

    for f in fallos:
        print(f"  - {f}")
    if fallos:
        return 1
    print(
        f"vecino comóvil: {n_vec} estrellas del contaminante a ΔPM={THEIA['delta_pm']}σ y "
        f"Δϖ={THEIA['delta_plx']}σ; el brazo A no cambia y el barrido separa"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
