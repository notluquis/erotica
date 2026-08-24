#!/usr/bin/env python3
"""D6: «no encontré nada» tiene que poder distinguirse de «me rompí».

Escrita antes que el arreglo, así que ahora falla.

`search_pseudoprobability` lanza `RuntimeError("Pseudo-probability search did not find candidate
clusters.")` cuando el campo no tiene estructura. **La respuesta es correcta** — no hay nada que
encontrar — pero llega con el mismo tipo que un fallo real, así que cada consumidor tiene que
decidirlo leyendo la cadena del mensaje. Hoy uno de ellos lo hace exactamente así, con un comentario
en vez de código: `except RuntimeError as exc:  # "did not find candidate clusters"`.

Medido en CTRL-seeds: **82 ocurrencias sobre 24 semillas de campo liso**, 70 celdas de los arms `5d`
y 12 de los `3d`, y **cero** en cuanto hay estructura que encontrar. O sea que la rama no es rara:
es la respuesta normal a un cielo sin cúmulo, y el prerregistro de ese hilo llegó a escribir que era
«una caída puntuada como acierto» antes de leer el mensaje.

El arreglo compatible es un **tipo propio que herede de `RuntimeError`**: quien ya captura
`RuntimeError` sigue funcionando, y quien quiera distinguir puede hacerlo sin mirar el texto.
Cambiar la excepción por un retorno rompería a los dos consumidores que hoy la esperan.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np

RAIZ = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ))

from astropy.table import QTable  # noqa: E402

from erotica.core.clustering import Clustering  # noqa: E402


def main() -> int:
    fallos: list[str] = []

    # El tipo tiene que existir y ser un RuntimeError, o romperíamos a quien ya captura.
    try:
        from erotica.core.clustering import NoCandidateClusters
    except ImportError:
        print("  - `NoCandidateClusters` no existe: no hay forma de distinguir sin leer el mensaje")
        return 1
    if not issubclass(NoCandidateClusters, RuntimeError):
        fallos.append("no hereda de RuntimeError: rompe a quien ya lo captura")

    # Campo liso puro: no hay nada que encontrar, y eso NO es un fallo.
    rng = np.random.default_rng(0)
    n = 400
    t = QTable(
        {
            "ra": rng.uniform(249.5, 250.5, n),
            "dec": rng.uniform(-32.5, -31.5, n),
            "pmra": rng.normal(-3.0, 4.0, n),
            "pmdec": rng.normal(-3.0, 4.0, n),
            "parallax": rng.gamma(3.0, 0.3, n),
        }
    )
    a = Clustering(t)
    try:
        a.search_pseudoprobability(
            columns=("pmra", "pmdec"), min_cluster_size_samples=range(150, 190)
        )
        print("  (este campo sí produjo candidatos; la sonda no ejercitó la rama)")
    except NoCandidateClusters as e:
        if "did not find candidate" not in str(e):
            fallos.append(f"el mensaje dejó de nombrar la causa: {e}")
    except RuntimeError as e:
        fallos.append(f"sigue llegando como RuntimeError pelado, indistinguible: {e}")

    for f in fallos:
        print(f"  - {f}")
    if fallos:
        return 1
    print(
        "`NoCandidateClusters` distingue «no hay nada» de «me rompí», y sigue siendo un RuntimeError"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
