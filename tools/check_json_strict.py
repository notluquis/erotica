#!/usr/bin/env python3
"""Que los JSON commiteados sean JSON, y no sólo relegibles por Python.

`json.dumps(..., default=float)` —el patrón que usan diez scripts de `tools/validation/`— emite
`NaN`, `Infinity` y `-Infinity` como tokens desnudos cuando un valor no es finito. Python los relee
sin chistar, así que el defecto es invisible desde acá; pero **no son JSON**: RFC 8259 no los define,
y `jq`, `JSON.parse`, Go y cualquier validador de esquema rechazan el fichero entero. Un sidecar
inválido no es "un valor raro en una fila", es un fichero que ningún lector ajeno puede abrir.

Existe porque el `check-json` de pre-commit **pasa sobre un fichero con 907 de estos tokens**:
usa `json.load`, o sea exactamente el lector que no distingue. Medido 2026-08-24 sobre
`tools/validation/benchmark_controls.json`.

`null` es la codificación correcta: es válido, y significa lo mismo que un NaN en estos sidecars —
la métrica no está definida para esa celda (un Jaccard entre conjuntos vacíos, un AUC con una sola
clase). No se pierde ningún número real.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent


def _estricto(constante: str):
    raise ValueError(constante)


def revisar(texto: str) -> str | None:
    """El primer token no estándar del documento, o None si es JSON válido."""
    try:
        json.loads(texto, parse_constant=_estricto)
    except ValueError as e:
        return str(e)
    return None


def main(argv: list[str]) -> int:
    if argv:
        # `.resolve()`: pre-commit pasa rutas RELATIVAS al directorio del repo, y `relative_to`
        # sobre una ruta relativa revienta con ValueError -- el hook moria con un traceback en vez
        # de reportar el fichero malo. Medido al mutar: reintroducir un NaN mataba el hook.
        ficheros = [Path(a).resolve() for a in argv]
    else:
        salida = subprocess.run(
            ["git", "-C", str(RAIZ), "ls-files", "*.json", "*.jsonl"],
            capture_output=True,
            text=True,
        ).stdout.split()
        ficheros = [RAIZ / f for f in salida]

    malos = []
    for f in ficheros:
        texto = f.read_text(errors="replace")
        # Un .jsonl es un objeto por línea, no un documento: se valida línea a línea o el propio
        # lector estricto falla por la segunda llave, que no sería el defecto que se busca.
        lineas = [texto] if f.suffix == ".json" else [x for x in texto.splitlines() if x.strip()]
        for i, linea in enumerate(lineas, 1):
            fallo = revisar(linea)
            if fallo:
                try:
                    donde = str(f.relative_to(RAIZ))
                except ValueError:
                    donde = str(f)
                donde += f":{i}" if f.suffix == ".jsonl" else ""
                malos.append(f"{donde}: token no estandar {fallo!r}")
                break

    if malos:
        print(f"{len(malos)} fichero(s) no son JSON valido pese a que Python los relee:")
        for m in malos:
            print(f"  {m}")
        print("\n  Arreglo: emitir None en vez de float('nan') en el script que lo escribe, y")
        print("  convertir el fichero existente (mismo significado, codificacion valida).")
        return 1
    print(f"{len(ficheros)} JSON/JSONL trackeados, todos validos bajo un lector estricto")
    return 0


if __name__ == "__main__":
    # Canarios: el detector se prueba a si mismo antes de acusar a nadie. Un doble escape o un
    # parse_constant mal cableado lo dejan aceptandolo todo, y el mensaje de exito es identico.
    assert revisar('{"x": NaN}') is not None, "el canario NaN no se caza: el detector esta ciego"
    assert revisar('{"x": Infinity}') is not None, "el canario Infinity no se caza"
    assert revisar('{"x": null, "y": 1.5}') is None, "un JSON valido se esta marcando como malo"
    raise SystemExit(main(sys.argv[1:]))
