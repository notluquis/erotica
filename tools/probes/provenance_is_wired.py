#!/usr/bin/env python3
"""F2: los scripts que produjeron las cifras publicadas no dejaban registro de qué las produjo.

Dos afirmaciones, medidas por separado porque fallan por razones distintas.

**La forma** — cada script de `comments_paper/review_repo/` que escribe una traza escribe también
su sidecar. Se comprueba estáticamente porque correr los seis cuesta cuatro cadenas de NUTS por
sitio, y lo que puede regresar es que alguien añada un `to_netcdf` sin su línea de al lado.

**El contenido** — que el sidecar traiga lo que dice traer. Se ejercita la llamada REAL con la
entrada REAL de los scripts, saltándose sólo el ajuste: `write_metadata` no sabe nada del muestreo,
así que un sidecar escrito sobre el catálogo de verdad es indistinguible del que escribirá la
re-corrida.

⚠ **Lo que esta sonda NO puede afirmar, y por eso se dice en voz alta**: las 14 trazas que ya están
en disco siguen sin registro y **no pueden adquirirlo**. Un sidecar escrito hoy grabaría el git y
las dependencias de hoy para una traza de hace semanas — un registro fabricado, que sobrevive a
cualquier comprobación que se le haga y por eso es peor que ninguno. Se arregla re-corriendo, o no
se arregla.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import tempfile

RAIZ = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(RAIZ))

REVIEW = RAIZ / "data/test/NGC6383/comments_paper/review_repo"
ENTRADA = (
    RAIZ
    / "data/test/NGC6383/comments_paper/radius_robustness/generated/40/paperfaithful_reference_p06.ecsv"
)


def forma() -> list[str]:
    """Todo script que escribe una traza escribe su sidecar."""
    fallos = []
    escriben = sorted(p for p in REVIEW.glob("*.py") if "to_netcdf" in p.read_text())
    if not escriben:
        return ["ningún script escribe trazas: el patrón murió o el directorio se movió"]
    for p in escriben:
        t = p.read_text()
        n_nc = len(re.findall(r"\.to_netcdf\(", t))
        n_side = len(re.findall(r"write_metadata\(", t))
        if n_side < n_nc:
            fallos.append(f"{p.name}: {n_nc} trazas escritas, {n_side} sidecars")
    return fallos


def contenido() -> list[str]:
    """El sidecar trae git, dependencias y el checksum de la entrada de verdad."""
    from erotica.analysis import write_metadata

    fallos = []
    if not ENTRADA.is_file():
        return [f"la entrada de los scripts no está: {ENTRADA}"]
    with tempfile.TemporaryDirectory() as d:
        destino = pathlib.Path(d) / "idata_probe_provenance.json"
        write_metadata(destino, inputs=[str(ENTRADA)], seeds=42, script="probe")
        rec = json.loads(destino.read_text())

    for clave in (
        "created_at",
        "erotica_version",
        "git",
        "python",
        "dependencies",
        "seeds",
        "inputs",
    ):
        if clave not in rec:
            fallos.append(f"el sidecar no trae `{clave}`")
    if rec.get("seeds") != 42:
        fallos.append(f"el seed no sobrevive al registro: {rec.get('seeds')!r}")
    # Sin checksum el sidecar no ata la traza a SU entrada: nombra un fichero que pudo cambiar.
    ins = rec.get("inputs") or []
    if not (
        isinstance(ins, list)
        and ins
        and any(
            "blake2b" in json.dumps(i) or "checksum" in json.dumps(i) or "digest" in json.dumps(i)
            for i in ins
        )
    ):
        fallos.append(f"la entrada se registra sin checksum: {json.dumps(ins)[:160]}")
    # Un git ausente convierte el registro en una fecha con adornos.
    if not isinstance(rec.get("git"), dict) or not rec["git"].get("commit"):
        fallos.append(f"sin commit de git el registro no identifica el código: {rec.get('git')!r}")

    # Y una entrada ilegible se ANOTA, no levanta: el registro nunca se lleva por delante la traza.
    with tempfile.TemporaryDirectory() as d:
        destino = pathlib.Path(d) / "falta.json"
        try:
            write_metadata(destino, inputs=[str(ENTRADA.parent / "no_existe.ecsv")], seeds=1)
        except Exception as e:  # noqa: BLE001
            fallos.append(
                f"una entrada ausente levanta en vez de anotarse ({type(e).__name__}); "
                "un registro que mata el resultado que describe es peor que ninguno"
            )
    return fallos


def main() -> int:
    fallos = [("forma", f) for f in forma()] + [("contenido", f) for f in contenido()]
    for que, f in fallos:
        print(f"  - {que}: {f}")
    if fallos:
        return 1
    n_nc = len(list(REVIEW.rglob("*.nc")))
    n_side = len(list(REVIEW.rglob("*_provenance*.json")))
    print(
        f"provenance cableada en los {len(sorted(p for p in REVIEW.glob('*.py') if 'to_netcdf' in p.read_text()))} "
        f"scripts que escriben trazas, y el sidecar trae git, dependencias y checksum de la entrada real"
    )
    print(
        f"  residuo declarado: {n_nc} trazas ya en disco, {n_side} con sidecar — no se retrofitean, "
        f"se re-corren (un sidecar de hoy sobre una traza de hace semanas es un registro fabricado)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
