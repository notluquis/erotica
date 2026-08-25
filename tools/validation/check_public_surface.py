#!/usr/bin/env python3
"""Nada de lo que este repositorio PUBLICA puede mandar al lector a un repo que no puede abrir.

`erotica` es **público**; `phd-hub`, `phd-kb` y `thesis-test` son **privados**. Medido el
2026-08-25: el repo público citaba `~/phd/...` en **doce sitios de la documentación publicada** y en
**cuatro docstrings del paquete**. Ninguno filtraba contenido —se comprobó: `methodology.md`,
`open-threads.md`, `model-landscape.md` y `PIPELINE.md` no existen dentro de este repo, o sea el
diseño de rutear-en-vez-de-copiar aguantó— pero **prometían al lector algo que no puede seguir**, y
cuatro de ellos con las palabras *«Full result:»* y *«Full derivation, scripts and numbers:»*.

**Por qué hacía falta un detector nuevo y no bastaba el que ya existe.** `~/phd/check_dead_paths.py`
resuelve las rutas contra un disco donde `~/phd` **sí existe**, así que las dieciséis le resolvían.
La vista del detector y la del lector público son distintas, y la que faltaba era la del lector.

**Qué cuenta como superficie publicada**, medido contra `docs/conf.py` y no supuesto:

- los `.md` bajo `docs/` que entran en un `toctree` — `myst_parser`;
- **el fuente entero del paquete** — `sphinx.ext.autodoc` publica los docstrings y
  `sphinx.ext.viewcode` publica el modulo completo, comentarios incluidos.

Lo que NO entra: `tests/`, `tools/`, `.github/`. Son visibles navegando GitHub, pero nadie llega a
ellos siguiendo una referencia de la documentacion, que es la clase que esto vigila.

Exit 1 nombrando cada sitio. No escribe nada.
"""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent.parent
#: Los repos privados del programa. Una ruta a cualquiera de ellos es inseguible para un lector.
PRIVADOS = (r"~/phd\b", r"phd-hub", r"phd-kb", r"thesis-test")
#: `AGENTS.md` y su symlink quedan fuera: `docs/conf.py` los excluye de la build a proposito, y son
#: instrucciones para agentes que trabajan CON el hub delante, no documentacion para un lector.
EXENTOS = ("AGENTS.md", "CLAUDE.md")


def superficie() -> list[str]:
    """Los ficheros versionados que Sphinx publica."""
    files = subprocess.run(
        ["git", "-C", str(REPO), "ls-files"], capture_output=True, text=True
    ).stdout.split()
    return [
        f
        for f in files
        if pathlib.PurePath(f).name not in EXENTOS
        and not f.startswith("docs/_build/")
        and (
            (f.startswith("docs/") and f.endswith(".md"))
            or (f.startswith("erotica/") and f.endswith(".py"))
        )
    ]


def canarios() -> list[str]:
    """Que los patrones sigan vivos: uno que DEBE casar y uno que no. → §K.1.16."""
    malos = []
    pos, neg = "see ~/phd/methodology.md PART K", "see docs/design-notes/decisions.md"
    if not any(re.search(p, pos) for p in PRIVADOS):
        malos.append("canario: una cita a un repo privado ya no casa; los patrones están muertos")
    if any(re.search(p, neg) for p in PRIVADOS):
        malos.append("canario: una cita interna al propio repo se marca como privada")
    fs = superficie()
    if not any(f.startswith("docs/") for f in fs) or not any(f.endswith(".py") for f in fs):
        malos.append(
            f"canario: la superficie tiene {len(fs)} ficheros y le falta una de sus dos mitades; "
            "un recorrido roto se lee igual que un repositorio limpio"
        )
    return malos


def main() -> int:
    malos = canarios()
    fs = superficie()
    for rel in fs:
        f = REPO / rel
        if not f.is_file():
            continue
        for n, linea in enumerate(f.read_text(errors="replace").split("\n"), 1):
            for pat in PRIVADOS:
                if re.search(pat, linea):
                    malos.append(
                        f"{rel}:{n} manda al lector a un repo privado — `{linea.strip()[:80]}`"
                    )
                    break
    for m in malos:
        print(f"  - {m}")
    if malos:
        print(
            "\n  -> arreglo: mete inline el número o la conclusión que la ruta prometía; si era un\n"
            "     puntero a doctrina, cita la fuente primaria que esa doctrina ya cita; si era\n"
            "     seguimiento interno, di que queda registrado y quita la ruta."
        )
        return 1
    print(f"superficie publicada: {len(fs)} ficheros, ninguno cita un repo privado")
    return 0


if __name__ == "__main__":
    sys.exit(main())
