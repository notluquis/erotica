#!/usr/bin/env python3
"""Lo que el paquete importa tiene que estar declarado en `pyproject.toml`.

El release-blocker decia *«verify in a clean venv/clone»*, y el sandbox bloqueo esa prueba en su
momento. Un venv limpio necesita red, tarda minutos y **no dice cual falta**: dice que un import
revento. Esto lo mide sin red, en un segundo, y nombra el modulo.

La consecuencia que motivo el blocker es concreta: `gaiadr3-zeropoint` se importa a nivel de MODULO
en `preprocess/preprocessor.py`, asi que su ausencia rompia `import erotica.preprocess` en una
instalacion fresca — no en una llamada rara, en el import.

**Dos niveles, y la distincion ES el detector.** La primera version agrupaba `dependencies` con
todos los extras y preguntaba solo *"esta declarado en algun sitio"*. Con esa regla, mover
`gaiadr3-zeropoint` de core a `[bayes]` pasa en verde y reintroduce el blocker entero — la union de
fuentes borra la atribucion, que es la forma exacta de `methodology.md` §K.1.36 y la escribi dentro
del detector el mismo dia que la documente.

| donde se importa | donde tiene que estar declarado |
|---|---|
| **a nivel de modulo, sin guardia** — corre con `import erotica.<algo>` | `[project.dependencies]`, y solo ahi |
| dentro de una funcion, o bajo `try/except ImportError` | core **o** cualquier extra |

`if TYPE_CHECKING:` no cuenta: no corre en runtime. Medido antes de apretar la regla: de 13 imports
de nivel de modulo no-stdlib, **uno** cae fuera de core y es `mpl_toolkits`, que ya estaba exento
por ser parte de matplotlib. Ruido cero sobre el corpus de hoy → §K.1.27.

**Las excepciones son por nombre y cada una trae su razon.** Una exencion sin motivo escrito es una
lista de cosas que alguien callo, no una decision.

Exit 1 nombrando cada import sin declarar. No escribe nada.
"""

from __future__ import annotations

import ast
import pathlib
import sys
import tomllib

REPO = pathlib.Path(__file__).resolve().parent.parent.parent

#: Nombre de distribucion -> nombre de import, donde diferen. `pip install scikit-learn` da
#: `import sklearn`, y comparar las cadenas crudas marcaria como ausente algo que esta.
ALIAS = {
    "scikit-learn": "sklearn",
    "adjusttext": "adjustText",
    "gaiadr3-zeropoint": "zero_point",
    "fast-histogram": "fast_histogram",
    "hr-selection-function": "hr_selection_function",
    "pytest-cov": "pytest_cov",
    "myst-parser": "myst_parser",
}
#: Exentos, con su razon. NO se amplia sin escribir una.
EXENTOS = {
    # Subpaquete de matplotlib, que si esta declarado: su top-level es otro nombre.
    "mpl_toolkits": "es parte de `matplotlib`, ya declarada",
    # Su propio ImportError lo dice: "pyUPMASK is not importable. It ships as scripts, not a
    # package". No hay nada que `pip install` pueda resolver, y por eso el guardia manda a correrlo
    # por fuera y pasarle la salida.
    "pyUPMASK": "se distribuye como scripts, no como paquete; no es instalable por pip",
}


def _nombres(specs: list[str]) -> set[str]:
    fuera = set()
    for x in specs:
        base = x.split("@")[0].split(">")[0].split("<")[0].split("=")[0].split("[")[0]
        base = base.strip().lower()
        fuera.add(base)
        fuera.add(ALIAS.get(base, base.replace("-", "_")))
    return fuera


def declaradas() -> tuple[set[str], set[str]]:
    """`(core, core|extras)` — separados a proposito. Agruparlos es el defecto que esto arregla."""
    d = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    core = _nombres(list(d.get("dependencies", [])))
    extras: list[str] = []
    for v in (d.get("optional-dependencies") or {}).values():
        extras += list(v)
    return core, core | _nombres(extras)


def _nivel_modulo(arbol: ast.Module):
    """Los import que corren con solo hacer `import erotica.<algo>`.

    NO cuentan: los de dentro de una funcion (perezosos), los de un `try/except ImportError`
    (guardados a proposito — el repo ya usa ese idioma) y los de `if TYPE_CHECKING:`, que no corren.
    """
    for n in arbol.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            yield n
        elif isinstance(n, ast.If) and "TYPE_CHECKING" not in ast.unparse(n.test):
            for m in n.body + n.orelse:
                if isinstance(m, (ast.Import, ast.ImportFrom)):
                    yield m


def _tops(n: ast.AST) -> list[str]:
    if isinstance(n, ast.Import):
        nombres = [a.name for a in n.names]
    elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
        nombres = [n.module]
    else:
        return []
    return [nm.split(".")[0] for nm in nombres]


def importados() -> tuple[dict[str, str], dict[str, str]]:
    """`(todos, solo_nivel_modulo)`, cada uno modulo de primer nivel -> primer sitio."""
    todos: dict[str, str] = {}
    nucleo: dict[str, str] = {}
    for f in sorted((REPO / "erotica").rglob("*.py")):
        try:
            arbol = ast.parse(f.read_text(errors="replace"))
        except SyntaxError:
            continue
        for n in ast.walk(arbol):
            for top in _tops(n):
                todos.setdefault(top, f"{f.relative_to(REPO)}:{n.lineno}")
        for n in _nivel_modulo(arbol):
            for top in _tops(n):
                nucleo.setdefault(top, f"{f.relative_to(REPO)}:{n.lineno}")
    return todos, nucleo


def problemas() -> list[str]:
    std = set(sys.stdlib_module_names)
    core, cualquiera = declaradas()
    todos, nucleo = importados()

    def falta(mod: str, donde: set[str]) -> bool:
        return mod not in donde and mod.replace("_", "-") not in donde

    malos = []
    vistos = 0
    for mod, donde in sorted(todos.items()):
        if mod in std or mod == "erotica" or mod in EXENTOS:
            continue
        vistos += 1
        if falta(mod, cualquiera):
            malos.append(
                f"`{mod}` se importa en {donde} y `pyproject.toml` no lo declara — "
                "en una instalación fresca eso es un ImportError, no un aviso"
            )
        elif mod in nucleo and falta(mod, core):
            malos.append(
                f"`{mod}` se importa a nivel de módulo en {nucleo[mod]} pero sólo está declarado "
                "en un extra: `pip install erotica` sin ese extra rompe al importar el paquete, "
                "no al usar la función. Va en `[project.dependencies]`, o el import se hace "
                "perezoso / bajo `try: ... except ImportError`"
            )
    # Canarios: si el recorrido o el parser se rompen, no encuentran nada y pasa en vacio.
    if vistos < 5:
        malos.append(
            f"canario: sólo {vistos} imports externos en `erotica/`; el recorrido está roto y "
            "un vacío se lee igual que un limpio"
        )
    if "numpy" not in nucleo:
        malos.append(
            "canario: `numpy` no aparece entre los imports de nivel de módulo; el recorrido de "
            "`_nivel_modulo` no está leyendo y el nivel estricto pasaría en vacío"
        )
    if not core or "numpy" not in core:
        malos.append("canario: `[project.dependencies]` no trae `numpy`; el parseo de core falla")
    return malos


def main() -> int:
    malos = problemas()
    for m in malos:
        print(f"  - {m}")
    if malos:
        print(
            "\n  -> arreglo: decláralo en `pyproject.toml` (en el extra que le toque), o —si de "
            "verdad\n     no es instalable— añádelo a EXENTOS **con su razón escrita**."
        )
        return 1
    todos, nucleo = importados()
    std = set(sys.stdlib_module_names)
    externos = [m for m in todos if m not in std and m != "erotica" and m not in EXENTOS]
    estrictos = [m for m in externos if m in nucleo]
    print(
        f"dependencias: {len(externos)} módulos externos importados, {len(estrictos)} de ellos a "
        "nivel de módulo (exigen core); todos declarados o exentos"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
