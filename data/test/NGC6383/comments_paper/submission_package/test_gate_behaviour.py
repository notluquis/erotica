#!/usr/bin/env python3
"""Lo que el gate promete sobre sí mismo, comprobado corriéndolo.

No revisa el manuscrito — eso lo hace `gate.py`. Revisa las promesas del propio gate, que son las
que una revisión de código encontró rotas el 2026-08-23 y que ningún check podía ver:

- `--quick` no bendice. Imprimía "OK - el paquete puede subirse" sin haber corrido la ranura
  obligatoria de NESTOR ni ninguna de las dos compilaciones.
- una omisión no sale 0. `REVISADO PARCIAL` salía 0, así que un `gate.py && subir` leía una omisión
  como éxito y sólo un humano leyendo las dos últimas líneas se enteraba.
- `--allow-skips` existe y es lo único que perdona una omisión, para que sea la máquina y no el
  lector quien imponga que en la máquina que sube no se omitió nada.

Es barato: usa `--quick`, que no compila.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE / "gate.py"


def run(*flags: str, env: dict | None = None) -> tuple[int, str]:
    r = subprocess.run([sys.executable, str(GATE), *flags], cwd=HERE, env=env,
                       capture_output=True, text=True, timeout=300)
    return r.returncode, r.stdout + r.stderr


def main() -> int:
    bad: list[str] = []

    code, out = run("--quick")
    if "puede subirse" in out:
        bad.append("--quick bendice el paquete sin haber corrido los checks lentos")
    if "REVISADO PARCIAL" not in out:
        bad.append("--quick no avisa de que es una revisión parcial")

    # Una omisión real dentro de --quick. La primera versión quitaba el zip, pero `c_zip` es un
    # check LENTO: con --quick no corre, así que no había omisión y el caso no probaba nada — el
    # test se cazó a sí mismo al primer intento. `c_typos` sí se omite cuando el binario no está,
    # y recortar el PATH lo provoca sin tocar un solo fichero.
    # Un PATH con chktex y lacheck pero sin typos. La versión anterior recortaba a `/usr/bin:/bin`
    # y se llevaba también los linters, así que `--allow-skips` salía distinto de 0 por un FALLO
    # real y no por la omisión — el caso probaba otra cosa de la que decía. Segunda vez que la
    # sonda de este mismo test era lo roto.
    env = {"PATH": "/Library/TeX/texbin:/usr/bin:/bin", "HOME": str(Path.home())}
    code_sin, out_sin = run("--quick", env=env)
    code_con, _ = run("--quick", "--allow-skips", env=env)
    if "omitido" not in out_sin:
        bad.append("recortar el PATH no produjo una omisión; el caso ya no prueba nada")
    if code_sin == 0:
        bad.append("una omisión sale 0 sin --allow-skips")
    if code_con != 0:
        bad.append("--allow-skips no perdona una omisión")

    for line in bad:
        print(f"  - {line}")
    if bad:
        return 1
    print("gate: --quick no bendice, una omisión no sale 0, --allow-skips la perdona")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
