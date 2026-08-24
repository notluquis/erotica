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

import os
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
    # El entorno se recorta para inducir UNA omisión, no para cambiarle el entorno a los demás
    # checks. `c_kb` se omite cuando ve `CI`/`GITHUB_ACTIONS` y **falla** cuando no las ve, así que
    # un env limpio convertía esa omisión en un FALLO y `--allow-skips` salía distinto de 0 por algo
    # que no era la omisión inducida. Tercera vez que la sonda de este test era lo roto, y la
    # primera que sólo se veía en CI: en local esas variables no están y el check se omite igual por
    # falta del repo hermano. → `methodology.md` §K.1.6c.
    #
    # `/usr/bin` primero: en Ubuntu chktex vive ahí, y `/Library/TeX/texbin` sólo existe en macOS.
    env = {"PATH": "/usr/bin:/bin:/Library/TeX/texbin", "HOME": str(Path.home())}
    for k in ("CI", "GITHUB_ACTIONS", "GITHUB_WORKSPACE"):
        if k in os.environ:
            env[k] = os.environ[k]
    code_sin, out_sin = run("--quick", env=env)
    code_con, _ = run("--quick", "--allow-skips", env=env)
    if "omitido" not in out_sin:
        bad.append("recortar el PATH no produjo una omisión; el caso ya no prueba nada")
    if code_sin == 0:
        bad.append("una omisión sale 0 sin --allow-skips")
    if code_con != 0:
        bad.append("--allow-skips no perdona una omisión")

    # La rama del nº12 sólo vive bajo un chktex < 1.7.9, y en esta máquina hay 1.7.9 — así que
    # `accept` queda en cortocircuito y esa rama NUNCA se ejercita en local. Un shim que reporta la
    # versión vieja y emite un nº12 la despierta sin tocar el manuscrito.
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        shim = Path(d) / "chktex"
        real = subprocess.run(["command", "-v", "chktex"], capture_output=True, text=True,
                              shell=False, executable="/bin/sh")
        ruta = real.stdout.strip() or "/usr/bin/chktex"
        shim.write_text("#!/bin/sh\n"
                        'if [ "$1" = "--version" ]; then echo "ChkTeX v1.7.8 - shim"; exit 0; fi\n'
                        f'"{ruta}" "$@"\n'
                        "printf '12|134|Interword spacing should perhaps be used.\\n'\n")
        shim.chmod(0o755)
        env = {**os.environ, "PATH": f"{d}:/usr/bin:/bin:/Library/TeX/texbin"}
        _, out_viejo = run("--quick", "--allow-skips", env=env)
        linea = next((l for l in out_viejo.split("\n") if "chktex" in l), "")
        if "nº12 aceptado" not in linea:
            bad.append(f"con chktex 1.7.8 el nº12 no se acepta: {linea.strip()[:70]}")
        if "FALLA" in linea:
            bad.append("la rama del nº12 falla bajo la versión que la necesita")

    # El gate no debe escribir encima de un fichero trackeado. Construia el PDF y el log ENCIMA
    # de clean_source/aanda.pdf y marked_changes/aanda_marked.pdf, y para evitar el churn de ~12 MB
    # por corrida restauraba los bytes viejos cuando el texto extraido no cambiaba. Eso revertia en
    # silencio una figura regenerada (una figura no cambia el texto), y devolvia el PDF viejo justo
    # antes de que c_deliverables leyera su /Producer, tapando el desajuste de motor de TeX. Se
    # comprueba la propiedad, no la ausencia de la linea: cualquier rescritura que vuelva a apuntar
    # el build al arbol trackeado falla aca aunque no se parezca al codigo que se quito.
    sys.path.insert(0, str(HERE))
    import gate

    rastreados = subprocess.run(["git", "ls-files"], cwd=HERE, capture_output=True, text=True).stdout.split()
    for tex in (gate.TEX, gate.MARKED):
        for salida in gate.build_paths(tex):
            rel = salida.relative_to(HERE).as_posix()
            if gate.BUILD_DIR not in salida.parts:
                bad.append(f"el gate escribe {rel} fuera de {gate.BUILD_DIR}/")
            if rel in rastreados:
                bad.append(f"el gate escribe encima de {rel}, que esta trackeado")

    # Y la otra mitad del mismo defecto: las paginas se leen del log del build. Con `-outdir` la
    # linea trae el prefijo del directorio, asi que un patron anclado en "{stem}.pdf" a secas deja
    # de casar y el check informa "no se pudo leer las paginas" sobre un build correcto.
    if gate.pages_in("Output written on _gate_build/aanda.pdf (26 pages, 5 bytes).", "aanda") != 26:
        bad.append("pages_in no lee la linea del log con outdir")
    if gate.pages_in("Output written on aanda.pdf (26 pages).", "aanda") != 26:
        bad.append("pages_in no lee la linea del log sin outdir")
    if gate.pages_in("Output written on otro.pdf (26 pages).", "aanda") is not None:
        bad.append("pages_in acepta un stem que no es el suyo")

    for line in bad:
        print(f"  - {line}")
    if bad:
        return 1
    print("gate: --quick no bendice, una omisión no sale 0, --allow-skips la perdona,\n      el build no toca ficheros trackeados, pages_in lee el log con y sin outdir")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
