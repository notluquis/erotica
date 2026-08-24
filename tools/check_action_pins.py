#!/usr/bin/env python3
"""Cada `uses:` está pineado a un SHA, y el comentario dice de qué ref salió ese SHA.

Las dos mitades importan y la segunda es la que deriva. Un pin es `acción@<40 hex>  # v4`, y nada
impide que alguien bumpee el SHA y deje el comentario, o al revés. Una sesión par encontró
exactamente eso en otro repo: un `# v1` cuyo SHA era un commit viejo, arrastrado desde una copia
ajena, y Dependabot lo reconcilia como un bump inexplicado en vez de uno deliberado.

Qué comprueba, sin red:

1. Ningún `uses:` con ref mutable — ni tag, ni rama, ni `@master`. El 2026-08-23 este repo tenía 24,
   entre ellos `crate-ci/typos@master` y `openjournals/openjournals-draft-action@master`, y las 20
   que GitHub reportaba como `zizmor/unpinned-uses` de severidad `error` llevaban ahí sin que nadie
   las mirara — porque el workflow que las reporta corre con `continue-on-error` y su verde sólo
   decía que había corrido.
2. Cada SHA pineado **está en la historia** del ref que su comentario declara.

   Alcanzabilidad, no igualdad. Exigir que el SHA fuera la cabeza ACTUAL del ref ponía en rojo todo
   pin deliberado en cuanto upstream avanzara, que es justo para lo que sirve pinear: GitHub
   re-apunta el tag rodante `v4` en cada release v4.x, y `master` se mueve en cada push, así que
   `actions/checkout@11d5960  # v4` habría fallado por estar correcto. Lo que un comentario promete
   es *de dónde salió este SHA*, y eso se comprueba preguntando si el ref desciende de él.

(2) necesita red y `gh`. Sin `--offline` y sin ninguno de los dos, degrada a (1) diciendo por qué:
antes `gh` ausente reventaba con un `FileNotFoundError` y una red caída ponía en rojo cada acción con
"no pude resolver", que convierte una falla de entorno en una acusación contra el fichero. Un check
que se calla lo que no pudo mirar es el mismo silencio que este repo persigue en todo lo demás; uno
que culpa al fichero por su propio entorno es peor, porque manda a arreglar lo que no está roto.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
USES = re.compile(r"uses: ([\w.-]+/[\w./-]+)@([\w./-]+)(?:\s+#\s*(\S+))?")


def hay_gh() -> bool:
    """`gh` instalado y autenticado. `subprocess.run` de un binario ausente lanza FileNotFoundError,
    no devuelve un rc: sin esto el check moría con un traceback en cualquier máquina sin `gh`."""
    try:
        return (
            subprocess.run(["gh", "auth", "status"], capture_output=True, timeout=30).returncode
            == 0
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def alcanzable(repo: str, sha: str, ref: str) -> tuple[bool | None, str]:
    """¿El `ref` desciende del `sha` pineado? (None, motivo) si no se pudo averiguar.

    `compare/{sha}...{ref}` devuelve `identical` si el ref sigue ahí mismo y `ahead` si avanzó por
    encima —los dos correctos para un pin—, y `behind`/`diverged` si el SHA no está en esa historia,
    que es el defecto real: un comentario que nombra un ref del que ese commit nunca salió.
    """
    base = "/".join(repo.split("/")[:2])
    try:
        r = subprocess.run(
            ["gh", "api", f"repos/{base}/compare/{sha}...{ref}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        return None, type(e).__name__
    if r.returncode != 0:
        return None, r.stderr.strip().splitlines()[-1][:80] if r.stderr.strip() else "gh api fallo"
    estado = json.loads(r.stdout).get("status")
    return estado in {"identical", "ahead"}, estado or "sin status"


def sha_of(repo: str, ref: str) -> str | None:
    base = "/".join(repo.split("/")[:2])
    for api in (f"repos/{base}/git/ref/tags/{ref}", f"repos/{base}/commits/{ref}"):
        r = subprocess.run(["gh", "api", api], capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            continue
        d = json.loads(r.stdout)
        if "object" not in d:
            return d.get("sha")
        obj = d["object"]
        if obj["type"] != "tag":
            return obj["sha"]
        t = subprocess.run(
            ["gh", "api", f"repos/{base}/git/tags/{obj['sha']}"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return json.loads(t.stdout)["object"]["sha"] if t.returncode == 0 else None
    return None


def main() -> int:
    online = "--offline" not in sys.argv
    motivo_offline = "--offline"
    if online and not hay_gh():
        online, motivo_offline = False, "`gh` ausente o sin autenticar"
    bad: list[str] = []
    sin_red: list[str] = []
    pinned = 0
    # `.yaml` tambien: el glob solo miraba `.yml`, asi que un workflow con la otra extension
    # —que GitHub acepta igual— quedaba fuera del check entero sin decirlo.
    for f in sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")]):
        for repo, ref, comment in USES.findall(f.read_text()):
            if not re.fullmatch(r"[0-9a-f]{40}", ref):
                bad.append(f"{f.name}: {repo}@{ref} no está pineado a un SHA")
                continue
            pinned += 1
            if not comment:
                bad.append(f"{f.name}: {repo} pineado sin comentario que diga de qué ref salió")
                continue
            if not online:
                continue
            ok, detalle = alcanzable(repo, ref, comment)
            if ok is None:
                sin_red.append(f"{f.name}: {repo}@{comment} ({detalle})")
            elif not ok:
                bad.append(
                    f"{f.name}: {repo} dice `# {comment}`, pero {ref[:12]} no esta en la historia "
                    f"de ese ref (compare: {detalle})"
                )

    for line in bad:
        print(f"  - {line}")
    if bad:
        return 1
    # Lo que no se pudo mirar se dice, y no se cuenta como aprobado. No falla: una red caida no es
    # un defecto del fichero, y hacerla fallar entrena a ignorar el rojo.
    for line in sin_red:
        print(f"  ? no se pudo comprobar: {line}")
    print(
        f"pins: {pinned} usos, todos a SHA"
        + (
            f", y cada SHA esta en la historia del ref que declara ({len(sin_red)} sin comprobar)"
            if online
            else f" — el contraste comentario/SHA NO se comprobo ({motivo_offline})"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
