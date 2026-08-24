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
2. Cada SHA pineado resuelve de verdad al ref que su comentario declara.

(2) necesita red. Sin ella el check hace (1) y **dice en voz alta que no hizo (2)**, en vez de salir
verde como si hubiera comprobado las dos cosas: un check que se calla lo que no pudo mirar es el
mismo silencio que este repo persigue en todo lo demás.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"
USES = re.compile(r"uses: ([\w.-]+/[\w./-]+)@([\w./-]+)(?:\s+#\s*(\S+))?")


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
    bad: list[str] = []
    pinned = 0
    for f in sorted(WORKFLOWS.glob("*.yml")):
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
            real = sha_of(repo, comment)
            if real is None:
                bad.append(f"{f.name}: no pude resolver {repo}@{comment}")
            elif real != ref:
                bad.append(
                    f"{f.name}: {repo} dice `# {comment}` y ese ref es {real[:12]}, "
                    f"no el pineado {ref[:12]}"
                )

    for line in bad:
        print(f"  - {line}")
    if bad:
        return 1
    print(
        f"pins: {pinned} usos, todos a SHA"
        + (
            ", y cada comentario resuelve a su SHA"
            if online
            else " — el contraste comentario/SHA NO se comprobó (--offline)"
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
