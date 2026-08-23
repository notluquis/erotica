#!/usr/bin/env python3
"""Deniega comandos que borran trabajo no respaldado, ANTES de que corran.

Por qué existe, con el incidente medido:

`~/.claude/settings.json` trae `Bash(git *)` en su lista de permitidos, así que **cada** comando
de git corre sin preguntar. El 2026-08-23, revirtiendo una mutación de prueba, un checkout que
descarta el árbol se llevó por delante una edición real de la Sect. 8 del manuscrito de P01 que no
estaba commiteada. No hubo prompt porque no podía haberlo: el patrón estaba preaprobado. De ahí
salió la regla de esa sesión — *las mutaciones se revierten desde copias en `/tmp`, nunca desde
git* — y una regla que depende de acordarse no es un guardarraíl.

Un hook `PreToolUse` sí corre sobre comandos preaprobados, y puede negar. Eso es lo único que
ataja este caso.

Adaptado del `check-destructive.sh` de `bukhr/datawarehouse`, con la lista reescrita: la de allá
cubre borrado recursivo, reset duro, push forzado y DDL de SQL, y **no cubre el checkout que
descarta el árbol**, que es justo el comando del incidente.

## Por qué tokeniza en vez de aplicar regex a la cadena

La primera versión era una lista de expresiones regulares sobre el comando crudo, igual que la de
origen. Una revisión que no la escribió midió seis agujeros, y los seis eran la misma causa — una
regex no ve la estructura del comando:

| pasaba sin bloquearse | por qué |
|---|---|
| `git -C /otro/repo checkout -- f.tex` | el subcomando no iba pegado a `git`, y **derrotaba las cinco reglas de git a la vez** |
| `git push origin --force` | la regla exigía la bandera pegada a `push` |
| `git clean -d -f`, `git clean --force` | la regla no cruzaba un segundo `-` |
| `rm -r -f x`, `rm --recursive --force x`, `sudo rm -rf x` | banderas separadas, forma larga, prefijo |
| `git checkout .` | la forma más común de perder el árbol entero |
| `git switch --discard-changes` | ausente |

Y al revés, denegaba cosas que no destruyen nada: `git restore --staged` sólo saca del índice, y
`git clean -fn` es un ensayo. Ese costo es el que apaga guardarraíles.

Un séptimo, de precedencia: `re.search(BOUNDARY + patron)` concatenaba sin agrupar, así que en una
regla con `|` de nivel superior el ancla aplicaba **sólo a la primera alternativa**. `rm -fr` se
marcaba en cualquier parte de la cadena y `rm -rf` no. Tokenizar borra la clase entera.

## Lo que sigue sin ver, dicho a propósito

- `git checkout <rama-o-fichero>` con un nombre simple y sin `/` ni extensión es indistinguible de
  un cambio de rama. Se cubren `--`, `.`, `./`, rutas con `/` y extensiones conocidas.
- Un `python -c "open(f,'w')"` o un `>` de shell borran igual y no son ninguno de estos comandos.
- Es una lista de comandos conocidos: sólo ve lo que alguien ya nombró.
- Un segmento cuyo texto EMPIECE con uno de estos comandos se marca aunque sea la línea de un
  mensaje de commit. Distinguirlo pide entender heredocs y comillas, o sea un parser de shell
  completo, y un guardia que intenta ser un parser falla de maneras más raras que el caso que
  arregla. Cuando estorbe, el trabajo se escribe a un fichero y se corre desde ahí — que es como se
  prueba este mismo guardia, porque sus casos de prueba son por construcción comandos destructivos.

Falla abierto ante cualquier excepción propia: un guardia nunca puede ser la razón de que un turno
se caiga. Eso significa que su silencio NO es prueba de que revisó — por eso se mutation-testea.
"""

from __future__ import annotations

import json
import pathlib
import re
import shlex
import sys

# Prefijos que envuelven al comando real sin cambiar lo que hace.
WRAPPERS = {"sudo", "command", "env", "nohup", "time", "builtin", "exec"}
# `git -C <ruta>` y compañía: opciones globales que consumen un argumento.
GIT_OPTS_WITH_ARG = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}
# Sufijos que delatan que el argumento de checkout es un fichero y no una rama.
FILEISH = re.compile(r"\.(tex|py|md|json|yaml|yml|toml|txt|cfg|ini|sh|csv|dat|bib|cls|sty)$")

DISCARDS_TREE = "descarta cambios sin commitear del árbol de trabajo"


def _strip_prefixes(toks: list[str]) -> list[str]:
    i = 0
    while i < len(toks) and (re.match(r"^\w+=", toks[i]) or toks[i] in WRAPPERS):
        i += 1
    return toks[i:]


def _git_subcommand(toks: list[str]) -> tuple[str, list[str]]:
    """Salta las opciones globales para llegar al subcomando real. `git -C x checkout` es checkout."""
    j = 1
    while j < len(toks) and toks[j].startswith("-"):
        j += 2 if toks[j] in GIT_OPTS_WITH_ARG else 1
    return (toks[j], toks[j + 1 :]) if j < len(toks) else ("", [])


def _has(rest: list[str], *names: str) -> bool:
    return any(t in names for t in rest)


def _short_flags(rest: list[str]) -> str:
    """Letras de las banderas cortas, juntas: `-d -f` y `-fd` dan lo mismo."""
    return "".join(t[1:] for t in rest if re.fullmatch(r"-[A-Za-z]+", t))


def _judge_git(sub: str, rest: list[str]) -> str | None:
    if sub == "checkout":
        # `-b`/`-B` crea rama: nunca toca ficheros. Va primero porque los nombres de rama de este
        # equipo llevan `/` (`feat/x`, `dp/DAT-…`) y la heurística de "parece ruta" los marcaba.
        if _has(rest, "-b", "-B"):
            return None
        if "--" in rest:
            return DISCARDS_TREE
        if _has(rest, "-f", "--force"):
            return DISCARDS_TREE
        for t in rest:
            if t.startswith("-"):
                continue
            if t in (".", "./") or "/" in t or FILEISH.search(t) or "*" in t:
                return DISCARDS_TREE
            break  # el primer argumento suelto es la rama; lo demás ya no decide
        return None
    if sub == "switch":
        return DISCARDS_TREE if _has(rest, "--discard-changes", "-f", "--force") else None
    if sub == "restore":
        # `--staged` a secas sólo saca del índice: no toca el árbol y denegarlo es el falso
        # positivo que termina apagando el guardia.
        if _has(rest, "--staged") and not _has(rest, "--worktree", "-W"):
            return None
        return DISCARDS_TREE
    if sub == "clean":
        forced = "f" in _short_flags(rest) or _has(rest, "--force")
        dry = "n" in _short_flags(rest) or _has(rest, "--dry-run")
        if forced and not dry:
            return "borra ficheros sin trackear, que suelen ser trabajo en curso"
        return None
    if sub == "reset":
        return "descarta commits y el árbol de trabajo a la vez" if _has(rest, "--hard") else None
    if sub == "push":
        if _has(rest, "--force-with-lease") or any(
            t.startswith("--force-with-lease=") for t in rest
        ):
            return None
        return "reescribe la rama remota" if _has(rest, "--force", "-f") else None
    return None


def _judge(seg: str) -> str | None:
    try:
        toks = _strip_prefixes(shlex.split(seg))
    except ValueError:  # comillas sin cerrar: no es un comando que podamos leer
        return None
    if not toks:
        return None
    if toks[0] == "rm":
        rest, short = toks[1:], _short_flags(toks[1:])
        recursive = "r" in short or "R" in short or _has(rest, "--recursive")
        forced = "f" in short or _has(rest, "--force")
        return "borra recursivamente sin papelera" if (recursive and forced) else None
    if toks[0] == "git":
        return _judge_git(*_git_subcommand(toks))
    return None


def verdict(cmd: str) -> str | None:
    """Un comando por segmento: `&&`, `||`, `;`, `|` y saltos de línea los separan."""
    for seg in re.split(r"&&|\|\||[;|\n]", cmd):
        loss = _judge(seg)
        if loss:
            return loss
    return None


HINT = (
    "Para revertir una mutación de prueba: respalda antes en /tmp y restaura desde ahí. "
    "Si esto es intencional, dilo explícitamente y córrelo tú con `! <comando>`."
)


# La misma lista tiene que proteger los dos repos donde se edita trabajo no commiteado: este hub y
# `erotica`, donde ocurrió el incidente. Son repos separados con remotos separados, así que el
# script viaja duplicado -- y CLAUDE.md dice que un hook son dos mitades, script y declaración, y
# que las dos viajan. Apuntar la declaración de erotica a la copia del hub por ruta absoluta
# resolvería la duplicación y falla abierto si el hub no está, que es peor: quedaría sin guardia
# creyendo que lo tiene.
#
# El fichero está en formato `ruff format` aunque este repo no corre ruff: el pre-commit de erotica
# sí, así que reformatea su copia en cada commit y las separa. Mantener las dos en el formato del
# repo más estricto es lo único que las deja quietas.
TWIN_REPOS = ("phd", "erotica")


def _twin_of(mine: pathlib.Path) -> pathlib.Path | None:
    """La copia del OTRO repo, derivada de dónde está ésta.

    Antes era una ruta fija a `~/erotica/...`, y eso hacía que `--check` corrido desde la copia de
    erotica se comparara consigo misma y saliera 0 **sin comparar nada** — verde aunque la del hub
    estuviera borrada. Derivarla del propio `__file__` hace que las dos direcciones comprueben.
    """
    rel = pathlib.Path(".claude") / "hooks" / mine.name
    for repo in TWIN_REPOS:
        root = mine.parent.parent.parent
        if root.name == repo:
            other = next(r for r in TWIN_REPOS if r != repo)
            return root.parent / other / rel
    return None


def _declared(repo_root: pathlib.Path) -> bool:
    """Que el otro repo además lo DECLARE. Comparar sólo el script deja pasar la mitad que falta:
    borrar el bloque `PreToolUse` de su settings.json dejaba ese repo sin guardia con este check en
    verde, que es la misma deriva silenciosa que el check existe para cerrar."""
    cfg = repo_root / ".claude" / "settings.json"
    if not cfg.exists():
        return False
    try:
        hooks = json.loads(cfg.read_text()).get("hooks", {}).get("PreToolUse", [])
    except Exception:
        return False
    return any(
        "guard_destructive.py" in h.get("command", "")
        for entry in hooks
        for h in entry.get("hooks", [])
    )


def check_twin() -> int:
    mine = pathlib.Path(__file__).resolve()
    twin = _twin_of(mine)
    if twin is None:
        print(f"no sé cuál es el repo gemelo de {mine}; esperaba uno de {TWIN_REPOS}")
        return 1
    repo_root = twin.parent.parent.parent
    if not repo_root.is_dir():
        # Falta el REPO: se omite, ruidosamente. Falta el FICHERO teniendo el repo: eso es un
        # borrado y falla. Colapsar los dos bloqueaba cada turno en cualquier clon que no tuviera
        # el repo hermano al lado, contradiciendo la regla que la skill machine-check declara.
        print(f"omitido: {repo_root} no está; no hay copia gemela que comparar")
        return 0
    if not twin.exists():
        print(
            f"{repo_root.name} tiene el repo pero no el guardarraíl en {twin}: quedó sin protección"
        )
        return 1
    if twin.read_bytes() != mine.read_bytes():
        print(f"las dos copias del guardarraíl divergieron: {mine} != {twin}")
        return 1
    if not _declared(repo_root):
        print(
            f"{repo_root.name} tiene el script pero su settings.json no declara el hook PreToolUse"
        )
        return 1
    return 0


def main() -> int:
    if "--check" in sys.argv:
        return check_twin()
    try:
        raw = sys.stdin.read()
        event = json.loads(raw) if raw.strip() else {}
        cmd = (event.get("tool_input") or {}).get("command") or ""
        if not cmd:
            return 0
        loss = verdict(cmd)
        if loss is None:
            return 0
        print(
            json.dumps(
                {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": f"Guardarraíl: este comando {loss}. {HINT}",
                    }
                }
            )
        )
    except Exception:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
