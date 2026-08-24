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
# Envoltorios que reciben el comando real como ARGUMENTO, con gramáticas distintas entre sí:
# `timeout 5 rm -rf x`, `xargs -n1 rm -rf`, `nice -n 19 rm -rf`, `watch rm -rf`. No se modela la
# gramática de cada uno —`timeout` come una duración, `xargs -I{}` come un patrón— porque
# equivocarse en una la deja pasar entera. Se juzga cada sufijo, que no depende de ninguna.
CMD_WRAPPERS = {"timeout", "xargs", "nice", "ionice", "stdbuf", "watch", "parallel"}
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
        # `-p`/`--patch` descarta trozos del árbol interactivamente, sin nombrar ninguna ruta.
        if _has(rest, "-p", "--patch"):
            return DISCARDS_TREE
        sueltos = [t for t in rest if not t.startswith("-")]
        # `git checkout <tree-ish> <ruta>...`: con dos o más argumentos sueltos, los que siguen al
        # primero SON rutas y la forma descarta el árbol. Antes se hacía `break` tras el primero
        # —"lo demás ya no decide"— y por eso `git checkout HEAD .` pasaba: la grafía exacta del
        # incidente que este fichero existe para impedir, con `HEAD` absorbiendo la decisión.
        if len(sueltos) >= 2:
            return DISCARDS_TREE
        # Con uno solo no se puede distinguir rama de ruta por sintaxis: `docs/notes` es las dos
        # cosas. Se resuelve conservador, con `/`, que es lo que hacia la version anterior y lo que
        # exige el fichero de casos (`checkout src/` deniega, `checkout dev` pasa). El precio es
        # denegar `git checkout feat/x` sobre una rama existente; cuesta un `!` y un falso negativo
        # aqui cuesta trabajo sin commitear. La rama NUEVA no paga nada: `-b`/`-B` sale antes.
        uno = sueltos[0] if sueltos else ""
        if uno in (".", "./") or "/" in uno or FILEISH.search(uno) or "*" in uno:
            return DISCARDS_TREE
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
    if toks[0] in CMD_WRAPPERS:
        # Se juzga cada sufijo en vez de modelar la gramática del envoltorio. Cuesta O(n^2) sobre
        # una lista de tokens corta y no depende de saber cuántos argumentos come `timeout` o
        # `xargs -I{}`; equivocarse en eso deja pasar la forma entera.
        for k in range(1, len(toks)):
            loss = _judge(shlex.join(toks[k:]))
            if loss:
                return loss
    return None


HEREDOC = re.compile(r"<<-?\s*(['\"]?)(\w+)\1")


def _strip_heredocs(cmd: str) -> str:
    """Quita el CUERPO de cada heredoc: es dato, no comandos.

    La defensa contra un mensaje de commit que DESCRIBE un comando era que la línea no empezara por
    él, y separar por `&` la rompió: un trozo nuevo puede empezar por el comando destructivo aunque
    la línea entera sea prosa. El commit de este mismo arreglo se bloqueó a sí mismo así, que es
    exactamente el caso que el fichero de casos ya protegía por el otro lado.

    Descartar el cuerpo entero no pierde cobertura: nada de lo que hay dentro de un heredoc se
    ejecuta. Lo que venga DESPUÉS del marcador de cierre sí se sigue juzgando.
    """
    lineas = cmd.split("\n")
    fuera, i = [], 0
    while i < len(lineas):
        fuera.append(lineas[i])
        marcas = [m.group(2) for m in HEREDOC.finditer(lineas[i])]
        i += 1
        for marca in marcas:
            while i < len(lineas) and lineas[i].strip() != marca:
                i += 1
            i += 1  # la línea de cierre tampoco es un comando
    return "\n".join(fuera)


def verdict(cmd: str) -> str | None:
    """Un comando por segmento: `&&`, `||`, `;`, `|`, `&` y saltos de línea los separan.

    El `&` suelto faltaba, y `&&` lo tapaba: `sleep 1 & rm -rf build` era UN segmento que empezaba
    por `sleep`, así que el `rm -rf` de la derecha no se juzgaba nunca. Va después de `&&` en la
    alternancia, que es lo que impide partir un `&&` por la mitad.
    """
    for seg in re.split(r"&&|\|\||[;|\n&]", _strip_heredocs(cmd)):
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
    """Que el repo además lo DECLARE, con el matcher que hace falta. Comparar sólo el script deja
    pasar la mitad que falta: borrar el bloque `PreToolUse` de su settings.json dejaba ese repo sin
    guardia con este check en verde, que es la misma deriva silenciosa que el check existe para
    cerrar.

    El `matcher` se comprueba porque una declaración con el matcher equivocado —`Read`, `Write`,
    cualquier cosa que no sean comandos— contaba como declarada y no habría corrido nunca sobre un
    `Bash`. El hook son dos mitades y la segunda tiene su propia mitad."""
    cfg = repo_root / ".claude" / "settings.json"
    if not cfg.exists():
        return False
    try:
        hooks = json.loads(cfg.read_text()).get("hooks", {}).get("PreToolUse", [])
    except Exception:
        return False
    return any(
        "guard_destructive.py" in h.get("command", "")
        and re.search(r"\bBash\b", entry.get("matcher", "") or "")
        for entry in hooks
        for h in entry.get("hooks", [])
    )


def check_twin() -> int:
    mine = pathlib.Path(__file__).resolve()
    # PRIMERO el repo desde el que se corre. `_declared` sólo se llamaba sobre el GEMELO, así que
    # el settings.json de este repo —el único desde el que este check se invoca— no lo comprobaba
    # nadie: borrar su bloque `PreToolUse` dejaba el hub sin guardarraíl con el check en verde.
    # Es el mismo agujero que `_declared` dice cerrar, un repo más cerca.
    aqui = mine.parent.parent.parent
    if not _declared(aqui):
        print(
            f"{aqui.name} tiene el script pero su settings.json no lo declara para Bash en "
            "PreToolUse: aqui el guardarrail no corre"
        )
        return 1
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
