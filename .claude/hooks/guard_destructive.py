#!/usr/bin/env python3
"""Deniega comandos que borran trabajo no respaldado, ANTES de que corran.

Por qué existe, con el incidente medido:

`~/.claude/settings.json` trae `Bash(git *)` en su lista de permitidos, así que **cada** comando
de git corre sin preguntar. El 2026-08-23, revirtiendo una mutación de prueba, un
`git checkout -- <fichero>` se llevó por delante una edición real de la Sect. 8 del manuscrito de
P01 que no estaba commiteada. No hubo prompt porque no podía haberlo: el patrón estaba
preaprobado. De ahí salió la regla de esa sesión — *las mutaciones se revierten desde copias en
`/tmp`, nunca desde git* — y una regla que depende de acordarse no es un guardarraíl.

Un hook `PreToolUse` sí corre sobre comandos preaprobados, y puede negar. Eso es lo único que
ataja este caso.

Adaptado del `check-destructive.sh` de `bukhr/datawarehouse`, con la lista reescrita: la de allá
cubre `rm -rf`, `git reset --hard`, `git push --force` y DDL de SQL, y **no cubre
`git checkout --`**, que es justo el comando del incidente. Absorber su regex verbatim habría
dejado un guardia que no atrapa la única pérdida que lo justifica. Se quedan sus tres entradas de
git/fs, se van las de SQL y Databricks — acá no hay warehouse.

Lo que NO ve, dicho a propósito:

- `git checkout <fichero>` sin `--` es igual de destructivo, y es indistinguible de
  `git checkout <rama>` sin consultar el árbol. No se intenta: un guardia que adivina mal molesta
  en cada cambio de rama y se termina desactivando.
- Un `python -c "open(f,'w')"` o un `>` de shell borran igual y no matchean nada de esto.
- Es una lista negra, así que sólo ve lo que alguien ya nombró.

Falla abierto ante cualquier excepción propia: un guardia nunca puede ser la razón de que un turno
se caiga. Eso significa que su silencio NO es prueba de que revisó — por eso se mutation-testea.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys

# Cada entrada es (patrón, qué se pierde). El motivo viaja con el patrón porque un "bloqueado por
# política" sin decir qué se pierde se lee como un obstáculo y se rodea; decir "esto descarta tu
# edición sin commitear" es lo que hace que alguien se detenga a mirar.
RULES: list[tuple[str, str]] = [
    (r"\bgit\s+checkout\s+(\S+\s+)?--\s", "descarta cambios sin commitear del árbol de trabajo"),
    (r"\bgit\s+restore\b", "descarta cambios sin commitear del árbol de trabajo"),
    (r"\bgit\s+clean\s+-\w*f", "borra ficheros sin trackear, que suelen ser trabajo en curso"),
    (r"\bgit\s+reset\s+--hard\b", "descarta commits y el árbol de trabajo a la vez"),
    (r"\bgit\s+push\s+(--force(?!-with-lease)|-f)\b", "reescribe la rama remota"),
    (r"\brm\s+-\w*[rR]\w*f|\brm\s+-\w*f\w*[rR]", "borra recursivamente sin papelera"),
]

HINT = (
    "Para revertir una mutación de prueba: respalda antes en /tmp y restaura desde ahí. "
    "Si esto es intencional, dilo explícitamente y córrelo tú con `! <comando>`."
)


# Un comando empieza al principio de la cadena, o después de `&&`, `||`, `;` o un salto de línea.
# Sin este ancla el guardia bloqueó su PROPIO commit: el mensaje *describía* el incidente y contenía
# la frase `git checkout --` dentro de un heredoc. Es la misma forma que K.1.6d — una cita es
# indistinguible de una orden para un matcher — y se arregla igual: anclando en vez de buscar en
# cualquier parte.
#
# Residuo aceptado y dicho en voz alta: una línea de heredoc que EMPIECE con uno de estos comandos
# se marca igual, y una cadena que contenga `; git restore x` también. Cerrar eso pide entender
# comillas y heredocs, o sea un parser de shell, y un guardia que intenta ser un parser falla de
# maneras más raras que el caso que arregla. Cuando estorbe, el trabajo se escribe a un fichero y se
# corre desde ahí — que es como se prueba este mismo guardia.
BOUNDARY = r"(?:^|[\n;|]|&&|\|\|)\s*"


def verdict(cmd: str) -> str | None:
    for pattern, loss in RULES:
        if re.search(BOUNDARY + pattern, cmd):
            return loss
    return None


# La misma lista tiene que proteger los dos repos donde se edita trabajo no commiteado: este hub y
# `erotica`, donde ocurrió el incidente. Son repos separados con remotos separados, así que el
# script viaja duplicado -- y CLAUDE.md dice que un hook son dos mitades, script y declaración, y
# que las dos viajan. Apuntar la declaración de erotica a la copia del hub por ruta absoluta
# resolvería la duplicación y falla abierto si el hub no está, que es peor: quedaría sin guardia
# creyendo que lo tiene. Se duplica, y `--check` compara las copias byte a byte.
#
# El fichero está en formato `ruff format` aunque este repo no corre ruff: el pre-commit de erotica
# sí, así que reformatea su copia en cada commit y las separa. Mantener las dos en el formato del
# repo más estricto es lo único que las deja quietas.
TWIN = pathlib.Path.home() / "erotica" / ".claude" / "hooks" / "guard_destructive.py"


def check_twin() -> int:
    mine = pathlib.Path(__file__).resolve()
    if not TWIN.exists():
        print(f"falta la copia gemela en {TWIN}: erotica queda sin guardarraíl")
        return 1
    if TWIN.resolve() == mine:
        return 0
    if TWIN.read_bytes() != mine.read_bytes():
        print(f"las dos copias del guardarraíl divergieron: {mine} != {TWIN}")
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
