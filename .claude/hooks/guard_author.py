#!/usr/bin/env python3
"""Tres exigencias del autor que dependían de la vigilancia, convertidas en una negación previa.

`methodology.md` §L.11 cuenta lo que el autor tuvo que pedir más de una vez, mensaje por mensaje, en
`agent-findings/author-thinking-catalog-2026-10-04.md`. Tres filas marcadas **vigilancia** tienen
una forma que un hook `PreToolUse` puede juzgar sin adivinar:

  P1   «tenías que cargar las skills para ocuparlas sipo» (10 mensajes)  -> herramienta Agent
  P14  «ya no editaremos esas cosas» — P01 y la tesis (4 mensajes)       -> Edit/Write/Bash
  P12  «hazme las preguntas alternativas y recomendado» (3 mensajes)      -> AskUserQuestion

Un hook `PreToolUse` corre aunque la herramienta esté preaprobada y puede negar con motivo; el
motivo le llega al modelo, que puede corregir y reintentar. Es lo único que ataja un olvido antes
de que ocurra, en vez de encontrarlo después.

## Lo medido antes de construirlo (2026-10-04, `--measure` sobre las transcripciones del hub)

| regla | unidad | juzgables | habría negado | lectura |
|---|---|---|---|---|
| P1 skills en el prompt | llamadas a Agent (sin exentos) | 149 | 43 | 37 de 47 antes del 2026-09-22, cuando el autor ya lo pedía; 6 de 102 después |
| P12 opciones + recomendada primera | llamadas a AskUserQuestion | 25 (52 preguntas) | 3 | 2 sí-o-no sin alternativa recomendada; 1 legítima (autoría: decide el autor) |

La de P12 tiene una exención explícita para la tercera: `sin recomendación: <motivo>` en el texto
de la pregunta, visible para el autor.

## Qué NO ve, dicho a propósito

- P1 mira que el prompt **nombre** una skill, no que el subagente la cargue: es lo que el hook puede
  leer. Un subagente que recibe el nombre y no la abre pasa.
- P14 sobre Bash es una lista de comandos que escriben (`>`, `tee`, `sed -i`, `mv`, `cp`, `rm`,
  `git commit`…). Un `python -c "open(f, 'w')"` o un `latexmk` dentro de `clean_source/` escriben
  igual y no están en ella. Es la misma limitación declarada de `guard_destructive.py`, del que
  reutiliza el corte en segmentos y el descarte de heredocs.
- No corre en sesiones abiertas desde el repo del paper (`~/paper-ngc6383-aa52082-24`) ni desde la
  tesis: se declara en `phd` y `erotica`, donde se trabaja.

## La excepción de P14 es del autor, no del modelo

`PHD_EDITAR_CONGELADO=1` en el entorno con el que **el autor lanza** `claude` (pruebas de imprenta,
una fe de erratas). Un `export` dentro de un Bash del modelo no llega al proceso del hook, así que el
modelo no puede abrirse la puerta a sí mismo.

Falla abierto ante cualquier excepción propia: un guardia nunca puede ser la razón de que un turno
se caiga. Por eso su silencio no prueba nada, y `test_guard_author.py` lo mutation-testea.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import re
import shlex
import sys

HOME = pathlib.Path.home()
AQUI = pathlib.Path(__file__).resolve()

# --- P14: lo congelado ---------------------------------------------------------------------------

CONGELADO = (
    HOME / "thesis-test",
    HOME / "paper-ngc6383-aa52082-24" / "submission_package" / "clean_source",
    # `cds_final/` es lo que se entregó al CDS: el repo lo declara fuente de verdad junto al .tex.
    HOME / "paper-ngc6383-aa52082-24" / "cds_final",
)
PERMISO = "PHD_EDITAR_CONGELADO"


def _destructivo():
    """El parser del guardarraíl hermano. Si no carga, P14-Bash no corre: `--check` lo dice."""
    spec = importlib.util.spec_from_file_location("gd", AQUI.with_name("guard_destructive.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _resolver(ruta: str, cwd: str) -> pathlib.Path:
    ruta = os.path.expandvars(os.path.expanduser(ruta))
    ruta = re.split(r"[*?\[]", ruta, maxsplit=1)[0] or "."
    p = pathlib.Path(ruta)
    if not p.is_absolute():
        p = pathlib.Path(cwd) / p
    return pathlib.Path(os.path.realpath(p))


def congelado(ruta: str, cwd: str) -> pathlib.Path | None:
    p = _resolver(ruta, cwd)
    for raiz in CONGELADO:
        r = pathlib.Path(os.path.realpath(raiz))
        if p == r or r in p.parents:
            return r
    return None


ESCRIBEN_TODO = {"rm", "rmdir", "mv", "tee", "touch", "truncate", "unlink", "chmod", "shred"}
ESCRIBEN_DESTINO = {"cp", "rsync", "install", "ln", "ditto"}
CON_I = {"sed", "perl", "gsed"}
GIT_ESCRIBE = {
    "commit",
    "checkout",
    "restore",
    "reset",
    "rm",
    "mv",
    "apply",
    "am",
    "merge",
    "rebase",
    "pull",
    "stash",
    "cherry-pick",
    "revert",
    "clean",
    "switch",
    "add",
}
REDIR = re.compile(r"(?<![<>&\d])\d?>>?\|?\s*([^\s;&|<>]+)")


def _args(toks: list[str]) -> list[str]:
    return [t for t in toks[1:] if not t.startswith("-")]


def juzgar_bash(cmd: str, cwd: str) -> str | None:
    gd = _destructivo()
    cmd = gd._strip_heredocs(cmd)
    for seg in re.split(r"&&|\|\||[;|\n&]", cmd):
        for m in REDIR.finditer(seg):
            destino = m.group(1)
            if destino.startswith("/dev/"):
                continue
            r = congelado(destino, cwd)
            if r:
                return f"redirige salida a {destino}, dentro de {r}"
        try:
            toks = gd._strip_prefixes(shlex.split(seg))
        except ValueError:
            continue
        if not toks:
            continue
        c = os.path.basename(toks[0])
        if c == "cd":
            destino = toks[1] if len(toks) > 1 else str(HOME)
            cwd = str(_resolver(destino, cwd))
            continue
        objetivos: list[str] = []
        if c in ESCRIBEN_TODO:
            objetivos = _args(toks)
        elif c in ESCRIBEN_DESTINO:
            objetivos = _args(toks)[-1:]
        elif c in CON_I and any(t.startswith(("-i", "--in-place")) or t == "-pi" for t in toks):
            objetivos = _args(toks)
        elif c == "git":
            sub, resto = gd._git_subcommand(toks)
            if sub not in GIT_ESCRIBE:
                continue
            repo = cwd
            for k, t in enumerate(toks[:-1]):
                if t == "-C":
                    repo = str(_resolver(toks[k + 1], cwd))
            objetivos = [repo] + [t for t in resto if not t.startswith("-")]
            cwd_git = repo
            for o in objetivos:
                r = congelado(o, cwd_git)
                if r:
                    return f"`git {sub}` escribe en {r}"
            continue
        for o in objetivos:
            r = congelado(o, cwd)
            if r:
                return f"`{c}` escribe en {o}, dentro de {r}"
    return None


# --- P1: skills en el prompt del subagente -------------------------------------------------------

# Los del autor (Explore, claude-code-guide) más dos declarados: `fork` hereda el contexto con las
# skills ya cargadas, y `statusline-setup` sólo edita la configuración de la terminal. `Plan` NO:
# planear un experimento es justo cuando `measurement-design` tiene que estar.
EXENTOS = {"Explore", "claude-code-guide", "statusline-setup", "fork"}
# Agentes que despacha un Workflow de plugin con prompts que ningún modelo puede reescribir: negarlos
# no hace que nombren una skill, sólo rompe el escaneo entero.
PREFIJOS_EXENTOS = ("claude-security:",)
SIN_SKILL = re.compile(r"sin[ -]skill:\s*\S.{8,}", re.I)


def skills_conocidas() -> list[str]:
    nombres: set[str] = set()
    for repo in (HOME / "phd", HOME / "erotica", AQUI.parent.parent.parent):
        d = repo / ".claude" / "skills"
        if d.is_dir():
            nombres.update(p.name for p in d.iterdir() if (p / "SKILL.md").is_file())
    return sorted(nombres)


def nombra_skill(prompt: str, nombres: list[str]) -> bool:
    if ".claude/skills/" in prompt:
        return True
    # `state/data-horizon.yaml` no nombra la skill `data-horizon`: ni ruta antes ni extensión después.
    return any(re.search(rf"(?<![\w/.-]){re.escape(n)}(?![\w-]|\.\w)", prompt) for n in nombres)


def juzgar_agente(entrada: dict) -> str | None:
    tipo = entrada.get("subagent_type") or "general-purpose"
    if tipo in EXENTOS or tipo.startswith(PREFIJOS_EXENTOS):
        return None
    prompt = entrada.get("prompt") or ""
    nombres = skills_conocidas()
    if not nombres or nombra_skill(prompt, nombres) or SIN_SKILL.search(prompt):
        return None
    return (
        "el prompt del subagente no nombra ninguna skill, y el autor pidió diez veces que cada "
        "subagente las use, también al verificar (methodology.md §L.11, P1). Agrega al prompt "
        "«Antes de nada lee y sigue `.claude/skills/<x>/SKILL.md`» con las que apliquen de: "
        + ", ".join(nombres)
        + ". Si de verdad ninguna aplica, escribe en el prompt «sin skill: <motivo>»."
    )


# --- P12: preguntar con alternativas y una recomendada -------------------------------------------

RECOMENDADA = re.compile(r"\((?:Recommended|Recomendad[oa])\b", re.I)
SIN_RECOMENDACION = re.compile(r"sin recomendaci[oó]n:\s*\S", re.I)


def juzgar_pregunta(entrada: dict) -> str | None:
    for q in entrada.get("questions") or []:
        if q.get("multiSelect"):
            continue
        texto = q.get("question") or ""
        if SIN_RECOMENDACION.search(texto):
            continue
        etiquetas = [o.get("label") or "" for o in q.get("options") or []]
        if len(etiquetas) < 2:
            return f"«{texto[:60]}» trae {len(etiquetas)} opción(es): el autor pide alternativas"
        marcadas = [i for i, e in enumerate(etiquetas) if RECOMENDADA.search(e)]
        if not marcadas:
            return f"«{texto[:60]}» no marca ninguna opción «(Recomendado)»"
        if marcadas[0] != 0:
            return f"«{texto[:60]}»: la recomendada va primera, y es la {marcadas[0] + 1}.ª"
    return None


# --- despacho ------------------------------------------------------------------------------------


def veredicto(evento: dict) -> str | None:
    herramienta = evento.get("tool_name") or ""
    entrada = evento.get("tool_input") or {}
    cwd = evento.get("cwd") or os.getcwd()
    if herramienta in ("Agent", "Task"):
        return juzgar_agente(entrada)
    if herramienta == "AskUserQuestion":
        m = juzgar_pregunta(entrada)
        return (
            m
            and m
            + ". Toda pregunta de decisión trae ≥2 opciones y la recomendada primera, con "
            "«(Recomendado)» en su etiqueta y la razón medida en su descripción (§L.11.4). Si la "
            "decisión es del autor y no hay qué recomendar, escribe «sin recomendación: <motivo>» "
            "en la pregunta."
        )
    if os.environ.get(PERMISO) == "1":
        return None
    if herramienta in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
        ruta = entrada.get("file_path") or entrada.get("notebook_path") or ""
        r = congelado(ruta, cwd) if ruta else None
        if r:
            return (
                f"{ruta} está dentro de {r}, que no se edita: P01 está aceptado y la tesis "
                "defendida; se unifica a futuro en trabajo nuevo (§L.11, P14). Si es una prueba de "
                f"imprenta, el autor relanza con {PERMISO}=1."
            )
        return None
    if herramienta == "Bash":
        m = juzgar_bash(entrada.get("command") or "", cwd)
        return m and (
            f"este comando {m}, que no se edita (§L.11, P14). Si es una prueba de imprenta, el "
            f"autor relanza con {PERMISO}=1."
        )
    return None


# --- --check: las dos mitades en los dos repos ---------------------------------------------------

MATCHER_NECESARIO = ("Agent", "AskUserQuestion", "Edit", "Write", "NotebookEdit", "Bash")


def _declarado(repo: pathlib.Path) -> list[str]:
    cfg = repo / ".claude" / "settings.json"
    if not cfg.is_file():
        return list(MATCHER_NECESARIO)
    try:
        entradas = json.loads(cfg.read_text()).get("hooks", {}).get("PreToolUse", [])
    except Exception:
        return list(MATCHER_NECESARIO)
    cubiertas: set[str] = set()
    for e in entradas:
        if any(AQUI.name in h.get("command", "") for h in e.get("hooks", [])):
            cubiertas.update((e.get("matcher") or "").split("|"))
    return [m for m in MATCHER_NECESARIO if m not in cubiertas]


def check() -> int:
    problemas = []
    try:
        gd = _destructivo()
        gd._strip_heredocs("x")
    except Exception as e:
        problemas.append(f"no carga guard_destructive.py ({e}): P14 sobre Bash falla abierto")
    aqui = AQUI.parent.parent.parent
    faltan = _declarado(aqui)
    if faltan:
        problemas.append(f"{aqui.name}/.claude/settings.json no declara {AQUI.name} para {faltan}")
    gemelo_repo = None
    if aqui.name in ("phd", "erotica"):
        gemelo_repo = aqui.parent / ("erotica" if aqui.name == "phd" else "phd")
    if gemelo_repo is None:
        problemas.append(f"no sé cuál es el repo gemelo de {aqui}")
    elif not gemelo_repo.is_dir():
        print(f"omitido: {gemelo_repo} no está; no hay copia gemela que comparar")
    else:
        gemelo = gemelo_repo / ".claude" / "hooks" / AQUI.name
        if not gemelo.is_file():
            problemas.append(f"{gemelo_repo.name} tiene el repo pero no {gemelo}")
        elif gemelo.read_bytes() != AQUI.read_bytes():
            problemas.append(f"las dos copias divergieron: {AQUI} != {gemelo}")
        else:
            faltan = _declarado(gemelo_repo)
            if faltan:
                problemas.append(
                    f"{gemelo_repo.name}/.claude/settings.json no declara {AQUI.name} para {faltan}"
                )
    for p in problemas:
        print(f"  - {p}")
    return 1 if problemas else 0


def medir(carpeta: str) -> int:
    """Cuántas llamadas reales de las transcripciones habría negado. No escribe nada."""
    vistas, contadas = set(), {"Agent": [0, 0], "AskUserQuestion": [0, 0]}
    for f in sorted(pathlib.Path(carpeta).expanduser().rglob("*.jsonl")):
        for linea in f.open(errors="ignore"):
            try:
                m = json.loads(linea).get("message") or {}
            except Exception:
                continue
            for b in m.get("content") or [] if isinstance(m, dict) else []:
                if not isinstance(b, dict) or b.get("type") != "tool_use":
                    continue
                n = b.get("name")
                if n not in ("Agent", "Task", "AskUserQuestion"):
                    continue
                clave = json.dumps(b.get("input"), sort_keys=True)
                if clave in vistas:
                    continue
                vistas.add(clave)
                k = "AskUserQuestion" if n == "AskUserQuestion" else "Agent"
                entrada = b.get("input") or {}
                tipo = entrada.get("subagent_type") or ""
                if k == "Agent" and (tipo in EXENTOS or tipo.startswith(PREFIJOS_EXENTOS)):
                    continue
                contadas[k][0] += 1
                juez = juzgar_agente if k == "Agent" else juzgar_pregunta
                contadas[k][1] += bool(juez(entrada))
    for k, (n, malas) in contadas.items():
        print(f"{k}: {n} llamadas juzgables, {malas} negadas")
    return 0


def main() -> int:
    if "--check" in sys.argv:
        return check()
    if len(sys.argv) > 2 and sys.argv[1] == "--measure":
        return medir(sys.argv[2])
    try:
        raw = sys.stdin.read()
        evento = json.loads(raw) if raw.strip() else {}
        motivo = veredicto(evento)
        if motivo:
            print(
                json.dumps(
                    {
                        "hookSpecificOutput": {
                            "hookEventName": "PreToolUse",
                            "permissionDecision": "deny",
                            "permissionDecisionReason": f"Exigencia del autor: {motivo}",
                        }
                    },
                    ensure_ascii=False,
                )
            )
    except Exception:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
