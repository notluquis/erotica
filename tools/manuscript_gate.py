#!/usr/bin/env python3
"""Reusable engine behind every paper's `gate.py` -- the command that has to pass before a
manuscript is uploaded or committed.

Extracted 2026-09-15 from the NGC 6383 (aa52082-24) gate, the only manuscript this repo had
CI for. That gate carried 26 checks and none of them was really about NGC 6383: a letter that
promises the referee something the paper no longer says, a marked diff that shows moved
figures as deleted ones, a CDS ReadMe that drifted from what NESTOR actually holds, a
knowledge-graph note quoting a superseded value, line numbers printed over the appendix text.
Every one of those failure classes can hit a second paper, so the checks moved here and each
paper supplies its own `gate.toml` -- which files, which word lists, which of the 26 checks do
not apply and why.

What did NOT move: the four checks that only make sense for NGC 6383 (Table 1's internal
consistency, the two literature-table cross-checks, the catalogue-derived numbers) stay in
that paper's own `gate_local.py`, loaded through the `local` key in its toml.

This is the canonical copy. The NGC 6383 paper repo (`paper-ngc6383-aa52082-24`, extracted
2026-09-22) vendors a copy of this file verbatim, pinned by commit + sha256 in its own
`tools/VENDORED.toml` -- editing this engine here does not update that copy until it is
re-vendored on purpose.

Usage (from a paper's own `gate.py` shim, never by running this file directly)
-------------------------------------------------------------------------------
    import manuscript_gate as mg
    mg.configure(Path(__file__).parent / "gate.toml")
    raise SystemExit(mg.main())

Exit code is 0 only if every applicable check passes. Each check prints what it compared, not
just a verdict, so a failure says what to look at -- that discipline is the one thing this file
must never relax, whatever else it makes configurable.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

# latexmk escribia el PDF y el log ENCIMA de los ficheros trackeados, asi que cada corrida lenta
# dejaba dos binarios de ~6 MB modificados en `git status`. El parche anterior era restaurar los
# bytes originales cuando el TEXTO extraido no cambiaba, y eso tenia cuatro modos de falla medidos:
# una figura regenerada no cambia el texto y se revertia en silencio; si `pdftotext` fallaba en los
# dos PDF las dos cadenas quedaban vacias y el build era un no-op permanente; la restauracion
# devolvia el PDF viejo justo antes de que c_deliverables leyera su /Producer, tapando el desajuste
# de motor de TeX que ese check acababa de volver fatal; y si el build fallaba no restauraba nada,
# que era el unico caso donde el arbol quedaba peor. Ninguna senal disponible decide bien: qpdf no
# esta y `pdfimages` solo ve 8 rasters porque las figuras A&A son PDF vectoriales.
# Se construye en un subdirectorio propio. El arbol trackeado no se toca, no hay nada que restaurar,
# y c_deliverables compara contra un build FRESCO, que es lo que siempre quiso comparar.
BUILD_DIR = "_gate_build"


def build_paths(tex: Path) -> tuple[Path, Path]:
    """(pdf, log) que produce el gate para `tex`, fuera del arbol trackeado."""
    stem = tex.with_suffix("").name
    return tex.parent / BUILD_DIR / f"{stem}.pdf", tex.parent / BUILD_DIR / f"{stem}.log"


def pages_in(log_text: str, stem: str) -> int | None:
    """Las paginas que declara un log de latexmk.

    Con `-outdir` la linea trae el prefijo del directorio, y el patron anclado en `{stem}.pdf`
    dejaba de casar -- devolviendo "no se pudo leer las paginas" sobre un build correcto. Con
    ruta ABSOLUTA no basta con tolerar el prefijo: LaTeX parte la linea del log a 79 caracteres
    y el nombre queda cortado en dos. Por eso el outdir es relativo, y esto tolera el prefijo.
    """
    m = re.search(rf"Output written on (?:\S*/)?{stem}\.pdf \((\d+) pages", log_text)
    return int(m.group(1)) if m else None


# --------------------------------------------------------------------------- registro de checks

results: list[tuple[str, bool, str]] = []
# @check registers a name only when the function is *called*, so a check that main() forgets to
# call is indistinguishable from one that passes -- which is how c_overclaim was written, printed
# nothing, and left the gate reporting 10/10. Declared names are recorded at import; main() checks
# every one of them ran (or was marked n/a, or is slow under --quick).
declared: list[str] = []
slow_checks: set[str] = set()
skipped: list[tuple[str, str]] = []
not_applicable_ran: list[tuple[str, str]] = []
# Dos registros paralelos: CHECKS por nombre corto (lo que usa `not_applicable` y las secciones del
# toml, p.ej. "kb" para c_kb) y BY_NAME por el string declarado en @check (lo que usa main() para
# despachar en orden de declaracion, agrupado por `group`).
CHECKS: dict[str, Callable] = {}
BY_NAME: dict[str, Callable] = {}


class Skipped(Exception):
    """Un check cuyo insumo vive fuera de este repo, o de este entorno, y no esta presente aca.

    Existe porque el gate se declaro "el duenio interino del manuscrito" y se cableo a CI, donde
    dos checks no podian pasar jamas: uno lee `~/phd/kb`, que es OTRO repo, y el runner no lo
    tiene. El resultado fue el peor de los dos mundos -- el gate local leia 26/26 mientras el
    workflow fallaba en las cinco corridas seguidas desde el 2026-08-17, y nadie miraba.

    La salida no es saltar en silencio: eso es exactamente el modo de falla que este repo
    persigue en todo lo demas. Un salto se imprime, se cuenta aparte en el resumen, y nombra su
    motivo, y desde 2026-08-24 **una omision no sale 0** salvo con `--allow-skips`, que solo pasa
    CI -- asi que en la maquina que sube una omision impide bendecir el paquete.

    La regla, corregida: **se omite cuando el insumo no puede existir en este entorno, y se falla
    cuando podria existir y no esta.** El repo `kb` en un runner y el binario `typos` fuera de CI
    no pueden estar; una nota borrada teniendo el repo, o un fichero que el MANIFEST declara
    indispensable, si podrian.

    `Skipped` no es lo mismo que `not_applicable` del toml. `Skipped` lo decide el CODIGO del
    check, en tiempo de ejecucion, sobre el ENTORNO (CI vs local, binario en PATH o no) -- y es
    igual para todos los papers. `not_applicable` lo decide la CONFIGURACION, por paper, sobre si
    el check tiene sentido para ESE manuscrito (una primera submision sin cartas de referee, por
    ejemplo) -- y el check ni siquiera se llama. Confundir los dos convertiria una omision de
    entorno en una excusa de contenido, o viceversa.
    """


def check(name: str, group: str = "consistencia", slow: bool = False):
    """Register a check. `slow` marks the ones that need a LaTeX run, at the single place that
    knows: the declaration. `group` es el encabezado bajo el que main() lo imprime -- tres grupos
    existen: "consistencia", "latex", "compilacion" -- y reemplaza la posicion en el fichero como
    forma de agrupar, porque main() ya no llama a cada check a mano."""

    def deco(fn):
        short = fn.__name__[2:] if fn.__name__.startswith("c_") else fn.__name__

        def wrapper(*a, **kw):
            # not_applicable se consulta ANTES de intentar nada: el check ni se llama. Es distinto
            # de Skipped (mas arriba) -- no cuenta como omitido, no bloquea la bendicion, y se
            # imprime con su propio prefijo para que no se confunda con un salto de entorno.
            if short in NOT_APPLICABLE:
                motivo = NOT_APPLICABLE[short]
                not_applicable_ran.append((name, motivo))
                print(f"no aplica  {name}: {motivo}")
                return True
            try:
                ok, detail = fn(*a, **kw)
            except Skipped as exc:
                skipped.append((name, str(exc)))
                results.append((name, True, f"omitido: {exc}"))
                print(f"omite  {name}: {exc}")
                return True
            except Exception as exc:  # a check that crashes is a failed check
                ok, detail = False, f"{type(exc).__name__}: {exc}"
            results.append((name, ok, detail))
            print(f"{'ok    ' if ok else 'FALLA '} {name}: {detail}")
            return ok

        wrapper.check_name = name
        wrapper.short_name = short
        wrapper.group = group
        declared.append(name)
        if slow:
            slow_checks.add(name)
        CHECKS[short] = wrapper
        BY_NAME[name] = wrapper
        return wrapper

    return deco


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


# ------------------------------------------------------------------------------ configuracion

TEX: Path | None = None
MARKED: Path | None = None
LETTERS: list[Path] = []
CDS_DAT: Path | None = None
CDS_README: Path | None = None
KB_ROOT: Path | None = None
KB_NOTES: list[Path] = []
DOCS: list[tuple[str, Path]] = []
NOT_APPLICABLE: dict[str, str] = {}

_CONFIG: dict = {}
_BASE: Path | None = None
_configured = False


def _resolve(base: Path, s: str) -> Path:
    p = Path(s).expanduser()
    return p if p.is_absolute() else (base / p)


def _no_empty_strings(node, path: str) -> None:
    """Un `""` en una lista de configuracion (una palabra, una excepcion, un guion aceptado)
    casa CUALQUIER COSA donde esa lista se usa como filtro -- `"" in frag` es `True` siempre, y
    `ACCEPTED_DASH` vacio en `dash_ok` deja de discriminar nada. Camina el toml entero, no una
    lista a mano, porque la proxima seccion que necesite esta proteccion no debe tener que
    acordarse de pedirla."""
    if isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, str) and v == "":
                raise SystemExit(
                    f"gate.toml: {path}[{i}] es una cadena vacia; en cualquier lista usada como "
                    "filtro (palabras, excepciones, guiones aceptados) eso acepta todo"
                )
            _no_empty_strings(v, f"{path}[{i}]")
    elif isinstance(node, dict):
        for k, v in node.items():
            _no_empty_strings(v, f"{path}.{k}" if path else k)


def configure(toml_path: Path, base: Path | None = None) -> dict:
    """Carga `gate.toml`, resuelve las rutas top-level y, si el paper lo declara, sus checks
    locales -- en ese orden, porque validar `not_applicable` necesita ver el roster COMPLETO de
    checks conocidos (los 22 genericos mas los locales), no solo los genericos.

    No se puede llamar dos veces en el mismo proceso: una segunda carga re-ejecutaria el modulo
    `local` y duplicaria sus entradas en `declared`, que es justo el conteo que
    `test_gate_mutations.py` usa para asegurarse de que no quedo ningun check sin mutacion.
    """
    global _configured, _CONFIG, _BASE, TEX, MARKED, LETTERS, CDS_DAT, CDS_README
    global KB_ROOT, KB_NOTES, DOCS, NOT_APPLICABLE

    if _configured:
        raise SystemExit(
            "manuscript_gate.configure() ya corrio en este proceso; no se puede "
            "reconfigurar (cada paper corre en su propio proceso)"
        )
    _configured = True

    toml_path = Path(toml_path).resolve()
    # `base` existe para que un test pueda leer una copia mutada del toml desde otro directorio
    # sin que sus rutas relativas dejen de apuntar al paper (ver GATE_TOML en el shim de P01).
    base = Path(base).resolve() if base is not None else toml_path.parent
    with toml_path.open("rb") as fh:
        cfg = tomllib.load(fh)

    _no_empty_strings(cfg, "")

    for key in ("tex", "letters", "cds_dat", "cds_readme"):
        if key not in cfg:
            raise SystemExit(f"gate.toml: falta la clave obligatoria '{key}'")

    TEX = _resolve(base, cfg["tex"])
    MARKED = _resolve(base, cfg["marked"]) if cfg.get("marked") else None
    LETTERS = [_resolve(base, s) for s in cfg["letters"]]
    CDS_DAT = _resolve(base, cfg["cds_dat"])
    CDS_README = _resolve(base, cfg["cds_readme"])
    # Los documentos a compilar: MARKED se omite solo si el paper no lo configuro (una primera
    # submision no tiene diff marcado que mostrar).
    DOCS = [("limpio", TEX)] + ([("marcado", MARKED)] if MARKED else [])

    kb_cfg = cfg.get("kb")
    if kb_cfg:
        KB_ROOT = Path(kb_cfg["root"]).expanduser()
        KB_NOTES = [KB_ROOT / n for n in kb_cfg["notes"]]
    else:
        KB_ROOT = None
        KB_NOTES = []

    if cfg.get("local"):
        local_path = _resolve(base, cfg["local"])
        spec = importlib.util.spec_from_file_location("manuscript_gate_local", local_path)
        local_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(local_mod)
        engine = sys.modules[__name__]
        for attr_name in dir(local_mod):
            if not attr_name.startswith("c_"):
                continue
            obj = getattr(local_mod, attr_name)
            if not hasattr(obj, "check_name"):
                raise SystemExit(
                    f"{local_path.name}: {attr_name} empieza con 'c_' pero no esta "
                    "decorado con @manuscript_gate.check"
                )
            setattr(engine, attr_name, obj)

    # not_applicable, validado DESPUES de cargar `local`: sus claves son nombres cortos de check
    # (el mismo vocabulario que las secciones -- "kb", no "c_kb" ni el string largo declarado en
    # @check), y deben nombrar un check que exista de verdad.
    na = cfg.get("not_applicable", {})
    for k, v in na.items():
        if k not in CHECKS:
            raise SystemExit(
                f"gate.toml: not_applicable declara '{k}', que no es ningun check "
                f"conocido ({sorted(CHECKS)})"
            )
        if not v:
            raise SystemExit(f"gate.toml: not_applicable.{k} no trae motivo")
    NOT_APPLICABLE = dict(na)

    _CONFIG = cfg
    _BASE = base
    return cfg


# --------------------------------------------------------------------------- consistency


@check("posterior claims")
def c_posterior():
    script = _resolve(_BASE, _CONFIG["posterior"]["script"])
    r = run([sys.executable, str(script)])
    tail = r.stdout.strip().split("\n")[-1] if r.stdout else "sin salida"
    return r.returncode == 0, tail


@check("cifras de las cartas sin deriva contra el manuscrito")
def c_letter_numbers():
    """Catch a letter quoting a *slightly different* value from the manuscript.

    "Absent from the manuscript" is the wrong test: the letters legitimately quote
    superseded values when disclosing a correction ("t_rh becomes 24.7 (was 30.5)") and
    refit results that only the letter reports. The defect class is narrower and nastier --
    the same quantity rounded twice. R7 quoted T_max = 42.45 arcmin against Table 1's 42.5,
    a third rounding of 42.4667, which no "is it present" check can see because both look
    like ordinary numbers.

    So: for every number in a letter that is not in the manuscript verbatim, ask whether the
    manuscript holds a number within 0.5% of it. If it does, the two are almost certainly
    the same quantity written twice. Values quoted as history ("was X", "from X to") are
    exempt, since disagreeing with the current value is the whole point of quoting them.
    """
    tex = TEX.read_text()
    tex_nums = sorted({float(x) for x in re.findall(r"(?<![\w.])\d+\.\d{1,4}(?![\w.])", tex)})
    drift = []
    for path in LETTERS:
        text = path.read_text()
        for m in re.finditer(r"(?<![\w.])(\d+\.\d{1,4})(?![\w.])", text):
            num = m.group(1)
            if num in tex:
                continue
            before = text[max(0, m.start() - 40) : m.start()].lower()
            if re.search(r"\bwas\b|\bfrom\b|\bpreviously\b|\binstead of\b", before):
                continue  # quoted as the superseded value on purpose
            v = float(num)
            near = [t for t in tex_nums if t and abs(t - v) / max(abs(v), 1e-9) < 0.005 and t != v]
            if near:
                drift.append(f"{path.name}: carta {num} vs manuscrito {near[0]}")
    return not drift, ("sin deriva" if not drift else f"{len(drift)} -> {drift[:5]}")


@check("grafo de conocimiento al dia con el manuscrito")
def c_kb():
    """The KB note on this paper must not quote superseded values.

    It quoted R_t = 40.4 arcmin -- the round-1 number the referee's R11 attacked -- for a
    month after the manuscript adopted 54 arcmin. Consulting it would have been worse than
    not consulting it, which is the reason it went unconsulted.
    """
    # La condicion es el ENTORNO, no la ruta. Antes bastaba con que `~/phd/kb` no estuviera, asi
    # que un `git worktree`, un clon en otro sitio o una reorganizacion apagaban en silencio el
    # unico check que existe porque la nota cargo el R_t superado durante un mes -- y lo apagaban
    # justo en la maquina donde es load-bearing. En CI no puede estar; en local, si no esta, falta.
    if os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI"):
        raise Skipped("CI no clona el repo hermano `kb`; este check corre en local")
    if not KB_ROOT.is_dir():
        return False, f"{KB_ROOT} no esta: el grafo no se pudo comprobar (fetch de phd-kb?)"
    missing = [p for p in KB_NOTES if not p.exists()]
    if missing:
        return False, f"notas no encontradas: {[p.name for p in missing]}"
    note = "\n".join(p.read_text() for p in KB_NOTES)
    tex = TEX.read_text()
    anchors = {label: (tex_re, note_re) for label, tex_re, note_re in _CONFIG["kb"]["anchors"]}
    stale = []
    for label, (in_tex, in_note) in anchors.items():
        if re.search(in_tex, tex) and re.search(in_note, note):
            stale.append(label)
    return not stale, (
        "sin cifras superadas" if not stale else f"la nota cita valores de ronda 1: {stale}"
    )


@check("registro del manuscrito")
def c_register():
    """Catch commentary on our own reasoning, and instructions standing in for statements.

    An editorial pass in Sect. 69 removed seven of these by hand and left nothing behind to catch
    the eighth, which is the failure this file exists to stop. The list is closed and specific
    rather than a general prose-quality opinion: each entry was actually written into this
    manuscript and actually removed.
    """
    phrases = _CONFIG["register"]["phrases"]
    text = TEX.read_text().lower()
    found = [phrase for phrase in phrases if phrase in text]
    return not found, "sin muletillas de registro" if not found else f"{len(found)} -> {found}"


@check("la carta no afirma mas fuerte que el manuscrito")
def c_overclaim():
    # No manuscript-wide exemption. The first version allowed a word in the letters as soon as it
    # occurred anywhere in the paper, in any sense: "unambiguously" is in Sect. 3.2 about branch
    # selection, which would have exempted it from an R11 contamination claim -- the exact
    # assertion this list exists to stop. Presence in the manuscript is not assertion of the same
    # claim. None of these words is needed in a letter; if one ever is, it goes on the list's
    # exception with the sentence that earns it.
    words = _CONFIG["overclaim"]["words"]
    found = []
    for path in LETTERS:
        text = path.read_text().lower()
        found += [f"{path.name}:{w}" for w in words if w in text]
    return not found, "sin sobre-afirmacion" if not found else f"{len(found)} -> {found}"


@check("las cartas no perdieron simbolos al pasar a texto plano")
def c_dropped_symbols():
    """A Greek letter that vanished in the conversion to .txt leaves a subscript with no symbol.

    NESTOR accepts only .pdf or .txt for the response, so the letters are plain text and every
    symbol is transliterated by hand -- "lambda", "p-tilde", "R-hat". Five did not survive: the
    referee's R2 read `p_HDBSCAN,i=_i/_max`, R3 twice referred to `_max`, R4 to the parallax
    dispersion `_parallax`, and R9 to a mixing length `(_MLT=1.82)`. Each is a definition the
    referee had explicitly asked for, delivered as a bare underscore.

    An underscore opening a subscript after a space, '=' or '(' is exactly that signature, and it
    cannot occur legitimately here: every real identifier in these letters starts with a letter
    (t_rh, T_max, R_t, sigma_parallax). Cheap, and general -- it is about the file format, not
    about this paper.
    """
    # Two signatures, because the conversion failed in two ways. A subscript opening on nothing
    # is the dropped symbol (`_max`); a doubled underscore is the symbol collapsing into the
    # underscore that introduced it (`R__sun` from R_\odot). Neither can occur legitimately: every
    # identifier in these letters starts with a letter, and none has an empty subscript level.
    bad = []
    for path in LETTERS:
        text = path.read_text()
        for m in re.finditer(r"(?<=[\s=(/])_[A-Za-z]|__", text):
            bad.append(
                f"{path.name}: ...{' '.join(text[max(0, m.start() - 30) : m.start() + 14].split())}"
            )
        # Third signature, from re-wrapping rather than from symbol loss: a line-end hyphen whose
        # two halves were merged onto one line without dropping it ("a tie- breaker"). Suspended
        # hyphens are legitimate and are the only exception ("window- or model-dependent").
        for m in re.finditer(r"[a-z]- [a-z]", text):
            frag = text[max(0, m.start() - 12) : m.end() + 10]
            if re.search(r"- (or|and|to|nor)\b", frag):
                continue
            bad.append(f"{path.name}: guion partido ...{' '.join(frag.split())}")
    return not bad, ("sin subindices sin simbolo" if not bad else f"{len(bad)} -> {bad[:4]}")


@check("ortografia US, como la carta le promete al editor")
def c_spelling():
    british = _CONFIG["spelling"]["british"]
    exceptions = _CONFIG["spelling"]["exceptions"]
    bad = []
    for path in [TEX] + LETTERS:
        text = path.read_text()
        for word in british:
            for m in re.finditer(rf"\b{word}\b", text, re.I):
                frag = " ".join(text[max(0, m.start() - 70) : m.end() + 40].split())
                # The licensed exceptions: CDS mandates the wording of the VizieR
                # acknowledgement, and the cover letter quotes the word to declare it.
                if any(exc in frag for exc in exceptions):
                    continue
                bad.append(f"{path.name}: {word} ...{frag[:70]}")
    return not bad, (
        "sin formas britanicas fuera de las excepciones declaradas"
        if not bad
        else f"{len(bad)} -> {bad[:3]}"
    )


@check("las cartas no citan secciones que el manuscrito no tiene")
def c_section_refs():
    """A reply that points at a section number the PDF does not contain reads as a stale PDF.

    The A&A editorial office wrote on 2026-08-17: "It looks like the compiled PDF is not the
    revised version: there is for instance no Sect. 3.1.1." The PDF was correct. The response
    letter said "The revised Sect. 2.1.1 now defines...", attaching *revised* to a round-1 number,
    while its own mapping sends 2.1 to 3.1 -- so a reader looking for the revised counterpart hunts
    for 3.1.1. The restructuring removed all nine subsubsections, so no Sect. X.Y.Z exists at all.

    Only numbers presented as belonging to the revised manuscript are checked. The reply headers
    deliberately carry both ("Sect. 2.1.1 -> 3.2") and the mapping table lists every round-1
    number, so both are skipped; what must resolve is any other "Sect. N" in the prose.
    """
    tex = TEX.read_text()
    body = tex[: tex.find(r"\begin{appendix}")] if r"\begin{appendix}" in tex else tex
    have, sec, sub = set(), 0, 0
    for m in re.finditer(r"\\(subsection|section)\{", body):
        if m.group(1) == "section":
            sec, sub = sec + 1, 0
            have.add(str(sec))
        else:
            sub += 1
            have.add(f"{sec}.{sub}")
    bad = []
    for path in LETTERS:
        text = path.read_text()
        for m in re.finditer(r"Sects?\.\s+(\d+(?:\.\d+)*)", text):
            # The arrow form presents both numbers on purpose -- reply headers and the note that
            # explains them -- and the mapping table lists every round-1 number by design.
            line = text[text.rfind("\n", 0, m.start()) + 1 : text.find("\n", m.end())]
            # A line using the arrow form is presenting both numbers on purpose: the reply
            # headers, the note that explains them, and R7's compound "2.1.5 / 2.2 -> 5 / 6.1".
            # The mapping table lists every round-1 number by design.
            if "->" in line or "|" in line:
                continue
            if m.group(1) not in have:
                bad.append(f"{path.name}: Sect. {m.group(1)} no existe en el manuscrito")
    return not bad, (
        f"todas las secciones citadas existen ({len(have)} en el manuscrito)"
        if not bad
        else f"{len(bad)} -> {sorted(set(bad))[:4]}"
    )


@check("copias de las cartas y del ReadMe sincronizadas")
def c_copies():
    pairs = [(_resolve(_BASE, a), _resolve(_BASE, b)) for a, b in _CONFIG["copies"]["pairs"]]
    bad = [f"{a.name}" for a, b in pairs if not b.exists() or a.read_bytes() != b.read_bytes()]
    return not bad, "todas iguales" if not bad else f"divergen: {bad}"


@check("la carta no declara el paquete CDS sin cambios si cambio")
def c_cds_claim():
    """The cover letter told the editor the CDS package was "unchanged from the previous
    submission" while its ReadMe had been corrected in the same round.

    That is not a wording slip: the upload plan is to *replace* the archive on NESTOR, and an
    editor reading "unchanged" has no reason to process a replacement. CDS would then receive the
    ReadMe that omits the null marker on the three 2MASS columns -- the defect that would have
    bounced the package. The superseded archive is kept under _legacy/, so the two ReadMes can
    simply be compared.

    ⚠ The baseline zip is load-bearing and must not be deleted while this paper is under review:
    it is the only local record of what NESTOR actually holds, and without it there is no way to
    tell whether the letter's account of the dataset is honest. A missing baseline fails rather
    than skips, on purpose -- a skip here would restore exactly the silence this check exists to
    break.
    """
    legacy = _resolve(_BASE, _CONFIG["cds_claim"]["submitted_zip"])
    if not legacy.exists():
        return False, (
            f"falta {legacy}, la referencia de lo que hay en NESTOR; restaurala desde "
            "git en vez de saltarse el chequeo"
        )
    import zipfile

    with zipfile.ZipFile(legacy) as z:
        old = next((n for n in z.namelist() if n.endswith("ReadMe") and "__MACOSX" not in n), None)
        if old is None:
            return False, "el zip ya subido no trae ReadMe"
        if z.read(old) == CDS_README.read_bytes():
            return True, "el ReadMe no cambio; nada que declarar"
    # It did change, so the letter has to say so. Asserted positively rather than by banning the
    # word "unchanged": the letter can be misleading without using it, and the editor needs to
    # know a replacement is coming, not merely to not be told the opposite.
    text = " ".join(LETTERS[0].read_text().lower().split())
    disclosed = "readme" in text and "replacement" in text
    return disclosed, (
        "el ReadMe cambio y la carta lo declara"
        if disclosed
        else "el ReadMe cambio y la carta de presentacion no lo declara"
    )


@check("el diff marcado deriva del manuscrito actual")
def c_marked_fresh():
    """The marked PDF is built from a *copy* of the manuscript, so editing one leaves the other.

    latexdiff runs on marked_changes/new_revised.tex, which MANIFEST.md tells you to `cp` from
    the paper's own clean_source/*.tex first (aanda.tex for NGC 6383, at the time this was
    written). Skip the copy and the marked PDF silently shows the previous revision -- the
    referee then reads a diff that omits the change they asked for. Two edits on 2026-08-17
    (the Appendix D wording and the Sect. 8 lead-in) left it stale with every other check
    green, which is what this catches: byte equality of the copy, and a marked source at
    least as new as it.
    """
    revised = MARKED.parent / "new_revised.tex"
    if not revised.exists():
        return False, "falta marked_changes/new_revised.tex"
    if revised.read_bytes() != TEX.read_bytes():
        return False, f"new_revised.tex != {TEX.parent.name}/{TEX.name}; falta el cp del MANIFEST"
    # Esto comparaba mtimes. git no preserva mtimes, asi que en un checkout limpio el orden es
    # arbitrario: el check no podia fallar en CI por la razon correcta ni pasar por ella. El sello
    # lo escribe set_diff_markup.py, que es obligatorio en la receta y corre justo despues de
    # latexdiff, sobre la misma fuente -- es de contenido y sobrevive a un clone.
    seal = MARKED.parent / "new_revised.sha256"
    if not seal.exists():
        return False, "falta new_revised.sha256; corre set_diff_markup.py tras latexdiff"
    want = hashlib.sha256(revised.read_bytes()).hexdigest()
    got = seal.read_text().strip()
    if got != want:
        return False, (
            f"el diff salio de otra fuente (sello {got[:12]}, actual {want[:12]}); "
            "falta correr latexdiff + set_diff_markup.py"
        )
    # Colour is the whole notation in this build -- deletions are not struck through -- so the key
    # set_diff_markup.py injects after \maketitle is load-bearing, and it is injected into a file
    # that two later scripts rewrite.
    if "set_diff_markup legend" not in MARKED.read_text():
        return False, "falta la leyenda de colores; corre set_diff_markup.py"
    return True, "al dia con clean_source, con leyenda de colores"


# NOT a check, and the reason is worth keeping: an intra-manuscript version of c_letter_numbers
# was written for NGC 6383 on 2026-08-17 and removed the same hour. Asking "is every Table N
# value present in the body" passes for the wrong reason -- mutating one value elsewhere left it
# green, because the old number still appears somewhere else, which is exactly the R11-class
# defect it claimed to catch. The symmetric form (any near-but-unequal pair between table and
# body) does catch that mutation and produced 12 false positives on the clean manuscript, all
# legitimate. Twelve alarms that are all noise is a check nobody will keep running. This class
# needs a reader, or a paper-specific local check with real context -- see gate_local.py.


@check("una sola forma de decir por que se adopta la ventana de 70 arcmin", group="consistencia")
def c_paraphrase():
    """A guarded rephrasing that says the same claim in different words is invisible to a
    presence check, and visible to a human only by luck. `variants` is a closed list of phrasings
    that were actually written into a manuscript and actually removed for overclaiming past what
    the analysis in the paper supports; `canonical` is the phrase that replaced them, required to
    occur at least once so the argument is still being made, just in the one licensed form."""
    variants = _CONFIG["paraphrase"]["variants"]
    canonical = _CONFIG["paraphrase"]["canonical"]
    tex = TEX.read_text().lower()
    found = [p for p in variants if p in tex]
    count = tex.count(canonical.lower())
    if found:
        return False, f"variantes que afirman de mas: {found}"
    return count > 0, (
        f"{count} usos de la formula canonica"
        if count
        else f"desaparecio la formula canonica {canonical!r}"
    )


# --------------------------------------------------------------------------------- CDS


@check("paquete CDS valido")
def c_cds():
    readme, dat = CDS_README.read_text(), CDS_DAT.read_text().rstrip("\n").split("\n")
    m = re.search(rf"{re.escape(CDS_DAT.name)}\s+(\d+)\s+(\d+)", readme)
    if not m:
        return False, "el ReadMe no declara Lrecl/Records"
    lrecl, records = int(m.group(1)), int(m.group(2))
    problems = []
    if records != len(dat):
        problems.append(f"Records {records} != {len(dat)} filas")
    if any(len(line) != lrecl for line in dat):
        problems.append(f"Lrecl {lrecl} != longitud real")
    rows = []
    for line in readme.split("\n"):
        mm = re.match(r"^\s*(\d+)(?:-(\d+))?\s+([IFAE]\d+(?:\.\d+)?)\s+(\S+)\s+(\S+)\s+(.*)$", line)
        if mm:
            rows.append(
                (int(mm.group(1)), int(mm.group(2) or mm.group(1)), mm.group(5), mm.group(6), line)
            )
    cols = {line.index(lab, 17) for _, _, lab, _, line in rows}
    if len(cols) > 1:
        problems.append(f"columna Label desalineada: {sorted(cols)}")
    for a, b, lab, expl, _ in rows:
        # Nulo = campo en blanco (el estandar del CDS: '?' en la explicacion) o '...' (lo que usaba
        # aa52082-24 hasta 2026-09-22, cuando el validador del upload de VizieR lo rechazo en F8.4
        # con "Bad decimal point").
        nulls = sum(1 for d in dat if d[a - 1 : b].strip() in ("", "..."))
        if bool(nulls) != expl.lstrip().startswith("?"):
            problems.append(f"{lab}: {nulls} nulos, marca '?' = {expl.lstrip()[:1]!r}")
    return (
        not problems,
        "Lrecl, registros, alineacion y nulos correctos" if not problems else "; ".join(problems),
    )


# ------------------------------------------------------------------------------- LaTeX


@check("chktex y lacheck", group="latex")
def c_linters():
    # Esto contaba avisos y aceptaba "hasta 2". Un umbral no dice CUALES: si el aviso de guion se
    # arreglara y apareciera otro distinto, la cuenta seguiria en 2 y el check verde -- pasa sin
    # ver nada. Y cuando fallo en CI (3 en vez de 2) no habia forma de saber cual era el tercero,
    # porque imprimia una cifra. Ahora acepta por NUMERO de aviso, no por cantidad: el 8 es el
    # largo de guion, una regla de estilo sobre prosa cuyos casos reales configura `accepted_dash`;
    # cualquier otro numero falla y se imprime.
    a = run(["chktex", "-q", "-f", "%n|%l|%c|%m\n", TEX.name], cwd=TEX.parent)
    # Medido: `chktex fichero_inexistente.tex` devuelve rc=0 con stdout vacio y el aviso en stderr.
    # Sin mirar stderr, `warns` quedaba vacio, `unexpected` vacio, y el check reportaba
    # "0 avisos, todos aceptados" -- verde. Un renombre del .tex, un cwd equivocado, un .chktexrc
    # corrupto o un build que rechace el `-f` convertian los dos linters en un pase incondicional.
    if "Unable to open" in a.stderr or "unable to open" in a.stderr.lower():
        return False, f"chktex no pudo leer {TEX.name}: {a.stderr.strip()[:90]}"
    # El `.replace("\\n", "\n")` que habia aqui era un no-op sobre la salida real -- Python pasa un
    # salto real en el `-f`, y chktex emite saltos reales -- y solo podia hacer dano: chktex
    # sustituye nombres de comando en varios mensajes, asi que uno que cite `\newline` o
    # `\noindent` se partia por la barra-n y producia un fragmento que no empieza por un numero
    # aceptado, o sea un fallo nombrando un aviso inexistente con el real truncado.
    warns = [w for w in a.stdout.split("\n") if w.strip()]
    # `.chktexrc` argumenta EXACTAMENTE lo contrario de aceptar el nº8 entero: "Relaxing NumDash
    # would silence them but would also stop catching a genuine 1-10 written for 1--10, which is
    # the more valuable check. Two explained warnings are cheaper than a weakened rule." Aceptar
    # todos los guiones del documento era la regla debilitada que ese comentario rechaza -- medido,
    # con `1-10 Myr` y `pp. 100-110` insertados, cuatro defectos tipograficos reales pasaban verdes.
    # Se acepta el TEXTO documentado por paper, que es lo que ambos avisos marcan y sobrevive a
    # que las lineas se muevan.
    accepted_dash = _CONFIG["linters"]["accepted_dash"]
    tex_lines = TEX.read_text().split("\n")

    def dash_ok(w: str) -> bool:
        """El aviso marca la OCURRENCIA aceptada, no una linea que la contenga en alguna parte.

        Aceptaba cualquier nº8 cuya LINEA contuviera el texto aceptado, que es la misma clase de
        agujero que este bloque dice haber cerrado -- de fichero entero a linea -- sin llegar al
        final: en `... in X, over 1-10 Myr ...` chktex emite dos avisos con la misma `%l` y los dos
        pasaban. Con `%c` se exige que la columna marcada caiga DENTRO de una ocurrencia del texto
        aceptado.
        """
        parts = w.split("|")
        if len(parts) < 3 or parts[0] != "8":
            return False
        try:
            src = tex_lines[int(parts[1]) - 1]
            col = int(parts[2])
        except (ValueError, IndexError):
            return False
        for accepted in accepted_dash:
            desde = 0
            while (i := src.find(accepted, desde)) >= 0:
                # 1-based, como los reporta chktex; el tramo va de i+1 a i+len inclusive.
                if i + 1 <= col <= i + len(accepted):
                    return True
                desde = i + 1
        return False

    # El nº12 ("interword spacing") es un falso positivo de chktex < 1.7.9 sobre abreviaturas
    # seguidas de un comando (p.ej. `($m=43$). \textsc{`): no distingue ese punto de uno de
    # oracion. 1.7.9 lo distingue y no lo emite. NO se desactiva globalmente: se acepta solo bajo
    # la version vieja, que es la que trae el runner de Ubuntu. La version se imprime, para que la
    # divergencia entre los dos entornos quede a la vista en ambos registros en vez de convertirse
    # en la clase de silencio que este gate existe para sacar.
    accept: list[str] = []
    ver = run(["chktex", "--version"]).stdout
    m = re.search(r"v(\d+)\.(\d+)\.(\d+)", ver)
    # Fallaba ABIERTO: sin `m`, `old_chktex` era True y la exencion del nº12 se concedia para
    # siempre. chktex 2.x, un build de distro, cualquier `--version` que no case, y la clase que el
    # nº12 vigila se aceptaba en silencio. Si no se sabe que version es, se asume la estricta: una
    # exencion se concede sabiendo, no por no saber.
    old_chktex = tuple(map(int, m.groups())) < (1, 7, 9) if m else False
    if old_chktex:
        accept.append("12|")
    unexpected = [w for w in warns if not dash_ok(w) and not any(w.startswith(k) for k in accept)]
    b = run(["lacheck", TEX.name], cwd=TEX.parent)
    if "not open" in b.stderr.lower() or "no such file" in b.stderr.lower():
        return False, f"lacheck no pudo leer {TEX.name}: {b.stderr.strip()[:90]}"
    lacheck = [
        ln for ln in b.stdout.split("\n") if ln.strip() and "Dots should be ellipsis" not in ln
    ]
    ok = not unexpected and not lacheck
    vtag = (m.group(0) if m else "version desconocida") + (", nº12 aceptado" if old_chktex else "")
    detail = f"chktex {vtag}: {len(warns)} avisos, todos aceptados; lacheck 0"
    if not ok:
        detail = ("chktex inesperados: " + "; ".join(unexpected[:3]) if unexpected else "") + (
            "  lacheck: " + "; ".join(lacheck[:3]) if lacheck else ""
        )
    return ok, detail


@check("typos", group="latex")
def c_typos():
    # En CI la cobertura puede existir por otra via: `crate-ci/typos@master` como paso propio del
    # workflow, sobre este directorio y con este `_typos.toml`, fallando el job por su cuenta.
    # Lo que esa accion no hace es dejar el binario en PATH, asi que invocarlo aca reventaba con
    # FileNotFoundError y contaba como check fallado. Se omite nombrando quien cubre el hueco --
    # no es una excepcion, es la misma revision corriendo un escalon mas arriba.
    if shutil.which("typos") is None:
        raise Skipped("binario ausente; si CI lo corre, es el paso crate-ci/typos del workflow")
    r = run(["typos", str(_BASE)])
    return r.returncode == 0, "limpio" if r.returncode == 0 else r.stdout.strip()[:160]


@check("floats movidos limpiados del diff marcado", group="latex")
def c_strip():
    """A moved float must not appear as a struck-through caption with no image.

    Without the strip, a moved figure reads to a referee as a deleted one.
    """
    if not MARKED.exists():
        return False, f"{MARKED.name} no existe; corre latexdiff primero"
    body = MARKED.read_text()
    body = body[body.index(r"\begin{document}") :]
    spans = len(re.findall(r"%DIFDELCMD < \\begin\{(?:figure|table)\*?\}", body))
    return spans == 0, f"{spans} spans de float comentados (deben ser 0 tras strip_moved_floats.py)"


@check("numeracion de linea apagada en los apendices", group="latex")
def c_linenumbers():
    """aa.cls's \\appendix ends in \\linenumbers and turns them back on."""
    tex = TEX.read_text()
    i = tex.find(r"\begin{appendix}")
    if i < 0:
        return False, "no hay bloque appendix"
    window = tex[i : i + 200]
    return r"\nolinenumbers" in window, (
        "reemitido tras \\begin{appendix}"
        if r"\nolinenumbers" in window
        else "FALTA: aa.cls las reactiva y se imprimen sobre el texto"
    )


# ------------------------------------------------------------------------- build (slow)


@check("ambos documentos compilan sin errores", group="compilacion", slow=True)
def c_build():
    """Build every configured document, because every one is uploaded and only one was ever
    audited when this check was first written for NGC 6383.

    Until 2026-08-17 that gate compiled clean_source alone. The marked document was left to
    whatever log happened to be on disk -- which is how it sat with four TeX errors
    ("Misplaced \\noalign", "Illegal unit of measure") from a stale run while the gate
    reported a clean build. c_overfull reads these logs, so building here also guarantees
    it is reading logs this run produced rather than a previous state of the manuscript.
    """
    out = []
    for tag, tex in DOCS:
        stem = tex.with_suffix("").name
        pdf, logp = build_paths(tex)
        # El clean sigue siendo necesario -- sin el, c_overfull y c_manifest_pages podrian leer el
        # log de una corrida anterior -- pero ahora solo borra el directorio de build.
        run(["latexmk", "-C", f"-outdir={BUILD_DIR}", stem], cwd=tex.parent)
        # `$bibtex_fudge=0`: bibtex corre desde el directorio del .tex, no desde BUILD_DIR. Con el
        # default, `\bibliography{methods,../paper}` de P02 se resolvia relativo a `_gate_build/` y
        # daba 2 citas indefinidas que un build directo no tiene (medido 2026-09-22; ni BIBINPUTS
        # ni -auxdir lo arreglan: kpathsea no busca rutas explicitamente relativas).
        run(
            [
                "latexmk",
                "-e",
                "$bibtex_fudge=0",
                "-pdf",
                "-bibtex",
                "-interaction=nonstopmode",
                f"-outdir={BUILD_DIR}",
                tex.name,
            ],
            cwd=tex.parent,
        )
        if not logp.exists():
            out.append(f"FALLA {tag}: latexmk no dejo log en {BUILD_DIR}/")
            continue
        log = logp.read_text(errors="replace")
        pages = pages_in(log, stem)
        bad = {
            "errores TeX": len(re.findall(r"^! ", log, re.M)),
            "LaTeX Error": log.count("LaTeX Error"),
            "refs indef": len(re.findall(r"Reference .* undefined", log)),
            "citas indef": len(re.findall(r"Citation .* undefined", log)),
        }
        if any(bad.values()) or pages is None:
            # Recorded, not returned: returning here would skip the remaining documents, leaving
            # their logs from a previous run for c_overfull and c_manifest_pages to read -- the
            # exact stale-log state this check was written to end.
            out.append(
                f"FALLA {tag}: {pages if pages is not None else '?'} pp, "
                + ", ".join(f"{k} {v}" for k, v in bad.items())
            )
        else:
            out.append(f"{tag} {pages} pp")
    failed = [o for o in out if o.startswith("FALLA")]
    return not failed, (
        "; ".join(out) + f", 0 errores y 0 indefinidas en los {len(DOCS)} documentos"
        if not failed
        else "; ".join(out)
    )


@check("los PDF que se suben son los recien construidos", group="compilacion", slow=True)
def c_deliverables():
    """The uploaded files are *copies*, and a copy is exactly as stale as you let it be.

    The clean_source build is the build; the files a journal or preprint server receives are
    what gets copied there. Rebuilding without re-copying uploads the previous revision -- on
    2026-08-17 (NGC 6383) the copies were 24 minutes older than the build and still carried the
    marked document's undefined citation. Bytes cannot be compared (every build stamps a new
    timestamp and a new /ID), so the extracted text is, which is what a referee reads.
    """

    def text(p: Path) -> str | None:
        r = run(["pdftotext", str(p), "-"])
        if r.returncode != 0:
            return None
        # aa.cls stamps \today into the running head, so a rebuild on a later day differs from the
        # copy in the date and nothing else. That is not a stale deliverable, and treating it as one
        # makes the check cry wolf every time the clock rolls over.
        return re.sub(r"\b[A-Z][a-z]+ \d{1,2}, \d{4}\b", "<fecha>", r.stdout)

    _prod_cache: dict[Path, str] = {}

    def producer(p: Path) -> str:
        # Memoizado: se llamaba varias veces sobre el mismo fichero construido, y un desajuste
        # costaba varios `pdfinfo` para leer los mismos PDF repetidas veces.
        if p in _prod_cache:
            return _prod_cache[p]
        r = run(["pdfinfo", str(p)])
        m = re.search(r"^Producer:\s*(.+)$", r.stdout, re.M)
        _prod_cache[p] = m.group(1).strip() if m else "?"
        return _prod_cache[p]

    tag_to_tex = dict(DOCS)
    pairs = []
    for tag, rel in _CONFIG["deliverables"]["pairs"]:
        if tag not in tag_to_tex:
            return (
                False,
                f"deliverables.pairs cita el tag {tag!r}, que no esta en DOCS ({list(tag_to_tex)})",
            )
        built = build_paths(tag_to_tex[tag])[0]
        pairs.append((built, _resolve(_BASE, rel)))

    missing_build = [built for built, _ in pairs if not built.exists()]
    if missing_build:
        raise Skipped(
            f"no hay build en {BUILD_DIR}/ contra el que comparar; c_build es quien lo crea"
        )
    # Este check compara un PDF recien construido con uno versionado, y eso solo significa algo si
    # los construyo el mismo motor. Otro TeX Live guiona y corta lineas distinto, asi que el texto
    # extraido difiere por el entorno y no por estar desactualizado. La condicion es medida, no una
    # bandera -- el propio PDF dice quien lo hizo.
    for built, sent in pairs:
        if sent.exists() and producer(built) != producer(sent):
            # En la maquina que sube esta omision ya no es benigna: desde que las omisiones son
            # fatales sin `--allow-skips`, un desajuste de motor aqui hace que el gate se niegue a
            # bendecir. Que es lo correcto -- tras actualizar TeX Live, los PDF versionados llevan
            # el motor viejo y todo build fresco el nuevo, asi que el check se apagaba exactamente
            # cuando su respuesta correcta era "reconstruye y vuelve a copiar".
            raise Skipped(
                f"construido por {producer(built)}, enviado por {producer(sent)}: "
                "otro motor de TeX. En la maquina que sube esto significa reconstruir "
                "los entregables y volver a copiarlos"
            )
    stale = []
    for built, sent in pairs:
        if not sent.exists():
            stale.append(f"{sent.name} no existe")
            continue
        a, b = text(built), text(sent)
        # None != None is False, so a pair of unreadable PDFs used to report "identical". A check
        # whose whole purpose is to break silence about stale uploads cannot have a silent pass.
        if a is None or b is None:
            stale.append(f"{sent.name}: pdftotext no pudo leer uno de los dos")
        elif a != b:
            stale.append(sent.name)
    return not stale, "identicos al build" if not stale else f"desactualizados: {stale}"


@check("el MANIFEST declara el numero de paginas real", group="compilacion", slow=True)
def c_manifest_pages():
    """MANIFEST.md is the upload instructions, and it states page counts as fact.

    For NGC 6383 it said "26 pp clean / 29 pp marked" in the header and "30 pp" in the file
    table, three lines apart, both hand-maintained. The reason to check rather than to stop
    writing them down is that the count is how a human notices a truncated or double-built PDF
    before sending it. Every "NN pp" in the file is compared against every configured build.
    """
    manifest = _BASE / "MANIFEST.md"
    if not manifest.exists():
        return False, "no existe MANIFEST.md"
    real = {}
    for tag, tex in DOCS:
        stem = tex.with_suffix("").name
        _, log = build_paths(tex)
        if not log.exists():
            return False, f"no hay build {tag} en {BUILD_DIR}/: corre c_build antes"
        n = pages_in(log.read_text(errors="replace"), stem)
        if n is None:
            return False, f"no se pudo leer las paginas del build {tag}"
        real[tag] = n
    claimed = {int(n) for n in re.findall(r"(\d+)\s*pp", manifest.read_text())}
    # Both directions. Checking only "no claimed count is wrong" passes on a MANIFEST that swaps
    # the two labels, drops one count, or states none at all -- and a missing or swapped count is
    # exactly the hand-maintenance error the check exists to stop.
    wrong = sorted(claimed - set(real.values()))
    absent = sorted(v for v in real.values() if v not in claimed)
    if wrong or absent:
        return False, (f"el MANIFEST afirma {wrong} que los builds no dan; " if wrong else "") + (
            f"y no declara {absent}" if absent else ""
        )
    return True, ", ".join(f"{tag} {n} pp" for tag, n in real.items()) + ", todos declarados"


@check("sin texto fuera de columna en ninguno de los dos PDF", group="compilacion", slow=True)
def c_overfull():
    """An Overfull \\hbox is text sticking out of the column, and in two-column A&A it lands
    on top of the neighbouring column.

    This check did not exist for NGC 6383 until a human looked at the marked PDF and said it
    looked awful. The clean build had been checked for Overfull and had none; the marked build
    had never been checked at all and had twelve, because latexdiff's default UNDERLINE markup
    strikes deleted text with ulem's \\sout, which cannot break across lines. A long struck
    citation list therefore ran off the column and overprinted the text beside it.

    The lesson is not about latexdiff. Eleven checks were written for that paper and none of
    them looked at the artefact the referee actually opens.
    """
    allowed = _CONFIG["overfull"]["allowed"]
    bad = {}
    for tag, tex in DOCS:
        if tag not in allowed:
            bad[tag] = f"overfull.allowed no declara el tag {tag!r}"
            continue
        log = build_paths(tex)[1]
        if not log.exists():
            bad[tag] = "sin log"
            continue
        n = log.read_text().count("Overfull \\hbox")
        if n != allowed[tag]:
            bad[tag] = f"{n} (esperado {allowed[tag]})"
    return not bad, (
        ", ".join(f"{tag} {allowed[tag]}" for tag, _ in DOCS)
        if not bad
        else f"cajas desbordadas: {bad}"
    )


@check("el zip enviado compila solo", group="compilacion", slow=True)
def c_zip():
    import tempfile
    import zipfile

    zp = _resolve(_BASE, _CONFIG["zip"]["path"])
    if not zp.exists():
        # Artefacto local y regenerable, tipicamente gitignorado: es la ranura obligatoria de la
        # revista, asi que el check pertenece a la maquina que sube, no al runner: alli no hay zip
        # que revisar y fallar solo dice que no lo hay.
        raise Skipped("el zip es artefacto local y no se versiona; el check corre donde se sube")
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        tex_files = [n for n in names if n.endswith(".tex")]
        if len(tex_files) != 1:
            return False, f"se exige un solo .tex; el zip trae {len(tex_files)}"
        # "It compiles" is not "it is the current manuscript". A zip built before the last edit
        # compiles perfectly and to the same page count -- for NGC 6383 on 2026-08-17 this check
        # passed on a zip 24 minutes stale, missing the last edit, while every other check was
        # green. This is the *mandatory* slot: the journal builds the referee's PDF from it, so a
        # stale zip is the one defect here that reaches print.
        # Every entry, not a named handful. The first version listed five files and skipped the
        # figures, so regenerating a figure without rebuilding the zip left the check green -- and
        # the journal builds the referee's PDF from this zip, so the referee would read the new
        # text against the old figure. Figure regeneration is routine in this tree.
        stale = []
        for n in names:
            if n.endswith("/"):
                continue
            src = TEX.parent / n
            if not src.exists():
                stale.append(f"{n} no existe en {TEX.parent.name}")
            elif z.read(n) != src.read_bytes():
                stale.append(n)
        if stale:
            return False, f"el zip no coincide con {TEX.parent.name} ({len(stale)}): {stale[:5]}"
        with tempfile.TemporaryDirectory() as td:
            z.extractall(td)
            run(["latexmk", "-pdf", "-bibtex", "-interaction=nonstopmode", TEX.name], cwd=Path(td))
            log = Path(td, f"{TEX.stem}.log")
            if not log.exists():
                return False, "no compilo"
            t = log.read_text()
            n = len(re.findall(r"(Reference|Citation) .* undefined", t))
            p = pages_in(t, TEX.stem)
            return (
                t.count("LaTeX Error") == 0 and n == 0,
                f"{p if p is not None else '?'} pp aislado, {n} indefinidas, 1 .tex",
            )


# ------------------------------------------------------------------------------------- main

GROUPS = ["consistencia", "latex", "compilacion"]
GROUP_HEADERS = {
    "consistencia": "=== consistencia entre el manuscrito y lo que lo describe ===",
    "latex": "=== fuente LaTeX ===",
    "compilacion": "=== compilacion ===",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="omite lo que necesita compilar LaTeX")
    ap.add_argument(
        "--allow-skips",
        action="store_true",
        help="una omision no es fatal. Solo para CI: en la maquina que sube, un check "
        "omitido es un check que no se hizo donde importa",
    )
    args = ap.parse_args()

    printed_header = False
    for group in GROUPS:
        if group == "compilacion" and args.quick:
            continue
        names_here = [n for n in declared if BY_NAME[n].group == group]
        if not names_here:
            continue
        print(("" if not printed_header else "\n") + GROUP_HEADERS[group])
        printed_header = True
        for n in names_here:
            BY_NAME[n]()

    # La guarda `forgotten`: un check declarado (via @check) tiene que terminar en exactamente uno
    # de estos tres estados -- corrido (entro a `results`, sea ok, FALLA u omitido), marcado n/a
    # por la config, o excusado por --quick al ser lento. Cualquier otro caso es un check que
    # quedo fuera del despacho por group() -- el bug que la version anterior, con una lista de
    # llamadas a mano en main(), tambien podia cometer con un nombre mal escrito.
    ran = {n for n, _, _ in results}
    na = {n for n, _ in not_applicable_ran}
    forgotten = [
        n
        for n in declared
        if n not in ran and n not in na and not (args.quick and n in slow_checks)
    ]
    if forgotten:
        print(f"\nFALLA  checks declarados que no se despacharon: {forgotten}")
        return 1

    failed = [n for n, ok, _ in results if not ok]
    print(
        f"\n{len(results) - len(failed) - len(skipped)}/{len(results)} pasan"
        + (f", {len(skipped)} omitidos" if skipped else "")
        + (f", {len(na)} no aplica" if na else "")
    )
    for n, why in skipped:
        print(f"  omitido  {n}: {why}")
    if failed:
        print("FALLAN: " + ", ".join(failed))
        return 1
    # La bendicion es lo unico sobre lo que alguien actua, asi que no puede sobrevivir a una
    # omision. `c_zip` puede pasar de fallar a omitirse cuando el zip no esta: en el runner eso es
    # correcto, pero en la maquina que sube significa que la ranura obligatoria de la revista no
    # existe -- y el gate igual imprimiria "puede subirse". Una omision aca abajo es informacion
    # que se pierde justo donde se toma la decision.
    # La bendicion se hizo consciente de las omisiones y no del modo rapido: `--quick` imprimia
    # "puede subirse" sin haber corrido la ranura obligatoria ni ninguna compilacion. Y
    # `REVISADO PARCIAL` salia 0, o sea que cualquier consumidor automatico -- un
    # `gate.py && subir`, un paso de CI -- leia una omision como exito, y solo un humano leyendo
    # las dos ultimas lineas se enteraba.
    #
    # `--allow-skips` es la salida: las omisiones son fatales por defecto y solo el runner las
    # perdona, porque es el unico sitio donde son legitimas. En la maquina que sube no hay ninguna,
    # asi que la bandera no cuesta nada aca y hace que la MAQUINA, no el lector, imponga que no se
    # omitio nada donde importa.
    # Las dos razones de una revision parcial se reportan JUNTAS, y la fatalidad la decide la
    # omision. La primera version devolvia 0 en cuanto veia `--quick`, antes de mirar `skipped`, asi
    # que el modo rapido se tragaba las omisiones y volvia a colar exactamente el agujero que este
    # bloque cierra. Lo caza `test_gate_behaviour.py`.
    parcial = []
    if args.quick:
        parcial.append("modo --quick: no corrieron " + ", ".join(sorted(slow_checks)))
    if skipped:
        parcial.append(
            f"{len(skipped)} omitidos: "
            + ", ".join(n for n, _ in skipped)
            + ". En la maquina que sube no deberia omitirse ninguno"
        )
    if parcial:
        print("REVISADO PARCIAL - " + "; ".join(parcial) + ".")
        return 1 if (skipped and not args.allow_skips) else 0
    print("OK - el paquete puede subirse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
