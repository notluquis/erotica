#!/usr/bin/env python3
"""One command that has to pass before aa52082-24 is uploaded or committed.

Why this file exists
--------------------
Every defect this package accumulated was found by a human looking in a place nobody had
looked before, and each was of the same shape: an artefact that *describes* the manuscript
drifting away from the manuscript, silently, because nothing compared them.

    - the two letters promised the referee things the paper no longer said
    - nine sentences called the isochrone ensemble a posterior after Sect. 4.4 denied it
    - the marked PDF showed ten figures as deleted when they had only moved
    - the CDS ReadMe on NESTOR was the superseded one, and a third copy had diverged
    - the knowledge-graph note still carried the round-1 R_t of 40.4 arcmin
    - line numbers printed on top of the appendix text on every page

None of those is exotic. Each is cheap to detect and impossible to notice by reading.

The other half of the reason is structural. This paper lives inside the EROTICA package
repository, under `data/`, and the package's `.pre-commit-config.yaml` excludes `^(data/...)`
because `data/` is data. So the manuscript receives no hooks at all -- not by a decision
about the manuscript but as collateral of where it sits. `ARCHITECTURE.md` in the hub calls
this "today's broken state" and defers the fix until after acceptance, because 42 scripts
hardcode paths into this directory. Until then, this file is the substitute owner.

Usage
-----
    python3 gate.py            # everything, ~4 min (rebuilds both PDFs)
    python3 gate.py --quick    # everything that does not need a LaTeX run, ~5 s

Exit code is 0 only if every check passes. Each check prints what it compared, not just a
verdict, so a failure says what to look at.
"""

from __future__ import annotations

import argparse
import math
import hashlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEX = HERE / "clean_source" / "aanda.tex"
MARKED = HERE / "marked_changes" / "aanda_marked.tex"
LETTERS = [HERE / "letters" / "cover_letter_round2.txt",
           HERE / "letters" / "response_to_referee_round2.txt"]
CDS_DAT = HERE.parent / "cds_final" / "ngc6383_members.dat"
CDS_README = HERE.parent / "cds_final" / "ReadMe"
KB_ROOT = Path.home() / "phd" / "kb"
KB_NOTES = [KB_ROOT / "papers" / "2024arXiv240509145P.md",
            KB_ROOT / "objects" / "ngc-6383.md"]

results: list[tuple[str, bool, str]] = []
# @check registers a name only when the function is *called*, so a check that main() forgets to
# call is indistinguishable from one that passes -- which is how c_overclaim was written, printed
# nothing, and left the gate reporting 10/10. Declared names are recorded at import; main() checks
# every one of them ran.
declared: list[str] = []
slow_checks: set[str] = set()
skipped: list[tuple[str, str]] = []


class Skipped(Exception):
    """Un check cuyo insumo vive fuera de este repo y no esta presente aca.

    Existe porque el gate se declaro "el duenio interino del manuscrito" y se cableo a CI, donde
    dos checks no podian pasar jamas: uno lee `~/phd/kb`, que es OTRO repo, y el runner no lo
    tiene. El resultado fue el peor de los dos mundos -- el gate local leia 26/26 mientras el
    workflow fallaba en las cinco corridas seguidas desde el 2026-08-17, y nadie miraba.

    La salida no es saltar en silencio: eso es exactamente el modo de falla que este repo
    persigue en todo lo demas. Un salto se imprime, se cuenta aparte en el resumen, y nombra su
    motivo. Y la condicion es angosta a proposito: se salta cuando falta el REPO entero, no
    cuando falta el fichero -- si el repo esta y la nota no, eso es un borrado y falla."""


def check(name: str, slow: bool = False):
    """Register a check. `slow` marks the ones that need a LaTeX run, at the single place that
    knows: the declaration. A second hand-kept list in main() drifts the moment a name is edited,
    which is the failure the `declared` list itself was added to remove."""
    def deco(fn):
        declared.append(name)
        if slow:
            slow_checks.add(name)

        def wrapper(*a, **kw):
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
        return wrapper
    return deco


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


# --------------------------------------------------------------------------- consistency

@check("posterior claims")
def c_posterior():
    r = run([sys.executable, str(HERE / "check_posterior_claims.py")])
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
            before = text[max(0, m.start() - 40):m.start()].lower()
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
    if not KB_ROOT.is_dir():
        raise Skipped(f"{KB_ROOT} no esta (es otro repo); el check corre en local, no en CI")
    missing = [p for p in KB_NOTES if not p.exists()]
    if missing:
        return False, f"notas no encontradas: {[p.name for p in missing]}"
    note = "\n".join(p.read_text() for p in KB_NOTES)
    tex = TEX.read_text()
    anchors = {
        "R_t": (r"R_t = 54\^\{\+7\}_\{-11\}", r"R_t\s*40\.4"),
        "R_c": (r"1\.96\^\{\+0\.19\}_\{-0\.16\}", r"R_c\s*1\.95"),
        "age/t_rh": (r"0\.14", r"t_rh\s*~\s*0\.19"),
    }
    stale = []
    for label, (in_tex, in_note) in anchors.items():
        if re.search(in_tex, tex) and re.search(in_note, note):
            stale.append(label)
    return not stale, ("sin cifras superadas" if not stale
                       else f"la nota cita valores de ronda 1: {stale}")


# Phrases that are about how the paper argues rather than about the cluster, or that instruct
# the reader instead of stating something. Every one was written into this manuscript during the
# 2026-08 review and removed in Sect. 69 of CHANGES.md; the list is what that pass found, not a
# general style opinion. It applies to the manuscript ONLY -- a response letter is legitimately
# about how one argues, and "we now state", "our argument" belong there.
REGISTER = [
    "for our argument",
    "should be read as such",
    "should be read accordingly",
    "rather than afterwards",
    "deliberately kept apart",
    "we make it from",
    "requires an argument",
    "it is worth noting",
    "it should be emphasized",
    "it should be emphasised",
    "needless to say",
    "as we have seen",
    "in our opinion",
    "we note two consequences",
]


@check("registro del manuscrito")
def c_register():
    """Catch commentary on our own reasoning, and instructions standing in for statements.

    An editorial pass in Sect. 69 removed seven of these by hand and left nothing behind to catch
    the eighth, which is the failure this file exists to stop. The list is closed and specific
    rather than a general prose-quality opinion: each entry was actually written into this
    manuscript and actually removed.
    """
    text = TEX.read_text().lower()
    found = [phrase for phrase in REGISTER if phrase in text]
    return not found, "sin muletillas de registro" if not found else f"{len(found)} -> {found}"


# The letter may not assert more strongly than the manuscript. Not a style rule: a referee reads
# both, and a letter that says the wider windows "demonstrably" admit contamination sends them to a
# Sect. 3.3 that says the behavior "is consistent with" it and would require further diagnostics to
# settle. The manuscript hedges there on purpose -- the outer population overlaps catalogued
# comoving clusters, which no contamination metric distinguishes. R11 carried both words below
# until 2026-08-17. Each is a claim of proof; if one is ever true of this analysis it belongs in
# the manuscript first, which is exactly what the second half of the check asks.
OVERCLAIM = ["demonstrably", "conclusively", "definitively", "unambiguously", "artificially",
             "indisputably", "proves that", "rules out"]


@check("la carta no afirma mas fuerte que el manuscrito")
def c_overclaim():
    # No manuscript-wide exemption. The first version allowed a word in the letters as soon as it
    # occurred anywhere in the paper, in any sense: "unambiguously" is in Sect. 3.2 about branch
    # selection, which would have exempted it from an R11 contamination claim -- the exact
    # assertion this list exists to stop. Presence in the manuscript is not assertion of the same
    # claim. None of these words is needed in a letter; if one ever is, it goes on the list's
    # exception with the sentence that earns it.
    found = []
    for path in LETTERS:
        text = path.read_text().lower()
        found += [f"{path.name}:{w}" for w in OVERCLAIM if w in text]
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
            bad.append(f"{path.name}: ...{' '.join(text[max(0, m.start()-30):m.start()+14].split())}")
        # Third signature, from re-wrapping rather than from symbol loss: a line-end hyphen whose
        # two halves were merged onto one line without dropping it ("a tie- breaker"). Suspended
        # hyphens are legitimate and are the only exception ("window- or model-dependent").
        for m in re.finditer(r"[a-z]- [a-z]", text):
            frag = text[max(0, m.start() - 12):m.end() + 10]
            if re.search(r"- (or|and|to|nor)\b", frag):
                continue
            bad.append(f"{path.name}: guion partido ...{' '.join(frag.split())}")
    return not bad, ("sin subindices sin simbolo" if not bad else f"{len(bad)} -> {bad[:4]}")



# US spelling, because the cover letter promises the editor exactly that: "The spelling is
# consistently US throughout (the single British form, 'catalogue', occurs only inside the
# CDS-mandated VizieR acknowledgement)." That claim is checkable, and it was true of the manuscript
# and false of the response letter, which said "Galactic-centre". The list is the ordinary
# British/US pairs an astronomy paper can hit; "towards", "whilst" and "amongst" are included
# because A&A house style prefers the shorter forms.
BRITISH = [
    "analyse", "analysed", "analysing", "characterise", "characterised", "characterisation",
    "normalise", "normalised", "normalisation", "minimise", "maximise", "optimise", "optimised",
    "organise", "organised", "organisation", "recognise", "summarise", "summarised",
    "emphasise", "emphasised", "parameterise", "parameterised", "marginalise", "marginalised",
    "utilise", "utilised", "generalise",
    "colour", "colours", "coloured", "behaviour", "favour", "neighbour", "neighbours",
    "neighbouring", "labour",
    "centre", "centres", "centred", "metre", "fibre",
    "modelling", "modelled", "labelling", "labelled", "travelled", "cancelled", "signalled",
    "fuelled", "totalled",
    "catalogue", "catalogues", "catalogued", "dialogue", "analogue", "grey", "programme",
    "defence", "licence", "ageing", "practise", "artefact", "artefacts", "sulphur", "aluminium",
    "whilst", "amongst", "towards",
]


@check("ortografia US, como la carta le promete al editor")
def c_spelling():
    bad = []
    for path in [TEX] + LETTERS:
        text = path.read_text()
        for word in BRITISH:
            for m in re.finditer(rf"\b{word}\b", text, re.I):
                frag = " ".join(text[max(0, m.start() - 70):m.end() + 40].split())
                # The two licensed exceptions: CDS mandates the wording of the VizieR
                # acknowledgement, and the cover letter quotes the word to declare it.
                if "VizieR" in frag or '"catalogue"' in frag:
                    continue
                bad.append(f"{path.name}: {word} ...{frag[:70]}")
    return not bad, ("sin formas britanicas fuera del agradecimiento a VizieR" if not bad
                     else f"{len(bad)} -> {bad[:3]}")


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
    body = tex[:tex.find(r"\begin{appendix}")] if r"\begin{appendix}" in tex else tex
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
            line = text[text.rfind("\n", 0, m.start()) + 1:text.find("\n", m.end())]
            # A line using the arrow form is presenting both numbers on purpose: the reply
            # headers, the note that explains them, and R7's compound "2.1.5 / 2.2 -> 5 / 6.1".
            # The mapping table lists every round-1 number by design.
            if "->" in line or "|" in line:
                continue
            if m.group(1) not in have:
                bad.append(f"{path.name}: Sect. {m.group(1)} no existe en el manuscrito")
    return not bad, (f"todas las secciones citadas existen ({len(have)} en el manuscrito)"
                     if not bad else f"{len(bad)} -> {sorted(set(bad))[:4]}")


@check("copias de las cartas y del ReadMe sincronizadas")
def c_copies():
    pairs = [
        (HERE / "letters" / "cover_letter_round2.txt",
         HERE.parent / "referee_round2" / "cover_letter_round2.txt"),
        (HERE / "letters" / "response_to_referee_round2.txt",
         HERE.parent / "referee_round2" / "response_letter.txt"),
        (CDS_README, HERE / "cds" / "ReadMe"),
        # marked_changes/ carries its own copy of the bibliography and the vendored A&A class,
        # so latexdiff's output builds standalone. Adding a reference to the manuscript without
        # copying cites.bib across leaves the marked PDF with an undefined citation printing as
        # "(?)" -- which is what 1987degc.book.....S did on p. 13 on 2026-08-17.
        (TEX.parent / "cites.bib", MARKED.parent / "cites.bib"),
        (TEX.parent / "aa.cls", MARKED.parent / "aa.cls"),
        (TEX.parent / "aa.bst", MARKED.parent / "aa.bst"),
    ]
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

    ⚠ `_legacy/cds_superseded/cds.zip` is load-bearing and must not be deleted while this paper is
    under review: it is the only local record of what NESTOR actually holds, and without it there
    is no way to tell whether the letter's account of the dataset is honest. MANIFEST.md calls it
    superseded, which is true of its contents and false of its role. A missing baseline fails
    rather than skips, on purpose -- a skip here would restore exactly the silence this check
    exists to break.
    """
    legacy = HERE.parent / "_legacy" / "cds_superseded" / "cds.zip"
    if not legacy.exists():
        return False, (f"falta {legacy.relative_to(HERE.parent)}, la referencia de lo que hay en "
                       "NESTOR; restaurala desde git en vez de saltarse el chequeo")
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
    return disclosed, ("el ReadMe cambio y la carta lo declara" if disclosed
                       else "el ReadMe cambio y la carta de presentacion no lo declara")


@check("el diff marcado deriva del manuscrito actual")
def c_marked_fresh():
    """The marked PDF is built from a *copy* of the manuscript, so editing one leaves the other.

    latexdiff runs on marked_changes/new_revised.tex, which MANIFEST.md tells you to `cp` from
    clean_source/aanda.tex first. Skip the copy and the marked PDF silently shows the previous
    revision -- the referee then reads a diff that omits the change they asked for. Two edits on
    2026-08-17 (the Appendix D wording and the Sect. 8 lead-in) left it stale with every other
    check green, which is what this catches: byte equality of the copy, and a marked source at
    least as new as it.
    """
    revised = MARKED.parent / "new_revised.tex"
    if not revised.exists():
        return False, "falta marked_changes/new_revised.tex"
    if revised.read_bytes() != TEX.read_bytes():
        return False, "new_revised.tex != clean_source/aanda.tex; falta el cp del MANIFEST"
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
        return False, (f"el diff salio de otra fuente (sello {got[:12]}, actual {want[:12]}); "
                       "falta correr latexdiff + set_diff_markup.py")
    # Colour is the whole notation in this build -- deletions are not struck through -- so the key
    # set_diff_markup.py injects after \maketitle is load-bearing, and it is injected into a file
    # that two later scripts rewrite.
    if "set_diff_markup legend" not in MARKED.read_text():
        return False, "falta la leyenda de colores; corre set_diff_markup.py"
    return True, "al dia con clean_source, con leyenda de colores"


@check("Table 1 es internamente consistente")
def c_table1():
    """Table 1 states four quantities bound by an equation, so three of them determine the fourth.

    t_seg = (<m>/m) t_rh (Eq. 5), and the text says its uncertainty comes from "propagating the
    uncertainties in <m>, m, and t_rh". All four values and all four errors sit in Table 1, so the
    relation is checkable without leaving the manuscript -- and it failed. Correcting t_rh from
    30.5 to 24.7 Myr this round moved the central t_seg correctly (2.94 -> 2.38) but left the
    error at 1.24, where the stated propagation gives 0.95; the round-1 pair (2.94 +/- 1.17)
    reproduces the same propagation exactly, which is what shows the recomputation was partial.
    A sentence written this round then leaned on the too-wide interval, claiming the 1 sigma range
    "already reaches the adopted age".

    The lesson generalises past this row: a correction that propagates into a derived quantity has
    to propagate into its uncertainty too, and nothing else here was watching. The tolerance is
    2% -- these are two-decimal printed values, not a fit.
    """
    tex = TEX.read_text()

    def row(label):
        m = re.search(re.escape(label) + r"[^&]*&\s*\$([-\d.]+)\s*\\pm\s*([\d.]+)\$", tex)
        return (float(m.group(1)), float(m.group(2))) if m else None

    need = {"mean": r"Mean stellar mass", "mmax": r"Most massive star",
            "trh": r"Half-mass relaxation time", "tseg": r"Minimum segregation time"}
    got = {k: row(v) for k, v in need.items()}
    missing = [k for k, v in got.items() if v is None]
    if missing:
        return False, f"no pude leer de Table 1: {missing}"
    (mm, e_mm), (mx, e_mx), (trh, e_trh), (tseg, e_tseg) = (got[k] for k in
                                                            ("mean", "mmax", "trh", "tseg"))
    val = mm / mx * trh
    err = val * math.sqrt((e_mm / mm) ** 2 + (e_mx / mx) ** 2 + (e_trh / trh) ** 2)
    bad = []
    if abs(val - tseg) > 0.02 * tseg:
        bad.append(f"t_seg: tabla {tseg}, (<m>/m)*t_rh = {val:.2f}")
    if abs(err - e_tseg) > 0.02 * max(err, e_tseg):
        bad.append(f"sigma(t_seg): tabla {e_tseg}, propagacion = {err:.2f}")
    # Sect. 7 spells the same +/- out as an interval and argues from it. The interval was 1.14-3.62
    # while the error was 1.24: correct then, and silently wrong the moment the error was fixed by
    # hand. Endpoints are what the referee reads, so they are checked against the table's own row
    # rather than trusted to have been updated alongside it.
    m = re.search(r"range of \$t_\{\\mathrm\{seg\}\}\$ alone, \$([\d.]+)\$--\$([\d.]+)", tex)
    if m is None:
        bad.append("no encuentro el intervalo 1 sigma de t_seg en Sect. 7")
    else:
        lo, hi = float(m.group(1)), float(m.group(2))
        if abs(lo - (tseg - e_tseg)) > 0.02 or abs(hi - (tseg + e_tseg)) > 0.02:
            bad.append(f"intervalo en Sect. 7 {lo}--{hi} != tabla {tseg-e_tseg:.2f}--{tseg+e_tseg:.2f}")
    # The same sentence states the separation in sigma, which is the third thing derived from this
    # row. Guarding the interval and not the sigma would strand it on the next refit, which is the
    # partial propagation this whole check exists for.
    s = re.search(r"places the adopted age only \$([\d.]+)\\sigma\$ above", tex)
    age = re.search(r"mode age \$t_\{\\mathrm\{age\}\} = ([\d.]+)\\,\\mathrm\{Myr\}\$", tex)
    if s is None or age is None:
        bad.append("no encuentro la separacion en sigma o la edad modal en Sect. 7")
    else:
        want = (float(age.group(1)) - tseg) / e_tseg
        if abs(float(s.group(1)) - want) > 0.05:
            bad.append(f"separacion: el texto dice {s.group(1)} sigma, la tabla da {want:.2f}")
    return not bad, (f"t_seg = ({mm}/{mx})*{trh} = {val:.2f} +/- {err:.2f} y el intervalo "
                     f"{tseg-e_tseg:.2f}--{tseg+e_tseg:.2f} de Sect. 7 concuerdan con la tabla"
                     if not bad else "; ".join(bad))


# NOT a check, and the reason is worth keeping: an intra-manuscript version of c_letter_numbers
# was written here on 2026-08-17 and removed the same hour. Asking "is every Table 1 value present
# in the body" passes for the wrong reason -- mutating the body's R_c from 1.96 to 1.95 left it
# green, because 1.96 still appears elsewhere, which is exactly the R11-class defect it claimed to
# catch. The symmetric form (any near-but-unequal pair between table and body) does catch that
# mutation and produces 12 false positives on the clean manuscript, all legitimate: Table D.2's
# other windows quote genuinely different means (0.906, 0.911, 1.711, 2.544), and the cluster
# centre 263.683 sits within 0.5% of six source coordinates in the Table 2 excerpt. Twelve alarms
# that are all noise is a check nobody will keep running. The structural relations that CAN be
# audited without context are covered by c_table1 and c_catalog_numbers; this class needs a reader.

# One fact, one phrasing. The 70 arcmin window's virtue is stated in six places -- abstract,
# Sect. 5, Appendix D, Sect. 8.4, the conclusions and Fig. D.1's caption -- and each variant below
# was actually written into one of them and removed, because it claims more than Appendix D allows.
# "Fully encloses the fitted profile" says the data determine R_t in the same sentence that reports
# the posterior truncated by the prior; "genuine background annulus" says b is real background
# where Appendix D deliberately refuses to choose among residual contamination, a corona, and
# projected populations. A closed list of variants actually removed, not a style opinion.
PARAPHRASE = [
    "fully encloses the fitted profile",
    "fully encloses the profile",
    "genuine background annulus",
    "genuine background level",
    "true background annulus",
    "real background beyond the cluster",
    "constrain the field background",
    "the field background beyond",
]


@check("una sola forma de decir por que se adopta la ventana de 70 arcmin")
def c_paraphrase():
    tex = TEX.read_text().lower()
    found = [p for p in PARAPHRASE if p in tex]
    canonical = tex.count("fitted background annulus")
    if found:
        return False, f"variantes que afirman de mas: {found}"
    return canonical > 0, (f"{canonical} usos de la formula canonica" if canonical
                           else "desaparecio la formula canonica 'fitted background annulus'")


@check("el texto y Table A.1 atribuyen el mismo valor a la misma referencia")
def c_literature_agreement():
    """Table A.1 compiles the literature; the running text quotes it. They must not disagree.

    The introduction gave Angelo et al. (2018) a distance of 0.840 kpc where Table A.1's row for
    the same paper says 0.83 +/- 0.16 -- and the Table A.1 caption argument, and the response
    letter, both use 0.83. One quantity, one source, two roundings, three places. This is the R7
    defect with a citation attached, which is what makes it checkable where the general
    intra-manuscript version was not (Sect. 74): the citation key says *which* two numbers are
    supposed to be the same, so there is no guessing and no noise.
    """
    tex = TEX.read_text()
    i = tex.find(r"\label{tab:literature}")
    a, b = tex.rfind(r"\begin{table", 0, i), tex.find(r"\end{table", i)
    # b is guarded too: find() returns -1 for a missing terminator, and tex[b:] is then the last
    # character of the file, which made the text-side scan run over a truncated document and pass
    # vacuously instead of reporting a malformed table.
    if i < 0 or a < 0 or b < 0:
        return False, "no encuentro Table A.1 completa (falta \\begin o \\end)"
    rows = {}
    for ln in tex[a:b].split("\n"):
        m = re.match(r"\s*\\citet\{([^}]+)\}", ln)
        if not m:
            continue
        cells = [c.strip() for c in ln.split("&")]
        if len(cells) > 5:
            d = re.search(r"(\d+\.\d+)", cells[5])
            if d:
                rows[m.group(1)] = d.group(1)
    body = tex[:a] + tex[b:]
    bad, compared = [], 0
    for m in re.finditer(r"\$(\d+\.\d+)\s*~?\\?,?\\mathrm\{kpc\}\$\s*\\citep\{([^},]+)\}", body):
        val, key = m.group(1), m.group(2)
        tab = rows.get(key)
        if tab is None:
            continue
        compared += 1
        if abs(float(val) - float(tab)) > 1e-9:
            bad.append(f"{key}: texto {val} vs Table A.1 {tab}")
    if not rows:
        return False, "no pude leer distancias de Table A.1"
    # Report comparisons made, not rows available. "19 referencias tabuladas" read as 19 verified
    # pairs when the text-side regex matches 3 -- the introduction's one sentence. The rest are
    # never quoted with a distance in the running text, and a value inside a multi-key \citep is
    # skipped outright. Saying so is what stops a maintainer assuming the coverage is complete.
    return not bad, (f"{compared} de {len(rows)} referencias tabuladas se citan con distancia en "
                     "el texto; sin desacuerdo" if not bad else f"{len(bad)} -> {bad}")


@check("el pie de Table A.1 declara el rango real de su propia columna")
def c_literature_span():
    """A caption that summarises its own table has to be recomputed from it, not remembered.

    Table A.1's caption states the span of adopted distances to argue the literature ages are not
    on a common scale. It said 0.83--1.70 kpc while the table also lists 2.13 and 0.76 -- correct
    for the fifteen rows that report an age, which is the set the argument is about, and wrong
    about "this table". The argument was sound and the sentence was not, which is the only kind of
    error a caption can make about the rows printed directly beneath it.
    """
    tex = TEX.read_text()
    i = tex.find(r"\label{tab:literature}")
    a = tex.rfind(r"\begin{table", 0, i)
    b = tex.find(r"\end{table", i)
    if i < 0 or a < 0 or b < 0:
        return False, "no encuentro Table A.1"
    dists, aged = [], 0
    for ln in tex[a:b].split("\n"):
        if r"\citet{" not in ln:
            continue
        cells = [c.strip() for c in ln.split("&")]
        if len(cells) < 9:
            continue
        age = cells[8].replace(r"\\", "").strip()
        if age in (r"$\cdots$", ""):
            continue  # no age quoted, so it is not on the axis the caption argues about
        aged += 1
        m = re.search(r"(\d+\.\d+)", cells[5])
        if m:
            dists.append(float(m.group(1)))
    if not dists:
        return False, "no pude leer distancias de Table A.1"
    claimed = re.search(r"span \$([\d.]+)\$--\$([\d.]+)\\,\\mathrm\{kpc\}", tex[a:b])
    if claimed is None:
        return False, "el pie de Table A.1 ya no declara un rango de distancias"
    lo, hi = float(claimed.group(1)), float(claimed.group(2))
    ok = abs(lo - min(dists)) < 0.005 and abs(hi - max(dists)) < 0.005
    # `dists` holds the rows that quote an age *and* a parseable distance; three rows quote an age
    # with no distance and never reach it. Saying "filas con edad" hid that, and the count printed
    # (12) contradicted the docstring's fifteen.
    return ok, (f"{lo}--{hi} kpc sobre las {len(dists)} de {aged} filas con edad que ademas citan "
                "distancia" if ok else f"el pie dice {lo}--{hi} y esas filas dan "
                f"{min(dists)}--{max(dists)}")


@check("las cifras del manuscrito se rederivan del catalogo entregado")
def c_catalog_numbers():
    """Recompute from the delivered .dat what the manuscript states, and require it to match.

    Every other consistency check here compares two texts. This one is the only external oracle
    in the package: the CDS catalogue is data, produced by the pipeline, not by the sentence that
    describes it. Twelve published quantities are recomputed: the four membership thresholds,
    compared against the single sentence that declares them; and eight more -- the reference-sample
    proper-motion means and dispersions, the parallax mean and dispersion over the
    delta_plx/plx < 0.1 subsample, the Sagitta PMS count and the YSO denominator -- required to
    appear somewhere in the manuscript, which is safe for them because a recomputed decimal that
    moves does not land on another number by accident.

    Six more were verified by hand on 2026-08-17 and are not guarded, because their printed form
    differs from their computed one and matching on a string would be a check that passes for the
    wrong reason: the four G < 19 subsets (288, 236, 191, 153), the brightest member (G = 8.80,
    computed as 8.8) and the median (G ~ 17.0). They are recorded here so the omission is a known
    gap rather than an assumed coverage.

    The comparison is deliberately "does the manuscript contain this value" rather than a list of
    expected constants: a hardcoded expectation drifts exactly like the prose it is meant to
    guard, whereas a number recomputed from the data and then looked for in the text fails when
    either side moves. It caught nothing on arrival -- all eighteen already matched -- so its
    value is entirely in the next refit.
    """
    import statistics as st
    if not CDS_DAT.exists():
        return False, "no existe el .dat del CDS"
    cols = {"Plx": (48, 55), "e_Plx": (57, 64), "pmRA": (66, 73), "pmDE": (84, 91),
            "Gmag": (102, 109), "pMember": (180, 186), "PMSProb": (190, 196),
            "Jmag": (129, 136), "Hmag": (138, 145), "Ksmag": (147, 154)}

    def val(line, a, b):
        s = line[a - 1:b].strip()
        return None if s in ("", "...") else float(s)

    rows = []
    for line in CDS_DAT.read_text().splitlines():
        r = {k: val(line, *v) for k, v in cols.items()}
        r["Ref"] = line[187:188].strip()
        rows.append(r)
    ref = [r for r in rows if r["Ref"] == "1"]
    has2m = lambda r: all(r[k] is not None for k in ("Jmag", "Hmag", "Ksmag"))

    # The four thresholds are anchored to the sentence that defines them rather than searched for
    # anywhere in the document. Tested: dropping one row makes the reference sample 253, which the
    # loose form does catch because "253" appears nowhere -- but the total becomes 320, which it
    # does NOT catch, because "320" happens to occur elsewhere in 26 pages. An integer check that
    # depends on the new value being absent by luck is decoration.
    # The Ref flag is documented as "1 if pMember >= 0.6 after clipping", so it is validated
    # against its own definition rather than trusted: every quantity below is computed over `ref`,
    # and a mis-set flag would otherwise agree with the manuscript while disagreeing with the
    # column it encodes.
    by_p = [r for r in rows if r["pMember"] >= 0.6]
    if len(ref) != len(by_p):
        return False, (f"la columna Ref marca {len(ref)} fuentes y pMember>=0.6 da {len(by_p)}; "
                       "el flag no cumple su propia definicion en el ReadMe")
    counts = [len([r for r in rows if r["pMember"] > 0.5]), len(by_p),
              len([r for r in rows if r["pMember"] >= 0.7]),
              len([r for r in rows if r["pMember"] >= 0.8])]
    sentence = re.search(r"NGC 6383 has [^.]*candidate members[^.]*\.", TEX.read_text())
    if sentence is None:
        return False, "no encuentro la frase que declara los cuatro umbrales de membresia"
    stated = [int(x) for x in re.findall(r"\$(\d+)\$", sentence.group(0))][:4]
    if stated != counts:
        return False, f"umbrales: el texto dice {stated}, el catalogo da {counts}"

    sub = [r for r in ref if r["e_Plx"] / abs(r["Plx"]) < 0.1]
    tex = TEX.read_text()
    # Each quantity is compared against the sentence that *states* it, not looked for anywhere in
    # the manuscript. Boundary-anchored containment is still not enough: with plain containment a
    # recomputed 193 matched inside the bibcode 1930LicOB..14..154T and 116 inside
    # 2005AA...438.1163K; with boundaries added, a recomputed 0.14 still matched the age/t_rh ratio
    # and 0.045 the metallicity prior. A number that lands on an unrelated quantity is a check
    # passing by coincidence, which is the rule the four thresholds above already follow.
    anchored = [
        ("con 2MASS en la referencia",
         len([r for r in ref if has2m(r)]),
         r"N_\{\\text\{cl\}\}=(\d+)\s*\$ is the number of reference-sample sources"),
        ("PMS>=0.6 en la referencia",
         len([r for r in ref if r["PMSProb"] is not None and r["PMSProb"] >= 0.6]),
         r"Applying Sagitta to the membership yields \$(\d+)\$"),
        ("media pmRA", round(st.mean(r["pmRA"] for r in ref), 3),
         r"The mean proper-motion values are \$([\d.]+)\\,"),
        ("media pmDE", round(st.mean(r["pmDE"] for r in ref), 3),
         r"in R\.A\. and \$(-[\d.]+)\\,"),
        ("dispersion pmRA", round(st.stdev(r["pmRA"] for r in ref), 3),
         r"with member dispersions of \$([\d.]+)\$"),
        ("dispersion pmDE", round(st.stdev(r["pmDE"] for r in ref), 3),
         r"with member dispersions of \$[\d.]+\$ and \$([\d.]+)\\,"),
        ("media de paralaje", round(st.mean(r["Plx"] for r in sub), 3),
         r"The mean parallax of the subsample used for the distance estimate is \$([\d.]+)\\,"),
        ("dispersion de paralaje", round(st.stdev(r["Plx"] for r in sub), 3),
         r"with a 1\$\\sigma\$ dispersion of \$([\d.]+)\\,\\mathrm\{mas\}\$"),
    ]
    missing = []
    for label, value, pattern in anchored:
        m = re.search(pattern, tex)
        if m is None:
            missing.append(f"{label}: no encuentro la frase que lo declara")
        elif abs(float(m.group(1)) - value) > 1e-9:
            missing.append(f"{label}: el texto dice {m.group(1)}, el catalogo da {value}")

    return not missing, (f"12 cantidades rederivadas del .dat y comparadas contra la frase que "
                         "las declara" if not missing else f"{len(missing)} -> {missing[:4]}")


# --------------------------------------------------------------------------------- CDS

@check("paquete CDS valido")
def c_cds():
    readme, dat = CDS_README.read_text(), CDS_DAT.read_text().rstrip("\n").split("\n")
    m = re.search(r"ngc6383_members\.dat\s+(\d+)\s+(\d+)", readme)
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
            rows.append((int(mm.group(1)), int(mm.group(2) or mm.group(1)),
                         mm.group(5), mm.group(6), line))
    cols = {line.index(lab, 17) for _, _, lab, _, line in rows}
    if len(cols) > 1:
        problems.append(f"columna Label desalineada: {sorted(cols)}")
    for a, b, lab, expl, _ in rows:
        nulls = sum(1 for d in dat if d[a - 1:b].strip() == "...")
        if bool(nulls) != expl.lstrip().startswith("?"):
            problems.append(f"{lab}: {nulls} nulos, marca '?' = {expl.lstrip()[:1]!r}")
    return not problems, "Lrecl, registros, alineacion y nulos correctos" if not problems else "; ".join(problems)


# ------------------------------------------------------------------------------- LaTeX

@check("chktex y lacheck")
def c_linters():
    # Esto contaba avisos y aceptaba "hasta 2". Un umbral no dice CUALES: si el aviso de guion se
    # arreglara y apareciera otro distinto, la cuenta seguiria en 2 y el check verde -- pasa sin
    # ver nada. Y cuando fallo en CI (3 en vez de 2) no habia forma de saber cual era el tercero,
    # porque imprimia una cifra. Ahora acepta por NUMERO de aviso, no por cantidad: el 8 es el
    # largo de guion, una regla de estilo sobre prosa cuyos casos reales ya cubre DashExcpt en
    # .chktexrc; cualquier otro numero falla y se imprime.
    a = run(["chktex", "-q", "-f", "%n|%l|%m\n", "aanda.tex"], cwd=TEX.parent)
    # Medido: `chktex fichero_inexistente.tex` devuelve rc=0 con stdout vacio y el aviso en stderr.
    # Sin mirar stderr, `warns` quedaba vacio, `unexpected` vacio, y el check reportaba
    # "0 avisos, todos aceptados" -- verde. Un renombre del .tex, un cwd equivocado, un .chktexrc
    # corrupto o un build que rechace el `-f` convertian los dos linters en un pase incondicional.
    if "Unable to open" in a.stderr or "unable to open" in a.stderr.lower():
        return False, f"chktex no pudo leer aanda.tex: {a.stderr.strip()[:90]}"
    warns = [w for w in a.stdout.replace("\\n", "\n").split("\n") if w.strip()]
    accept = ["8|"]  # nº8: largo de guion, ver DashExcpt en .chktexrc
    # El nº12 ("interword spacing") es un falso positivo de chktex < 1.7.9 sobre `($m=43$).
    # \textsc{`:
    # no distingue ese punto de una abreviatura. 1.7.9 lo distingue y no lo emite. La clase que el
    # 12
    # vigila es real aca -- 113 abreviaturas en el manuscrito -- asi que NO se desactiva: se acepta
    # solo bajo la version vieja, que es la que trae el runner de Ubuntu. La version se imprime,
    # para
    # que la divergencia entre los dos entornos quede a la vista en ambos registros en vez de
    # convertirse en la clase de silencio que este gate existe para sacar.
    ver = run(["chktex", "--version"]).stdout
    m = re.search(r"v(\d+)\.(\d+)\.(\d+)", ver)
    # Fallaba ABIERTO: sin `m`, `old_chktex` era True y la exencion del nº12 se concedia para
    # siempre. chktex 2.x, un build de distro, cualquier `--version` que no case, y la clase que el
    # nº12 vigila -- 113 abreviaturas en este manuscrito -- se aceptaba en silencio. Si no se sabe
    # que version es, se asume la estricta: una exencion se concede sabiendo, no por no saber.
    old_chktex = tuple(map(int, m.groups())) < (1, 7, 9) if m else False
    if old_chktex:
        accept.append("12|")
    unexpected = [w for w in warns if not any(w.startswith(k) for k in accept)]
    b = run(["lacheck", "aanda.tex"], cwd=TEX.parent)
    if "not open" in b.stderr.lower() or "no such file" in b.stderr.lower():
        return False, f"lacheck no pudo leer aanda.tex: {b.stderr.strip()[:90]}"
    lacheck = [l for l in b.stdout.split("\n")
               if l.strip() and "Dots should be ellipsis" not in l]
    ok = not unexpected and not lacheck
    vtag = (m.group(0) if m else "version desconocida") + (", nº12 aceptado" if old_chktex else "")
    detail = f"chktex {vtag}: {len(warns)} avisos, todos aceptados; lacheck 0"
    if not ok:
        detail = ("chktex inesperados: " + "; ".join(unexpected[:3]) if unexpected else "") + \
                 ("  lacheck: " + "; ".join(lacheck[:3]) if lacheck else "")
    return ok, detail


@check("typos")
def c_typos():
    # En CI la cobertura existe pero por otra via: `crate-ci/typos@master` corre como paso propio
    # del workflow, sobre ESTE directorio y con ESTE _typos.toml, y falla el job por su cuenta.
    # Lo que esa accion no hace es dejar el binario en PATH, asi que invocarlo aca reventaba con
    # FileNotFoundError y contaba como check fallado. Se omite nombrando quien cubre el hueco --
    # no es una excepcion, es la misma revision corriendo un escalon mas arriba.
    if shutil.which("typos") is None:
        raise Skipped("binario ausente; lo corre el paso crate-ci/typos del workflow")
    r = run(["typos", str(HERE)])
    return r.returncode == 0, "limpio" if r.returncode == 0 else r.stdout.strip()[:160]


@check("floats movidos limpiados del diff marcado")
def c_strip():
    """A moved float must not appear as a struck-through caption with no image.

    Without the strip, ten figures read to a referee as ten figures cut.
    """
    if not MARKED.exists():
        return False, "aanda_marked.tex no existe; corre latexdiff primero"
    body = MARKED.read_text()
    body = body[body.index(r"\begin{document}"):]
    spans = len(re.findall(r"%DIFDELCMD < \\begin\{(?:figure|table)\*?\}", body))
    return spans == 0, f"{spans} spans de float comentados (deben ser 0 tras strip_moved_floats.py)"


@check("numeracion de linea apagada en los apendices")
def c_linenumbers():
    """aa.cls's \\appendix ends in \\linenumbers and turns them back on."""
    tex = TEX.read_text()
    i = tex.find(r"\begin{appendix}")
    if i < 0:
        return False, "no hay bloque appendix"
    window = tex[i:i + 200]
    return r"\nolinenumbers" in window, (
        "reemitido tras \\begin{appendix}" if r"\nolinenumbers" in window
        else "FALTA: aa.cls las reactiva y se imprimen sobre el texto")


# ------------------------------------------------------------------------- build (slow)

@check("ambos documentos compilan sin errores", slow=True)
def c_build():
    """Build both, because both are uploaded and only one was ever audited.

    Until 2026-08-17 this compiled clean_source alone. The marked document was left to
    whatever log happened to be on disk -- which is how it sat with four TeX errors
    ("Misplaced \\noalign", "Illegal unit of measure") from a stale run while the gate
    reported a clean build. c_overfull reads these logs, so building here also guarantees
    it is reading logs this run produced rather than a previous state of the manuscript.
    """
    out = []
    for tag, tex in (("limpio", TEX), ("marcado", MARKED)):
        stem = tex.with_suffix("").name
        run(["latexmk", "-C", stem], cwd=tex.parent)
        run(["latexmk", "-pdf", "-bibtex", "-interaction=nonstopmode", tex.name], cwd=tex.parent)
        log = (tex.parent / f"{stem}.log").read_text(errors="replace")
        pages = re.search(rf"Output written on {stem}\.pdf \((\d+) pages", log)
        bad = {
            "errores TeX": len(re.findall(r"^! ", log, re.M)),
            "LaTeX Error": log.count("LaTeX Error"),
            "refs indef": len(re.findall(r"Reference .* undefined", log)),
            "citas indef": len(re.findall(r"Citation .* undefined", log)),
        }
        if any(bad.values()) or pages is None:
            # Recorded, not returned: returning here skipped the marked rebuild, leaving its log
            # from a previous run for c_overfull and c_manifest_pages to read -- the exact stale-log
            # state this check was written to end.
            out.append(f"FALLA {tag}: {pages.group(1) if pages else '?'} pp, "
                       + ", ".join(f"{k} {v}" for k, v in bad.items()))
        else:
            out.append(f"{tag} {pages.group(1)} pp")
    failed = [o for o in out if o.startswith("FALLA")]
    return not failed, ("; ".join(out) + ", 0 errores y 0 indefinidas en ambos" if not failed
                        else "; ".join(out))


@check("los PDF que se suben son los recien construidos", slow=True)
def c_deliverables():
    """The uploaded files are *copies*, and a copy is exactly as stale as you let it be.

    clean_source/aanda.pdf is the build; aa52082-24_marked_changes.pdf and friends are what
    NESTOR receives. Rebuilding without re-copying uploads the previous revision -- on
    2026-08-17 the copies were 24 minutes older than the build and still carried the marked
    document's undefined citation. Bytes cannot be compared (every build stamps a new
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

    def producer(p: Path) -> str:
        r = run(["pdfinfo", str(p)])
        m = re.search(r"^Producer:\s*(.+)$", r.stdout, re.M)
        return m.group(1).strip() if m else "?"

    pairs = [(TEX.parent / "aanda.pdf", HERE / "aanda_revised_clean.pdf"),
             (MARKED.parent / "aanda_marked.pdf", HERE / "aanda_revised_marked.pdf"),
             (MARKED.parent / "aanda_marked.pdf", HERE / "aa52082-24_marked_changes.pdf")]
    # Este check compara un PDF recien construido con uno versionado, y eso solo significa algo si
    # los construyo el mismo motor. Otro TeX Live guiona y corta lineas distinto, asi que el texto
    # extraido difiere por el entorno y no por estar desactualizado: en CI marcaba los tres como
    # obsoletos siempre. La condicion es medida, no una bandera -- el propio PDF dice quien lo hizo.
    for built, sent in pairs:
        if sent.exists() and producer(built) != producer(sent):
            # En la maquina que sube esta omision ya no es benigna: desde que las omisiones son
            # fatales sin `--allow-skips`, un desajuste de motor aqui hace que el gate se niegue a
            # bendecir. Que es lo correcto -- tras actualizar TeX Live, los PDF versionados llevan
            # el motor viejo y todo build fresco el nuevo, asi que el check se apagaba exactamente
            # cuando su respuesta correcta era "reconstruye y vuelve a copiar".
            raise Skipped(f"construido por {producer(built)}, enviado por {producer(sent)}: "
                          "otro motor de TeX. En la maquina que sube esto significa reconstruir "
                          "los entregables y volver a copiarlos")
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


@check("el MANIFEST declara el numero de paginas real", slow=True)
def c_manifest_pages():
    """MANIFEST.md is the upload instructions, and it states page counts as fact.

    It said "26 pp clean / 29 pp marked" in the header and "30 pp" in the file table, three
    lines apart, both hand-maintained. The reason to check rather than to stop writing them
    down is that the count is how a human notices a truncated or double-built PDF before
    sending it. Every "NN pp" in the file is compared against the two builds.
    """
    manifest = HERE / "MANIFEST.md"
    if not manifest.exists():
        return False, "no existe MANIFEST.md"
    real = {}
    for tag, tex in (("limpio", TEX), ("marcado", MARKED)):
        stem = tex.with_suffix("").name
        log = tex.parent / f"{stem}.log"
        m = re.search(rf"Output written on {stem}\.pdf \((\d+) pages", log.read_text(errors="replace"))
        if not m:
            return False, f"no se pudo leer las paginas del build {tag}"
        real[tag] = int(m.group(1))
    claimed = {int(n) for n in re.findall(r"(\d+)\s*pp", manifest.read_text())}
    # Both directions. Checking only "no claimed count is wrong" passes on a MANIFEST that swaps
    # the two labels, drops one count, or states none at all -- and a missing or swapped count is
    # exactly the hand-maintenance error the check exists to stop.
    wrong = sorted(claimed - set(real.values()))
    absent = sorted(v for v in real.values() if v not in claimed)
    if wrong or absent:
        return False, (f"el MANIFEST afirma {wrong} que los builds no dan; " if wrong else "") + \
                      (f"y no declara {absent}" if absent else "")
    return True, f"limpio {real['limpio']} pp, marcado {real['marcado']} pp, ambos declarados"


@check("sin texto fuera de columna en ninguno de los dos PDF", slow=True)
def c_overfull():
    """An Overfull \\hbox is text sticking out of the column, and in two-column A&A it lands
    on top of the neighbouring column.

    This check did not exist until a human looked at the marked PDF and said it looked
    awful. The clean build had been checked for Overfull and had none; the marked build had
    never been checked at all and had twelve, because latexdiff's default UNDERLINE markup
    strikes deleted text with ulem's \\sout, which cannot break across lines. A long struck
    citation list therefore ran off the column and overprinted the text beside it.

    The lesson is not about latexdiff. Eleven checks were written and none of them looked at
    the artefact the referee actually opens.
    """
    # The marked build carries one documented residual: the footnote that shows the old
    # GitHub URL beside the new one, two long unbreakable URLs on a single footnote line at
    # footnote size, 20.4pt over. xurl and \\sloppy were both tried and neither absorbs it.
    # It is allowed by exact count, so a second overfull box still fails the gate.
    allowed = {"limpio": 0, "marcado": 1}
    bad = {}
    for tag, log in (("limpio", TEX.parent / "aanda.log"),
                     ("marcado", MARKED.parent / "aanda_marked.log")):
        if not log.exists():
            bad[tag] = "sin log"
            continue
        n = log.read_text().count("Overfull \\hbox")
        if n != allowed[tag]:
            bad[tag] = f"{n} (esperado {allowed[tag]})"
    return not bad, ("limpio 0, marcado 1 documentado" if not bad
                     else f"cajas desbordadas: {bad}")


@check("el zip enviado compila solo", slow=True)
def c_zip():
    import tempfile, zipfile
    zp = HERE / "aa52082-24_source.zip"
    if not zp.exists():
        # Artefacto local y regenerable (.gitignore lo excluye a proposito, y MANIFEST.md trae el
        # comando). Es la ranura obligatoria de NESTOR, asi que el check pertenece a la maquina que
        # sube, no al runner: alli no hay zip que revisar y fallar solo dice que no lo hay.
        raise Skipped("el zip es artefacto local y no se versiona; el check corre donde se sube")
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        tex_files = [n for n in names if n.endswith(".tex")]
        if len(tex_files) != 1:
            return False, f"NESTOR exige un solo .tex; el zip trae {len(tex_files)}"
        # "It compiles" is not "it is the current manuscript". A zip built before the last edit
        # compiles perfectly and to the same page count -- on 2026-08-17 this check passed on a
        # zip 24 minutes stale, missing the Sect. 8 edit, while every other check was green. This
        # is the *mandatory* slot: NESTOR builds the referee's PDF from it, so a stale zip is the
        # one defect here that reaches print.
        # Every entry, not a named handful. The first version listed five files and skipped the
        # 21 figures, so regenerating a figure without rebuilding the zip left the check green --
        # and NESTOR builds the referee's PDF from this zip, so the referee would read the new text
        # against the old figure. Figure regeneration is routine in this tree.
        stale = []
        for n in names:
            if n.endswith("/"):
                continue
            src = TEX.parent / n
            if not src.exists():
                stale.append(f"{n} no existe en clean_source")
            elif z.read(n) != src.read_bytes():
                stale.append(n)
        if stale:
            return False, f"el zip no coincide con clean_source ({len(stale)}): {stale[:5]}"
        with tempfile.TemporaryDirectory() as td:
            z.extractall(td)
            run(["latexmk", "-pdf", "-bibtex", "-interaction=nonstopmode", "aanda.tex"], cwd=Path(td))
            log = Path(td, "aanda.log")
            if not log.exists():
                return False, "no compilo"
            t = log.read_text()
            n = len(re.findall(r"(Reference|Citation) .* undefined", t))
            p = re.search(r"Output written on aanda\.pdf \((\d+) pages", t)
            return (t.count("LaTeX Error") == 0 and n == 0,
                    f"{p.group(1) if p else '?'} pp aislado, {n} indefinidas, 1 .tex")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quick", action="store_true", help="omite lo que necesita compilar LaTeX")
    ap.add_argument("--allow-skips", action="store_true",
                    help="una omision no es fatal. Solo para CI: en la maquina que sube, un check "
                         "omitido es un check que no se hizo donde importa")
    args = ap.parse_args()

    print("=== consistencia entre el manuscrito y lo que lo describe ===")
    c_posterior(); c_letter_numbers(); c_kb(); c_register(); c_overclaim()
    c_dropped_symbols(); c_spelling(); c_section_refs(); c_copies(); c_cds_claim(); c_marked_fresh(); c_cds(); c_table1(); c_paraphrase(); c_literature_agreement(); c_literature_span(); c_catalog_numbers()
    print("\n=== fuente LaTeX ===")
    c_linters(); c_typos(); c_strip(); c_linenumbers()
    # The slow group is measured, not listed. A second hand-maintained copy of these five names
    # is the drift the `declared`/`forgotten` guard exists to remove: renaming a @check string, or
    # adding a sixth slow check, would make --quick fail while naming a check that did run.
    if not args.quick:
        print("\n=== compilacion ===")
        c_build(); c_zip(); c_deliverables(); c_manifest_pages(); c_overfull()

    ran = {n for n, _, _ in results}
    forgotten = [n for n in declared
                 if n not in ran and not (args.quick and n in slow_checks)]
    if forgotten:
        print(f"\nFALLA  checks declarados que main() no llama: {forgotten}")
        return 1

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed) - len(skipped)}/{len(results)} pasan"
          + (f", {len(skipped)} omitidos" if skipped else ""))
    for n, why in skipped:
        print(f"  omitido  {n}: {why}")
    if failed:
        print("FALLAN: " + ", ".join(failed))
        return 1
    # La bendicion es lo unico sobre lo que alguien actua, asi que no puede sobrevivir a una
    # omision. `c_zip` paso de fallar a omitirse cuando el zip no esta: en el runner eso es
    # correcto, pero en la maquina que sube significa que la ranura obligatoria de NESTOR no
    # existe -- y el gate igual imprimia "puede subirse". Una omision aca abajo es informacion
    # que se pierde justo donde se toma la decision.
    # La bendicion se hizo consciente de las omisiones y no del modo rapido: `--quick` imprimia
    # "puede subirse" sin haber corrido la ranura obligatoria de NESTOR ni ninguna de las dos
    # compilaciones. Y `REVISADO PARCIAL` salia 0, o sea que cualquier consumidor automatico
    # -- un `gate.py && subir`, un paso de CI -- leia una omision como exito, y solo un humano
    # leyendo las dos ultimas lineas se enteraba.
    #
    # `--allow-skips` es la salida: las omisiones son fatales por defecto y solo el runner las
    # perdona, porque es el unico sitio donde son legitimas. En la maquina que sube no hay ninguna,
    # asi que la bandera no cuesta nada aqui y hace que la MAQUINA, no el lector, imponga que no se
    # omitio nada donde importa.
    # Las dos razones de una revision parcial se reportan JUNTAS, y la fatalidad la decide la
    # omision. La primera version devolvia 0 en cuanto veia `--quick`, antes de mirar `skipped`, asi
    # que el modo rapido se tragaba las omisiones y volvia a colar exactamente el agujero que este
    # bloque cierra. Lo caza `test_gate_behaviour.py`.
    parcial = []
    if args.quick:
        parcial.append("modo --quick: no corrieron " + ", ".join(sorted(slow_checks)))
    if skipped:
        parcial.append(f"{len(skipped)} omitidos: " + ", ".join(n for n, _ in skipped)
                       + ". En la maquina que sube no deberia omitirse ninguno")
    if parcial:
        print("REVISADO PARCIAL - " + "; ".join(parcial) + ".")
        return 1 if (skipped and not args.allow_skips) else 0
    print("OK - el paquete puede subirse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
