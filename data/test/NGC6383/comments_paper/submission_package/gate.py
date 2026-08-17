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
import re
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
KB_NOTES = [Path.home() / "phd" / "kb" / "papers" / "2024arXiv240509145P.md",
            Path.home() / "phd" / "kb" / "objects" / "ngc-6383.md"]

results: list[tuple[str, bool, str]] = []
# @check registers a name only when the function is *called*, so a check that main() forgets to
# call is indistinguishable from one that passes -- which is how c_overclaim was written, printed
# nothing, and left the gate reporting 10/10. Declared names are recorded at import; main() checks
# every one of them ran.
declared: list[str] = []


def check(name: str):
    def deco(fn):
        declared.append(name)

        def wrapper(*a, **kw):
            try:
                ok, detail = fn(*a, **kw)
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
    tex = TEX.read_text().lower()
    found = []
    for path in LETTERS:
        text = path.read_text().lower()
        found += [f"{path.name}:{w}" for w in OVERCLAIM if w in text and w not in tex]
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
    return not bad, ("sin subindices sin simbolo" if not bad else f"{len(bad)} -> {bad[:4]}")



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
    if MARKED.stat().st_mtime < revised.stat().st_mtime:
        return False, "aanda_marked.tex mas viejo que new_revised.tex; falta correr latexdiff"
    return True, "al dia con clean_source"


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
    if i < 0 or a < 0:
        return False, "no encuentro Table A.1"
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
    bad = []
    for m in re.finditer(r"\$(\d+\.\d+)\s*~?\\?,?\\mathrm\{kpc\}\$\s*\\citep\{([^},]+)\}", body):
        val, key = m.group(1), m.group(2)
        tab = rows.get(key)
        if tab is None:
            continue
        if abs(float(val) - float(tab)) > 1e-9:
            bad.append(f"{key}: texto {val} vs Table A.1 {tab}")
    if not rows:
        return False, "no pude leer distancias de Table A.1"
    return not bad, (f"{len(rows)} referencias tabuladas, sin desacuerdo con el texto" if not bad
                     else f"{len(bad)} -> {bad}")


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
    if i < 0 or a < 0:
        return False, "no encuentro Table A.1"
    dists = []
    for ln in tex[a:b].split("\n"):
        if r"\citet{" not in ln:
            continue
        cells = [c.strip() for c in ln.split("&")]
        if len(cells) < 9:
            continue
        age = cells[8].replace(r"\\", "").strip()
        if age in (r"$\cdots$", ""):
            continue  # no age quoted, so it is not on the axis the caption argues about
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
    return ok, (f"{lo}--{hi} kpc sobre las {len(dists)} filas con edad" if ok
                else f"el pie dice {lo}--{hi} y las filas con edad dan "
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
    counts = [len([r for r in rows if r["pMember"] > 0.5]), len(ref),
              len([r for r in rows if r["pMember"] >= 0.7]),
              len([r for r in rows if r["pMember"] >= 0.8])]
    sentence = re.search(r"NGC 6383 has [^.]*candidate members[^.]*\.", TEX.read_text())
    if sentence is None:
        return False, "no encuentro la frase que declara los cuatro umbrales de membresia"
    stated = [int(x) for x in re.findall(r"\$(\d+)\$", sentence.group(0))][:4]
    if stated != counts:
        return False, f"umbrales: el texto dice {stated}, el catalogo da {counts}"

    derived = {
        "con 2MASS en la referencia": len([r for r in ref if has2m(r)]),
        "PMS>=0.6 en la referencia": len([r for r in ref if r["PMSProb"] is not None
                                          and r["PMSProb"] >= 0.6]),
        "media pmRA": round(st.mean(r["pmRA"] for r in ref), 3),
        "media pmDE": round(st.mean(r["pmDE"] for r in ref), 3),
        "dispersion pmRA": round(st.stdev(r["pmRA"] for r in ref), 3),
        "dispersion pmDE": round(st.stdev(r["pmDE"] for r in ref), 3),
    }
    sub = [r for r in ref if r["e_Plx"] / abs(r["Plx"]) < 0.1]
    derived["media de paralaje"] = round(st.mean(r["Plx"] for r in sub), 3)
    derived["dispersion de paralaje"] = round(st.stdev(r["Plx"] for r in sub), 3)

    tex = TEX.read_text()
    missing = [f"{k}={v}" for k, v in derived.items() if str(v).lstrip("-") not in tex]
    return not missing, (f"4 umbrales anclados a su frase + {len(derived)} cantidades "
                         "rederivadas del .dat y presentes en el texto"
                         if not missing else f"el texto no dice: {missing}")


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
    a = run(["chktex", "-q", "-f", "%n\n", "aanda.tex"], cwd=TEX.parent)
    n_chktex = len([x for x in a.stdout.strip().split("\n") if x.strip()])
    b = run(["lacheck", "aanda.tex"], cwd=TEX.parent)
    n_lacheck = len([l for l in b.stdout.split("\n")
                     if l.strip() and "Dots should be ellipsis" not in l])
    # 2 documented chktex residuals: "Sh 2-012" on lines 31 and 48, see .chktexrc
    return (n_chktex <= 2 and n_lacheck == 0,
            f"chktex {n_chktex} (2 residuos documentados), lacheck {n_lacheck}")


@check("typos")
def c_typos():
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

@check("ambos documentos compilan sin errores")
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
            return False, f"{tag}: {pages.group(1) if pages else '?'} pp, " + \
                          ", ".join(f"{k} {v}" for k, v in bad.items())
        out.append(f"{tag} {pages.group(1)} pp")
    return True, ", ".join(out) + ", 0 errores y 0 indefinidas en ambos"


@check("los PDF que se suben son los recien construidos")
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
        return r.stdout if r.returncode == 0 else None

    pairs = [(TEX.parent / "aanda.pdf", HERE / "aanda_revised_clean.pdf"),
             (MARKED.parent / "aanda_marked.pdf", HERE / "aanda_revised_marked.pdf"),
             (MARKED.parent / "aanda_marked.pdf", HERE / "aa52082-24_marked_changes.pdf")]
    stale = []
    for built, sent in pairs:
        if not sent.exists():
            stale.append(f"{sent.name} no existe")
        elif text(built) != text(sent):
            stale.append(sent.name)
    return not stale, "identicos al build" if not stale else f"desactualizados: {stale}"


@check("el MANIFEST declara el numero de paginas real")
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
    wrong = sorted(claimed - set(real.values()))
    return not wrong, (f"limpio {real['limpio']} pp, marcado {real['marcado']} pp"
                       if not wrong else f"el MANIFEST afirma {wrong} y los builds dan {real}")


@check("sin texto fuera de columna en ninguno de los dos PDF")
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


@check("el zip enviado compila solo")
def c_zip():
    import tempfile, zipfile
    zp = HERE / "aa52082-24_source.zip"
    if not zp.exists():
        return False, "no existe el zip"
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
        stale = [n for n in ("aanda.tex", "cites.bib", "aanda.bbl", "aa.cls", "aa.bst")
                 if n in names and z.read(n) != (TEX.parent / n).read_bytes()]
        if stale:
            return False, f"el zip no coincide con clean_source: {stale}"
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
    args = ap.parse_args()

    print("=== consistencia entre el manuscrito y lo que lo describe ===")
    c_posterior(); c_letter_numbers(); c_kb(); c_register(); c_overclaim()
    c_dropped_symbols(); c_copies(); c_cds_claim(); c_marked_fresh(); c_cds(); c_table1(); c_paraphrase(); c_literature_agreement(); c_literature_span(); c_catalog_numbers()
    print("\n=== fuente LaTeX ===")
    c_linters(); c_typos(); c_strip(); c_linenumbers()
    if not args.quick:
        print("\n=== compilacion ===")
        c_build(); c_zip(); c_deliverables(); c_manifest_pages(); c_overfull()

    ran = {n for n, _, _ in results}
    slow = {"ambos documentos compilan sin errores", "el zip enviado compila solo", "los PDF que se suben son los recien construidos", "el MANIFEST declara el numero de paginas real",
            "sin texto fuera de columna en ninguno de los dos PDF"}
    forgotten = [n for n in declared if n not in ran and not (args.quick and n in slow)]
    if forgotten:
        print(f"\nFALLA  checks declarados que main() no llama: {forgotten}")
        return 1

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} pasan")
    if failed:
        print("FALLAN: " + ", ".join(failed))
        return 1
    print("OK - el paquete puede subirse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
