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
KB_NOTE = Path.home() / "phd" / "kb" / "papers" / "2024arXiv240509145P.md"

results: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
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
    if not KB_NOTE.exists():
        return False, f"nota no encontrada: {KB_NOTE}"
    note, tex = KB_NOTE.read_text(), TEX.read_text()
    anchors = {
        "R_t": (r"R_t = 54\^\{\+7\}_\{-11\}", r"R_t\s*40\.4"),
        "R_c": (r"1\.96\^\{\+0\.19\}_\{-0\.16\}", r"R_c\s*1\.95"),
    }
    stale = []
    for label, (in_tex, in_note) in anchors.items():
        if re.search(in_tex, tex) and re.search(in_note, note):
            stale.append(label)
    return not stale, ("sin cifras superadas" if not stale
                       else f"la nota cita valores de ronda 1: {stale}")


@check("copias de las cartas y del ReadMe sincronizadas")
def c_copies():
    pairs = [
        (HERE / "letters" / "cover_letter_round2.txt",
         HERE.parent / "referee_round2" / "cover_letter_round2.txt"),
        (HERE / "letters" / "response_to_referee_round2.txt",
         HERE.parent / "referee_round2" / "response_letter.txt"),
        (CDS_README, HERE / "cds" / "ReadMe"),
    ]
    bad = [f"{a.name}" for a, b in pairs if not b.exists() or a.read_bytes() != b.read_bytes()]
    return not bad, "todas iguales" if not bad else f"divergen: {bad}"


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

@check("compilacion limpia")
def c_build():
    run(["latexmk", "-C", "aanda"], cwd=TEX.parent)
    run(["latexmk", "-pdf", "-bibtex", "-interaction=nonstopmode", "aanda.tex"], cwd=TEX.parent)
    log = (TEX.parent / "aanda.log").read_text()
    pages = re.search(r"Output written on aanda\.pdf \((\d+) pages", log)
    bad = {
        "LaTeX Error": log.count("LaTeX Error"),
        "refs indefinidas": len(re.findall(r"Reference .* undefined", log)),
        "citas indefinidas": len(re.findall(r"Citation .* undefined", log)),
    }
    return (not any(bad.values()) and pages is not None,
            f"{pages.group(1) if pages else '?'} pp, " + ", ".join(f"{k} {v}" for k, v in bad.items()))


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
    c_posterior(); c_letter_numbers(); c_kb(); c_copies(); c_cds()
    print("\n=== fuente LaTeX ===")
    c_linters(); c_typos(); c_strip(); c_linenumbers()
    if not args.quick:
        print("\n=== compilacion ===")
        c_build(); c_zip()

    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} pasan")
    if failed:
        print("FALLAN: " + ", ".join(failed))
        return 1
    print("OK - el paquete puede subirse.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
