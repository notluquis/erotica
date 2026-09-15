#!/usr/bin/env python3
"""Cada check del gate, roto a propósito una vez, tiene que ponerse rojo.

`test_gate_behaviour.py` prueba lo que el gate promete de sí mismo (`--quick` no bendice, una
omisión no sale 0). Esto prueba la otra mitad, que no tenía fichero: que **cada uno de los 26
checks puede fallar** por la razón que lo justifica. Hasta 2026-09-15 esas mutaciones se habían
hecho a mano, de a una, al escribir cada check, y nada las repetía.

Existe para el refactor `TOOL-gate-generalize` (hub, `open-threads.md`): el gate parametrizado no
está hecho hasta que los 26 veredictos se reproduzcan **y** estas mutaciones vuelvan a ponerse rojas
contra él. Por eso los checks se llaman a través de un adaptador: hoy el único es `legacy`, que carga
una copia de este `gate.py` sin modificarlo.

Nunca toca el árbol trackeado. Copia `submission_package/`, `cds_final/`, `referee_round3/`, la base
de `_legacy/` y las dos notas de `phd-kb` a un directorio temporal; el `gate.py` copiado resuelve
`HERE` desde su propia ubicación, así que todo lo relativo apunta a la copia, y `KB_ROOT` se reasigna
a mano porque es absoluto.

Cada mutación exige que su ancla exista antes de mutar: una mutación que no llegó al objeto deja el
check verde por la razón equivocada, y eso se reporta como sonda rota, no como check sordo.

    python3 test_gate_mutations.py          # los 21 rápidos, ~1 min
    python3 test_gate_mutations.py --slow   # más los 5 que compilan, ~15 min
    python3 test_gate_mutations.py --allow-skips   # CI: sin `kb` ni `typos`, esas se omiten
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent  # comments_paper/


class ProbeMissed(Exception):
    """La mutación no encontró su ancla: no se puede concluir nada del veredicto."""


journal: dict[Path, bytes | None] = {}


def _remember(p: Path) -> None:
    if p not in journal:
        journal[p] = p.read_bytes() if p.exists() else None


def restore() -> None:
    for p, data in journal.items():
        if data is None:
            p.unlink(missing_ok=True)
        else:
            p.write_bytes(data)
    journal.clear()


def sub(p: Path, pattern: str, repl, count: int = 1) -> None:
    _remember(p)
    text = p.read_text()
    new, n = re.subn(pattern, repl, text, count=count)
    if n == 0:
        raise ProbeMissed(f"{p.name}: no casa {pattern!r}")
    p.write_text(new)


def append(p: Path, text: str) -> None:
    _remember(p)
    p.write_text(p.read_text() + text)


def insert_body(root: Path, text: str) -> None:
    sub(root / "submission_package/clean_source/aanda.tex", r"(?=\\begin\{appendix\})",
        lambda _m: f"\n{text}\n\n")


def copy_over(src: Path, dst: Path) -> None:
    _remember(dst)
    dst.write_bytes(src.read_bytes())


# ---------------------------------------------------------------------------------- adaptadores

class Legacy:
    """El `gate.py` de hoy, cargado desde la copia. Sus checks devuelven un bool y dejan
    `(nombre, ok, detalle)` en `results`; una omisión queda en `skipped`."""

    def __init__(self, root: Path):
        spec = importlib.util.spec_from_file_location(
            "gate_under_test", root / "submission_package" / "gate.py")
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        kb = root / "kb"
        self.mod.KB_NOTES = [kb / p.relative_to(self.mod.KB_ROOT) for p in self.mod.KB_NOTES]
        self.mod.KB_ROOT = kb

    def run(self, fn: str) -> tuple[str, str]:
        n_skip = len(self.mod.skipped)
        getattr(self.mod, fn)()
        _, ok, detail = self.mod.results[-1]
        if len(self.mod.skipped) > n_skip:
            return "skip", detail
        return ("ok" if ok else "fail"), detail


ADAPTERS = {"legacy": Legacy}


def make_copy(dst: Path) -> Path:
    ignore = shutil.ignore_patterns("_gate_build", "__pycache__")
    for d in ("submission_package", "cds_final", "referee_round3", "_legacy/cds_round2_submitted"):
        shutil.copytree(PAPER / d, dst / d, ignore=ignore, symlinks=True)
    import importlib.util as u
    spec = u.spec_from_file_location("gate_paths", HERE / "gate.py")
    g = u.module_from_spec(spec)
    spec.loader.exec_module(g)
    for note in g.KB_NOTES:
        if not note.exists():
            continue  # CI no clona `kb`; c_kb se omite alli y la mutacion lo cuenta como omitida
        target = dst / "kb" / note.relative_to(g.KB_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(note, target)
    return dst


# ------------------------------------------------------------------------------------ mutaciones
# (función del check, qué rompe, mutación). La mutación rompe lo que el check PROMETE, no el diff
# que lo introdujo; varios checks llevan más de una porque vigilan más de un ancla.

SP = "submission_package"
TEX = f"{SP}/clean_source/aanda.tex"
RESP = f"{SP}/letters/response_to_referee_round3.txt"
COVER = f"{SP}/letters/cover_letter_round3.txt"


def _near_number(root: Path) -> str:
    """Un decimal a <0.5% de uno del manuscrito que el manuscrito no contiene."""
    tex = (root / TEX).read_text()
    for m in re.finditer(r"(?<![\w.])(\d{2}\.\d{2})(?![\w.])", tex):
        v = float(m.group(1)) * 1.003
        s = f"{v:.2f}"
        if s != m.group(1) and s not in tex:
            return s
    raise ProbeMissed("no hay decimal NN.NN en el manuscrito")


QUICK = [
    ("c_posterior", "frase R17 verbatim en la carta de ronda 3",
     lambda r: append(r / RESP, "\nWe show posterior distributions for the astrometric, "
                                "structural, and age parameters.\n")),
    ("c_letter_numbers", "la carta cita un valor redondeado distinto del manuscrito",
     lambda r: append(r / RESP, f"\nThe value is {_near_number(r)} in the text.\n")),
    ("c_kb", "la nota vuelve a citar R_t 40.4",
     lambda r: append(r / "kb/objects/ngc-6383.md", "\nR_t 40.4 arcmin\n")),
    ("c_kb", "la nota vuelve a citar R_c 1.95",
     lambda r: append(r / "kb/objects/ngc-6383.md", "\nR_c 1.95 arcmin\n")),
    ("c_kb", "la nota vuelve a citar t_rh ~ 0.19",
     lambda r: append(r / "kb/objects/ngc-6383.md", "\nt_rh ~ 0.19\n")),
    ("c_register", "muletilla de registro en el cuerpo",
     lambda r: insert_body(r, "It is worth noting that the cluster is young.")),
    ("c_overclaim", "la carta afirma de más",
     lambda r: append(r / COVER, "\nThis demonstrably settles it.\n")),
    ("c_dropped_symbols", "subíndice sin símbolo",
     lambda r: append(r / RESP, "\nthe ratio _max is defined\n")),
    ("c_dropped_symbols", "guion partido al re-envolver",
     lambda r: append(r / RESP, "\na tie- breaker was used\n")),
    ("c_spelling", "forma británica en el manuscrito",
     lambda r: insert_body(r, "The colour index is used.")),
    ("c_section_refs", "la carta cita una sección inexistente",
     lambda r: append(r / RESP, "\nSee Sect. 9.9 for details.\n")),
    ("c_copies", "la copia en referee_round3 diverge",
     lambda r: append(r / "referee_round3/cover_letter_round3.txt", " ")),
    ("c_copies", "cites.bib de marked_changes diverge",
     lambda r: append(r / f"{SP}/marked_changes/cites.bib", "\n")),
    ("c_cds_claim", "el ReadMe cambia y la carta no lo declara",
     lambda r: append(r / "cds_final/ReadMe", "\n")),
    ("c_marked_fresh", "se edita el manuscrito sin copiarlo a new_revised.tex",
     lambda r: insert_body(r, "Edited.")),
    ("c_table1", "sigma(t_seg) vuelve al valor sin propagar",
     lambda r: sub(r / TEX, r"(Minimum segregation time\s*&\s*\$[\d.]+\s*\\pm\s*)[\d.]+",
                   r"\g<1>1.24")),
    ("c_paraphrase", "variante que afirma de más",
     lambda r: insert_body(r, "The window is a genuine background annulus.")),
    ("c_literature_agreement", "el texto y Table A.1 discrepan en una distancia",
     lambda r: sub(r / TEX, r"\$0\.83~\\mathrm\{kpc\}\$(\s*\\citep\{2018AA\.\.\.610A\.\.30A\})",
                   r"$0.840~\\mathrm{kpc}$\g<1>")),
    ("c_literature_span", "el pie de Table A.1 declara otro rango",
     # Anclada a `kpc`: la primera versión casó un "span $" de Sect. 6 y dejó el pie intacto.
     lambda r: sub(r / TEX, r"span \$[\d.]+\$(--\$[\d.]+\\,\\mathrm\{kpc\})", r"span $0.50$\g<1>")),
    ("c_catalog_numbers", "el texto cambia la media de pmRA",
     lambda r: sub(r / TEX, r"(The mean proper-motion values are \$)([\d.]+)",
                   lambda m: m.group(1) + f"{float(m.group(2)) + 0.001:.3f}")),
    ("c_catalog_numbers", "el catálogo pierde una fila (lado de los datos)",
     lambda r: sub(r / "cds_final/ngc6383_members.dat", r"\A[^\n]*\n", "")),
    ("c_cds", "Records del ReadMe no coincide con las filas",
     lambda r: sub(r / "cds_final/ReadMe", r"(ngc6383_members\.dat\s+\d+\s+)(\d+)",
                   lambda m: m.group(1) + str(int(m.group(2)) + 1))),
    ("c_linters", "rango con guion corto",
     lambda r: insert_body(r, "Ages of 1-10 Myr are common.")),
    ("c_typos", "errata",
     lambda r: insert_body(r, "This is " + "t" + "eh cluster.")),
    ("c_strip", "float movido comentado en el diff marcado",
     lambda r: sub(r / f"{SP}/marked_changes/aanda_marked.tex", r"(\\begin\{document\})",
                   lambda m: m.group(1) + "\n%DIFDELCMD < \\begin{figure}\n")),
    ("c_linenumbers", "falta \\nolinenumbers tras \\begin{appendix}",
     lambda r: sub(r / TEX, r"(\\begin\{appendix\}[\s\S]{0,200}?)\\nolinenumbers", r"\g<1>")),
]

SLOW_CHECKS = ["c_build", "c_zip", "c_deliverables", "c_manifest_pages", "c_overfull"]


def check_all_quick(gate, names, omitted: list[str] | None) -> list[str]:
    bad = []
    for fn in names:
        status, detail = gate.run(fn)
        if status == "skip" and omitted is not None:
            omitted.append(f"{fn} (base)")
        elif status != "ok":
            bad.append(f"{fn}: {status} en la copia sin mutar -> {detail}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--slow", action="store_true", help="incluye los 5 checks que compilan")
    ap.add_argument("--allow-skips", action="store_true",
                    help="un check omitido no es un problema. Solo para CI, igual que en gate.py")
    ap.add_argument("--target", choices=sorted(ADAPTERS), default="legacy")
    args = ap.parse_args()

    bad: list[str] = []
    omitted: list[str] = []
    with tempfile.TemporaryDirectory(prefix="gate_mut_") as td:
        root = make_copy(Path(td))
        gate = ADAPTERS[args.target](root)

        declared = {name for name in gate.mod.declared}
        fns = sorted({n for n in dir(gate.mod) if n.startswith("c_")})
        covered = {fn for fn, _, _ in QUICK} | set(SLOW_CHECKS)
        if len(fns) != len(declared) or set(fns) - covered:
            bad.append(f"checks sin mutación: {sorted(set(fns) - covered)} "
                       f"({len(fns)} funciones, {len(declared)} declarados)")

        quick_fns = sorted({fn for fn, _, _ in QUICK})
        base = check_all_quick(gate, quick_fns, omitted if args.allow_skips else None)
        print(f"base sin mutar: {len(quick_fns) - len(base)}/{len(quick_fns)} rápidos en verde")
        bad += base

        for fn, what, mutate in QUICK:
            try:
                mutate(root)
            except ProbeMissed as exc:
                bad.append(f"{fn} [{what}]: sonda rota -> {exc}")
                restore()
                continue
            status, detail = gate.run(fn)
            restore()
            if status == "skip" and args.allow_skips:
                omitted.append(f"{fn} [{what}]")
                continue
            mark = "rojo " if status == "fail" else "SORDO"
            print(f"  {mark} {fn} [{what}]: {detail[:110]}")
            if status != "fail":
                bad.append(f"{fn} [{what}]: {status} con la mutación puesta -> {detail}")

        if args.slow:
            bad += slow_suite(gate, root)

    print(f"\n{'FALLA' if bad else 'OK'} - {len(bad)} problema(s)"
          + (f", {len(omitted)} omitido(s): {omitted}" if omitted else ""))
    for b in bad:
        print(f"  - {b}")
    return 1 if bad else 0


def slow_suite(gate, root: Path) -> list[str]:
    bad = []

    def expect(fn: str, want: str, what: str) -> None:
        status, detail = gate.run(fn)
        mark = "ok   " if status == want else "MAL  "
        print(f"  {mark} {fn} [{what}]: esperado {want}, dio {status}: {detail[:100]}")
        if status != want:
            bad.append(f"{fn} [{what}]: esperado {want}, dio {status} -> {detail}")

    sp = root / SP
    print("lentos, base:")
    for fn in SLOW_CHECKS:
        expect(fn, "ok", "sin mutar")

    print("lentos, mutaciones:")
    sub(sp / "MANIFEST.md", r"\b30 pp\b", "29 pp", count=0)
    expect("c_manifest_pages", "fail", "el MANIFEST declara otra cuenta de páginas")
    restore()

    copy_over(sp / "aanda_revised_marked.pdf", sp / "aanda_revised_clean.pdf")
    expect("c_deliverables", "fail", "el PDF limpio enviado no es el recién construido")
    restore()

    append(sp / "clean_source/cites.bib", "\n% stale\n")
    expect("c_zip", "fail", "el zip no coincide con clean_source")
    restore()

    insert_body(root, r"\thiscontrolsequenceisundefined")
    expect("c_build", "fail", "secuencia de control indefinida")
    restore()

    insert_body(root, r"Overfull here \hbox to 1.5\hsize{} and there.")
    gate.run("c_build")
    expect("c_overfull", "fail", "caja desbordada en el limpio")
    restore()
    return bad


if __name__ == "__main__":
    sys.exit(main())
