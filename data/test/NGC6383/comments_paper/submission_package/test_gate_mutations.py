#!/usr/bin/env python3
"""Cada check del gate, roto a propósito una vez, tiene que ponerse rojo.

`test_gate_behaviour.py` prueba lo que el gate promete de sí mismo (`--quick` no bendice, una
omisión no sale 0). Esto prueba la otra mitad, que no tenía fichero: que **cada uno de los 26
checks puede fallar** por la razón que lo justifica. Hasta 2026-09-15 esas mutaciones se habían
hecho a mano, de a una, al escribir cada check, y nada las repetía.

Existe para el refactor `TOOL-gate-generalize` (hub, `open-threads.md`): el gate parametrizado no
estuvo hecho hasta que los 26 veredictos se reprodujeron **y** estas mutaciones se pusieron rojas
contra el motor genérico. Por eso los checks se llaman a través de un adaptador -- `generic`, que
carga `tools/manuscript_gate.py` y lo configura con `gate.toml`. Hasta 2026-09-15 hubo un segundo
adaptador, `legacy`, que cargaba el `gate.py` monolítico de antes del refactor; se quitó una vez
medida la paridad de los dos (mismo veredicto, mismo detalle, en rápido y en lento) porque `gate.py`
ya es el shim que configura este mismo motor -- mantener `legacy` habría sido probar el motor contra
una copia de sí mismo con otro nombre.

Nunca toca el árbol trackeado. Copia `submission_package/`, `cds_final/`, `referee_round3/`, la base
de `_legacy/` y las dos notas de `phd-kb` a un directorio temporal, más `tools/manuscript_gate.py`
junto al resto; el `gate.py` copiado resuelve `HERE` desde su propia ubicación, así que todo lo
relativo apunta a la copia, y `KB_ROOT` se reasigna a mano porque es absoluto.

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

for _parent in HERE.parents:
    _candidate = _parent / "tools" / "manuscript_gate.py"
    if _candidate.exists():
        ENGINE_SRC = _candidate
        break
else:
    raise SystemExit(f"no encuentro tools/manuscript_gate.py subiendo desde {HERE}")


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
    sub(root / "submission_package/clean_source/aa52082-24.tex", r"(?=\\begin\{appendix\})",
        lambda _m: f"\n{text}\n\n")


def copy_over(src: Path, dst: Path) -> None:
    _remember(dst)
    dst.write_bytes(src.read_bytes())


# ---------------------------------------------------------------------------------- adaptadores

class Generic:
    """El motor generico (`tools/manuscript_gate.py`), configurado con la copia de
    `submission_package/gate.toml`. Corre el mismo codigo que produccion -- la copia de
    `make_copy` incluye el motor, no solo el paper."""

    def __init__(self, root: Path):
        engine_path = root / "tools" / "manuscript_gate.py"
        spec = importlib.util.spec_from_file_location("manuscript_gate", engine_path)
        self.mod = importlib.util.module_from_spec(spec)
        sys.modules["manuscript_gate"] = self.mod
        spec.loader.exec_module(self.mod)
        self.mod.configure(root / "submission_package" / "gate.toml")
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


ADAPTERS = {"generic": Generic}


def make_copy(dst: Path) -> Path:
    ignore = shutil.ignore_patterns("_gate_build", "__pycache__")
    for d in ("submission_package", "cds_final", "referee_round3", "_legacy/cds_round2_submitted"):
        shutil.copytree(PAPER / d, dst / d, ignore=ignore, symlinks=True)
    (dst / "tools").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ENGINE_SRC, dst / "tools" / "manuscript_gate.py")
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
TEX = f"{SP}/clean_source/aa52082-24.tex"
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
    ("c_kb", "la nota vuelve a citar R_t 40.4",
     lambda r: append(r / "kb/objects/ngc-6383.md", "\nR_t 40.4 arcmin\n")),
    ("c_kb", "la nota vuelve a citar R_c 1.95",
     lambda r: append(r / "kb/objects/ngc-6383.md", "\nR_c 1.95 arcmin\n")),
    ("c_kb", "la nota vuelve a citar t_rh ~ 0.19",
     lambda r: append(r / "kb/objects/ngc-6383.md", "\nt_rh ~ 0.19\n")),
    ("c_register", "muletilla de registro en el cuerpo",
     lambda r: insert_body(r, "It is worth noting that the cluster is young.")),
    ("c_spelling", "forma británica en el manuscrito",
     lambda r: insert_body(r, "The colour index is used.")),
    ("c_copies", "la copia en referee_round3 diverge",
     lambda r: append(r / "referee_round3/cover_letter_round3.txt", " ")),
    ("c_copies", "el ReadMe de cds_final diverge de cds/ReadMe",
     lambda r: append(r / "cds_final/ReadMe", " ")),
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
     lambda r: sub(r / "cds_final/table2.dat", r"\A[^\n]*\n", "")),
    ("c_cds", "Records del ReadMe no coincide con las filas",
     lambda r: sub(r / "cds_final/ReadMe", r"(table2\.dat\s+\d+\s+)(\d+)",
                   lambda m: m.group(1) + str(int(m.group(2)) + 1))),
    ("c_linters", "rango con guion corto",
     lambda r: insert_body(r, "Ages of 1-10 Myr are common.")),
    ("c_typos", "errata",
     lambda r: insert_body(r, "This is " + "t" + "eh cluster.")),
    ("c_linenumbers", "falta \\nolinenumbers tras \\begin{appendix}",
     lambda r: sub(r / TEX, r"(\\begin\{appendix\}[\s\S]{0,200}?)\\nolinenumbers", r"\g<1>")),
]

# Retirados 2026-09-15 junto con `marked` en gate.toml (paso 5: sin diff marcado que subir en esta
# ronda, ver el `[not_applicable]` de gate.toml para el motivo de cada uno). Un check n/a no puede
# mutarse por QUICK: not_applicable se consulta ANTES de llamar a la función real (ver `check()` en
# manuscript_gate.py), así que nunca entra a `results` y `gate.run(fn)` leería `results[-1]` de otro
# check -- SIEMPRE "ok" o basura, nunca la mutación puesta. Se excusan aquí, con motivo, en vez de
# dejar que `covered` se quede corto y la guarda de abajo los reporte como "checks sin mutación",
# que sería la razón equivocada (no faltan, son n/a).
NOT_APPLICABLE_THIS_PAPER = {
    "c_letter_numbers": "n/a: no hay carta de ronda 4 que subir",
    "c_overclaim": "n/a: idem",
    "c_dropped_symbols": "n/a: idem",
    "c_section_refs": "n/a: idem",
    "c_marked_fresh": "n/a: no hay diff marcado que subir",
    "c_strip": "n/a: idem",
    "c_cds_claim": "n/a: el CDS se envía directo al CDS, no vía NESTOR",
}

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
    ap.add_argument("--target", choices=sorted(ADAPTERS), default="generic")
    args = ap.parse_args()

    bad: list[str] = []
    omitted: list[str] = []
    with tempfile.TemporaryDirectory(prefix="gate_mut_") as td:
        root = make_copy(Path(td))
        gate = ADAPTERS[args.target](root)

        declared = {name for name in gate.mod.declared}
        fns = sorted({n for n in dir(gate.mod) if n.startswith("c_")})
        covered = {fn for fn, _, _ in QUICK} | set(SLOW_CHECKS) | set(NOT_APPLICABLE_THIS_PAPER)
        if len(fns) != len(declared) or set(fns) - covered:
            bad.append(f"checks sin mutación: {sorted(set(fns) - covered)} "
                       f"({len(fns)} funciones, {len(declared)} declarados)")
        # Todo lo que se excusó por n/a tiene que SEGUIR siendo n/a en el gate.toml real -- si algún
        # día uno de estos vuelve a aplicar (una ronda 4 con carta, por ejemplo), esta lista queda
        # mintiendo sobre por qué no se mutó, exactamente la clase de deriva que motivó agregarla.
        na_reales = {f"c_{corto}" for corto in gate.mod.NOT_APPLICABLE}
        na_no_declarados = set(NOT_APPLICABLE_THIS_PAPER) - na_reales
        if na_no_declarados:
            bad.append(f"excusados aquí pero YA NO son n/a en gate.toml: {sorted(na_no_declarados)} "
                       "-- necesitan una mutación real, no una excusa")

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
    # 27 pp desde el 2026-09-15 (pase de layout post-aceptacion: se quitaron los \clearpage
    # entre apendices y dos figuras pasaron a \sidecaption a 0.6\textwidth). Antes eran 30 pp;
    # la cifra se actualiza aca CADA VEZ que MANIFEST.md cambia la suya, o esta sonda deja de
    # casar nada (re.subn con count=0 devuelve 0 reemplazos) y sale como sonda rota, no como
    # mutacion roja.
    sub(sp / "MANIFEST.md", r"\b27 pp\b", "26 pp", count=0)
    expect("c_manifest_pages", "fail", "el MANIFEST declara otra cuenta de páginas")
    restore()

    # aanda_revised_marked.pdf ya no existe (paso 5: se quito junto con el resto del diff
    # marcado). marked_changes/aanda_marked.pdf si sigue en el arbol -- es registro, no se borra
    # -- y su contenido es real y distinto del limpio, así que sirve igual de bien para simular
    # una copia desactualizada.
    copy_over(sp / "marked_changes/aanda_marked.pdf", sp / "aa52082-24_revised_clean.pdf")
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
