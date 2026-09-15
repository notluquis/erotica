#!/usr/bin/env python3
"""Checks specific to NGC 6383 (aa52082-24), loaded by `manuscript_gate.configure()` via the
`local` key in `gate.toml`.

These four read `mg.TEX`/`mg.CDS_DAT` inside the function body, never at import time (never
`from manuscript_gate import TEX`), because the rebind of KB roots and corpus paths that the
mutation-test harness performs on a copied tree needs the CURRENT values, read at call time --
importing the name once would freeze it at whatever it was when this module first loaded.
"""

from __future__ import annotations

import math
import re

import manuscript_gate as mg


@mg.check("Table 1 es internamente consistente")
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
    tex = mg.TEX.read_text()

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


@mg.check("el texto y Table A.1 atribuyen el mismo valor a la misma referencia")
def c_literature_agreement():
    """Table A.1 compiles the literature; the running text quotes it. They must not disagree.

    The introduction gave Angelo et al. (2018) a distance of 0.840 kpc where Table A.1's row for
    the same paper says 0.83 +/- 0.16 -- and the Table A.1 caption argument, and the response
    letter, both use 0.83. One quantity, one source, two roundings, three places. This is the R7
    defect with a citation attached, which is what makes it checkable where the general
    intra-manuscript version was not (see manuscript_gate.py's NOT-a-check comment above
    c_paraphrase): the citation key says *which* two numbers are supposed to be the same, so
    there is no guessing and no noise.
    """
    tex = mg.TEX.read_text()
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


@mg.check("el pie de Table A.1 declara el rango real de su propia columna")
def c_literature_span():
    """A caption that summarises its own table has to be recomputed from it, not remembered.

    Table A.1's caption states the span of adopted distances to argue the literature ages are not
    on a common scale. It said 0.83--1.70 kpc while the table also lists 2.13 and 0.76 -- correct
    for the fifteen rows that report an age, which is the set the argument is about, and wrong
    about "this table". The argument was sound and the sentence was not, which is the only kind of
    error a caption can make about the rows printed directly beneath it.
    """
    tex = mg.TEX.read_text()
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


@mg.check("las cifras del manuscrito se rederivan del catalogo entregado")
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
    if not mg.CDS_DAT.exists():
        return False, "no existe el .dat del CDS"
    cols = {"Plx": (48, 55), "e_Plx": (57, 64), "pmRA": (66, 73), "pmDE": (84, 91),
            "Gmag": (102, 109), "pMember": (180, 186), "PMSProb": (190, 196),
            "Jmag": (129, 136), "Hmag": (138, 145), "Ksmag": (147, 154)}

    def val(line, a, b):
        s = line[a - 1:b].strip()
        return None if s in ("", "...") else float(s)

    rows = []
    for line in mg.CDS_DAT.read_text().splitlines():
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
    sentence = re.search(r"NGC 6383 has [^.]*candidate members[^.]*\.", mg.TEX.read_text())
    if sentence is None:
        return False, "no encuentro la frase que declara los cuatro umbrales de membresia"
    stated = [int(x) for x in re.findall(r"\$(\d+)\$", sentence.group(0))][:4]
    if stated != counts:
        return False, f"umbrales: el texto dice {stated}, el catalogo da {counts}"

    sub = [r for r in ref if r["e_Plx"] / abs(r["Plx"]) < 0.1]
    tex = mg.TEX.read_text()
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
