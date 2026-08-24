#!/usr/bin/env python3
"""Make the marked diff legible: additions in the body font, deletions readable.

latexdiff's markup styles all trade something away, and the trade has to be made against a
two-column A&A page rather than in the abstract.

    UNDERLINE (the default)  ulem's \\sout and \\uwave, which cannot break across lines. In a
                             narrow column a struck citation list runs off the column and
                             prints on top of its neighbour. Measured: 12 overfull boxes,
                             and the result was visibly unreadable.
    soul's \\st and \\ul      breaks lines correctly, and does not compile here at all: 28
                             errors, because the diff text is full of math and macros that
                             soul cannot take as an argument.
    CFONT                    marks by colour, so everything reflows: 0 overfull boxes. But
                             it changes two different axes -- additions get `\\sf`, a
                             different *family* from the serif body, and deletions get
                             `\\scriptsize`, which is small enough to be a struggle to read.

This script keeps CFONT's mechanism and fixes its typography:

    additions   the body font, in blue. Colour alone marks them, which is one of the two
                options the editor's letter allows ("boldface or colored text").
    deletions   red at `\\footnotesize`. Still visually subordinate, so a paragraph reads as
                the new sentence with the old text receding, but large enough to read.

The subordination is not decoration. Setting both at the same size was tried and abandoned:
with equal weight the two texts interleave into one unreadable run, and you cannot tell what
was removed without looking at the colour word by word.

Cost, measured: one extra page, and one overfull box of 20.4pt in the footnote that carries
the old GitHub URL beside the new one -- two long unbreakable URLs on one footnote line at
this size. `xurl` and `\\sloppy` were both tried and neither absorbs it.

Usage, on the output of `latexdiff --type=CFONT` (see MANIFEST.md):

    python3 set_diff_markup.py aanda_marked.tex
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

CFONT_ADD = r"\providecommand{\DIFaddtex}[1]{{\protect\color{blue} \sf #1}}"
CFONT_DEL = r"\providecommand{\DIFdeltex}[1]{{\protect\color{red} \scriptsize #1}}"

OURS_ADD = r"\providecommand{\DIFaddtex}[1]{{\protect\color[rgb]{0,0,0.7}#1}}"
OURS_DEL = r"\providecommand{\DIFdeltex}[1]{{\protect\color[rgb]{0.65,0,0}\footnotesize #1}}"

# Colour is the entire notation here: deletions are not struck through, because ulem's \sout
# cannot break across a line and ran off the column (see above). Without a key the referee has to
# infer which colour means what from the fact that one of them is smaller. The clean manuscript
# never sees this line; it is injected into the marked file only.
LEGEND_MARK = "% set_diff_markup legend"
LEGEND = (LEGEND_MARK + "\n"
          r"\noindent{\small\textit{Marked-up version.} "
          r"{\color[rgb]{0,0,0.7}Text added in this revision is shown in blue}; "
          r"{\color[rgb]{0.65,0,0}text removed is shown in red, at a smaller size}. "
          r"The comparison is against the previously submitted revision, so only the changes "
          r"made in this revision are marked.}\medskip" + "\n")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    target = Path(argv[1])
    if not target.exists():
        print(f"no existe: {target}")
        return 1

    text = target.read_text()
    done = []
    # La sustitucion CFONT -> OURS es la UNICA senal de que latexdiff acaba de correr sobre esta
    # fuente: los marcadores CFONT solo existen en su salida cruda. Se lleva en un booleano propio y
    # no se deduce de los mensajes de `done`, porque reescribir un mensaje habria cambiado en
    # silencio quien puede firmar el sello.
    latexdiff_fresco = False

    # Two independent concerns, each checked on its own. An early "already processed" exit keyed on
    # the markup alone made the legend unreachable on every file the previous version of this
    # script had produced: gate.py then failed asking for a run that provably did nothing.
    if OURS_ADD in text:
        done.append("marcado ya ajustado")
    else:
        missing = [name for name, needle in (("DIFaddtex", CFONT_ADD), ("DIFdeltex", CFONT_DEL))
                   if text.count(needle) != 1]
        if missing:
            print(f"no encontre las definiciones de CFONT para {missing}; "
                  "corre latexdiff con --type=CFONT")
            return 1
        text = text.replace(CFONT_ADD, OURS_ADD).replace(CFONT_DEL, OURS_DEL)
        latexdiff_fresco = True
        done.append("adiciones en la letra del cuerpo (azul), borrados en rojo footnotesize")

    if LEGEND_MARK in text:
        done.append("leyenda ya presente")
    elif r"\maketitle" not in text:
        print("no encuentro \\maketitle; no puedo insertar la leyenda de colores")
        return 1
    else:
        text = text.replace(r"\maketitle", "\\maketitle\n\n" + LEGEND, 1)
        done.append("leyenda de colores insertada")

    target.write_text(text)

    # Sello de procedencia: de que fuente salio este diff. gate.py lo compara con el sha actual de
    # new_revised.tex. Antes comparaba mtimes, y git no preserva mtimes: en un checkout limpio el
    # orden es arbitrario, asi que el check no podia fallar en CI ni por la razon correcta ni por
    # ninguna otra. Se escribe aca porque este script es obligatorio en la receta y corre justo
    # despues de latexdiff -- nada queda a que alguien se acuerde de un paso extra.
    # SOLO si latexdiff acaba de correr. Se escribia siempre, y re-correrlo sobre un marcado ya
    # procesado —que es justo lo que invita a hacer el mensaje de error del gate— reescribia el sello
    # al sha nuevo sin que latexdiff hubiera corrido: el diff viejo quedaba certificado como fresco y
    # el arbitro leia un diff sin el cambio que pidio.
    #
    # El primer arreglo condiciono en "hizo algo", y eso seguia certificando de mas. En el estado
    # PARCIAL —marcado ya ajustado, leyenda ausente, que es exactamente lo que dejan los dos scripts
    # que reescriben este fichero despues— insertar la leyenda contaba como trabajo y el sello se
    # reescribia igual. Insertar una leyenda es cosmetica: no dice nada sobre de que fuente salio el
    # diff. La unica evidencia de procedencia es haber convertido marcadores CFONT, que solo existen
    # en la salida cruda de latexdiff.
    src = target.parent / "new_revised.tex"
    if src.exists() and latexdiff_fresco:
        sha = hashlib.sha256(src.read_bytes()).hexdigest()
        (target.parent / "new_revised.sha256").write_text(sha + "\n")
        done.append(f"sello de new_revised.tex {sha[:12]}")
    elif src.exists():
        done.append("latexdiff no corrio en esta pasada: el sello NO se toca")

    print("; ".join(done))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
