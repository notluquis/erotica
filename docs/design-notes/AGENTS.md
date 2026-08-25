# docs/design-notes/ — the "why this" log

> **Contexto raíz:** bajo la convención `AGENTS.md` **gana el fichero más cercano y no se
> concatena** — al contrario que `CLAUDE.md`, que los concatena todos. Si tu agente sólo lee
> éste, lee también el `AGENTS.md` de la raíz del repo. Y `~/phd` es el **hub**, un repo
> aparte que puede no estar presente: lo que dependa de él va marcado.

These notes are the reasoning behind the methods, and they feed the papers directly. They are more
detailed and more provisional than the user guide, and they are **append-only about mistakes**.

## The rule that makes them worth anything

**Record what was wrong and why, not only the fix.** A struck-through row in a results table is worth
more than the row that replaced it, because it is the only thing that stops the same design being
tried again. Examples already here that earn their space: the fractal coverage row that measured
nothing and *why* the construction hid the effect; the three retractions at the top of
`king_model_validity.md`; the resolution artefact that under-read a selection function 4.5×.

Never silently correct a number that has been quoted. Mark the old value, the new value, and the
cause.

## What a claim in here must carry

- **A citation with a bibcode**, verified on ADS/SciX — not a remembered attribution. Two have been
  wrong: a µas floor credited to the wrong paper, and King's `+ b` attributed to King.
- **A distinction between verified and inferred**, stated in the text. "I read this in the source"
  and "this follows from what they say" are different claims.
- **A script**, for anything numeric — see `tools/validation/AGENTS.md`. A number without a
  regenerable script is a number that will be wrong later and unfixable.
- **The falsifier.** What result would overturn this? If nothing would, it is not a finding.

## Structure conventions

- MyST admonitions carry the weight: `{danger}` for a result that would have been published and is
  false, `{warning}` for retractions, `{important}` for the load-bearing claim of a section.
- Sphinx must build with **zero warnings** — `python -m sphinx -T -b html docs docs/_build/html`.
- New notes go in `index.md`'s toctree, or they are unreachable.
- Claims that bear on novelty belong in `~/phd/software-landscape.md` as well, which is where the
  adjudication against competitors lives.

## Where each kind of content belongs (Diátaxis)

The documentation follows [Diátaxis](https://diataxis.fr), and the quadrant boundaries are quoted
because they are easy to blur:

| quadrant | orientation | what it is |
|---|---|---|
| Tutorials | learning | *"an experience that takes place under the guidance of a tutor … its purpose is not to help the user get something done, but to help them learn."* |
| How-to guides | goals | *"directions that guide the reader through a problem or towards a result."* |
| Reference | information | *"technical descriptions of the machinery and how to operate it … It should be austere. One hardly reads reference material; one consults it."* |
| **Explanation** | understanding | *"a discursive treatment of a subject, that permits reflection."* |

**Design notes are Explanation, and Diátaxis says so explicitly**: *"Provide background and context in
your explanation: explain why things are so — design decisions, historical reasons, technical
constraints."* That is exactly this directory. Keep the *why* here and out of the API reference, which
must stay consultable rather than readable.

The corollary that matters: **a measured limitation is Explanation, not a caveat buried in Reference.**
"The tidal radius is prior-determined for this footprint" belongs in a design note with the evidence,
and is *summarised* in the docstring with a pointer — not the other way round.

## Before writing a "nobody has done X"

Check it. Six such claims in this programme have been falsified, two after being written into a
dossier. A gap is more often an abandoned attempt than an unthought thought — search proceedings,
theses, and the negative phrasing ("surprising", "does not work", "we will investigate further").
See `~/phd/methodology.md` PART K.4.
