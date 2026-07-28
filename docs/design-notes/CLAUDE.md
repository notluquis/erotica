# docs/design-notes/ — the "why this" log

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
- **A script**, for anything numeric — see `tools/validation/CLAUDE.md`. A number without a
  regenerable script is a number that will be wrong later and unfixable.
- **The falsifier.** What result would overturn this? If nothing would, it is not a finding.

## Structure conventions

- MyST admonitions carry the weight: `{danger}` for a result that would have been published and is
  false, `{warning}` for retractions, `{important}` for the load-bearing claim of a section.
- Sphinx must build with **zero warnings** — `python -m sphinx -T -b html docs docs/_build/html`.
- New notes go in `index.md`'s toctree, or they are unreachable.
- Claims that bear on novelty belong in `~/phd/software-landscape.md` as well, which is where the
  adjudication against competitors lives.

## Before writing a "nobody has done X"

Check it. Six such claims in this programme have been falsified, two after being written into a
dossier. A gap is more often an abandoned attempt than an unthought thought — search proceedings,
theses, and the negative phrasing ("surprising", "does not work", "we will investigate further").
See `~/phd/methodology.md` PART K.4.
