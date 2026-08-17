#!/usr/bin/env python3
"""Fail if the manuscript or either letter calls the isochrone ensemble a posterior.

Sect. 4.4 states that the DEMetropolis isochrone ensemble does not meet the convergence
criteria, so its marginals are a consistency check and not sampled posteriors. Nine
sentences elsewhere contradicted that -- in Table 1's note, in R17 of the referee letter,
in Appendix B's title, and, worst, in the caption of Fig. B.4, one paragraph below the
text that says the opposite. Captions are read before body text.

They were found by hand three times, in three passes, because each pass looked in a
narrower place than the last: the body, then the letters, then the captions and
appendices. This is that search, made repeatable.

The King-profile posteriors are deliberately NOT flagged: those samplers converge and the
word is correct for them. The trigger is an assertive phrasing (`posterior distributions`,
`sampled posterior`, `posterior width`, `credible intervals`) in the same sentence as an
isochrone-fit token, with nothing in that sentence denying it.

What it does NOT catch, measured rather than assumed. Of the three real defects this pass
removed it catches two: R17's "posterior distributions ... for the astrometric, structural,
and age parameters", and Fig. B.4's "Posterior distributions of the parameters A_V, dm, loga,
met". It misses Appendix B's "The credible intervals ... correspond to these marginal
posteriors", because that sentence names no isochrone token -- its subject is "these", and no
sentence-level regex resolves an anaphor. This is a screen, not a proof: it makes the cheap
class impossible to reintroduce silently and leaves the referential class to a human reading
the paragraph.

Its first version was worse than useless: it required the token BEFORE the word, so it passed
its own mutation test while the real R17 sentence, which puts them the other way round, sailed
through. It also let a later "ensemble" in the same sentence excuse an earlier false claim.
Both are fixed; both are why the mutation test is run against the verbatim original sentences
rather than against paraphrases.

Usage (see MANIFEST.md; run it before rebuilding the deliverables):
    python3 check_posterior_claims.py            # exit 1 on any unqualified claim
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGETS = [
    HERE / "clean_source" / "aanda.tex",
    HERE / "letters" / "response_to_referee_round2.txt",
    HERE / "letters" / "cover_letter_round2.txt",
]

# Whose "posterior" is under suspicion: the isochrone fit only. Order-agnostic -- the
# original R17 defect put "posterior distributions" BEFORE "age parameters", so a regex that
# required the token first could not see it. "age" is in the list for the same reason.
ISOCHRONE = re.compile(r"\b(isochrone|ASteCA|DEMetropolis|corner plot|loga|age)\b", re.I)

# Only ASSERTIVE phrasings. A bare "posterior median" is correct for the King fit, and the
# Table 1 caption's "1-sigma posterior uncertainty" is the convention the tablefoot scopes;
# flagging either buries the real defects in noise, which is how a check stops being read.
SUSPECT = re.compile(
    r"posterior distributions?|sampled posteriors?|posterior width|credible intervals?",
    re.I,
)

# What makes it acceptable: the sentence DENIES the claim. "ensemble" is deliberately NOT a
# qualifier -- it appears in correct and incorrect sentences alike, and letting it whitewash
# a sentence is what made the first version of this check pass its own mutation test.
QUALIFIER = re.compile(
    r"\bnot\b|\bdoes not\b|\brather than\b|\binstead of\b|\bwithout\b",
    re.I,
)


def sentences(text: str):
    return re.split(r"(?<=[.;])\s+", text.replace("\n", " "))


def offenders(text: str) -> list[str]:
    return [
        s
        for s in sentences(text)
        if ISOCHRONE.search(s) and SUSPECT.search(s) and not QUALIFIER.search(s)
    ]


def main() -> int:
    total = 0
    for path in TARGETS:
        if not path.exists():
            print(f"FALTA  {path}")
            return 2
        bad = offenders(path.read_text())
        total += len(bad)
        print(f"{'FALLA ' if bad else 'ok    '} {path.name}: {len(bad)}")
        for sentence in bad:
            print(f"    ! {' '.join(sentence.split())[:200]}")
    if total:
        print(
            f"\n{total} frase(s) llaman posterior al ensemble de isocronas sin matizar. "
            "Sect. 4.4 dice que no converge; el manuscrito y las cartas deben decir lo mismo."
        )
        return 1
    print("\nOK - ningun documento contradice la no-convergencia del ensemble.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
