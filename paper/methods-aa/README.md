# P02 — A&A methods paper (draft)

`latexmk -pdf aanda.tex`. Builds clean as of 2026-08-04: 3 pages, 0 undefined references,
0 undefined citations, 0 errors.

## Template provenance — read before submitting

| file | version | date | origin |
|---|---|---|---|
| `aa.cls` | **9.4** | 2025/11/27 | copied from the NGC 6383 submission package (`comments_paper/submission_package/clean_source/`) |
| `aa.bst` | — | — | same |
| `linenoaa.sty` | — | — | same |

⚠ **This was NOT verified against the official EDP Sciences package**, and it should be before
submission. Attempted 2026-08-04 and blocked:

- `https://www.aanda.org/doc_journal/instructions/aa-package.zip` → **HTTP 403** (Cloudflare
  challenge; the same trap that makes `pypi.org/project/<name>/` return 200 for non-existent
  projects — an automated fetch here tells you nothing about the file)
- `https://www.aanda.org/images/stories/aa/aa-package.zip` → **HTTP 403**
- `http://ftp.edpsciences.org/pub/aa/aa-package.zip` → **HTTP 503**; over HTTPS the certificate
  does not cover that hostname (`DNS:www.edpsciences.com` only)

Published release notes describe **v9.4, March 2026**, whose change is that `[longauth]` shifts
the whole author list and affiliations after the references. Our copy is v9.4 dated
**2025/11/27** — same major version, possibly a point revision behind. We do not use
`[longauth]`, so the documented change does not affect this manuscript, but that is an argument
for *why it probably does not matter*, not evidence that the file is current.

**To resolve:** download `aa-package.zip` from <https://www.aanda.org/for-authors> in a browser,
replace `aa.cls`, `aa.bst`, `linenoaa.sty`, and rebuild. Then delete this warning and record the
verified version and date in the table above.

## Provenance rule for the manuscript

Every number in `aanda.tex` carries a `%<-` comment naming the script and JSON sidecar that
produced it. A number without one is not yet evidence.

This is not bureaucracy. In this repository a test count in the JOSS paper matched no install of
the package (478 / 482 / 532 against a claimed 488), commit counts were stale the day they were
written, "docs 0 warnings" was copied from a build weeks old and was actually 2, and a
knowledge-graph size asserted in four separate files could not be re-derived under any counting
rule. Hand-copied integers drift from the builds that produced them; the comment is what makes
the drift findable.

`\TODO{...}` marks a claim whose measurement is still running — not a gap to write around.
