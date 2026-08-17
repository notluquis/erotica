# NGC 6383 (aa52082-24), Submission package manifest

A&A ROUND-2 resubmission. Compiles to 24 pp clean / 28 pp marked / 0 errors / 0 undefined refs / 0 undefined citations. (Last full rebuild: 2026-08-16, co-author pass: 41 annotations from P. Cerulo applied - 24 figure-caption cuts, 11 wording fixes, the Table 1 caption moved to \tablefoot per A&A house style - plus four corrections to the Kalari 2019 comparison found while re-checking that source, including the adopted-distance systematic the previous version omitted. See referee_round2/PIERLUIGI_REVIEW_TRIAGE.md. Round-2 restructure + R1-R17 fixes are in CHANGES.md §42. latexdiff baseline = round-1 submitted version, so the marked PDF still shows only round-2 changes.)

## FILES TO SEND
(Round-1 letters archived in `letters/round1_archive/`, do NOT send.)
| File | What | Upload as |
|---|---|---|
| `aa52082-24_source.zip` | LaTeX source: aanda.tex, cites.bib, aanda.bbl, aa.cls, aa.bst, linenoaa.sty, Figures/ (21, all used) | Manuscript source (self-contained, test-compiles clean: 24 pp, 0 undefined) |
| `aanda_revised_clean.pdf` | Clean compiled manuscript | Reviewer PDF |
| `aa52082-24_marked_changes.pdf` | latexdiff vs ROUND-1 submitted version (blue=add, red=del, citations black) | Marked-changes PDF |
| `aa52082-24_cds_members.zip` | CDS catalog: ngc6383_members.dat + ReadMe (321 candidates, 254-member reference flagged) | CDS data table |
| `letters/cover_letter_round2.txt` | Round-2 cover letter (restructure + R_t + self-corrections) | Cover letter |
| `letters/response_to_referee_round2.md` | Round-2 point-by-point response (R1-R17 + old-to-new section map) | Response to referee |

## WORKING DIRS (NOT sent, kept for our records)
- `clean_source/`, master LaTeX source (6 source files + Figures/ 21 used). Edit here, then rebuild the zip.
- `marked_changes/`, latexdiff inputs (old_submitted.tex, aanda_marked.tex, aanda_revised.tex) to regenerate the marked PDF.
- `cds/`, CDS table sources.
- `gaia_quality_next_run/`, HD 159176 Gaia-quality analysis (source of the RUWE / excess-noise numbers).

## OUR RECORDS (NOT sent)
- `CHANGES.md`, full original→current changelog (10 categories).
- `AANDA_GUIDELINES_AUDIT.md`, A&A house-style audit.

## Rebuild the source zip after editing clean_source/
```
cd clean_source && pdflatex aanda && bibtex aanda && pdflatex aanda && pdflatex aanda
# zip -q -FS ../aa52082-24_source.zip aanda.tex aanda.bbl cites.bib aa.cls aa.bst linenoaa.sty Figures/*.pdf
```
