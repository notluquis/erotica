# NGC 6383 (aa52082-24), Submission package manifest

A&A ROUND-2 resubmission. Compiles to 26 pp clean / 29 pp marked / 0 errors / 0 undefined refs / 0 undefined citations. (Last full rebuild: 2026-08-17, co-author pass: 41 annotations from P. Cerulo applied - 24 figure-caption cuts, 11 wording fixes, the Table 1 caption moved to \tablefoot per A&A house style - plus four corrections to the Kalari 2019 comparison found while re-checking that source, including the adopted-distance systematic the previous version omitted. A later point-by-point re-read of the referee report found three internal contradictions, now fixed: the Table 1 \tablefoot scoped to the five rows whose intervals are not posterior widths, the 1.2 T_max prior collision stated explicitly, and the R17 element (i) no longer claiming a converged isochrone posterior; Appendix D also gained the argument for why R_t survives the contamination that biases R_c, and a \clearpage so Fig. D.1, the load-bearing evidence for R11, now sits on the same page as its text (p. 25) instead of five pages later. Both letters were then cross-checked against the manuscript and realigned, see CHANGES.md Sect. 49; a further pass on 2026-08-17 (Sect. 70) realigned R11 and the Spitzer attribution in the response letter, and added five gate checks after finding the marked PDF, the source zip and the sent PDFs all stale or broken while the gate was green. See referee_round2/PIERLUIGI_REVIEW_TRIAGE.md. Round-2 restructure + R1-R17 fixes are in CHANGES.md §42. latexdiff baseline = round-1 submitted version, so the marked PDF still shows only round-2 changes.)

## FILES TO SEND — mapped to the NESTOR upload slots
(Round-1 letters archived in `letters/round1_archive/`, do NOT send.)

NESTOR asks for a zip **containing exactly one .tex**, plus optional files. It builds the reviewer
PDF from the zip itself, so `aanda_revised_clean.pdf` is **not uploaded** — it is our local proof.

| NESTOR slot | File | Notes |
|---|---|---|
| **Updated source files** (mandatory) | `aa52082-24_source.zip` | aanda.tex (the only .tex), aanda.bbl, cites.bib, aa.cls, aa.bst, linenoaa.sty, Figures/ (21, all used). Clean version only, per the editor's instruction. Verified to compile standalone in an empty directory: 26 pp, 0 errors, 0 undefined. |
| Latexdiff or bold pdf (optional) | `aa52082-24_marked_changes.pdf` | 29 pp. latexdiff against the round-1 submitted version, so it shows round-2 changes only. |
| Reply to the referee(s) (optional; .pdf or .txt) | `letters/response_to_referee_round2.txt` | Plain text, and now the master: the Markdown version was deleted on 2026-08-16, since NESTOR accepts only .pdf or .txt and keeping two formats had already let them drift apart once. Edit this file directly. |
| Cover letter for the Editor (optional; .pdf or .txt) | `letters/cover_letter_round2.txt` | Lists the changes, states that the author list is unchanged, and reports the editor's three pre-submission checks. |
| Datasets | `aa52082-24_cds_members.zip` | ⚠ **The dataset currently uploaded is the wrong one.** NESTOR shows `cds.zip`, 81.74 Kb, which is `_legacy/cds_superseded/cds.zip` (2026-05-18): it nests everything under a `cds/` folder, carries `__MACOSX/` resource forks, and ships two files CDS does not want. The `.dat` is byte-identical to ours, so no science differs, but its ReadMe is the older one. Replace it with `aa52082-24_cds_members.zip` (ReadMe + ngc6383_members.dat at top level, 321 rows). The cover letter now discloses the two ReadMe corrections and that a replacement archive is enclosed; it previously said the package was unchanged, which would have left this one in place. ⚠ Do **not** delete `_legacy/cds_superseded/cds.zip` while the paper is under review: `gate.py`'s `c_cds_claim` uses it as the baseline for what NESTOR actually holds, and without it nothing can check that the letter's account of the dataset is honest. It is superseded as content, not as a record. |

`aanda_revised_clean.pdf` (26 pp) and `aanda_revised_marked.pdf` are local copies for checking; only the files in the table above go to NESTOR.

## WORKING DIRS (NOT sent, kept for our records)
- `clean_source/`, master LaTeX source (6 source files + Figures/ 21 used). Edit here, then rebuild the zip.
- `marked_changes/`, latexdiff inputs (old_submitted.tex, aanda_marked.tex, aanda_revised.tex) to regenerate the marked PDF.
- `cds/`, CDS table sources.
- `gaia_quality_next_run/`, HD 159176 Gaia-quality analysis (source of the RUWE / excess-noise numbers).

## OUR RECORDS (NOT sent)
- `CHANGES.md`, full original→current changelog (10 categories).
- `AANDA_GUIDELINES_AUDIT.md`, A&A house-style audit.

## Before rebuilding anything: the consistency check
ok     aanda.tex: 0
ok     response_to_referee_round2.txt: 0
ok     cover_letter_round2.txt: 0

OK - ningun documento contradice la no-convergencia del ensemble.
Sect. 4.4 says the DEMetropolis isochrone ensemble does not converge. Nine sentences across the
manuscript and the two letters said the opposite, in three separate passes, each found only after
the previous pass declared itself complete. This screens the cheap class of that defect; its own
docstring records the class it cannot see (a sentence whose subject is "these").

## Rebuild the marked-changes PDF after editing clean_source/
```
cp clean_source/aanda.tex marked_changes/new_revised.tex
cd marked_changes
latexdiff --type=CFONT old_submitted.tex new_revised.tex > aanda_marked.tex
python3 set_diff_markup.py aanda_marked.tex        # REQUIRED, see below
python3 strip_moved_floats.py aanda_marked.tex     # REQUIRED, see below
python3 fit_marked_tables.py aanda_marked.tex      # REQUIRED, see below
pdflatex aanda_marked && bibtex aanda_marked && pdflatex aanda_marked && pdflatex aanda_marked
```

`--type=CFONT` is not optional either. latexdiff's default UNDERLINE strikes deleted text with
ulem's `\sout`, which cannot break across lines; in A&A's two columns a long struck citation list
runs off its column and prints on top of the text beside it. Measured: **12 overfull boxes with
UNDERLINE, 1 with CFONT**, and the marked PDF was visibly unreadable in places. CFONT marks by
colour, which is one of the two options the editor's letter allows ("boldface or colored text").

`set_diff_markup.py` also injects a colour key after `\maketitle` in the marked file only: colour is the entire notation in this build, since deletions are not struck through, and without a key the referee has to infer which colour means what from the fact that one is smaller. `gate.py` fails if the key is missing, because two later scripts rewrite that file.

`set_diff_markup.py` fixes CFONT's typography, which changes two different axes at once:
additions get `\sf`, a different font *family* from the serif body, and deletions get
`\scriptsize`, small enough to be a struggle to read. It rewrites them to additions in the body
font in blue and deletions in red at `\footnotesize` -- subordinate, so a paragraph reads as the
new sentence with the old receding, but legible. Setting both at the same size was tried and
abandoned: with equal weight the two texts interleave into one unreadable run. The strikeout some
readers expect is not available: it is ulem's, the very thing that overflowed, and `soul`, whose
strikeout does break lines, fails to compile against this document's math and macros.

`fit_marked_tables.py` closes the width problem. Marking a table whose every cell changed puts the old
and the new value in each cell, so Table D.2 came out 93.9pt too wide. The script compiles, reads
the log, and wraps **only** the tabulars that actually overflowed — wrapping the ones that already
fit would shrink them for nothing. The clean manuscript is never touched: those tables fit there.
Result: the clean PDF has **0** overfull boxes and the marked one has **1**, documented: the
footnote carrying the old GitHub URL beside the new one, two long unbreakable URLs on one line,
20.4pt over. `xurl` and `\sloppy` were both tried. `gate.py` enforces those exact counts, so a
second box still fails.
`strip_moved_floats.py` is not optional. latexdiff has no move detection (upstream #162), so
each float relocated by the round-2 restructure leaves a struck-through caption with no image
at its old position, which reads as "this figure was cut". The script removes such a span only
when every graphic and label inside it is alive elsewhere in the *typeset* text, so a genuine
deletion still shows. Acceptance, unchanged from CHANGES.md Sect. 47: 21 unique figures live,
0 "??" in the compiled PDF. Currently strips 10 spans, 29 pp.

## Rebuild the source zip after editing clean_source/
```
cd clean_source && pdflatex aanda && bibtex aanda && pdflatex aanda && pdflatex aanda
# zip -q -FS ../aa52082-24_source.zip aanda.tex aanda.bbl cites.bib aa.cls aa.bst linenoaa.sty Figures/*.pdf
```
