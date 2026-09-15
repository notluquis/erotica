# NGC 6383 (aa52082-24), Submission package manifest

A&A ROUND-3 resubmission (minor revision, decision 2026-09; **paper accepted 2026-09-15**, this
pass is the post-acceptance layout cleanup the editorial office asked for by email the same day).
Compiles to 26 pp clean / 0 errors / 0 undefined refs / 0 undefined citations. The
clean count dropped from 30 pages to 27 in the post-acceptance pass: the three explicit `\clearpage`
commands between appendices were removed, Figs. B.4 and C.3 moved to `\sidecaption` at
`0.6\textwidth`, and Fig. C.4 was regenerated as a 1x3 row instead of a 3x1 column. A single
targeted `\FloatBarrier` (package `placeins`) was kept before Appendix D only -- without it, Fig.
D.1 drifted five pages from the text that discusses it, undoing the exact R11 fix this same file
documented on 2026-08-17 below; measured against the appendix pages rendered to PNG, not assumed.

**2026-09-15, second compaction pass: 27 pages to 26.** The editor's "reduce figures when needed and
remove the empty spaces" was only partly met by the first pass above -- pp. 21-25 still had between
~30% and ~75% of their height blank, measured with a per-page, per-column blank-row script
(`whitespace.py`) rendering the PDF at 40 dpi, not eyeballed. Cause, confirmed by testing each lever
in isolation and re-measuring: two-column LaTeX cannot backfill a column once text has advanced past
it, so a short single-column figure followed by a `figure*` always strands the other column, and a
`figure*` (needing a fresh page top) can never join a page whose earlier float already consumed it.
Fig. C.5 (`cumulative_by_mass_and_type_mseg.pdf`) was narrowed from `\hsize` to `0.7\textwidth`,
just enough for it to join Figs. C.3-C.4 on one page (23: 32% blank to 10% blank), eliminating a
whole page. Fig. C.1 (`ngc6383_mass_binary.pdf`) was regenerated left/right instead of stacked
upper/lower (`review_repo/regen_massbinary_side_by_side.py`, `conda run -n cosmic`, rasterized
scatter markers, same ASteCA/masses data, no pipeline re-run) so it could run as a `figure*` at
`0.9\textwidth` instead of a single column at full height; a size-only fix (shrinking the original
stacked figure to fit a leftover column) was tried and rejected because it drove the embedded tick
labels below 3 pt at print size -- illegible. Captions for both changed "Upper/Lower panel" to
"Left/Right panel" to match. Tried and reverted because they cost pages or gained nothing measured:
`stfloats` for `figure*[b]` (re-flowed the *main text*, pp. 1-18, the instant it was loaded in the
preamble -- unshippable, confirmed by diffing the caption-to-page map before/after); removing the
pre-Appendix-D `\FloatBarrier` (dumped every Appendix C figure after the Appendix D title, breaking
the same ordering constraint this file's first pass fixed); shrinking Figs. B.4, C.2, C.4, C.6 below
their first-pass or original size (each is the sole content of its page or column, so shrinking it
only grows the blank tail below it -- measured making pp. 21/22/24 worse before being reverted).
Remaining blank pages (21: 40%, 22: 34%, 24: 64%, both isolated single floats; 25: 34%, Appendix D
text + Fig. D.1) are the structural floor of this mechanism, not unexamined: a short single-column
float can never be followed, in the same twocolumn pass, by the very next `figure*` in source order.

**2026-09-15, third pass: 26 to 25 pages.** The "structural floor" claim just above held for figures
that must precede their *own* appendix's title, but Fig. C.6 only has to precede Fig. C.6's own
appendix (C); nothing requires it to precede Appendix D's title too, as long as it does not drift
onto a page *after* D's title or past it in reading order on the same page. The single
`\FloatBarrier` kept before `\section{Appendix D}` was moved back six lines, to right after Fig.
C.5's `\end{figure*}` and before Fig. C.6's `\begin{figure}` -- so it still stops every earlier
Appendix C float from drifting past a title, but Fig. C.6, declared after the barrier, is free to
float forward. Result: Fig. C.6 lands at the top of the left column of the first Appendix D page,
Appendix D's title and both intro paragraphs follow in the same column below it, and Fig. D.1 lands
at the top of the right column of that same page -- same page as the paragraph that names it
("Fig.~D.1 shows that..."), same as required. p24 (Fig. C.6 alone) is gone; the page it shared with
Appendix D went from 64% blank to 6%. Checked, not assumed: rendered PNG read confirms C.6 sits
above the Appendix D title in reading order on that page; the caption-to-page map confirms pp. 1-18
did not move; the build log has 0 errors, 0 `Overfull \hbox`, 0 "No sufficient room for the legend".
This is a refinement of the previous pass's `\FloatBarrier` experiment, not a contradiction of it:
that pass tried *removing* the barrier outright, which let every Appendix C figure (not just C.6)
drift past the Appendix D title -- moving it, keeping exactly one float on the far side, does not.
Also tried this pass, measured, and reverted: Fig. C.2 (`luminosity_function.pdf`) at `0.8\hsize` to
see whether the saved height let Fig. C.3 join p22 -- it did not (C.3 stayed on p23), and p22 (the
only page this could have helped) went from 34% blank to 40%, so the change bought nothing and was
reverted. Remaining blank pages (21: 40%, Fig. B.4 + Appendix C's title and intro, isolated for the
same column-backfill reason as before; 22: 34%, Figs. C.1+C.2) are the same structural floor as the
second pass, now narrower by one page.

**2026-09-15, fourth pass (independent review): 25 to 26 pp.** An outside review of the accepted,
already-compacted manuscript found Table A.1 (`tab:literature`) landing on p. 20, in reading order
*after* the title of Appendix B on p. 19 -- the round-3 cover letter's own promise ("Each appendix
figure and table now appears within its own appendix") broken by the third pass above, which had
removed every `\clearpage` between appendices without re-checking what a `table*` too tall for
Appendix A's one short paragraph would do next. Three fixes were tried on a working copy, measured
with `whitespace.py` and checked against the rendered PNGs of pp. 19-21, not assumed from the source
order: (i) moving the `table*` block to right before `\section{Historical parameters}` cost nothing
in pages (still 25) but the appendix-letter counter had not yet incremented at that point in the
source, so the table's own number came out "Table .1" -- a new defect, not a fix, discarded on sight
of the rendered page; (ii) a `\FloatBarrier` right before `\section{HDBSCAN diagnostic...}` forces
the table to p. 20 (p. 19 left 85.6% blank) but lets Appendix B's title and text share p. 20 with it
(11.6% + 23.8% blank); (iii) a `\clearpage` at the same point also reaches p. 20 but strands the
table alone there before Appendix B starts fresh on p. 21 (64.1% + 3.8% blank) -- same page count as
(ii), more blank. (ii) was adopted: one page more than the third pass, but the only one of the three
that both satisfies the round-3 promise and does not spend more blank than the alternative at the
same page count. The same pass also regenerated Fig. B.4 (`plot_pair_trace.pdf`) to prune the tick
labels of its bottom row and first column (`MaxNLocator(nbins=3, prune='both')`, only on the axes
that carry a label), fixing adjacent-panel tick labels that had been rendering fused (e.g. "10.500");
same trace data, `rasterized=True` and `dpi=300` unchanged, checked at 200 dpi. Text-only fixes from
the same review (broken CDS URL scheme, a PyPI package name, a scope-narrowed methods sentence, an
abbreviation-first-use cleanup, a table-footnote off-by-one between "rows" and "quantities", two
gendered pronouns rewritten, a Spitzer, Jr. BibTeX name-field order fix, and the minor items listed
in this repository's commit history for aa52082-24) did not move the page count on their own.

**No more marked diff, no more referee response, this round.** The paper is accepted; the editor
asked for a clean version only, and there is no referee to answer. `gate.toml`'s `marked` key and
the seven checks that depended on a live upload of the diff or the letters (letter numbers,
overclaim, dropped symbols, section refs, marked-fresh, strip, the CDS-change claim) were retired
to `not_applicable` 2026-09-15, each with its own reason. `marked_changes/` and `letters/` are
**not** deleted -- they stay as the record of what round 3 actually said -- but the two tracked
deliverable copies that only existed to be uploaded, `aa52082-24_marked_changes.pdf` and
`aanda_revised_marked.pdf`, were removed with `git rm`: nothing in `gate.toml` compares against
them any more, and an untracked-from-the-check binary is exactly the kind of drift this file exists
to prevent, not something to leave sitting in the tree "just in case". `aa52082-24_revised_clean.pdf`
stays; it is still the local proof copy for the one document that is still uploaded.
(Last full rebuild: 2026-08-17, co-author pass: 41 annotations from P. Cerulo applied - 24 figure-caption cuts, 11 wording fixes, the Table 1 caption moved to \tablefoot per A&A house style - plus four corrections to the Kalari 2019 comparison found while re-checking that source, including the adopted-distance systematic the previous version omitted. A later point-by-point re-read of the referee report found three internal contradictions, now fixed: the Table 1 \tablefoot scoped to the five rows whose intervals are not posterior widths, the 1.2 T_max prior collision stated explicitly, and the R17 element (i) no longer claiming a converged isochrone posterior; Appendix D also gained the argument for why R_t survives the contamination that biases R_c, and a \clearpage so Fig. D.1, the load-bearing evidence for R11, now sits on the same page as its text (p. 25) instead of five pages later. Both letters were then cross-checked against the manuscript and realigned, see CHANGES.md Sect. 49; a further pass on 2026-08-17 (Sect. 70) realigned R11 and the Spitzer attribution in the response letter, and added five gate checks after finding the marked PDF, the source zip and the sent PDFs all stale or broken while the gate was green. See referee_round2/PIERLUIGI_REVIEW_TRIAGE.md. Round-2 restructure + R1-R17 fixes are in CHANGES.md §42. latexdiff baseline = round-1 submitted version, so the marked PDF still shows only round-2 changes.)

## FILES TO SEND — mapped to the NESTOR upload slots
(Round-1 letters archived in `letters/round1_archive/`, do NOT send.)

NESTOR asks for a zip **containing exactly one .tex**, plus optional files. It builds the reviewer
PDF from the zip itself, so `aa52082-24_revised_clean.pdf` is **not uploaded** — it is our local proof.

| NESTOR slot | File | Notes |
|---|---|---|
| **Updated source files** (mandatory) | `aa52082-24_source.zip` | aa52082-24.tex (the only .tex), aa52082-24.bbl, cites.bib, aa.cls, aa.bst, linenoaa.sty, Figures/ (21, all used). Clean version only, per the editor's instruction. Verified to compile standalone in an empty directory: 26 pp, 0 errors, 0 undefined. |

**CDS deposit is no longer a NESTOR slot.** Until 2026-09-15 this table carried a "Datasets" row
for `aa52082-24_cds_members.zip`, uploaded to NESTOR so the journal would forward it to the CDS.
Kept here as the record of why that replacement was needed, not deleted: NESTOR had shown `cds.zip`
(`_legacy/cds_superseded/cds.zip`, 2026-05-18), which nested everything under a `cds/` folder,
carried `__MACOSX/` resource forks, and shipped two files CDS does not want; the `.dat` was
byte-identical to ours, so no science differed, but its ReadMe was the older one. The cover letter
for that round disclosed the two ReadMe corrections and the replacement archive. ⚠ Do **not** delete
`_legacy/cds_superseded/cds.zip` or `_legacy/cds_round2_submitted/cds.zip`: `gate.toml`'s
`cds_claim.submitted_zip` still names the round-2 one as a record even though `c_cds_claim` itself is
`not_applicable` now (see the note near the top of this file) -- the file staying missing would make
that reference dangle, not the check pass differently. As of this round the CDS table is **sent
directly to the CDS**, not via NESTOR: the ready-to-upload archive is `aa52082-24_cds_table2.zip`
(`cds/ReadMe` + `cds/table2.dat`, flat, no `cds/` folder, no resource forks -- same lesson as above,
applied to the new name). It is not tracked in git, for the same reason `aa52082-24_source.zip`
isn't: both are regenerable from tracked sources (`cds/ReadMe` + `cds/table2.dat` here, `clean_source/`
for the .tex zip) rather than records of what a third party already received.

`aa52082-24_revised_clean.pdf` (26 pp) is a local copy for checking; only the files in the table above go to NESTOR. ⚠ The archive to upload is `aa52082-24_source.zip`. A byte-identical duplicate named `clean_source.zip` used to sit beside it, referenced by nothing and documented nowhere; it was deleted on 2026-08-17, because two archives with the same contents and different names is how the stale one gets uploaded the day only one of them is rebuilt.

## WORKING DIRS (NOT sent, kept for our records)
- `clean_source/`, master LaTeX source (6 source files + Figures/ 21 used). Edit here, then rebuild the zip.
- `marked_changes/`, latexdiff inputs (old_submitted.tex, aanda_marked.tex, aanda_revised.tex) to regenerate the marked PDF.
- `cds/`, CDS table sources.
- `gaia_quality_next_run/`, HD 159176 Gaia-quality analysis (source of the RUWE / excess-noise numbers).

## OUR RECORDS (NOT sent)
- `CHANGES.md`, full original→current changelog (10 categories).
- `AANDA_GUIDELINES_AUDIT.md`, A&A house-style audit.

## Before rebuilding anything: the consistency check
ok     aa52082-24.tex: 0
ok     response_to_referee_round2.txt: 0
ok     cover_letter_round2.txt: 0

OK - ningun documento contradice la no-convergencia del ensemble.
Sect. 4.4 says the DEMetropolis isochrone ensemble does not converge. Nine sentences across the
manuscript and the two letters said the opposite, in three separate passes, each found only after
the previous pass declared itself complete. This screens the cheap class of that defect; its own
docstring records the class it cannot see (a sentence whose subject is "these").

## Rebuild the marked-changes PDF after editing clean_source/ (HISTORICAL -- retired 2026-09-15)
Kept as a record of how the marked PDF was built for rounds 2-3, not as an active step: the paper
is accepted, the editor asked for a clean version only, and no check in `gate.toml` reconstructs or
compares this document any more (`marked` was removed from the config; see the note near the top of
this file). Do not run this recipe as part of the current workflow.
```
cp clean_source/aa52082-24.tex marked_changes/new_revised.tex
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
0 "??" in the compiled PDF. Round 3 (diff against the round-2 version): strips 0 spans, 31 pages (historical fact, not a live page count -- see the HISTORICAL note above).

## Rebuild the source zip after editing clean_source/
```
cd clean_source && pdflatex aa52082-24 && bibtex aa52082-24 && pdflatex aa52082-24 && pdflatex aa52082-24
# zip -q -FS ../aa52082-24_source.zip aa52082-24.tex aa52082-24.bbl cites.bib aa.cls aa.bst linenoaa.sty Figures/*.pdf
```
