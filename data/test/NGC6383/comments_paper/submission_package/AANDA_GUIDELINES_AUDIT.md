> **SNAPSHOT 2026-05-19 — superseded by later builds (see CHANGES.md). Page counts/zip names below are outdated.**

# A&A guideline audit for aa52082-24

Date checked: 2026-05-19

Official sources checked:

- A&A Author's guide PDF: https://www.aanda.org/doc_journal/instructions/aadoc.pdf
- A&A language guide: https://www.aanda.org/images/stories/author/EnglishGuide-2021.pdf
- A&A tables and CDS guidance: https://www.aanda.org/for-authors/latex-issues/tables
- A&A references guidance: https://www.aanda.org/for-authors/latex-issues/references
- A&A appendices guidance: https://www.aanda.org/about-aa/65-author-information/information-files
- NESTOR front page and 2025 page-length policy: https://nestor.aanda.org/ and https://www.aanda.org/component/content/article/11-news/3046-shaping-a-sustainable-future-for-a-a-new-policy-on-paper-length

## Current package status

- Clean source upload candidate: `aa52082-24_clean_source_aanda.zip`
- Clean compiled PDF: `aanda_revised_clean.pdf`
- Marked-change optional PDF: `aa52082-24_marked_changes.pdf`
- Response letter: `letters/response_to_referee.md` and `.txt`
- Cover letter: `letters/cover_letter.md` and `.txt`
- CDS table package: `aa52082-24_cds_members.zip`

## Passes

- The clean source archive contains only the source file, bibliography, local A&A class/style files, generated `.bbl`, and figures used by `aanda.tex`.
- The marked-change file is separate from the clean source, as requested by the A&A revision letter.
- The manuscript compiles to a 17-page A4 PDF.
- The marked-change PDF also compiles to 17 pages.
- The abstract is below the A&A 300-word limit.
- The keyword list has six entries, matching the A&A maximum.
- All figure and table labels are referenced at least once in the manuscript.
- The appendix uses the `appendix` environment and appears after the bibliography.
- The CDS table is provided as a fixed-width ASCII `.dat` file with a CDS-style `ReadMe`.
- The manuscript includes a short printed excerpt of the CDS table and describes the complete column set, rather than printing the full large table.
- The LaTeX logs were checked for fatal errors, undefined references/citations, overfull/underfull boxes, natbib warnings, and hyperref warnings.

## Corrections made in this audit pass

- Standardized the manuscript and letters toward US spelling for `catalog`, matching existing use of `color`, `center`, and `Characterizing`.
- Replaced `Sec.` with A&A-preferred `Sect.` in a figure caption.
- Removed a direct imperative phrasing in a caption (`please refer`) and replaced it with a neutral statement.
- Recompiled the clean and marked PDFs and regenerated the clean source zip after these changes.

## Remaining editorial risks

- Page-length policy: NESTOR now advertises the A&A policy effective for manuscripts submitted from 2025-04-02. The page count is established at submission. This manuscript has reference `aa52082-24` and a referee decision dated 2025-01-26, so it is likely outside the new-submission trigger. If A&A nevertheless applies the policy during resubmission, the current 17-page PDF has the bibliography on page 15 and appendices starting on page 16, which is above the 12-page main-body cap for regular papers.
- Several figure captions are still long. A&A allows explanatory captions, but the guide asks captions to label and explain the figure concisely, with detailed scientific discussion in the main text. Further compression could reduce page count and improve house-style compliance.
- The large CDS table is formatted correctly for the paper/CDS split, but final CDS acceptance may still request small ReadMe wording or byte-by-byte formatting changes after their automated consistency check.
- The response letter no longer lists local intermediate paths; it refers to the prepared upload products by file name.
