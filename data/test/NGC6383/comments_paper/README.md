# comments_paper/ — NGC 6383 A&A revision workspace (aa52082-24)

Round-2 state (2026-07). Map:

- submission_package/  — THE deliverables: clean_source/ (master LaTeX),
  marked_changes/ (latexdiff vs round-1), PDFs, source zip, letters/, MANIFEST.md
  (upload checklist), CHANGES.md (full changelog).
- referee_round2/      — round-2 report, response letter, cover letter, backup.
- review_repo/         — figure factory (regen_*.py), published posteriors
  (idata_*.nc), round-1 audit notes (0*.md), provenance scripts.
- radius_robustness/   — per-radius (40/50/60/70') production samples + dill.
- clustering_audit/    — membership audit exports (feeds the COSMIC methods paper).
- cds_final/           — authoritative 321-source member catalog.
- hd159176_gaia_quality/, rauw_halpha/ — special-case analyses used in the text.
- 6383_old_paper/      — original 2024 submission (historical record).
- cluster_data.ecsv    — LIVE input: Sagitta columns for the CDS catalog
  (ngc6383_generate_cds_table.py). Do not remove despite its age.
- _legacy/             — archived pre-referee / superseded material (see its README).

Authoritative member catalog: cds_final/ngc6383_members.ecsv (321; Ref==1 = 254).
