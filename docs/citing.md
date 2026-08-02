# Citing EROTICA and the methods it uses

Astropy's affiliated-package criteria define "Good" documentation as including *"a preferred method of
citation of this package and citations to relevant papers and software included where appropriate"*.
This page is that, and it is deliberately weighted toward the second half: **most of what EROTICA does
is other people's method, implemented.**

## The package

Machine-readable metadata lives in [`CITATION.cff`](https://github.com/notluquis/erotica/blob/dev/CITATION.cff),
which GitHub renders as a *Cite this repository* button.

```{warning}
**Pre-release.** The author list, ORCIDs and Zenodo DOI in `CITATION.cff` are not final. A software
paper is in preparation; once it has a DOI the `preferred-citation` field will redirect the citation
button to it, which is the correct target for a methods paper with a companion code release.
```

```{note}
**Every bibcode on this page was verified against SciX/ADS on 2026-08-02**, not transcribed from
memory. That is not pedantry: this project has already mis-attributed the Gaia parallax zero-point
floor once (it is Maíz Apellániz, Pantaleoni González & Barbá 2021, not Vasiliev & Baumgardt), and a
wrong bibcode propagates silently into every paper that copies it.
```

## The methods — cite these, they are not ours

### Radial structure

| method | cite | note |
|---|---|---|
| King profile | King (1962), AJ 67, 471 — `1962AJ.....67..471K` | **Empirical**, in his own words *"merely a convenient fitting formula"*. The additive background is **not** in his Eq. (14); see {func}`~erotica.analysis.structure.king_profile`. |
| King dynamical model | King (1966), AJ 71, 64 — `1966AJ.....71...64K` | A different object from the 1962 law. There is no "King 1964". |
| EFF profile | Elson, Fall & Freeman (1987), ApJ 323, 54 — `1987ApJ...323...54E` | Convention trap: `γ_MvdM = γ_EFF + 1` against McLaughlin & van der Marel (2005). Their abstract is also the earliest statement of the corona problem: *"At least some and perhaps all the clusters in the sample extend beyond their eventual tidal radii, with **up to 50% of the total masses in unbound halos**."* |
| Corona component | Danilov & Putkov (2012), Astron. Rep. 56, 609; Seleznev (2016), MNRAS 456, 3757 | Seleznev states the mechanism the corona term exists to fix. |
| Model comparison scale | Kass & Raftery (1995), JASA 90, 773 | Applied to `2 ln B`. |

### Membership, selection and calibration

| method | cite |
|---|---|
| HDBSCAN | Campello, Moulavi & Sander (2013); McInnes, Healy & Astels (2017), JOSS 2, 205 |
| Gaia DR3 selection function | Cantat-Gaudin et al. (2023), A&A 669, A55 — `2023A&A...669A..55C`, via [`gaiaunlimited`](https://github.com/gaia-unlimited/gaiaunlimited) |
| Subsample selection functions | Castro-Ginard et al. (2023), A&A 677, A37 — `2023A&A...677A..37C` |
| Census cross-match, masses, Jacobi radii | Hunt & Reffert (2024), A&A 686, A42 — `2024A&A...686A..42H` |
| Parallax zero point | Lindegren et al. (2021); the 10.3 µas floor is Maíz Apellániz, Pantaleoni González & Barbá (2021), A&A 649, A13 — `2021A&A...649A..13M` |
| Distances | Bailer-Jones et al. (2021), AJ 161, 147 |

### Statistics and validation

| method | cite |
|---|---|
| Power-scaling prior sensitivity | Kallioinen, Paananen, Bürkner & Vehtari (2024), Stat. Comput. 34, 57 — `arXiv:2107.14054` |
| Weakly informative scale priors | Gelman (2006); Polson & Scott (2012); in-field precedent Olivares et al. (2018), A&A 612, A70 |
| Recoverability criteria | Muñoz, Padmanabhan & Geha (2012), ApJ 745, 127 — `2012ApJ...745..127M` |
| Unbounded intervals under weak identification | Dufour (1997), Econometrica 65, 1365 |
| Synthetic cluster structure | Goodwin & Whitworth (2004), A&A 413, 929 — `2004A&A...413..929G`; McLuster, Küpper et al. (2011), MNRAS 417, 2300 |
| Substructure statistic | Cartwright & Whitworth (2004), MNRAS 348, 589 — with the caution of Daffern-Powell & Parker (2020) |
| Mass segregation `Λ_MSR` | Allison et al. (2009), MNRAS 395, 1449 |
| Sampler | Hoffman & Gelman (2014) NUTS, via PyMC — Abril-Pla et al. (2023), PeerJ CS 9, e1516 |
| Diagnostics | ArviZ — Kumar et al. (2019), JOSS 4, 1143 |

## Software this depends on

`numpy`, `scipy`, `astropy`, `pandas`, `scikit-learn`, `hdbscan`, `pymc`, `pytensor`, `arviz`,
`astroquery`, `gaiaunlimited`. Astropy asks that dependencies be cited where appropriate: at minimum
cite Astropy (Astropy Collaboration 2013, 2018, 2022) and PyMC if you fit anything.

## If you use a documented limitation

Several results in {doc}`design-notes/index` are measurements in their own right — the
prior-determined tidal radius, the recoverability boundary for the EFF slope, the background term
absorbing corona rather than contamination. If you rely on one, cite the package and point at the
design note; each names the script under `tools/validation/` that produced it.
