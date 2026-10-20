r"""PARSEC isochrones (CMD web-interface output) as a pseudo-EEP backend.

References: Bressan et al. 2012 (``2012MNRAS.427..127B``, PARSEC v1.2S); Chen et al. 2014, 2015 and
Tang et al. 2014 (low-mass and massive tracks, as the CMD header lists them); Nguyen et al. 2022
(``2022A&A...665A.126N``, v2.0, rotating). **Which PARSEC for which cluster:** the v2.0 *isochrone*
product starts at log t = 7.0 (landscape finding §15.2), so for NGC 6383 (log t ~ 6.5) the usable
family is v1.2S; v2.0 is for clusters of >= 10 Myr.

**Reading.** CMD writes one table with ``Zini MH logAge Mini ... logL logTe ... label <bands>``;
rows are grouped by ``(MH, logAge)``. Rows with a magnitude >= 50 (CMD's ``99.999`` padding, which
ASteCA also drops) and phases >= 9 (post-AGB, "in preparation" per the CMD FAQ, which ASteCA drops
by default) are removed. The EEP coordinate is built by :mod:`.pseudo_eep` after replacing the label
by its running maximum along the isochrone (the raw label is not monotone in 65 % of the
isochrones of the local CMD 3.7 file).

**Metallicity.** The fit coordinate is CMD's ``MH`` column. Measured on the local file
(2026-10-04, ``tools/validation/isochrone_grids/feh_conventions.py``): ``MH = log10(Z/X) -
log10(0.0207)`` with ``Y = 0.2485 + 1.78 Z`` reproduces all six Z of the file to < 1e-4 dex,
i.e. PARSEC's (Z/X)_sun = 0.0207 (measured). Z_sun = 0.0152 and the Caffau et al. 2011 mixture
are from memory of Bressan+12 [I], not read in this session.

**Passbands.** Whatever the CMD request asked for; the local files are "Gaia EDR3 (all Vegamags,
Gaia passbands from ESA/Gaia website)" with ``Gmag G_BPmag G_RPmag`` and OBC bolometric corrections
(Marigo+08, Girardi+08) at A_V = 0. CMD 3.8 with YBC (Chen+19) can use star-by-star extinction
coefficients, which YBC says are needed for Gaia at A_V >~ 0.5; these files do not.

**Licence.** No data licence declared ("All rights reserved" footer of the PARSEC pages): files are
fetched on demand (``ezpadova``, MIT, as the CMD client) and never redistributed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .base import GridNode, GridProvenance, IsochroneGrid
from .pseudo_eep import effective_phase, resample

#: EEP blocks per effective phase: PMS and MS get the resolution, later phases less
PARSEC_BLOCKS = {
    0: (0, 200),
    1: (200, 200),
    2: (400, 40),
    3: (440, 60),
    4: (500, 40),
    5: (540, 40),
    6: (580, 40),
    7: (620, 40),
    8: (660, 40),
}


class PARSECGrid(IsochroneGrid):
    """PARSEC (CMD output) with pseudo-EEPs.

    Parameters
    ----------
    path : str or Path
        One CMD output file, or a directory of them.
    bands : tuple of str
        Magnitude columns. Default: Gaia EDR3 ``Gmag G_BPmag G_RPmag``.
    scheme : {"arclength", "index", "massquantile"}
        Pseudo-EEP construction (see :mod:`.pseudo_eep`). ``"arclength"`` is the default; the other
        two exist to be measured against it.
    loga_range : tuple, optional
    max_mass : float, optional
        Drop rows above this initial mass (the >= 20 Msun branch of v1.2S comes from a different
        set of tracks and has implausible Gaia magnitudes at its top, G ~ -16 to -20).
    """

    eep_kind = "pseudo"
    default_bands = ("Gmag", "G_BPmag", "G_RPmag")
    default_effl = (6390.7, 5182.6, 7825.1)

    def __init__(
        self,
        path: str | Path,
        bands: tuple[str, ...] | None = None,
        *,
        scheme: str = "arclength",
        loga_range: tuple[float, float] | None = None,
        max_mass: float | None = None,
        n_massquantile: int = 400,
    ) -> None:
        super().__init__()
        import pandas as pd

        self.bands = tuple(bands) if bands is not None else self.default_bands
        self.scheme = scheme
        p = Path(path)
        files = [p] if p.is_file() else sorted(f for f in p.iterdir() if f.is_file())
        frames, release, hashes = [], set(), []
        for fp in files:
            names = None
            with open(fp, encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if not line.startswith("#"):
                        continue
                    if "PARSEC release" in line or "PARSEC version" in line:
                        release.add(line.split("PARSEC")[-1].replace("release", "").strip())
                    if line.lstrip("#").split()[:2] == ["Zini", "MH"]:
                        names = line.lstrip("#").split()
            if names is None:
                continue
            frames.append(
                pd.read_csv(
                    fp,
                    sep=r"\s+",
                    comment="#",
                    header=None,
                    names=names,
                    float_precision="round_trip",
                )
            )
            hashes.append(f"{fp.name}:{hashlib.sha256(fp.read_bytes()).hexdigest()}")
        if not frames:
            raise FileNotFoundError(f"no CMD/PARSEC table in {p}")
        d = pd.concat(frames, ignore_index=True)
        missing = [b for b in self.bands if b not in d]
        if missing:
            raise ValueError(f"PARSEC table lacks bands {missing}")
        d = d[d["label"] < 9]
        for b in self.bands:
            d = d[d[b] < 50.0]
        if max_mass is not None:
            d = d[d["Mini"] <= max_mass]
        if loga_range is not None:
            d = d[(d["logAge"] >= loga_range[0] - 1e-9) & (d["logAge"] <= loga_range[1] + 1e-9)]
        # global metric weights: equal contribution of log Teff and log L (Dotter 2016 §II.3)
        w = (1.0 / np.ptp(d["logTe"]) ** 2, 1.0 / np.ptp(d["logL"]) ** 2)
        self.metric_weights = w
        for (mh, loga), s in d.groupby(["MH", "logAge"]):
            s = s.sort_values("Mini", kind="stable")
            cols = {
                "mass": s["Mini"].to_numpy(float),
                "logte": s["logTe"].to_numpy(float),
                "logl": s["logL"].to_numpy(float),
            }
            cols.update({b: s[b].to_numpy(float) for b in self.bands})
            ph = effective_phase(s["label"].to_numpy(float))
            eep, r = resample(
                cols,
                scheme=scheme,
                phase=ph,
                phase_blocks=PARSEC_BLOCKS,
                weights=w,
                n_total=n_massquantile,
            )
            node = GridNode(
                eep,
                r["mass"],
                {b: r[b] for b in self.bands},
                {"logte": r["logte"], "logl": r["logl"]},
            )
            self._add(round(float(mh), 5), round(float(loga), 4), node, float(s["Zini"].iloc[0]))
        self.version = ",".join(sorted(release)) or "?"
        self.name = f"PARSEC {self.version} ({scheme})"
        self.provenance = GridProvenance(
            family="PARSEC",
            version=self.version,
            references=("2012MNRAS.427..127B",),
            source="http://stev.oapd.inaf.it/cgi-bin/cmd (CMD 3.7/3.8)",
            license="no data licence declared ('All rights reserved'); not redistributed",
            metallicity="[M/H] = log10(Z/X) - log10(0.0207), Y = 0.2485 + 1.78 Z (measured)",
            passbands=", ".join(self.bands),
            sha256=tuple(hashes),
            notes=f"pseudo-EEP scheme {scheme}; labels by running maximum",
        )
