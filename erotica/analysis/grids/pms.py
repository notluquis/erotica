r"""Pre-main-sequence-only grids, for use as **controls** of the main fit: BHAC15 and SPOTS.

Both are tabulated on (almost) the same initial masses at every age, so the EEP coordinate is the
index of the mass in the grid's global mass list (``eep_kind = "mass"``): interpolating between ages
at fixed index is interpolating along that mass's track, valid while no phase boundary is crossed
between the two ages -- true inside their PMS/early-MS window.

**Limits that a fit with these grids inherits, and must declare:**

=========  ==========================  ==========  ===============================================
grid       mass                        [Fe/H]      Gaia
=========  ==========================  ==========  ===============================================
BHAC15     0.01-1.4 Msun               0 only      ``G_RSV G G_BP G_RP``; release not declared in
                                                   the file (dated 2020-08-10, i.e. before EDR3
                                                   passbands were public: DR2-era [I])
SPOTS      0.10-1.30 Msun, step 0.05   0 only      Gaia **DR2** G, BP, RP from empirical MS colour
                                                   tables (Somers+20 §2.3), -99 outside their
                                                   calibrated range: at f = 0.34 and log t 6-7 only
                                                   M >= 0.55-0.65 Msun have Gaia magnitudes, while
                                                   2MASS J, H, Ks reach 0.10 Msun (measured on the
                                                   Zenodo files, 2026-10-04)
=========  ==========================  ==========  ===============================================

So a control with either grid fixes [Fe/H] = 0 (the main fit must be re-run with
``grid.select(feh=0.0)`` to compare like with like) and fits a magnitude window inside the mass range
(``IsochroneFitter.setup(mag_window=...)`` with :func:`~erotica.analysis.grids.safe_window`).
Neither grid applies a star-by-star extinction coefficient in the Gaia bands.

References: Baraffe et al. 2015 (``2015A&A...577A..42B``); Somers, Cao & Pinsonneault 2020
(``2020ApJ...891...29S``), data Zenodo 10.5281/zenodo.3593339 (v1.0, CC BY 4.0).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np

from .base import GridNode, GridProvenance, IsochroneGrid


def _mass_index(masses: np.ndarray, grid_masses: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(grid_masses, np.round(masses, 4))
    if not np.allclose(grid_masses[idx], np.round(masses, 4)):
        raise ValueError("mass not in the grid's mass list")
    return idx.astype(float)


class BHAC15Grid(IsochroneGrid):
    """Baraffe et al. (2015) isochrones from ``BHAC15_iso.<system>`` (one solar composition)."""

    eep_kind = "mass"
    default_bands = ("G", "G_BP", "G_RP")
    default_effl = (6390.7, 5182.6, 7825.1)

    def __init__(
        self,
        path: str | Path,
        bands: tuple[str, ...] | None = None,
        *,
        loga_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.bands = tuple(bands) if bands is not None else self.default_bands
        fp = Path(path)
        if fp.is_dir():
            fp = fp / "BHAC15_iso.GAIA"
        blocks: list[tuple[float, list[str], list[list[float]]]] = []
        age, names = None, None
        for line in fp.read_text().splitlines():
            m = re.search(r"t \(Gyr\)\s*=\s*([0-9.]+)", line)
            if m:
                age, names = float(m.group(1)), None
                blocks.append((age, [], []))
                continue
            s = line.strip()
            if s.startswith("!") and "M/Ms" in s:
                names = s.lstrip("!").split()
                blocks[-1] = (age, names, [])
                continue
            if not s or s.startswith("!") or names is None:
                continue
            blocks[-1][2].append([float(v) for v in s.split()])
        all_m = np.unique(
            np.round(np.concatenate([np.asarray(r, float)[:, 0] for _, _, r in blocks if r]), 4)
        )
        self.mass_list = all_m
        for age_gyr, names, rows in blocks:
            if not rows:
                continue
            loga = round(float(np.log10(age_gyr * 1e9)), 4)
            if loga_range is not None and not (
                loga_range[0] - 1e-9 <= loga <= loga_range[1] + 1e-9
            ):
                continue
            a = np.asarray(rows, float)
            col = {n: a[:, i] for i, n in enumerate(names)}
            missing = [b for b in self.bands if b not in col]
            if missing:
                raise ValueError(f"{fp.name} lacks bands {missing}; it has {names}")
            mags = {b: col[b] for b in self.bands}
            ok = np.all([np.isfinite(v) & (v < 50) for v in mags.values()], axis=0)
            if ok.sum() < 2:
                continue
            mass = col["M/Ms"][ok]
            self._add(
                0.0,
                loga,
                GridNode(
                    _mass_index(mass, all_m),
                    mass,
                    {b: v[ok] for b, v in mags.items()},
                    {"logte": np.log10(col["Teff"][ok]), "logl": col["L/Ls"][ok]},
                ),
                None,
            )
        self.name = "BHAC15"
        self.provenance = GridProvenance(
            family="BHAC15",
            version=f"{fp.name} (file dated by the server; Gaia file updated 2020-08-10)",
            references=("2015A&A...577A..42B",),
            source="http://perso.ens-lyon.fr/isabelle.baraffe/BHAC15dir/",
            license="no licence declared; READ_INFO asks to cite Baraffe et al. 2015; not redistributed",
            metallicity="solar only ([Fe/H] = 0); Z not stated in the file",
            passbands=f"{', '.join(self.bands)}; Gaia release not declared (DR2-era file [I])",
            sha256=(f"{fp.name}:{hashlib.sha256(fp.read_bytes()).hexdigest()}",),
        )


class SPOTSGrid(IsochroneGrid):
    """SPOTS (Somers+2020) isochrones at one spot filling factor, from ``fXXX.isoc``.

    Parameters
    ----------
    path : str or Path
        The ``.isoc`` file, or the directory holding them (then ``fspot`` picks the file).
    fspot : float
        Nominal covering fraction: 0, 0.17, 0.34, 0.51, 0.68 or 0.85.
    bands : tuple of str
        Default Gaia DR2 ``G_mag BP_mag RP_mag``; ``("J_mag", "J_mag", "K_mag")``-style 2MASS
        choices reach lower masses (see the module table).
    """

    eep_kind = "mass"
    default_bands = ("G_mag", "BP_mag", "RP_mag")
    # The same CCM89 wavelengths as every other backend, on purpose: a control must not change the
    # extinction treatment together with the grid (landscape finding §15.2 point 3)
    default_effl = (6390.7, 5182.6, 7825.1)

    def __init__(
        self,
        path: str | Path,
        fspot: float = 0.0,
        bands: tuple[str, ...] | None = None,
        *,
        loga_range: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.bands = tuple(bands) if bands is not None else self.default_bands
        fp = Path(path)
        if fp.is_dir():
            fp = fp / f"f{int(round(fspot * 100)):03d}.isoc"
        names = None
        rows: list[list[float]] = []
        for line in fp.read_text().splitlines():
            s = line.strip()
            if s.startswith("##"):
                tok = s.lstrip("#").split()
                if tok[:2] == ["logAge", "Mass"]:
                    names = tok
                continue
            if s and names is not None:
                rows.append([float(v) for v in s.split()])
        if names is None:
            raise ValueError(f"{fp}: no SPOTS column line")
        a = np.asarray(rows, float)
        col = {n: a[:, i] for i, n in enumerate(names)}
        missing = [b for b in self.bands if b not in col]
        if missing:
            raise ValueError(f"{fp.name} lacks bands {missing}; it has {names}")
        self.fspot = float(np.median(col["Fspot"]))
        all_m = np.unique(np.round(col["Mass"], 4))
        self.mass_list = all_m
        loga_all = np.round(col["logAge"], 4)
        for loga in np.unique(loga_all):
            if loga_range is not None and not (
                loga_range[0] - 1e-9 <= loga <= loga_range[1] + 1e-9
            ):
                continue
            sel = loga_all == loga
            mags = {b: col[b][sel] for b in self.bands}
            ok = np.all([v > -90 for v in mags.values()], axis=0)
            if ok.sum() < 2:
                continue
            mass = col["Mass"][sel][ok]
            self._add(
                0.0,
                float(loga),
                GridNode(
                    _mass_index(mass, all_m),
                    mass,
                    {b: v[ok] for b, v in mags.items()},
                    {"logte": col["log(Teff)"][sel][ok], "logl": col["log(L/Lsun)"][sel][ok]},
                ),
                None,
            )
        self.name = f"SPOTS f={self.fspot:.2f}"
        self.provenance = GridProvenance(
            family="SPOTS",
            version="Zenodo 3593339 v1.0 (2019-12-26)",
            references=("2020ApJ...891...29S",),
            source="https://zenodo.org/records/3593339",
            license="CC BY 4.0 (declared on the Zenodo record)",
            metallicity="solar only ([Fe/H] = 0)",
            passbands=f"{', '.join(self.bands)}; Gaia DR2 via empirical MS colour tables",
            sha256=(f"{fp.name}:{hashlib.sha256(fp.read_bytes()).hexdigest()}",),
            notes=f"fspot = {self.fspot}",
        )
