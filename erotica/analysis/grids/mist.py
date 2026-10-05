r"""MIST isochrones (v1.2 and v2.5) as an :class:`~.base.IsochroneGrid` backend.

References: Dotter 2016 (``2016ApJS..222....8D``, EEPs and isochrone construction); Choi et al. 2016
(``2016ApJ...823..102C``, MIST v1); Dotter et al. 2026 (``2026ApJS..283...64D``, MIST II = v2.5).

**Reading.** Each ``.iso.cmd`` / ``*.iso.<system>`` file holds one composition and every age. The
composition comes from the header line ``Yinit Zinit [Fe/H] [a/Fe] v/vcrit`` and is checked against
the ``[Fe/H]_init`` data column when present (a mismatch raises: it would mean the header and the
models disagree about what was computed). Rows with a non-finite magnitude or non-positive mass are
dropped, as ``MISTIsochrones._read_file`` does; the regression test
``tests/test_grids.py::test_mist_backend_reproduces_the_legacy_reader`` holds the two readers to the
same arrays.

**Metallicity conventions, measured on the file headers (2026-10-04)** -- see
``tools/validation/isochrone_grids/feh_conventions.py`` for the script and its JSON:

=========  ============  =========================================  ==========================
version    Z(label 0)    label definition that fits all 15 files     max residual
=========  ============  =========================================  ==========================
v1.2       0.0142857     ``log10(Z / 0.0142857)``                    < 1e-4 dex
v2.5       0.0163577     ``log10(Z/X) - log10(Z/X)_sun``             < 1e-4 dex
=========  ============  =========================================  ==========================

with ``Y = 0.2490 + 1.490 Z`` (v1.2) and ``Y = 0.2490 + 1.301 Z`` (v2.5), both exact to 1e-4.
The mixtures are Asplund et al. 2009 (v1.2, protosolar) and Grevesse & Sauval 1998 (v2.5) -- stated by
``mist.science`` and MIST II, not derivable from the headers. Consequence: a v1.2 "[Fe/H] = +0.5"
and a v2.5 "[Fe/H] = +0.5" are different compositions (``log(Z/X)`` differs by 0.049 dex), and an
ASteCA-style ``log10(Z/Z_sun)`` reproduces v1.2 exactly and v2.5 to within 0.045 dex at +0.5.

**MIST v2.5 content, measured on the UBVRIplus files (2026-10-04).** Gaia columns present:
``Gaia_{G,BP,RP}_DR2Rev``, ``Gaia_G_MAW``, ``Gaia_BP_MAWb``, ``Gaia_BP_MAWf``, ``Gaia_RP_MAW``,
``Gaia_{G,BP,RP}_EDR3`` (the same set as v1.2), plus 2MASS. Magnitudes are finite below
``T_eff = 3500 K`` (the lower edge of MIST II's ATLAS12/SYNTHE BC grid) down to ~3000 K; how they are
obtained there is not stated in what was read [I]. At log t = 6.0 the [Fe/H] = 0 isochrone starts at
0.251 Msun (0.133 at 6.3; 0.10 from 6.5), against 0.10 at every age in v1.2: at the youngest ages
v2.5 has no model for the lowest-mass members, so a fit there must cut the faint end
(``IsochroneFitter.setup(mag_window=...)``).

**Licence.** No data licence is declared by ``mist.science`` ("(c) MIST. All rights reserved"
footer; the MIST II article is CC BY 4.0). The files are therefore never committed or redistributed
by EROTICA: :mod:`.fetch` downloads them into a user cache and records their SHA-256.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .base import GridNode, GridProvenance, IsochroneGrid

MIST_BIBCODES = {
    "1.2": ("2016ApJS..222....8D", "2016ApJ...823..102C"),
    "2.5": ("2016ApJS..222....8D", "2026ApJS..283...64D"),
}


def _header(fpath: Path) -> tuple[dict, list[str]]:
    """Header values and the column names of a MIST isochrone file."""
    meta: dict = {}
    names: list[str] | None = None
    lines: list[str] = []
    with open(fpath, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith("#"):
                break
            lines.append(line.lstrip("#").strip())
    for i, ln in enumerate(lines):
        tok = ln.split()
        if ln.startswith("MIST version number"):
            meta["version"] = ln.split("=")[1].strip()
        if "Zinit" in tok and i + 1 < len(lines):
            vals = lines[i + 1].split()
            for k, v in zip(tok, vals, strict=False):
                meta[k] = float(v)
        if names is None and "EEP" in tok and "initial_mass" in tok:
            names = tok
    if names is None or "Zinit" not in meta:
        raise ValueError(f"{fpath.name}: not a MIST isochrone file (no Zinit or no column line)")
    return meta, names


def _sha256(fpath: Path) -> str:
    h = hashlib.sha256()
    with open(fpath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


class MISTGrid(IsochroneGrid):
    """MIST v1.2 or v2.5 isochrones read from a directory of single-composition files.

    Parameters
    ----------
    path : str or Path
        Directory with the files (``*.iso.cmd`` for v1.2, ``*.iso.*`` for v2.5), or one file.
    bands : tuple of str
        Magnitude columns to keep. Default: Gaia EDR3 G, BP, RP.
    afe : float, default 0.0
        [alpha/Fe] to keep (v2.5 ships -0.2 ... +0.6; v1.2 only 0).
    vvcrit : float, default 0.4
        Initial rotation v/v_crit to keep (both versions ship 0.0 and 0.4).
    loga_range : tuple, optional
        Keep only ages inside this range (memory: a v2.5 file is ~84 MB of text).
    hash_files : bool, default False
        Record the SHA-256 of every file read in :attr:`provenance` (slow on v2.5).
    """

    eep_kind = "native"
    default_bands = ("Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3")
    default_effl = (6390.7, 5182.6, 7825.1)

    def __init__(
        self,
        path: str | Path,
        bands: tuple[str, ...] | None = None,
        *,
        afe: float = 0.0,
        vvcrit: float = 0.4,
        loga_range: tuple[float, float] | None = None,
        hash_files: bool = False,
    ) -> None:
        super().__init__()
        import pandas as pd

        self.bands = tuple(bands) if bands is not None else self.default_bands
        p = Path(path)
        files = [p] if p.is_file() else sorted(f for f in p.iterdir() if ".iso" in f.name)
        if not files:
            raise FileNotFoundError(f"no MIST isochrone files in {p}")
        versions, hashes, header_rows = set(), [], []
        for fp in files:
            meta, names = _header(fp)
            if (
                abs(meta.get("[a/Fe]", 0.0) - afe) > 1e-6
                or abs(meta.get("v/vcrit", 0.4) - vvcrit) > 1e-6
            ):
                continue
            versions.add(meta.get("version", "?"))
            feh, z = float(meta["[Fe/H]"]), float(meta["Zinit"])
            header_rows.append((meta.get("Yinit", np.nan), z, feh))
            missing = [b for b in self.bands if b not in names]
            if missing:
                raise ValueError(f"{fp.name} lacks bands {missing}; it has {names}")
            use = ["EEP", "log10_isochrone_age_yr", "initial_mass", *self.bands]
            extra = [c for c in ("log_Teff", "log_L", "phase", "[Fe/H]_init") if c in names]
            df = pd.read_csv(
                fp,
                sep=r"\s+",
                comment="#",
                header=None,
                names=names,
                usecols=use + extra,
                # pandas' default fast parser differs from Python's float() in the last ulp;
                # round_trip makes the arrays bit-identical to MISTIsochrones (regression test)
                float_precision="round_trip",
            )
            if "[Fe/H]_init" in df and not np.allclose(df["[Fe/H]_init"], feh, atol=1e-6):
                raise ValueError(f"{fp.name}: header [Fe/H] {feh} != data column [Fe/H]_init")
            loga = np.round(df["log10_isochrone_age_yr"].to_numpy(), 4)
            for a in np.unique(loga):
                if loga_range is not None and not (
                    loga_range[0] - 1e-9 <= a <= loga_range[1] + 1e-9
                ):
                    continue
                s = df[loga == a]
                mags = {b: s[b].to_numpy(float) for b in self.bands}
                mass = s["initial_mass"].to_numpy(float)
                ok = np.isfinite(mass) & (mass > 0)
                for v in mags.values():
                    ok &= np.isfinite(v)
                if ok.sum() < 2:
                    continue
                node = GridNode(
                    s["EEP"].to_numpy(float)[ok],
                    mass[ok],
                    {b: v[ok] for b, v in mags.items()},
                    {
                        std: s[c].to_numpy(float)[ok]
                        for c, std in (("log_Teff", "logte"), ("log_L", "logl"), ("phase", "phase"))
                        if c in s
                    },
                )
                self._add(feh, float(a), node, z)
            if hash_files:
                hashes.append(f"{fp.name}:{_sha256(fp)}")
        if not self._nodes:
            raise ValueError(f"no node read from {p} with [a/Fe]={afe}, v/vcrit={vvcrit}")
        if len(versions) != 1:
            raise ValueError(f"mixed MIST versions in {p}: {sorted(versions)}")
        self.version = versions.pop()
        self.header_compositions = np.array(sorted(header_rows, key=lambda r: r[2]))
        self.name = f"MIST v{self.version}"
        conv = {
            "1.2": "[Fe/H] label = log10(Z/0.0142857); Asplund+09 protosolar mixture",
            "2.5": "[Fe/H] label = log10(Z/X) - log10(Z/X)_sun, Z_sun = 0.0163577; GS98 mixture",
        }.get(self.version, "not measured for this version")
        self.provenance = GridProvenance(
            family="MIST",
            version=self.version,
            references=MIST_BIBCODES.get(self.version, ("2016ApJS..222....8D",)),
            source="https://mist.science/model_grids.html",
            license="no data licence declared ('(c) MIST. All rights reserved'); not redistributed",
            metallicity=conv,
            passbands=", ".join(self.bands),
            sha256=tuple(hashes),
            notes=f"[a/Fe]={afe}, v/vcrit={vvcrit}",
        )

    def feh_convention_residuals(self) -> dict[str, float]:
        """Max |label - formula| over the files read, for the two candidate definitions.

        ``log10(Z/Z0)`` and ``log10(Z/X) - log10(Z0/X0)``, with ``Z0, X0`` from the label-0 file.
        Needs the [Fe/H] = 0 file among those read.
        """
        y, z, feh = self.header_compositions.T
        i0 = int(np.argmin(np.abs(feh)))
        if abs(feh[i0]) > 1e-9:
            raise ValueError("the [Fe/H] = 0 file was not read")
        x = 1.0 - y - z
        return {
            "log10(Z/Zsun)": float(np.max(np.abs(np.log10(z / z[i0]) - feh))),
            "log10(Z/X)-sun": float(
                np.max(np.abs(np.log10(z / x) - np.log10(z[i0] / x[i0]) - feh))
            ),
            "Zsun": float(z[i0]),
        }
