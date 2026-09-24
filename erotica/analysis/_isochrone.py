"""Isochrone fitting: native MIST reader + PyMC NUTS.

Forward model with ASteCA's ingredients, likelihood unbinned per star (since 2026-09-22):
  - Linear-Z metallicity (Zinit) as stored in MIST files; isochrones interpolated at fixed EEP
  - Chabrier (2014) IMF
  - Offner binary fraction + Duchêne & Kraus (2013) mass-ratio distribution, marginalised
  - Per-star photometric errors, a fitted error model for the completeness cut
  - Cardelli, Clayton & Mathis (1989) + O'Donnell (1994) extinction law
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import QTable

from .._membership import COLUMNA_ISOCRONA, select_by_probability


def _require_pymc():
    """Import PyMC, or fail with the name of the extra that supplies it.

    The isochrone module was the one sampler path that imported ``pymc`` bare, so a user
    without the extra got a raw ``ModuleNotFoundError`` naming a package they never asked
    for, while every other entry point named the extra. ``paper/paper.md`` claimed all of
    them were guarded; this makes that true rather than softening the claim.
    """
    try:
        import pymc as pm
    except ImportError as exc:  # pragma: no cover - exercised in the no-bayes CI leg
        raise ImportError(
            "PyMC is required for erotica.analysis isochrone fitting. Install the 'bayes' extra."
        ) from exc
    return pm


# ---------------------------------------------------------------------------
# CCM89 + O'Donnell 94  (A_lambda / A_V)
# ---------------------------------------------------------------------------


def _ccm89(lam_aa: float, Rv: float = 3.1) -> float:
    x = 1e4 / lam_aa  # Å → 1/μm
    if 1.1 <= x <= 3.3:
        y = x - 1.82
        a = (
            1
            + 0.17699 * y
            - 0.50447 * y**2
            - 0.02427 * y**3
            + 0.72085 * y**4
            + 0.01979 * y**5
            - 0.77530 * y**6
            + 0.32999 * y**7
        )
        b = (
            1.41338 * y
            + 2.28305 * y**2
            + 1.07233 * y**3
            - 5.38434 * y**4
            - 0.62251 * y**5
            + 5.30260 * y**6
            - 2.09002 * y**7
        )
    elif 3.3 < x <= 8.0:
        Fa = Fb = 0.0
        if x >= 5.9:
            y = x - 5.9
            Fa = -0.04473 * y**2 - 0.009779 * y**3
            Fb = 0.2130 * y**2 + 0.1207 * y**3
        a = 1.752 - 0.316 * x - 0.104 / ((x - 4.67) ** 2 + 0.341) + Fa
        b = -3.090 + 1.825 * x + 1.206 / ((x - 4.62) ** 2 + 0.263) + Fb
    else:
        a = 0.574 * x**1.61
        b = -0.527 * x**1.61
    return a + b / Rv


# ---------------------------------------------------------------------------
# Chabrier (2003/2014) IMF weights
# ---------------------------------------------------------------------------


def _chabrier2014_weights(mass: np.ndarray) -> np.ndarray:
    """IMF weight per EEP, normalised to sum = 1.

    Chabrier (2014) system IMF:
      m < m0 : log-normal  ξ(m) = exp(-0.5*(log10(m)-log10(mc))²/σ²) / m
      m ≥ m0 : power-law   ξ(m) ∝ m^{-2.35}  (Salpeter slope)
    with mc = 0.20 M☉, σ = 0.55, transition at m0 = 1.0 M☉.
    Weight = ξ(m) × |Δm|.
    """
    mc, sigma, m0 = 0.20, 0.55, 1.0
    log10_m = np.log10(np.clip(mass, 1e-9, None))
    log10_mc = np.log10(mc)

    xi_ln = np.exp(-0.5 * (log10_m - log10_mc) ** 2 / sigma**2) / np.clip(mass, 1e-9, None)
    xi_pl = mass ** (-2.35)

    # Scale power-law to match log-normal at m0 (continuity)
    xi_ln_at_m0 = np.exp(-0.5 * (np.log10(m0) - log10_mc) ** 2 / sigma**2) / m0
    xi_pl_at_m0 = m0 ** (-2.35)
    scale = xi_ln_at_m0 / xi_pl_at_m0 if xi_pl_at_m0 > 0 else 1.0

    xi = np.where(mass < m0, xi_ln, xi_pl * scale)
    dm = np.abs(np.gradient(mass))
    w = xi * dm
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    s = w.sum()
    return w / s if s > 0 else np.ones(len(mass)) / len(mass)


# ---------------------------------------------------------------------------
# Duchêne & Kraus (2013) mean mass-ratio q = m2/m1
# ---------------------------------------------------------------------------


def _dk_mean_q(mass: np.ndarray) -> np.ndarray:
    """Expected value of q for each primary mass, D&K (2013) power-law distribution.

    Marginalises the power-law  f(q) ∝ q^γ  over [0, 1]:
      E[q] = (γ+1)/(γ+2)
    with mass-dependent γ from Table 1 of Duchêne & Kraus (2013).
    """
    gamma = np.where(
        mass <= 0.1,
        4.2,
        np.where(mass <= 0.6, 0.4, np.where(mass <= 1.4, 0.3, np.where(mass <= 6.5, -0.5, 0.0))),
    )
    return (gamma + 1.0) / (gamma + 2.0)


# ---------------------------------------------------------------------------
# Combine two magnitudes into an unresolved pair
# ---------------------------------------------------------------------------


def _mag_combine(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return -2.5 * np.log10(10 ** (-0.4 * m1) + 10 ** (-0.4 * m2))


# ---------------------------------------------------------------------------
# MIST isochrone reader
# ---------------------------------------------------------------------------


class MISTIsochrones:
    """Read MIST ``.iso.cmd`` files and expose an isochrone grid.

    Metallicity is stored as **linear Z** (``Zinit``), matching ASteCA.
    """

    def __init__(
        self,
        isochs_path: str | Path,
        magnitude_col: str = "Gaia_G_EDR3",
        color_col1: str = "Gaia_BP_EDR3",
        color_col2: str = "Gaia_RP_EDR3",
        mass_col: str = "initial_mass",
        loga_col: str = "log10_isochrone_age_yr",
    ) -> None:
        self.isochs_path = Path(isochs_path)
        self.magnitude_col = magnitude_col
        self.color_col1 = color_col1
        self.color_col2 = color_col2
        self.mass_col = mass_col
        self.loga_col = loga_col

        # (Z, loga) → (mass, G_abs, BP_abs, RP_abs)
        self._grid: dict[
            tuple[float, float],
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        # (Z, loga) -> EEP of each point, same order as ``_grid`` (empty if the file has no EEP)
        self._eep: dict[tuple[float, float], np.ndarray] = {}
        self._met_values: np.ndarray | None = None
        self._loga_values: np.ndarray | None = None
        self._load()

    # ------------------------------------------------------------------
    # Z extraction — reads "Zinit" from MIST header (like ASteCA)
    # ------------------------------------------------------------------

    @staticmethod
    def _z_from_header_lines(lines: list[str]) -> float | None:
        """Find 'Zinit' column in already-collected header lines, read its value."""
        for i, line in enumerate(lines):
            tokens = line.split()
            if "Zinit" in tokens:
                col_idx = tokens.index("Zinit")
                for next_line in lines[i + 1 :]:
                    vals = next_line.split()
                    if len(vals) > col_idx:
                        try:
                            return float(vals[col_idx])
                        except ValueError:
                            continue
        return None

    def _z_from_header(self, fpath: Path) -> float | None:
        """Find 'Zinit' column in header, read value from the following line."""
        lines: list[str] = []
        with open(fpath, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith("#"):
                    break
                lines.append(line.strip().lstrip("#").strip())
        return self._z_from_header_lines(lines)

    def _z_from_name(self, fpath: Path) -> float | None:
        """Fallback: parse [Fe/H] from filename and convert to Z."""
        name = fpath.stem.lower()
        # e.g.  feh_p0.00  or  feh+0.00  or  feh-0.50
        m = re.search(r"feh_?([mp][0-9]+\.?[0-9]*)", name)
        if m:
            raw = m.group(1).replace("p", "+").replace("m", "-")
            try:
                feh = float(raw)
                return 0.0152 * 10**feh
            except ValueError:
                pass
        m = re.search(r"feh([+-]?[0-9]+\.?[0-9]*)", name)
        if m:
            feh = float(m.group(1))
            return 0.0152 * 10**feh
        return None

    # ------------------------------------------------------------------
    # File parsing
    # ------------------------------------------------------------------

    def _read_file(self, fpath: Path) -> None:
        # Single-pass: collect header lines and data rows simultaneously
        header_lines: list[str] = []
        colnames: list[str] | None = None
        ncols: int = 0
        rows: list[list[float]] = []

        with open(fpath, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("#"):
                    stripped = line.strip().lstrip("#").strip()
                    header_lines.append(stripped)
                    tokens = stripped.split()
                    if colnames is None and any(
                        t in tokens
                        for t in [
                            self.magnitude_col,
                            self.loga_col,
                            "initial_mass",
                            "EEP",
                        ]
                    ):
                        colnames = tokens
                        ncols = len(tokens)
                else:
                    if colnames is None or not line.strip():
                        continue
                    vals = line.split()
                    if len(vals) != ncols:
                        continue
                    try:
                        rows.append([float(v) for v in vals])
                    except ValueError:
                        continue

        Z = self._z_from_header_lines(header_lines)
        if Z is None:
            Z = self._z_from_name(fpath)
        if Z is None:
            warnings.warn(f"Cannot determine Zinit for {fpath.name}; assuming Z_sun.", stacklevel=2)
            Z = 0.0152
        Z = round(Z, 6)

        if colnames is None:
            warnings.warn(f"No column header in {fpath.name}; skipping.", stacklevel=2)
            return

        col_idx = {c: i for i, c in enumerate(colnames)}
        needed = [self.loga_col, self.magnitude_col, self.color_col1, self.color_col2]
        if not all(c in col_idx for c in needed):
            missing = [c for c in needed if c not in col_idx]
            warnings.warn(f"Missing columns {missing} in {fpath.name}; skipping.", stacklevel=2)
            return

        i_loga = col_idx[self.loga_col]
        i_G = col_idx[self.magnitude_col]
        i_BP = col_idx[self.color_col1]
        i_RP = col_idx[self.color_col2]
        i_mass = col_idx.get(self.mass_col)
        i_eep = col_idx.get("EEP")

        if not rows:
            return

        data = np.array(rows, dtype=np.float64)
        loga_all = data[:, i_loga]
        G_all = data[:, i_G]
        BP_all = data[:, i_BP]
        RP_all = data[:, i_RP]
        mass_all = data[:, i_mass] if i_mass is not None else np.ones(len(data))
        eep_all = data[:, i_eep] if i_eep is not None else np.full(len(data), np.nan)

        for loga_val in np.unique(np.round(loga_all, 4)):
            mask = np.round(loga_all, 4) == loga_val
            G = G_all[mask]
            BP = BP_all[mask]
            RP = RP_all[mask]
            mass = mass_all[mask]
            valid = (
                np.isfinite(G) & np.isfinite(BP) & np.isfinite(RP) & np.isfinite(mass) & (mass > 0)
            )
            if valid.sum() < 2:
                continue
            # stable sort, so equal masses (MIST repeats a few near the ZAMS) keep EEP order
            sidx = np.argsort(mass[valid], kind="stable")
            self._grid[(Z, float(loga_val))] = (
                mass[valid][sidx],
                G[valid][sidx],
                BP[valid][sidx],
                RP[valid][sidx],
            )
            self._eep[(Z, float(loga_val))] = eep_all[mask][valid][sidx]

    def _load(self) -> None:
        seen: set[Path] = set()
        for pat in ("*.iso.cmd", "*.cmd", "*.dat", "*.txt"):
            for fp in sorted(self.isochs_path.glob(pat)):
                seen.add(fp)
        if not seen:
            raise FileNotFoundError(f"No isochrone files found in {self.isochs_path}")
        for fp in sorted(seen):
            self._read_file(fp)
        if not self._grid:
            raise ValueError(f"No valid isochrones parsed from {self.isochs_path}")
        self._met_values = np.array(sorted({k[0] for k in self._grid}))
        self._loga_values = np.array(sorted({k[1] for k in self._grid}))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def met_age_dict(self) -> dict[str, np.ndarray]:
        return {"met": self._met_values, "loga": self._loga_values}

    def get_isochrone(
        self, met: float, loga: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(mass, G_abs, BP_abs, RP_abs)`` at the nearest grid point."""
        mi = int(np.argmin(np.abs(self._met_values - met)))  # type: ignore[arg-type]
        ai = int(np.argmin(np.abs(self._loga_values - loga)))  # type: ignore[arg-type]
        key = (float(self._met_values[mi]), float(self._loga_values[ai]))  # type: ignore[index]
        return self._grid[key]

    def get_isochrone_eep(
        self, met: float, loga: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(eep, mass, G_abs, BP_abs, RP_abs)`` at the nearest node, ordered by EEP.

        :meth:`get_isochrone` orders by mass, which callers rely on for ``np.interp``; this
        accessor orders by the equivalent evolutionary phase, the coordinate in which MIST
        isochrones are interpolated (Dotter 2016, ``2016ApJS..222....8D``; Choi et al. 2016,
        ``2016ApJ...823..102C``). Raises ``ValueError`` if the file had no ``EEP`` column.
        """
        mi = int(np.argmin(np.abs(self._met_values - met)))  # type: ignore[arg-type]
        ai = int(np.argmin(np.abs(self._loga_values - loga)))  # type: ignore[arg-type]
        key = (float(self._met_values[mi]), float(self._loga_values[ai]))  # type: ignore[index]
        eep = self._eep[key]
        if not np.all(np.isfinite(eep)):
            raise ValueError(f"isochrone {key} was read without an EEP column")
        o = np.argsort(eep, kind="stable")
        mass, G, BP, RP = self._grid[key]
        return eep[o], mass[o], G[o], BP[o], RP[o]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _error_poly(obs_mag: np.ndarray, e: np.ndarray) -> np.ndarray:
    """Coefficients ``(c0, c1, c2)`` of ``log10 e = c0 + c1 m + c2 m**2``.

    Falls back to a constant (the median error) when there is not enough information to
    constrain a quadratic: a degree-2 fit needs at least three *distinct* magnitudes, and with
    fewer the Vandermonde matrix is rank-deficient and the polynomial arbitrary.
    """
    from numpy.polynomial.polynomial import Polynomial

    ok = np.isfinite(obs_mag) & np.isfinite(e) & (e > 0)
    if np.unique(obs_mag[ok]).size < 3:
        # the median of the column as given, as before 2026-09-22 (a test pins it)
        return np.array([np.log10(float(np.nanmedian(e))), 0.0, 0.0])
    return np.asarray(Polynomial.fit(obs_mag[ok], np.log10(e[ok]), 2).convert().coef, float)


def _fit_error_model(obs_mag: np.ndarray, e_mag: np.ndarray, e_col: np.ndarray) -> tuple[Any, Any]:
    """Fit quadratic log-error vs magnitude models for interpolation.

    Falls back to a constant (the median error) when there is not enough
    information to constrain a quadratic. A degree-2 fit needs at least three
    *distinct* magnitudes: with fewer, or with repeated magnitudes, the
    Vandermonde matrix is rank-deficient and the returned polynomial is
    arbitrary rather than merely imprecise.

    Notes
    -----
    Before 2026-07-27 the guard read ``mag_ok.sum() < 3``, but ``mag_ok`` is the
    *filtered magnitude array*, not a boolean mask -- so it summed magnitudes
    (two stars at G = 14, 15 give 29) instead of counting points. The fallback
    therefore never fired for any realistic input, and under-determined fits were
    accepted silently: NumPy's ``RankWarning`` was swallowed by a blanket
    ``ignore::RuntimeWarning`` in ``pytest.ini``. Both are fixed.
    """

    def _callable(coef: np.ndarray) -> Any:
        return lambda m: (
            10 ** np.polynomial.polynomial.polyval(np.atleast_1d(m).astype(float), coef)
        )

    return _callable(_error_poly(obs_mag, e_mag)), _callable(_error_poly(obs_mag, e_col))


def _chabrier2014_xi(m: Any, xp: Any = np) -> Any:
    r"""Chabrier (2014) system IMF :math:`\xi(m) = dN/dm`, unnormalised; ``xp`` is numpy or
    ``pytensor.tensor``. Same constants and continuity at :math:`m_0` as
    :func:`_chabrier2014_weights`."""
    mc, sigma, m0 = 0.20, 0.55, 1.0
    scale = float(np.exp(-0.5 * (np.log10(mc) / sigma) ** 2))  # xi_ln(m0) / m0**-2.35, m0 = 1
    lm = xp.log10(m)
    xi_ln = xp.exp(-0.5 * ((lm - np.log10(mc)) / sigma) ** 2) / m
    return xp.where(m < m0, xi_ln, scale * m**-2.35)


# Duchêne & Kraus (2013) Table 1: f(q) ∝ q**gamma, gamma by primary-mass range. The model uses
# a logistic step of 0.005 dex so the likelihood stays differentiable when an interpolated
# primary mass crosses a range boundary; the synthetic-cluster generators use the exact step.
_DK_EDGES = (0.1, 0.6, 1.4, 6.5)
_DK_GAMMAS = (4.2, 0.4, 0.3, -0.5, 0.0)
_DK_WIDTH_DEX = 0.005
# Mass-ratio nodes. Dense near q = 1, where the combined magnitude changes fastest, and at
# small q, where f(q) ∝ q**-0.5 (1.4-6.5 Msun primaries) piles the pairs up: with a single
# [0, 0.3] piece the uniform deposit along it misplaced them, and a Monte Carlo of the toy
# family at 0.01 mag errors gave chi2/dof 2.45 (1.17 with these nodes; measured 2026-09-23).
_Q_NODES = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.65, 0.75, 0.85, 0.93, 1.0])


def _dk_gamma(mass: Any, xp: Any = np, smooth: bool = False) -> Any:
    """Power-law index of the D&K (2013) mass-ratio distribution for each primary mass."""
    if not smooth:
        m = np.asarray(mass, float)
        return np.select([m <= e for e in _DK_EDGES], list(_DK_GAMMAS[:-1]), default=_DK_GAMMAS[-1])
    lm = xp.log10(mass)
    g = _DK_GAMMAS[0]
    for edge, g0, g1 in zip(_DK_EDGES, _DK_GAMMAS[:-1], _DK_GAMMAS[1:], strict=True):
        x = xp.clip((lm - np.log10(edge)) / _DK_WIDTH_DEX, -50.0, 50.0)
        g = g + (g1 - g0) / (1.0 + xp.exp(-x))
    return g


def _erfc_np(x: Any) -> Any:
    from scipy.special import erfc

    return erfc(x)


def _segment_density(
    xg: Any,
    xc: Any,
    sg: Any,
    sc: Any,
    Ag: Any,
    Ac: Any,
    Bg: Any,
    Bc: Any,
    xp: Any = np,
    smear: tuple[Any, Any] | None = None,
    first_moment: bool = False,
) -> Any:
    r"""Density of star ``i`` from a uniform deposit along the segment ``A_k -> B_k``.

    .. math::
       d_{ik} = \int_0^1 \mathcal{N}_2\!\left(x_i;\ A_k + t\,(B_k - A_k),\ \Sigma_{ik}\right) dt
       = \frac{e^{-\rho_{ik}^2/2}}{2\pi \sqrt{|\Sigma_{ik}|}}\,
         \frac{\sqrt{2\pi}}{L_{ik}}\left[\Phi\!\big(L_{ik}(1-t^*_{ik})\big)
         - \Phi\!\big(-L_{ik} t^*_{ik}\big)\right],

    in coordinates whitened by :math:`\Sigma_{ik}` (Cholesky): :math:`L` is the whitened
    segment length, :math:`t^*` the parameter of closest approach and :math:`\rho` the
    whitened perpendicular distance. Exact for a straight segment. :math:`\Sigma_{ik}` is
    the star's own ``diag(sg_i**2, sc_i**2)``, plus, if ``smear = (vg_k, vc_k)`` is given,
    :math:`v_k v_k^\top/12` -- the covariance of a uniform spread along :math:`v_k` (used for
    the primary-mass direction of the binary sheet). Stars ``(N,)``, segments ``(S,)``;
    returns ``(N, S)``.

    With ``first_moment``, also returns the first moment in the segment parameter,
    :math:`\int_0^1 t\,\mathcal{N}_2(\cdot)\,dt`, which a density varying linearly along the
    segment needs; in whitened coordinates
    :math:`\int_0^1 t\,e^{-L^2(t-t^*)^2/2}dt = t^* I_0 + \big[e^{-L^2 t^{*2}/2}
    - e^{-L^2(1-t^*)^2/2}\big]/L^2`, with :math:`I_0` the integral above. Returns ``(J0, J1)``.
    """
    s11 = (sg**2)[:, None]
    s22 = (sc**2)[:, None]
    s12 = 0.0 * s11
    if smear is not None:
        vg, vc = smear
        s11 = s11 + (vg**2 / 12.0)[None, :]
        s22 = s22 + (vc**2 / 12.0)[None, :]
        s12 = s12 + (vg * vc / 12.0)[None, :]
    # Cholesky of [[s11, s12], [s12, s22]]: whitened u1 = dg / l11, u2 = (dc - l21 u1) / l22
    l11 = xp.sqrt(s11)
    l21 = s12 / l11
    l22 = xp.sqrt(s22 - l21**2)
    ag = (Ag[None, :] - xg[:, None]) / l11
    ac = ((Ac[None, :] - xc[:, None]) - l21 * ag) / l22
    dg = (Bg - Ag)[None, :] / l11
    dc = ((Bc - Ac)[None, :] - l21 * dg) / l22
    L2 = dg**2 + dc**2
    long_ = L2 > 1e-8
    L2s = xp.where(long_, L2, 1.0)  # safe value: no NaN in the branch not taken, nor its grad
    L = xp.sqrt(L2s)
    t0 = -(ag * dg + ac * dc) / L2s
    perp2 = (ag + t0 * dg) ** 2 + (ac + t0 * dc) ** 2
    erf = xp.erf if xp is not np else _erf_np
    cdf_diff = 0.5 * (erf(L * (1 - t0) / np.sqrt(2.0)) - erf(-L * t0 / np.sqrt(2.0)))
    I0 = np.sqrt(2 * np.pi) / L * cdf_diff
    seg = xp.exp(-0.5 * perp2) * I0
    point = xp.exp(-0.5 * (ag**2 + ac**2))
    norm = 1.0 / (2 * np.pi * l11 * l22)
    J0 = xp.where(long_, seg, point) * norm
    if not first_moment:
        return J0
    I1 = t0 * I0 + (xp.exp(-0.5 * L2s * t0**2) - xp.exp(-0.5 * L2s * (1 - t0) ** 2)) / L2s
    J1 = xp.where(long_, xp.exp(-0.5 * perp2) * I1, 0.5 * point) * norm
    return J0, J1


def _erf_np(x: Any) -> Any:
    from scipy.special import erf

    return erf(x)


# ---------------------------------------------------------------------------
# IsochroneFitter
# ---------------------------------------------------------------------------


class IsochroneFitter:
    r"""Bayesian isochrone fitting: unbinned per-star likelihood over EEP-interpolated MIST
    isochrones, sampled with NUTS (PyMC).

    Forward model (ASteCA's ingredients, without the binned Hess):

    * **Isochrone at any (Z, log t).** Bilinear interpolation between the four bracketing MIST
      nodes, in :math:`\log_{10} Z` and :math:`\log_{10} t`, **at fixed EEP** -- the equivalent
      evolutionary phase, the coordinate MIST isochrones are built to be interpolated in
      (Dotter 2016, ``2016ApJS..222....8D``). Points whose EEP a node lacks are clamped to that
      node's last point, so their mass step and hence IMF weight go to zero continuously.
    * **Single stars** are deposited along the segments between consecutive EEP points with the
      Chabrier (2014) IMF weights :math:`w_k = \xi(\bar m_k)\,\Delta m_k`, as a density per
      unit length that is continuous along the track (linear within each segment; see
      :meth:`_segments`).
    * **Unresolved binaries**: fraction :math:`b(m) = \alpha + \beta/(1 + 1.4/m)` (Offner et al.
      2023, ``2023ASPC..534..275O``, as in ASteCA), mass ratio marginalised over
      :math:`f(q \mid m) \propto q^{\gamma(m)}` (Duchêne & Kraus 2013,
      ``2013ARA&A..51..269D``, Table 1) by a uniform deposit along the binary locus between
      the mass-ratio nodes ``_Q_NODES``, each piece weighted by its exact probability
      :math:`q_{j+1}^{\gamma+1} - q_j^{\gamma+1}`.
    * **Extinction**: CCM89 + O'Donnell (1994), :math:`A_\lambda = k_\lambda A_V`.

    Likelihood, for star :math:`i` with photometric errors :math:`(e_{G,i}, e_{c,i})`:

    .. math::
       \ln\mathcal{L} = \sum_i \omega_i \ln\!\left[(1 - f_{\rm bg})\,
       \frac{\sum_k w_k\, d_{ik}(\theta)}{F(\theta)} + \frac{f_{\rm bg}}{A}\right],
       \qquad s_{\cdot,i}^2 = e_{\cdot,i}^2 + \sigma_{\rm floor}^2 + \sigma_{\rm int}^2,

    with :math:`d_{ik}` the segment density of :func:`_segment_density`,
    :math:`F(\theta) = \sum_k w_k\,\bar P_k`, with :math:`\bar P_k` the probability that a model
    star deposited along segment :math:`k` is observed brighter than the faintest member,
    integrated exactly along the segment (:meth:`_p_observed`; the completeness cut, as ASteCA's
    ``cut_max_mag``), :math:`A` the area of the members' CMD bounding box, :math:`f_{\rm bg}` a
    uniform field fraction, :math:`\sigma_{\rm int}` a free intrinsic width and
    :math:`\sigma_{\rm floor}` = ``SIGMA_FLOOR`` = 0.01 mag a fixed floor, the forward model's
    own approximation error (see the constant). :math:`\omega_i` are the optional MS/PMS weights of :meth:`setup` (1 by default).
    The likelihood is conditioned on the number of members, so there is no amplitude parameter.

    Why not the binned Hess that stood here until 2026-09-22: it shifted a *precomputed*
    histogram, so log L depended on the grid's internal reference and posteriors locked onto
    ``(dm_mu, mean(Av_range))`` and onto metallicity nodes. A per-evaluation binned deposit was
    measured as the alternative and rejected: the bin-scale smoothing it needs for smooth
    gradients biased ``dm`` by about -0.25 mag on synthetic clusters, and without it the
    likelihood was rough. See ``docs/design-notes/decisions.md`` (2026-09-22/23).

    Notes
    -----
    Workflow:

    1. ``fitter = IsochroneFitter(isochs_path, ...)``
    2. ``fitter.setup(cluster_data, prob_threshold=0.6)``
    3. ``fitter.save_grid(path)``  /  ``fitter.load_grid(path)``  ← optional (node table)
    4. ``idata = fitter.fit(draws=2000, tune=1000, chains=4)``
    5. ``obs_mag, obs_col, cmds = fitter.posterior_cmd(idata, num_samples=30)``
    """

    # Width floor, in magnitudes, in quadrature with each star's errors: the forward model's own
    # approximation error. IMF-weighted 68th percentile of the leave-one-node-out residual of
    # the EEP interpolation, over the stars visible at NGC 6383's priors: 0.016-0.026 mag in
    # age (at 0.10 dex, twice the native step), 0.025-0.032 in Z (0.5 dex, twice native);
    # scaled to the native steps as the square of the step, 0.005 and 0.008. Measured
    # 2026-09-23 (tools/validation/isochrone_likelihood_d1/budget.py).
    SIGMA_FLOOR = 0.01
    # The binary sheet over (primary, q): exact along q between _Q_NODES; along the primary it
    # is built on every BINARY_STRIDE-th EEP point, BINARY_SUBSAMPLE points per coarse segment,
    # each spread uniformly along the primary step (moment-matched Gaussian) when BINARY_SMEAR.
    # Without the spread the sheet is a set of ridges that stars with mmag errors fall between.
    BINARY_SUBSAMPLE = 1
    BINARY_SMEAR = True
    BINARY_STRIDE = 3
    # candidates of the lattice stage polished with the full model (find_start)
    SEARCH_POLISH = 5
    LIKELIHOOD_VERSION = "unbinned-eep-v1"

    def __init__(
        self,
        isochs_path: str | Path,
        *,
        magnitude: str = "Gaia_G_EDR3",
        magnitude_effl: float = 6390.7,
        color: tuple[str, str] = ("Gaia_BP_EDR3", "Gaia_RP_EDR3"),
        color_effl: tuple[float, float] = (5182.6, 7825.1),
        model: str = "MIST",
        Rv: float = 3.1,
        # Binary fraction (Offner et al. 2022 / ASteCA default)
        alpha: float = 0.09,
        beta: float = 0.94,
        loga_range: tuple[float, float],
        Av_range: tuple[float, float] = (0.0, 3.0),
        dm_mu: float,
        dm_sigma: float = 0.3,
        dm_range: tuple[float, float],
        M_met: int = 200,
        M_loga: int = 200,
    ) -> None:
        r"""Configure the forward model. Reads no isochrones -- :meth:`setup` does that.

        Parameters
        ----------
        isochs_path : str or Path
            Location of the theoretical isochrone files. Stored only; the read
            happens in :meth:`setup`. The files must carry an ``EEP`` column.
        magnitude : str, default "Gaia_G_EDR3"
            Name of the magnitude column in the isochrone files.
        magnitude_effl : float, default 6390.7
            Effective wavelength of that band **in Ångström**. Feeds the CCM89 /
            O'Donnell-94 law that turns :math:`A_V` into the band's extinction
            coefficient :math:`A_\lambda/A_V`, so an error here biases the
            fitted extinction rather than merely shifting a plot. The default is
            Gaia :math:`G` (EDR3).
        color : tuple of str, default ("Gaia_BP_EDR3", "Gaia_RP_EDR3")
            The two bands whose difference forms the colour, blue first.
        color_effl : tuple of float, default (5182.6, 7825.1)
            Their effective wavelengths **in Ångström**, same role and same
            ordering as `color`.
        model : str, default "MIST"
            Isochrone family label. Stored as ``self.model_name``.
        Rv : float, default 3.1
            Total-to-selective extinction ratio :math:`R_V = A_V/E(B-V)`,
            dimensionless. 3.1 is the diffuse-ISM average; sightlines through
            dense cloud material run higher. It is **fixed, not fitted**.
        alpha, beta : float, default 0.09 and 0.94
            Binary-fraction parameters in
            :math:`b_p = \alpha + \beta / (1 + 1.4/m)`, with :math:`m` in
            :math:`M_\odot` and the result clipped to :math:`[0, 1]`. The
            defaults are the Offner et al. (2022) relation, matching ASteCA, so
            the binary fraction **rises with mass** instead of being one number
            for the whole cluster. ``alpha = beta = 0`` fits single stars only.
        loga_range : tuple of float
            Prior bounds on :math:`\log_{10}(\mathrm{age}/\mathrm{yr})`. The node table
            holds the isochrones that bracket this range, so widening it after
            :meth:`setup` requires rebuilding the table.
        Av_range : tuple of float, default (0.0, 3.0)
            Prior bounds on :math:`A_V`, in **magnitudes**.
        dm_mu : float
            Prior mean of the distance modulus, in **magnitudes**.

            ⚠ **`loga_range`, `dm_mu` y `dm_range` son OBLIGATORIOS y no tienen default a
            propósito.** Hasta 2026-08-26 valían ``(6.0, 7.0)``, ``10.2`` y ``(9.5, 10.7)``,
            que son **la edad y el módulo de distancia de NGC 6383** horneados en una API
            general. Medido contra el censo Hunt & Reffert 2024: el cúmulo MEDIANO está a
            2.259 kpc, o sea DM = 11.77, **fuera** de ese rango, y el percentil 99.5 da
            14.54. Un ajuste sobre cualquier cúmulo que no fuera NGC 6383 se truncaba en el
            borde del prior y devolvía una respuesta segura y equivocada, sin decir nada.

            ``dm_sigma``, ``Av_range`` y ``Rv`` **sí** conservan default: la anchura del
            prior de DM es una elección de modelado, ``Av`` de 0 a 3 cubre la mayoría de
            las líneas de visión, y ``Rv = 3.1`` es el valor estándar del medio
            interestelar.

            Since 2026-09-22 it is **only a prior**. Until then it was also the reference
            frame of the precomputed Hess grid, of the Hess window and of the error kernel,
            and the posterior locked onto it; ``tests/test_isochrone.py`` now checks that the
            likelihood does not depend on it.
        dm_sigma : float, default 0.3
            Prior standard deviation of the distance modulus, in magnitudes.
        dm_range : tuple of float
            Hard truncation bounds on the distance modulus, in magnitudes.
        M_met, M_loga : int, default 200
            **Unused since 2026-09-22**, kept so existing calls do not break. They set the
            resolution of the precomputed regular Hess grid, which no longer exists: the
            isochrone is interpolated between the MIST nodes at every evaluation.

        Notes
        -----
        Nothing is validated or read here; every argument is simply stored.
        Failures from an unreadable `isochs_path`, an unknown band name, or a
        wavelength outside the CCM89 domain surface from :meth:`setup`.
        """
        self.isochs_path = Path(isochs_path)
        self.magnitude = magnitude
        self.magnitude_effl = magnitude_effl
        self.color = color
        self.color_effl = color_effl
        self.model_name = model
        self.Rv = Rv
        self.alpha = alpha
        self.beta = beta
        self.loga_range = loga_range
        self.Av_range = Av_range
        self.dm_mu = dm_mu
        self.dm_sigma = dm_sigma
        self.dm_range = dm_range
        self.M_met = M_met
        self.M_loga = M_loga

        self._isochs: MISTIsochrones | None = None
        self._obs_mag: np.ndarray | None = None
        self._obs_col: np.ndarray | None = None
        self._e_obs_mag: np.ndarray | None = None  # per-star errors that enter the likelihood
        self._e_obs_col: np.ndarray | None = None
        self._star_weights: np.ndarray | None = None
        self._e_mag_fn: Any = None  # callable: apparent_mag -> error (posterior_cmd)
        self._e_col_fn: Any = None
        self._e_mag_coef: np.ndarray | None = None  # same model, as coefficients (F(theta))
        self._N_obs: int | None = None
        self._mag_lim: float | None = None  # completeness cut: the faintest member
        self._box_area: float | None = None
        self._kG: float | None = None
        self._kBP: float | None = None
        self._kRP: float | None = None
        self._k_col1: float | None = None  # = kBP - kRP
        # node table: (n_Z, n_age, rows, n_eep); rows = mass, G, colour, then per q node G, colour
        self._nodes: np.ndarray | None = None
        self._node_logz: np.ndarray | None = None
        self._node_loga: np.ndarray | None = None
        self._nodes_tensor: Any = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(
        self,
        cluster_data: QTable,
        *,
        prob_threshold: float = 0.6,
        probability_column: str = COLUMNA_ISOCRONA,
        precompute_grid: bool = True,
        pms_column: str | None = None,
        pms_max: float = 0.5,
        ms_weight: float = 1.0,
    ) -> None:
        """Read the isochrones, select the members, and (optionally) build the node table.

        Parameters
        ----------
        cluster_data : QTable
            Must contain ``Gmag``, ``G_BPmag``, ``G_RPmag``, ``e_Gmag``,
            ``e_G_BPmag``/``e_G_RPmag`` (or ``e_BP_RP``), and
            ``probability_hdbscan``.
        prob_threshold : float
            Minimum membership probability.
        precompute_grid : bool
            ``False`` skips building the EEP node table; :meth:`build_grid` or
            :meth:`load_grid` does it later. (The name predates the 2026-09-22 rewrite.)
        pms_column : str, optional
            Column name with PMS probability (e.g. ``"pms_sagitta"``).
            Used together with ``ms_weight`` to upweight MS stars.
        pms_max : float
            Stars with ``pms_column >= pms_max`` are treated as PMS (weight 1).
            Stars below this threshold are treated as MS (weight ``ms_weight``).
        ms_weight : float
            Multiplicative weight :math:`\\omega_i` on the log-likelihood of MS stars.
            ``ms_weight=1`` (default) is the likelihood; any other value makes it a
            weighted pseudo-likelihood, which anchors the fit to the main sequence but
            whose posterior width is no longer calibrated.
        """
        col1_name, col2_name = self.color
        self._isochs = MISTIsochrones(
            self.isochs_path,
            magnitude_col=self.magnitude,
            color_col1=col1_name,
            color_col2=col2_name,
        )

        members = select_by_probability(cluster_data, probability_column, prob_threshold)
        self._N_obs = len(members)

        weights = np.ones(len(members), dtype=float)
        if pms_column is not None and pms_column in members.colnames and ms_weight != 1.0:
            pms_prob = np.asarray(members[pms_column], dtype=float)
            is_ms = ~(np.isfinite(pms_prob) & (pms_prob >= pms_max))
            weights[is_ms] = ms_weight
            print(
                f"[setup] Weighted likelihood: {int(is_ms.sum())} MS stars (w={ms_weight}) + "
                f"{int((~is_ms).sum())} PMS stars (w=1.0)."
            )
        self._star_weights = weights

        obs_mag = np.asarray(members["Gmag"], dtype=float)
        obs_col = np.asarray(members["G_BPmag"], dtype=float) - np.asarray(
            members["G_RPmag"], dtype=float
        )
        e_mag = np.asarray(members["e_Gmag"], dtype=float)
        if "e_BP_RP" in members.colnames:
            e_col = np.asarray(members["e_BP_RP"], dtype=float)
        else:
            e_col = np.hypot(
                np.asarray(members["e_G_BPmag"], dtype=float),
                np.asarray(members["e_G_RPmag"], dtype=float),
            )
        self._obs_mag, self._obs_col = obs_mag, obs_col
        self._e_mag_fn, self._e_col_fn = _fit_error_model(obs_mag, e_mag, e_col)
        self._e_mag_coef = _error_poly(obs_mag, e_mag)
        # a missing or non-positive per-star error is replaced by the error model at that star
        bad_m = ~(np.isfinite(e_mag) & (e_mag > 0))
        bad_c = ~(np.isfinite(e_col) & (e_col > 0))
        e_mag = np.where(bad_m, self._e_mag_fn(obs_mag), e_mag)
        e_col = np.where(bad_c, self._e_col_fn(obs_mag), e_col)
        self._e_obs_mag, self._e_obs_col = e_mag, e_col

        # Window, from the data only: the completeness cut is the faintest member (ASteCA's
        # ``cut_max_mag``); the field component is uniform over the members' bounding box.
        self._mag_lim = float(obs_mag.max())
        span_m = max(float(np.ptp(obs_mag)), 1e-3)
        span_c = max(float(np.ptp(obs_col)), 1e-3)
        self._box_area = span_m * span_c

        self._kG, self._kBP, self._kRP = self._compute_ext_coefs()
        self._k_col1 = self._kBP - self._kRP

        if precompute_grid:
            self._build_nodes()

    def _compute_ext_coefs(self) -> tuple[float, float, float]:
        kG = _ccm89(self.magnitude_effl, self.Rv)
        kBP = _ccm89(self.color_effl[0], self.Rv)
        kRP = _ccm89(self.color_effl[1], self.Rv)
        return kG, kBP, kRP

    # ------------------------------------------------------------------
    # EEP node table
    # ------------------------------------------------------------------

    def _build_nodes(self) -> None:
        """Stack every (Z, age) node that can bracket the prior on one EEP axis.

        Rows per node: initial mass, ``G``, colour, then ``G`` and colour of the unresolved
        pair at each mass-ratio node in ``_Q_NODES`` (companion mass clipped to the node's
        lowest mass, as the generators do). EEPs a node lacks are clamped to its first/last
        point, where the mass step is zero, so they carry no IMF weight.
        """
        iso = self._isochs
        if iso is None:
            raise RuntimeError("Call setup() first.")
        zs = np.asarray(iso._met_values, float)
        ages = np.asarray(iso._loga_values, float)
        lo, hi = self.loga_range
        a0 = int(np.clip(np.searchsorted(ages, lo, side="right") - 1, 0, len(ages) - 1))
        a1 = int(np.clip(np.searchsorted(ages, hi, side="left"), 0, len(ages) - 1))
        ages = ages[a0 : a1 + 1]
        if lo < ages[0] - 1e-9 or hi > ages[-1] + 1e-9:
            raise ValueError(
                f"loga_range {self.loga_range} is outside the isochrone ages "
                f"[{ages[0]}, {ages[-1]}] in {self.isochs_path}"
            )

        per_node = {}
        e_lo, e_hi = np.inf, -np.inf
        for z in zs:
            for a in ages:
                eep, mass, G, BP, RP = iso.get_isochrone_eep(float(z), float(a))
                ms = np.argsort(mass, kind="stable")
                rows = [mass, G, BP - RP]
                for q in _Q_NODES:
                    m2 = np.clip(q * mass, mass.min(), None)
                    G2, BP2, RP2 = (np.interp(m2, mass[ms], x[ms]) for x in (G, BP, RP))
                    rows += [
                        _mag_combine(G, G2),
                        _mag_combine(BP, BP2) - _mag_combine(RP, RP2),
                    ]
                per_node[(z, a)] = (eep, rows)
                e_lo, e_hi = min(e_lo, eep.min()), max(e_hi, eep.max())

        axis = np.arange(int(e_lo), int(e_hi) + 1, dtype=float)
        T = np.empty((len(zs), len(ages), 3 + 2 * len(_Q_NODES), len(axis)))
        for i, z in enumerate(zs):
            for j, a in enumerate(ages):
                eep, rows = per_node[(z, a)]
                for r, x in enumerate(rows):
                    T[i, j, r] = np.interp(axis, eep, x)  # clamps outside the node's EEPs
        if np.any(np.diff(T[:, :, 0], axis=-1) < 0):
            raise ValueError("initial mass decreases along EEP in some isochrone node")
        self._set_nodes(T, np.log10(zs), ages)
        print(f"  EEP node table: {T.shape} (Z x age x rows x EEP)")

    def _set_nodes(self, T: np.ndarray, logz: np.ndarray, loga: np.ndarray) -> None:
        import pytensor

        self._nodes, self._node_logz, self._node_loga = T, logz, loga
        # static shape: slicing rows of a shape-less shared variable gives JAX a traced length
        self._nodes_tensor = pytensor.shared(T, name="isochrone_nodes", shape=T.shape)

    @staticmethod
    def _bracket(nodes: np.ndarray, x: Any, xp: Any) -> tuple[Any, Any, Any]:
        """Lower index, upper index and weight of ``x`` between sorted ``nodes``."""
        n = len(nodes)
        if n == 1:
            zero = 0 if xp is np else xp.constant(0, dtype="int64")
            return zero, zero, x * 0.0
        if xp is np:
            k = int(np.clip(np.searchsorted(nodes, x, side="right") - 1, 0, n - 2))
            return k, k + 1, float(np.clip((x - nodes[k]) / (nodes[k + 1] - nodes[k]), 0, 1))
        nt = xp.constant(nodes)
        k = xp.clip(xp.sum(xp.le(nt, x)) - 1, 0, n - 2).astype("int64")
        w = xp.clip((x - nt[k]) / (nt[k + 1] - nt[k]), 0.0, 1.0)
        return k, k + 1, w

    def _interp_isochrone(self, met: Any, loga: Any, xp: Any = np) -> Any:
        """Isochrone rows (mass, G, colour, q-node photometry) at ``(met, loga)``, at fixed EEP."""
        if self._nodes is None:
            raise RuntimeError("No node table: call setup() or build_grid() first.")
        T = self._nodes if xp is np else self._nodes_tensor
        i0, i1, wz = self._bracket(self._node_logz, xp.log10(met), xp)
        j0, j1, wa = self._bracket(self._node_loga, loga, xp)
        return (
            (1 - wz) * (1 - wa) * T[i0, j0]
            + wz * (1 - wa) * T[i1, j0]
            + (1 - wz) * wa * T[i0, j1]
            + wz * wa * T[i1, j1]
        )

    # ------------------------------------------------------------------
    # Likelihood
    # ------------------------------------------------------------------

    def _deposit(self, met: Any, loga: Any, dm: Any, Av: Any, xp: Any = np) -> dict:
        """Segments and weights of the model population at the sampled parameters."""
        X = self._interp_isochrone(met, loga, xp)
        mass = xp.maximum(X[0], 1e-9)
        dmag, dcol = dm + self._kG * Av, self._k_col1 * Av
        G, col = X[1] + dmag, X[2] + dcol
        Gq, cq = X[3::2] + dmag, X[4::2] + dcol  # (n_q, n_eep)

        # IMF weight per segment and per point (each segment split between its two ends)
        m_mid = 0.5 * (mass[1:] + mass[:-1])
        w_seg = _chabrier2014_xi(m_mid, xp) * (mass[1:] - mass[:-1])
        w_norm = xp.sum(w_seg)
        w_seg = w_seg / w_norm
        # Binaries: the pair's locus is a sheet over (primary mass, q). Along q it is deposited
        # exactly (segments between the q nodes); along the primary it is sampled at K points
        # per EEP segment, because a sheet sampled only at the EEP points is a set of ridges
        # 0.02-0.07 mag apart, which stars with mmag errors fall between (measured 2026-09-22:
        # with K = 1 the likelihood preferred a mode 0.11 mag off in dm on a synthetic cluster).
        K = self.BINARY_SUBSAMPLE
        t = (np.arange(K) + 0.5) / K
        n_e = self._nodes.shape[-1]
        idx = np.unique(np.r_[np.arange(0, n_e, self.BINARY_STRIDE), n_e - 1])

        def _sub(A: Any) -> Any:  # (..., n_coarse) -> (..., (n_coarse - 1) * K), linear in EEP
            a0, a1 = A[..., :-1], A[..., 1:]
            out = a0[..., None] + (a1 - a0)[..., None] * t
            return out.reshape((-1,)) if A.ndim == 1 else out.reshape((A.shape[0], -1))

        def _step(A: Any) -> Any:  # same shape as _sub(A): the sub-primary spacing
            d = (A[..., 1:] - A[..., :-1]) / K
            out = d[..., None] + 0.0 * t
            return out.reshape((-1,)) if A.ndim == 1 else out.reshape((A.shape[0], -1))

        # the binary sheet on a coarser EEP axis (every BINARY_STRIDE-th point), its IMF weight
        # per coarse segment normalised like the singles', so the two populations sum to one
        mass_c, Gq_c, cq_c = mass[idx], Gq[:, idx], cq[:, idx]
        w_c = _chabrier2014_xi(0.5 * (mass_c[1:] + mass_c[:-1]), xp) * (mass_c[1:] - mass_c[:-1])
        w_c = w_c / w_norm
        m_b = _sub(mass_c)
        b_sub = xp.clip(self.alpha + self.beta / (1.0 + 1.4 / m_b), 0.0, 1.0)
        w_rep = (w_c[:, None] * np.ones(K)).reshape((-1,)) / K
        b = xp.clip(self.alpha + self.beta / (1.0 + 1.4 / mass), 0.0, 1.0)

        # mass-ratio probabilities per q piece: q_{j+1}**(g+1) - q_j**(g+1), q_0 = 0
        g1 = _dk_gamma(m_b, xp, smooth=True) + 1.0
        cdf = [m_b * 0.0] + [m_b * 0.0 + 1.0 if q == 1.0 else q**g1 for q in _Q_NODES[1:]]
        pq = [cdf[k + 1] - cdf[k] for k in range(len(_Q_NODES) - 1)]
        return dict(
            G=G,
            col=col,
            Gq=_sub(Gq_c),
            cq=_sub(cq_c),
            # step between neighbouring sub-primaries along the primary direction, per q node
            vq=_step(Gq_c),
            vc=_step(cq_c),
            m_b=m_b,
            w_single=w_seg * (1 - 0.5 * (b[1:] + b[:-1])),
            w_bin=[w_rep * b_sub * p for p in pq],
        )

    def _star_loglike(
        self, met: Any, loga: Any, dm: Any, Av: Any, sigma_int: Any, f_bg: Any, xp: Any = np
    ) -> Any:
        """Per-star log-likelihood vector ``(N,)`` (weights :math:`\\omega_i` not applied)."""
        s2 = self.SIGMA_FLOOR**2 + sigma_int**2
        xg, xc = self._obs_mag, self._obs_col
        sg = xp.sqrt(self._e_obs_mag**2 + s2)
        sc = xp.sqrt(self._e_obs_col**2 + s2)
        single, binary = self._segments(met, loga, dm, Av, xp)
        Ag, Ac, Bg, Bc, c0, c1 = single
        J0, J1 = _segment_density(xg, xc, sg, sc, Ag, Ac, Bg, Bc, xp, first_moment=True)
        dens = xp.dot(J0, c0) + xp.dot(J1, c1)
        if binary is not None:
            Ag, Ac, Bg, Bc, w, vg, vc = binary
            dens = dens + xp.dot(
                _segment_density(xg, xc, sg, sc, Ag, Ac, Bg, Bc, xp, smear=(vg, vc)), w
            )
        F = self._detected_fraction(met, loga, dm, Av, sigma_int, xp, segs=(single, binary))
        return xp.log((1 - f_bg) * dens / F + f_bg / self._box_area)

    def _detected_fraction(
        self, met: Any, loga: Any, dm: Any, Av: Any, sigma_int: Any, xp: Any = np, segs: Any = None
    ) -> Any:
        r""":math:`F(\theta)`: the fraction of the model population observed brighter than the
        completeness cut -- each segment's :meth:`_p_observed` under its own weights (linear
        along the track for singles, uniform along q for the binary pieces)."""
        s2 = self.SIGMA_FLOOR**2 + sigma_int**2
        single, binary = self._segments(met, loga, dm, Av, xp) if segs is None else segs
        Ag, _, Bg, _, c0, c1 = single
        P, T = self._p_observed(Ag, Bg, s2, xp, first_moment=True)
        F = xp.sum(c0 * P + c1 * T)
        if binary is not None:
            Ag, _, Bg, _, w, _, _ = binary
            F = F + xp.sum(w * self._p_observed(Ag, Bg, s2, xp))
        return F

    def _segments(self, met: Any, loga: Any, dm: Any, Av: Any, xp: Any = np) -> tuple:
        r"""The model population as segments: ``(single, binary)``.

        ``single = (Ag, Ac, Bg, Bc, c0, c1)``: the isochrone between consecutive EEP points,
        with a density per unit length along the track that is **continuous**: linear within
        each segment between point values
        :math:`\rho_j = (w_{j-1} + w_j)/(\ell_{j-1} + \ell_j)` (IMF weights :math:`w`, CMD
        lengths :math:`\ell`), so segment :math:`k` carries :math:`c_{0,k} + c_{1,k}\,t` per
        unit :math:`t` with :math:`c_0 = \ell_k\rho_k`, :math:`c_1 = \ell_k(\rho_{k+1}-\rho_k)`,
        rescaled so the singles keep their total weight :math:`\sum_k w_k`.

        Why not :math:`w_k` spread uniformly, which stood here until 2026-09-24: MIST's EEP
        points are sparse where the isochrone moves fast (segments up to 1.7 mag long on
        NGC 6383's priors, per-length weights differing 2x between neighbours, points moving
        along the track at ~38 mag/dex in log t), so as log t changed a segment boundary swept
        across a clump of stars and each saw its density step. log L was rough in log t (second
        differences -0.83 to +1.03 at 0.0005 dex against a median -0.07) and NUTS mixed at
        ESS 167 on a synthetic cluster. Measured in the hub finding
        ``isochrone-nuts-convergence-2026-09.md`` §10.13.

        ``binary = (Ag, Ac, Bg, Bc, w, vg, vc)`` or ``None`` without binaries: the pieces along
        q, uniform, with their spread along the primary.
        """
        d = self._deposit(met, loga, dm, Av, xp)
        G, col, Gq, cq = d["G"], d["col"], d["Gq"], d["cq"]

        w = d["w_single"]
        # CMD length of each segment; the 1e-9 keeps the gradient finite on the zero-length
        # segments of EEPs clamped at a node's end (their weight is zero there)
        ell = xp.sqrt((G[1:] - G[:-1]) ** 2 + (col[1:] - col[:-1]) ** 2 + 1e-18)
        zero = w[:1] * 0.0
        wl = xp.concatenate([zero, w, zero])
        ll = xp.concatenate([zero, ell, zero])
        rho = (wl[:-1] + wl[1:]) / (ll[:-1] + ll[1:])
        c0 = ell * rho[:-1]
        c1 = ell * (rho[1:] - rho[:-1])
        scale = xp.sum(w) / xp.sum(c0 + 0.5 * c1)
        single = (G[:-1], col[:-1], G[1:], col[1:], c0 * scale, c1 * scale)
        if not (self.alpha or self.beta):  # decided at build time, not sampled
            return single, None

        # All binary pieces in ONE density call: a Python loop over q pieces multiplied the
        # symbolic graph, and PyTensor's rewrite of it took ~20 min before the first sample
        # (measured 2026-09-23).
        nq = len(d["w_bin"])
        on = 1.0 if self.BINARY_SMEAR else 0.0
        binary = (
            Gq[:-1].reshape((-1,)),
            cq[:-1].reshape((-1,)),
            Gq[1:].reshape((-1,)),
            cq[1:].reshape((-1,)),
            xp.stack(d["w_bin"]).reshape((-1,)),
            on * (0.5 * (d["vq"][:nq] + d["vq"][1:])).reshape((-1,)),
            on * (0.5 * (d["vc"][:nq] + d["vc"][1:])).reshape((-1,)),
        )
        return single, binary

    def _p_observed(self, A: Any, B: Any, s2: Any, xp: Any, first_moment: bool = False) -> Any:
        r"""Probability that a model star deposited uniformly along the segment from apparent
        magnitude ``A`` to ``B`` is observed brighter than the completeness cut :math:`G_{\lim}`.

        .. math::
           \bar P_k = \frac{1}{B_k - A_k}\int_{A_k}^{B_k} \Phi\!\left(\frac{G_{\lim}-G}{s_k}\right)
           dG = \frac{s_k}{B_k - A_k}\Big[\psi\big(\tfrac{G_{\lim}-A_k}{s_k}\big)
           - \psi\big(\tfrac{G_{\lim}-B_k}{s_k}\big)\Big],\qquad \psi(x) = x\,\Phi(x) + \varphi(x),

        with :math:`s_k^2 = e(\bar G_k)^2 + s^2`, the error model at the segment's midpoint
        (clipped to the observed range, because the quadratic log-error fit diverges when
        extrapolated). Zero-length segments use :math:`\Phi` at the point.

        With ``first_moment`` also :math:`T_k = \int_0^1 t\,\Phi(u_0 - \delta t)\,dt`
        (:math:`u_0 = (G_{\lim}-A_k)/s_k`, :math:`\delta = (B_k-A_k)/s_k`), for a weight linear
        along the segment: :math:`T = \delta^{-2}[u_0(\psi(u_0)-\psi(u_1)) - (\chi(u_0)-\chi(u_1))]`
        with :math:`\chi(u) = ((u^2-1)\Phi(u) + u\varphi(u))/2`, the antiderivative of
        :math:`u\Phi(u)`; for :math:`|\delta| < 0.1`, where that difference cancels, the
        expansion :math:`\bar P/2 - \delta\,\varphi(u_{\rm mid})/12`. Returns ``(P, T)``.

        Why the integral and not :math:`\Phi` at the midpoint, which stood here until
        2026-09-24: with the midpoint, a whole segment's IMF weight switches on within about
        :math:`\pm s` of shift as its midpoint crosses the cut, so :math:`N \ln F` -- the same
        step for every star -- jumps once per segment length.
        """
        c = self._e_mag_coef
        mid = 0.5 * (A + B)
        Gc = xp.clip(mid, float(self._obs_mag.min()), float(self._obs_mag.max()))
        e = 10.0 ** (c[0] + c[1] * Gc + c[2] * Gc**2)
        s = xp.sqrt(e**2 + s2)
        erfc = xp.erfc if xp is not np else _erfc_np

        def Phi(x: Any) -> Any:
            return 0.5 * erfc(-x / np.sqrt(2.0))

        def phi(x: Any) -> Any:
            return xp.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)

        def psi(x: Any) -> Any:  # antiderivative of Phi
            return x * Phi(x) + phi(x)

        L = self._mag_lim
        d = B - A
        short = xp.abs(d) < 1e-6
        D = xp.where(short, 1.0, d)  # safe value: no NaN in the branch not taken, nor its grad
        u0, u1 = (L - A) / s, (L - B) / s
        P = xp.where(short, Phi((L - mid) / s), s / D * (psi(u0) - psi(u1)))
        if not first_moment:
            return P

        def chi(x: Any) -> Any:  # antiderivative of x Phi(x)
            return 0.5 * ((x**2 - 1.0) * Phi(x) + x * phi(x))

        delta = d / s
        small = xp.abs(delta) < 0.1
        dl = xp.where(small, 1.0, delta)
        exact = (u0 * (psi(u0) - psi(u1)) - (chi(u0) - chi(u1))) / dl**2
        T = xp.where(small, 0.5 * P - delta * phi((L - mid) / s) / 12.0, exact)
        return P, T

    def _compiled_loglike(self, mode: str | None = None) -> Any:
        """Compiled ``(log L, d log L / d params)`` of the six parameters, for optimisation."""
        import importlib.util

        import pytensor
        import pytensor.tensor as pt

        v = [pt.dscalar(n) for n in ("met", "loga", "dm", "Av", "sigma_int", "f_bg")]
        ll = pt.sum(self._star_weights * self._star_loglike(*v, pt))
        # JAX only when asked: initialising it in a process that later forks (PyMC's
        # multiprocess sampler) raises JAX's fork warning, an error under a strict policy
        if mode == "JAX" and importlib.util.find_spec("jax") is None:
            mode = None
        fn = pytensor.function(v, [ll, *pytensor.grad(ll, v)], mode=mode)
        return lambda x: (lambda out: (float(out[0]), np.array(out[1:], float)))(fn(*x))

    def _seeded_inverse_mass(self, model: Any, start_info: dict) -> np.ndarray:
        r"""Diagonal initial inverse mass matrix in the sampler's unconstrained space.

        For an interval-bounded parameter :math:`x \in (a, b)` sampled as
        :math:`y = \mathrm{logit}((x-a)/(b-a))`, the local standard deviation :math:`s_x` at the
        mode :math:`x^*` maps to :math:`s_y = s_x (b-a)/((x^*-a)(b-x^*))`; the entry is
        :math:`s_y^2`. ``sigma_int`` and ``f_bg`` get 1. Why: the physical parameters' local
        widths are 1e-5 to 3e-4 in that space and the nuisance ones order 1; from an identity
        matrix, warmup spanned both scales with saturated trees. Measured 2026-09-23 on a binary
        synthetic cluster, 100 warmup iterations: 42 118 leapfrog steps (median tree 263, 90th
        percentile 1023) unseeded, 2 426 (median 15) seeded. Adaptation continues from it.
        """
        bounds = {
            "met": (10.0 ** float(self._node_logz[0]), 10.0 ** float(self._node_logz[-1])),
            "loga": tuple(self.loga_range),
            "dm": tuple(self.dm_range),
            "Av": tuple(self.Av_range),
        }
        diag = []
        for rv in model.free_RVs:
            if rv.name in bounds and rv.name in start_info["local_sd"]:
                a, b = bounds[rv.name]
                x = float(
                    np.clip(start_info["mode"][rv.name], a + 1e-9 * (b - a), b - 1e-9 * (b - a))
                )
                sd = float(start_info["local_sd"][rv.name])
                diag.append((sd * (b - a) / ((x - a) * (b - x))) ** 2)
            else:
                diag.append(1.0)
        return np.asarray(diag, dtype=float)

    def find_start(
        self, n_chains: int = 4, rng: np.random.Generator | None = None, mode: str | None = None
    ) -> dict:
        """Locate the dominant likelihood mode and draw chain starts around it.

        1. Every (Z, age) node inside the priors, with ``(dm, A_V)`` optimised at a wide
           intrinsic width (0.05 mag), where the likelihood is smooth.
        2. The ``SEARCH_POLISH`` best nodes are polished (L-BFGS-B, bounded) on all six parameters of the
           full model; the best is the mode.
        3. Local standard deviations from the curvature of log L at the mode; each chain
           starts at the mode plus a uniform offset of up to 3 of them (clipped to the prior).

        Returns a dict with ``mode``, ``loglike``, ``runner_up`` (the other polished maxima,
        with their log L), ``local_sd`` and ``chain_starts`` (list of dicts for ``initvals``).
        """
        from scipy.optimize import minimize

        rng = np.random.default_rng() if rng is None else rng
        if self._nodes is None:
            raise RuntimeError("Call setup() or build_grid() first.")
        zlo, zhi = 10 ** float(self._node_logz[0]), 10 ** float(self._node_logz[-1])
        lo = np.array(
            [zlo * (1 + 1e-9), self.loga_range[0], self.dm_range[0], self.Av_range[0], 1e-4, 1e-4]
        )
        hi = np.array(
            [zhi * (1 - 1e-9), self.loga_range[1], self.dm_range[1], self.Av_range[1], 0.5, 0.5]
        )
        if len(self._node_logz) == 1:
            lo[0] = hi[0] = zlo

        # Stage 1 uses the full model. A single-star proxy was tried on 2026-09-23 to save time
        # and put binary seed 1 on a local maximum 14.6 log-units below the full-model one
        # (-88.34 vs -73.75); the time it saved was not the bottleneck (sampling is).
        f1 = self._compiled_loglike(mode)
        cands = []
        ages = [a for a in self._node_loga if lo[1] <= a <= hi[1]] or [
            float(np.mean(self.loga_range))
        ]
        for z in np.clip(10.0**self._node_logz, lo[0], hi[0]):
            for a in ages:

                def nll2(y: np.ndarray, z: float = float(z), a: float = float(a)) -> tuple:
                    v, g = f1([z, a, y[0], y[1], 0.05, 0.02])
                    return -v, -g[2:4]

                y0 = [self.dm_mu, float(np.mean(self.Av_range))]
                r = minimize(
                    nll2,
                    y0,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=list(zip(lo[2:4], hi[2:4], strict=True)),
                )
                cands.append((float(r.fun), float(z), float(a), *map(float, r.x)))
        cands.sort(key=lambda c: c[0])

        f6 = self._compiled_loglike(mode)

        def nll6(y: np.ndarray) -> tuple:
            v, g = f6(y)
            return -v, -g

        polished = []
        for c in cands[: self.SEARCH_POLISH]:
            y0 = np.clip(np.array([*c[1:], 0.03, 0.02]), lo, hi)
            r = minimize(
                nll6, y0, jac=True, method="L-BFGS-B", bounds=list(zip(lo, hi, strict=True))
            )
            polished.append((-float(r.fun), np.asarray(r.x, float)))
        polished.sort(key=lambda p: -p[0])
        best_ll, best = polished[0]

        sd = np.empty(4)
        for k in range(4):
            h = 1e-4 * max(abs(best[k]), 1e-3)
            yp, ym = best.copy(), best.copy()
            yp[k] += h
            ym[k] -= h
            curv = -(f6(yp)[1][k] - f6(ym)[1][k]) / (2 * h)  # -d2 logL / dx2
            sd[k] = 1 / np.sqrt(curv) if curv > 0 else 0.1 * (hi[k] - lo[k])
        names = ("met", "loga", "dm", "Av")
        starts = []
        for _ in range(n_chains):
            x = np.clip(best[:4] + rng.uniform(-3, 3, 4) * sd, lo[:4], hi[:4])
            st = {n: float(v) for n, v in zip(names, x, strict=True)}
            st["sigma_int"] = float(max(best[4], 1e-3) * rng.uniform(0.7, 1.4))
            st["f_bg"] = float(np.clip(best[5], 1e-3, 0.5) * rng.uniform(0.7, 1.4))
            if len(self._node_logz) == 1:
                st.pop("met")
            starts.append(st)
        return {
            "mode": dict(zip((*names, "sigma_int", "f_bg"), best.tolist(), strict=True)),
            "loglike": float(best_ll),
            "local_sd": dict(zip(names, sd.tolist(), strict=True)),
            "runner_up": [(float(ll), x.tolist()) for ll, x in polished[1:]],
            "chain_starts": starts,
        }

    def loglike(
        self,
        met: float,
        loga: float,
        dm: float,
        Av: float,
        sigma_int: float = 0.0,
        f_bg: float = 0.0,
    ) -> float:
        """Total (weighted) log-likelihood at fixed parameters, in NumPy -- for profiles,
        diagnostics and tests; the sampler uses the PyTensor graph of the same function."""
        v = self._star_loglike(met, loga, dm, Av, sigma_int, f_bg, np)
        return float(np.sum(self._star_weights * v))

    # ------------------------------------------------------------------
    # Prior configuration
    # ------------------------------------------------------------------

    def set_priors(self, priors: dict) -> None:
        """Update prior parameters before building the model.

        Parameters
        ----------
        priors : dict
            Any subset of: ``dm_mu``, ``dm_sigma``, ``dm_range``,
            ``loga_range``, ``Av_range``.
        """
        valid = {"dm_mu", "dm_sigma", "dm_range", "loga_range", "Av_range"}
        unknown = set(priors) - valid
        if unknown:
            raise ValueError(f"Unknown prior keys: {unknown!r}. Valid: {valid}")
        for k, v in priors.items():
            setattr(self, k, v)

    # ------------------------------------------------------------------
    # Node table: build / cache
    # ------------------------------------------------------------------

    def build_grid(
        self,
        M_met: int | None = None,
        M_loga: int | None = None,
        *,
        grid_cache: str | Path | None = None,
    ) -> None:
        """Build (or load from cache) the EEP node table.

        Call this after ``setup(precompute_grid=False)`` and (optionally) ``set_priors()``.
        ``M_met`` / ``M_loga`` are accepted and ignored (see :meth:`__init__`).
        """
        if grid_cache is not None and Path(grid_cache).exists():
            print(f"Loading isochrone node table from cache: {grid_cache}")
            self.load_grid(grid_cache)
        else:
            self._build_nodes()
            if grid_cache is not None:
                self.save_grid(grid_cache)
                print(f"Isochrone node table saved to: {grid_cache}")

    def save_grid(self, path: str | Path) -> None:
        if self._nodes is None:
            raise RuntimeError("No node table to save. Run setup() first.")
        np.savez_compressed(
            path,
            likelihood=np.array(self.LIKELIHOOD_VERSION),
            nodes=self._nodes,
            node_logz=self._node_logz,
            node_loga=self._node_loga,
            q_nodes=_Q_NODES,
        )

    def load_grid(self, path: str | Path) -> None:
        d = np.load(path)
        if "likelihood" not in d.files or str(d["likelihood"]) != self.LIKELIHOOD_VERSION:
            raise ValueError(
                f"{path} is a precomputed Hess grid from before 2026-09-22, whose likelihood "
                "depended on the grid's internal reference and locked the posterior onto it. "
                "It cannot be reused; rebuild the node table with build_grid()."
            )
        if not np.array_equal(d["q_nodes"], _Q_NODES):
            raise ValueError(f"{path} was built with other mass-ratio nodes; rebuild it.")
        self._set_nodes(d["nodes"], d["node_logz"], d["node_loga"])

    # ------------------------------------------------------------------
    # PyMC model
    # ------------------------------------------------------------------

    def build_model(self) -> Any:
        pm = _require_pymc()
        import pytensor.tensor as pt

        if self._nodes is None or self._obs_mag is None:
            raise RuntimeError(
                "Call setup() (or setup(precompute_grid=False) + build_grid()/load_grid()) first."
            )
        loga_min, loga_max = self.loga_range
        if loga_min < self._node_loga[0] - 1e-9 or loga_max > self._node_loga[-1] + 1e-9:
            raise ValueError(
                f"loga_range {self.loga_range} reaches beyond the node table "
                f"[{self._node_loga[0]}, {self._node_loga[-1]}]; rebuild it with build_grid()."
            )
        met_min, met_max = 10.0 ** float(self._node_logz[0]), 10.0 ** float(self._node_logz[-1])
        weights = self._star_weights

        with pm.Model() as model:
            if len(self._node_logz) > 1:
                met = pm.Uniform("met", lower=met_min, upper=met_max)
            else:  # one metallicity file: Uniform(Z, Z) has no density, so Z is fixed
                met = pm.Deterministic("met", pt.as_tensor_variable(met_min))
            loga = pm.Uniform("loga", lower=loga_min, upper=loga_max)
            dm = pm.TruncatedNormal(
                "dm",
                mu=self.dm_mu,
                sigma=self.dm_sigma,
                lower=self.dm_range[0],
                upper=self.dm_range[1],
            )
            Av = pm.Uniform("Av", lower=self.Av_range[0], upper=self.Av_range[1])
            # intrinsic width: unmodelled spread (differential reddening, rotation, model error)
            sigma_int = pm.HalfNormal("sigma_int", sigma=0.05)
            # field fraction among the selected members, uniform over the CMD box
            f_bg = pm.Beta("f_bg", alpha=1.0, beta=19.0)

            def _logp(value: Any, met: Any, loga: Any, dm: Any, Av: Any, s: Any, f: Any) -> Any:
                return weights * self._star_loglike(met, loga, dm, Av, s, f, pt)

            pm.CustomDist(
                "y",
                met,
                loga,
                dm,
                Av,
                sigma_int,
                f_bg,
                logp=_logp,
                observed=np.zeros(self._N_obs),
            )
        return model

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def fit(
        self,
        *,
        draws: int = 2000,
        tune: int = 1000,
        chains: int | None = None,
        cores: int | None = None,
        target_accept: float = 0.8,
        nuts_sampler: str = "pymc",
        random_seed: int | None = None,
        progressbar: bool = True,
        # Initialisation
        init: str = "auto",
        initvals: dict | None = None,
        # Posterior log-likelihood (needed for LOO / WAIC)
        log_likelihood: bool = False,
        # Extra kwargs forwarded verbatim to the chosen NUTS backend
        nuts_sampler_kwargs: dict | None = None,
        start: str = "search",
        start_info: dict | None = None,
    ) -> Any:
        """Run NUTS sampling and return ArviZ InferenceData.

        Parameters
        ----------
        draws : int
            Number of posterior samples per chain.
        tune : int
            Number of tuning (warm-up) steps.  Increase to 2000+ for
            complex posteriors; blackjax window-adaptation uses this budget.
        chains : int or None
            Number of independent chains.  ``None`` delegates to PyMC's
            default: ``max(2, cores)``.  Use ≥ 4 for reliable R̂ / ESS.
        cores : int or None
            Number of parallel processes.  ``None`` delegates to PyMC's
            default: ``min(4, cpu_count())``.
        target_accept : float
            Dual-averaging target acceptance rate.  PyMC default is 0.8.
            Raise to 0.9–0.99 if you see divergences.
        nuts_sampler : str
            ``"blackjax"`` (JAX/GPU-friendly) or ``"pymc"``
            (default PyMC implementation).
        init : str
            Initialisation strategy passed to ``pm.sample``.  Options:
            ``"auto"`` (default), ``"advi"``, ``"advi+adapt_diag"``,
            ``"map"``, ``"adapt_diag"``, ``"adapt_full"``.
            ``"advi"`` finds a variational approximation first and uses it
            to seed the chains — helps on difficult posteriors.
        initvals : dict, optional
            Explicit starting values, e.g. from ``pm.find_MAP()``.
            Overrides ``init`` for the initial point.
        log_likelihood : bool
            Store the pointwise log-likelihood in ``idata``.  Required for
            ``az.loo()`` / ``az.waic()`` model comparison.  Adds memory and
            post-processing time proportional to ``draws × N_members``.
        nuts_sampler_kwargs : dict, optional
            Extra keyword arguments forwarded verbatim to the backend sampler.
            For blackjax: ``{"step_size": 0.01}`` to override the initial
            step size.
        start : {"search", "prior"}, default "search"
            ``"search"`` starts the chains at ``initvals`` if given, else runs
            :meth:`find_start` (the likelihood maximum plus a uniform offset of up to 3 local
            standard deviations per parameter); either way with PyMC's jitter off, and with
            the initial inverse mass matrix seeded from the local standard deviations at the
            mode (see :meth:`_seeded_inverse_mass`). ``start_info`` passes a
            :meth:`find_start` result already computed.
            ``"prior"`` is PyMC's default (initial point plus uniform jitter in [-1, 1] in the
            transformed space).

            Why the default is not PyMC's: the per-star likelihood is sharp, and away from
            the dominant mode it has local maxima in (Z, log t). Measured 2026-09-22 on a
            synthetic cluster (Z 0.0143, log t 6.55, dm 10.47, A_V 1.10): a chain from the
            prior centre settled at Z 0.0237, log t 6.886, dm 10.37, A_V 1.34 with a field
            fraction of 0.12, 242 log-units below the truth. **So with ``"search"`` R-hat
            certifies mixing within the mode the search found, not that no other mode
            exists**; :meth:`find_start` returns the runner-up modes so that can be checked.
        """
        pm = _require_pymc()

        model = self.build_model()
        step = None
        if start == "search":
            n_chains = chains if chains is not None else 4
            jax_sampler = nuts_sampler in ("numpyro", "blackjax")
            if start_info is None:
                rng = np.random.default_rng(random_seed)
                start_info = self.find_start(n_chains, rng, mode="JAX" if jax_sampler else None)
            if initvals is None:
                initvals = start_info["chain_starts"]
            inv_mass = self._seeded_inverse_mass(model, start_info)
            if jax_sampler:
                from pymc.sampling.jax import sample_jax_nuts

                user = dict(nuts_sampler_kwargs or {})
                seed = np.diag(inv_mass) if user.get("dense_mass") else inv_mass
                kw = {"inverse_mass_matrix": seed, **user}
                with model:
                    return sample_jax_nuts(
                        draws=draws,
                        tune=tune,
                        chains=n_chains,
                        target_accept=target_accept,
                        random_seed=random_seed,
                        initvals=initvals,
                        jitter=False,
                        progressbar=progressbar,
                        nuts_sampler=nuts_sampler,
                        nuts_kwargs=kw,
                        idata_kwargs={"log_likelihood": log_likelihood},
                    )
            # PyMC's own NUTS, seeded the same way (and still adapting); no jitter on the starts
            from pymc.step_methods.hmc.quadpotential import QuadPotentialDiagAdapt

            with model:
                ip = model.initial_point()
                ip.update(
                    {
                        vv.name: model.rvs_to_transforms[rv]
                        .forward(initvals[0][rv.name], *rv.owner.inputs)
                        .eval()
                        for rv, vv in zip(model.free_RVs, model.value_vars, strict=True)
                        if rv.name in initvals[0]
                    }
                )
                mean = np.concatenate([np.atleast_1d(ip[v.name]) for v in model.value_vars])
                pot = QuadPotentialDiagAdapt(len(inv_mass), mean, inv_mass, 10)
                step = pm.NUTS(potential=pot, target_accept=target_accept)
        with model:
            # with our own step, target_accept already lives in it (pm.sample rejects it twice)
            extra = {} if step is not None else {"target_accept": target_accept, "init": init}
            if step is None and nuts_sampler_kwargs:
                extra["nuts_sampler_kwargs"] = nuts_sampler_kwargs
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                step=step,
                cores=cores,
                nuts_sampler=nuts_sampler,
                random_seed=random_seed,
                progressbar=progressbar,
                initvals=initvals,
                idata_kwargs={"log_likelihood": log_likelihood},
                **extra,
            )
        return idata

    # ------------------------------------------------------------------
    # Posterior utilities
    # ------------------------------------------------------------------

    def posterior_cmd(
        self, idata: Any, *, num_samples: int = 20, rng: np.random.Generator | None = None
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Generate synthetic CMDs drawn from the posterior, with the likelihood's forward model.

        Each draw samples ``N_members`` stars from the EEP-interpolated isochrone at a posterior
        sample: mass uniformly within IMF-weighted segments, binaries with the Offner fraction and
        a D&K mass ratio (interpolated between the same mass-ratio nodes the likelihood uses),
        errors from the fitted error model, and the completeness cut.

        Returns
        -------
        obs_mag, obs_col : np.ndarray
        cmds : list of (mag_arr, col_arr)
            Access: ``syn = cmds[i]; mag = syn[0]; col = syn[1]``.
        """
        import arviz as az

        rng = np.random.default_rng() if rng is None else rng
        ds = az.extract(idata, num_samples=num_samples, var_names=["met", "loga", "dm", "Av"])
        cols = [np.asarray(ds[v]).ravel() for v in ("met", "loga", "dm", "Av")]
        sint = (
            np.asarray(az.extract(idata, num_samples=num_samples, var_names=["sigma_int"])).ravel()
            if "sigma_int" in idata.posterior
            else np.zeros(len(cols[0]))
        )
        N_out = self._N_obs if self._N_obs else 500
        cmds: list[tuple[np.ndarray, np.ndarray]] = []
        for (met_v, loga_v, dm_v, Av_v), s_v in zip(zip(*cols, strict=True), sint, strict=True):
            mag_s, col_s = self._draw_stars(
                float(met_v), float(loga_v), float(dm_v), float(Av_v), float(s_v), N_out, rng
            )
            cmds.append((mag_s, col_s))
        return self._obs_mag, self._obs_col, cmds  # type: ignore[return-value]

    def _draw_stars(
        self,
        met: float,
        loga: float,
        dm: float,
        Av: float,
        sigma_int: float,
        n: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """``n`` observed stars (after the completeness cut) from the forward model."""
        d = self._deposit(met, loga, dm, Av, np)
        g1 = _dk_gamma(d["m_b"]) + 1.0
        w_single = d["w_single"]
        w_pt_bin = sum(d["w_bin"])  # weight of each binary sub-primary, all q
        p = np.concatenate([w_single, w_pt_bin])
        p = np.clip(p, 0, None) / np.clip(p, 0, None).sum()
        out_m, out_c = [], []
        s2 = self.SIGMA_FLOOR**2 + sigma_int**2
        for _ in range(100):  # a model entirely beyond the cut returns what it has, not a hang
            if sum(len(x) for x in out_m) >= n:
                break
            k = rng.choice(len(p), size=4 * n, p=p)
            single = k < len(w_single)
            G = np.empty(k.size)
            C = np.empty(k.size)
            ks, t = k[single], rng.uniform(size=int(single.sum()))
            G[single] = d["G"][ks] + t * (d["G"][ks + 1] - d["G"][ks])
            C[single] = d["col"][ks] + t * (d["col"][ks + 1] - d["col"][ks])
            kb = k[~single] - len(w_single)
            q = rng.uniform(size=kb.size) ** (1.0 / g1[kb])
            j = np.clip(np.searchsorted(_Q_NODES, q, side="right") - 1, 0, len(_Q_NODES) - 2)
            f = (q - _Q_NODES[j]) / (_Q_NODES[j + 1] - _Q_NODES[j])
            G[~single] = d["Gq"][j, kb] + f * (d["Gq"][j + 1, kb] - d["Gq"][j, kb])
            C[~single] = d["cq"][j, kb] + f * (d["cq"][j + 1, kb] - d["cq"][j, kb])
            eg = np.sqrt(
                self._e_mag_fn(np.clip(G, self._obs_mag.min(), self._obs_mag.max())) ** 2 + s2
            )
            ec = np.sqrt(
                self._e_col_fn(np.clip(G, self._obs_mag.min(), self._obs_mag.max())) ** 2 + s2
            )
            Go = G + rng.normal(size=G.size) * eg
            Co = C + rng.normal(size=G.size) * ec
            keep = Go <= self._mag_lim
            out_m.append(Go[keep])
            out_c.append(Co[keep])
        return np.concatenate(out_m)[:n], np.concatenate(out_c)[:n]

    def _median_isochrone(self, idata: Any) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        """Posterior-median parameters and the apparent single-star isochrone there."""
        import arviz as az

        post = az.extract(idata, var_names=["met", "loga", "dm", "Av"])
        med = {
            v: float(np.median(np.asarray(post[v]).ravel())) for v in ["met", "loga", "dm", "Av"]
        }
        X = self._interp_isochrone(med["met"], med["loga"], np)
        G_app = X[1] + med["dm"] + self._kG * med["Av"]  # type: ignore[operator]
        col_app = X[2] + self._k_col1 * med["Av"]  # type: ignore[operator]
        return med, X[0], G_app, col_app

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------

    def cmd_distances(
        self,
        idata: Any,
        cluster_table: QTable,
        *,
        prob_threshold: float = 0.6,
        probability_column: str = COLUMNA_ISOCRONA,
        pms_column: str | None = "pms_sagitta",
    ) -> Any:
        """For each cluster member compute its distance to the posterior median isochrone.

        PMS stars that fall close to the PMS portion of the isochrone AND have
        high ``pms_sagitta`` probability are the most physically self-consistent
        PMS candidates.

        Parameters
        ----------
        idata : arviz.InferenceData
            Posterior from ``fit()``.
        cluster_table : QTable
            Full cluster member table (same one passed to ``setup()``).
        prob_threshold : float
            Membership probability cut.
        pms_column : str or None
            Column with PMS probability to include in the output.

        Returns
        -------
        pandas.DataFrame
            One row per member star with columns:

            - ``source_id``
            - ``Gmag``, ``BP_RP`` (observed)
            - ``iso_Gmag``, ``iso_BP_RP`` (nearest isochrone point)
            - ``d_cmd``  — Euclidean distance in CMD space (mag)
            - ``delta_col`` — observed BP-RP minus isochrone BP-RP at same G
                             (positive = redder than MS → PMS side)
            - ``mass_est`` — primary mass of nearest isochrone point (M☉)
            - ``pms_prob`` — pms_sagitta value (NaN if column absent)
            - ``pms_score`` — combined score = pms_prob / (1 + d_cmd)
                              (higher → better PMS candidate)
        """
        import pandas as pd

        # Apparent isochrone at the posterior median (EEP-interpolated, as in the likelihood)
        _, mass, G_app, col_app = self._median_isochrone(idata)

        # Observed members
        members = select_by_probability(cluster_table, probability_column, prob_threshold)
        obs_g = np.asarray(members["Gmag"], dtype=float)
        obs_col = np.asarray(members["G_BPmag"], dtype=float) - np.asarray(
            members["G_RPmag"], dtype=float
        )

        # Normalise CMD space by typical observed spreads so both axes contribute equally
        sig_g = float(np.nanstd(obs_g)) or 1.0
        sig_col = float(np.nanstd(obs_col)) or 1.0

        iso_g_n = G_app / sig_g
        iso_col_n = col_app / sig_col

        rows = []
        for i in range(len(members)):
            g_n = obs_g[i] / sig_g
            c_n = obs_col[i] / sig_col
            dists = np.hypot(iso_g_n - g_n, iso_col_n - c_n)
            j = int(np.argmin(dists))

            d_cmd = float(np.hypot(G_app[j] - obs_g[i], col_app[j] - obs_col[i]))
            delta_col = float(obs_col[i] - col_app[j])  # + = redder than MS

            pms_prob = float("nan")
            if pms_column and pms_column in members.colnames:
                val = members[pms_column][i]
                pms_prob = float(val) if np.isfinite(float(val)) else float("nan")

            pms_score = pms_prob / (1.0 + d_cmd) if np.isfinite(pms_prob) else float("nan")

            rows.append(
                dict(
                    source_id=int(members["source_id"][i]),
                    Gmag=float(obs_g[i]),
                    BP_RP=float(obs_col[i]),
                    iso_Gmag=float(G_app[j]),
                    iso_BP_RP=float(col_app[j]),
                    mass_est=float(mass[j]),
                    d_cmd=d_cmd,
                    delta_col=delta_col,
                    pms_prob=pms_prob,
                    pms_score=pms_score,
                )
            )

        df = pd.DataFrame(rows)
        df.sort_values("pms_score", ascending=False, inplace=True, ignore_index=True)
        return df

    def plot_pms_selection(
        self,
        idata: Any,
        cluster_table: QTable,
        *,
        prob_threshold: float = 0.6,
        pms_column: str | None = "pms_sagitta",
        pms_threshold: float = 0.5,
        top_n: int | None = None,
        figsize: tuple[float, float] = (7, 8),
        save: str | None = None,
    ) -> Any:
        """CMD plot highlighting PMS stars by their isochrone-fit quality.

        Stars are colour-coded by ``pms_score`` (pms_prob / (1 + d_cmd)).
        The posterior median isochrone is overlaid.

        Parameters
        ----------
        top_n : int, optional
            If given, circle the top-N PMS candidates.

        Returns
        -------
        pandas.DataFrame
            Same as :meth:`cmd_distances`, sorted by ``pms_score`` descending.
        """
        import matplotlib.colors as mcolors
        import matplotlib.pyplot as plt

        df = self.cmd_distances(
            idata,
            cluster_table,
            prob_threshold=prob_threshold,
            pms_column=pms_column,
        )

        _, _, G_app, col_app = self._median_isochrone(idata)
        cut = G_app < float(df["Gmag"].max()) + 0.5

        # Separate MS and PMS stars
        if pms_column and pms_column in df.columns:
            is_pms = df["pms_prob"] >= pms_threshold
        else:
            is_pms = df["delta_col"] > 0.1  # fallback: redder than isochrone

        ms_df = df[~is_pms]
        pms_df = df[is_pms]

        scores = pms_df["pms_score"].values
        finite = np.isfinite(scores)
        if finite.any():
            norm = mcolors.Normalize(
                vmin=float(np.nanmin(scores[finite])), vmax=float(np.nanmax(scores[finite]))
            )
        else:
            norm = mcolors.Normalize(0, 1)
        # plt.get_cmap, not matplotlib.cm.get_cmap: the latter was removed for good in matplotlib
        # 3.11.0 (removed in 3.9.0, re-added in 3.9.1, removed again in 3.11.0), while the
        # changelog explicitly preserves plt.get_cmap. This path has no test, so nothing here
        # would have caught it -- it was found by reading the 3.11 changelog against a grep.
        cmap = plt.get_cmap("plasma")

        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(
            ms_df["BP_RP"],
            ms_df["Gmag"],
            s=10,
            alpha=0.4,
            color="steelblue",
            label="MS members",
            zorder=2,
        )

        sc = ax.scatter(
            pms_df["BP_RP"],
            pms_df["Gmag"],
            c=pms_df["pms_score"],
            cmap=cmap,
            norm=norm,
            s=20,
            alpha=0.8,
            zorder=3,
            label="PMS members",
        )

        if top_n is not None:
            top = pms_df.head(top_n)
            ax.scatter(
                top["BP_RP"],
                top["Gmag"],
                s=80,
                facecolors="none",
                edgecolors="red",
                linewidths=1.5,
                zorder=5,
                label=f"Top {top_n} PMS",
            )

        ax.plot(col_app[cut], G_app[cut], color="k", lw=1.5, zorder=4, label="Median isochrone")

        ax.invert_yaxis()
        ax.set_xlabel("BP − RP")
        ax.set_ylabel("G")
        ax.legend(fontsize=9, framealpha=0.4)
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label("pms_score  (pms_prob / (1 + d_cmd))")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=200, bbox_inches="tight")
        plt.show()
        return df

    def plot_cmd(
        self,
        idata: Any,
        *,
        num_samples: int = 30,
        figsize: tuple[float, float] = (6, 7),
        obs_kw: dict | None = None,
        syn_kw: dict | None = None,
        median_kw: dict | None = None,
        save: str | None = None,
    ) -> None:
        """CMD with observed stars and posterior synthetic draws.

        Parameters
        ----------
        idata : arviz.InferenceData
            Posterior from ``fit()``.
        num_samples : int
            Number of posterior draws to overlay as faint scatter.
        save : str, optional
            File path to save the figure (e.g. ``"cmd.pdf"``).
        """
        import matplotlib.pyplot as plt

        obs_kw = {"s": 8, "alpha": 0.6, "color": "steelblue", "zorder": 3, **(obs_kw or {})}
        syn_kw = {"s": 3, "alpha": 0.15, "color": "tomato", "zorder": 2, **(syn_kw or {})}
        median_kw = {"s": 4, "alpha": 0.6, "color": "firebrick", "zorder": 4, **(median_kw or {})}

        obs_mag, obs_col, cmds = self.posterior_cmd(idata, num_samples=num_samples)

        # Median posterior isochrone (noiseless, no IMF sampling)
        _, _, G_med, col_med = self._median_isochrone(idata)
        cut = G_med < self._obs_mag.max() + 0.5  # type: ignore[union-attr]

        fig, ax = plt.subplots(figsize=figsize)

        for syn in cmds:
            ax.scatter(syn[1], syn[0], **syn_kw, label="_nolegend_")

        ax.scatter(obs_col, obs_mag, label="Observed", **obs_kw)
        ax.scatter(col_med[cut], G_med[cut], label="Median posterior", **median_kw)

        ax.invert_yaxis()
        ax.set_xlabel("BP − RP")
        ax.set_ylabel("G")
        ax.legend(fontsize=10, framealpha=0.4)
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=200, bbox_inches="tight")
        plt.show()

    def plot_hess(
        self,
        idata: Any,
        *,
        figsize: tuple[float, float] = (13, 4),
        cmap_hess: str = "Blues",
        cmap_residual: str = "RdBu_r",
        n_draw: int = 200_000,
        save: str | None = None,
    ) -> None:
        """Three-panel Hess diagram: observed | model at the posterior median | residuals.

        The likelihood is unbinned; this is a diagnostic only. The model panel histograms
        ``n_draw`` stars from the forward model (:meth:`_draw_stars`), scaled to the members.
        """
        import matplotlib.pyplot as plt
        from astropy.stats import knuth_bin_width

        if self._obs_mag is None or self._nodes is None:
            raise RuntimeError("Call setup() and build_grid() first.")
        med, _, _, _ = self._median_isochrone(idata)
        _, bm = knuth_bin_width(self._obs_mag, return_bins=True)
        _, bc = knuth_bin_width(self._obs_col, return_bins=True)
        H_obs, em, ec = np.histogram2d(self._obs_mag, self._obs_col, bins=[bm, bc])
        g, c = self._draw_stars(
            med["met"], med["loga"], med["dm"], med["Av"], 0.0, n_draw, np.random.default_rng(0)
        )
        H_syn, _, _ = np.histogram2d(g, c, bins=[em, ec])
        H_syn = H_syn * self._N_obs / n_draw
        residual = H_obs - H_syn
        vlim = np.nanpercentile(np.abs(residual), 98)

        fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)
        extent = [ec[0], ec[-1], em[-1], em[0]]
        ims = [
            axes[0].imshow(H_obs, extent=extent, aspect="auto", cmap=cmap_hess, origin="upper"),
            axes[1].imshow(H_syn, extent=extent, aspect="auto", cmap=cmap_hess, origin="upper"),
            axes[2].imshow(
                residual,
                extent=extent,
                aspect="auto",
                cmap=cmap_residual,
                origin="upper",
                vmin=-vlim,
                vmax=vlim,
            ),
        ]
        for ax, title, im in zip(
            axes, ["Observed", "Model @ median", "Obs − Model"], ims, strict=True
        ):
            ax.set_title(title)
            ax.set_xlabel("BP − RP")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[0].set_ylabel("G")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=200, bbox_inches="tight")
        plt.show()


__all__ = ["IsochroneFitter", "MISTIsochrones"]
