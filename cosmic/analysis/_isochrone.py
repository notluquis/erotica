"""Isochrone fitting: native MIST reader + PyMC NUTS.

Matches ASteCA's forward model:
  - Linear-Z metallicity (Zinit) as stored in MIST files
  - Chabrier (2014) IMF
  - Duchêne & Kraus (2013) binary fraction + mass-ratio distribution
  - Magnitude-sorted observational error model
  - Cardelli, Clayton & Mathis (1989) + O'Donnell (1994) extinction law
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from astropy.table import QTable
from scipy.ndimage import convolve, gaussian_filter


# ---------------------------------------------------------------------------
# CCM89 + O'Donnell 94  (A_lambda / A_V)
# ---------------------------------------------------------------------------

def _ccm89(lam_aa: float, Rv: float = 3.1) -> float:
    x = 1e4 / lam_aa          # Å → 1/μm
    if 1.1 <= x <= 3.3:
        y = x - 1.82
        a = (1 + 0.17699*y - 0.50447*y**2 - 0.02427*y**3
             + 0.72085*y**4 + 0.01979*y**5 - 0.77530*y**6 + 0.32999*y**7)
        b = (1.41338*y + 2.28305*y**2 + 1.07233*y**3 - 5.38434*y**4
             - 0.62251*y**5 + 5.30260*y**6 - 2.09002*y**7)
    elif 3.3 < x <= 8.0:
        Fa = Fb = 0.0
        if x >= 5.9:
            y = x - 5.9
            Fa = -0.04473*y**2 - 0.009779*y**3
            Fb =  0.2130*y**2  + 0.1207*y**3
        a = 1.752 - 0.316*x - 0.104 / ((x - 4.67)**2 + 0.341) + Fa
        b = -3.090 + 1.825*x + 1.206 / ((x - 4.62)**2 + 0.263) + Fb
    else:
        a =  0.574 * x**1.61
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
    log10_m  = np.log10(np.clip(mass, 1e-9, None))
    log10_mc = np.log10(mc)

    xi_ln = np.exp(-0.5 * (log10_m - log10_mc)**2 / sigma**2) / np.clip(mass, 1e-9, None)
    xi_pl = mass**(-2.35)

    # Scale power-law to match log-normal at m0 (continuity)
    xi_ln_at_m0 = np.exp(-0.5 * (np.log10(m0) - log10_mc)**2 / sigma**2) / m0
    xi_pl_at_m0 = m0**(-2.35)
    scale       = xi_ln_at_m0 / xi_pl_at_m0 if xi_pl_at_m0 > 0 else 1.0

    xi = np.where(mass < m0, xi_ln, xi_pl * scale)
    dm = np.abs(np.gradient(mass))
    w  = xi * dm
    w  = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    s  = w.sum()
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
        mass <= 0.1, 4.2,
        np.where(
            mass <= 0.6, 0.4,
            np.where(
                mass <= 1.4, 0.3,
                np.where(mass <= 6.5, -0.5, 0.0)
            )
        )
    )
    return (gamma + 1.0) / (gamma + 2.0)


# ---------------------------------------------------------------------------
# Combine two magnitudes into an unresolved pair
# ---------------------------------------------------------------------------

def _mag_combine(m1: np.ndarray, m2: np.ndarray) -> np.ndarray:
    return -2.5 * np.log10(10**(-0.4 * m1) + 10**(-0.4 * m2))


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
        self.color_col1    = color_col1
        self.color_col2    = color_col2
        self.mass_col      = mass_col
        self.loga_col      = loga_col

        # (Z, loga) → (mass, G_abs, BP_abs, RP_abs)
        self._grid: dict[
            tuple[float, float],
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        ] = {}
        self._met_values:  np.ndarray | None = None
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
                for next_line in lines[i + 1:]:
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
                    if colnames is None and any(t in tokens for t in [
                        self.magnitude_col, self.loga_col, "initial_mass", "EEP",
                    ]):
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
            warnings.warn(f"Cannot determine Zinit for {fpath.name}; assuming Z_sun.")
            Z = 0.0152
        Z = round(Z, 6)

        if colnames is None:
            warnings.warn(f"No column header in {fpath.name}; skipping.")
            return

        col_idx = {c: i for i, c in enumerate(colnames)}
        needed  = [self.loga_col, self.magnitude_col, self.color_col1, self.color_col2]
        if not all(c in col_idx for c in needed):
            missing = [c for c in needed if c not in col_idx]
            warnings.warn(f"Missing columns {missing} in {fpath.name}; skipping.")
            return

        i_loga = col_idx[self.loga_col]
        i_G    = col_idx[self.magnitude_col]
        i_BP   = col_idx[self.color_col1]
        i_RP   = col_idx[self.color_col2]
        i_mass = col_idx.get(self.mass_col)

        if not rows:
            return

        data     = np.array(rows, dtype=np.float64)
        loga_all = data[:, i_loga]
        G_all    = data[:, i_G]
        BP_all   = data[:, i_BP]
        RP_all   = data[:, i_RP]
        mass_all = data[:, i_mass] if i_mass is not None else np.ones(len(data))

        for loga_val in np.unique(np.round(loga_all, 4)):
            mask  = np.round(loga_all, 4) == loga_val
            G     = G_all[mask]
            BP    = BP_all[mask]
            RP    = RP_all[mask]
            mass  = mass_all[mask]
            valid = (
                np.isfinite(G) & np.isfinite(BP) & np.isfinite(RP)
                & np.isfinite(mass) & (mass > 0)
            )
            if valid.sum() < 2:
                continue
            sidx = np.argsort(mass[valid])
            self._grid[(Z, float(loga_val))] = (
                mass[valid][sidx],
                G[valid][sidx],
                BP[valid][sidx],
                RP[valid][sidx],
            )

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
        self._met_values  = np.array(sorted({k[0] for k in self._grid}))
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
        mi = int(np.argmin(np.abs(self._met_values  - met)))   # type: ignore[arg-type]
        ai = int(np.argmin(np.abs(self._loga_values - loga)))  # type: ignore[arg-type]
        key = (float(self._met_values[mi]), float(self._loga_values[ai]))  # type: ignore[index]
        return self._grid[key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _smooth2d(H: np.ndarray) -> np.ndarray:
    ker = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=float)
    ker /= ker.sum()
    return convolve(H.astype(float), ker, mode="constant", cval=0.0)


def _fit_error_model(
    obs_mag: np.ndarray, e_mag: np.ndarray, e_col: np.ndarray
) -> tuple[Any, Any]:
    """Fit quadratic log-error vs magnitude models for interpolation.

    Uses ``numpy.polynomial.Polynomial.fit`` (SVD-based, numerically stable)
    rather than the legacy ``np.polyfit`` which raises RankWarning on poorly
    conditioned inputs.
    """
    from numpy.polynomial.polynomial import Polynomial

    valid_m = np.isfinite(obs_mag) & np.isfinite(e_mag) & (e_mag > 0)
    valid_c = np.isfinite(obs_mag) & np.isfinite(e_col) & (e_col > 0)

    def _fit(mag_ok, e_ok, fallback):
        if mag_ok.sum() < 3:
            return lambda m: np.full_like(np.atleast_1d(m), fallback, dtype=float)
        poly = Polynomial.fit(mag_ok, np.log10(e_ok), 2)
        return lambda m: 10 ** poly(np.atleast_1d(m).astype(float))

    f_emag = _fit(obs_mag[valid_m], e_mag[valid_m], float(np.nanmedian(e_mag)))
    f_ecol = _fit(obs_mag[valid_c], e_col[valid_c], float(np.nanmedian(e_col)))
    return f_emag, f_ecol


# ---------------------------------------------------------------------------
# IsochroneFitter
# ---------------------------------------------------------------------------

class IsochroneFitter:
    """Bayesian isochrone fitting using a MIST forward model and PyMC NUTS.

    Replicates ASteCA's forward model (Chabrier IMF, D&K binaries, CCM89
    extinction, magnitude-sorted error model) without any dependency on ASteCA.

    Workflow
    --------
    1. ``fitter = IsochroneFitter(isochs_path, ...)``
    2. ``fitter.setup(cluster_data, prob_threshold=0.6)``
    3. ``fitter.save_grid(path)``  /  ``fitter.load_grid(path)``  ← optional
    4. ``idata = fitter.fit(draws=2000, tune=1000, chains=4)``
    5. ``obs_mag, obs_col, cmds = fitter.posterior_cmd(idata, num_samples=30)``
    """

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
        loga_range: tuple[float, float] = (6.0, 7.0),
        Av_range:   tuple[float, float] = (0.0, 3.0),
        dm_mu:    float = 10.2,
        dm_sigma: float = 0.3,
        dm_range: tuple[float, float] = (9.5, 10.7),
        M_met:  int = 200,
        M_loga: int = 200,
    ) -> None:
        self.isochs_path    = Path(isochs_path)
        self.magnitude      = magnitude
        self.magnitude_effl = magnitude_effl
        self.color          = color
        self.color_effl     = color_effl
        self.model_name     = model
        self.Rv             = Rv
        self.alpha          = alpha
        self.beta           = beta
        self.loga_range     = loga_range
        self.Av_range       = Av_range
        self.dm_mu          = dm_mu
        self.dm_sigma       = dm_sigma
        self.dm_range       = dm_range
        self.M_met          = M_met
        self.M_loga         = M_loga

        self._isochs:     MISTIsochrones | None = None
        self._obs_mag:    np.ndarray | None = None
        self._obs_col:    np.ndarray | None = None
        self._e_mag_fn:   Any = None   # callable: apparent_mag → error
        self._e_col_fn:   Any = None
        self._N_obs:      int | None = None
        self._obs_tensor: Any = None
        self._H_grid:     np.ndarray | None = None
        self._met_grid:   np.ndarray | None = None
        self._loga_grid:  np.ndarray | None = None
        self._mag_range:  tuple[float, float] | None = None
        self._col_range:  tuple[float, float] | None = None
        self._Nbins:      tuple[int, int] | None = None
        self._binw_mag:   float | None = None
        self._binw_col:   float | None = None
        self._kG:         float | None = None
        self._kBP:        float | None = None
        self._kRP:        float | None = None
        self._k_col1:     float | None = None  # = kBP - kRP
        self._I_tensor:   Any = None  # precomputed bin-index tensor (Nb_mag, 1)
        self._J_tensor:   Any = None  # precomputed bin-index tensor (1, Nb_col)
        self._H_tensor:   Any = None  # pytensor.shared wrapping _H_grid
        self._obs_hess:   np.ndarray | None = None  # flat observed Hess (Nb_mag*Nb_col,)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(
        self,
        cluster_data: QTable,
        *,
        prob_threshold: float = 0.6,
        precompute_grid: bool = True,
        pms_column: str | None = None,
        pms_max: float = 0.5,
        ms_weight: float = 1.0,
    ) -> None:
        """Prepare binning, error model, and optionally the H_grid.

        Parameters
        ----------
        cluster_data : QTable
            Must contain ``Gmag``, ``G_BPmag``, ``G_RPmag``, ``e_Gmag``,
            ``e_G_BPmag``/``e_G_RPmag`` (or ``e_BP_RP``), and
            ``probability_hdbscan``.
        prob_threshold : float
            Minimum membership probability.
        precompute_grid : bool
            ``False`` skips the expensive H_grid build; use when loading from cache.
        pms_column : str, optional
            Column name with PMS probability (e.g. ``"pms_sagitta"``).
            Used together with ``ms_weight`` to upweight MS stars in the Hess.
        pms_max : float
            Stars with ``pms_column >= pms_max`` are treated as PMS (weight 1).
            Stars below this threshold are treated as MS (weight ``ms_weight``).
        ms_weight : float
            Multiplicative weight applied to MS stars in the Hess histogram.
            ``ms_weight=1`` (default) → uniform weights, no preference.
            ``ms_weight=3`` → MS stars count 3× more than PMS in the likelihood,
            anchoring the isochrone fit to the main sequence while still
            allowing PMS stars to contribute.
        """
        col1_name, col2_name = self.color
        self._isochs = MISTIsochrones(
            self.isochs_path,
            magnitude_col=self.magnitude,
            color_col1=col1_name,
            color_col2=col2_name,
        )

        mask    = np.array(cluster_data["probability_hdbscan"]) >= prob_threshold
        members = cluster_data[mask]
        self._N_obs = int(mask.sum())

        # Per-star Hess weights: MS stars get ms_weight, PMS stars get 1.0.
        # When pms_column is None or ms_weight==1, all weights are uniform.
        hess_weights = np.ones(len(members), dtype=float)
        if pms_column is not None and pms_column in members.colnames and ms_weight != 1.0:
            pms_prob = np.asarray(members[pms_column], dtype=float)
            is_ms    = ~(np.isfinite(pms_prob) & (pms_prob >= pms_max))
            hess_weights[is_ms] = ms_weight
            n_ms  = int(is_ms.sum())
            n_pms = int((~is_ms).sum())
            print(f"[setup] Weighted Hess: {n_ms} MS stars (w={ms_weight}) + "
                  f"{n_pms} PMS stars (w=1.0).")
        ms_members = members   # all stars still enter the histogram

        # Use ALL members for range, error model, and posterior_cmd
        obs_mag = np.asarray(members["Gmag"],    dtype=float)
        bp      = np.asarray(members["G_BPmag"], dtype=float)
        rp      = np.asarray(members["G_RPmag"], dtype=float)
        obs_col = bp - rp

        e_mag = np.asarray(members["e_Gmag"], dtype=float)
        if "e_BP_RP" in members.colnames:
            e_col = np.asarray(members["e_BP_RP"], dtype=float)
        else:
            e_bp  = np.asarray(members["e_G_BPmag"], dtype=float)
            e_rp  = np.asarray(members["e_G_RPmag"], dtype=float)
            e_col = np.hypot(e_bp, e_rp)

        self._obs_mag = obs_mag
        self._obs_col = obs_col
        # Error model fitted on all members (wider magnitude baseline)
        self._e_mag_fn, self._e_col_fn = _fit_error_model(obs_mag, e_mag, e_col)

        # MS-only photometry for the Hess histogram (the fit target)
        ms_obs_mag = np.asarray(ms_members["Gmag"],    dtype=float)
        ms_bp      = np.asarray(ms_members["G_BPmag"], dtype=float)
        ms_rp      = np.asarray(ms_members["G_RPmag"], dtype=float)
        ms_obs_col = ms_bp - ms_rp

        # Extinction coefficients (A_lambda / A_V)
        self._kG, self._kBP, self._kRP = self._compute_ext_coefs()
        self._k_col1 = self._kBP - self._kRP

        # Histogram range: observed + reference synthetic at (dm_mu, Av_mean).
        # Use the midpoint of the *prior* loga_range (not the full file range) so
        # that the reference isochrone has the right age — otherwise a young-cluster
        # prior (loga~6) gets compared against a median-file isochrone (loga~7.6),
        # which misses the bright OB sequence and sets mag_min too faint.
        Av_mid   = float(np.mean(self.Av_range))
        met_arr  = self._isochs.met_age_dict["met"]
        loga_mid = float(np.mean(self.loga_range))
        _, ref_G, ref_BP, ref_RP = self._isochs.get_isochrone(
            float(np.median(met_arr)), loga_mid
        )
        ref_app_mag = ref_G  + self.dm_mu + self._kG    * Av_mid
        ref_app_col = (ref_BP - ref_RP) + self._k_col1 * Av_mid

        mag_min = float(min(obs_mag.min(), np.nanmin(ref_app_mag)))
        mag_max = float(max(obs_mag.max(), np.nanmax(ref_app_mag)))
        col_min = float(min(obs_col.min(), np.nanmin(ref_app_col)))
        col_max = float(max(obs_col.max(), np.nanmax(ref_app_col)))
        self._mag_range = (mag_min, mag_max)
        self._col_range = (col_min, col_max)

        Nb_mag, Nb_col = self._compute_nbins(obs_mag, obs_col)
        self._Nbins    = (Nb_mag, Nb_col)
        self._binw_mag = (mag_max - mag_min) / Nb_mag
        self._binw_col = (col_max - col_min) / Nb_col

        import pytensor.tensor as pt
        self._I_tensor = pt.constant(
            np.arange(Nb_mag, dtype="float64").reshape(Nb_mag, 1), name="I_bins"
        )
        self._J_tensor = pt.constant(
            np.arange(Nb_col, dtype="float64").reshape(1, Nb_col), name="J_bins"
        )

        # Build weighted Hess: numpy.histogram2d supports weights;
        # fast_histogram does not, so we only fall back to it for the uniform case.
        if np.all(hess_weights == 1.0):
            from fast_histogram import histogram2d as _h2d
            cl_histo = _h2d(
                ms_obs_mag, ms_obs_col,
                bins=[Nb_mag, Nb_col],
                range=[[mag_min, mag_max], [col_min, col_max]],
            ).astype(np.float64)
        else:
            cl_histo, _, _ = np.histogram2d(
                ms_obs_mag, ms_obs_col,
                bins=[Nb_mag, Nb_col],
                range=[[mag_min, mag_max], [col_min, col_max]],
                weights=hess_weights,
            )
            cl_histo = cl_histo.astype(np.float64)
        # Store as plain numpy — passed directly to pm.Poisson(observed=...)
        # so PyMC wraps it in ConstantData internally (idiomatic, no graph bloat).
        self._obs_hess = cl_histo.ravel()

        if precompute_grid:
            self._precompute_H_grid()

        if self._H_grid is not None:
            import pytensor
            # pytensor.shared → JAX/blackjax receives array as a runtime input
            # rather than baking it into the JIT trace as a literal constant.
            self._H_tensor = pytensor.shared(self._H_grid, name="H_grid")

    def _compute_ext_coefs(self) -> tuple[float, float, float]:
        kG  = _ccm89(self.magnitude_effl, self.Rv)
        kBP = _ccm89(self.color_effl[0],  self.Rv)
        kRP = _ccm89(self.color_effl[1],  self.Rv)
        return kG, kBP, kRP

    def _compute_nbins(self, mag: np.ndarray, col: np.ndarray) -> tuple[int, int]:
        try:
            from astropy.stats import knuth_bin_width
            _, bm = knuth_bin_width(mag, return_bins=True)
            _, bc = knuth_bin_width(col, return_bins=True)
            return max(int(len(bm) - 1), 5), max(int(len(bc) - 1), 5)
        except Exception:
            nb = max(int(np.ceil(len(mag) ** (1 / 3))), 5)
            return nb, nb

    # ------------------------------------------------------------------
    # Hess grid precomputation  (matches ASteCA's generate() in expectation)
    # ------------------------------------------------------------------

    def _hess_for_isochrone(
        self,
        mass: np.ndarray,
        G_abs: np.ndarray,
        BP_abs: np.ndarray,
        RP_abs: np.ndarray,
    ) -> np.ndarray:
        """Build a Hess diagram at dm=0, Av=0, matching ASteCA's expected CMD."""
        Nb_mag, Nb_col = self._Nbins            # type: ignore[misc]
        mag_min, mag_max = self._mag_range      # type: ignore[misc]
        col_min, col_max = self._col_range      # type: ignore[misc]
        Av_mid = float(np.mean(self.Av_range))

        # IMF weights
        w_imf = _chabrier2014_weights(mass)

        # ---- Binary photometry ----------------------------------------
        # Per-star binary probability (Offner et al. 2022)
        b_p = np.clip(
            self.alpha + self.beta / (1.0 + 1.4 / np.clip(mass, 1e-9, None)), 0.0, 1.0
        )
        # Expected mass-ratio from D&K (2013)
        q   = _dk_mean_q(mass)
        M2  = q * mass

        # Interpolate secondary G, BP, RP onto the same isochrone
        G2  = np.interp(M2, mass, G_abs,  left=G_abs[0],  right=G_abs[-1])
        BP2 = np.interp(M2, mass, BP_abs, left=BP_abs[0], right=BP_abs[-1])
        RP2 = np.interp(M2, mass, RP_abs, left=RP_abs[0], right=RP_abs[-1])

        G_comb  = _mag_combine(G_abs,  G2)
        BP_comb = _mag_combine(BP_abs, BP2)
        RP_comb = _mag_combine(RP_abs, RP2)
        col_comb = BP_comb - RP_comb
        col_sing = BP_abs  - RP_abs

        # Cut at max(obs_mag) in apparent magnitude (like ASteCA's cut_max_mag)
        max_app = self._obs_mag.max()       # type: ignore[union-attr]
        app_G_sing = G_abs + self.dm_mu + self._kG * Av_mid   # type: ignore[operator]
        cut = app_G_sing < max_app

        # ---- Build weighted histogram ----------------------------------
        # Single-star contribution
        w_sing  = w_imf * (1.0 - b_p) * cut
        H_sing, _, _ = np.histogram2d(
            G_abs, col_sing,
            bins=[Nb_mag, Nb_col],
            range=[[mag_min, mag_max], [col_min, col_max]],
            weights=w_sing,
        )

        # Binary contribution
        app_G_comb = G_comb + self.dm_mu + self._kG * Av_mid
        cut_b = app_G_comb < max_app
        w_bin  = w_imf * b_p * cut_b
        H_bin, _, _ = np.histogram2d(
            G_comb, col_comb,
            bins=[Nb_mag, Nb_col],
            range=[[mag_min, mag_max], [col_min, col_max]],
            weights=w_bin,
        )

        H = H_sing + H_bin

        # ---- Convolve with magnitude-dependent errors ------------------
        # Evaluate errors at apparent magnitude ≈ abs_mag + dm_mu + kG*Av_mid
        app_mag_rep = G_abs + self.dm_mu + self._kG * Av_mid   # type: ignore[operator]
        e_m = self._e_mag_fn(app_mag_rep)    # type: ignore[misc]
        e_c = self._e_col_fn(app_mag_rep)    # type: ignore[misc]
        # IMF-weighted mean errors
        e_m_eff = float(np.sum(w_imf * e_m) / np.sum(w_imf))
        e_c_eff = float(np.sum(w_imf * e_c) / np.sum(w_imf))
        sig_m_px = e_m_eff / self._binw_mag   # type: ignore[operator]
        sig_c_px = e_c_eff / self._binw_col   # type: ignore[operator]
        if sig_m_px > 0.1 or sig_c_px > 0.1:
            H = gaussian_filter(H, sigma=[sig_m_px, sig_c_px])

        return _smooth2d(H)

    def _precompute_H_grid(self) -> None:
        met_arr  = self._isochs.met_age_dict["met"]   # type: ignore[union-attr]
        loga_arr = self._isochs.met_age_dict["loga"]  # type: ignore[union-attr]
        met_grid  = np.linspace(float(met_arr[0]),  float(met_arr[-1]),  self.M_met)
        loga_grid = np.linspace(float(loga_arr[0]), float(loga_arr[-1]), self.M_loga)
        self._met_grid  = met_grid
        self._loga_grid = loga_grid

        Nb_mag, Nb_col = self._Nbins  # type: ignore[misc]
        H_grid = np.zeros((self.M_met, self.M_loga, Nb_mag, Nb_col), dtype=np.float64)
        total  = self.M_met * self.M_loga
        done   = 0

        # Cache by actual nearest-neighbor file indices — many linspace points
        # map to the same file isochrone, so this avoids redundant Hess builds.
        file_met  = self._isochs._met_values   # type: ignore[union-attr]
        file_loga = self._isochs._loga_values  # type: ignore[union-attr]
        _hess_cache: dict[tuple[int, int], np.ndarray] = {}

        for i, met in enumerate(met_grid):
            mi = int(np.argmin(np.abs(file_met - met)))
            for j, loga in enumerate(loga_grid):
                ai = int(np.argmin(np.abs(file_loga - loga)))
                cache_key = (mi, ai)
                if cache_key not in _hess_cache:
                    mass, G, BP, RP = self._isochs.get_isochrone(  # type: ignore[union-attr]
                        float(met), float(loga)
                    )
                    _hess_cache[cache_key] = (
                        self._hess_for_isochrone(mass, G, BP, RP)
                        if len(mass) >= 2
                        else np.zeros((Nb_mag, Nb_col), dtype=np.float64)
                    )
                H_grid[i, j] = _hess_cache[cache_key]
                done += 1
                if done % 1000 == 0:
                    print(f"  H_grid: {done}/{total} ({100*done/total:.0f}%)")

        self._H_grid = H_grid
        print(f"  H_grid complete: {H_grid.shape}")

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
    # Grid construction (step 3 of the step-by-step workflow)
    # ------------------------------------------------------------------

    def build_grid(
        self,
        M_met: int | None = None,
        M_loga: int | None = None,
        *,
        grid_cache: str | Path | None = None,
    ) -> None:
        """Precompute (or load from cache) the H_grid and set up tensors.

        Call this after ``setup(precompute_grid=False)`` and (optionally)
        ``set_priors()``.  Loading from cache is much faster than recomputing.

        Parameters
        ----------
        M_met, M_loga : int, optional
            Override the grid resolution set in ``__init__``.
        grid_cache : path, optional
            If the file exists, load from it; otherwise compute and save.
        """
        import pytensor

        if M_met is not None:
            self.M_met = M_met
        if M_loga is not None:
            self.M_loga = M_loga

        if grid_cache is not None and Path(grid_cache).exists():
            print(f"Loading H_grid from cache: {grid_cache}")
            self.load_grid(grid_cache)
        else:
            self._precompute_H_grid()
            self._H_tensor = pytensor.shared(self._H_grid, name="H_grid")
            if grid_cache is not None:
                self.save_grid(grid_cache)
                print(f"H_grid saved to: {grid_cache}")

    # ------------------------------------------------------------------
    # Grid cache
    # ------------------------------------------------------------------

    def save_grid(self, path: str | Path) -> None:
        if self._H_grid is None:
            raise RuntimeError("No H_grid to save. Run setup() first.")
        np.savez_compressed(
            path,
            H_grid=self._H_grid,
            met_grid=self._met_grid,
            loga_grid=self._loga_grid,
            Nbins=np.array(self._Nbins),
            mag_range=np.array(self._mag_range),
            col_range=np.array(self._col_range),
            ext_coefs=np.array([self._kG, self._kBP, self._kRP]),
        )

    def load_grid(self, path: str | Path) -> None:
        import pytensor.tensor as pt

        d = np.load(path)
        self._H_grid    = d["H_grid"]
        self._met_grid  = d["met_grid"]
        self._loga_grid = d["loga_grid"]
        self._Nbins     = tuple(d["Nbins"].tolist())
        self._mag_range = tuple(d["mag_range"].tolist())
        self._col_range = tuple(d["col_range"].tolist())
        self._kG   = float(d["ext_coefs"][0])
        self._kBP  = float(d["ext_coefs"][1])
        self._kRP  = float(d["ext_coefs"][2])
        self._k_col1 = self._kBP - self._kRP
        Nb_mag, Nb_col  = self._Nbins
        self._binw_mag  = (self._mag_range[1] - self._mag_range[0]) / Nb_mag
        self._binw_col  = (self._col_range[1] - self._col_range[0]) / Nb_col
        import pytensor
        # Rebuild PyTensor tensors so build_model() works after load_grid()
        self._I_tensor = pt.constant(
            np.arange(Nb_mag, dtype="float64").reshape(Nb_mag, 1), name="I_bins"
        )
        self._J_tensor = pt.constant(
            np.arange(Nb_col, dtype="float64").reshape(1, Nb_col), name="J_bins"
        )
        self._H_tensor = pytensor.shared(self._H_grid, name="H_grid")

    # ------------------------------------------------------------------
    # PyTensor ops (differentiable)
    # ------------------------------------------------------------------

    def _interp_H(self, met_v: Any, loga_v: Any) -> Any:
        import pytensor.tensor as pt
        H_t   = self._H_tensor  # pt.constant set during setup() / load_grid()
        met0  = float(self._met_grid[0])                          # type: ignore[index]
        dmet  = float(self._met_grid[1]  - self._met_grid[0])    # type: ignore[index]
        loga0 = float(self._loga_grid[0])                        # type: ignore[index]
        dloga = float(self._loga_grid[1] - self._loga_grid[0])   # type: ignore[index]

        mpos = (met_v  - met0)  / dmet
        apos = (loga_v - loga0) / dloga

        i0 = pt.cast(pt.clip(pt.floor(mpos), 0, self.M_met  - 2), "int64")
        j0 = pt.cast(pt.clip(pt.floor(apos), 0, self.M_loga - 2), "int64")
        i1, j1 = i0 + 1, j0 + 1
        wm = mpos - pt.cast(i0, "float64")
        wa = apos - pt.cast(j0, "float64")

        return (
            (1 - wm) * (1 - wa) * H_t[i0, j0]
            + wm * (1 - wa) * H_t[i1, j0]
            + (1 - wm) * wa  * H_t[i0, j1]
            + wm * wa        * H_t[i1, j1]
        )

    def _shift_histogram(self, H: Any, dmag: Any, dcol: Any) -> Any:
        import pytensor.tensor as pt
        su = dmag / self._binw_mag
        sv = dcol / self._binw_col
        Nb_mag, Nb_col = self._Nbins  # type: ignore[misc]

        U, V = self._I_tensor - su, self._J_tensor - sv
        u0, v0 = pt.floor(U), pt.floor(V)
        fu = pt.cast(U - u0, "float64")
        fv = pt.cast(V - v0, "float64")
        u0i = pt.cast(u0, "int64")
        v0j = pt.cast(v0, "int64")
        u1i = u0i + 1
        v1j = v0j + 1

        valid = pt.cast((U >= 0) & (U <= Nb_mag - 1) & (V >= 0) & (V <= Nb_col - 1), "float64")

        def _g(A: Any, r: Any, c: Any) -> Any:
            r = pt.cast(pt.clip(r, 0, Nb_mag - 1), "int64")
            c = pt.cast(pt.clip(c, 0, Nb_col - 1), "int64")
            return A[r, c]

        return (
            (1 - fu) * (1 - fv) * _g(H, u0i, v0j)
            + fu  * (1 - fv) * _g(H, u1i, v0j)
            + (1 - fu) * fv  * _g(H, u0i, v1j)
            + fu  * fv       * _g(H, u1i, v1j)
        ) * valid

    # ------------------------------------------------------------------
    # PyMC model
    # ------------------------------------------------------------------

    def build_model(self) -> Any:
        import pymc as pm
        import pytensor.tensor as pt

        if self._H_grid is None or self._obs_hess is None:
            raise RuntimeError(
                "Call setup() (or setup(precompute_grid=False) + load_grid()) first."
            )

        met_arr          = self._isochs.met_age_dict["met"]  # type: ignore[union-attr]
        met_min, met_max = float(met_arr[0]), float(met_arr[-1])
        loga_min, loga_max = self.loga_range

        with pm.Model() as model:
            met  = pm.Uniform("met",  lower=met_min,  upper=met_max)
            loga = pm.Uniform("loga", lower=loga_min, upper=loga_max)
            dm   = pm.TruncatedNormal(
                "dm", mu=self.dm_mu, sigma=self.dm_sigma,
                lower=self.dm_range[0], upper=self.dm_range[1],
            )
            Av = pm.Uniform("Av", lower=self.Av_range[0], upper=self.Av_range[1])

            H0   = self._interp_H(met, loga)
            dmag = dm + self._kG * Av
            dcol = self._k_col1 * Av
            Hsyn = self._shift_histogram(H0, dmag, dcol)

            lam   = pt.maximum(Hsyn.reshape((-1,)), 1e-6)
            # log_s scales the normalized Hess (sum ≈ 1) to match N_obs counts.
            # Centering at log(N_obs) gives NUTS a sensible starting region.
            log_s = pm.Normal("log_s", mu=float(np.log(self._N_obs)), sigma=1.0)
            bg    = pm.HalfNormal("bg", sigma=0.2)
            mu    = pt.exp(log_s) * lam + bg

            # Pass raw numpy array — PyMC wraps it in ConstantData internally,
            # which is the idiomatic pattern and avoids embedding it in the graph.
            pm.Poisson("y", mu=mu, observed=self._obs_hess)

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
        nuts_sampler: str = "blackjax",
        random_seed: int | None = None,
        progressbar: bool = True,
        # Initialisation
        init: str = "auto",
        initvals: dict | None = None,
        # Posterior log-likelihood (needed for LOO / WAIC)
        log_likelihood: bool = False,
        # Extra kwargs forwarded verbatim to the chosen NUTS backend
        nuts_sampler_kwargs: dict | None = None,
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
            ``"blackjax"`` (JAX/GPU-friendly), ``"numpyro"``, or ``"pymc"``
            (default PyMC C++ implementation).
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
            post-processing time proportional to ``draws × Nb_mag × Nb_col``.
        nuts_sampler_kwargs : dict, optional
            Extra keyword arguments forwarded verbatim to the backend sampler.
            For blackjax: ``{"step_size": 0.01}`` to override the initial
            step size; for numpyro: ``{"adapt_step_size": True}``.
        """
        import pymc as pm

        model = self.build_model()
        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                nuts_sampler=nuts_sampler,
                target_accept=target_accept,
                random_seed=random_seed,
                progressbar=progressbar,
                init=init,
                initvals=initvals,
                nuts_sampler_kwargs=nuts_sampler_kwargs or {},
                idata_kwargs={"log_likelihood": log_likelihood},
            )
        return idata

    # ------------------------------------------------------------------
    # Posterior utilities
    # ------------------------------------------------------------------

    def posterior_cmd(
        self, idata: Any, *, num_samples: int = 20
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Generate synthetic CMDs drawn from the posterior.

        Returns
        -------
        obs_mag, obs_col : np.ndarray
        cmds : list of (mag_arr, col_arr)
            Access: ``syn = cmds[i]; mag = syn[0]; col = syn[1]``.
        """
        import arviz as az

        ds = az.extract(idata, num_samples=num_samples, var_names=["met", "loga", "dm", "Av"])
        # Use .values.flat to iterate regardless of the ArviZ sample-dim name
        met_vals  = np.asarray(ds["met"]).ravel()
        loga_vals = np.asarray(ds["loga"]).ravel()
        dm_vals   = np.asarray(ds["dm"]).ravel()
        Av_vals   = np.asarray(ds["Av"]).ravel()

        N_out   = self._N_obs if self._N_obs else 500
        max_app = float(self._obs_mag.max())   # type: ignore[union-attr]

        cmds: list[tuple[np.ndarray, np.ndarray]] = []
        for met_v, loga_v, dm_v, Av_v in zip(met_vals, loga_vals, dm_vals, Av_vals):
            met_v  = float(met_v)
            loga_v = float(loga_v)
            dm_v   = float(dm_v)
            Av_v   = float(Av_v)

            mass, G_abs, BP_abs, RP_abs = self._isochs.get_isochrone(  # type: ignore[union-attr]
                met_v, loga_v
            )
            if len(mass) < 2:
                continue

            # Apply distance modulus and per-band extinction
            G_app  = G_abs  + dm_v + self._kG  * Av_v   # type: ignore[operator]
            BP_app = BP_abs + dm_v + self._kBP * Av_v   # type: ignore[operator]
            RP_app = RP_abs + dm_v + self._kRP * Av_v   # type: ignore[operator]
            col_app = BP_app - RP_app

            # Cut at max observed magnitude (ASteCA: cut_max_mag)
            cut = G_app < max_app

            # Binary contribution — draw q from D&K power-law f(q) ∝ q^γ
            # Inverse CDF: q = u^{1/(γ+1)} for u ~ Uniform(0,1), γ > -1.
            gamma_q = np.where(
                mass <= 0.1, 4.2,
                np.where(mass <= 0.6, 0.4,
                np.where(mass <= 1.4, 0.3,
                np.where(mass <= 6.5, -0.5, 0.0)))
            )
            u_q = np.random.uniform(size=len(mass))
            q = np.clip(u_q ** (1.0 / (gamma_q + 1.0)), 0.0, 1.0)
            M2   = mass * q
            G2   = np.interp(M2, mass, G_abs,  left=G_abs[0],  right=G_abs[-1])
            BP2  = np.interp(M2, mass, BP_abs, left=BP_abs[0], right=BP_abs[-1])
            RP2  = np.interp(M2, mass, RP_abs, left=RP_abs[0], right=RP_abs[-1])

            # Combined apparent magnitudes — shift property ensures correct extinction
            G_comb   = _mag_combine(G_abs,  G2)  + dm_v + self._kG  * Av_v   # type: ignore[operator]
            BP_comb  = _mag_combine(BP_abs, BP2) + dm_v + self._kBP * Av_v   # type: ignore[operator]
            RP_comb  = _mag_combine(RP_abs, RP2) + dm_v + self._kRP * Av_v   # type: ignore[operator]
            col_comb = BP_comb - RP_comb

            b_p = np.clip(
                self.alpha + self.beta / (1.0 + 1.4 / np.clip(mass, 1e-9, None)), 0.0, 1.0
            )
            is_binary = (np.random.uniform(size=len(mass)) < b_p) & cut

            G_use   = np.where(is_binary, G_comb,   G_app)
            col_use = np.where(is_binary, col_comb, col_app)

            # Sample N_out stars from Chabrier IMF
            w = _chabrier2014_weights(mass)
            idx = np.random.choice(len(mass), size=N_out, replace=True, p=w)

            mag_s = G_use[idx]
            col_s = col_use[idx]

            # Add magnitude-sorted errors (like ASteCA's add_errors)
            e_m = self._e_mag_fn(mag_s)   # type: ignore[misc]
            e_c = self._e_col_fn(mag_s)   # type: ignore[misc]
            # Sort by magnitude descending (faintest first), like ASteCA
            sidx    = np.argsort(-mag_s)
            mag_s   = mag_s[sidx] + np.random.normal(0.0, 1.0, N_out) * e_m[sidx]
            col_s   = col_s[sidx] + np.random.normal(0.0, 1.0, N_out) * e_c[sidx]

            valid = (
                (mag_s >= self._mag_range[0]) & (mag_s <= self._mag_range[1])   # type: ignore[index]
                & (col_s >= self._col_range[0]) & (col_s <= self._col_range[1])  # type: ignore[index]
            )
            cmds.append((mag_s[valid], col_s[valid]))

        return self._obs_mag, self._obs_col, cmds   # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Diagnostic plots
    # ------------------------------------------------------------------

    def cmd_distances(
        self,
        idata: Any,
        cluster_table: QTable,
        *,
        prob_threshold: float = 0.6,
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
        import arviz as az
        import pandas as pd

        # Posterior median parameters
        post = az.extract(idata, var_names=["met", "loga", "dm", "Av"])
        med  = {v: float(np.median(np.asarray(post[v]).ravel()))
                for v in ["met", "loga", "dm", "Av"]}

        # Apparent isochrone at posterior median
        mass, G_abs, BP_abs, RP_abs = self._isochs.get_isochrone(  # type: ignore[union-attr]
            med["met"], med["loga"]
        )
        G_app   = G_abs  + med["dm"] + self._kG  * med["Av"]   # type: ignore[operator]
        BP_app  = BP_abs + med["dm"] + self._kBP * med["Av"]   # type: ignore[operator]
        RP_app  = RP_abs + med["dm"] + self._kRP * med["Av"]   # type: ignore[operator]
        col_app = BP_app - RP_app

        # Observed members
        mask    = np.array(cluster_table["probability_hdbscan"]) >= prob_threshold
        members = cluster_table[mask]
        obs_g   = np.asarray(members["Gmag"],    dtype=float)
        obs_col = np.asarray(members["G_BPmag"], dtype=float) \
                - np.asarray(members["G_RPmag"], dtype=float)

        # Normalise CMD space by typical observed spreads so both axes contribute equally
        sig_g   = float(np.nanstd(obs_g))   or 1.0
        sig_col = float(np.nanstd(obs_col)) or 1.0

        iso_g_n   = G_app   / sig_g
        iso_col_n = col_app / sig_col

        rows = []
        for i in range(len(members)):
            g_n   = obs_g[i]   / sig_g
            c_n   = obs_col[i] / sig_col
            dists = np.hypot(iso_g_n - g_n, iso_col_n - c_n)
            j     = int(np.argmin(dists))

            d_cmd     = float(np.hypot(G_app[j]   - obs_g[i],
                                       col_app[j] - obs_col[i]))
            delta_col = float(obs_col[i] - col_app[j])   # + = redder than MS

            pms_prob = float("nan")
            if pms_column and pms_column in members.colnames:
                val = members[pms_column][i]
                pms_prob = float(val) if np.isfinite(float(val)) else float("nan")

            pms_score = (pms_prob / (1.0 + d_cmd)
                         if np.isfinite(pms_prob) else float("nan"))

            rows.append(dict(
                source_id = int(members["source_id"][i]),
                Gmag      = float(obs_g[i]),
                BP_RP     = float(obs_col[i]),
                iso_Gmag  = float(G_app[j]),
                iso_BP_RP = float(col_app[j]),
                mass_est  = float(mass[j]),
                d_cmd     = d_cmd,
                delta_col = delta_col,
                pms_prob  = pms_prob,
                pms_score = pms_score,
            ))

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
        import matplotlib.pyplot as plt
        import matplotlib.cm as mcm
        import matplotlib.colors as mcolors

        df = self.cmd_distances(
            idata, cluster_table,
            prob_threshold=prob_threshold,
            pms_column=pms_column,
        )

        import arviz as az
        post = az.extract(idata, var_names=["met", "loga", "dm", "Av"])
        med  = {v: float(np.median(np.asarray(post[v]).ravel()))
                for v in ["met", "loga", "dm", "Av"]}
        mass, G_abs, BP_abs, RP_abs = self._isochs.get_isochrone(  # type: ignore[union-attr]
            med["met"], med["loga"]
        )
        G_app   = G_abs  + med["dm"] + self._kG  * med["Av"]   # type: ignore[operator]
        col_app = (BP_abs - RP_abs) + self._k_col1 * med["Av"]  # type: ignore[operator]
        cut = G_app < float(df["Gmag"].max()) + 0.5

        # Separate MS and PMS stars
        if pms_column and pms_column in df.columns:
            is_pms = df["pms_prob"] >= pms_threshold
        else:
            is_pms = df["delta_col"] > 0.1   # fallback: redder than isochrone

        ms_df  = df[~is_pms]
        pms_df = df[is_pms]

        scores = pms_df["pms_score"].values
        finite = np.isfinite(scores)
        if finite.any():
            norm = mcolors.Normalize(vmin=float(np.nanmin(scores[finite])),
                                     vmax=float(np.nanmax(scores[finite])))
        else:
            norm = mcolors.Normalize(0, 1)
        cmap = mcm.get_cmap("plasma")

        fig, ax = plt.subplots(figsize=figsize)

        ax.scatter(ms_df["BP_RP"], ms_df["Gmag"],
                   s=10, alpha=0.4, color="steelblue", label="MS members", zorder=2)

        sc = ax.scatter(pms_df["BP_RP"], pms_df["Gmag"],
                        c=pms_df["pms_score"], cmap=cmap, norm=norm,
                        s=20, alpha=0.8, zorder=3, label="PMS members")

        if top_n is not None:
            top = pms_df.head(top_n)
            ax.scatter(top["BP_RP"], top["Gmag"],
                       s=80, facecolors="none", edgecolors="red",
                       linewidths=1.5, zorder=5, label=f"Top {top_n} PMS")

        ax.plot(col_app[cut], G_app[cut],
                color="k", lw=1.5, zorder=4, label="Median isochrone")

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
        import arviz as az

        obs_kw    = {"s": 8,  "alpha": 0.6, "color": "steelblue",  "zorder": 3, **(obs_kw    or {})}
        syn_kw    = {"s": 3,  "alpha": 0.15, "color": "tomato",    "zorder": 2, **(syn_kw    or {})}
        median_kw = {"s": 4,  "alpha": 0.6,  "color": "firebrick", "zorder": 4, **(median_kw or {})}

        obs_mag, obs_col, cmds = self.posterior_cmd(idata, num_samples=num_samples)

        # Median posterior isochrone (noiseless, no IMF sampling)
        post = az.extract(idata, var_names=["met", "loga", "dm", "Av"])
        med  = {v: float(np.median(np.asarray(post[v]).ravel()))
                for v in ["met", "loga", "dm", "Av"]}
        mass, G_abs, BP_abs, RP_abs = self._isochs.get_isochrone(med["met"], med["loga"])  # type: ignore[union-attr]
        G_med  = G_abs  + med["dm"] + self._kG  * med["Av"]   # type: ignore[operator]
        col_med = (BP_abs + med["dm"] + self._kBP * med["Av"]) \
                - (RP_abs + med["dm"] + self._kRP * med["Av"])
        cut = G_med < self._obs_mag.max() + 0.5   # type: ignore[union-attr]

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
        save: str | None = None,
    ) -> None:
        """Three-panel Hess diagram: observed | synthetic | residuals.

        Parameters
        ----------
        idata : arviz.InferenceData
            Posterior from ``fit()``.
        save : str, optional
            File path to save the figure.
        """
        import matplotlib.pyplot as plt
        import arviz as az
        from fast_histogram import histogram2d as _h2d

        if self._obs_hess is None or self._H_grid is None:
            raise RuntimeError("Call setup() and build_grid() first.")

        Nb_mag, Nb_col     = self._Nbins           # type: ignore[misc]
        mag_min, mag_max   = self._mag_range        # type: ignore[misc]
        col_min, col_max   = self._col_range        # type: ignore[misc]

        # Observed Hess (2-D, not flat)
        H_obs = _h2d(
            self._obs_mag, self._obs_col,           # type: ignore[arg-type]
            bins=[Nb_mag, Nb_col],
            range=[[mag_min, mag_max], [col_min, col_max]],
        ).astype(np.float64)

        # Synthetic Hess @ posterior median
        post = az.extract(idata, var_names=["met", "loga", "dm", "Av"])
        med  = {v: float(np.median(np.asarray(post[v]).ravel()))
                for v in ["met", "loga", "dm", "Av"]}
        H0   = self._interp_H(med["met"], med["loga"]).eval()
        dmag = med["dm"] + self._kG  * med["Av"]   # type: ignore[operator]
        dcol = self._k_col1          * med["Av"]   # type: ignore[operator]
        # Use scipy.ndimage.shift for the numpy context (avoids PyTensor indexing)
        from scipy.ndimage import shift as _ndshift
        su   = dmag / self._binw_mag   # type: ignore[operator]
        sv   = dcol / self._binw_col   # type: ignore[operator]
        Hsyn = _ndshift(H0, [su, sv], order=1, mode="constant", cval=0.0)

        # Scale synthetic to observed counts
        scale = H_obs.sum() / max(Hsyn.sum(), 1e-12)
        Hsyn_scaled = Hsyn * scale

        residual = H_obs - Hsyn_scaled
        vlim = np.nanpercentile(np.abs(residual), 98)

        fig, axes = plt.subplots(1, 3, figsize=figsize, sharex=True, sharey=True)

        extent = [col_min, col_max, mag_max, mag_min]   # imshow: origin top-left → invert Y

        im0 = axes[0].imshow(H_obs,       extent=extent, aspect="auto",
                             cmap=cmap_hess, origin="upper")
        im1 = axes[1].imshow(Hsyn_scaled, extent=extent, aspect="auto",
                             cmap=cmap_hess, origin="upper")
        im2 = axes[2].imshow(residual,    extent=extent, aspect="auto",
                             cmap=cmap_residual, origin="upper",
                             vmin=-vlim, vmax=vlim)

        titles = ["Observed", "Synthetic @ median", "Obs − Syn"]
        ims    = [im0, im1, im2]
        for ax, title, im in zip(axes, titles, ims):
            ax.set_title(title)
            ax.set_xlabel("BP − RP")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        axes[0].set_ylabel("G")
        fig.tight_layout()
        if save:
            fig.savefig(save, dpi=200, bbox_inches="tight")
        plt.show()


__all__ = ["IsochroneFitter", "MISTIsochrones"]
