"""Phase-by-phase isochrone fits of NGC 6383: the shared pieces (hub finding
``agent-findings/isochrone-ms-vs-pms.md``, pre-registered there, section 0).

WHY THIS EXISTS
---------------
The author looked at the run registry and saw MIST follow the pre-main sequence (G 14-20) but not the
turn-on / upper main sequence (G 10-14), where MIST bends horizontally at G ~ 11 while the stars run
diagonally. A single isochrone fitted to all 254 members is dominated by the 219 PMS stars, so a
mismatch in the 35 bright ones is invisible in the global fit. The literature (Naylor 2009; Bell et al.
2013; Herczeg & Hillenbrand 2015; Feiden 2016) measures exactly this: ages from different mass ranges /
evolutionary phases of the same young cluster disagree, and the practice is to fit them separately and
report the disagreement as a model systematic. This module is that fit.

Every choice carries the process it models (methodology §A.1):

* **Phase = a window in observed G** (``IsochroneFitter.setup(mag_window=...)``). The likelihood of the
  stars inside the window is the exact conditional likelihood p(star | G in window, theta): both edges
  enter F(theta) (``_p_observed``), so the stars outside the window cannot bias it. Windows are in the
  observed magnitude, the variable on which nothing else selects (membership was astrometric). The
  windows are fixed in ``WINDOWS`` before any phase age was read (hub finding §0). What it LOSES: by
  conditioning on the number of stars per window it throws away the ratio N(G<14)/N(G>=14), itself an
  age + IMF clock; that ratio is reported separately (``window_counts``), not fitted.
* **Shared parameters fixed at the global fit** ([Fe/H], dm, A_V; arm ``a``): the same distance and
  extinction for every phase, as asked. dm is the parallax's (N(10.223, 0.026) in the global fit).
  Arm ``b`` frees A_V per window: if the ages reconcile and A_V differs, the tension is extinction
  (differential reddening hits the turn-on hardest, its colour is the most A_V-sensitive part).
  ``sigma_int`` and ``f_bg`` are profiled per window: the field box area changes with the window.
* **Interval** = profile likelihood Delta ln L = 0.5. NOT calibrated: the screen measured ~25 %
  coverage for this class of interval (``isochrone-model-screen.md`` §4). The tension is read against
  the synthetic null distribution of Delta log t (``synth_phases.py``), not against these widths.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))
sys.path.insert(0, str(HERE.parent / "isochrone_colour"))
from common import EDR3, PRI_PLX, SAMPLE, mist, search  # noqa: E402,F401

# Windows in observed Gaia G (apparent). Fixed before any phase age was read (hub finding §0.2).
# Primary pair: the 35 members brighter than G = 14 (upper MS + turn-on + radiative PMS, ~1.5-8 Msun
# at 1-3 Myr) and the 219 fainter (convective PMS). Secondary split of the bright side at G = 12.
G_BRIGHT_EDGE = 5.0  # well above every member (brightest 8.80) and every model star that matters
WINDOWS = {
    "ms": (G_BRIGHT_EDGE, 14.0),
    "pms": (14.0, None),  # None = the faintest member, the global completeness cut
    "ums": (G_BRIGHT_EDGE, 12.0),
    "to": (12.0, 14.0),
}
PRIMARY = ("ms", "pms")
LOGA_PROFILE = np.round(np.arange(5.70, 7.0001, 0.05), 3)
NAMES = ("feh", "loga", "dm", "Av", "sigma_int", "f_bg")


def faint_limit(data) -> float:
    return float(np.max(np.asarray(data["Gmag"], float))) + 1e-6


def window_fitter(grid, data, window, pri=PRI_PLX):
    """An ``IsochroneFitter`` on ``grid`` restricted to the stars with G in ``window``."""
    from erotica.analysis._isochrone import IsochroneFitter

    lo, hi = window
    hi = faint_limit(data) if hi is None else hi
    f = IsochroneFitter(grid=grid, magnitude=EDR3[0], color=EDR3[1:], **pri)
    f.setup(data, prob_threshold=0.0, mag_window=(lo, hi))
    return f


def full_fitter(grid, data, pri=PRI_PLX):
    from erotica.analysis._isochrone import IsochroneFitter

    f = IsochroneFitter(grid=grid, magnitude=EDR3[0], color=EDR3[1:], **pri)
    f.setup(data, prob_threshold=0.0)
    return f


def profile_loga(f, fixed: dict, free=("sigma_int", "f_bg"), loga_grid=LOGA_PROFILE) -> dict:
    """Profile log-likelihood of log t on ``f``: at each log t, maximise over ``free`` (L-BFGS-B,
    bounded, warm-started from the neighbouring point); every other parameter at ``fixed``.
    The likelihood alone (no dm prior: dm is fixed)."""
    from scipy.optimize import minimize

    fl = f._compiled_loglike(None)
    zlo, zhi = f._met_bounds()
    bounds = {
        "feh": (zlo, zhi),
        "loga": f.loga_range,
        "dm": f.dm_range,
        "Av": f.Av_range,
        "sigma_int": (1e-4, 0.5),
        "f_bg": (1e-4, 0.5),
    }
    idx = [NAMES.index(k) for k in free]
    start = {"sigma_int": 0.05, "f_bg": 0.05, **{k: v for k, v in fixed.items() if k in free}}
    x0 = np.array([start.get(k, fixed.get(k, 0.5)) for k in free], float)
    out_ll, out_x = [], []
    t0 = time.time()
    # sweep from the middle outwards in both directions so the warm start is always a neighbour
    order = np.argsort(np.abs(loga_grid - np.median(loga_grid)), kind="stable")
    done = {}
    for j in order:
        a = float(loga_grid[j])
        if not (f.loga_range[0] <= a <= f.loga_range[1]):
            continue
        nb = [k for k in done if abs(loga_grid[k] - a) < 0.051]
        y0 = done[nb[0]][1] if nb else x0

        def nll(y, a=a):
            full = np.array([fixed.get(n, 0.0) for n in NAMES], float)
            full[1] = a
            full[idx] = y
            v, g = fl(full)
            return -v, -g[idx]

        r = minimize(nll, y0, jac=True, method="L-BFGS-B", bounds=[bounds[k] for k in free])
        done[j] = (-float(r.fun), np.asarray(r.x, float))
    keys = sorted(done)
    out_a = [float(loga_grid[k]) for k in keys]
    out_ll = [done[k][0] for k in keys]
    out_x = [dict(zip(free, done[k][1].tolist(), strict=True)) for k in keys]
    return {
        "loga": out_a,
        "loglike": out_ll,
        "free": list(free),
        "free_at": out_x,
        "fixed": {k: float(v) for k, v in fixed.items() if k not in free and k != "loga"},
        "n": int(f._N_obs),
        "seconds": round(time.time() - t0, 1),
    }


def profile_summary(p: dict) -> dict:
    """MLE of log t (parabola through the best grid point and its neighbours) and the Delta ln L =
    0.5 and 2.0 crossings (linear interpolation on the grid). Crossings beyond the grid are reported
    as the grid edge with ``open_lo``/``open_hi`` set: an interval that is not closed is said so."""
    a = np.asarray(p["loga"], float)
    ll = np.asarray(p["loglike"], float)
    k = int(np.argmax(ll))
    if 0 < k < len(a) - 1:
        x, y = a[k - 1 : k + 2], ll[k - 1 : k + 2]
        c = np.polyfit(x, y, 2)
        best = float(-c[1] / (2 * c[0])) if c[0] < 0 else float(a[k])
        best = float(np.clip(best, x[0], x[-1]))
        llmax = float(max(np.polyval(c, best), ll[k]))
    else:
        best, llmax = float(a[k]), float(ll[k])
    out = {"loga_hat": best, "loglike_max": llmax, "at_grid_edge": k in (0, len(a) - 1)}
    for d, tag in ((0.5, "1"), (2.0, "2")):
        thr = llmax - d
        lo, hi = a[0], a[-1]
        open_lo = open_hi = True
        for i in range(k, 0, -1):
            if ll[i - 1] < thr <= ll[i]:
                lo = a[i - 1] + (thr - ll[i - 1]) / (ll[i] - ll[i - 1]) * (a[i] - a[i - 1])
                open_lo = False
                break
        for i in range(k, len(a) - 1):
            if ll[i + 1] < thr <= ll[i]:
                hi = a[i] + (ll[i] - thr) / (ll[i] - ll[i + 1]) * (a[i + 1] - a[i])
                open_hi = False
                break
        out[f"lo{tag}"], out[f"hi{tag}"] = float(lo), float(hi)
        out[f"open_lo{tag}"], out[f"open_hi{tag}"] = open_lo, open_hi
    return out


def one_age_vs_two(p_ms: dict, p_pms: dict) -> dict:
    """Likelihood ratio between one log t shared by the two windows and one per window, with the
    other parameters as profiled in each. Both profiles on the same grid. 2 Delta ln L is NOT read
    against chi^2_1 (the shared parameters were fixed at a fit to the same stars): it is read against
    the synthetic null."""
    a1, l1 = np.asarray(p_ms["loga"]), np.asarray(p_ms["loglike"])
    a2, l2 = np.asarray(p_pms["loga"]), np.asarray(p_pms["loglike"])
    common_a = np.intersect1d(np.round(a1, 3), np.round(a2, 3))
    s = np.array([l1[np.isclose(a1, x)][0] + l2[np.isclose(a2, x)][0] for x in common_a])
    s1, s2 = profile_summary(p_ms), profile_summary(p_pms)
    one = float(s.max())
    two = s1["loglike_max"] + s2["loglike_max"]
    return {
        "loga_shared": float(common_a[int(np.argmax(s))]),
        "two_dlnL": float(2 * (two - one)),
        "delta_loga": s1["loga_hat"] - s2["loga_hat"],
    }


def window_counts(f_full, theta: dict, edges=(8.80, 12.0, 14.0)) -> dict:
    """Model-predicted share of the observed members in each G slice, at ``theta`` on the full fitter
    (faint cut = the faintest member): F(theta) with a two-sided window over F with the faint cut
    alone, times N. Also the expected number brighter than the brightest member (G 8.80), which
    answers whether the empty upper part of the isochrone is treated right (Poisson P(0))."""
    lim = f_full._mag_lim
    keep_b = getattr(f_full, "_mag_bright", None)
    args = [theta[k] for k in ("feh", "loga", "dm", "Av", "sigma_int")]
    tot = float(f_full._detected_fraction(*args, xp=np))
    out = {"N": int(f_full._N_obs)}
    cuts = [-np.inf, *edges, lim]
    try:
        for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
            f_full._mag_lim = hi
            f_full._mag_bright = None if not np.isfinite(lo) else lo
            frac = float(f_full._detected_fraction(*args, xp=np)) / tot
            key = f"G<{hi:g}" if not np.isfinite(lo) else f"{lo:g}<=G<{hi:g}"
            out[key] = {"frac": frac, "expected": frac * (1 - theta["f_bg"]) * f_full._N_obs}
    finally:
        f_full._mag_lim, f_full._mag_bright = lim, keep_b
    g = f_full._obs_mag
    for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
        key = f"G<{hi:g}" if not np.isfinite(lo) else f"{lo:g}<=G<{hi:g}"
        hi_o = hi + 1e-6 if hi == lim else hi  # the faintest member sits exactly on the cut
        out[key]["observed"] = int(((g >= lo) & (g < hi_o)).sum())
    return out
