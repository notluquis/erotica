"""Injection-recovery for the SED fit (pre-registered §0.4), run before any fit of real data.

For every real member: its own band mask (configs a and b) and its own errors; truth drawn
continuous (off the MIST nodes): log T_eff ~ U[3.52, 4.30] (EEM truth: capped at its 10 645 K range),
A_V ~ U[0.3, 1.6], log g ~ U[3.6, 4.6]. R replicas per star. Arms:
  ctl        MIST truth, MIST fit
  eem-truth  EEM intrinsic colours + MIST extinction, MIST fit   (atmosphere systematic)
  irx        MIST truth + 0.3 mag in Ks and 0.1 in H, MIST fit using Ks/H wherever the star has them
  floor2     MIST truth, MIST fit with the systematic floor x2
Coverage of the p16-p84 interval and median bias, per G bin (<14, 14-17, >17), not only in aggregate.
Writes synth_<arm>_<config>.json and synth_perstar.npz.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import data  # noqa: E402
import sedgrid as sg  # noqa: E402

R = 5
BINS = [(-np.inf, 14), (14, 17), (17, np.inf)]


def draw(rng, arm):
    while True:
        hi = np.log10(10500.0) if arm == "eem-truth" else 4.30
        lt = rng.uniform(3.52, hi)
        # off-node: at least 0.002 dex from any MIST node and from the fit grid step
        T, g, A, _ = sg.load_mist_bc(0.0)
        if np.min(np.abs(np.log10(T) - lt)) > 0.002:
            break
    return lt, rng.uniform(3.6, 4.6), rng.uniform(0.3, 1.6)


def main():
    t, mag, err, have, flags = data.load()
    G = mag[:, 0]
    logt, bc = sg.mist_fine(0.0)
    out = {}
    per = {}
    rng = np.random.default_rng(20261011)
    for config in ("b", "a"):
        use_all = data.mask(have, config)
        for arm in ("ctl", "eem-truth", "irx", "floor2"):
            t0 = time.time()
            res = []
            for i in range(len(t)):
                use = use_all[i].copy()
                if use.sum() < 3:
                    continue
                if arm == "irx":  # a missed disc: every measured band of the config, flags ignored
                    use = (np.isfinite(mag[i]) & np.isfinite(err[i])
                           & np.array(data.CONFIGS[config], bool))
                s_gen = data.sigma(err[i:i + 1])[0]
                s_fit = data.sigma(err[i:i + 1], 2.0 if arm == "floor2" else 1.0)[0]
                for r in range(R):
                    lt, lg, av = draw(rng, arm)
                    model = "eem" if arm == "eem-truth" else "mist"
                    m0 = sg.truth_mags(model, lt, lg, av) + 15.0
                    if arm == "irx":
                        m0 = m0 - np.array([0, 0, 0, 0, 0.1, 0.3])
                    m = m0 + rng.normal(0, 1, 6) * s_gen
                    p, cmin = sg.fit_star(bc, m, s_fit, use)
                    sm = sg.summarise(p, logt)
                    res.append(dict(i=i, G=float(G[i]), lt=lt, av=av, chi2=cmin, nb=int(use.sum()), **sm))
            key = f"{arm}_{config}"
            per[key] = res
            out[key] = summarise_arm(res)
            print(f"{key}: {len(res)} fits in {time.time() - t0:.0f}s  "
                  + " | ".join(f"{b}: {v}" for b, v in out[key].items() if b != "all"))
            print(f"   all: {out[key]['all']}")
    (HERE / "synth_summary.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(HERE / "synth_perstar.npz", **{k: np.array([list(r.values()) for r in v])
                                                        for k, v in per.items()},
                        columns=np.array(list(per["ctl_b"][0].keys())))


def summarise_arm(res):
    def block(rr):
        if not rr:
            return None
        lt = np.array([r["lt"] for r in rr])
        av = np.array([r["av"] for r in rr])
        lt50 = np.array([r["logt50"] for r in rr])
        av50 = np.array([r["av50"] for r in rr])
        cov_t = np.mean([(r["logt16"] <= r["lt"] <= r["logt84"]) for r in rr])
        cov_a = np.mean([(r["av16"] <= r["av"] <= r["av84"]) for r in rr])
        return dict(n=len(rr), bias_av=round(float(np.median(av50 - av)), 3),
                    p16_84_dav=[round(float(x), 3) for x in np.percentile(av50 - av, [16, 84])],
                    bias_logt=round(float(np.median(lt50 - lt)), 4),
                    p16_84_dlogt=[round(float(x), 4) for x in np.percentile(lt50 - lt, [16, 84])],
                    cov_av=round(float(cov_a), 3), cov_logt=round(float(cov_t), 3),
                    med_sd_av=round(float(np.median([r["sd_av"] for r in rr])), 3),
                    med_sd_logt=round(float(np.median([r["sd_logt"] for r in rr])), 4))

    o = {f"G{lo}-{hi}": block([r for r in res if lo <= r["G"] < hi]) for lo, hi in BINS}
    o["all"] = block(res)
    return o


if __name__ == "__main__":
    main()
