#!/usr/bin/env python3
"""Applies the pre-registered rules of the hub finding ``isochrone-ms-vs-pms.md`` §0.5 to
``synth_T*.json`` and ``real_*.json``. Output: ``phases_summary.json`` (and stdout).

Every number the finding cites comes from here; the rules are code, not prose."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from phases import profile_summary
from scipy.stats import beta

HERE = Path(__file__).resolve().parent
WIN = ("ms", "pms", "ums", "to")


def _summary(p: dict) -> dict:
    """Recomputed from the stored profile, never read from the JSON: the stored ``summary`` was
    written by the ``profile_summary`` of the run (up to 4828762 its intervals could exclude their
    own ``loga_hat`` and come out open at both edges). ``loga_hat`` is the same in both versions."""
    return profile_summary(p["profile"])


def _cp(k: int, n: int) -> list | None:
    """Clopper-Pearson 68 % interval of a binomial proportion k/n."""
    if n == 0:
        return None
    lo = 0.0 if k == 0 else float(beta.ppf(0.16, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(0.84, k + 1, n - k))
    return [round(lo, 3), round(hi, 3)]


def _av(ph, w, arm):
    p = ph[w][arm]
    s = _summary(p)
    k = int(np.argmin(np.abs(np.asarray(p["profile"]["loga"]) - s["loga_hat"])))
    return p["profile"]["free_at"][k].get("Av")


def synth(truth: str) -> dict | None:
    f = HERE / f"synth_{truth}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    t = d["truth"]["loga"]
    reps = d["reps"]
    out = {
        "R": len(reps),
        "truth": d["truth"],
        "global_bias_median": float(np.median([r["global"]["mode"]["loga"] - t for r in reps])),
    }
    for key, arms in (("phases", ("a", "b")), ("phases_truth", ("a",))):
        for arm in arms:
            for w in WIN:
                if w not in reps[0][key]:
                    continue
                ss = [_summary(r[key][w][arm]) for r in reps]
                bias = np.array([s["loga_hat"] - t for s in ss])
                cov = sum(s["lo1"] <= t <= s["hi1"] for s in ss)
                opn = [s["open_lo1"] or s["open_hi1"] for s in ss]
                unr = [s["unresolved1"] and not o for s, o in zip(ss, opn, strict=True)]
                res = [not (o or u) for o, u in zip(opn, unr, strict=True)]
                cov_res = sum(s["lo1"] <= t <= s["hi1"] for s, ok in zip(ss, res, strict=True) if ok)
                out[f"{key}:{arm}:{w}"] = {
                    "bias_median": float(np.median(bias)),
                    "bias_mean": float(bias.mean()),
                    "bias_sd": float(bias.std(ddof=1)) if len(bias) > 1 else None,
                    "coverage_dlnL0.5": f"{cov}/{len(ss)}",
                    "halfwidth_median": float(np.median([(s["hi1"] - s["lo1"]) / 2 for s in ss])),
                    "open_intervals": int(sum(opn)),
                    "unresolved_intervals": int(sum(unr)),
                    "coverage_resolved": f"{cov_res}/{sum(res)}",
                    "coverage_resolved_cp68": _cp(cov_res, sum(res)),
                    "coverage_all_cp68": _cp(cov, len(ss)),
                    "estimator_outside_interval": int(
                        sum(not (s["lo1"] <= s["loga_hat"] <= s["hi1"]) for s in ss)
                    ),
                }
            ten = [r["tension"][f"{key}:{arm}"] for r in reps]
            dl = np.array([x["delta_loga"] for x in ten])
            out[f"null:{key}:{arm}"] = {
                "delta_mean": float(dl.mean()),
                "delta_median": float(np.median(dl)),
                "delta_sd": float(dl.std(ddof=1)) if len(dl) > 1 else None,
                "two_dlnL": [round(x["two_dlnL"], 2) for x in ten],
                "delta": [round(float(x), 3) for x in dl],
            }
    dav = np.array([_av(r["phases"], "ms", "b") - _av(r["phases"], "pms", "b") for r in reps])
    out["null:dAv_ms_minus_pms:b"] = {
        "mean": float(dav.mean()),
        "sd": float(dav.std(ddof=1)) if len(dav) > 1 else None,
        "values": [round(float(x), 3) for x in dav],
    }
    a = out["phases:a:ms"]["bias_median"], out["phases:a:pms"]["bias_median"]
    out["method_interpretable"] = bool(
        max(abs(a[0]), abs(a[1])) <= 0.10 and abs(out["null:phases:a"]["delta_median"]) <= 0.10
    )
    return out


def real(grid: str, null: dict | None) -> dict | None:
    f = HERE / f"real_{grid}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    out = {
        "global": d["global"]["mode"],
        "global_at_bound": d["global"]["at_bound"],
        "global_feh0": d.get("global_feh0", {}).get("mode"),
        "counts": d.get("counts"),
        "counts_feh0": d.get("counts_feh0"),
        "gencheck": d.get("gencheck"),
    }
    for key, arms in (("phases", ("a", "b")), ("phases_feh0", ("a",))):
        for arm in arms:
            for w in WIN:
                if w in d.get(key, {}):
                    s = _summary(d[key][w][arm])
                    out[f"{key}:{arm}:{w}"] = {
                        k: s[k]
                        for k in ("loga_hat", "lo1", "hi1", "open_lo1", "open_hi1", "unresolved1")
                    }
                    if arm == "b":
                        out[f"{key}:{arm}:{w}"]["Av"] = _av(d[key], w, "b")
    out["tension"] = d.get("tension")
    if null and d.get("tension"):
        dec = {}
        for arm, nk in (("a", "null:phases:a"), ("b", "null:phases:b"), ("c", "null:phases:a")):
            mu, sd = null[nk]["delta_mean"], null[nk]["delta_sd"]
            z = (d["tension"][arm]["delta_loga"] - mu) / sd
            dec[arm] = {
                "delta": d["tension"][arm]["delta_loga"],
                "z": float(z),
                "verdict": "tension" if abs(z) > 3 else ("none" if abs(z) < 2 else "marginal"),
            }
        if "phases" in d and "b" in d["phases"]["ms"]:
            dav = _av(d["phases"], "ms", "b") - _av(d["phases"], "pms", "b")
            n = null["null:dAv_ms_minus_pms:b"]
            dec["dAv_b"] = {"value": dav, "z": (dav - n["mean"]) / n["sd"]}
        out["decision"] = dec
    return out


def main() -> None:
    res = {"synth": {t: synth(t) for t in ("T1", "T2")}}
    null = res["synth"]["T1"]
    res["real"] = {g: real(g, null) for g in ("mist12", "mist25", "parsec")}
    c = [
        res["real"][g]["decision"]["c"]["delta"]
        for g in res["real"]
        if res["real"][g] and "decision" in res["real"][g]
    ]
    if c and null:
        res["grid_spread_c"] = {
            "range": float(max(c) - min(c)),
            "range_over_s0": float((max(c) - min(c)) / null["null:phases:a"]["delta_sd"]),
        }
    (HERE / "phases_summary.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
