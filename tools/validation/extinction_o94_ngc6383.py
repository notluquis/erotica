#!/usr/bin/env python3
r"""CCM89 -> CCM89 + O'Donnell (1994): the coefficient shift and the closed-form map of a fit.

WHY THIS EXISTS
---------------
The 2026-11-05 entry of ``docs/design-notes/decisions.md`` ("``"ccm89_o94"``: O'Donnell (1994) in
the optical") quotes three kinds of number:

1. the change of :math:`k_G, k_{BP}, k_{RP}` (and :math:`k_{BP} - k_{RP}`) at the fitter's Gaia
   effective wavelengths and :math:`R_V = 3.1`;
2. where a stored CCM89 optimum of NGC 6383 lands under the closed-form map
   :math:`A_V' = A_V k_{col}/k_{col}'`, :math:`dm' = dm + k_G A_V - k_G' A_V'`;
3. that the log-likelihood is the same at the mapped point ("the dm prior is the only term that
   breaks the identity").

The entry first pointed only at a hub finding (``~/phd/agent-findings/extinction-o94-2026-11-05.md``),
whose likelihood is a two-age, empirical-field frame of a hub script. This script regenerates all
three from this repository, with ``IsochroneFitter``'s own wavelengths, ``_ccm89`` and ``loglike``.

WHAT IS RECOMPUTED AND WHAT IS QUOTED
-------------------------------------
* (1) and (2) are recomputed; (2) takes as input the stored CCM89 optimum of the hub's metallicity
  tornado (``ngc6383_feh_tornado/ccm_free.json`` "free", 2026-10-20), embedded below as data. When
  the hub is present its file is read and must equal the embedded values, and the mapped point is
  compared with the hub's own ``o94_free.json`` "control_mapped_params".
* (3) is recomputed on this fitter's single-age ``loglike`` (uniform background, ``f_bg`` and
  ``sigma_int`` at the stored values) for the 254 NGC 6383 members of the P01 reference sample.
  That is a different log L from the hub frame's (-545.27...), which adds a second age and an
  empirical field density; the identity holds in both because both apply the extinction law only
  as the constant shifts ``(dm + k_G A_V, k_col A_V)`` (``IsochroneFitter._deposit``). The hub
  value is quoted in the sidecar, not recomputed. The structural claim is also a test that runs in
  CI on synthetic data: ``tests/test_isochrone.py::TestODonnell94::test_constant_laws_are_one_reparametrisation``.

WHAT WOULD FALSIFY IT
---------------------
A |Delta log L| at the mapped point above floating-point noise (gate: 1e-9 on a log L of order
-500), which would mean the likelihood reads A_V or dm somewhere other than through those two
shifts. Controls, run and reported: O94 at the *unmapped* CCM89 point, and the map with only the dm
shift (A_V not rescaled, the mutation the hub saw red). Both must move log L by far more than the
gate, or the identity check could not fail.

INPUTS (local, untracked, as in ``isochrone_nuts_convergence.py``)
------------------------------------------------------------------
``data/test/NGC6383/MIST/UBVRIplus`` and
``data/test/NGC6383/comments_paper/radius_robustness/generated/40/paperfaithful_reference_p06.ecsv``
(N = 254). Without them stage (3) is skipped and the sidecar says so; (1) and (2) still run.

    python tools/validation/extinction_o94_ngc6383.py      # writes extinction_o94_ngc6383.json
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from erotica.analysis._isochrone import IsochroneFitter, _ccm89  # noqa: E402

B = REPO / "data" / "test" / "NGC6383"
if not (B / "MIST").is_dir():  # a worktree: the untracked inputs live in the main checkout
    B = Path("/Users/notluquis/erotica/data/test/NGC6383")
MIST = B / "MIST" / "UBVRIplus"
SAMPLE = B / "comments_paper/radius_robustness/generated/40/paperfaithful_reference_p06.ecsv"
HUB = Path(os.path.expanduser("~/phd/agent-findings/scripts/ngc6383_feh_tornado"))
OUT = HERE / "extinction_o94_ngc6383.json"
GATE = 1e-9

# The stored CCM89 optimum ("free", start S2) of the hub's metallicity tornado, 2026-10-20:
# ~/phd/agent-findings/scripts/ngc6383_feh_tornado/ccm_free.json. Quoted, full precision.
CCM_OPT = {
    "met": 0.004640373382362379,
    "loga": 5.98555294680986,
    "dm": 10.212464101476325,
    "Av": 0.7586135468774481,
    "sigma_int": 0.13549275115159165,
    "f_bg": 0.17000000000000004,
}
HUB_FRAME_LOGL = -545.2738871479884  # the hub frame's log L there, and the O94 one at the map


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def coefficients() -> dict:
    f = IsochroneFitter(isochs_path=".", loga_range=(6.0, 7.0), dm_mu=10.0, dm_range=(9.0, 11.0))
    lams = {"G": f.magnitude_effl, "BP": f.color_effl[0], "RP": f.color_effl[1]}
    k = {
        law: {b: _ccm89(lam, f.Rv, opt) for b, lam in lams.items()}
        for law, opt in (("ccm89", "ccm89"), ("ccm89_o94", "o94"))
    }
    for law in k:
        k[law]["BP-RP"] = k[law]["BP"] - k[law]["RP"]
    rel = {b: 100.0 * (k["ccm89_o94"][b] / k["ccm89"][b] - 1.0) for b in k["ccm89"]}
    return {"Rv": f.Rv, "effl_AA": lams, "k": k, "o94_vs_ccm89_percent": rel}


def mapped(p: dict, kc: dict, ko: dict, rescale_av: bool = True) -> dict:
    """The CCM89 point that gives the same (dm + k_G A_V, k_col A_V) under O94."""
    q = dict(p)
    q["Av"] = p["Av"] * kc["BP-RP"] / ko["BP-RP"] if rescale_av else p["Av"]
    q["dm"] = p["dm"] + kc["G"] * p["Av"] - ko["G"] * q["Av"]
    return q


def hub_check(m: dict) -> dict:
    ccm, o94 = HUB / "ccm_free.json", HUB / "o94_free.json"
    if not (ccm.exists() and o94.exists()):
        return {"skipped": f"hub files absent ({HUB})"}
    stored = json.loads(ccm.read_text())["free"]
    assert {n: stored["params"][n] for n in CCM_OPT} == CCM_OPT, "embedded optimum != hub file"
    assert stored["logL"] == HUB_FRAME_LOGL
    hub_map = json.loads(o94.read_text())["control_mapped_params"]
    d = {n: abs(hub_map[n] - m[n]) for n in ("Av", "dm")}
    assert max(d.values()) < 1e-12, d
    return {
        "ccm_free.json_sha256": _sha256(ccm),
        "o94_free.json_sha256": _sha256(o94),
        "max_abs_diff_mapped_vs_hub": max(d.values()),
        "hub_control_mapped_logL": json.loads(o94.read_text())["control_mapped_logL"],
    }


def loglike_identity(kc: dict, ko: dict) -> dict:
    if not (MIST.is_dir() and SAMPLE.exists()):
        return {"skipped": f"untracked inputs absent ({MIST}, {SAMPLE})"}
    from astropy.table import QTable, Table

    data = QTable(Table.read(SAMPLE))
    kw = dict(
        isochs_path=MIST,
        loga_range=(5.0, 7.0),
        Av_range=(0.0, 3.0),
        dm_mu=10.3,
        dm_sigma=0.2,
        dm_range=(9.5, 10.7),
    )
    fit = {}
    for law in ("ccm89", "ccm89_o94"):
        fit[law] = IsochroneFitter(extinction_law=law, **kw)
        fit[law].setup(data, prob_threshold=0.0)
    p = CCM_OPT
    m = mapped(p, kc, ko)
    m_dm_only = mapped(p, kc, ko, rescale_av=False)

    def ll(law: str, q: dict) -> float:
        return fit[law].loglike(q["met"], q["loga"], q["dm"], q["Av"], q["sigma_int"], q["f_bg"])

    ref = ll("ccm89", p)
    res = {
        "N": int(fit["ccm89"]._N_obs),
        "sample_sha256": _sha256(SAMPLE),
        "logL_ccm89_at_optimum": ref,
        "logL_o94_at_mapped": ll("ccm89_o94", m),
        "control_logL_o94_at_unmapped": ll("ccm89_o94", p),
        "control_logL_o94_dm_shift_only": ll("ccm89_o94", m_dm_only),
    }
    res["abs_delta_mapped"] = abs(res["logL_o94_at_mapped"] - ref)
    res["abs_delta_unmapped"] = abs(res["control_logL_o94_at_unmapped"] - ref)
    res["abs_delta_dm_shift_only"] = abs(res["control_logL_o94_dm_shift_only"] - ref)
    res["gate"] = GATE
    assert res["abs_delta_mapped"] < GATE, res
    assert min(res["abs_delta_unmapped"], res["abs_delta_dm_shift_only"]) > 1e3 * GATE, res
    return res


def main() -> None:
    c = coefficients()
    kc, ko = c["k"]["ccm89"], c["k"]["ccm89_o94"]
    m = mapped(CCM_OPT, kc, ko)
    out = {
        "script": "tools/validation/extinction_o94_ngc6383.py",
        "coefficients": c,
        "ccm89_optimum_quoted": {
            "source": "~/phd/agent-findings/scripts/ngc6383_feh_tornado/"
            "ccm_free.json ['free'] (2026-10-20)",
            "params": CCM_OPT,
            "hub_frame_logL": HUB_FRAME_LOGL,
        },
        "mapped_o94": {"Av": m["Av"], "dm": m["dm"]},
        "hub": hub_check(m),
        "loglike_identity": loglike_identity(kc, ko),
    }
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    r = c["o94_vs_ccm89_percent"]
    print("k O94 vs CCM89 (%): " + ", ".join(f"{b} {v:+.2f}" for b, v in r.items()))
    print(
        f"mapped: A_V {CCM_OPT['Av']:.4f} -> {m['Av']:.4f}, dm {CCM_OPT['dm']:.4f} -> {m['dm']:.4f}"
    )
    li = out["loglike_identity"]
    if "skipped" in li:
        print("log L identity SKIPPED:", li["skipped"])
    else:
        print(
            f"log L identity (N = {li['N']}): |dlogL| mapped {li['abs_delta_mapped']:.2e}, "
            f"unmapped {li['abs_delta_unmapped']:.3f}, dm shift only {li['abs_delta_dm_shift_only']:.3f}"
        )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
