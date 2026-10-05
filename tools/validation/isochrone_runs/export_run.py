#!/usr/bin/env python3
r"""Exporta una corrida de ajuste de isócronas a un JSON versionado, para ver evolucionar la isócrona.

POR QUÉ EXISTE
--------------
El autor quiere comparar la isócrona por versión del código y por modelo de grilla. Cada corrida
deja su resultado en un formato distinto (``ngc_0.json`` + cadenas por tramos en
``~/.cache/erotica-c1``, ablaciones en ``isochrone_nuts_convergence.json``, una tabla en P01), y
ninguno trae la curva en el CMD. Este script escribe un JSON por corrida con el esquema de
``README.md`` y lo deja en ``runs/``; la GUI ``~/phd-dashboard`` lo lee desde ``$EROTICA_RUNS``.

QUÉ NO HACE
-----------
No reimplementa la física. La curva sale de ``IsochroneFitter._interp_isochrone`` (interpolación
bilineal en log Z y log t a EEP fijo) y del desplazamiento por distancia y extinción con los
coeficientes CCM89 del propio ajustador (``_kG``, ``_k_col1``), las mismas dos líneas que
``IsochroneFitter._median_isochrone``; el test lo verifica contra ese método. Lo único escrito aquí
es la transformación de las cadenas del espacio no acotado al de los parámetros, que son las de
PyMC (intervalo: ``lo + (hi - lo) * sigmoid(z)``; log: ``exp(z)``; logodds: ``sigmoid(z)``), y se
verifica reproduciendo los cuantiles de ``ngc_0.json``, que salieron del grafo de PyMC.

⚠ Las curvas de corridas viejas se recalculan con la interpolación **actual** del paquete en la
mediana de esa corrida; la likelihood de esa corrida pudo usar otra (la grilla Hess de 507f779).
Cada JSON lo dice en ``isochrone.computed_with``.

QUÉ LO FALSARÍA
---------------
``tests/test_isochrone_runs.py``: cuantiles exportados ≠ los de ``ngc_0.json`` a 1e-5 (están
redondeados a 5 decimales), o curva de la mediana ≠ ``_median_isochrone`` del paquete.

USO
---
    OMP_NUM_THREADS=1 python tools/validation/isochrone_runs/export_run.py c1
    OMP_NUM_THREADS=1 python tools/validation/isochrone_runs/export_run.py hess507
    OMP_NUM_THREADS=1 python tools/validation/isochrone_runs/export_run.py p01

Necesita los ficheros MIST locales (``data/test/NGC6383/MIST/UBVRIplus``, rutas de esta máquina:
landmine del repo) y, para ``c1``, las cadenas en ``~/.cache/erotica-c1/c1/chain{0..3}``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VALIDATION = HERE.parent
REPO = VALIDATION.parent.parent
RUNS = HERE / "runs"
SCHEMA_VERSION = 2  # 2 (2026-10-04): top-level `grid` block; see README
PARAMS = ["met", "loga", "dm", "Av", "sigma_int", "f_bg"]
QS = (5, 16, 50, 84, 95)
C1_CHAINS = Path.home() / ".cache/erotica-c1/c1"
C1_SUMMARY = VALIDATION / "isochrone_unbinned/ngc_0.json"
# MIST v1.2 / Asplund+09 protosolar Z, the value P01 quotes (aa52082-24.tex:261)
Z_SUN_MIST = 0.0142
N_DRAWS_BAND = 400  # posterior draws used for the 16-84 band
N_DRAWS_CURVES = 30  # thinned draws stored as individual polylines


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True
    ).stdout.strip()


# --------------------------------------------------------------------------- data and fitter
def fitter(loga_range=(6.0, 7.0)):
    """The C1 configuration: same MIST files, same 254 stars, same priors."""
    sys.path.insert(0, str(VALIDATION))
    from astropy.table import QTable, Table
    from isochrone_nuts_convergence import MIST, PRIORS, SAMPLE

    from erotica.analysis._isochrone import IsochroneFitter

    pri = {k: v for k, v in PRIORS.items() if k not in ("M_met", "M_loga")}
    pri["loga_range"] = loga_range
    f = IsochroneFitter(isochs_path=MIST, **pri)
    table = QTable(Table.read(SAMPLE))
    f.setup(table, prob_threshold=0.0)
    return f, pri, table, MIST, SAMPLE


def observed_cmd(f, table) -> dict:
    """The points the likelihood used: G, BP-RP and the per-star errors after the error-model
    fill-in of :meth:`IsochroneFitter.setup` (which is what enters the likelihood)."""
    from erotica._membership import COLUMNA_ISOCRONA

    return {
        "x": "BP-RP",
        "y": "G",
        "source_id": [int(s) for s in table["source_id"]],
        "color": [round(float(v), 5) for v in f._obs_col],
        "mag": [round(float(v), 5) for v in f._obs_mag],
        "e_color": [round(float(v), 5) for v in f._e_obs_col],
        "e_mag": [round(float(v), 5) for v in f._e_obs_mag],
        "membership_prob": [round(float(v), 4) for v in table[COLUMNA_ISOCRONA]],
        "membership_column": COLUMNA_ISOCRONA,
        "mag_limit": float(f._mag_lim),
    }


def apparent_isochrone(f, p: dict):
    """Single-star isochrone at ``p`` in the observed system: the two lines of
    ``IsochroneFitter._median_isochrone``. Returns (eep_index, mass, G_app, col_app)."""
    met = p.get(getattr(f, "_met_name", "met"), p.get("met"))
    X = f._interp_isochrone(met, p["loga"], np)
    G = X[1] + p["dm"] + f._kG * p["Av"]
    col = X[2] + f._k_col1 * p["Av"]
    return np.arange(X.shape[-1]), X[0], G, col


def _keep(mass: np.ndarray) -> np.ndarray:
    """EEP points that carry the isochrone: drop the clamped tails, where the node lacks the EEP
    and the mass step is exactly zero (``_build_nodes``)."""
    step = np.diff(mass)
    keep = np.r_[True, step > 0] & np.r_[step > 0, True]
    return keep


def isochrone_block(f, median: dict, draws: dict | None, rng, computed_with: str) -> dict:
    eep, mass, G, col = apparent_isochrone(f, median)
    k = _keep(mass)
    out = {
        "computed_with": computed_with,
        "definition": "single-star isochrone (no binaries, no errors) at the posterior median of "
        "met/loga/dm/Av, in the observed system: G and BP-RP with CCM89+O'Donnell extinction, R_V=3.1",
        "median": {
            "eep": [int(v) for v in eep[k]],
            "mass": [round(float(v), 5) for v in mass[k]],
            "color": [round(float(v), 5) for v in col[k]],
            "mag": [round(float(v), 5) for v in G[k]],
        },
        "band_16_84": None,
        "draws": None,
    }
    if draws is None:
        return out
    n = len(draws["met"])
    idx = rng.choice(n, size=min(N_DRAWS_BAND, n), replace=False)
    Gs, Cs = [], []
    for i in idx:
        _, _, g, c = apparent_isochrone(
            f, {p: float(draws[p][i]) for p in ("met", "loga", "dm", "Av")}
        )
        Gs.append(g[k])
        Cs.append(c[k])
    Gs, Cs = np.array(Gs), np.array(Cs)
    out["band_16_84"] = {
        "definition": "per EEP point (same EEP indices as median), 16th and 84th percentiles of "
        f"colour and of G separately over {len(idx)} posterior draws; not a band at fixed G, "
        "because a ~1 Myr isochrone is not monotonic in G at the PMS turn-on",
        "n_draws": int(len(idx)),
        "color_q16": [round(float(v), 5) for v in np.percentile(Cs, 16, axis=0)],
        "color_q84": [round(float(v), 5) for v in np.percentile(Cs, 84, axis=0)],
        "mag_q16": [round(float(v), 5) for v in np.percentile(Gs, 16, axis=0)],
        "mag_q84": [round(float(v), 5) for v in np.percentile(Gs, 84, axis=0)],
    }
    sel = idx[:N_DRAWS_CURVES]
    out["draws"] = []
    for i in sel:
        p = {q: float(draws[q][i]) for q in ("met", "loga", "dm", "Av")}
        _, _, g, c = apparent_isochrone(f, p)
        out["draws"].append(
            {
                "params": {q: round(v, 6) for q, v in p.items()},
                "color": [round(float(v), 4) for v in c[k][::3]],
                "mag": [round(float(v), 4) for v in g[k][::3]],
            }
        )
    return out


# --------------------------------------------------------------------------- posterior
def chain_bounds(f, pri) -> dict:
    return {
        "met": (10.0 ** float(f._node_logz[0]), 10.0 ** float(f._node_logz[-1])),
        "loga": tuple(pri["loga_range"]),
        "dm": tuple(pri["dm_range"]),
        "Av": tuple(pri["Av_range"]),
    }


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def transform(value_vars: list[str], z: list[np.ndarray], bounds: dict) -> dict:
    """Unconstrained draws -> parameters, with PyMC's default transforms."""
    out = {}
    for name, v in zip(value_vars, z, strict=True):
        kind = name
        if kind.endswith("_interval__"):
            base = kind[: -len("_interval__")]
            lo, hi = bounds[base]
            out[base] = lo + (hi - lo) * _sigmoid(v)
        elif kind.endswith("_log__"):
            base = kind[: -len("_log__")]
            out[base] = np.exp(v)
        elif kind.endswith("_logodds__"):
            base = kind[: -len("_logodds__")]
            out[base] = _sigmoid(v)
        else:
            raise ValueError(f"unknown transform in {name!r}")
    return out


def load_chunked_chains(root: Path, n_chains: int = 4):
    """Post-warmup draws and stats from ``isochrone_c1_chunked.py run`` directories, in chain order
    (the same concatenation as its ``_load_draws``)."""
    zs, stats, cfg0 = [], [], None
    keys = ("num_steps", "step_size", "accept_prob", "diverging", "potential_energy", "energy")
    for c in range(n_chains):
        od = root / f"chain{c}"
        cfg = json.loads((od / "config.json").read_text())
        cfg0 = cfg0 or cfg
        parts = [np.load(p) for p in sorted(od.glob("chunk_*.npz"))]
        nv = len(cfg["value_vars"])
        z = [np.concatenate([p[f"z{i}"] for p in parts], axis=1) for i in range(nv)]
        st = {k: np.concatenate([p[k] for p in parts], axis=1) for k in keys}
        w, total = cfg["warmup"], cfg["warmup"] + cfg["draws"]
        if z[0].shape[1] < total:
            raise SystemExit(f"{od}: only {z[0].shape[1]} of {total} iterations on disk")
        zs.append([v[:, w:total] for v in z])
        stats.append({k: v[:, w:total] for k, v in st.items()})
    z = [np.concatenate([zz[i] for zz in zs], axis=0) for i in range(len(zs[0]))]
    st = {k: np.concatenate([s[k] for s in stats], axis=0) for k in stats[0]}
    return cfg0, z, st


def summarize(post: dict, st: dict | None) -> tuple[dict, dict]:
    """Quantiles, R-hat and bulk ESS with arviz (``az.rhat``/``az.ess``, not ``az.summary``,
    which rounds at the threshold), and the sampler diagnostics, as ``_summ`` computes them."""
    import arviz as az
    from arviz_base import from_dict

    idata = from_dict({"posterior": post})
    rh, eb = az.rhat(idata.posterior), az.ess(idata.posterior, method="bulk")
    posterior = {}
    for p in PARAMS:
        a = np.asarray(post[p])
        q = np.percentile(a.ravel(), QS)
        posterior[p] = {
            "q05": float(q[0]),
            "q16": float(q[1]),
            "q50": float(q[2]),
            "q84": float(q[3]),
            "q95": float(q[4]),
            "rhat": float(np.asarray(rh[p])),
            "ess_bulk": float(np.asarray(eb[p])),
            "chain_means": [float(v) for v in a.mean(1)],
        }
    diag = None
    if st is not None:
        E = st["energy"]
        diag = {
            "chains": int(E.shape[0]),
            "draws_per_chain": int(E.shape[1]),
            "divergences": int(st["diverging"].sum()),
            "bfmi": [
                float(v) for v in np.mean(np.diff(E, axis=1) ** 2, axis=1) / np.var(E, axis=1)
            ],
            "tree_depth_chain_mean": [
                float(v) for v in (np.log2(st["num_steps"]).astype(int) + 1).mean(1)
            ],
            "step_size_chain_mean": [float(v) for v in st["step_size"].mean(1)],
        }
    return posterior, diag


def _met_note(z: float) -> str:
    return f"met is linear Z (MIST Zinit); [M/H] = log10(Z/{Z_SUN_MIST}) = {np.log10(z / Z_SUN_MIST):+.2f}"


def _write(run: dict) -> Path:
    RUNS.mkdir(exist_ok=True)
    p = RUNS / f"{run['id']}.json"
    p.write_text(json.dumps(run, indent=1) + "\n")
    return p


LEGACY_GRID = {
    "name": "MIST v1.2",
    "family": "MIST",
    "version": "1.2",
    "eep_kind": "native",
    "path": "legacy isochs_path (MISTIsochrones)",
    "metallicity_parameter": "met = linear Z (Zinit), uniform in Z",
    "bands": ["Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"],
}


def _base(f, pri, table, mist, sample) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "grid": dict(LEGACY_GRID),
        "cluster": "NGC 6383",
        "model": {
            "grid": "MIST",
            "version": "v1.2",
            "files": f"{Path(mist).name}: MIST_v1.2_feh_*_afe_p0.0_vvcrit0.4_UBVRIplus.iso.cmd",
            "photometry": "Gaia EDR3 G, BP, RP",
            "z_sun": Z_SUN_MIST,
            "z_sun_source": "Asplund+09 protosolar, MIST v1.2 (Choi+2016)",
        },
        "cmd": observed_cmd(f, table),
        "sample": str(Path(sample).name),
    }


# --------------------------------------------------------------------------- the runs
def export_c1() -> Path:
    f, pri, table, mist, sample = fitter()
    cfg, z, st = load_chunked_chains(C1_CHAINS)
    post = transform(cfg["value_vars"], z, chain_bounds(f, pri))
    posterior, diag = summarize(post, st)
    summary = json.loads(C1_SUMMARY.read_text())
    flat = {p: post[p].ravel() for p in PARAMS}
    med = {p: posterior[p]["q50"] for p in PARAMS}
    run = {
        "id": "c1-ngc6383-unbinned-eep-v1-96ce4c0",
        "date": "2026-09-30",
        "erotica_commit": summary["head"],
        "result_commit": "96ce4c0",
        "method": "NUTS (numpyro kernel via isochrone_c1_chunked.py), unbinned per-star likelihood, "
        "EEP-interpolated isochrones (IsochroneFitter.LIKELIHOOD_VERSION unbinned-eep-v1)",
        **_base(f, pri, table, mist, sample),
        "config": {
            "priors": {
                "met": f"Uniform{tuple(round(v, 7) for v in chain_bounds(f, pri)['met'])} (linear Z)",
                "loga": f"Uniform{tuple(pri['loga_range'])}",
                "dm": f"TruncatedNormal(mu={pri['dm_mu']}, sigma={f.dm_sigma}, {tuple(pri['dm_range'])})",
                "Av": f"Uniform{tuple(pri['Av_range'])}",
                "sigma_int": "HalfNormal(0.05)",
                "f_bg": "Beta(1, 19)",
            },
            "binaries": {"alpha": f.alpha, "beta": f.beta, "q": "Duchene & Kraus 2013"},
            "imf": "Chabrier 2014",
            "Rv": f.Rv,
            "sigma_floor": f.SIGMA_FLOOR,
            "sampler": {
                k: summary["config"][k]
                for k in ("warmup", "draws", "target_accept", "seed", "mass")
            },
            "chains": 4,
            "start": "find_start (search), see ngc_0.json -> search",
        },
        "posterior": posterior,
        "diagnostics": diag,
        "isochrone": isochrone_block(
            f,
            med,
            flat,
            np.random.default_rng(20261003),
            f"erotica IsochroneFitter._interp_isochrone at {_git('rev-parse', '--short', 'HEAD')}",
        ),
        "notes": [
            _met_note(med["met"]),
            "posterior loga sits against the lower prior bound 6.0 (q05 6.006); see the hub finding "
            "agent-findings/isochrone-c1-age-review.md",
            "source of record: tools/validation/isochrone_unbinned/ngc_0.json; chains: "
            "~/.cache/erotica-c1/c1/chain0..3 (not in git)",
            "C3 (binary injection-recovery) not met: loga biased +0.013 to +0.024 dex",
        ],
    }
    return _write(run)


def export_hess507() -> Path:
    """The precomputed-Hess likelihood at 507f779, 4 chains: converged (R-hat <= 1.0021) but locked
    onto the grid's internal reference -- convergence is not calibration. Summary only."""
    f, pri, table, mist, sample = fitter()
    src = json.loads((VALIDATION / "isochrone_nuts_convergence.json").read_text())
    a = src["ablation"]["new_frame_interp_4ch"]
    posterior = {}
    for p, key in (
        ("met", "met"),
        ("loga", "loga"),
        ("dm", "dm"),
        ("Av", "Av"),
        ("sigma_int", None),
        ("f_bg", None),
    ):
        if key is None:
            posterior[p] = None
            continue
        q = a[key]["q05_16_50_84_95"]
        posterior[p] = dict(zip(("q05", "q16", "q50", "q84", "q95"), q, strict=True)) | {
            "rhat": a[key]["rhat"],
            "ess_bulk": a[key]["ess_bulk"],
            "chain_means": a[key]["chain_means"],
        }
    med = {p: posterior[p]["q50"] for p in ("met", "loga", "dm", "Av")}
    run = {
        "id": "hess-ngc6383-binned-507f779",
        "date": "2026-09-22",
        "erotica_commit": "507f779",
        "method": "NUTS (numpyro), precomputed binned Hess-diagram Poisson likelihood "
        "(new frame, node interpolation); replaced the same day by the unbinned likelihood",
        **_base(f, pri, table, mist, sample),
        "config": {
            "priors": "as C1 for met/loga/dm/Av; nuisance log_s and bg instead of sigma_int and f_bg",
            "chains": 4,
            "nuisance": {k: a[k] for k in ("log_s", "bg")},
        },
        "posterior": posterior,
        "diagnostics": {
            k: a[k]
            for k in ("divergences", "bfmi", "tree_depth_chain_mean", "step_size_chain_mean")
        }
        | {"chains": 4},
        "isochrone": isochrone_block(
            f,
            med,
            None,
            None,
            "curve recomputed at this run's median with the CURRENT EEP interpolation "
            f"({_git('rev-parse', '--short', 'HEAD')}); the run's own likelihood used the Hess grid of 507f779",
        ),
        "notes": [
            _met_note(med["met"]),
            "converged but not calibrated: the posterior locked onto the grid's internal reference "
            "(dm_mu, mean(Av_range)) and injection-recovery failed (dm outside the 90% interval in 16/16); "
            "hub finding isochrone-nuts-convergence-2026-09.md section 0",
            "source: tools/validation/isochrone_nuts_convergence.json -> ablation.new_frame_interp_4ch; "
            "no chains kept, so no 16-84 band",
        ],
    }
    return _write(run)


def export_p01() -> Path:
    """The published P01 values (ASteCA + MIST v1.2, DEMetropolis, not converged): a reference
    point, not a posterior. aa52082-24.tex Table 1 and Sect. 5.1."""
    f, pri, table, mist, sample = fitter()
    point = {"met": 0.024, "loga": 6.55, "dm": 10.30, "Av": 1.24}
    posterior = {p: None for p in PARAMS}
    posterior.update(
        {
            "met": {"q50": 0.024, "pm": 0.008},
            "loga": {"q50": 6.55, "plus": 0.40, "minus": 0.20},
            "dm": {"q50": 10.30, "pm": 0.09},
            "Av": {"q50": 1.24, "pm": 0.26},
        }
    )
    run = {
        "id": "p01-ngc6383-asteca-demetropolis",
        "date": None,  # the ASteCA run is undated in the paper; 2026-09-15 was only the file rename
        "erotica_commit": None,
        "method": "ASteCA synthetic-cluster likelihood sampled with DEMetropolis (300 chains x 1000 "
        "after 1500); NOT converged per P01 itself; 'q50' is the mode",
        **_base(f, pri, table, mist, sample),
        "config": {
            "priors": "Z U(0.001,0.045), loga U(6,7), Av U(0.5,2), dm N(10.3,0.2) (P01 Sect. 5.1)"
        },
        "posterior": posterior,
        "diagnostics": None,
        "isochrone": isochrone_block(
            f,
            point,
            None,
            None,
            "curve recomputed at P01's mode with erotica's EEP interpolation "
            f"({_git('rev-parse', '--short', 'HEAD')}); P01's own curves came from ASteCA",
        ),
        "notes": [
            _met_note(point["met"]),
            "source: ~/paper-ngc6383-aa52082-24/submission_package/clean_source/aa52082-24.tex "
            "Table 1 and lines 258-271; the age interval +0.40/-0.20 is a stated systematic, not a posterior width",
        ],
    }
    return _write(run)


def migrate_v1() -> list[Path]:
    """Schema 1 -> 2 without re-running anything: add the ``grid`` block (every schema-1 run went
    through the legacy MIST v1.2 path or, for P01, ASteCA on MIST v1.2) and bump the version."""
    out = []
    for p in sorted(RUNS.glob("*.json")):
        run = json.loads(p.read_text())
        if run.get("schema_version") != 1:
            continue
        grid = dict(LEGACY_GRID)
        if run["id"].startswith("p01-"):
            grid["path"] = "ASteCA 0.6.9 reading MIST v1.2 (not EROTICA)"
        run = {
            "schema_version": 2,
            "grid": grid,
            **{k: v for k, v in run.items() if k != "schema_version"},
        }
        p.write_text(json.dumps(run, indent=1) + "\n")
        out.append(p)
    return out


def export_grids() -> list[Path]:
    """NGC 6383 with every grid backend, from ``isochrone_grids/ngc6383_grids.json``: maximum
    likelihood points with Laplace widths (NOT posteriors), same 254 stars as C1, so the GUI can
    show the same cluster with several grids side by side."""
    sys.path.insert(0, str(VALIDATION))
    sys.path.insert(0, str(VALIDATION / "isochrone_grids"))
    from astropy.table import QTable, Table
    from isochrone_nuts_convergence import PRIORS, SAMPLE
    from ngc6383_grids import CACHE, NGC

    from erotica.analysis import grids as G
    from erotica.analysis._isochrone import IsochroneFitter

    src = json.loads((VALIDATION / "isochrone_grids/ngc6383_grids.json").read_text())
    pri = {k: v for k, v in PRIORS.items() if k not in ("M_met", "M_loga")}
    table = QTable(Table.read(SAMPLE))
    lr = (5.9, 7.1)
    builders = {
        "MIST v1.2": lambda fehs: G.MISTGrid(NGC / "MIST/UBVRIplus", loga_range=lr).select(
            feh=fehs
        ),
        "MIST v2.5": lambda fehs: G.MISTGrid(
            CACHE / "mist_v2.5/UBVRIplus_afe0_vvcrit0.4", loga_range=lr
        ).select(feh=fehs),
        "PARSEC v1.2S": lambda fehs: G.PARSECGrid(
            NGC / "PARSEC/gaiaedr3", loga_range=lr, max_mass=20.0
        ),
    }
    head = _git("rev-parse", "--short", "HEAD")
    out = []
    for name, fit in src["arms"].get("main", {}).items():
        d = fit["grid"]
        g = builders[name](d["feh_nodes"])
        b = fit["bands"]
        f = IsochroneFitter(grid=g, magnitude=b[0], color=(b[1], b[2]), **pri)
        f.setup(table, prob_threshold=0.0)
        m = fit["mode"]
        posterior = {k: {"q50": float(m[k]), "laplace_sd": fit["laplace_sd"].get(k)} for k in m}
        slug = name.lower().replace(" ", "-").replace(".", "")
        run = {
            "schema_version": SCHEMA_VERSION,
            "grid": {
                **d,
                "name": d["grid"],
                "path": "erotica.analysis.grids (grid=)",
                "metallicity_parameter": "feh = the grid's own [Fe/H] label, uniform in it",
            },
            "id": f"map-ngc6383-{slug}-{head}",
            "date": "2026-10-04",
            "erotica_commit": head,
            "cluster": "NGC 6383",
            "model": {
                "grid": d["family"],
                "version": d["version"],
                "files": d["grid"],
                "photometry": ", ".join(b),
                "z_sun": None,
                "z_sun_source": "not used: the fit coordinate is the [Fe/H] label",
            },
            "method": "maximum likelihood (IsochroneFitter.find_start: node lattice + L-BFGS-B), "
            "Laplace widths at the mode; NOT a posterior",
            "config": {
                "priors": {k: list(v) if isinstance(v, tuple) else v for k, v in pri.items()},
                "binaries": {"alpha": f.alpha, "beta": f.beta},
                "sigma_floor": f.SIGMA_FLOOR,
                "at_prior_bound": fit["at_prior_bound"],
            },
            "posterior": posterior,
            "diagnostics": None,
            "cmd": observed_cmd(f, table),
            "sample": str(Path(SAMPLE).name),
            "isochrone": isochrone_block(
                f, m, None, None, f"erotica IsochroneFitter(grid=) at {head}"
            ),
            "notes": [
                f"mode at prior bound in {fit['at_prior_bound']}"
                if fit["at_prior_bound"]
                else "mode inside the prior box",
                "source: tools/validation/isochrone_grids/ngc6383_grids.json",
            ],
        }
        out.append(_write(run))
    return out


if __name__ == "__main__":
    which = sys.argv[1:] or ["c1", "hess507", "p01"]
    for w in which:
        print(
            {
                "c1": export_c1,
                "hess507": export_hess507,
                "p01": export_p01,
                "migrate": migrate_v1,
                "grids": export_grids,
            }[w]()
        )
