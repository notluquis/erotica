#!/usr/bin/env python3
"""False-fail rate of the per-chain split R-hat clause of the convergence gate (prereg §0.4:
max over 6 parameters x 4 chains <= 1.01) for a PERFECT sampler: stationary AR(1) chains of 500
draws whose per-chain ESS matches the measured one (bulk ESS 840-1800 over 4 chains, i.e.
~210-450 per chain). Writes gate_noise.json."""
import json
from pathlib import Path

import arviz as az
import numpy as np

rng = np.random.default_rng(1)
out = {}
for ess_chain in (210, 300, 450):
    # AR(1) with ESS/N = (1-phi)/(1+phi)
    r = ess_chain / 500
    phi = (1 - r) / (1 + r)
    fails = []
    for _ in range(400):
        mx = 0.0
        for _p in range(6):
            e = rng.normal(size=(4, 500))
            x = np.empty_like(e)
            x[:, 0] = e[:, 0] / np.sqrt(1 - phi**2)
            for t in range(1, 500):
                x[:, t] = phi * x[:, t - 1] + e[:, t]
            for c in range(4):
                mx = max(mx, float(az.rhat(x[c].reshape(2, -1))))
        fails.append(mx > 1.01)
    out[str(ess_chain)] = {"phi": phi, "false_fail_rate": float(np.mean(fails)), "n_sim": 400}
    print(ess_chain, out[str(ess_chain)], flush=True)
Path(__file__).with_name("gate_noise.json").write_text(json.dumps(out, indent=1) + "\n")
