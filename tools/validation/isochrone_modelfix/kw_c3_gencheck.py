#!/usr/bin/env python3
"""Population check of the independent generator of ``kw_c3.py`` against the likelihood's own deposit
(hub finding ``isochrone-intervals-nuts.md`` §2.17), noiseless, in the window 14 <= G <= 19 and at the
truths of cell A replicate 0. Two-sample KS on G and on colour:

* singles: independent vs deposit (alpha = beta = 0) -- must NOT reject (p >= 0.01);
* binaries: independent vs deposit at BINARY_STRIDE 3 (production) and 1;
* positive control: independent WITH binaries vs deposit WITHOUT -- must reject (the test has power
  for the effect the experiment is about).

Writes ``kw_c3_gencheck.json``."""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aniso as A  # noqa: E402
import kw_c3 as X  # noqa: E402
import kw_cells as K  # noqa: E402

C = A.C


def win(G, c):
    m = (G >= 14) & (G <= 19)
    return G[m], c[m]


def ks(a, b):
    r = [stats.ks_2samp(a[i], b[i]) for i in (0, 1)]
    return {"G_D": float(r[0].statistic), "G_p": float(r[0].pvalue), "col_D": float(r[1].statistic),
            "col_p": float(r[1].pvalue), "n_ind": int(a[0].size), "n_dep": int(b[0].size)}


def main(n=60000):
    real = C.real_table()
    gen = A.make_fitter(real, "F")
    th = K.truth_for("A", 0, np.random.default_rng(6000))
    args = (th["feh"], th["loga"], th["dm"], th["Av"], n)

    def ind(a, b):
        return win(*X.ind_true_stars(gen._isochs, gen._kG, gen._kBP, gen._kRP, *args,
                                     np.random.default_rng(1), a, b))

    def dep(a, b, stride):
        g = A.make_fitter(real, "F")
        g.alpha, g.beta, g.BINARY_STRIDE = a, b, stride
        return win(*C._true_stars(g, *args, np.random.default_rng(2)))

    al, be = gen.alpha, gen.beta
    out = {"truth": {k: th[k] for k in ("feh", "loga", "dm", "Av")}, "n": n,
           "singles": ks(ind(0, 0), dep(0, 0, 3)),
           "binaries_stride3": ks(ind(al, be), dep(al, be, 3)),
           "binaries_stride1": ks(ind(al, be), dep(al, be, 1)),
           "positive_control_binaries_vs_singles": ks(ind(al, be), dep(0, 0, 3))}
    (HERE / "kw_c3_gencheck.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
