import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from d1_proto import AGES, KC, KG, ZS, Data, Phi, isochrone, mle, synthetic_cluster  # noqa: E402


def make_binned(fs):
    def ll(D, th, zs=ZS, ages=AGES, bg=0.01, binaries=True):
        met, loga, dm, Av = th
        mass, w, bp, G, col, Gb, cb = isochrone(met, loga, zs, ages, binaries)
        lam = 0
        for GG, cc, ww in ((G, col, w * (1 - bp)), (Gb, cb, w * bp)):
            Ga, ca = GG + dm + KG * Av, cc + KC * Av
            em, ec = D.sig(Ga)
            sm = np.sqrt(em**2 + fs * D.bw_m**2)
            sc = np.sqrt(ec**2 + fs * D.bw_c**2)
            Pm = np.diff(Phi((D.em[:, None] - Ga[None]) / sm[None]), axis=0)
            Pc = np.diff(Phi((D.ec[:, None] - ca[None]) / sc[None]), axis=0)
            lam = lam + (Pm * ww) @ Pc.T
        lam = np.maximum(lam, 1e-12)
        n = D.H
        s = n.sum() / lam.sum()
        for _ in range(20):
            mu = s * lam + bg
            s *= np.sum(n * lam / mu) / lam.sum()
        mu = s * lam + bg
        return float(np.sum(n * np.log(mu) - mu))

    return ll


class DataFine(Data):
    def __init__(self, t, k):
        super().__init__(t)
        nm = len(self.em) - 3
        nc = len(self.ec) - 3
        g0 = self.G.min()
        c0 = self.c.min()
        self.bw_m /= k
        self.bw_c /= k
        self.em = g0 + self.bw_m * np.arange(-2 * k, nm * k + 1)
        self.ec = c0 + self.bw_c * np.arange(-k, nc * k + 2 * k)
        self.H, _, _ = np.histogram2d(self.G, self.c, bins=[self.em, self.ec])


out = {}
for vname, fs, k in (
    ("fs0.5_k1", 0.5, 1),
    ("fs0_k1", 0.0, 1),
    ("fs0.05_k1", 0.05, 1),
    ("fs0.5_k3", 0.5, 3),
    ("fs0_k3", 0.0, 3),
):
    fn = make_binned(fs)
    rows = []
    for seed in (1, 2, 3):
        t = synthetic_cluster(0.014286, 6.55, 10.47, 1.10, seed, binaries=False)
        D = Data(t) if k == 1 else DataFine(t, k)
        truth = np.array([0.014286, 6.55, 10.47, 1.10])
        xh, lh = mle(fn, D, [truth, np.array([0.0143, 6.5, 10.3, 1.25])], binaries=False)
        rows.append(np.round(xh, 4).tolist() + [round(lh - fn(D, truth, binaries=False), 2)])
    out[vname] = rows
    print(vname, rows, flush=True)
json.dump(out, open(HERE / "d1_variants.json", "w"), indent=1)
