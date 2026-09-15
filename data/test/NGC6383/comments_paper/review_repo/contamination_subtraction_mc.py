"""Round 3 (2026-09-14). Does P01's mass-segregation signal survive removing the field contamination that
its own King fit predicts? The King background b (stars arcmin^-2, Table 1: 0.011 +- 0.007) of the 40'
reference sample implies b * area background stars per radial annulus. Each Monte Carlo realization removes
that many stars at random from each annulus (never the 20 most massive), then recomputes Lambda_MSR with
its permutation p (P01's null) and the single-vs-binary KS. Seeded; env erotica-bench."""
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import ks_2samp

B = "/Users/notluquis/erotica/data/test/NGC6383"
rng = np.random.default_rng(2026)
ast = pd.read_csv(f"{B}/ASteCA/output/NGC_6383_dr3_all/NGC_6383_dr3_all.csv", sep=r"\s+")
m = pd.read_csv(f"{B}/data/40/masses_asteca_069.csv")
binr = m.binar_prob.values > 0.7
mass = m.m1.values + np.where(binr, np.nan_to_num(m.m2.values), 0.0)
bp = m.binar_prob.values
ra0, de0 = 263.6826, -32.5838
x = (ast.ra.values - ra0) * np.cos(np.radians(de0)) * 60; y = (ast.dec.values - de0) * 60
r = np.hypot(x, y); P = np.column_stack([x, y])
EDGES = np.arange(0, 45, 5.0)
NS = [5, 10, 15, 20]; NDRAW = 1000
mst = lambda Q: minimum_spanning_tree(squareform(pdist(Q))).toarray().sum()
protect = np.argsort(mass)[::-1][:20]

def stats(idx):
    Q, mm = P[idx], mass[idx]; order = np.argsort(mm)[::-1]; out = []
    for n in NS:
        lm = mst(Q[order[:n]]); rr = np.array([mst(Q[rng.choice(len(Q), n, replace=False)]) for _ in range(NDRAW)])
        out.append((rr.mean() / lm, (rr <= lm).mean()))
    b_ = bp[idx] >= 0.6
    return out, ks_2samp(r[idx][~b_], r[idx][b_]).pvalue

def remove(bdens):
    keep = np.ones(len(r), bool)
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        hi_ = min(hi, 40.0); area = np.pi * (hi_**2 - lo**2)
        k = rng.poisson(bdens * area)
        cand = np.where((r >= lo) & (r < hi) & ~np.isin(np.arange(len(r)), protect))[0]
        k = min(k, len(cand))
        if k: keep[rng.choice(cand, k, replace=False)] = False
    return np.where(keep)[0]

base, ksb = stats(np.arange(len(r)))
print("no removal: N=254 | " + " | ".join(f"L={l:.2f} p={p:.4f}" for l, p in base) + f" | KS single-vs-binary p={ksb:.3f}")
for bdens in [0.004, 0.011, 0.018]:
    L = {n: [] for n in NS}; Pv = {n: [] for n in NS}; K = []; Nk = []
    for it in range(60):
        idx = remove(bdens); s, ks = stats(idx); Nk.append(len(idx)); K.append(ks)
        for n, (l, p) in zip(NS, s): L[n].append(l); Pv[n].append(p)
    print(f"b={bdens:.3f}: N_kept median {int(np.median(Nk))} | " +
          " | ".join(f"N={n}: L med {np.median(L[n]):.2f} p med {np.median(Pv[n]):.3f} (p<0.01 in {np.mean(np.array(Pv[n])<0.01)*100:.0f}%)" for n in NS) +
          f" | KS p med {np.median(K):.3f} (p<0.05 in {np.mean(np.array(K)<0.05)*100:.0f}%)")
