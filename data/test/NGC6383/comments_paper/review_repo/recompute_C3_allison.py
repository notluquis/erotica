#!/usr/bin/env python3
"""Reproducible recompute for NGC 6383 referee response.

Covers:
  - C3 error propagation: R_hl, R_hm (bootstrap arcmin error + full pc propagation incl. distance)
  - R_c / R_t pc propagation
  - t_rh / t_seg cascade from the corrected r_hm(pc)
  - Allison et al. (2009) MST Lambda_MSR mass-segregation profile (+figure)
  - T-Tauri binary probability lookup

Run from /Users/notluquis/COSMIC.
Inputs (40 arcmin production run):
  ASteCA/output/NGC_6383_dr3_all/NGC_6383_dr3_all.csv   (254 ref sample: designation, ra, dec, Gmag)
  data/40/masses_asteca.csv                             (254: m1, m2, binar_prob, aligned by row)
"""
import numpy as np, pandas as pd
import astropy.units as u, astropy.constants as const
from astropy.coordinates import SkyCoord, angular_separation
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "/Users/notluquis/COSMIC/data/test/NGC6383"
rng = np.random.default_rng(7)
CENTER = SkyCoord(263.6826 * u.deg, -32.5838 * u.deg)
DIST, DERR = 1.110, 0.060      # kpc (parallax distance)
N = 254
LAM = 0.11
MC, MC_ERR = 902.27, 92.3      # Msun, cluster mass from Hunt et al. 2024
MMAX = 14.7                    # Msun, most massive star (paper)
AGE = 3.53                     # Myr

ast = pd.read_csv(f"{BASE}/ASteCA/output/NGC_6383_dr3_all/NGC_6383_dr3_all.csv", sep=r"\s+")
mas = pd.read_csv(f"{BASE}/data/40/masses_asteca.csv")
assert len(ast) == len(mas) == 254
df = ast.reset_index(drop=True).copy()
df["m1"] = mas["m1"].values
df["m2"] = mas["m2"].values
df["binar_prob"] = mas["binar_prob"].values
df["mtot"] = df["m1"] + np.where(df["binar_prob"] > 0.7, df["m2"], 0.0)
sc = SkyCoord(df["ra"].values * u.deg, df["dec"].values * u.deg)
df["sep"] = angular_separation(sc.ra, sc.dec, CENTER.ra, CENTER.dec).to(u.arcmin).value


def r_half_light(d):
    s, g = d["sep"].values, d["Gmag"].values
    m = np.isfinite(s) & np.isfinite(g); s, g = s[m], g[m]
    f = 10 ** (-0.4 * (g - g.min())); o = np.argsort(s)
    cs = np.cumsum(f[o]); i = np.searchsorted(cs, cs[-1] / 2)
    return s[o][min(i, len(s) - 1)]


def r_half_mass(d):
    s, mm = d["sep"].values, d["mtot"].values
    m = np.isfinite(s) & np.isfinite(mm); s, mm = s[m], mm[m]
    o = np.argsort(s); cm = np.cumsum(mm[o]); i = np.searchsorted(cm, cm[-1] / 2)
    return s[o][min(i, len(s) - 1)]


def to_pc(theta, e_theta):
    r = (theta * u.arcmin * DIST * u.kpc).to(u.pc, u.dimensionless_angles()).value
    return r, r * np.hypot(e_theta / theta, DERR / DIST)


rhl, rhm = r_half_light(df), r_half_mass(df)
B = 4000
ehl = np.std([r_half_light(df.sample(254, replace=True, random_state=rng.integers(1e9))) for _ in range(B)])
ehm = np.std([r_half_mass(df.sample(254, replace=True, random_state=rng.integers(1e9))) for _ in range(B)])

print("=== C3 radii (real 254 reference sample) ===")
for name, th, e, paper in [("R_c", 1.95, 0.19, "0.632+-0.00621"),
                            ("R_t", 40.4, 14.3, "13.10+-4.71"),
                            ("R_hl", rhl, ehl, "1.960+-0.00020"),
                            ("R_hm", rhm, ehm, "1.650+-0.066")]:
    r, er = to_pc(th, e)
    print(f"  {name}: {th:.2f}+-{e:.2f} arcmin -> {r:.3f} +- {er:.3f} pc   (paper {paper})")

# t_rh / t_seg cascade with corrected r_hm
rhm_pc, erhm_pc = to_pc(rhm, ehm)
trh = (0.17 * N / np.log(LAM * N) * np.sqrt((rhm_pc * u.pc) ** 3 / (const.G * MC * u.Msun))).to(u.Myr).value
rel = np.hypot(1.5 * erhm_pc / rhm_pc, 0.5 * MC_ERR / MC)
mmean = df["mtot"].mean()
tseg = mmean / MMAX * trh
mcut = mmean * trh / AGE
print(f"\n=== dynamical cascade (corrected r_hm={rhm_pc:.3f} pc) ===")
print(f"  t_rh = {trh:.1f} +- {trh*rel:.1f} Myr        (paper 13.6, used wrong r_hm=1.65 pc)")
print(f"  min t_seg (14.7 Msun) = {tseg:.2f} +- {tseg*rel:.2f} Myr")
print(f"  <m_tot> = {mmean:.3f} Msun ; mass cutoff t_seg=age -> {mcut:.2f} Msun (paper used 6.08)")

# Allison 2009 Lambda_MSR
dra = (df["ra"].values - CENTER.ra.deg) * np.cos(np.deg2rad(CENTER.dec.deg))
ddec = df["dec"].values - CENTER.dec.deg
pts = np.column_stack([dra, ddec])
order = np.argsort(df["mtot"].values)[::-1]
mst = lambda p: minimum_spanning_tree(squareform(pdist(p))).toarray().sum()
Ns = [5, 8, 10, 15, 20, 30, 40, 50]
lams, sigs = [], []
print("\n=== Allison Lambda_MSR ===")
for n in Ns:
    lm = mst(pts[order[:n]])
    rnd = np.array([mst(pts[rng.choice(254, n, replace=False)]) for _ in range(500)])
    lams.append(rnd.mean() / lm); sigs.append(rnd.std() / lm)
    print(f"  N={n:3d}: Lambda={lams[-1]:.2f}+-{sigs[-1]:.2f}")

fig, ax = plt.subplots(figsize=(5, 3.6))
ax.errorbar(Ns, lams, yerr=sigs, fmt="o-", color="purple", capsize=3)
ax.axhline(1.0, ls="--", color="gray", lw=1)
ax.set_xlabel(r"$N_{\rm MST}$ (most massive)")
ax.set_ylabel(r"$\Lambda_{\rm MSR}$")
ax.set_title("NGC 6383 mass segregation (Allison et al. 2009)")
fig.tight_layout()
out = f"{BASE}/comments_paper/submission_package/clean_source/Figures/mass_segregation_mst.pdf"
fig.savefig(out)
print(f"\nfigure -> {out}")

# T-Tauri binary
tt = df[df["designation"].astype(str).str.contains("4054567805963940352")]
print("\n=== T-Tauri Gaia DR3 4054567805963940352 ===")
print("  in 254 reference sample:", len(tt) > 0, "| binar_prob =",
      tt["binar_prob"].values if len(tt) else "n/a")
