"""Round 3 (2026-09-14). Backs the Lambda_MSR sample-sensitivity sentences of Sect. 7 and point (i) of the
response letter. Input hunt_reffert2024_members_ngc6383.tsv = VizieR J/A+A/686/A42/members, Name=NGC_6383 (322 rows).
Random draws are seeded (default_rng(1)), so the run reproduces the quoted numbers; p-values use 2000 draws.

Why Zhang+2026 (AJ 172, 85) get Lambda ~1 for NGC 6383 and P01 gets 2.5-3.6.

Factorial: membership (Hunt&Reffert 2024 prob>=0.8 vs P01 reference 254) x masses (Hunt Mass50 vs
ASteCA 0.6.9 system masses) x geometry (2D vs Zhang's 3D shrinkage). Plus the 3D asymmetry test:
same 3D reconstruction but with one common e_plx for all stars.
"""
import numpy as np, pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

B = "/Users/notluquis/erotica/data/test/NGC6383"
rng = np.random.default_rng(1)
NS = [5, 10, 15, 20]
TRIALS = 500

def mst(P):
    return minimum_spanning_tree(squareform(pdist(P))).toarray().sum()

def lam(P, mass, n):
    order = np.argsort(mass)[::-1]
    lm = mst(P[order[:n]])
    r = np.array([mst(P[rng.choice(len(P), n, replace=False)]) for _ in range(TRIALS)])
    return r.mean() / lm, (r.mean() + r.std()) / lm < 1

# --- Hunt & Reffert 2024 members (VizieR J/A+A/686/A42/members) ---
lines = [l for l in open(f"{B}/comments_paper/review_repo/hunt_reffert2024_members_ngc6383.tsv") if not l.startswith("#") and l.strip()]
hdr = lines[0].rstrip("\n").split("\t")
rows = [l.rstrip("\n").split("\t") for l in lines[3:]]
h = pd.DataFrame(rows, columns=hdr)
for c in ["Prob", "RA_ICRS", "DE_ICRS", "Plx", "e_Plx", "Gmag", "Mass50"]:
    h[c] = pd.to_numeric(h[c], errors="coerce")
h["GaiaDR3"] = h["GaiaDR3"].str.strip()
print("Hunt members:", len(h), "| prob>=0.8:", (h.Prob >= 0.8).sum(), "| Mass50 NaN:", h.Mass50.isna().sum())
hz = h[(h.Prob >= 0.8)].dropna(subset=["Mass50", "RA_ICRS", "DE_ICRS", "Plx", "e_Plx"]).reset_index(drop=True)
print("Zhang working sample (prob>=0.8, finite):", len(hz))

# --- P01 reference sample with ASteCA 0.6.9 masses (as finalize_v069.py) ---
ast = pd.read_csv(f"{B}/ASteCA/output/NGC_6383_dr3_all/NGC_6383_dr3_all.csv", sep=r"\s+")
m = pd.read_csv(f"{B}/data/40/masses_asteca_069.csv")
binr = m.binar_prob.values > 0.7
ast["mass"] = m.m1.values + np.where(binr, np.nan_to_num(m.m2.values), 0.0)
ast["GaiaDR3"] = ast.designation.str.replace("Gaia DR3 ", "").str.strip()
print("P01 reference:", len(ast))

P_CLUS, E_CLUS = 0.88573121, 0.00508136   # Hunt cluster table; Zhang uses max(e, 0.01)
S_C = max(E_CLUS, 0.01)

def coords2d(ra, dec):
    d = 1000.0 / P_CLUS
    ra, dec = np.radians(ra), np.radians(dec)
    return np.column_stack([d*np.cos(dec)*np.cos(ra), d*np.cos(dec)*np.sin(ra)])

def coords3d(ra, dec, plx, eplx):
    pc = (plx*S_C**2 + P_CLUS*eplx**2) / (S_C**2 + eplx**2)
    d = 1000.0 / pc
    ra, dec = np.radians(ra), np.radians(dec)
    return np.column_stack([d*np.cos(dec)*np.cos(ra), d*np.cos(dec)*np.sin(ra), d*np.sin(dec)]), d

def report(label, P, mass):
    out = []
    for n in NS:
        l, s = lam(P, mass, n)
        out.append(f"{l:5.2f}{'*' if s else ' '}")
    print(f"{label:58s} " + "  ".join(out))

print("\nLambda for N =", NS, "  (* = Zhang's inverse criterion Lambda+sigma<1)")
# 1. reproduce Zhang
report("Zhang repro 2D: Hunt memb p>=0.8, Hunt Mass50", coords2d(hz.RA_ICRS, hz.DE_ICRS), hz.Mass50.values)
P3, d3 = coords3d(hz.RA_ICRS, hz.DE_ICRS, hz.Plx, hz.e_Plx)
report("Zhang repro 3D: Hunt memb p>=0.8, Hunt Mass50", P3, hz.Mass50.values)
# 2. 3D asymmetry test: common e_plx (median) for every star
Pe, _ = coords3d(hz.RA_ICRS, hz.DE_ICRS, hz.Plx, np.full(len(hz), np.median(hz.e_Plx)))
report("3D with one common e_plx (median) for all stars", Pe, hz.Mass50.values)
# LOS scatter of the massive vs all
order = np.argsort(hz.Mass50.values)[::-1]
for n in [5, 10, 20]:
    print(f"  top-{n:2d} massive: median e_plx {np.median(hz.e_Plx.values[order[:n]]):.3f} mas, std(d) {np.std(d3[order[:n]]):5.1f} pc"
          f" | all: median e_plx {np.median(hz.e_Plx):.3f}, std(d) {np.std(d3):5.1f} pc | r50 1.30 pc")
# 3. P01 as published (2D, own masses) -- sky-plane km scaled as Zhang for comparability
report("P01 2D: P01 memb (254), ASteCA masses", coords2d(ast.ra, ast.dec), ast.mass.values)
# 4. factorial on the common stars
j = ast.merge(h[["GaiaDR3", "Mass50", "Prob"]], on="GaiaDR3", how="inner")
print(f"\nP01 reference stars also in Hunt: {len(j)} of {len(ast)}; of those Hunt prob>=0.8: {(j.Prob>=0.8).sum()}")
jj = j.dropna(subset=["Mass50"])
report(f"common stars ({len(jj)}), ASteCA masses", coords2d(jj.ra, jj.dec), jj.mass.values)
report(f"common stars ({len(jj)}), Hunt Mass50", coords2d(jj.ra, jj.dec), jj.Mass50.values)
print("\nTop-10 by ASteCA mass (P01) vs Hunt Mass50 for the same star:")
for _, r in j.sort_values("mass", ascending=False).head(10).iterrows():
    print(f"  {r.GaiaDR3:>20s} G={r.Gmag:5.2f} ASteCA {r.mass:6.2f}  Hunt {r.Mass50:6.2f}  prob {r.Prob:.2f}")
print("Top-10 by Hunt Mass50 (Zhang sample):")
for _, r in hz.sort_values("Mass50", ascending=False).head(10).iterrows():
    inP = r.GaiaDR3 in set(ast.GaiaDR3)
    print(f"  {r.GaiaDR3:>20s} G={r.Gmag:5.2f} Hunt {r.Mass50:6.2f}  e_plx {r.e_Plx:.3f}  in P01 ref: {inP}")

# --- extent of each sample, and permutation p-value (P01's null) ---
from astropy.coordinates import SkyCoord, angular_separation
import astropy.units as u
C = SkyCoord(263.6826*u.deg, -32.5838*u.deg)
def rad(ra, dec):
    return angular_separation(np.radians(ra), np.radians(dec), C.ra.rad, C.dec.rad) * 180/np.pi*60
def perm(label, ra, dec, mass):
    P = coords2d(ra, dec); order = np.argsort(mass)[::-1]; r = rad(np.asarray(ra), np.asarray(dec))
    out = []
    for n in NS:
        lm = mst(P[order[:n]]); rr = np.array([mst(P[rng.choice(len(P), n, replace=False)]) for _ in range(2000)])
        out.append(f"L={rr.mean()/lm:4.2f} p={(rr <= lm).mean():.4f}")
    print(f"{label:34s} N*={len(P):3d} r_med={np.median(r):5.2f}' r90={np.percentile(r,90):5.2f}' top10 r_med={np.median(r[order[:10]]):4.2f}' | " + " | ".join(out))
print("\nPermutation p = P(random MST <= massive MST), 2000 draws; N =", NS)
perm("P01 254, ASteCA masses", ast.ra, ast.dec, ast.mass.values)
perm("common 178, ASteCA masses", jj.ra, jj.dec, jj.mass.values)
perm("common 178, Hunt Mass50", jj.ra, jj.dec, jj.Mass50.values)
perm("Hunt all 322, Hunt Mass50", h.RA_ICRS, h.DE_ICRS, h.Mass50.values)
perm("Hunt p>=0.8 (69), Hunt Mass50", hz.RA_ICRS, hz.DE_ICRS, hz.Mass50.values)

# --- is P01's signal carried by outer members that could be contaminants? tests inside P01's own sample ---
from astropy.table import Table
cds = Table.read(f"{B}/comments_paper/cds_final/ngc6383_members.ecsv").to_pandas()
cds["GaiaDR3"] = cds.GaiaDR3.astype(str).str.strip()
a2 = ast.merge(cds[["GaiaDR3", "pMember"]], on="GaiaDR3", how="left")
a2["r"] = rad(a2.ra.values, a2.dec.values)
a2["inHunt"] = a2.GaiaDR3.isin(set(h.GaiaDR3))
print(f"\nP01 254: pMember NaN {a2.pMember.isna().sum()}; not in Hunt {(~a2.inHunt).sum()}, their r_med {a2.r[~a2.inHunt].median():.1f}' vs in-Hunt {a2.r[a2.inHunt].median():.1f}'")
print("  mass of not-in-Hunt: median %.2f, max %.2f; in-Hunt median %.2f" % (a2.mass[~a2.inHunt].median(), a2.mass[~a2.inHunt].max(), a2.mass[a2.inHunt].median()))
for lab, sel in [("P01 pMember>=0.8", a2.pMember >= 0.8), ("P01 r<20'", a2.r < 20), ("P01 r<15'", a2.r < 15),
                 ("P01 r<10'", a2.r < 10), ("P01 in Hunt (178)", a2.inHunt)]:
    s = a2[sel]
    perm(lab, s.ra, s.dec, s.mass.values)

# --- KS single vs binary and Sagitta PMS fraction by sample/radius (Sect. 7 numbers) ---
from scipy.stats import ks_2samp
m2 = pd.read_csv(f"{B}/data/40/masses_asteca_069.csv")
a2["bp"] = m2.binar_prob.values
def ks_sb(lab, s):
    b_ = s.bp >= 0.6
    print(f"{lab:22s} N={len(s):3d} KS single-vs-binary p={ks_2samp(s.r[~b_], s.r[b_]).pvalue:.3f} (Ns={int((~b_).sum())}, Nb={int(b_.sum())})")
print()
ks_sb("P01 254", a2); ks_sb("P01 in Hunt 178", a2[a2.inHunt]); ks_sb("P01 r<15'", a2[a2.r < 15])
print("not-in-Hunt classified single:", int((a2.bp[~a2.inHunt] < 0.6).sum()), "of", int((~a2.inHunt).sum()))
ref = cds[cds.Ref == 1].copy()
ref["r"] = rad(ref.RAdeg.values, ref.DEdeg.values)
w2m = ~(ref.Jmag.isna() & ref.Hmag.isna() & ref.Ksmag.isna())
for lab, sel in [("r<10'", ref.r < 10), ("10-20'", (ref.r >= 10) & (ref.r < 20)), ("20-40'", ref.r >= 20)]:
    s = ref[sel & w2m]
    print(f"PMS fraction {lab:7s}: {(s.PMSProb >= 0.6).mean()*100:.0f}% of {len(s)} with 2MASS")
