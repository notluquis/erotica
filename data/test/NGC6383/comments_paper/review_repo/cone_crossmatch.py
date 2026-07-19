#!/usr/bin/env python3
"""Source-level crossmatch of the 40/50/60/70 arcmin reference samples (p>=0.6):
pairwise overlaps, survival of the adopted 254, core intersection, and where the
extra 60/70 sources live (inside vs outside the 40 arcmin footprint)."""
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord, angular_separation
import astropy.units as u
B="/Users/notluquis/COSMIC/data/test/NGC6383/comments_paper/radius_robustness/generated"
c=SkyCoord(263.6826*u.deg,-32.5838*u.deg)
S={};D={}
for r in [40,50,60,70]:
    t=Table.read(f"{B}/{r}/paperfaithful_reference_p06.ecsv")
    ids=set(np.asarray(t["source_id"],dtype=np.int64))
    sc=SkyCoord(np.asarray(t["ra"])*u.deg,np.asarray(t["dec"])*u.deg)
    dc=angular_separation(sc.ra,sc.dec,c.ra,c.dec).to(u.arcmin).value
    S[r]=ids; D[r]=dict(zip(np.asarray(t["source_id"],dtype=np.int64),dc))
    print(f"[{r}'] N={len(ids)}")
print("\npairwise overlap (count, Jaccard, frac of smaller):")
import itertools
for a,b in itertools.combinations([40,50,60,70],2):
    inter=len(S[a]&S[b]); uni=len(S[a]|S[b]); small=min(len(S[a]),len(S[b]))
    print(f"  {a}'x{b}': common={inter}  Jaccard={inter/uni:.2f}  frac(min)={inter/small:.3f}")
core=S[40]&S[50]&S[60]&S[70]
print(f"\ncore (in all four): {len(core)}  = {len(core)/254*100:.1f}% of the adopted 254")
print("\nsurvival of the adopted 254 in each wider run:")
for r in [50,60,70]:
    surv=len(S[40]&S[r])
    print(f"  in {r}': {surv}/254 ({surv/254*100:.1f}%)  | lost: {254-surv}")
print("\nextra sources (not in the 254) per run, split by radius vs the 40' footprint:")
for r in [50,60,70]:
    extra=S[r]-S[40]
    inside=sum(1 for s in extra if D[r][s]<=40.0)
    print(f"  {r}': extra={len(extra)}  inside 40'={inside}  beyond 40'={len(extra)-inside}")
print("\nlost-254 diagnostics (where do the dropped 254 sit?):")
for r in [50,60,70]:
    lost=S[40]-S[r]
    if lost:
        dl=[D[40][s] for s in lost]
        print(f"  {r}': lost={len(lost)}  median r={np.median(dl):.1f}'  min={min(dl):.1f}'  max={max(dl):.1f}'")
print("\n50' vs 40' two-way: only-in-50 =",len(S[50]-S[40]),"| only-in-40 =",len(S[40]-S[50]))
