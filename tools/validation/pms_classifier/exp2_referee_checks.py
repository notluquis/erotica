"""POST HOC, revision adversarial (subagente con la skill adversarial-referee, 2026-10-11) del segundo experimento. Nada de esto estaba
prerregistrado ni cambia la regla: mide cuanto dependen los veredictos de §3 de la particion A/B (M1), de un cumulo (M2), de la semilla
del bootstrap (M3), del prior P1 (M4) y de la codificacion de Sagitta en G2 (M6). Recalculos sobre cache/exp2_oof.npz; sin reentrenar.
Corre: SAGITTA_SRC=~/.local/lib/sagitta-1.2.3 OMP_NUM_THREADS=2 ~/miniforge3/envs/erotica/bin/python exp2_referee_checks.py \
       > exp2_referee_checks_measured.txt
"""
import json
import sys

import numpy as np

print("=== [R-a] semilla del bootstrap de G1 y FPR en B de las configuraciones D1 (ref1)")
import exp2_common as E, exp2_grid as GR
from pathlib import Path
C = Path('cache')
d = GR.prepare()
oofs = np.load(C/'exp2_oof.npz')
summ = json.load(open(C/'exp2_summary.json'))
sel = json.load(open(C/'exp2_selected.json'))
print('chosen', sel['chosen'], sel['th'])
B, A, grp, mg0 = d['B'], d['A'], d['grp'], d['mg0']
# disjointness and fold check
print('A&B', np.sum(A&B), 'A clusters', len(np.unique(grp[A])), 'B clusters', len(np.unique(grp[B])), 'overlap', len(set(grp[A])&set(grp[B])))
def boot(pr, g, R, seed):
    rng = np.random.default_rng(seed)
    ug, inv = np.unique(g, return_inverse=True)
    n = np.bincount(inv, minlength=len(ug)).astype(float); k = np.bincount(inv, weights=pr.astype(float), minlength=len(ug))
    v = [k[i].sum()/n[i].sum() for i in (rng.integers(0,len(ug),len(ug)) for _ in range(R))]
    return np.percentile(v, 97.5)
p = oofs['T0P1K0F0'].astype(float)
dec = E.decide(p, mg0, np.array(sel['th']))
ups = [boot(dec[B], grp[B], 1000, s) for s in range(40)]
print('G1 boot upper over 40 seeds R=1000: min %.4f med %.4f max %.4f frac>0.10 %.3f' % (np.min(ups), np.median(ups), np.max(ups), np.mean(np.array(ups)>0.10)))
print('R=20000 seed 1:', boot(dec[B], grp[B], 20000, 1))
# tramo +3..+5 and -5..+3 gate stability
k = E.tramo_index(mg0)
for i in range(4):
    s = B & (k==i)
    u = [boot(dec[s], grp[s], 1000, 100+j) for j in range(20)]
    print(E.TRAMO_NAMES[i], 'upper min/med/max %.4f %.4f %.4f frac>0.10 %.2f' % (min(u), np.median(u), max(u), np.mean(np.array(u)>0.10)))
# all configs on B
print('config  B global FPR, boot upper(R=4000), per-tramo FPR, recw')
for key, r in summ.items():
    if r['cfg']['D']!='D1' and key!='T1P1K2F1D0': continue
    pp = oofs[key[:-2]].astype(float)
    dd = E.decide(pp, mg0, np.array(r['th']))
    fpr = dd[B].mean(); up = boot(dd[B], grp[B], 4000, 7)
    tr = [dd[B&(k==i)].mean() for i in range(4)]
    trup = [boot(dd[B&(k==i)], grp[B&(k==i)], 2000, 8) for i in range(4)]
    print(f"{key:12s} {fpr:.4f} up {up:.4f} {'PASS' if up<=0.10 else 'FAIL'} tramos {np.round(tr,3).tolist()} ups {np.round(trup,3).tolist()} recw {r['recw']:.3f}")
print("=== [R-b] concentracion por cumulo en B (ref2)")
import exp2_common as E, exp2_grid as GR
from pathlib import Path
sel = json.load(open('cache/exp2_selected.json'))
dec = E.decide(oofs['T0P1K0F0'].astype(float), d['mg0'], np.array(sel['th']))
k = E.tramo_index(d['mg0']); B=d['B']; g=d['grp']
s = B&(k==1)
u, inv = np.unique(g[s], return_inverse=True)
n = np.bincount(inv); x = np.bincount(inv, weights=dec[s].astype(float))
o = np.argsort(-x)
print('+3..+5 B: total FP', x.sum(), 'n', n.sum(), 'clusters', len(u))
for i in o[:6]: print(u[i], int(x[i]), n[i], round(x[i]/n[i],3), 'logt %.2f'%np.median(d['logt'][s][inv==i]))
print('FP share top3', x[o[:3]].sum()/x.sum(), ' FPR without top1', (x.sum()-x[o[0]])/(n.sum()-n[o[0]]))
# MG0 of FPs in +3..+5 relative to cluster turnoff? print logt dist of FP clusters
oc = g=='OC_0011'
print('OC_0011 in B:', (oc&B).sum(), 'FP all tramos', dec[oc&B].sum(), 'per tramo', [int(dec[oc&B&(k==i)].sum()) for i in range(4)], 'n per tramo', [int((oc&B&(k==i)).sum()) for i in range(4)])
Bx = B&~oc
print('G1 without OC_0011: FPR %.4f' % dec[Bx].mean())
rng=np.random.default_rng(3); uu,iv=np.unique(g[Bx],return_inverse=True); nn=np.bincount(iv); xx=np.bincount(iv,weights=dec[Bx].astype(float))
v=[xx[i].sum()/nn[i].sum() for i in (rng.integers(0,len(uu),len(uu)) for _ in range(4000))]; print('boot upper', np.percentile(v,97.5))
import astropy.table as at
m=d['m'][oc]; print({c: float(np.median(np.asarray(m[c],float))) for c in ('logAge50','logAge16','logAge84','AV50','dist50') if c in m.colnames})
print("=== [R-c] 200 particiones A/B y FPR contra AV50 (ref3)")
import exp2_common as E, exp2_grid as GR
from scipy.stats import beta
p = oofs['T0P1K0F0'].astype(float)
g, old, mg0 = d['grp'], d['old'], d['mg0']; k = E.tramo_index(mg0)
uo = np.unique(g[old]); rng = np.random.default_rng(99)
def bootup(pr, gg, R=1000):
    uu, iv = np.unique(gg, return_inverse=True); nn=np.bincount(iv); xx=np.bincount(iv, weights=pr.astype(float))
    return np.percentile([xx[i].sum()/nn[i].sum() for i in (rng.integers(0,len(uu),len(uu)) for _ in range(R))], 97.5)
synth_ok = [True, True, True, False]
res=[]; sets={}
for it in range(200):
    perm = rng.permutation(uo); a=set(perm[:len(uo)//2])
    A = old & np.isin(g, list(a)); B = old & ~A
    th = E.tramo_thresholds(p, mg0, A, target_fpr=0.03)
    dec = E.decide(p, mg0, th)
    x, n = dec[B].sum(), B.sum()
    cpu = beta.ppf(0.975, x+1, n-x)
    g1 = cpu<=0.10 and bootup(dec[B], g[B])<=0.10
    est = tuple(int(bootup(dec[B&(k==i)], g[B&(k==i)], 500)<=0.10 and synth_ok[i]) for i in range(4))
    res.append(g1); sets[est]=sets.get(est,0)+1
    if it==0: print('example th', th)
print('P(G1 pasa) over 200 splits:', np.mean(res))
print('established sets (-5..+3,+3..+5,+5..+7,+7..+12):', sets)
# FPR per cluster vs AV50 over all old
uu, iv = np.unique(g[old], return_inverse=True)
th0 = np.array(json.load(open('cache/exp2_selected.json'))['th']); dec0 = E.decide(p, mg0, th0)
fp = np.bincount(iv, weights=dec0[old].astype(float)); nn=np.bincount(iv)
av = np.array([np.median(np.asarray(d['m']['AV50'],float)[old][iv==j]) for j in range(len(uu))])
from scipy.stats import spearmanr
m10 = nn>=10
print('spearman FPR vs AV50 (n>=10):', spearmanr(fp[m10]/nn[m10], av[m10]))
for lo,hi in [(0,1),(1,2),(2,3),(3,4)]:
    s=(av>=lo)&(av<hi); print(f'AV {lo}-{hi}: clusters {s.sum()} FPR pooled {fp[s].sum()/max(nn[s].sum(),1):.3f}')
o=np.argsort(-fp/np.maximum(nn,1)); print([(uu[j], round(fp[j]/nn[j],2), nn[j], round(av[j],2)) for j in o[:6] if nn[j]>=10])
print("=== [R-d] P0 contra P1 con D1: FPR en B")
B, mg0 = d["B"], d["mg0"]
for K in ("K0", "K1", "K2"):
    for F in ("F0", "F1"):
        for T in ("T0", "T1"):
            r0, r1 = summ[f"{T}P0{K}{F}D1"], summ[f"{T}P1{K}{F}D1"]
            f0 = E.decide(oofs[f"{T}P0{K}{F}"].astype(float), mg0, np.array(r0["th"]))[B].mean()
            f1 = E.decide(oofs[f"{T}P1{K}{F}"].astype(float), mg0, np.array(r1["th"]))[B].mean()
            print(f"  {T}{K}{F}: P0 {f0:.4f} -> P1 {f1:.4f} (recall ponderado {r0['recw']:.3f} -> {r1['recw']:.3f})")
print("=== [R-e] mitad sintetica de G2 para Sagitta (0,6), en las dos codificaciones")
import apply_ngc6383 as AP
import sagitta_ref as SG
from exp2_confirm import synth
from astropy.coordinates import SkyCoord
import astropy.units as u
gal = SkyCoord(263.673 * u.deg, -32.577 * u.deg).galactic
for kind in ("ms",):
    ob = synth(kind, 2000, AP.MOD, AP.AV)
    N = len(ob["G"]); plx = np.full(N, 1000 / 1100.9)
    av = SG.predict_av(np.full(N, gal.l.deg), np.full(N, gal.b.deg), plx)
    ps = SG.predict_pms(plx, av, ob["G"], ob["BP"], ob["RP"], ob["J"], ob["H"], ob["Ks"])
    kk = E.tramo_index(ob["G"] - AP.MOD - 0.789 * AP.AV)
    inr = ob["G"] <= 20.66
    print("  FPR Sagitta(0,6) MS log t 9,0 (faltante -> token; el sintetico no tiene U): " + "  ".join(
        f"{nm} {np.mean(ps[kk == i] >= .6):.3f} (G<=20,66: {np.mean(ps[(kk == i) & inr] >= .6):.3f})" for i, nm in enumerate(E.TRAMO_NAMES)))
