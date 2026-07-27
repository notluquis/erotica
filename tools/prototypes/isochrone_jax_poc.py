"""VALIDATED PoC (2026-07-21) — differentiable MIST metallicity interpolation.

Proves the fix for the isochrone-NUTS staircase (see docs/isochrone_sampler_fix.md):
map_coordinates(order=1) over an EEP-aligned regular grid gives a smooth, fully
differentiable G([Fe/H]) with nonzero gradient everywhere.

RESULT on the 15 NGC 6383 MIST files (age 6.6):
  feh sweep (200 pts): 200 unique G values   (was 11/200 nearest-neighbor)
  d(G)/d(feh):         0.0% zero-gradient     (was 95.0%)
  => staircase eliminated; NUTS gradients are now usable.

Next: 3D (age,[Fe/H],EEP) grid + NumPyro per-star mixture (docs/hierarchical_isochrone_design.md).
"""
import glob, re, numpy as np, jax, jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
jax.config.update("jax_enable_x64", True)

FILES=sorted(glob.glob("/Users/notluquis/erotica/data/test/NGC6383/MIST/UBVRIplus/*.iso.cmd"))
C_EEP,C_AGE,C_FEH,C_G,C_BP,C_RP=0,1,8,30,31,32
TARGET_AGE=6.60  # young

def read_blocks(fn):
    """yield (log_age, arr[EEP,G,BP,RP,feh]) per isochrone block."""
    rows=[]; 
    for line in open(fn):
        if line.startswith('#') or not line.strip(): 
            if rows: yield np.array(rows); rows=[]
            continue
        v=line.split(); rows.append([float(v[C_EEP]),float(v[C_AGE]),float(v[C_FEH]),
                                     float(v[C_G]),float(v[C_BP]),float(v[C_RP])])
    if rows: yield np.array(rows)

# common EEP axis (main-sequence-ish window present at young age across all feh)
EEP_AXIS=np.arange(202,454,2.0)   # 126 EEP points
fehs=[]; gridG=[]; gridC=[]
for fn in FILES:
    best=None
    for blk in read_blocks(fn):
        if blk.size==0: continue
        la=blk[0,1]
        if best is None or abs(la-TARGET_AGE)<abs(best[0,1]-TARGET_AGE): best=blk
    feh=best[0,2]; eep=best[:,0]; G=best[:,3]; col=best[:,4]-best[:,5]
    # interp this isochrone onto the common EEP axis
    Gi=np.interp(EEP_AXIS,eep,G,left=np.nan,right=np.nan)
    Ci=np.interp(EEP_AXIS,eep,col,left=np.nan,right=np.nan)
    fehs.append(feh); gridG.append(Gi); gridC.append(Ci)
order=np.argsort(fehs); fehs=np.array(fehs)[order]
gridG=np.array(gridG)[order]; gridC=np.array(gridC)[order]   # (N_feh, N_EEP)
# drop EEP columns with any NaN (not covered at all feh) -> clean regular grid
good=~np.isnan(gridG).any(0); gridG=gridG[:,good]; gridC=gridC[:,good]; EEPg=EEP_AXIS[good]
print(f"grid: {gridG.shape} (N_feh={len(fehs)}, N_EEP={good.sum()}) | feh range {fehs.min()}..{fehs.max()} at age {TARGET_AGE}")
gG=jnp.asarray(gridG)

def G_of(feh, ieep):  # differentiable G at continuous feh index-space
    fi=jnp.interp(feh, jnp.asarray(fehs), jnp.arange(len(fehs)))   # feh -> fractional feh-index
    return map_coordinates(gG, [fi[None], jnp.asarray([float(ieep)])], order=1, mode="nearest")[0]

# --- VALIDATION ---
feh_sweep=np.linspace(fehs.min()+1e-3, fehs.max()-1e-3, 200)
ieep=60  # a mid main-sequence EEP
Gvals=np.array([float(G_of(f,ieep)) for f in feh_sweep])
uniq=len(np.unique(np.round(Gvals,6)))
grad=jax.grad(lambda f: G_of(f,ieep))
grads=np.array([float(grad(f)) for f in feh_sweep])
zerofrac=np.mean(np.abs(grads)<1e-9)
print(f"feh sweep (200 pts): {uniq} unique G values (was 11/200 nearest-neighbor)")
print(f"d(G)/d(feh): {100*zerofrac:.1f}% zero-gradient (was 95.0%) | grad range [{grads.min():.3f},{grads.max():.3f}]")
print("VERDICT:", "FIXED — smooth, nonzero gradient" if uniq>150 and zerofrac<0.05 else "still staircased")
