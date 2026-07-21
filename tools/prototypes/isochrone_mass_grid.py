"""VALIDATED mass-based differentiable MIST grid (2026-07-21).
Research-recommended basis (initial mass, not EEP) for young/PMS + multi-grid.
Spans a proper young CMD: G[-0.3,9.4], BP-RP[-0.26,2.90] (blue massive -> red low-mass PMS).
Differentiable via jax.scipy.ndimage.map_coordinates(order=1). isochrone(log_age,feh,mass)->(G,color).
"""
import glob, numpy as np, jax, jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates
jax.config.update("jax_enable_x64", True)
_F=sorted(glob.glob("/Users/notluquis/COSMIC/data/test/NGC6383/MIST/UBVRIplus/*.iso.cmd"))
_C=dict(EEP=0,AGE=1,FEH=8,MASS=2,G=30,BP=31,RP=32)   # MASS = initial_mass col idx2
def _blocks(fn):
    rows=[]
    for line in open(fn):
        if line.startswith('#') or not line.strip():
            if rows: yield np.array(rows); rows=[]
            continue
        v=line.split(); rows.append([float(v[_C[k]]) for k in ('EEP','AGE','FEH','MASS','G','BP','RP')])
    if rows: yield np.array(rows)
AGE_AXIS=np.round(np.arange(6.0,8.01,0.05),2)
MASS_AXIS=np.round(np.geomspace(0.15,7.0,80),4)   # log-spaced, PMS->intermediate (common at young age)
_fehs=[]; _pf=[]
for fn in _F:
    isos={}; feh=None
    for blk in _blocks(fn):
        if blk.size==0: continue
        feh=round(blk[0,2],4)
        # cols: [EEP,AGE,FEH,MASS,G,BP,RP] -> MASS=3,G=4,BP=5,RP=6
        isos[round(blk[0,1],2)]=blk[np.argsort(blk[:,3])]   # sort by initial_mass
    _fehs.append(feh); _pf.append(isos)
_o=np.argsort(_fehs); _fehs=np.array(_fehs)[_o]; _pf=[_pf[i] for i in _o]
def _onM(blk,col): return np.interp(MASS_AXIS,blk[:,3],blk[:,col],np.nan,np.nan)
def _iso_at(isos,a,col):
    ages=np.array(sorted(isos))
    if a<=ages[0]: return _onM(isos[ages[0]],col)
    if a>=ages[-1]: return _onM(isos[ages[-1]],col)
    j=np.searchsorted(ages,a); a0,a1=ages[j-1],ages[j]; w=(a-a0)/(a1-a0)
    return (1-w)*_onM(isos[a0],col)+w*_onM(isos[a1],col)
_gG=np.array([[_iso_at(_pf[jf],a,4) for jf in range(len(_fehs))] for a in AGE_AXIS])
_gBP=np.array([[_iso_at(_pf[jf],a,5) for jf in range(len(_fehs))] for a in AGE_AXIS])
_gRP=np.array([[_iso_at(_pf[jf],a,6) for jf in range(len(_fehs))] for a in AGE_AXIS])
_gC=_gBP-_gRP
_good=~(np.isnan(_gG).any((0,1))|np.isnan(_gC).any((0,1)))
_gG=_gG[:,:,_good]; _gC=_gC[:,:,_good]; mass_axis=jnp.asarray(MASS_AXIS[_good])
_gGj=jnp.asarray(_gG); _gCj=jnp.asarray(_gC)
_AGEj=jnp.asarray(AGE_AXIS); _FEHj=jnp.asarray(_fehs); _NM=int(_good.sum())
def isochrone(log_age, feh, mass):
    ai=jnp.interp(log_age,_AGEj,jnp.arange(len(_AGEj)))
    fi=jnp.interp(feh,_FEHj,jnp.arange(len(_FEHj)))
    mi=jnp.interp(mass,mass_axis,jnp.arange(_NM))
    ai=jnp.broadcast_to(ai,mi.shape); fi=jnp.broadcast_to(fi,mi.shape)
    return (map_coordinates(_gGj,[ai,fi,mi],order=1,mode="nearest"),
            map_coordinates(_gCj,[ai,fi,mi],order=1,mode="nearest"))
_m=np.asarray(mass_axis); _w=np.where(_m<0.5,_m**-1.3,0.5**-1.3*(_m/0.5)**-2.3)*np.gradient(_m)
_w/=_w.sum(); imf_logw=jnp.asarray(np.log(np.clip(_w,1e-30,None)))
AGE_MIN,AGE_MAX=float(AGE_AXIS[0]),float(AGE_AXIS[-1]); FEH_MIN,FEH_MAX=float(_fehs[0]),float(_fehs[-1])
if __name__=="__main__":
    G,C=isochrone(6.6,0.0,mass_axis)
    print(f"mass-grid: NMASS={_NM}, mass[{float(mass_axis.min()):.2f},{float(mass_axis.max()):.2f}]")
    print(f"iso(6.6,0): G[{float(G.min()):.1f},{float(G.max()):.1f}] col[{float(C.min()):.2f},{float(C.max()):.2f}] (should span blue..red)")
