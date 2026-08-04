"""EEP<->mass mapping vs age: choose the emulator box honestly."""
import glob
import numpy as np, pandas as pd
FILES = sorted(glob.glob("/Users/notluquis/erotica/data/test/NGC6383/MIST/UBVRIplus/*.iso.cmd"))
USE=[0,1,2,8,30,31,32,33]; NAMES=["eep","logage","mini","feh","G","BP","RP","phase"]
fn=[f for f in FILES if "p0.00" in f][0]
df=pd.read_csv(fn,sep=r"\s+",comment="#",header=None,usecols=USE,names=NAMES)
print("mass at fixed EEP, vs age  (feh=0.00)")
print(f"{'logage':>7} " + " ".join(f"{'E'+str(e):>9}" for e in (148,202,300,343,400,454,500,605)) + f"{'Mmax':>9}{'EEPmax':>8}")
for a in np.arange(6.0,8.01,0.2):
    b=df[np.isclose(df.logage,a)].sort_values("eep")
    row=[]
    for e in (148,202,300,343,400,454,500,605):
        row.append(f"{np.interp(e,b.eep,b.mini,np.nan,np.nan):9.3f}")
    print(f"{a:7.2f} "+" ".join(row)+f"{b.mini.max():9.2f}{b.eep.max():8.0f}")

print("\nmagnitude at fixed EEP vs age (how much CMD is inside EEP<=343?)")
for a in (6.0,6.6,7.2,8.0):
    b=df[np.isclose(df.logage,a)].sort_values("eep")
    inb=b[(b.eep>=148)&(b.eep<=343)]
    print(f" logage {a}: EEP[148,343] -> mass[{inb.mini.min():.3f},{inb.mini.max():.2f}] "
          f"G[{inb.G.min():.2f},{inb.G.max():.2f}] | FULL iso G[{b.G.min():.2f},{b.G.max():.2f}] "
          f"mass[{b.mini.min():.2f},{b.mini.max():.1f}]")

print("\n-- how EEPmin/EEPmax move with age (feh=0) --")
for a in np.arange(6.0,8.01,0.1):
    b=df[np.isclose(df.logage,a)]
    print(f"  {a:.2f}  EEP[{b.eep.min():4.0f},{b.eep.max():5.0f}]  N={len(b):5d}", end="")
    if abs(a*10 % 3) < 1e-6: print()
print()
