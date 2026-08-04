"""Throwaway inspection: MIST grid structure -- age nodes, EEP coverage, mass range."""
import glob, time
import numpy as np, pandas as pd

FILES = sorted(glob.glob("/Users/notluquis/erotica/data/test/NGC6383/MIST/UBVRIplus/*.iso.cmd"))
print(f"{len(FILES)} feh files")
USE = [0, 1, 2, 8, 30, 31, 32, 33]   # EEP, logage, Mini, feh, G, BP, RP, phase
NAMES = ["eep", "logage", "mini", "feh", "G", "BP", "RP", "phase"]

fn = [f for f in FILES if "p0.00" in f][0]
t0 = time.time()
df = pd.read_csv(fn, sep=r"\s+", comment="#", header=None, usecols=USE, names=NAMES,
                 engine="c")
print(f"parsed {fn.split('/')[-1]} in {time.time()-t0:.1f}s -> {df.shape}")
ages = np.unique(df.logage.values)
print(f"age nodes: {len(ages)}  {ages[:3]} ... {ages[-3:]}  spacing {np.diff(ages)[:5]}")

sub = df[(df.logage >= 5.99) & (df.logage <= 8.01)]
print(f"\nage window [6.0,8.0]: {len(np.unique(sub.logage))} nodes, {len(sub)} rows")
print(f"{'logage':>8} {'nEEP':>5} {'EEPmin':>7} {'EEPmax':>7} {'Mmin':>7} {'Mmax':>8} {'phases'}")
for a in np.unique(sub.logage)[::8]:
    b = sub[sub.logage == a]
    print(f"{a:8.2f} {len(b):5d} {b.eep.min():7.0f} {b.eep.max():7.0f} "
          f"{b.mini.min():7.3f} {b.mini.max():8.2f} {sorted(set(b.phase.values.astype(int)))}")

# common EEP range across the whole age window
eepmin = max(sub[sub.logage == a].eep.min() for a in np.unique(sub.logage))
eepmax = min(sub[sub.logage == a].eep.max() for a in np.unique(sub.logage))
print(f"\ncommon EEP range across ALL ages in [6,8]: [{eepmin:.0f}, {eepmax:.0f}]")
# EEP monotone in mass?
b = sub[sub.logage == 6.60]
print(f"at 6.60: EEP sorted={np.all(np.diff(b.eep.values)>0)}, "
      f"mass monotone={np.all(np.diff(b.mini.values)>0)}")
