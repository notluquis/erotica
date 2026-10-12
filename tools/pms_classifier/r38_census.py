"""R38 §4.4: censo de NGC 6383 en los tramos establecidos por el POOL fuera de muestra (G2-OOS + sintetico de §2.4: -5..+3, +3..+5,
+5..+7). No reentrena: suma, sorteo a sorteo, las 30 realizaciones conjuntas (modelo + calibracion + A_V/MOD) de exp2_census
(exp2_census_measured.txt, lineas "b k: ... total [t0, t1, t2, t3]"). El censo determinista CAL/BEST/T2M por tramo es [N1] de ese fichero.
Corre: ~/miniforge3/envs/erotica/bin/python r38_census.py > r38_census_measured.txt
"""
import re

import numpy as np

tot = np.array([[int(v) for v in re.search(r"total \[([^\]]+)\]", l).group(1).split(",")]
                for l in open("exp2_census_measured.txt") if re.match(r"\s+b \d+:", l)])
assert tot.shape == (30, 4), tot.shape
for nm, cols in (("-5..+3 y +5..+7 (B, §3.3)", [0, 2]), ("-5..+3, +3..+5 y +5..+7 (pool, §4)", [0, 1, 2]), ("+3..+5", [1]),
                 ("+5..+7", [2]), ("-5..+3", [0]), ("+7..+12 (no establecido)", [3])):
    s = tot[:, cols].sum(1)
    print(f"{nm:38s} mediana {np.median(s):.0f}  [2,5; 97,5 %] [{np.percentile(s, 2.5):.0f}; {np.percentile(s, 97.5):.0f}]  "
          f"68 % [{np.percentile(s, 16):.0f}; {np.percentile(s, 84):.0f}]  (B = 30: el 95 % es casi el rango; min {s.min()} max {s.max()})")
