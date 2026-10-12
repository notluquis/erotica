"""Cuenta, en el cruce 2MASS por X-Match de este directorio (cache/tmass.ecsv: el vecino mas cercano por source_id), las bandas con
ph_qual = U, si traen magnitud y si traen error. Respaldo de §3.4 del hallazgo y de la receta §4b de la skill astro-data-access.
Corre: ~/miniforge3/envs/erotica/bin/python tmass_u_counts.py > tmass_u_counts_measured.txt
"""
import numpy as np
from astropy.table import Table

t = Table.read("cache/tmass.ecsv")
q = np.array([str(x) for x in t["ph_qual"]])
print(f"filas (source_id con contraparte a <= 1''): {len(t)}; separacion mediana {np.median(np.asarray(t['angDist'], float)):.3f}''")
for i, (b, e) in enumerate((("j_m", "e_Jmag"), ("h_m", "e_Hmag"), ("ks_m", "e_Kmag"))):
    u = np.array([len(s) == 3 and s[i] == "U" for s in q])
    mm, em = np.ma.getmaskarray(np.ma.asarray(t[b])), np.ma.getmaskarray(np.ma.asarray(t[e]))
    print(f"  {b}: U {u.sum()}; con magnitud {np.sum(~mm[u])}; con error nulo {np.sum(em[u])}; sin U y error nulo {np.sum(em[~u])} de {(~u).sum()}")
