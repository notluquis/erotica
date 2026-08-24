"""Arnés de mutación que VERIFICA que la mutación llegó al fichero. → K.1.28.

La primera versión usaba `count=1` sobre un objetivo que aparecía dos veces y no comprobaba nada:
mutaba pmra y dejaba pmdec intacto, la sonda pasaba, y la conclusión habría sido "el guardia tiene
un agujero" sobre un guardia que funcionaba.
"""

import pathlib
import shutil
import subprocess
import sys

RAIZ = pathlib.Path("/Users/notluquis/erotica")
F = RAIZ / "tools/validation/benchmark_erotica_vs_asteca.py"
SONDA = RAIZ / "tools/probes/neighbour_injection.py"
PY = "/Users/notluquis/miniforge3/envs/erotica-bench/bin/python"
BK = pathlib.Path(
    "/private/tmp/claude-501/-Users-notluquis-phd/9a0f9b71-83f6-46ec-ba5e-2fc967bc6851/scratchpad/b6_good.py"
)

MUTS = [
    (
        "vecino con dispersion de campo",
        "rng.normal(0.0, SIGMA_PM_INT, n_neigh)",
        "rng.normal(0.0, FIELD_PM_SIGMA, n_neigh)",
        -1,
    ),  # -1 = todas las apariciones
    (
        "contaminacion excluye al vecino",
        "(n_neigh + n_field) / max(n_tot, 1)",
        "n_field / max(n_tot, 1)",
        1,
    ),
    ("delta_pm anulado en un eje", "delta_pm * SIGMA_PM_INT * np.cos(ang)", "0.0 * np.cos(ang)", 1),
    ("delta_pm anulado en LOS DOS ejes", "delta_pm * SIGMA_PM_INT", "0.0 * SIGMA_PM_INT", -1),
    ("delta_plx anulado", "plx_n_c = PLX_C + delta_plx * SIGMA_PLX_INT", "plx_n_c = PLX_C", 1),
    (
        "el vecino contado como MIEMBRO",
        "np.zeros(n_neigh + n_field, bool)",
        "np.concatenate([np.ones(n_neigh, bool), np.zeros(n_field, bool)])",
        1,
    ),
    ("el vecino no se reparte, se suma", "n_field -= n_neigh", "pass", 1),
]

shutil.copy(F, BK)
bueno = BK.read_text()
agujeros = []
for nombre, viejo, nuevo, n in MUTS:
    F.write_text(bueno)
    apariciones = bueno.count(viejo)
    if apariciones == 0:
        print(f"  [SONDA MALA] {nombre}: el objetivo no existe")
        continue
    F.write_text(bueno.replace(viejo, nuevo) if n == -1 else bueno.replace(viejo, nuevo, n))
    # LA verificación que faltaba: ¿el fichero cambió?
    if F.read_text() == bueno:
        print(f"  [SONDA MALA] {nombre}: el reemplazo fue un no-op")
        continue
    hechas = apariciones if n == -1 else min(n, apariciones)
    r = subprocess.run([PY, str(SONDA)], capture_output=True, text=True, cwd=RAIZ)
    verde = r.returncode == 0
    if verde:
        agujeros.append(nombre)
    ultima = (r.stdout.strip().splitlines() or [""])[-1]
    marca = "PASA <- AGUJERO" if verde else "roja"
    print(f"  [{marca:15s}] {nombre} ({hechas}/{apariciones} sitios)")
    if not verde:
        print(f"                    {ultima.strip()[:96]}")

F.write_text(bueno)
r = subprocess.run([PY, str(SONDA)], capture_output=True, text=True, cwd=RAIZ)
print(f"\n  restaurado: {'verde' if r.returncode == 0 else 'ROJO — algo quedó mal'}")
print(f"  agujeros reales: {agujeros or 'ninguno'}")
sys.exit(1 if agujeros or r.returncode else 0)
