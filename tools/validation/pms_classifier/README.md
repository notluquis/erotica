# Clasificador PMS no circular para NGC 6383

Clasificador pre-secuencia principal (XGBoost sobre el CMD absoluto desenrojecido por cúmulo, Gaia + 2MASS)
entrenado **sin** NGC 6383 ni sus vecinos de Sh 2-012, para que el censo PMS de NGC 6383 no sea circular:
Sagitta (McBride et al. 2021) tiene 58 de sus 254 miembros en su entrenamiento. Prerregistro, resultados y la
revisión adversarial (R38) viven en el hallazgo del hub `pms-classifier-non-circular.md`; cada `*_measured.txt`
es la salida de su script homónimo en la rama `pms-classifier` (`300e028`, conservada en el tag
`archive/pms-classifier`), y el código se trae sin reformatear para que siga siendo el que imprimió esas cifras
(`pyproject.toml`, per-file-ignores de este directorio). Lo único que cambió son las rutas de entrada: leían
`Path.home()/…` y ahora leen las variables de abajo (`core.input_table`).

## Entradas que no viven en este repositorio

| variable | qué es |
|---|---|
| `NGC6383_VIRAC2_MEMBERS` | `virac2_ngc6383_members_clean.ecsv`: miembros de NGC 6383 con VIRAC2 y 2MASS (no publicada) |
| `P01_CDS_MEMBERS` | `ngc6383_members.ecsv`: tabla de miembros del CDS de Pulgar-Escobar et al. (A&A 715, A228) |
| `PMS_MIST_UBVRIPLUS` | isócrona MIST v1.2 UBVRIplus ([Fe/H] = 0, v/v_crit = 0,4); por defecto `data/test/NGC6383/MIST/UBVRIplus/` del checkout |
| `SAGITTA_SRC` | sdist extraído de `sagitta-1.2.3.tar.gz` (control con Sagitta; ver `sagitta_ref.py`) |

Los datos de entrenamiento y validación los bajan `fetch_training.py`, `fetch_validation.py` y `fetch_oos.py` a
`cache/` (ignorado por git).

## Dependencias

Fuera de las del paquete: `xgboost`, `scikit-learn`, `astroquery`, `pyvo` y, para el control con Sagitta,
`torch`. No son extras de `erotica`: nada del paquete las importa. `test_core.py` se salta sin `xgboost`
(`python -m pytest -q test_core.py` desde este directorio; `pytest` en la raíz no lo recoge).
