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

## Resultados

Los scripts escriben sus resultados completos en `cache/` (ignorado). `results/` guarda las cinco sidecars
agregadas de la corrida que imprimió los `*_measured.txt` (copiadas de `cache/` del worktree de la rama):
`exp2_summary.json` (las 48 configuraciones de la grilla en la mitad A), `exp2_selected.json` (la elegida),
`exp2_confirm.json` (compuertas en la mitad B y tramos establecidos), `r38_oos_eval.json` (FPR en el pool de
352 cúmulos viejos fuera de muestra) y `r38_s2.json` (umbrales del brazo S2). Los modelos (`model_*.json`,
~2 MB cada uno) y las predicciones por estrella no se versionan.

## Defecto conocido: la dispersión de las simulaciones de potencia

`exp2_power.py` y `r38_power.py` (`sim`) generan el FPR por cúmulo como Beta(p a, (1 − p) a) con
`a = (1 − p(1 + cv²)) / cv²`; para un coeficiente de variación `cv` la concentración es
`a = (1 − p(1 + cv²)) / (p cv²)`. Falta el factor `1/p`: con `cv = 1` el CV simulado es 3,16 a p = 0,05,
4,08 a p = 0,03 y 1,83 a p = 0,15 (2·10⁶ draws; con la fórmula corregida, 1,00 en los tres). Las columnas
`disp` de `exp2_power_measured.txt` y `r38_power_measured.txt` describen entonces una dispersión entre cúmulos
mayor que la declarada; las columnas `binom` no dependen de esto. El código y las salidas quedan como se
corrieron porque son las que cita el hallazgo; corregirlas y releerlas es una decisión del análisis, no de
este traslado.

**Corregido el 2026-11-12 (R113-01).** La concentración vive ahora en `power_sim.py` (`beta_concentration`,
`cluster_fpr`), que usan los dos scripts; `tests/test_pms_power_sim.py` verifica la media y el CV de la Beta
(momentos exactos y 2·10⁶ draws) y se pone en rojo con la fórmula sin 1/p. Las corridas nuevas están **al lado**
de las viejas: `exp2_power_r113_measured.txt` y `r38_power_r113_measured.txt` (env `erotica-2026-10`, misma semilla).
Con la dispersión declarada (CV = 1), el falso fallo con FPR 0,05 de la mitad confirmatoria B baja de 0,41–0,49 a
0,007–0,14 por tramo (0,013 en «todos») y con FPR 0,03 de 0,19–0,27 a 0,000; en el pool fuera de muestra, todas
las columnas `disp` quedan en 0,000 (eran 0,003–0,21). Las columnas `binom` no dependen de la fórmula; la única
que cambió (+7..+12, falso fallo binomial 0,007 → 0,000, es decir 2/300 → 0/300) se mueve porque el generador es
compartido y la Beta consume otra cantidad de uniformes: ruido Monte Carlo, no efecto. Los `*_measured.txt` viejos
quedan como el registro que citaba el hallazgo.
