# Registro de corridas de isócronas

Un JSON por corrida en `runs/`, para ver cómo cambia la isócrona por versión del código y por modelo
de grilla. Lo escribe `export_run.py` (ver su docstring: qué fuente lee cada corrida y qué no
reimplementa); lo verifica `tests/test_isochrone_runs.py`. La GUI `~/phd-dashboard` lee este
directorio desde `$EROTICA_RUNS` (apúntalo a `.../isochrone_runs/runs`).

## Corridas presentes

| id | qué es | cadenas |
|---|---|---|
| `p01-ngc6383-asteca-demetropolis` | P01 publicado (ASteCA + MIST v1.2, DEMetropolis **no convergido**); punto de referencia, `q50` = moda | no |
| `hess-ngc6383-binned-507f779` | likelihood Hess binada, 4 cadenas; convergió pero **anclada a la referencia de la grilla** (convergencia ≠ calibración) | no (sólo resumen) |
| `c1-ngc6383-unbinned-eep-v1-96ce4c0` | C1: NUTS 4×2000, likelihood no binada por estrella, EEP | sí (`~/.cache/erotica-c1`, fuera de git) |

No se exportan: `isochrone_unbinned/ngc_attempt1_start_on_bound` (3 de 4 cadenas congeladas en el
arranque, 6000 divergencias: no es un posterior), las corridas `A_*`/`B_*`/`loo_*` (cúmulos
sintéticos, no NGC 6383) ni la corrida de 2026-06-11 (sin resumen en el repo). Añadirlas no exige
MCMC nuevo salvo esta última.

## Esquema (`schema_version` 1)

| campo | tipo | contenido |
|---|---|---|
| `schema_version` | int | 1 |
| `id` | str | = nombre del fichero sin `.json` |
| `date` | str | fecha de la corrida (ISO) |
| `erotica_commit` | str \| null | commit del código que corrió (`null` si no fue EROTICA); `result_commit` opcional = commit que guardó el resultado |
| `model` | obj | `grid`, `version`, `files`, `photometry`, `z_sun`, `z_sun_source` |
| `cluster` | str | |
| `method` | str | muestreador + likelihood |
| `config` | obj | priors (texto), binarias, IMF, R_V, piso, muestreador — lo que haya |
| `posterior` | obj | por parámetro (`met`, `loga`, `dm`, `Av`, `sigma_int`, `f_bg`): `q05`,`q16`,`q50`,`q84`,`q95`, `rhat`, `ess_bulk`, `chain_means`; `null` si la corrida no lo tiene. P01 usa `q50` + `pm` o `plus`/`minus` |
| `diagnostics` | obj \| null | `divergences`, `bfmi` (por cadena), `tree_depth_chain_mean`, `step_size_chain_mean`, `chains`, `draws_per_chain` |
| `cmd` | obj | puntos usados: `source_id`, `color` (BP−RP), `mag` (G), `e_color`, `e_mag` (los que entran a la likelihood), `membership_prob` + `membership_column`, `mag_limit` |
| `isochrone` | obj | ver abajo |
| `notes` | [str] | procedencia y advertencias |

**`met` es Z lineal** (`Zinit` de MIST), no [M/H]: [M/H] = log10(Z / `model.z_sun`). La nota de cada
corrida trae la conversión.

### `isochrone`

- `computed_with`: con qué código se calculó la curva. **Las corridas viejas se recalculan con la
  interpolación EEP actual** en su mediana; su propia likelihood pudo usar otra.
- `median`: isócrona de **estrellas simples** (sin binarias ni errores) en la mediana de
  met/loga/dm/Av, ya en el sistema observado (G, BP−RP con CCM89 + O'Donnell, R_V = 3,1).
  Polilínea ordenada por EEP: `eep`, `mass` (inicial, M☉), `color`, `mag`. Sin las colas
  sujetadas de la tabla de nodos.
- `band_16_84` (o `null` sin cadenas): **por punto EEP** (mismos índices que `median`), percentiles
  16 y 84 del color y de G **por separado**, sobre `n_draws` extracciones del posterior. No es una
  banda a G fijo: una isócrona de ~1 Myr no es monótona en G en la llegada a la secuencia
  principal. Para dibujarla: dos polilíneas `(color_q16, mag_q16)` y `(color_q84, mag_q84)`, o
  mejor `draws`.
- `draws` (o `null`): ~30 isócronas de extracciones del posterior, cada una con sus `params` y la
  polilínea diezmada (1 de cada 3 puntos EEP).

## Regenerar

```bash
cd ~/erotica-wt-runs   # rama isochrone-runs
OMP_NUM_THREADS=1 ~/miniforge3/envs/cosmic/bin/python tools/validation/isochrone_runs/export_run.py
~/miniforge3/envs/cosmic/bin/python -m pytest tests/test_isochrone_runs.py
```

El test de la curva necesita los MIST locales y se salta en CI; el de cuantiles sólo lee JSON
versionados y corre en cualquier sitio.
