#!/bin/bash
# control-only preps, one at a time at low priority (the NUTS fits hold 4 cores)
cd "$(dirname "$0")"
mkdir -p logs
PY=${PYTHON:-python}
for item in "$@"; do
  cell=${item%%:*}; rep=${item##*:}
  [ -f runs/${cell}_${rep}/starts.json ] && continue
  NPROC=1 XLA_FLAGS=--xla_force_host_platform_device_count=1 nice -n 15 $PY prep.py --cell $cell --rep $rep > logs/prep_${cell}_${rep}.log 2>&1
done
