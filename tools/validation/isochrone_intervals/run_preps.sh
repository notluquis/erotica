#!/bin/bash
# prep (control + starts) for a list of cell:rep, at most 4 at a time, one core each
cd "$(dirname "$0")"
for item in "$@"; do
  while [ "$(pgrep -f 'prep.py --cell' | wc -l)" -ge 4 ]; do sleep 10; done
  cell=${item%%:*}; rep=${item##*:}
  if [ -f runs/${cell}_${rep}/starts.json ]; then continue; fi
  NPROC=1 XLA_FLAGS=--xla_force_host_platform_device_count=1 nohup ~/miniforge3/envs/erotica/bin/python prep.py --cell $cell --rep $rep > logs/prep_${cell}_${rep}.log 2>&1 &
  sleep 2
done
wait
