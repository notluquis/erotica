#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=/Users/notluquis/erotica-wt-magnetic
PY=~/miniforge3/envs/erotica/bin/python
cd "$(dirname "$0")"
until grep -q "ALL3 DONE" logs/lanes.log; do sleep 30; done
run() { echo "$(date +%T) START $*" >> logs/lanes.log; nice -n 19 $PY run_arms.py "$@" >> "logs/$(echo $* | tr ' ,-' '___').log" 2>&1; echo "$(date +%T) END($?) $*" >> logs/lanes.log; }
( run where_null --reps 4-7 ) &
( run mech --which rv ) &
( run mech --which bin; run mech --which bg ) &
wait
echo "$(date +%T) ALL4 DONE" >> logs/lanes.log
