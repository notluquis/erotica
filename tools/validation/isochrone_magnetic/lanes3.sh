#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=/Users/notluquis/erotica-wt-magnetic
PY=~/miniforge3/envs/erotica/bin/python
cd "$(dirname "$0")"
until grep -q "ALL2 DONE" logs/lanes.log; do sleep 30; done
run() { echo "$(date +%T) START $*" >> logs/lanes.log; nice -n 19 $PY run_arms.py "$@" >> "logs/$(echo $* | tr ' ,-' '___').log" 2>&1; echo "$(date +%T) END($?) $*" >> logs/lanes.log; }
( run e3 --src identity --arm m; run e3 --src spots85 --arm m; run e3 --src spots34 --arm sm ) &
( run e3 --src feiden --arm m; run e3 --src identity --arm sm ) &
( run e3 --src spots34 --arm m; run e3 --src feiden --arm sm ) &
wait
echo "$(date +%T) ALL3 DONE" >> logs/lanes.log
