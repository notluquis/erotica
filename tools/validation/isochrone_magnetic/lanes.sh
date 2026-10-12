#!/bin/bash
# 4 lanes, sequential jobs per lane; nice 19; one BLAS thread each.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=/Users/notluquis/erotica-wt-magnetic
PY=~/miniforge3/envs/erotica/bin/python
cd "$(dirname "$0")"
run() { echo "$(date +%T) START $*" >> logs/lanes.log; nice -n 19 $PY run_arms.py "$@" >> "logs/$(echo $* | tr ' ,-' '___').log" 2>&1; echo "$(date +%T) END($?) $*" >> logs/lanes.log; }
( run e3 --src identity --arm a; run e3 --src spots85 --arm a; run e3 --src spots34 --arm s; run e3 --src spots17 --arm a ) &
( run e3 --src spots34 --arm a; run e3 --src identity --arm s; run e3 --src spots85 --arm s; run e3 --src spots51 --arm a ) &
( run e3 --src feiden --arm a; run e3 --src feiden --arm s; run e3 --src spots34 --arm a --bc iso; run e3 --src feiden --arm a --bc iso ) &
( run where; run where_null --reps 0-7; run mech --which bg; run mech --which rv; run mech --which bin ) &
wait
echo "$(date +%T) ALL DONE" >> logs/lanes.log
