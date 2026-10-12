#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=/Users/notluquis/erotica-wt-magnetic
PY=~/miniforge3/envs/erotica/bin/python
cd "$(dirname "$0")"

run() { echo "$(date +%T) START $*" >> logs/lanes.log; nice -n 19 $PY run_arms.py "$@" >> "logs/$(echo $* | tr ' ,-' '___').log" 2>&1; echo "$(date +%T) END($?) $*" >> logs/lanes.log; }
( run e3 --src identity --arm g; run e3 --src spots51 --arm s; run e3 --src spots34_2T --arm a; run e3 --src spots85 --arm a --bc iso ) &
( run e3 --src spots34 --arm g; run e3 --src spots17 --arm s; run e3 --src spots34_2T --arm s; run e3 --src spots51 --arm s --bc iso ) &
( run e3 --src feiden --arm g; run e3 --src spots34 --arm s --bc iso; run e3 --src feiden --arm s --bc iso; run e3 --src spots85 --arm g ) &
wait
echo "$(date +%T) ALL2 DONE" >> logs/lanes.log
