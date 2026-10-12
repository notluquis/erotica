#!/bin/bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONPATH=/Users/notluquis/erotica-wt-magnetic
PY=~/miniforge3/envs/erotica/bin/python
cd "$(dirname "$0")"
run() { echo "$(date +%T) START $*" >> logs/lanes.log; nice -n 19 $PY run_arms.py "$@" >> "logs/$(echo $* | tr ' ,-' '___').log" 2>&1; echo "$(date +%T) END($?) $*" >> logs/lanes.log; }
( run e3 --src spots51_2T --arm s; run e3 --src spots51_2T --arm s --bc iso; run e3 --src spots85_2T --arm s; run e3 --src spots85_2T --arm s --bc iso; run e3 --src spots34_2T --arm s --bc iso; run e3 --src identity --arm a --windows pms19; run e3 --src spots34 --arm a --windows pms19 ) &
( run e3 --src spots85 --arm m --bc iso; run e3 --src identity --arm s --windows ums,b11; run e3 --src feiden --arm s --windows ums,b11; run e3 --src identity --arm m --windows ums,b11; run e3 --src feiden --arm m --windows ums,b11; TRANSPLANT_TAPER=0 run e3 --src feiden --arm m; run e3 --src feiden --arm a --windows pms19 ) &
wait
echo "$(date +%T) ALL5 DONE" >> logs/lanes.log
