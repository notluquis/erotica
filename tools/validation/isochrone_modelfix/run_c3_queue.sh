#!/bin/bash
# C3' (hub §2.17): one worker = a list of "stage arm rep" jobs, one at a time, 1 thread each.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/miniforge3/envs/erotica/bin/python
while read -r stage arm rep; do
  [ -z "$stage" ] && continue
  case $stage in
    prepL) [ -f kw_runs/${arm}_$rep/prep.json ] || nice -n 5 $PY kw_c3.py prep --arm $arm --rep $rep --no-search8 > logs/kwc3prep_${arm}_$rep.log 2>&1 ;;
    prep)  [ -f kw_runs/${arm}_$rep/prep.json ] || nice -n 5 $PY kw_c3.py prep --arm $arm --rep $rep > logs/kwc3prep_${arm}_$rep.log 2>&1 ;;
    glob)  [ -f kw_runs/${arm}_$rep/global.json ] || nice -n 5 $PY kw_c3.py glob --arm $arm --rep $rep > logs/kwc3glob_${arm}_$rep.log 2>&1 ;;
  esac
  echo "$(date '+%F %T') $stage $arm $rep done"
done
