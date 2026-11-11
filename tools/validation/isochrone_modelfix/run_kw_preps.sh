#!/bin/bash
# Prepare the K x W interval cells (hub finding isochrone-intervals-nuts.md §2): A_0..7 first, then
# the data, then C_0..7; at most P in parallel (1 thread each). Skips a fit whose prep.json exists.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
P=${P:-8}
PY=~/miniforge3/envs/erotica/bin/python
{ for k in 0 1 2 3 4 5 6 7; do echo "A $k"; done; echo "real 0"; for k in 0 1 2 3 4 5 6 7; do echo "C $k"; done; } |
  xargs -P "$P" -L 1 sh -c 'd=kw_runs/$0_$1; [ "$0" = real ] && d=kw_runs/real_0; [ -f $d/prep.json ] || nice -n 5 '"$PY"' kw_cells.py prep --cell $0 --rep $1 > logs/kwprep_$0_$1.log 2>&1; echo "$(date "+%F %T") prep $0 $1 done"'
