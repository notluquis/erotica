#!/bin/bash
# three at a time, single-threaded each (machine shared; <= 3 cores)
cd "$(dirname "$0")"; mkdir -p logs fits
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
PY=~/miniforge3/envs/erotica/bin/python
for w in F W; do
  for a in I V VR; do nice -n 10 $PY fit_arm.py $a $w > logs/fit_${a}_${w}.log 2>&1 & done
  wait
done
