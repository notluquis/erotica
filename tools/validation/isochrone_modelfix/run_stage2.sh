#!/bin/bash
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=~/miniforge3/envs/erotica/bin/python
for a in K B KB; do nice -n 10 $PY fit_arm.py $a W > logs/fit_${a}_W.log 2>&1 & done; wait
for a in K B KB; do nice -n 10 $PY ppc2.py $a W 200 > logs/ppc_${a}_W_R200.log 2>&1 & done; wait
