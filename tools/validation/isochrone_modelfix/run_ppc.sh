#!/bin/bash
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=~/miniforge3/envs/erotica/bin/python; R=${R:-200}
for grp in "I_F V_F VR_F" "I_W V_W VR_W"; do
  for aw in $grp; do a=${aw%_*}; w=${aw#*_}; nice -n 10 $PY ppc2.py $a $w $R > logs/ppc_${a}_${w}_R$R.log 2>&1 & done; wait
done
