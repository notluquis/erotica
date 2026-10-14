#!/bin/bash
# Continuation sweeps of profile_loga.py (hub §2.16) in parallel Jacobi passes until max gain <= 0.01.
cd "$(dirname "$0")" || exit 1
NW=${NW:-6}
PY=~/miniforge3/envs/erotica/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
pts=($(ls kw_runs/real_0/profile/t_*.json | sed 's/.*t_//; s/.json//' | sort -n))
for pass in 0 1 2 3 4 5; do
  pids=()
  for w in $(seq 0 $((NW - 1))); do
    mine=()
    for i in "${!pts[@]}"; do [ $((i % NW)) -eq "$w" ] && mine+=("${pts[$i]}"); done
    PYTENSOR_FLAGS="compiledir=$HOME/.pytensor/profile_w$w" $PY profile_loga.py sweep_pass $pass "${mine[@]}" \
      > "logs/sweep_p${pass}_w$w.log" 2>&1 &
    pids+=($!)
  done
  echo "${pids[@]}" > logs/sweep.pids
  for p in "${pids[@]}"; do wait "$p"; done
  g=$(grep -h '^GAIN' logs/sweep_p${pass}_w*.log | awk '{print $2}' | sort -g | tail -1)
  echo "pass $pass max gain $g"
  awk -v g="$g" 'BEGIN{exit !(g <= 0.01)}' && break
done
echo SWEEP_DONE
