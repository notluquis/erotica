#!/bin/bash
# Profile of log t (hub isochrone-intervals-nuts.md §2.16): NW workers, one BLAS thread and one
# pytensor compiledir each, round-robin over the grid points without a JSON yet. PIDs in logs/profile.pids.
cd "$(dirname "$0")" || exit 1
NW=${NW:-6}
PY=~/miniforge3/envs/erotica/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1
todo=()
for t in $($PY profile_loga.py grid); do
  [ -f "kw_runs/real_0/profile/t_${t}.json" ] || todo+=("$t")
done
: > logs/profile.pids
for w in $(seq 0 $((NW - 1))); do
  mine=()
  for i in "${!todo[@]}"; do [ $((i % NW)) -eq "$w" ] && mine+=("${todo[$i]}"); done
  [ ${#mine[@]} -eq 0 ] && continue
  PYTENSOR_FLAGS="compiledir=$HOME/.pytensor/profile_w$w" nohup $PY profile_loga.py run "${mine[@]}" \
    > "logs/profile_w$w.log" 2>&1 &
  echo $! >> logs/profile.pids
done
cat logs/profile.pids
