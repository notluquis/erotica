#!/bin/bash
# NUTS on the K x W model (hub finding isochrone-intervals-nuts.md §2) for a list of kw_runs/<fit>,
# one fit at a time, its 4 chains in parallel (1 core each); replicates 500 + 500, real_* 1000 + 1000.
# Waits for the fit's starts.json; resumes from each chain's state.npz; finalize writes nuts.json,
# the pair/trace figures and draws.npz. Run two of these (disjoint lists) for 8 cores.
cd "$(dirname "$0")"
export ARM=K WINDOW=W OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=~/miniforge3/envs/erotica/bin/python
for r in "$@"; do
  until [ -f kw_runs/$r/prep.json ]; do sleep 30; done
  if [ -f kw_runs/$r/nuts.json ]; then continue; fi
  case $r in real*) W=1000; D=1000;; *) W=500; D=500;; esac
  for c in 0 1 2 3; do
    if [ ! -f kw_runs/$r/chain$c/DONE ]; then
      NPROC=1 XLA_FLAGS=--xla_force_host_platform_device_count=1 nohup nice -n 5 $PY nuts8.py run --data kw_runs/$r/data.ecsv --starts kw_runs/$r/starts.json --chain $c --outdir kw_runs/$r/chain$c --warmup $W --draws $D > logs/kwnuts_${r}_c$c.log 2>&1 &
    fi
  done
  wait
  truth=$($PY -c "import json;d=json.load(open('kw_runs/$r/prep.json'));print(json.dumps(d['truth_pymc']) if d.get('truth_pymc') else '')")
  NPROC=1 $PY nuts8.py finalize --data kw_runs/$r/data.ecsv --outdirs kw_runs/$r/chain0,kw_runs/$r/chain1,kw_runs/$r/chain2,kw_runs/$r/chain3 --json kw_runs/$r/nuts.json --fig kw_runs/$r/pair.png --truth "$truth" --title "K x W $r" --save-draws kw_runs/$r/draws.npz > logs/kwfinal_$r.log 2>&1
  $PY ../isochrone_intervals/chain_z.py kw_runs/$r > logs/kwchainz_$r.log 2>&1
  echo "$(date '+%F %T') $r finalized"
done
