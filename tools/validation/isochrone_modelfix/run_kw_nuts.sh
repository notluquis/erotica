#!/bin/bash
# NUTS on the K x W model (hub finding isochrone-intervals-nuts.md §2) for a list of fits in kw_runs/ (or $MODELFIX_RUNS),
# one fit at a time, its 4 chains in parallel (1 core each); replicates 500 + 500, real_* 1000 + 1000.
# Waits for the fit's starts.json; resumes from each chain's state.npz; finalize writes nuts.json,
# the pair/trace figures and draws.npz. Run two of these (disjoint lists) for 8 cores.
cd "$(dirname "$0")"
mkdir -p logs
PY=${PYTHON:-python}
export MODELFIX_RUNS="${MODELFIX_RUNS:-kw_runs}"
R_="$MODELFIX_RUNS"
export ARM=K WINDOW=W OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
for r in "$@"; do
  until [ -f $R_/$r/prep.json ]; do sleep 30; done
  if [ -f $R_/$r/nuts.json ]; then continue; fi
  case $r in real*) W=1000; D=1000;; *) W=500; D=500;; esac
  for c in 0 1 2 3; do
    if [ ! -f $R_/$r/chain$c/DONE ]; then
      NPROC=1 XLA_FLAGS=--xla_force_host_platform_device_count=1 nohup nice -n 5 $PY nuts8.py run --data $R_/$r/data.ecsv --starts $R_/$r/starts.json --chain $c --outdir $R_/$r/chain$c --warmup $W --draws $D > logs/kwnuts_${r}_c$c.log 2>&1 &
    fi
  done
  wait
  truth=$($PY -c "import json;d=json.load(open('$R_/$r/prep.json'));print(json.dumps(d['truth_pymc']) if d.get('truth_pymc') else '')")
  NPROC=1 $PY nuts8.py finalize --data $R_/$r/data.ecsv --outdirs $R_/$r/chain0,$R_/$r/chain1,$R_/$r/chain2,$R_/$r/chain3 --json $R_/$r/nuts.json --fig $R_/$r/pair.png --truth "$truth" --title "K x W $r" --save-draws $R_/$r/draws.npz > logs/kwfinal_$r.log 2>&1
  $PY ../isochrone_intervals/chain_z.py $R_/$r > logs/kwchainz_$r.log 2>&1
  echo "$(date '+%F %T') $r finalized"
done
