#!/bin/bash
# NUTS for a list of run dirs (cell_rep), one fit at a time, its 4 chains in parallel (1 core each);
# finalize writes runs/<r>/nuts.json and the pair/trace figures. Resumes from state.npz if present.
# WARMUP/DRAWS from the environment (default 500/500 for synthetic replicates).
cd "$(dirname "$0")"
W=${WARMUP:-500}; D=${DRAWS:-500}
for r in "$@"; do
  until [ -f runs/$r/starts.json ]; do sleep 20; done
  if [ -f runs/$r/nuts.json ]; then continue; fi
  for c in 0 1 2 3; do
    if [ ! -f runs/$r/chain$c/DONE ]; then
      NPROC=1 XLA_FLAGS=--xla_force_host_platform_device_count=1 nohup ~/miniforge3/envs/erotica/bin/python nuts_chain.py run --data runs/$r/data.ecsv --starts runs/$r/starts.json --chain $c --outdir runs/$r/chain$c --warmup $W --draws $D > logs/nuts_${r}_c$c.log 2>&1 &
    fi
  done
  wait
  truth=$(~/miniforge3/envs/erotica/bin/python -c "import json;d=json.load(open('runs/$r/prep.json'));print(json.dumps(d.get('truth')) if d.get('truth') else '')")
  NPROC=2 ~/miniforge3/envs/erotica/bin/python nuts_chain.py finalize --data runs/$r/data.ecsv --outdirs runs/$r/chain0,runs/$r/chain1,runs/$r/chain2,runs/$r/chain3 --json runs/$r/nuts.json --fig runs/$r/pair.png --truth "$truth" --title "$r" --save-draws runs/$r/draws.npz > logs/final_$r.log 2>&1
  echo "$(date '+%F %T') $r finalized"
done
