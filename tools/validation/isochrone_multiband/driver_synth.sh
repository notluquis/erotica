#!/bin/zsh
# Gate (coordinator, 2026-10-05 21:00, option B): start a fit only with memory_pressure >= 25 % free
# and swap free >= 0.5 GB; while running, SIGSTOP if memory_pressure < 20 %, SIGCONT when >= 25 %.
cd /Users/notluquis/erotica-wt-multiband/tools/validation/isochrone_multiband
export PYTHONPATH=/Users/notluquis/erotica-wt-multiband OMP_NUM_THREADS=1
PY=~/miniforge3/envs/cosmic/bin/python
mp() { memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print int($2)}'; }
sw() { sysctl -n vm.swapusage | sed -E 's/.*free = ([0-9.]+)M.*/\1/' | cut -d. -f1; }
gate() { while [ $(mp) -lt 25 -o $(sw) -lt 500 ]; do echo "gate closed mp=$(mp) swap=$(sw) $(date +%T)"; sleep 60; done; }
run() {
  gate
  nice -n 19 $PY colour_synth_mb.py --arm $1 --reps $2 &
  local pid=$!; local paused=0
  while kill -0 $pid 2>/dev/null; do
    sleep 20
    local m=$(mp)
    if [ $paused -eq 0 -a $m -lt 20 ]; then kill -STOP $pid; paused=1; echo "PAUSE $1 $2 mp=$m $(date +%T)"; fi
    if [ $paused -eq 1 -a $m -ge 25 ]; then kill -CONT $pid; paused=0; echo "RESUME $1 $2 mp=$m $(date +%T)"; fi
  done
  wait $pid || echo "FAILED $1 $2"
}
for r in 1 2 3 4 5 6 7 8; do for arm in control lori; do run $arm $r; done; done
for r in 1 2 3 4 5 6 7 8; do run nirJ $r; done
echo DRIVER_DONE
