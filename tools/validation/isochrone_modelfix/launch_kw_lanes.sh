#!/bin/bash
# Start the two NUTS lanes without exceeding 8 cores: lane 1 when <= 4 preps run, lane 2 when none.
cd "$(dirname "$0")"
np() { ps aux | grep "python kw_cells.py prep" | grep -v -E "grep|sh -c" | wc -l; }
until [ "$(np)" -le 4 ]; do sleep 30; done
nohup ./run_kw_nuts.sh A_0 A_2 A_4 A_6 real_0 C_0 C_2 C_4 C_6 > logs/kw_lane1.out 2>&1 &
echo "$(date '+%F %T') lane1 started"
until [ "$(np)" -eq 0 ]; do sleep 30; done
nohup ./run_kw_nuts.sh A_1 A_3 A_5 A_7 C_1 C_3 C_5 C_7 > logs/kw_lane2.out 2>&1 &
echo "$(date '+%F %T') lane2 started"
wait
