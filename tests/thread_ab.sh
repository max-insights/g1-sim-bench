#!/bin/bash
# Thread-limit A/B of the DWBC chain in sim (private netns, ROS domain 78, DDS domain 7; never touches the real chain).
#   bash thread_ab.sh <seconds per run> <rounds>
# Configs: base (no limit), t1 (1 thread), t2 (2 threads), alternated per round. Per run: CPU% / threads of the two
# processes (top over 10 s, mid-run) + dwbc_latency summary.
SECS=${1:-60}; ROUNDS=${2:-2}
OUT=$HOME/piper_ws/dwbc_logs/thread_ab_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
run() {  # name nthreads
  local name=$1 n=$2 env=()
  [ "$n" != 0 ] && env=(OMP_NUM_THREADS=$n MKL_NUM_THREADS=$n OPENBLAS_NUM_THREADS=$n NUMEXPR_NUM_THREADS=$n)
  echo "== $name $(date +%T) load $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$OUT/runs.txt"
  env "${env[@]}" unshare -rn bash ~/piper_ws/dwbc/tests/lat_sim_test.sh "$SECS" > "$OUT/$name.txt" 2>&1 &
  local W=$!
  sleep $((SECS / 2 + 5))
  local pids; pids=$(pgrep -f "[d]wbc_run\.py (control|teleop)" | paste -sd,)
  { echo "-- top $name"; top -b -n 2 -d 10 -p "$pids" | awk '/^top -/{k++} k==2' | grep -E "python|PID";
    for p in ${pids//,/ }; do echo "pid $p threads $(ls /proc/$p/task | wc -l) $(tr '\0' ' ' < /proc/$p/cmdline | grep -oE '(control|teleop) ')"; done
  } >> "$OUT/$name.cpu" 2>&1
  wait $W
  grep -m1 -o "lattest_[0-9_]*" "$OUT/$name.txt" | head -1 >> "$OUT/runs.txt"
}
for r in $(seq "$ROUNDS"); do
  run base_r$r 0; run t1_r$r 1; run t2_r$r 2
done
echo "DONE $OUT"
