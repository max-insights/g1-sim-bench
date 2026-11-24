#!/bin/bash
# Arm speed-cap A/B of the DWBC chain in sim (private netns, ROS domain 78, DDS domain 7; never touches the real chain).
#   bash speed_ab.sh <seconds per run> <rounds> [caps, default "3.0 4.0 5.0"]
# Fake PICO in fast-reach mode (FAKE_PICO_PROFILE=fast): min-jerk reaches every 2.5 s that push the arm joints into
# the --upper-body-joint-speed cap. Per run: dwbc_latency report + joint speeds (tests/joint_speed.py).
SECS=${1:-60}; ROUNDS=${2:-2}; CAPS=${3:-"3.0 4.0 5.0"}
HERE=$(cd "$(dirname "$0")/.." && pwd)
OUT=$HOME/piper_ws/dwbc_logs/speed_ab_$(date +%Y%m%d_%H%M%S); mkdir -p "$OUT"
for r in $(seq "$ROUNDS"); do
  for c in $CAPS; do
    name=cap${c}_r$r
    echo "== $name $(date +%T) load $(cut -d' ' -f1 /proc/loadavg)" | tee -a "$OUT/runs.txt"
    DWBC_ARM_SPEED=$c FAKE_PICO_PROFILE=fast unshare -rn bash "$HERE/tests/lat_sim_test.sh" "$SECS" > "$OUT/$name.txt" 2>&1
    d=$(grep -m1 -o "lattest_[0-9_]*" "$OUT/$name.txt" | head -1)
    echo "$name $d" >> "$OUT/runs.txt"
    [ -n "$d" ] && python3 "$HERE/tests/joint_speed.py" "$HOME/piper_ws/dwbc_logs/$d" >> "$OUT/runs.txt"
  done
done
echo "DONE $OUT"
