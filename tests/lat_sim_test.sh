#!/bin/bash
# Sim test of the latency monitor (dwbc_latency.py), no robot, no headset:
#   unshare -rn bash ~/piper_ws/dwbc/tests/lat_sim_test.sh [seconds]
# Private network namespace (own loopback), ROS domain 78, Unitree DDS domain 7 (MuJoCo sim in the control process),
# fake PICO SDK (tests/fake_pico/xrobotoolkit_sdk.py: A+X at 4 s, arms move from 6 s, 20-50 ms simulated Wi-Fi delay).
# Nothing reaches the real chain (ROS domain 77, DDS domain 0 on enp13s0) or its pid files in run/.
# Expect: [1] about 20-60 ms raw clock difference, every hop filled in, lat_summary.txt written.
HERE=$(cd "$(dirname "$0")/.." && pwd)
ip link set lo up || { echo "run me inside: unshare -rn bash $0"; exit 1; }
source "$HERE/dwbc_env.sh"
set -u
O=$DWBC_LOGS/lattest_$(date +%Y%m%d_%H%M%S); mkdir -p "$O"
export ROS_DOMAIN_ID=78 DWBC_DDS_DOMAIN=7 DWBC_LAT_DIR=$O PYTHONPATH=$HERE/tests/fake_pico:$PYTHONPATH
cd /tmp
python -u "$HERE/dwbc_run.py" control --interface sim --keyboard-dispatcher-type ros --no-enable-onscreen \
    --upper-body-joint-speed "${DWBC_ARM_SPEED:-3.0}" > "$O/control.log" 2>&1 &
C=$!
for i in $(seq 120); do grep -q "ROS keyboard dispatcher started" "$O/control.log" && break; sleep 0.5; done
python -u "$HERE/dwbc_run.py" teleop --body-control-device pico --hand-control-device pico > "$O/teleop.log" 2>&1 &
P=$!
sleep "${1:-30}"
kill -INT $P $C; sleep 3; kill -9 $P $C 2>/dev/null
grep -h "dwbc_latency" "$O/control.log" "$O/teleop.log"
python3 "$HERE/dwbc_latency.py" report "$O"
