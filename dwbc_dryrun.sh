#!/bin/bash
# Dry run of the REAL decoupled-WBC chain without the robot: dwbc_real.sh against dwbc_fake_robot.py inside a
# private network namespace whose lo carries 192.168.123.99 (so `--interface real` takes the real code path).
#   unshare -rn bash ~/piper_ws/dwbc/dwbc_dryrun.sh
# A preflight  B start (ramp, control loop, guard)  C leg policy on + band off: stand  D estop -> damping
# E control loop killed -> guard damps  F thermal trip (guard with a 0 C limit on a dummy process)
HERE=$HOME/piper_ws/dwbc
L=$HOME/piper_ws/dwbc_logs/dry_$(date +%Y%m%d_%H%M%S); mkdir -p "$L"
exec > >(tee "$L/dryrun.log") 2>&1
ip link set lo up && ip addr add 192.168.123.99/24 dev lo || { echo "run me with: unshare -rn bash $0"; exit 1; }
export DWBC_DRY=1 DWBC_IFACE=lo DWBC_QUIET=1
source "$HERE/dwbc_env.sh"
cd /tmp
FRL=$L/fake_robot.log
python -u "$HERE/dwbc_fake_robot.py" > "$FRL" 2>&1 &
FR=$!
for i in $(seq 80); do grep -q "fake robot up" "$FRL" && break; sleep 0.5; done
sleep 2
fr() { echo "   fake robot: $(tail -1 "$FRL")"; }
echo "== A preflight"; "$HERE/dwbc_real.sh" check
echo "== B start"; "$HERE/dwbc_real.sh" start --confirmed --no-teleop --no-grippers || { kill -9 $FR; exit 1; }
fr; sleep 3; fr
echo "== C gantry down until the feet carry weight, leg policy on, then rope slack; stand; walk 3 s with keys"
touch /tmp/dwbc_fake_lower; sleep 4; fr
"$HERE/dwbc_real.sh" key "]"; sleep 2; fr
touch /tmp/dwbc_fake_slack
for i in 1 2 3 4; do sleep 2; fr; done
"$HERE/dwbc_real.sh" key w w; sleep 3; "$HERE/dwbc_real.sh" key z
for i in 1 2 3; do sleep 1.5; fr; done
"$HERE/dwbc_real.sh" status | head -2
echo "== D estop"
"$HERE/dwbc_real.sh" estop; sleep 1.2; fr; "$HERE/dwbc_real.sh" status
echo "== E start again, kill -9 the control loop: the guard must damp"
"$HERE/dwbc_real.sh" start --confirmed --no-teleop --no-grippers > "$L/start2.log" 2>&1; tail -1 "$L/start2.log" | head -c 200; echo
sleep 2; fr
kill -9 -- -"$(cat "$HERE/run/real_control.pid")"; sleep 2.5
tail -3 "$(readlink -f "$HOME/piper_ws/dwbc_logs/real_latest")/guard.log"; fr
"$HERE/dwbc_real.sh" estop > /dev/null 2>&1
echo "== F thermal trip"
setsid sleep 300 & DUMMY=$!; echo $DUMMY > /tmp/dwbc_dummy.pid; sleep 0.3
python -u "$HERE/dwbc_guard.py" --pidfile /tmp/dwbc_dummy.pid --iface lo --stop-winding 0 --hold 0.5 --quiet --flag /tmp/dwbc_estop_dry
kill -0 $DUMMY 2>/dev/null && echo "   dummy still alive: FAIL" || echo "   dummy killed by the guard: OK"
sleep 1; fr
kill -9 $FR; wait $FR 2>/dev/null   # SIGTERM is swallowed (rclpy + non-daemon sim thread)
echo "dry run done: $L"
